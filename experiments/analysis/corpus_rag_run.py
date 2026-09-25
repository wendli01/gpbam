"""The retrieval arm re-run over the rebuilt corpus, end to end.

``rebuild_kb_with_titles.py`` changes two things at once: the embedding model
(bge-large-en-v1.5, English-only, to jina-v5) and what goes into the vector (the
statute body alone, to the citation line plus the body).  ``corpus_comparison.py``
says whether the norms became *findable*.  This says whether that reaches the
score -- same generators and same judges as the gold oracle study, so the numbers
sit next to Table~\\ref{tab:oracle_panel} rather than starting a new scale.

Everything here is free: the five generators run on NHR@FAU and InnKube, and the
judges are the two ensemble seats that are free on NHR@FAU.  ``gpt-5-nano`` is
deliberately absent, which is why the table recomputes the published baselines'
median over the *same two* judges -- a two-judge median and a three-judge median
are different scales, and mixing them is exactly the error the panel plumbing
exists to prevent.

Queries come from the cache ``corpus_comparison.py`` writes, so the recall figure
describes the queries the generator actually issued rather than a fresh sample.

Two phases per model, both idempotent, so an interrupted queue resumes by being
re-run:

``generate``
    essays only -- the expensive artefact, saved before any judge is called
``judge``
    tops up whichever panel judges the stored file is missing, in place

Run from ``experiments/``::

    PYTHONPATH=analysis python analysis/corpus_rag_run.py run --kb ./my_knowledge_base_bayern_titled
    PYTHONPATH=analysis python analysis/corpus_rag_run.py table
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('..'))

from oracle_experiment import (JUDGE_ALIASES, PANEL_ARMS, PUBLISHED, api_keys, cases,
                               fmt, fmt_delta, judge_col, published_panel)
from oracle_experiment import D as ORACLE_D
from retrieval_ablation import norm_of
import refs as refs_mod
from retrieval_diagnostics import OUT

#: the five generators of the gold oracle study, under the identifiers their
#: endpoints answer to (see endpoints.yaml)
MODELS = ('deepseek-ai/DeepSeek-V4-Flash', 'openai/gpt-oss-120b', 'gemma4-31b-it',
          'qwen36-35b', 'qwen3-next-80b-a3b-instruct',
          # Same base model as qwen36-35b, served by NHR@FAU instead of InnKube,
          # and reported as its own row. InnKube hung three times in one
          # afternoon with no error and no retry, which makes the two InnKube
          # models unextendable; this one is reachable.
          'Qwen/Qwen3.6-35B-A3B-FP8')

#: the ensemble minus gpt-5-nano -- the two seats that cost nothing
FREE_PANEL = ('Qwen/Qwen3.6-35B-A3B-FP8', 'deepseek-ai/DeepSeek-V4-Flash')
FREE_COLS = tuple(judge_col(m) for m in FREE_PANEL)

#: the paper's full ensemble. gpt-5-nano is billed on OpenRouter at roughly
#: $0.005 per essay -- it answers with ~9k reasoning tokens under the repo's
#: reasoning_effort='high', which is what the published run used. Reachable only
#: through --judges panel, which prints the bill and refuses without --yes,
#: because the default has to stay free: this module is invoked from queue
#: scripts that run unattended overnight.
PAID_PANEL = FREE_PANEL + ('openai/gpt-5-nano',)
#: The DeepSeek slug is the DeepInfra-pinned OpenRouter fallback for the NHR@FAU
#: seat (see endpoints.yaml). Without it the guard below estimates a real pass at
#: $0.00 and waives its own --yes.
COST_PER_ESSAY = {'openai/gpt-5-nano': 0.005,
                  'deepseek/deepseek-v4-flash-0731': 0.004}
#: OpenRouter routing for the fallback. Of the four providers serving that slug
#: one is biased +13.75 marks high and one refuses 11 of 12 prompts, so this is
#: load-bearing, not a preference. Only the OpenRouter branch of llm.py reads it,
#: so it is harmless to pass alongside an NHR@FAU seat.
DEEPINFRA = {'order': ['DeepInfra'], 'allow_fallbacks': False,
             'data_collection': 'deny'}
#: A third free seat. gpt-oss-120b is served by NHR@FAU like the other two, so
#: it costs nothing to add -- but note it is also one of the *generators* in
#: these tables, so in the gpt-oss row it is scoring its own essays. The same
#: is already true of Qwen3.6 for the FAU Qwen row and DeepSeek for its own, and
#: the per-case median over three seats limits how far any one of them can pull
#: a cell; still, it is a reason to read the diagonal with care.
FREE_PANEL3 = FREE_PANEL + ('openai/gpt-oss-120b',)
#: free3 minus DeepSeek. Not a scale anything is reported on -- pt.med() wants
#: all three seats and returns None otherwise -- but a way to bank two thirds of
#: a judging pass while the DeepSeek deployment is busy generating. judge() only
#: fills seats a file lacks, so a later `--judges free3` run tops up the third
#: without redoing these two. Needed on 09-11, when the oracle ablation held
#: DeepSeek for ten hours and six ls_text arms sat generated and unjudged.
NODS_PANEL = ('Qwen/Qwen3.6-35B-A3B-FP8', 'openai/gpt-oss-120b')
#: The DeepSeek seat alone, through the OpenRouter fallback -- the other half of
#: ``nods``, for the days the FAU deployment is down rather than merely busy.
#: Writes ``score_Judge (deepseek-v4-flash-0731)``, which
#: ``paper_table.SEAT_ALIASES`` resolves onto the canonical seat.
DS_OR_PANEL = ('deepseek/deepseek-v4-flash-0731',)
PANELS = {'free': FREE_PANEL, 'free3': FREE_PANEL3, 'panel': PAID_PANEL,
          'nods': NODS_PANEL, 'ds_or': DS_OR_PANEL}

D = 'zubaers_result/essay_writing/rag_titled/ji2'
KB = './my_knowledge_base_bayern_titled'

#: Both rebuilt corpora, each with its own output directory.  Running the
#: generation arm over federal-only as well as the combined corpus is what makes
#: the score side answer the same question the recall side does: whether the
#: missing Landesrecht was the binding constraint, or only the embedding was.
KBS = {'federal': './my_knowledge_base_titled',
       'combined': './my_knowledge_base_bayern_titled'}

#: Retrieval configurations worth generating under, newest last.  ``roundrobin``
#: is what the first pass used and what the stored arms hold; the others change
#: only how the sub-query result lists are combined, which recall_battery.py
#: measured at 6.90% -> 8.52% recall@50 for no extra search.
PIPELINES = {
    'roundrobin': dict(merge='roundrobin', candidates=None),
    # The ladder the paper walks: dense -> hybrid -> citations -> citations
    # +hybrid. Every rung holds the sub-query fusion at rrf/100 so that
    # consecutive rungs differ in one thing. 'rrf' here is what merges the
    # rewriter's several sub-queries; under 'dense' there is no lexical list to
    # fuse with, so the name still says what the retriever is.
    'dense': dict(merge='rrf', candidates=100, query_type='vector'),
    'bm25': dict(merge='rrf', candidates=100, query_type='fts'),
    # Citations resolved by exact lookup and nothing else -- no search fills the
    # remaining budget, so this arm usually hands over fewer than top_k norms.
    # That is the measurement: at k=10 citation lookup alone already reaches
    # 11.90% recall against cite_rrf's 12.14%, so the search leg contributes
    # 0.24pp and the score table should say whether that buys anything.
    'cite_only': dict(merge='rrf', candidates=100, cite_first=True, cite_only=True),
    'rrf': dict(merge='rrf', candidates=100),
    # RRF's pool reordered by InnKube's Qwen3-Reranker-4B. The best re-ranking
    # option this project has and the cheapest: 4B, served off-box, no VRAM, and
    # free. In recall_battery.py it is the strongest search-only configuration
    # measured -- 5.12% recall@10 against merge_rrf's 3.39% and the deployed
    # baseline's 2.26% -- so it isolates one question the ladder cannot answer
    # from its citation rungs: does a *better search* move the essay score, or
    # only a better key lookup?
    #
    # Deliberately not stacked on cite_first. The re-ranker reorders the search
    # leg, and at k=10 that leg is worth 0.24pp over cite_only (12.14 against
    # 11.90), so cite+rerank would be a null by construction. Here the search
    # leg is the whole arm.
    'rerank': dict(merge='rrf', candidates=100, rerank='innkube'),
    # Norms the model names, resolved by exact lookup, ahead of search results.
    # Best configuration in recall_battery.py at every budget, and by the largest
    # margin at the one generation actually uses: 8.07% recall@10 against the
    # deployed baseline's 2.64%, with 97.5% of cases getting at least one cited
    # norm in the top ten.
    'cite_rrf': dict(merge='rrf', candidates=100, cite_first=True),
    # Both exact-key channels plus search. Best configuration measured, by a
    # wide margin and at every budget: 23.92% recall@50 against cite_rrf's
    # 16.49% and the deployed baseline's 6.90%, and 19.07% on state law where
    # the deployed pipeline gets 9.23%.
    'cite_ls_rrf': dict(merge='rrf', candidates=100, cite_first=True, two_hop=True),
    # The two-hop alone, to separate its contribution from the citations'.
    'ls_rrf': dict(merge='rrf', candidates=100, two_hop=True),
    # The Leitsätze themselves, as text, instead of only as a route to a
    # Normenkette. drop_norms makes this the one pipeline whose context holds no
    # statutes at all, so it is also the one that takes NORM_FREE_PREAMBLE.
    # Point --leitsatz-index at ./my_case_corpus_bayern and the same arm becomes
    # "decision bodies as context": hop one over case text, hop two suppressed.
    # Simpler to explain than the two-hop, and -- unlike anything
    # that ends in statute rows -- not bounded by the gold-norm oracle, which
    # supplies the right norms and no reasoning whatsoever.
    'ls_text': dict(merge='rrf', candidates=100, two_hop=True,
                    ls_context=10, drop_norms=True),
    # Both: statutes for the wording that must be cited, Leitsätze for the
    # doctrine that says which wording matters.
    'cite_ls_mixed': dict(merge='rrf', candidates=100, cite_first=True,
                          two_hop=True, ls_context=10),
}

#: Leitsätze consulted per case before following their Normenketten.
LS_DEPTH = 20
LEITSATZ_KB = './my_leitsatz_index'

#: How the retrieved passages are presented to the generator.
#:
#: Retrieval hands over a *ranked* list and, since ``cite_rrf``, a list whose
#: rank means something quite specific: exact-key resolutions lead, and roughly
#: 43% of those land in the top ten against ~18% of search hits.  None of that
#: is currently communicated -- ``format_context`` deliberately emits no running
#: index, on the reasoning that a position number is noise the model might echo
#: into its citations.  That reasoning is about *citation* hygiene and says
#: nothing about whether the ordering itself is useful, which has never been
#: measured.  These four styles measure it.  Retrieval is identical across all
#: of them; only the presentation changes.
CONTEXT_STYLES = {
    # what every published arm used: no index, no ordering claim, no provenance
    'plain': '',
    # state the ordering the model is already being given
    'ranked':
        '- Die Normtexte sind nach abnehmender Trefferrelevanz sortiert: die '
        'zuerst genannte Norm passt nach Einschätzung der Suche am besten, die '
        'zuletzt genannte am schlechtesten. Das Attribut rang gibt diesen Rang an.',
    # same claim, opposite document order -- the diagnostic. If position is being
    # used at all, primacy and recency should not score the same.
    'reversed':
        '- Die Normtexte sind nach zunehmender Trefferrelevanz sortiert: die '
        'zuletzt genannte Norm passt nach Einschätzung der Suche am besten, die '
        'zuerst genannte am schlechtesten. Das Attribut rang gibt den Rang an, '
        'wobei rang 1 die beste Übereinstimmung bezeichnet.',
    # where a passage came from, rather than a score. Scores are not comparable
    # across channels -- an RRF score and an exact-match "score" are not the same
    # quantity, and inventing 1.0 for a lookup would be a claim we cannot support.
    'provenance':
        '- Das Attribut quelle gibt an, wie eine Norm gefunden wurde: '
        '"zitat" = als einschlägig benannt und unmittelbar nachgeschlagen; '
        '"leitsatz" = aus der Normenkette einer gerichtlichen Entscheidung zu '
        'einem ähnlichen Sachverhalt; "suche" = durch Ähnlichkeitssuche im '
        'Gesetzestext gefunden. Normen mit quelle "zitat" oder "leitsatz" sind '
        'erfahrungsgemäß deutlich häufiger einschlägig als solche mit "suche".',
}

#: Cases per generation chunk.  The partial file is rewritten after each chunk,
#: so a killed run loses at most this many essays and resumes from the rest.
GEN_CHUNK = 20

#: Marks which channel produced a row, for the ``provenance`` style.
CHANNEL_KEY = '_channel'
#: Marks a row as a Leitsatz rather than a norm, so it is rendered as one.
KIND_KEY = '_kind'

#: Bullet added whenever Leitsätze are handed over as text.  They are not norms
#: and must not be cited as if they were, which is the one way this arm could
#: quietly make the essays worse.
#: The whole statute half of the ``QA_USER_RAG`` preamble, and what replaces it
#: when the arm hands over no statutes at all.
#:
#: ``ls_text`` sets ``drop_norms``, so its context holds court material and
#: nothing else -- yet the shared preamble opens "Im Abschnitt <context> findest
#: du Normtexte" and then spends four bullets on how to read <norm> elements
#: that are not there. The stored ``ls_text`` arm was generated that way: 0 norm
#: elements against a prompt describing five properties of them. It is the one
#: arm whose point is context that is not a norm list, so a preamble insisting
#: it is a norm list is the wrong instrument for it.
#:
#: A replacement rather than another ``CONTEXT_STYLES`` bullet, because the
#: defect is text that must come *out*: styles only insert.
NORM_PREAMBLE = (
    'Im Abschnitt <context> findest du Normtexte, die durch eine automatische '
    'Suche zu diesem Sachverhalt gefunden wurden. Beachte dabei:\n'
    '- Die Auswahl ist unvollständig und teilweise unpassend. Sie ersetzt nicht '
    'deine eigene Kenntnis des deutschen Rechts.\n'
    '- Insbesondere kann Landesrecht (z. B. PAG, BayBO, BayVwVfG, BV, GO, '
    'LStVG) vollständig fehlen, obwohl es einschlägig ist.\n'
    '- Wenn eine Vorschrift dort nicht auftaucht, heißt das nicht, dass sie '
    'nicht existiert oder nicht einschlägig ist. Zitiere die einschlägigen '
    'Normen unabhängig davon, ob ihr Wortlaut unten abgedruckt ist.\n'
    '- Verwende die abgedruckten Normtexte, soweit sie einschlägig sind, und '
    'ignoriere die übrigen. Übernimm keine Vorschrift nur deshalb, weil sie '
    'unten steht.\n'
    '- Die Normtexte stehen in <norm>-Elementen mit den Attributen zitat, '
    'gebiet und ueberschrift. Zitiere im Gutachten die Norm selbst (etwa '
    '„§ 34 BauGB“), nicht das Element oder seine Position.')

NORM_FREE_PREAMBLE = (
    'Im Abschnitt <context> findest du Rechtsprechung, die durch eine '
    'automatische Suche zu diesem Sachverhalt gefunden wurde. Beachte dabei:\n'
    '- Die Auswahl ist unvollständig und teilweise unpassend. Sie ersetzt nicht '
    'deine eigene Kenntnis des deutschen Rechts.\n'
    '- Es sind KEINE Gesetzestexte abgedruckt. Zitiere die einschlägigen '
    'Normen aus deiner eigenen Kenntnis, unabhängig davon, ob sie im Kontext '
    'erwähnt werden.')

LEITSATZ_BULLET = (
    '- Neben den Normtexten enthält der Kontext <leitsatz>-Elemente: abstrakte '
    'Rechtssätze aus Entscheidungen bayerischer Gerichte zu ähnlichen '
    'Sachverhalten, mit dem Gericht, dem Aktenzeichen und der Normenkette der '
    'Entscheidung. Sie sind Rechtsprechung, kein Gesetzestext. Nutze sie, um die '
    'einschlägigen Normen und die dogmatischen Maßstäbe zu erkennen, und zitiere '
    'im Gutachten die Norm selbst; ein Leitsatz kann zusätzlich als Rechtsprechung '
    'angeführt werden, ersetzt aber keine Norm.')

#: The same bullet for an arm with no statutes in it. Only the opening clause
#: changes -- "Neben den Normtexten" names a thing the norm-free context does
#: not contain -- and the rest is deliberately identical, so the two ls_text
#: arms still differ in exactly one thing.
LEITSATZ_BULLET_ONLY = LEITSATZ_BULLET.replace(
    '- Neben den Normtexten enthält der Kontext <leitsatz>-Elemente:',
    '- Der Kontext besteht aus <leitsatz>-Elementen:')

#: Citations the rewriter names, under the prompt recall_battery.py caches them
#: with. A citation is a key, not a query: asking for these norms *as search
#: queries* was the worst configuration measured (3.09% recall@50), because
#: "§ 34 BauGB" has almost nothing for a dense model to embed and BM25 splits it
#: into tokens that match thousands of norms. Looked up exactly, the same
#: citations are the strongest signal available.
CITE_REWRITES = f'{OUT}/rewrites_citations.json'


def leitsatz_tag(leitsatz_kb):
    """Path fragment naming a non-default Leitsatz index, or '' for the default.

    v1 keeps the unsuffixed names every stored two-hop arm was written under.
    """
    if not leitsatz_kb or os.path.normpath(leitsatz_kb) == os.path.normpath(LEITSATZ_KB):
        return ''
    rest = os.path.basename(os.path.normpath(leitsatz_kb))
    # the leitsatz replace has to run first: 'my_leitsatz_index_v2' has to
    # reduce to 'v2' so the stored _lsv2 arms keep their names, and stripping
    # the 'my_' prefix first would leave 'leitsatz_index_v2' and rename them.
    rest = rest.replace('my_leitsatz_index', '').lstrip('_')
    rest = rest[3:] if rest.startswith('my_') else rest
    return f'_ls{rest}' if rest else '_lsalt'


def out_dir(corpus, pipeline='roundrobin', style='plain', top_k=10, leitsatz_kb=None):
    """Where a (corpus, pipeline, style, top_k) arm's answers live.

    ``roundrobin``, ``plain`` and ``top_k=10`` keep the unsuffixed directory
    names the earlier passes wrote, so those runs stay readable without being
    regenerated.

    ``top_k`` is in the path because ``generate`` skips any arm whose file
    already exists.  Without it, a ``--top-k 50`` run would find the ``top_k=10``
    file, print "reusing", and report the old essays as the new arm's result --
    a wrong number that looks like a successful run.
    """
    base = f'rag_titled_{corpus}' + ('' if pipeline == 'roundrobin' else f'_{pipeline}')
    if style != 'plain':
        base = f'{base}_{style}'
    if top_k != 10:
        base = f'{base}_k{top_k}'
    # Only the two-hop pipelines resolve through the Leitsatz index, so only
    # they fork a directory for it -- an rrf arm is byte-identical either way
    # and splitting it would strand half its models in a second tree.
    #
    # Without this an arm run against v2 would land on top of its v1 twin, and
    # ``generate`` skips any arm whose file already exists: it would print
    # "reusing", and report the v1 essays as the v2 result. Same failure the
    # top_k suffix above exists to prevent.
    if PIPELINES.get(pipeline, {}).get('two_hop'):
        base += leitsatz_tag(leitsatz_kb)
    return f'zubaers_result/essay_writing/{base}/ji2'


CORPORA = {c: (kb, out_dir(c)) for c, kb in KBS.items()}
#: the rewrites corpus_comparison.py measured recall against
REWRITES = f'{OUT}/rewrites_qwen3.6-35b-a3b-fp8.json'


class CachedReWriter:
    """Replays stored query rewrites instead of sampling new ones.

    ``qa.ReWriter`` samples, so a fresh call would hand the generator different
    queries from the ones recall was measured on and the two numbers would stop
    describing the same pipeline.  Same interface as the real rewriter, minus the
    API calls.
    """

    def __init__(self, path=REWRITES):
        rewrites = json.load(open(path))
        # keyed off the full case list, so a --limit run still finds its rewrites
        all_facts = cases()[0]
        if len(rewrites) != len(all_facts):
            raise SystemExit(f'{path}: {len(rewrites)} rewrites for {len(all_facts)} cases')
        self.by_facts = dict(zip(all_facts, rewrites))
        self.name = f'CachedReWriter ({os.path.basename(path)})'

    def predict(self, queries):
        missing = [q for q in queries if q not in self.by_facts]
        if missing:
            raise SystemExit(f'{len(missing)} case(s) absent from {REWRITES}; '
                             'regenerate the cache with corpus_comparison.py')
        return [self.by_facts[q] for q in queries]


class CitationFirstRetriever:
    """A LawRetriever whose results are led by norms resolved as exact keys.

    Wraps rather than extends, and lives here rather than in ``src/rag.py``,
    because resolving a citation needs ``analysis/refs.py`` and the pipeline
    should not depend on the analysis layer.

    Two channels can lead, and ``recall_battery.py`` says they are complementary
    rather than redundant -- 16.49% and 18.92% recall@50 separately, 23.92%
    together, which is the whole reason both are wired here:

    ``citations``
        Norms the rewriter named from the model's own memory.  Precise but
        capped: it plateaus at 11.16% from k=20 onward, because a model can only
        name what it already knows, and it names ~10 norms per case.
    ``two-hop``
        Norms a *retrieved authority* names.  Hop one searches 38,966 Leitsätze
        -- abstracted rule statements, the only text in any of our corpora
        written in the register an exam question is written in -- and hop two
        reads the Normenkette off each hit.  No memory ceiling, and the best
        state-law recall measured anywhere (19.68%).

    Search fills whatever budget is left after both.
    """

    def __init__(self, retriever, citations_by_query=None, top_k=10,
                 leitsatz_kb=None, rewrites_by_query=None, ls_depth=LS_DEPTH,
                 ls_context=0, drop_norms=False, cite_only=False):
        self.ret = retriever
        self.by_query = citations_by_query or {}
        self.top_k = top_k
        self.rewrites = rewrites_by_query or {}
        self.ls_depth = ls_depth
        #: how many Leitsätze go into the context as text, rather than only
        #: being read for their Normenkette and thrown away
        self.ls_context = ls_context
        #: hand over Leitsätze only, with no statute text at all
        self.drop_norms = drop_norms
        #: citations only: nothing fills the budget behind them
        self.cite_only = cite_only
        self._index = None
        self._ls = None
        if leitsatz_kb:
            import lancedb
            self._ls = lancedb.connect(leitsatz_kb).open_table('documents')
            print(f'  Leitsatz index: {self._ls.count_rows()} rules', flush=True)

    @property
    def index(self):
        """``(law_book, section) -> row``, built once from the corpus itself."""
        if self._index is None:
            tbl = self.ret.table_.to_lance().to_table(
                columns=['text', 'title', 'law_book', 'paragraph', 'source_path']
            ).to_pylist()
            idx = {}
            for r in tbl:
                n = norm_of(r)
                if n and n not in idx:   # first wins; later ones are reprints
                    idx[n] = r
            self._index = idx
            print(f'  citation index: {len(idx)} norms addressable by citation',
                  flush=True)
        return self._index

    def cited(self, query):
        """Norms named for this case, resolved against the corpus.

        Extraction runs over the whole rewrite at once rather than line by line:
        ``refs.extract`` resolves citations that share a book -- "§§ 34, 35
        BauGB" -- which splitting first breaks apart. Worth ~6% more norms
        (486 recovered against 458). No cap on how many citations are taken;
        an exact lookup costs nothing per extra one.
        """
        out, seen = [], set()
        text = str(self.by_query.get(query, ''))
        for c in refs_mod.canonicalise(refs_mod.extract(text)):
            key = (refs_mod.corpus_key(c.book), c.section.lower())
            if key in self.index and key not in seen:
                seen.add(key)
                out.append(self.index[key])
        return out

    def hopped(self, query):
        """Norms named by the Leitsätze this case's rewritten queries retrieve.

        Searched with the same cached rewrites the statute half uses, one
        sub-query at a time and fused by rank, so a Leitsatz several sub-queries
        agree on outranks one a single lucky sub-query found -- every chain
        followed costs budget that cannot be spent on another.
        """
        return self._resolve(self.leitsaetze(query))

    def leitsaetze(self, query):
        """The Leitsätze this case's rewritten queries retrieve, best first.

        Kept as a separate step because the Leitsatz text is useful in its own
        right, not only as a route to a Normenkette. It is the doctrinal
        register -- Rücksichtnahmegebot, Bebauungszusammenhang -- that no statute
        contains, already abstracted into a rule, at ~300 characters. Handing it
        to the generator directly is both simpler to explain than the two-hop and
        not bounded by the gold-norm oracle, which supplies norms and no
        reasoning at all.
        """
        if self._ls is None:
            return []
        subs = [s.strip(' \t-*0123456789.')
                for s in str(self.rewrites.get(query, '')).splitlines()]
        subs = [s for s in subs if len(s) > 12] or [str(query)]
        lists = []
        for s in subs:
            clean = re.sub(r'[^\wÄÖÜäöüß§ ]+',
                           ' ', s).lower()
            clean = re.sub(r'\s+', ' ', clean).strip()
            if not clean:
                continue
            lists.append(self._ls.search(clean, query_type='hybrid')
                         .limit(self.ls_depth).to_pandas().to_dict('records'))
        score, rows = {}, {}
        for lst in lists:
            for rank, r in enumerate(lst):
                key = r.get('source_path')
                score[key] = score.get(key, 0.0) + 1.0 / (60 + rank)
                rows.setdefault(key, r)
        return [rows[k] for k in sorted(score, key=score.get, reverse=True)][:self.ls_depth]

    def _resolve(self, ranked):
        """Normenkette of each retrieved Leitsatz, resolved to statute rows."""
        out, seen = [], set()
        for key in refs_mod.chain_norms_ordered(r.get('norms') or '' for r in ranked):
            if key in self.index and key not in seen:
                seen.add(key)
                out.append(self.index[key])
        return out

    def predict(self, queries):
        # cite_only never reads the search list, so do not pay for it: the dense
        # pass over 81 cases is the expensive half of this pipeline
        found = ([() for _ in queries] if self.cite_only
                 else self.ret.predict(queries))
        out = []
        for q, rows in zip(queries, found):
            # citations first: the more precise channel leads, the broader one
            # fills behind it, and a norm reached both ways appears once
            lead, lead_seen = [], set()
            # tagged with the channel that found them, which the `provenance`
            # context style shows the generator. Rows are copied first: they come
            # from the shared citation index, so writing the tag in place would
            # mislabel the same norm on a later case that found it another way.
            for r, channel in ([(x, 'zitat') for x in self.cited(q)]
                               + [(x, 'leitsatz') for x in self.hopped(q)]):
                key = (r.get('law_book'), r.get('paragraph'))
                if key not in lead_seen:
                    lead_seen.add(key)
                    lead.append(dict(r, **{CHANNEL_KEY: channel}))
            texts = []
            if self.ls_context:
                texts = [dict(r, **{CHANNEL_KEY: 'leitsatz', KIND_KEY: 'leitsatz'})
                         for r in self.leitsaetze(q)[:self.ls_context]]
            if self.drop_norms:
                out.append(texts[:self.top_k])
                continue
            rest = [] if self.cite_only else [
                dict(r, **{CHANNEL_KEY: 'suche'}) for r in rows
                if (r.get('law_book'), r.get('paragraph')) not in lead_seen]
            # Leitsätze lead: they are the shortest passages and the only ones
            # that state a rule rather than a norm text, so they are the part
            # most likely to be read if the generator reads only the top of the
            # context.
            out.append((texts + lead + rest)[:self.top_k])
        return out


def styled_generator(style, extra_bullets=(), norm_free=False, **kwargs):
    """An ``AnswerGenerator`` that tells the model what its context ordering means.

    Subclassed here rather than parameterised in ``src/qa.py`` because the
    default has to keep producing byte-identical prompts: every published arm
    used it, and a prompt change would silently make those numbers describe a
    different experiment.
    """
    from src import qa
    from src.prompts import QA_USER_RAG

    extra = '\n'.join([b for b in (CONTEXT_STYLES[style], *extra_bullets) if b])
    # The bullet goes next to the one that already explains the <norm> element,
    # so all the instructions about how to read the context sit together.
    anchor = ('- Die Normtexte stehen in <norm>-Elementen mit den Attributen '
              'zitat, gebiet und ueberschrift.')
    if norm_free:
        # No <norm> element exists to anchor against, so the statute half of the
        # preamble is replaced wholesale and the extra bullets go where it ended.
        if NORM_PREAMBLE not in QA_USER_RAG:
            raise SystemExit('QA_USER_RAG no longer contains the statute '
                             'preamble verbatim; NORM_PREAMBLE needs updating')
        replacement = NORM_FREE_PREAMBLE + (f'\n{extra}' if extra else '')
        prompt = QA_USER_RAG.replace(NORM_PREAMBLE, replacement)
    elif extra:
        if anchor not in QA_USER_RAG:
            raise SystemExit('QA_USER_RAG no longer contains the <norm> bullet; '
                             'the context-style prompts need re-anchoring')
        prompt = QA_USER_RAG.replace(anchor, f'{extra}\n{anchor}')
    else:
        prompt = QA_USER_RAG

    class StyledAnswerGenerator(qa.AnswerGenerator):
        def format_context(self, results):
            rows = list(results)
            # rank is assigned before any reordering, so `rang` always means
            # "retrieval's opinion" and only the reading order changes
            ranked = list(enumerate(rows, start=1))
            if style == 'reversed':
                ranked = ranked[::-1]
            blocks = []
            for rank, row in ranked:
                if row.get(KIND_KEY) == 'leitsatz':
                    blocks.append(
                        f'<leitsatz gericht="{row.get("court", "")}" '
                        f'datum="{row.get("date", "")}" '
                        f'az="{str(row.get("paragraph", "")).split(" LS")[0]}" '
                        f'normenkette="{row.get("norms", "")}">\n'
                        f'{str(row.get("text", "")).strip()}\n</leitsatz>')
                    continue
                attrs = [f'zitat="{self._citation(row)}"',
                         f'gebiet="{self._jurisdiction(row)}"']
                if style in ('ranked', 'reversed'):
                    attrs.append(f'rang="{rank}"')
                if style == 'provenance':
                    attrs.append(f'quelle="{row.get(CHANNEL_KEY, "suche")}"')
                attrs.append(f'ueberschrift="{self._heading(row)}"')
                blocks.append(f'<norm {" ".join(attrs)}>\n'
                              f'{str(row.get("text", "")).strip()}\n</norm>')
            return '\n'.join(blocks)

    return StyledAnswerGenerator(prompt=prompt, **kwargs)


def slug(model):
    return model.replace('/', '_')


def path_for(model, d=D):
    return f'{d}/rag_{slug(model)}.csv'


def missing_judges(path, panel=FREE_PANEL):
    """Panel judges that have not scored ``path`` yet.

    A seat counts as unscored when its column is absent *or* holds nothing.
    Testing presence alone is what let a failed judging pass look finished:
    the pass writes the column, every value comes back NaN, and from then on
    every retry skips the file because the column is there. Eleven cells of
    the ladder sat blank that way -- essays generated, two seats scored, the
    third an empty column no rejudge would ever refill. ``med`` refuses a
    column with no values, so nothing was ever printed wrong; it was simply
    never printed.
    """
    if not os.path.exists(path):
        return list(panel)
    d = pd.read_csv(path)
    return [m for m in panel
            if judge_col(m) not in d.columns or not d[judge_col(m)].notna().any()]


def make_reranker(name):
    """Build the re-ranker a pipeline asks for, or None.

    Only the InnKube-served one is wired in. The local cross-encoders are a
    swap away, but they need the retrieve-then-unload dance to share 8 GB with
    the query embedder, and they are all weaker than this one anyway.
    """
    if name is None:
        return None
    if name != 'innkube':
        raise SystemExit(f"unknown reranker {name!r}; only 'innkube' is wired in")
    from recall_battery import InnkubeReranker
    return InnkubeReranker()


#: DeepSeek-V4's reasoning effort is only a text prefix at the very start of the
#: prompt (encoding/encoding_dsv4.py, REASONING_EFFORT_PROMPTS; the system template
#: is plain "{content}", and this generator's system prompt is empty, so prefix as
#: the system message is the encoding verbatim). GWDG, like NHR@FAU in August,
#: sends `high` without it: effort `low` plus this prefix is DeepSeek's real high.
EFFORT_PREFIX = {
    'high': ("Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
             "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
             "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n"),
}


def generate(model, kb=KB, top_k=10, d=D, max_tokens=None, device='cuda', limit=None,
             pipeline='roundrobin', style='plain', leitsatz_kb=None, provider=None,
             store_as=None, concurrency=None, effort=None, endpoint=None, effort_prefix=None):
    """Essays for one model over ``kb``; returns the path, reusing an existing file.

    ``store_as`` separates the arm's identity from the endpoint that serves it.
    The DeepSeek weights are reachable under two names -- NHR@FAU's
    ``deepseek-ai/DeepSeek-V4-Flash-0731`` and OpenRouter's lowercase
    ``deepseek/deepseek-v4-flash-0731`` -- and ``path_for`` derives the filename
    from whichever one is passed. Generating through the fallback without this
    would write ``rag_deepseek_deepseek-v4-flash-0731.csv`` where
    ``ladder_table`` looks for ``rag_deepseek-ai_DeepSeek-V4-Flash``, so the arm
    would be invisible to the table that needs it.
    """
    stored = store_as or model
    out = path_for(stored, d)
    if os.path.exists(out):
        print(f'reusing {out}', flush=True)
        return out
    os.makedirs(d, exist_ok=True)
    facts, sols = cases()
    if limit:  # smoke test: exercise the whole path on a couple of cases
        facts, sols = facts[:limit], sols[:limit]

    from src import qa, rag
    knobs = PIPELINES[pipeline]
    ret = rag.LawRetriever(kb_name=kb, top_k=top_k, device=device, verbose=0,
                           rewriter=CachedReWriter(),
                           merge=knobs['merge'],
                           candidates=knobs['candidates'] or top_k,
                           query_type=knobs.get('query_type', 'hybrid'),
                           reranker=make_reranker(knobs.get('rerank')))
    if knobs.get('cite_first') or knobs.get('two_hop'):
        facts_all = cases()[0]
        by_cite = None
        if knobs.get('cite_first'):
            cites = json.load(open(CITE_REWRITES))
            if len(cites) != len(facts_all):
                raise SystemExit(f'{CITE_REWRITES}: {len(cites)} for {len(facts_all)} cases')
            by_cite = dict(zip(facts_all, cites))
        by_rewrite = None
        if knobs.get('two_hop'):
            # the same cached concept rewrites the statute half searches with,
            # so both halves of the pipeline see the same queries
            by_rewrite = dict(zip(facts_all, json.load(open(REWRITES))))
        ret = CitationFirstRetriever(
            ret, by_cite, top_k=top_k,
            leitsatz_kb=(leitsatz_kb or LEITSATZ_KB) if knobs.get('two_hop') else None,
            rewrites_by_query=by_rewrite,
            # a Leitsätze-only arm fills the whole budget with Leitsätze, so
            # it is compared against the norm arms at the same context size
            # rather than at a tenth of it
            ls_context=(top_k if knobs.get('drop_norms')
                        else knobs.get('ls_context', 0)),
            drop_norms=knobs.get('drop_norms', False),
            cite_only=knobs.get('cite_only', False))
    # drop_norms is the only case where the context holds no statutes at all, so
    # it is also the only one that takes the norm-free preamble and the Leitsatz
    # bullet that does not open by referring to Normtexte.
    free = knobs.get('drop_norms', False)
    bullet = (LEITSATZ_BULLET_ONLY if free else LEITSATZ_BULLET,) \
        if knobs.get('ls_context') else ()
    # one request per essay: see oracle_experiment.GEN_TIMEOUT -- NHR@FAU turns
    # every client timeout into a second, concurrent generation of the same essay
    gen = styled_generator(style, retriever=ret, model=model, max_tokens=max_tokens,
                           norm_free=free, extra_bullets=bullet, timeout=4 * 3600,
                           **({'provider': provider} if provider else {}),
                           **({'max_concurrency': concurrency} if concurrency else {}),
                           **({'reasoning_effort': effort} if effort else {}),
                           # GWDG: 30/min, 200/h, 1000/day per key -- rate limits are
                           # waited out (hours), not turned into blank essays
                           **(dict(inference_endpoint=os.environ['ENDPOINT_AC'], token_var='API_KEY_AC',
                                   stream=True, max_retries=400, max_backoff=90)
                              if endpoint == 'gwdg' else {}),
                           **({'system_prompt': EFFORT_PREFIX[effort_prefix]} if effort_prefix else {}))
    print(f'\n=== generating {len(facts)} answers with {model} '
          f'over {kb} (top_k={top_k}, {pipeline}, style={style}) ===', flush=True)
    # Generated in chunks, with the partial file rewritten after each one.
    #
    # ``gen.predict`` is atomic -- it returns only once every case is done -- so
    # a run that dies at case 70 of 81 used to leave nothing at all behind. That
    # happened four times in one afternoon (a hung endpoint, an OOM kill, and
    # twice by an over-broad kill pattern) and cost about eighty essays. Chunking
    # bounds the loss to one chunk and makes a restart resume instead of repeat.
    part = out + '.part'
    frames, start = [], 0
    if os.path.exists(part):
        prev = pd.read_csv(part)
        # only trust a prefix: the cases are generated in order
        start = int(prev['index'].max()) + 1 if len(prev) else 0
        frames = [prev]
        print(f'  resuming from {part}: {start} case(s) already generated', flush=True)

    for i in range(start, len(facts), GEN_CHUNK):
        block = list(facts[i:i + GEN_CHUNK])
        answers, info = gen.predict(block, return_raw=True)
        # A generation that exhausts its retries comes back as None here --
        # qa.predict deliberately keeps the row rather than dropping it, and
        # pd.DataFrame chokes on the None one line later, taking the chunk with
        # it. That is what cost gemma both of its k=50 arms on 23 August: ~11.5 h
        # of generation, no .part written, because the failure landed inside the
        # first chunk. src/evaluate.py has carried this guard since the Kimi run.
        df = pd.DataFrame([i if isinstance(i, dict) else {} for i in info]) \
               .rename(columns={'prompt': 'qa_prompt'})
        df.insert(0, 'answer', answers)
        df.insert(0, 'model', stored)
        df.insert(0, 'index', range(i, i + len(df)))
        frames.append(df)
        pd.concat(frames, ignore_index=True).to_csv(part + '.tmp', index=False)
        os.replace(part + '.tmp', part)
        print(f'  {min(i + GEN_CHUNK, len(facts))}/{len(facts)} generated', flush=True)

    df = pd.concat(frames, ignore_index=True)
    df['kb'] = kb
    df['top_k'] = top_k
    df['pipeline'] = pipeline
    df['context_style'] = style
    df['endpoint'] = endpoint or 'default'
    df['reasoning_effort'] = effort or 'high'
    df['effort_prefix'] = effort_prefix
    # Written before a single judge call: retrieval plus generation over 81 cases
    # is the part that takes hours, judging is cheap and repeatable.
    df.to_csv(out + '.tmp', index=False)
    os.replace(out + '.tmp', out)
    if os.path.exists(part):
        os.remove(part)
    answers = list(df['answer'])
    n_empty = sum(1 for a in answers if not str(a).strip() or a is None)
    print(f'written to {out}  ({len(df)} rows, {n_empty} empty)', flush=True)
    return out


def judge(path, panel=FREE_PANEL, judge_instruction='ji2', provider=None):
    """Score ``path`` with whichever panel judges it is missing, in place.

    One seat at a time, written after each. The seats are independent, and this
    endpoint returns 502s often enough that judging all of them into one frame
    and saving at the end loses the finished ones: qwen3-next's two k=50 arms
    were judged three times and stored nothing, because the DeepSeek seat died
    around case 20 each time and took the completed Qwen3.6 seat with it. A seat
    that fails now leaves the earlier ones on disk and is simply still missing,
    which is what ``missing_judges`` looks for on the next run.
    """
    todo = missing_judges(path, panel)
    if not todo:
        print(f'  all {len(panel)} judges present in {os.path.basename(path)}', flush=True)
        return
    facts_all, sols_all = cases()

    from src import scoring, prompts, evaluate
    instruction = prompts.JUDGE_INSTRUCTIONS_BY_NAME[judge_instruction]
    failed = []
    for m in todo:
        # re-read: the previous seat added a column to the file
        src = pd.read_csv(path).sort_values('index')
        # rows are stored in gpbam task order, so a short file is a prefix --
        # which is what --limit produces
        facts, sols = facts_all[:len(src)], sols_all[:len(src)]
        ensemble = scoring.JudgeEnsemble(
            [scoring.Judge(model=m, prompt=prompts.build_judge_user(instruction),
                           provider=provider)],
            verbose=True)
        print(f'\n=== judging {os.path.basename(path)} ({len(src)} essays) '
              f'<- {m} ===', flush=True)
        try:
            df = evaluate.rejudge_model(src.answer.fillna('').tolist(), facts, sols,
                                        judge=ensemble,
                                        model_name=str(src.model.iloc[0]), verbose=True)
        except Exception as e:  # a dead seat should not cost the finished ones
            print(f'!! seat {m} failed: {type(e).__name__}: {e}', flush=True)
            failed.append(m)
            continue

        add = [c for c in df.columns if c.endswith(f'Judge ({m.split("/")[-1]})')]
        merged = src.reset_index(drop=True).drop(columns=add, errors='ignore')
        for c in add:
            merged[c] = df[c].values
        assert len(merged) == len(src) and judge_col(m) in merged.columns
        merged.to_csv(path + '.tmp', index=False)
        os.replace(path + '.tmp', path)
        have = [c for c in merged.columns if c in FREE_COLS]
        med = merged[have].median(axis=1, skipna=True)
        print(f'  panel median over {len(have)} judge(s) {100 * med.mean():.2f}  '
              f'-> {os.path.basename(path)}', flush=True)
    if failed:
        raise RuntimeError(f'{len(failed)} seat(s) unfilled: {", ".join(failed)}')


def subscale(df):
    """Per-essay median over the two free judges, times 100.

    ``None`` if either seat is unfilled -- a one-judge median is not the same
    quantity, and silently falling back to it is how arms end up incomparable.
    """
    out = {}
    for c in FREE_COLS:
        for alias in JUDGE_ALIASES.get(c, (c,)):
            if alias in df.columns and df[alias].notna().any():
                out[c] = pd.to_numeric(df[alias], errors='coerce').values
                break
        else:
            return None
    return (100 * pd.DataFrame(out).median(axis=1, skipna=True)).values


def baseline_scores(model_key, arm, d=ORACLE_D):
    """One comparison arm for one model, on the two-judge subscale.

    Follows ``PANEL_ARMS`` rather than always reading the published run: gemma
    and qwen36-35b generate on InnKube here, and the oracle study already stores
    same-host no-retrieval baselines for them precisely so the comparison is not
    a comparison of deployments.
    """
    spec = PANEL_ARMS[model_key].get(arm)
    if spec is None:
        return None
    if spec == PUBLISHED:
        names = PANEL_ARMS[model_key].get('main', ())
        pub = published_panel()
        sel = pub[(pub.arm == arm) & (pub.model.isin(names))].sort_values('index')
        return subscale(sel) if len(sel) else None
    p = f'{d}/{spec}'
    return subscale(pd.read_csv(p).sort_values('index')) if os.path.exists(p) else None


def table(corpora=CORPORA, out_csv=f'{OUT}/corpus_rag_panel.csv',
          out_tex=f'{OUT}/corpus_rag_panel.tex'):
    """New retrieval arms against the published ones, all on the two-judge subscale."""
    rows = []
    for key, files in PANEL_ARMS.items():
        model = next((m for m in MODELS if slug(m).endswith(key)), None)
        arms = []
        for label, (_, d) in corpora.items():
            p = path_for(model, d) if model else None
            arms.append((f'rag_{label}',
                         subscale(pd.read_csv(p).sort_values('index'))
                         if p and os.path.exists(p) else None))
        row = dict(model=key)
        base = baseline_scores(key, 'no_rag')
        # The oracle arm stores all three judges, so it can be put on this
        # table's two-judge subscale rather than quoted from the three-judge
        # table -- same essays, same judges, so the columns are comparable and
        # the oracle reads as what it is: the ceiling retrieval is aiming at.
        for name, s in ([('no_rag', base)] + arms
                        + [('oracle', baseline_scores(key, 'oracle'))]):
            if s is None:
                row[name] = np.nan
                continue
            row[name] = np.nanmean(s)
            row[f'{name}_sem'] = float(pd.Series(s).sem())
            if base is not None and name != 'no_rag':
                paired = pd.Series(s) - pd.Series(base)
                paired = paired.dropna()
                row[f'{name}_delta'] = paired.mean()
                row[f'{name}_delta_sem'] = paired.sem()
        rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print('\n=== retrieval over the rebuilt corpus, two free judges ===')
    print('(medians over %s; gpt-5-nano deliberately excluded, so these are NOT\n'
          ' comparable to the three-judge numbers in oracle_panel.csv)'
          % ', '.join(m.split('/')[-1] for m in FREE_PANEL))
    cols = [f'rag_{l}' for l in corpora] + ['oracle']
    hdr = f'{"model":<30}{"no-RAG":>10}' + ''.join(f'{c:>20}' for c in cols)
    print(hdr)
    print('-' * len(hdr))
    for _, r in df.iterrows():
        line = f'{r.model:<30}{fmt(r.get("no_rag"), r.get("no_rag_sem")):>10}'
        for c in cols:
            line += f'{fmt_delta(r.get(f"{c}_delta"), sem=r.get(f"{c}_delta_sem")):>20}'
        print(line)
    print(f'\nwritten to {out_csv}')
    return df


def main(cmd='run', corpora=CORPORA, top_k=10, models=MODELS, max_tokens=None,
         device='cuda', limit=None, pipeline='roundrobin', style='plain',
         leitsatz_kb=None, provider=None, store_as=None, concurrency=None, effort=None,
         endpoint=None, effort_prefix=None):
    if cmd == 'table':
        table(corpora)
        return
    api_keys()
    todo = [(label, kb, d, m) for label, (kb, d) in corpora.items() for m in models]
    for n, (label, kb, d, model) in enumerate(todo, 1):
        print(f'\n{"#" * 70}\n# [{n}/{len(todo)}] {model} over {label} '
              f'({pipeline}, {style})\n{"#" * 70}', flush=True)
        try:
            p = generate(model, kb, top_k, d, max_tokens, device, limit, pipeline,
                         style, leitsatz_kb, provider=provider, store_as=store_as,
                         concurrency=concurrency, effort=effort, endpoint=endpoint,
                         effort_prefix=effort_prefix)
            if cmd == 'run':
                judge(p, provider=provider)
        except Exception as e:  # one dead endpoint should not take the queue with it
            print(f'!! {model} / {label} failed: {type(e).__name__}: {e}', flush=True)
    table(corpora)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('cmd', choices=['run', 'generate', 'judge', 'table'], default='run',
                    nargs='?')
    ap.add_argument('--corpora', nargs='*', default=list(KBS), choices=list(KBS),
                    help='which rebuilt corpora to generate over (default: both)')
    ap.add_argument('--pipeline', default='roundrobin', choices=list(PIPELINES),
                    help='how sub-query result lists are combined')
    ap.add_argument('--judges', default='free', choices=list(PANELS),
                    help="'free' is the two NHR@FAU seats and costs nothing; "
                         "'panel' adds gpt-5-nano at ~$0.005/essay and needs --yes")
    ap.add_argument('--yes', action='store_true',
                    help='confirm a billed judging run')
    ap.add_argument('--leitsatz-index', default=None,
                    help='Leitsatz index the two-hop pipelines resolve through. '
                         'v2 carries 71,207 rules against v1 38,966; a non-default '
                         'index suffixes the output directory so the arm cannot land '
                         'on top of its v1 twin and be silently reused.')
    ap.add_argument('--style', default='plain', choices=list(CONTEXT_STYLES),
                    help='how the retrieved passages are presented to the generator')
    ap.add_argument('--kb', default=None,
                    help='override the knowledge base; requires a single --corpora')
    ap.add_argument('--top-k', type=int, default=10,
                    help='passages handed to the generator (the published run used 5)')
    ap.add_argument('--out-dir', default=None, help='override the output directory')
    ap.add_argument('--models', nargs='*', default=list(MODELS))
    ap.add_argument('--max-tokens', type=int, default=None)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=None,
                    help='cap the number of cases -- a smoke test, not a result')
    ap.add_argument('--store-as', default=None,
                    help='write the arm under this model name instead of the one '
                         'generated with -- for serving the same weights through a '
                         'fallback endpoint without renaming the arm. One --models '
                         'only.')
    ap.add_argument('--concurrency', type=int, default=None,
                    help='requests in flight; the Generator default is 10')
    ap.add_argument('--pin-deepinfra', action='store_true',
                    help='pin OpenRouter routing to DeepInfra, the validated '
                         'fallback for the DeepSeek seat; no effect on an '
                         'NHR@FAU endpoint')
    ap.add_argument('--effort', default=None,
                    help="reasoning_effort for generation (default: src/llm.py's 'high'). "
                         "DeepSeek-V4-Flash on DeepInfra: 'low' is the regime NHR@FAU "
                         "served the August ladder at; see DS4-ablation-rerun.md")
    ap.add_argument('--endpoint', default=None, choices=['gwdg'],
                    help='generate on the GWDG academic cloud (ENDPOINT_AC / API_KEY_AC); '
                         'pass the model as GWDG names it, deepseek-v4-flash-0731, with --store-as')
    ap.add_argument('--effort-prefix', default=None, choices=list(EFFORT_PREFIX),
                    help="DeepSeek's own effort prefix as the system prompt; with --effort low "
                         "this is DeepSeek's real high where the server drops the prefix")
    a = ap.parse_args()
    corpora = {k: (KBS[k], out_dir(k, a.pipeline, a.style, a.top_k, a.leitsatz_index))
               for k in a.corpora}
    if a.kb or a.out_dir:
        if len(corpora) != 1:
            ap.error('--kb/--out-dir override a single corpus; pass one --corpora')
        (label, (kb, d)), = corpora.items()
        corpora = {label: (a.kb or kb, a.out_dir or d)}
    if a.cmd == 'judge':
        api_keys()
        panel = PANELS[a.judges]
        todo = [p for _, d in corpora.values() for m in a.models
                for p in [path_for(m, d)] if os.path.exists(p)]
        billed = [(p, [j for j in missing_judges(p, panel) if j in COST_PER_ESSAY])
                  for p in todo]
        cost = sum(COST_PER_ESSAY[j] * len(pd.read_csv(p, usecols=[0]))
                   for p, js in billed for j in js)
        if cost:
            print(f'{sum(len(js) for _, js in billed)} billed judge pass(es) over '
                  f'{sum(1 for _, js in billed if js)} file(s): est. ${cost:.2f}', flush=True)
            if not a.yes:
                raise SystemExit('refusing to spend without --yes')
        provider = DEEPINFRA if a.pin_deepinfra else None
        for p in todo:
            judge(p, panel=panel, provider=provider)
        table(corpora)
    else:
        if a.store_as and len(a.models) != 1:
            ap.error('--store-as renames a single arm; pass one --models')
        main(a.cmd, corpora, a.top_k, a.models, a.max_tokens, a.device, a.limit,
             a.pipeline, a.style, a.leitsatz_index,
             provider=DEEPINFRA if a.pin_deepinfra else None,
             store_as=a.store_as, concurrency=a.concurrency, effort=a.effort,
             endpoint=a.endpoint, effort_prefix=a.effort_prefix)

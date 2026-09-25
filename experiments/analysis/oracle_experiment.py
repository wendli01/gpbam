"""The oracle retrieval condition, end to end: generate, judge, tabulate.

The one experiment that separates "retrieval cannot find the norms" from "the
norms do not help".  Instead of searching, the context is assembled from the
reference solution's own citations, restricted to those present in the corpus --
so it is the best case a retriever over *this* corpus could reach, not an
unattainable one.

Three steps, three subcommands:

``run``
    Generate the oracle answers for one model, then re-judge the published
    no-RAG and with-RAG answers for the same model with the same judge, so the
    three conditions are on one scale.  **Calls the generation and judge APIs.**
    Roughly \\$0.14-2.45 of generation per model over the 81 cases, plus judging.
    Appends its summary to ``out/oracle_three_way.csv``.

``rejudge``
    Re-score answers already stored by ``run`` with the two ensemble judges
    reachable on NHR@FAU, writing ``<name>_panel.csv`` next to each.  This is
    what puts the oracle numbers on the paper's own scale; ``run`` alone leaves
    each model on its judge's scale.

``table``
    Read the stored answers and write both paper tables --
    ``out/oracle_condition.tex`` (single judge, with paired t-tests) and
    ``out/oracle_panel.{csv,tex}`` (the ensemble scale).  Offline.

Run from ``experiments/``::

    PYTHONPATH=analysis python analysis/oracle_experiment.py run --model qwen36-35b --gold gold
    PYTHONPATH=analysis python analysis/oracle_experiment.py rejudge --arm gold
    PYTHONPATH=analysis python analysis/oracle_experiment.py table
"""

import argparse
import glob
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

sys.path.insert(0, os.path.abspath('..'))

import refs as refs_mod
from retrieval_diagnostics import refex_norms, GPBAM, OUT

D = 'zubaers_result/essay_writing/oracle_rag/ji2'
#: The oracle reads the same corpus the retrieval arms do. It used to read
#: ./my_knowledge_base, which is federal-only, patched with the 100-norm
#: landesrecht.json sample -- and that sample contains no GO and no BV, two of
#: the four most-cited Bavarian books at 293 and 253 gold citations. So the
#: upper bound was crippled on exactly the axis the corpus section argues is
#: binding: 35.0 of 51.9 cited norms per case and 34.4% of the state-law ones,
#: against 47.4 and 94.9% here. Arms generated before this change carry the old
#: corpus; they are tagged separately rather than overwritten.
KB = './my_knowledge_base_bayern_titled'
LANDESRECHT = 'analysis/out/landesrecht.json'
NORAG_CSV = 'zubaers_result/essay_writing/without_rag/ji2/no_rag_ji2_result.csv'
SUMMARY = f'{OUT}/oracle_three_way.csv'
#: fixed, so --shuffle-refs is reproducible run to run
SHUFFLE_SEED = 20260818

#: The paper's three-judge ensemble.  Two are free on NHR@FAU; ``gpt-5-nano`` is
#: billed on OpenRouter at roughly \$0.005 per essay -- it answers with ~9k
#: reasoning tokens under the repo's default ``reasoning_effort='high'``, which
#: is what the published run used, so do not lower it here or the levels stop
#: matching Table~\ref{tab:main_results}.
PANEL = ('Qwen/Qwen3.6-35B-A3B-FP8', 'deepseek-ai/DeepSeek-V4-Flash', 'openai/gpt-5-nano')
#: rough OpenRouter cost per essay for each billed judge, for the run estimate.
#: The DeepSeek slug is the DeepInfra-pinned fallback for the NHR@FAU seat (see
#: endpoints.yaml); it belongs here because without it ``rejudge`` estimates a
#: real ~$1.50 pass at $0.00 and waives its own --yes guard. $0.004 is
#: 22.8k prompt tokens at $0.06/M plus ~13k output at $0.18/M, the midpoint of
#: the 7.5k (FAU) to 19k (high effort) output range measured for this seat.
COST_PER_ESSAY = {'openai/gpt-5-nano': 0.005,
                  'deepseek/deepseek-v4-flash-0731': 0.004}
#: the two NHR@FAU seats, which cost nothing
FREE_PANEL = tuple(m for m in PANEL if m not in COST_PER_ESSAY)
#: see corpus_rag_run.FREE_PANEL3 -- a third free seat, with the caveat that
#: gpt-oss is also a generator here
FREE_PANEL3 = FREE_PANEL + ('openai/gpt-oss-120b',)
PANELS = {'free': FREE_PANEL, 'free3': FREE_PANEL3, 'panel': PANEL}


def judge_col(model):
    return f'score_Judge ({model.split("/")[-1]})'


PANEL_COLS = tuple(judge_col(m) for m in PANEL)

#: The qwen ensemble judge appears under two column names in the published runs,
#: one per deployment that served it -- OpenRouter's ``qwen/qwen3.6-35b-a3b`` and
#: NHR@FAU's ``Qwen/Qwen3.6-35B-A3B-FP8``.  Same judge model, so either column
#: satisfies that seat on the panel.
JUDGE_ALIASES = {
    'score_Judge (Qwen3.6-35B-A3B-FP8)': ('score_Judge (Qwen3.6-35B-A3B-FP8)',
                                          'score_Judge (qwen3.6-35b-a3b)'),
}


def panel_frame(df):
    """The three judges' per-essay scores, or ``None`` if a seat is unfilled."""
    out = {}
    for c in PANEL_COLS:
        for alias in JUDGE_ALIASES.get(c, (c,)):
            if alias in df.columns and df[alias].notna().any():
                out[c] = df[alias].values
                break
        else:
            return None
    return pd.DataFrame(out)


def api_keys():
    """endpoints.yaml expects these names; .env in this checkout uses *_UP / *_FAU."""
    load_dotenv('../.env', override=True)
    load_dotenv(override=True)
    # API_KEY_OR is the funded OpenRouter key; API_KEY is an older one that may
    # be out of credit, so it is only the fallback.
    if 'API_KEY_OR' in os.environ:
        os.environ['openrouter_api'] = os.environ['API_KEY_OR']
    for env_name, yaml_name in (('API_KEY_UP', 'innkube_api'),
                                ('API_KEY_FAU', 'nhr_fau_api'),
                                ('API_KEY', 'openrouter_api')):
        if env_name in os.environ:
            os.environ.setdefault(yaml_name, os.environ[env_name])


def cases():
    d = json.load(open(GPBAM))
    order = sorted(d['solutions'], key=lambda x: int(x))
    return [d['facts'][k] for k in order], [d['solutions'][k] for k in order]


def shuffle_ranked(ranked, seed=SHUFFLE_SEED):
    """Permute each case's reference order, seeded per case.

    ``gold_sets`` orders the norms by how often the reference solution cites
    them, so the oracle context hands over a prominence ranking on top of the
    selection: not only which provisions the case turns on but which of them
    carry the argument, in descending order. That is a channel no real
    retriever has, because it is derived from a solution that does not exist
    yet, and it has never been ablated -- ``cites`` and ``unc`` both inherit it.

    Shuffling changes presentation order and nothing else, as long as the
    passage cap does not bind; ``run`` checks that and refuses if it does,
    since a binding cap would let the order decide *membership* too and the
    arm would no longer be a single-factor manipulation.
    """
    out = {}
    for i, order in ranked.items():
        order = list(order)
        random.Random(seed + i).shuffle(order)
        out[i] = order
    return out


def gold_sets(sols, gold_set='gold'):
    """Cited norms per case, and the order to spend the passage budget in.

    'refex' matches the published legal_ref_sim universe (20.8 norms/case);
    'gold' also sees the Art. form, i.e. GG and the Bavarian codes, and so
    reaches 53.1 norms/case -- about half of which the corpus cannot supply.
    Ranked by citation count, so a binding cap keeps the norms the solution
    leans on rather than an alphabetical slice.
    """
    gold, ranked = {}, {}
    for i, s in enumerate(sols):
        g = (refex_norms(s) if gold_set == 'refex'
             else {(refs_mod.corpus_key(x.book), x.section.lower())
                   for x in refs_mod.canonicalise(refs_mod.extract(s))})
        gold[i] = g
        c = Counter((refs_mod.corpus_key(x.book), x.section.lower())
                    for x in refs_mod.canonicalise(refs_mod.extract(s)))
        ranked[i] = ([n for n, _ in c.most_common() if n in g]
                     + [n for n in sorted(g) if n not in c])
    return gold, ranked


# --------------------------------------------------------------------------
# 1. run


class OracleRetriever:
    """The solution's own norms, in the shape ``AnswerGenerator`` expects.

    ``AnswerGenerator.predict`` calls ``retriever.predict(X)`` with the list of
    case facts and formats whatever comes back through ``format_context``, so a
    drop-in object with a ``predict`` method is all that is needed.
    """

    def __init__(self, facts, gold, kb=KB, max_passages=20, max_chars=1500,
                 verbose=True, ranked=None, include_uncovered=False,
                 landesrecht=LANDESRECHT, cites_only=False, cites_from=None,
                 pad_junk=0, pad_from=None):
        import lancedb
        t = lancedb.connect(kb).open_table('documents')
        df = t.to_lance().to_table(
            columns=['text', 'title', 'law_book', 'paragraph']).to_pandas()
        df['key'] = df.law_book.map(refs_mod.corpus_key)
        df['sect'] = df.paragraph.map(
            lambda p: (lambda m: re.sub(r'\s+', '', m.group(1)).lower() if m else None)(
                re.search(r'(?:§+|Art\.?)\s*(\d+\s?[a-z]?)', str(p))))
        self.by_norm = {}
        for r in df.itertuples():
            if r.sect and (r.key, r.sect) not in self.by_norm:
                self.by_norm[(r.key, r.sect)] = dict(
                    text=str(r.text)[:max_chars], title=r.title,
                    law_book=r.law_book, paragraph=r.paragraph)
        # State law fetched from BAYERN.RECHT: the federal corpus has none, and
        # roughly half of what these cases cite is Bavarian.
        if landesrecht and os.path.exists(landesrecht):
            extra = 0
            for key, row in json.load(open(landesrecht)).items():
                if not row:
                    continue
                book, sect = key.split('|', 1)
                if (book, sect) not in self.by_norm:
                    self.by_norm[(book, sect)] = dict(
                        text=row['text'][:max_chars], title=row['title'],
                        law_book=row['law_book'], paragraph=row['paragraph'])
                    extra += 1
            if verbose:
                print(f'loaded {extra} state-law norms from {landesrecht}')

        self.index_of = {f: i for i, f in enumerate(facts)}
        self.gold = gold
        # Dilution. Every arm so far is either all gold (the oracle) or mostly
        # junk (a real pipeline, ~1.7 of 10 norms on target), and the two differ
        # in far more than purity. Padding the oracle with norms the solution
        # does not cite holds the gold set fixed and varies only how much sits
        # around it, which is the one manipulation that prices junk while the
        # effect being diluted is known to be real (+6.6).
        #
        # The junk is lifted from a real retrieval arm rather than drawn at
        # random from the corpus: random norms would be obviously off-topic --
        # tax law in a building case -- and a judge or a generator can discard
        # those for free. What a pipeline actually returns is topically adjacent
        # and therefore the hard case.
        self.pad_junk = pad_junk
        self.pad_from = {}
        if pad_junk:
            if not pad_from:
                raise SystemExit('--pad-junk needs --pad-from: a stored arm CSV '
                                 'whose retrieved norms supply the junk')
            d = pd.read_csv(pad_from).sort_values('index')
            for idx, q in zip(d['index'], d['qa_prompt']):
                q = str(q)
                cites = (re.findall(r'<norm zitat="([^"]*)"', q)
                         or re.findall(r'\[SOURCE \d+: (.*?) \(', q))
                keys = []
                for cite in cites:
                    for c in refs_mod.canonicalise(refs_mod.extract(cite)):
                        k = (refs_mod.corpus_key(c.book), c.section.lower())
                        if k not in keys:
                            keys.append(k)
                self.pad_from[int(idx)] = keys
            have = [len(v) for v in self.pad_from.values()]
            if not sum(have):
                raise SystemExit(f'{pad_from}: no citations parsed out of qa_prompt')
            # the binding quantity is junk that is both non-gold for this case
            # and present in the corpus, not either half on its own
            short = [i for i, v in self.pad_from.items()
                     if len([k for k in v if k not in self.gold.get(i, ())
                             and k in self.by_norm]) < pad_junk]
            if verbose:
                print(f'padding with {pad_junk} non-cited norm(s) per case from '
                      f'{os.path.basename(pad_from)} '
                      f'({sum(have) / len(have):.1f} available per case'
                      + (f', {len(short)} case(s) short)' if short else ')'))
        self.ranked = ranked or {i: sorted(g) for i, g in gold.items()}
        self.max_passages = max_passages
        # Norms the reference solution cites that the corpus does not contain --
        # overwhelmingly state law. Their text cannot be supplied, but naming
        # them still hands over the issue-spotting, which is the thing a
        # retriever is supposed to do.
        self.include_uncovered = include_uncovered
        # Names the same norms the text arm supplies, and supplies none of their
        # text. The oracle's gain confounds two channels: the statute wording,
        # and the *selection* -- which provisions this case turns on, which is
        # the reference solution's issue list and so the skeleton of the
        # argument. Naming them without their text isolates the second. Paired
        # against the text oracle it has to name exactly the covered set (~45 of
        # ~52 per case), not all of them, or selection breadth differs too and
        # the contrast is unreadable; combine with ``include_uncovered`` for the
        # full citation list with no text anywhere.
        self.cites_only = cites_only
        # Exact-match mode. gold_sets() runs through refs.canonicalise(), and
        # that has drifted: the stored gold oracle carries 45.4 norms per case
        # and rebuilding it with today's refs.py yields 35.0. Pairing a new
        # citations-only arm against the stored text arm under those terms
        # would vary selection breadth alongside the text, which is precisely
        # the confound the arm exists to remove -- so the citation list can be
        # lifted verbatim out of the stored run's own prompt instead.
        self.cites_from = None
        if cites_from:
            d = pd.read_csv(cites_from).sort_values('index')
            # Two context formats exist in the stored runs. Anything generated
            # before format_context started emitting <norm> elements carries
            # `[SOURCE i: BauNVO § 15 (...)]`; current runs carry
            # `<norm zitat="§ 15 BauNVO" ...>`. Both are read, because the arm
            # this lifts from may be either -- and a regex that matches neither
            # returns an empty list, which would quietly turn this into a
            # no-context arm that still scores and still looks like a result.
            def cites_of(q):
                q = str(q)
                return (re.findall(r'\[SOURCE \d+: (.*?) \(', q)
                        + re.findall(r'<norm zitat="([^"]*)"', q))

            self.cites_from = {int(i): cites_of(q)
                               for i, q in zip(d['index'], d['qa_prompt'])}
            n = [len(v) for v in self.cites_from.values()]
            if not sum(n):
                raise SystemExit(
                    f'{cites_from}: no citations found in qa_prompt -- the context is '
                    'in neither the [SOURCE ...] nor the <norm zitat=...> format, so '
                    'this arm would have run with an empty context')
            if verbose:
                print(f'citation list lifted from {os.path.basename(cites_from)}: '
                      f'{sum(n) / len(n):.1f} norms per case')
        if verbose:
            covered = [len([n for n in gold[i] if n in self.by_norm]) for i in gold]
            print(f'oracle context: {sum(covered) / len(covered):.1f} of '
                  f'{sum(len(g) for g in gold.values()) / len(gold):.1f} cited norms per case '
                  f'are in the corpus and will be supplied '
                  f'(capped at {max_passages})')

    def predict(self, X):
        out = []
        for facts in X:
            i = self.index_of[facts]
            rows = [self.by_norm[n] for n in self.ranked[i] if n in self.by_norm]
            rows = rows[:self.max_passages]
            if self.pad_junk:
                seen = {(r['law_book'], r['paragraph']) for r in rows}
                junk = [self.by_norm[n] for n in self.pad_from.get(i, [])
                        if n not in self.gold[i] and n in self.by_norm
                        and (self.by_norm[n]['law_book'],
                             self.by_norm[n]['paragraph']) not in seen]
                rows = rows + junk[:self.pad_junk]
                # interleave, or the junk is a tail the model can simply stop
                # reading and the dilution is not a dilution. Same seed scheme as
                # --shuffle-refs, so the padded arm pairs against the shuffled
                # oracle and order is held constant across the pair.
                random.Random(SHUFFLE_SEED + i).shuffle(rows)
            if self.cites_only:
                cites = (', '.join(self.cites_from[i]) if self.cites_from is not None
                         else ', '.join(f'{r["law_book"]} {r["paragraph"]}' for r in rows))
                rows = [dict(
                    law_book='EINSCHLAEGIGE NORMEN',
                    paragraph='(nur Fundstellen)',
                    title='vom Gutachten zu pruefende Vorschriften',
                    text='Die folgenden Vorschriften sind fuer diesen Fall '
                         'einschlaegig; ihr Wortlaut liegt hier nicht vor: '
                         + cites)] if cites else []
            if self.include_uncovered:
                missing = [n for n in self.ranked[i] if n not in self.by_norm]
                if missing:
                    cites = ', '.join(f'{b.upper()} {s}' for b, s in missing)
                    rows.append(dict(
                        law_book='WEITERE EINSCHLAEGIGE NORMEN',
                        paragraph='(Volltext nicht verfuegbar)',
                        title='vom Gutachten zu pruefende Vorschriften',
                        text='Die folgenden Vorschriften sind fuer diesen Fall ebenfalls '
                             'einschlaegig; ihr Wortlaut liegt hier nicht vor: ' + cites))
            out.append(rows)
        return out


#: cases per chunk. ``evaluate_model`` generates and judges a whole list before
#: returning anything, so an unchunked call is atomic: a Kimi-class model at
#: ~7 min/essay spends nine hours in one call and a death at hour eight leaves
#: nothing. ``corpus_rag_run.generate`` learned this the hard way -- its comment
#: records losing about eighty essays in one afternoon -- and this is the same
#: fix, bounding a loss to one chunk and making a restart resume.
#:
#: Twenty was too coarse to survive a supervisor that caps an attempt. A
#: watchdog running ``run`` under ``timeout`` banks nothing until the first
#: chunk closes, so at the ~10 min/case FAU has served DeepSeek at lately, a
#: 20-case chunk needs 3h20 and a 45-minute attempt writes no ``.part`` at
#: all -- twelve retries of the top-10 arm produced exactly nothing on 09-10,
#: and five more on 09-11. Five bounds the loss to under an hour at that rate
#: and still amortises the write, which is a few seconds against minutes of
#: generation. ``--chunk`` overrides it for a model that is genuinely fast.
GEN_CHUNK = 5
#: client timeout for one oracle essay, in seconds -- see ``run``. Four hours is
#: one 39k-token essay at 3 tok/s, and stays under the supervisors' 6h cap.
GEN_TIMEOUT = 4 * 3600


def generate_chunked(gen, facts, sols, judge, model, out_path, chunk=GEN_CHUNK):
    """``evaluate_model`` over chunks, with a ``.part`` file between them."""
    from src import evaluate
    part = f'{out_path}.part'
    frames, start = [], 0
    if os.path.exists(part):
        prev = pd.read_csv(part)
        # only a prefix is trustworthy: cases are generated in task order
        start = int(prev['index'].max()) + 1 if len(prev) else 0
        frames = [prev]
        print(f'  resuming from {os.path.basename(part)}: {start} case(s) done',
              flush=True)
    for i in range(start, len(facts), chunk):
        df = evaluate.evaluate_model(gen, facts[i:i + chunk], sols[i:i + chunk],
                                     judge=judge, verbose=True, model_name=model)
        # evaluate_model numbers each call from zero; shift onto task position
        df['index'] = range(i, i + len(df))
        frames.append(df)
        pd.concat(frames, ignore_index=True).to_csv(f'{part}.tmp', index=False)
        os.replace(f'{part}.tmp', part)
        print(f'  {min(i + chunk, len(facts))}/{len(facts)} generated', flush=True)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(out_path, index=False)
    if os.path.exists(part):
        os.remove(part)
    return out


def run(model, judge_model, judge_instruction='ji2', max_passages=20, out_dir=D,
        max_tokens=None, existing_model=None, gold_set='gold',
        include_uncovered=False, tag='', fresh_baseline=False, dry_run=False,
        qa_prompt='qa1', cites_only=False, cites_from=None,
        shuffle_refs=False, pad_junk=0, pad_from=None, kb=KB,
        chunk=GEN_CHUNK, concurrency=None):
    # the endpoint's identifier need not match the name the published runs stored:
    # InnKube serves 'qwen35-397b', the CSVs call it 'qwen/qwen3.5-397b-a17b'
    existing_model = existing_model or model
    api_keys()

    facts, sols = cases()
    gold, ranked = gold_sets(sols, gold_set)
    if shuffle_refs:
        binding = [i for i, g in gold.items() if len(g) > max_passages]
        if binding:
            raise SystemExit(
                f'--shuffle-refs with --max-passages {max_passages}: the cap binds on '
                f'{len(binding)} case(s) (largest gold set {max(len(g) for g in gold.values())}), '
                'so the order would decide which norms are supplied and the arm would '
                'confound order with selection. Raise --max-passages.')
        ranked = shuffle_ranked(ranked)
        print(f'reference order shuffled per case, seed {SHUFFLE_SEED}', flush=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = model.replace('/', '_') + tag
    oracle_path = out_dir / f'oracle_{slug}.csv'

    if dry_run:
        ret = OracleRetriever(facts, gold, kb=kb, max_passages=max_passages, ranked=ranked,
                              include_uncovered=include_uncovered,
                              cites_only=cites_only, cites_from=cites_from,
                              pad_junk=pad_junk, pad_from=pad_from)
        ctx = ret.predict(facts[:1])[0]
        print(f'\nexample context for case 0 -- {len(ctx)} passages:')
        for r in ctx:
            print(f'  {r["law_book"]} {r["paragraph"]}  ({len(r["text"])} chars)')
        print('\ndry run: no API calls made')
        return

    from src import qa, scoring, prompts, evaluate
    # Generator defaults to max_concurrency=10. OpenRouter holds its per-stream
    # rate well past that, and the oracle arms are long: 27k completion tokens
    # per essay measured, 20 minutes each, so ten at a time is a 5-hour arm and
    # forty is a 1.3-hour one. Raise it only for endpoints that fan out.
    conc = {'max_concurrency': concurrency} if concurrency else {}
    # The Generator's 300 s client timeout is shorter than one oracle essay
    # (30-39k tokens on NHR@FAU). Its gateway keeps generating a request the
    # client abandoned and starts a new copy for every resend, so a short
    # timeout multiplied our own load on the deployment: what looked like a
    # slow DeepSeek on 09-11 (3.4 tok/s, 29 tok/s aggregate) was mostly our
    # duplicates. One request per essay, however long it takes.
    conc['timeout'] = GEN_TIMEOUT
    # ``AnswerGenerator`` swaps in the RAG prompt itself, but only when it is
    # still holding the v1 default, so both variants are named explicitly here.
    qa_plain, qa_rag = prompts.QA_PROMPTS_BY_NAME[qa_prompt]
    instruction = prompts.JUDGE_INSTRUCTIONS_BY_NAME[judge_instruction]
    judge = scoring.JudgeEnsemble(
        [scoring.Judge(model=judge_model, prompt=prompts.build_judge_user(instruction))],
        verbose=True)

    # --- generate the oracle condition
    if oracle_path.exists():
        print(f'reusing {oracle_path}')
        oracle = pd.read_csv(oracle_path)
    else:
        ret = OracleRetriever(facts, gold, kb=kb, max_passages=max_passages, ranked=ranked,
                              include_uncovered=include_uncovered,
                              cites_only=cites_only, cites_from=cites_from,
                              pad_junk=pad_junk, pad_from=pad_from)
        gen = qa.AnswerGenerator(retriever=ret, model=model, max_tokens=max_tokens,
                                 prompt=qa_rag, **conc)
        print(f'\n=== generating oracle answers with {model} ===')
        oracle = generate_chunked(gen, facts, sols, judge, model,
                                  str(oracle_path), chunk=chunk)
        print(f'written to {oracle_path}')

    # --- re-judge the two existing conditions with the same judge
    scores = {}
    judge_col = f'score_Judge ({judge_model.split("/")[-1]})'
    scores['oracle'] = oracle[judge_col]

    # A baseline generated on a different host than the oracle arm would confound
    # the condition with the deployment. When the stored answers come from another
    # endpoint, regenerate the no-retrieval arm here instead.
    if fresh_baseline:
        # tagged like the oracle arm above: a variant run (a different prompt,
        # say) needs its own baseline, and writing to the untagged name would
        # either silently reuse the default-prompt one or overwrite it
        fresh_path = out_dir / f'norag_fresh_{slug}.csv'
        if fresh_path.exists():
            print(f'reusing {fresh_path}')
            fresh = pd.read_csv(fresh_path)
        else:
            print(f'\n=== generating a same-host no-retrieval baseline with {model} ===')
            plain = qa.AnswerGenerator(model=model, max_tokens=max_tokens,
                                       prompt=qa_plain, **conc)
            fresh = generate_chunked(plain, facts, sols, judge, model,
                                     str(fresh_path), chunk=chunk)
        scores['no_rag'] = fresh[judge_col]

    for name, path in (('no_rag', NORAG_CSV),):
        if name in scores:
            continue
        cache = out_dir / f'rejudged_{name}_{model.replace("/", "_")}.csv'  # judge-only, shared across variants
        if cache.exists():
            print(f'reusing {cache}')
            df = pd.read_csv(cache)
        else:
            src = pd.read_csv(path)
            src = src[src.model == existing_model].sort_values('index')
            if not len(src):
                print(f'!! {existing_model} not present in {path}; skipping {name}')
                continue
            print(f'\n=== re-judging {name} for {existing_model} '
                  f'({len(src)} essays) with {judge_model} ===')
            df = evaluate.rejudge_model(src.answer.tolist(), facts, sols, judge=judge,
                                        model_name=model, verbose=True)
            df.to_csv(cache, index=False)
        scores[name] = df[judge_col]

    # --- report
    print(f'\n=== {model}, judged by {judge_model} on {judge_instruction} ===')
    base = scores.get('no_rag')
    rows = []
    for name in ('no_rag', 'oracle'):
        if name not in scores:
            continue
        s = pd.to_numeric(scores[name], errors='coerce') * 100
        # tag and qa_prompt are in the row because without them the log is
        # ambiguous: five runs on one evening appended fifteen rows all labelled
        # no_rag/with_rag/oracle, differing only in a prompt and a context
        # variant that nothing recorded. The condition name alone does not
        # identify a run.
        row = dict(model=model, gold_set=gold_set, condition=name,
                   tag=tag or '', qa_prompt=qa_prompt,
                   cites_only=bool(cites_only), include_uncovered=bool(include_uncovered),
                   n=int(s.notna().sum()), mean=s.mean(), sem=s.sem())
        if base is not None and name != 'no_rag':
            paired = (s - pd.to_numeric(base, errors='coerce') * 100).dropna()
            row['delta_vs_norag'] = paired.mean()
            row['delta_sem'] = paired.sem()
        rows.append(row)
    res = pd.DataFrame(rows)
    print(res.round(2).to_string(index=False))
    # append: one table for every run, newest last.  Re-running a config adds
    # rows rather than replacing them, so the file is a log -- take the last
    # rows per (model, gold_set) if you only want the current numbers.
    res.to_csv(SUMMARY, mode='a', header=not os.path.exists(SUMMARY), index=False)
    print(f'\nappended to {SUMMARY}')


# --------------------------------------------------------------------------
# 2. rejudge


def arm_of(path):
    """``gold``, ``refex`` or ``other`` -- the oracle reference set an arm used."""
    base = os.path.basename(path)
    if base.endswith('_refex.csv'):
        return 'refex'
    if base.endswith('_gold.csv'):
        return 'gold'
    return 'other'


def targets(d=D, arm='all'):
    """Stored answer files that need panel scores, gold arms first.

    ``arm='table'`` is every file the panel table actually reads, wherever it
    lives: the oracle arms produced here *and* the re-judged baselines they are
    compared against.  A judge added to only one arm would put the arms on
    different scales, so the whole table has to move together.

    The other selectors glob the raw generations instead.  The refex oracle only
    illustrates what the metric cannot see, so it is the arm to lose if the run
    has to be cut short.
    """
    if arm == 'table':
        seen, out = set(), []
        for files in PANEL_ARMS.values():
            for a in ('no_rag', 'oracle'):
                f = files.get(a)
                if f and f != PUBLISHED and f not in seen:
                    seen.add(f)
                    out.append(f'{d}/{f}')
        return [p for p in out if os.path.exists(p)]

    # rejudged_no_rag_* belongs here for the same reason norag_fresh_* does: it
    # is a no-retrieval baseline the ladder is measured against. Leaving it out
    # of the glob is what let two of them keep a single seat unnoticed, and a
    # baseline on a different scale than its arms turns a seat shift into an
    # apparent retrieval effect.
    out = [p for p in sorted(glob.glob(f'{d}/*.csv'))
           if not os.path.basename(p).endswith('_panel.csv')
           and os.path.basename(p).startswith(
               ('oracle_', 'norag_fresh_', 'rejudged_no_rag_'))]
    if arm != 'all':
        out = [p for p in out if arm_of(p) == arm]
    return sorted(out, key=lambda p: (arm_of(p) == 'refex', p))


def panel_path(p):
    """Where the panel-scored version of ``p`` lives.

    A file that already carries a panel judge is topped up in place; a raw
    generation carrying only the old single judge gets a ``_panel.csv`` beside
    it, so the original scores stay readable next to the new ones.
    """
    if p.endswith('_panel.csv') or not os.path.exists(p):
        return p
    cols = set(pd.read_csv(p, nrows=0).columns)
    return p if cols & set(PANEL_COLS) else p.replace('.csv', '_panel.csv')


def missing_judges(p, panel=PANEL):
    """Panel judges that have not scored ``p`` yet.

    A seat counts as unscored when its column is absent *or* holds nothing.
    Testing presence alone is what let a failed judging pass look finished:
    the pass writes the column, every value comes back NaN, and from then on
    every retry skips the file because the column is there. Eleven cells of
    the ladder sat blank that way -- essays generated, two seats scored, the
    third an empty column no rejudge would ever refill. ``med`` refuses a
    column with no values, so nothing was ever printed wrong; it was simply
    never printed.
    """
    out = panel_path(p)
    if not os.path.exists(out):
        return list(panel)
    d = pd.read_csv(out)
    return [m for m in panel
            if judge_col(m) not in d.columns or not d[judge_col(m)].notna().any()]


def n_live(p):
    """Rows of ``p`` that hold an essay, i.e. the calls a seat really costs.

    ``Judge.predict`` skips a blank answer rather than scoring it zero, so an
    arm the FAU outage cut off mid-run is cheaper than its row count: _gold_pad
    is 55 essays in 81 rows. Multiplying by a literal 81 overstated both the
    printed estimate and the --yes threshold by up to a third.
    """
    src = panel_path(p) if os.path.exists(panel_path(p)) else p
    try:
        return int(pd.read_csv(src, usecols=['answer']).answer.notna().sum())
    except Exception:
        return len(cases()[0])


def rejudge(dry_run=False, panel=PANEL, judge_instruction='ji2', limit=None,
            arm='all', d=D, confirmed=False, provider=None, concurrency=None):
    api_keys()
    facts, sols = cases()

    todo = [(p, m) for p in targets(d, arm) for m in [missing_judges(p, panel)] if m]
    live = {p: n_live(p) for p, _ in todo}
    calls = sum(len(m) * live[p] for p, m in todo)
    cost = sum(COST_PER_ESSAY.get(j, 0) * live[p] for p, m in todo for j in m)
    print(f'{len(todo)} file(s) to score, {calls} judge calls, '
          f'est. ${cost:.2f} on OpenRouter', flush=True)
    for p, m in todo:
        print(f'  [{arm_of(p)}] {os.path.basename(p)}  <- {", ".join(m)}', flush=True)
    if dry_run:
        print('\ndry run: no API calls made')
        return
    if cost and not confirmed:
        raise SystemExit(f'refusing to spend ${cost:.2f} without --yes')
    if limit:
        todo = todo[:limit]

    from src import scoring, prompts, evaluate
    instruction = prompts.JUDGE_INSTRUCTIONS_BY_NAME[judge_instruction]

    for n, (p, models) in enumerate(todo, 1):
        out_path = panel_path(p)
        src = pd.read_csv(out_path if os.path.exists(out_path) else p).sort_values('index')
        print(f'\n=== [{n}/{len(todo)}] {os.path.basename(p)} ({len(src)} essays, '
              f'{arm_of(p)}) <- {", ".join(models)} ===', flush=True)
        # ``provider`` is OpenRouter routing, and only the OpenRouter branch of
        # llm.py reads it, so it is safe to pass for an FAU seat in the same
        # panel -- which is what a mixed free3 top-up is.
        #
        # ``concurrency`` matters more here than the per-call latency suggests.
        # One DeepSeek judge call on the oracle prompt takes ~300 s, so the
        # Generator's default of 10 puts a 358-call pass at three hours; the
        # calls are independent and the arm is one file, so the only thing that
        # shortens it is fanning out wider.
        kw = {'max_concurrency': concurrency} if concurrency else {}
        judge = scoring.JudgeEnsemble(
            [scoring.Judge(model=m, prompt=prompts.build_judge_user(instruction),
                           provider=provider, **kw)
             for m in models], verbose=True)
        df = evaluate.rejudge_model(src.answer.tolist(), facts, sols, judge=judge,
                                    model_name=str(src.model.iloc[0]), verbose=True)
        # Keep everything already stored and graft on only the new judges'
        # columns; `score`/`judging_cost` in ``df`` describe this pass alone.
        add = [c for c in df.columns
               if any(c.endswith(f'Judge ({m.split("/")[-1]})') for m in models)]
        merged = src.reset_index(drop=True).drop(columns=add, errors='ignore')
        for c in add:
            merged[c] = df[c].values
        assert len(merged) == len(src) and judge_col(models[-1]) in merged.columns
        # Most of these files are the only copy of a run that costs money to
        # reproduce, and several are topped up in place -- so never truncate the
        # original until a complete replacement is on disk.
        merged.to_csv(out_path + '.tmp', index=False)
        os.replace(out_path + '.tmp', out_path)
        have = [c for c in merged.columns if c in PANEL_COLS]
        med = merged[have].median(axis=1, skipna=True)
        print(f'  panel median over {len(have)} judge(s) {100 * med.mean():.2f}  '
              f'(+${df.judging_cost.sum():.2f})  ->  {os.path.basename(out_path)}',
              flush=True)


# --------------------------------------------------------------------------
# 3. table


SINGLE_JUDGE_COL = 'score_Judge (qwen36-35b)'
CONDITIONS = [('no_rag', 'No retrieval'), ('oracle', 'Oracle retrieval')]

#: Which stored answers hold which arm.  Only the oracle arm is generated here,
#: so only it has to be judged here: the published runs already carry all three
#: ensemble judges, and ``PUBLISHED`` reads the baselines straight out of them
#: instead of paying to re-score essays that are already on the paper's scale.
#:
#: ``main`` lists the names the published runs store for the same model -- they
#: do not match the oracle run's identifiers (InnKube serves ``qwen36-35b``, the
#: published CSVs call it ``Qwen/Qwen3.6-35B-A3B-FP8`` without RAG and
#: ``qwen/qwen3.6-35b-a3b`` with it).
#:
#: Two models name a local ``norag_fresh_*`` file instead of ``PUBLISHED``: their
#: oracle answers came from InnKube while the published baseline came from
#: NHR@FAU, and comparing across deployments would confound the condition with
#: the host.  ``None`` means the arm does not exist (gemma4-31b-it was never run
#: with RAG).
PUBLISHED = 'published'
PANEL_ARMS = {
    'DeepSeek-V4-Flash': dict(
        main=('deepseek-ai/DeepSeek-V4-Flash',),
        no_rag=PUBLISHED,
        oracle='oracle_deepseek-ai_DeepSeek-V4-Flash_gold_panel.csv'),
    'gpt-oss-120b': dict(
        main=('openai/gpt-oss-120b',),
        no_rag=PUBLISHED,
        oracle='oracle_openai_gpt-oss-120b_gold_panel.csv'),
    # Named for FAU's serving now, and drawing both arms from it. The row used
    # to pair a `main` of RedHatAI/... against an InnKube no_rag and an InnKube
    # oracle -- a host comparison wearing a condition's name, which is exactly
    # what the qwen36 comment below warns about. The move was forced by speed:
    # InnKube serves this model at 31-47 min per essay on the 120-passage oracle
    # prompt, 45-63 hours for one arm.
    'gemma-4-31B-it-FP8-block': dict(
        main=('RedHatAI/gemma-4-31B-it-FP8-block',),
        no_rag='norag_fresh_RedHatAI_gemma-4-31B-it-FP8-block_panel.csv',
        oracle='oracle_RedHatAI_gemma-4-31B-it-FP8-block_gold_combined.csv'),
    'qwen36-35b': dict(
        main=('Qwen/Qwen3.6-35B-A3B-FP8', 'qwen/qwen3.6-35b-a3b'),
        no_rag='norag_fresh_qwen36-35b_panel.csv',
        oracle='oracle_qwen36-35b_gold_panel.csv'),
    # The same base model served by NHR@FAU, kept as its own row rather than
    # merged into the one above: the two servings score 2.34 points apart on the
    # same 81 no-RAG cases, which is the size of the retrieval effects being
    # measured. Its baseline is the published no_rag, which was generated there.
    'Qwen3.6-35B-A3B-FP8': dict(
        main=('Qwen/Qwen3.6-35B-A3B-FP8',),
        no_rag=PUBLISHED,
        oracle=None),
    'qwen3-next-80b-a3b-instruct': dict(
        main=('qwen3-next-80b-a3b-instruct',),
        no_rag=PUBLISHED,
        oracle='oracle_qwen3-next-80b-a3b-instruct_gold_panel.csv'),
}
MAIN = {'no_rag': NORAG_CSV}
MAIN_CACHE = f'{OUT}/main_results_models.json'
#: the published runs are 868 MB and 795 MB; the judge columns are a few kB
PUBLISHED_CACHE = f'{OUT}/main_panel_scores.csv'


def fmt(v, sem=None, digits=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '--'
    s = f'{v:.{digits}f}'
    if sem is not None and not np.isnan(sem):
        s += f'$_{{\\pm{sem:.1f}}}$'
    return s


def fmt_delta(v, p=None, sem=None):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '--'
    if sem is not None and not np.isnan(sem):
        return f'${v:+.2f}_{{\\pm{sem:.2f}}}$'
    s = f'{v:+.1f}'
    if p is not None and not np.isnan(p):
        s += f' ({"$p<0.001$" if p < 0.001 else f"$p={p:.2f}$"})'
    return s


def single_judge_stats(model, d=D):
    """{condition: per-case score}, single judge, indexed by case."""
    paths = {'oracle': f'{d}/oracle_{model}.csv',
             'no_rag': f'{d}/rejudged_no_rag_{model}.csv'}
    s = {}
    for cond, p in paths.items():
        if os.path.exists(p):
            df = pd.read_csv(p)
            if SINGLE_JUDGE_COL in df:
                s[cond] = df.set_index('index')[SINGLE_JUDGE_COL] * 100
    if 'no_rag' not in s:
        return None
    row = {'model': model, 'n': int(s['no_rag'].notna().sum())}
    for cond, _ in CONDITIONS:
        if cond not in s:
            continue
        row[cond] = s[cond].mean()
        row[f'{cond}_sem'] = s[cond].sem()
        if cond != 'no_rag':
            paired = pd.concat([s[cond], s['no_rag']], axis=1).dropna()
            dd = paired.iloc[:, 0] - paired.iloc[:, 1]
            row[f'{cond}_delta'] = dd.mean()
            row[f'{cond}_delta_sem'] = dd.sem()
            row[f'{cond}_p'] = stats.ttest_rel(paired.iloc[:, 0], paired.iloc[:, 1]).pvalue
    return row


def single_judge_table(d=D, out_tex=f'{OUT}/oracle_condition.tex'):
    models = sorted({re.sub(r'^oracle_|\.csv$', '', os.path.basename(p))
                     for p in glob.glob(f'{d}/oracle_*.csv')
                     if not p.endswith('_panel.csv')})
    rows = [r for r in (single_judge_stats(m, d) for m in models) if r]
    if not rows:
        print('no completed models yet')
        return None
    lines = [
        r'% Generated by experiments/analysis/oracle_experiment.py -- do not hand-edit.',
        r'\begin{table}[htbp]', r'\centering',
        r'\caption{\textbf{An oracle retrieval condition does not improve essay quality.}',
        r'\emph{Oracle retrieval} supplies, instead of search results, the statutes the',
        r'reference solution itself cites, restricted to those present in the federal corpus',
        r'(17.6 of 20.8 cited norms per case) and ranked by how often the solution relies on',
        r'them. It is therefore the best case any retriever over this corpus could reach.',
        r'$\Delta$ is the paired difference against the same model on the same case without',
        r'retrieval, over 81 cases, with the $p$-value of a paired $t$-test. All three',
        r'conditions are re-scored here by a single judge (\texttt{qwen36-35b}) rather than the',
        r'three-judge ensemble used elsewhere, because two of those judges were unavailable;',
        r'the levels are therefore not comparable with Table~\ref{tab:main_results}, but the',
        r'contrasts within a row are.}',
        r'\label{tab:oracle_condition}', r'\small',
        r'\begin{tabular}{lrrr}', r'\toprule',
        r'& \multicolumn{2}{c}{Score} & \\',
        r'\cmidrule(lr){2-3}',
        r'Model & No RAG & Oracle & $\Delta$ vs.\ no retrieval \\',
        r'\midrule']
    for r in rows:
        lines.append(
            f"{r['model'].replace('_', chr(92) + '_')} & "
            f"{fmt(r.get('no_rag'), r.get('no_rag_sem'))} & "
            f"{fmt(r.get('oracle'), r.get('oracle_sem'))} & "
            f"{fmt_delta(r.get('oracle_delta'), r.get('oracle_p'))} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    open(out_tex, 'w').write('\n'.join(lines))
    print(f'written to {out_tex}')
    return pd.DataFrame(rows)


def main_models(cache=MAIN_CACHE):
    """Model names in each published run, cached -- no_rag_ji2_result.csv is 868 MB."""
    if os.path.exists(cache):
        return {k: set(v) for k, v in json.load(open(cache)).items()}
    out = {}
    for arm, path in MAIN.items():
        out[arm] = sorted(pd.read_csv(path, usecols=['model']).model.unique())
        print(f'  {arm}: {len(out[arm])} models in {os.path.basename(path)}', flush=True)
    json.dump(out, open(cache, 'w'), indent=1)
    return {k: set(v) for k, v in out.items()}


def published_panel(cache=PUBLISHED_CACHE):
    """Per-essay judge scores from the published runs, cached to a small CSV."""
    if os.path.exists(cache):
        return pd.read_csv(cache)
    aliases = [a for c in PANEL_COLS for a in JUDGE_ALIASES.get(c, (c,))]
    out = []
    for arm, path in MAIN.items():
        have = set(pd.read_csv(path, nrows=0).columns)
        df = pd.read_csv(path, usecols=['model', 'index'] + [a for a in aliases if a in have])
        df.insert(0, 'arm', arm)
        out.append(df)
        print(f'  cached judge columns for {arm} ({len(df)} rows)', flush=True)
    w = pd.concat(out, ignore_index=True)
    w.to_csv(cache, index=False)
    return w


def panel_scores(spec, main_names=(), arm=None, d=D):
    """Per-essay panel score for one arm, or ``None`` if it was not panel-judged.

    Score convention follows the paper: per-essay median across all three
    judges, times 100.  A source missing any judge stays out -- a two-judge
    median is a different scale from a three-judge one, and mixing them is the
    failure this step exists to fix.
    """
    if spec is None:
        return None
    if spec == PUBLISHED:
        pub = published_panel()
        sel = pub[(pub.arm == arm) & (pub.model.isin(main_names))].sort_values('index')
        if not len(sel):
            return None
        f = panel_frame(sel)
    else:
        p = panel_path(f'{d}/{spec}')
        if not os.path.exists(p):
            return None
        f = panel_frame(pd.read_csv(p).sort_values('index'))
    if f is None:
        return None
    return (100 * f.median(axis=1, skipna=True)).values


def panel_table(d=D, require=('no_rag',), out_csv=f'{OUT}/oracle_panel.csv',
                out_tex=f'{OUT}/oracle_panel.tex'):
    """The three conditions on the ensemble scale, for every model the paper has.

    A model belongs here only if the published run has a no-RAG level for it, so
    the oracle column can be read next to a published baseline.  with-RAG is not
    required: gemma4-31b-it has a published no-RAG result but no with-RAG one,
    and its oracle gain still counts.
    """
    published = main_models()
    keep, dropped = {}, []
    for m, f in PANEL_ARMS.items():
        if all(set(f.get('main', ())) & published[arm] for arm in require):
            keep[m] = f
        else:
            dropped.append(m)
    if dropped:
        print(f'not in the published run(s) {", ".join(require)}, dropped: '
              f'{", ".join(dropped)}\n')

    rows = []
    for model, files in keep.items():
        s = {arm: panel_scores(files.get(arm), files.get('main', ()), arm, d)
             for arm in ('no_rag', 'oracle')}
        base = s['no_rag']
        r = {'model': model}
        for arm in ('no_rag', 'oracle'):
            v = s[arm]
            r[arm] = float(np.mean(v)) if v is not None else np.nan
            if arm != 'no_rag':
                if v is not None and base is not None:
                    dv = v - base
                    r[f'delta_{arm}'] = float(dv.mean())
                    r[f'delta_{arm}_sem'] = float(dv.std(ddof=1) / np.sqrt(len(dv)))
                else:
                    r[f'delta_{arm}'] = r[f'delta_{arm}_sem'] = np.nan
            if v is None:
                print(f'  no panel-scored answers: {model} / {arm}')
        rows.append(r)
    w = pd.DataFrame(rows)
    print(w.round(2).to_string(index=False))
    w.round(4).to_csv(out_csv, index=False)

    lines = [r'% Generated by experiments/analysis/oracle_experiment.py',
             r'\begin{table}[htbp]', r'\centering',
             r"\caption{\textbf{The oracle condition on the paper's judge scale.} "
             r'Answers re-scored with the two ensemble judges reachable on NHR@FAU '
             r'(per-essay median $\times$ 100). Deltas are paired per case. '
             r'Arms without panel-scored answers are left blank rather than mixed '
             r'with single-judge scores.}',
             r'\label{tab:oracle_panel}', r'\small',
             r'\begin{tabular}{lrrr}', r'\toprule',
             r'Model & no RAG & oracle & $\Delta$ oracle \\',
             r'\midrule']
    for _, r in w.iterrows():
        lines.append(f"{r.model} & {fmt(r.no_rag, digits=2)} & "
                     f"{fmt(r.oracle, digits=2)} & "
                     f"{fmt_delta(r.delta_oracle, sem=r.delta_oracle_sem)} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    open(out_tex, 'w').write('\n'.join(lines))
    print(f'\nwritten to {out_csv} and {out_tex}')
    return w


def table(d=D, require=('no_rag',)):
    print('=== single judge ===')
    single_judge_table(d)
    print('\n=== ensemble panel ===')
    panel_table(d, require)


# --------------------------------------------------------------------------


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('run', help='generate the oracle arm and score all three conditions')
    p.add_argument('--model', required=True)
    p.add_argument('--judge', default='qwen36-35b')
    p.add_argument('--judge-instruction', default='ji2')
    p.add_argument('--gold', choices=['gold', 'refex'], default='gold')
    p.add_argument('--kb', default=KB,
                   help='corpus the oracle resolves norms against')
    p.add_argument('--max-passages', type=int, default=20)
    p.add_argument('--chunk', type=int, default=GEN_CHUNK, metavar='N',
                   help='cases per .part checkpoint during generation; lower '
                        'it when the run is supervised by a timeout')
    p.add_argument('--max-tokens', type=int, default=None)
    p.add_argument('--concurrency', type=int, default=None, metavar='N',
                   help='parallel generation requests (default 10; raise only '
                        'for endpoints that fan out, not for NHR@FAU)')
    p.add_argument('--existing-model', default=None,
                   help='the name the published runs store for this model')
    p.add_argument('--include-uncovered', action='store_true',
                   help='also name the cited norms whose text the corpus lacks')
    p.add_argument('--fresh-baseline', action='store_true',
                   help='regenerate the no-retrieval arm on this host')
    p.add_argument('--tag', default='')
    p.add_argument('--cites-only', action='store_true',
                   help='name the gold norms instead of supplying their text, so the '
                        'delta against the text oracle isolates the selection channel')
    p.add_argument('--cites-from', default=None,
                   help='lift the citation list verbatim from this stored run\'s '
                        'qa_prompt, so the arm names exactly what its text-arm '
                        'counterpart supplied')
    p.add_argument('--pad-junk', type=int, default=0, metavar='N',
                   help='add N norms the solution does not cite, interleaved, to '
                        'price what junk context costs; pairs with --shuffle-refs')
    p.add_argument('--pad-from', default=None,
                   help='stored arm CSV supplying the junk, e.g. the k=50 RRF arm')
    p.add_argument('--shuffle-refs', action='store_true',
                   help='permute the reference order per case, to ablate the '
                        'prominence ranking gold_sets otherwise leaks')
    p.add_argument('--qa-prompt', default='qa1', choices=['qa1', 'qa2'],
                   help="generation prompt: 'qa1' is the published one, 'qa2' adds "
                        'the methodological block. Pair with --tag so the variant '
                        'writes beside the default rather than over it.')
    p.add_argument('--dry-run', action='store_true',
                   help='build the oracle context and print it; makes no API calls')

    p = sub.add_parser('rejudge', help='re-score stored answers with the ensemble panel')
    # Default is 'free'. This used to default to the full PANEL, so `rejudge`
    # with no arguments spent money -- and it is the command reached for when
    # topping up a seat, which is usually the free one.
    p.add_argument('--judges', default='free', choices=list(PANELS),
                   help="'free' is the two NHR@FAU seats and costs nothing; "
                        "'panel' adds gpt-5-nano at ~$0.005/essay and needs --yes")
    p.add_argument('--yes', action='store_true', help='confirm a billed run')
    p.add_argument('--arm', choices=['table', 'all', 'gold', 'refex', 'other'], default='table',
                   help="'table' scores every file the panel table reads; the others "
                        'select an oracle arm, gold first either way')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--dry-run', action='store_true')

    p = sub.add_parser('table', help='write both LaTeX tables from the stored answers')
    p.add_argument('--require', nargs='*', default=['no_rag'], choices=['no_rag'],
                   help='published runs a model must appear in to be listed')

    a = ap.parse_args()
    if a.cmd == 'run':
        run(a.model, a.judge, a.judge_instruction, a.max_passages, max_tokens=a.max_tokens,
            existing_model=a.existing_model, gold_set=a.gold,
            include_uncovered=a.include_uncovered, tag=a.tag,
            fresh_baseline=a.fresh_baseline, dry_run=a.dry_run,
            qa_prompt=a.qa_prompt, cites_only=a.cites_only,
            cites_from=a.cites_from, shuffle_refs=a.shuffle_refs,
            pad_junk=a.pad_junk, pad_from=a.pad_from, kb=a.kb,
            chunk=a.chunk, concurrency=a.concurrency)
    elif a.cmd == 'rejudge':
        rejudge(dry_run=a.dry_run, limit=a.limit, arm=a.arm,
                panel=PANELS[a.judges], confirmed=a.yes)
    else:
        table(require=tuple(a.require))

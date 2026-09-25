"""The retrieval ladder: what each pipeline finds, and what that buys the essay.

One column layout, two blocks of rows, so a reader can run their eye down a
column and see retrieval quality and essay score for the same pipeline.

The two blocks do not share a baseline, and cannot. An essay arm is measured
against the same model with no retrieval at all; a retrieval arm has no such
condition -- "no RAG" retrieves nothing, so its recall is zero by construction
and a delta against it would just restate the recall. The retrieval block is
therefore measured against ``round-robin``, the simplest pipeline that actually
searches, and the essay block against no RAG. Shading is normalised inside each
block, so a colour is comparable down a block and not across the two.

Column-to-config mapping matters here and is easy to get wrong. The recall row
and the score row in one column have to describe the *same* retriever:

    dense       dense_rrf         (NOT mode_vector -- that one inherits the
                                   round-robin merge and so is a different
                                   pipeline from the dense generation arm)
    BM25        bm25_rrf          (retrieval only; never generated under)
    round-robin baseline
    RRF         merge_rrf
    re-rank     rerank_innkube
    cite-only   cite_only
    cite+RRF    cite_then_search

``COLS`` is the full set and stays that way; ``PAPER_COLS`` is the subset the
LaTeX table prints, and the first three of those pipelines are not in it. Pass
``--full`` for the twelve-column version. The console dump always shows all of
them.

The recall rows read ``recall_battery.csv``. ``recall_battery_v2.csv`` measures
the two-hop configurations against the v2 Leitsatz index and describes a
retriever no column here uses.

``oracle`` has no battery config: it does not search. Its recall is the share of
each solution's cited norms the corpus can supply at all, which is the ceiling
every other column is working against.

Run from ``experiments/``::

    PYTHONPATH=analysis python analysis/ladder_table.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import refs as refs_mod
import paper_table as pt
from judge_recall_slope import gold_by_case, context_norms

BATTERY = 'analysis/out/recall_battery.csv'
#: The oracle arm the coverage row is read off: `_gold_combined`, the oracle
#: over the same corpus the retrieval arms search. 47.4 of 51.9 cited norms per
#: case and 94.9% of the state-law ones, against the stored `_gold` arms' 45.4
#: and 90.9% and the federal-corpus `_gold_now`'s 35.0 and 34.4%.
ORACLE_CTX = 'oracle_deepseek-ai_DeepSeek-V4-Flash_gold_combined.csv'
OUT = 'analysis/out/ladder_table.tex'
K = 10

#: display name -> (battery config, arm directory)
#:
#: Every column is the same ten-passage budget over the combined corpus. The
#: ``_k25``/``_k50`` directories are a different budget and a different question,
#: so they are not folded in here -- a column whose recall row is measured at
#: k=10 and whose score row was generated at k=50 would be two experiments
#: wearing one heading.
#:
#: ``bm25`` has no generation arm: it is the lexical leg on its own, measured
#: because ``dense`` and ``RRF`` are only interpretable next to it, and never
#: generated under because it is the weakest retriever in the battery. Its essay
#: cells are blank by design rather than by omission.
#:
#: The last three columns are the same pipelines at a 50-passage budget rather
#: than ten, which is why each column carries its own ``k``: the recall row and
#: the score row in one column must describe the same retrieval, and a recall
#: measured at k=10 above essays generated at k=50 would be two experiments
#: wearing one heading. They are here because the budget axis is not inert once
#: the exact-key channels are on -- ``rrf`` at k=50 moves nothing, ``cite_ls_rrf``
#: at k=10 moves nothing, and the two together are the only non-oracle arms on
#: record that beat no RAG beyond twice their standard error.
#:
#: ``cite_ls_rrf`` reaches this table only at k=50, and only as the ceiling of
#: the budget row. Its Leitsatz two-hop retrieves case law, which is the
#: follow-up's claim, not this paper's; it is kept as the comparison that says
#: whether the statute-only ``cite_rrf`` at the same budget is enough.
COLS = [
    ('no_rag', 'no RAG', None, None, K),
    ('dense', 'dense', ('dense_rrf',), 'rag_titled_combined_dense', K),
    ('bm25', 'BM25', ('bm25_rrf',), 'rag_titled_combined_bm25', K),
    ('roundrobin', 'round-robin', ('baseline',), 'rag_titled_combined', K),
    ('rrf', 'RRF', ('merge_rrf',), 'rag_titled_combined_rrf', K),
    ('rerank', 're-rank', ('rerank_innkube',), 'rag_titled_combined_rerank', K),
    ('cite_only', 'cite-only', ('cite_only',), 'rag_titled_combined_cite_only', K),
    ('cite_rrf', 'cite+RRF', ('cite_then_search',), 'rag_titled_combined_cite_rrf', K),
    ('rrf50', r'RRF$_{50}$', ('merge_rrf',), 'rag_titled_combined_rrf_k50', 50),
    ('cite_rrf50', r'cite+RRF$_{50}$', ('cite_then_search',),
     'rag_titled_combined_cite_rrf_k50', 50),
    # Headnote *text* in the context instead of the norms the headnote cites.
    # It sets drop_norms, so the generator sees court material and no statute
    # text at all -- but the retrieval block is still defined for it, and
    # ``ls_only`` measures it: which gold norms the retrieved Leitsaetze are
    # about. Read it as reachability rather than supply. That distinction is
    # the arm's whole point, and it is why the essay-score and citation cells
    # move while the norms themselves are never printed into the prompt.
    ('ls_text', 'LS-text', ('ls_only',), 'rag_titled_combined_ls_text', K),
    ('cite_ls_rrf50', r'cite+LS+RRF$_{50}$', ('cite_ls_search',),
     'rag_titled_combined_cite_ls_rrf_k50', 50),
    ('oracle', 'oracle', None, None, K),
]

#: The columns the paper prints, in order. ``dense``, ``BM25`` and
#: ``round-robin`` are the three single-strategy legs -- the embedding index
#: alone, the full-text index alone, and the two interleaved -- and RRF is the
#: standard hybrid fusion of exactly those two. All three sit within noise of it
#: on every row, so printing them spends three of twelve columns restating that
#: hybrid retrieval is the sensible default rather than showing anything the
#: reader cannot take from the literature. They stay in ``COLS`` and in the
#: console dump: ``--full`` prints them, and the retrieval block is then still
#: measured against RRF, so the two versions of the table are the same numbers.
#:
#: RRF and not round-robin as the baseline for the retrieval block, because with
#: the legs gone RRF is the simplest pipeline the table still shows, and a block
#: measured against a column that is not there cannot be read.
PAPER_COLS = ['no_rag', 'rrf', 'rerank', 'cite_only', 'cite_rrf',
              'rrf50', 'cite_rrf50', 'ls_text', 'cite_ls_rrf50', 'oracle']
RETRIEVAL_BASE = 'rrf'


#: One row per model, and Qwen3.6-35B appears once, as the NHR@FAU serving.
#: InnKube serves the same base model under `qwen36-35b` and the two are not
#: interchangeable: over the four arms both have, the no-RAG step is +2.35 while
#: the RAG arms sit within +/-1.7 with the sign flipping, so the offset does not
#: cancel in the deltas -- FAU's average +4.21 against InnKube's +1.65, and the
#: two disagree beyond 2 SEM on rrf. Splicing them would put a deployment
#: difference inside a row that reads as a pipeline comparison. FAU is kept
#: because it is the serving with the full ladder; InnKube keeps its own row in
#: paper_table.py, which is the table gpt-5-nano was bought for.
#:
#: Oracle files are `_gold_combined`: one condition, every model, reproducible
#: by current code, and over the corpus the retrieval arms actually search. This
#: supersedes both the stored `_gold_panel` arms, which no code path can
#: regenerate and which no FAU Qwen version exists of, and `_gold_now`, which
#: read the federal-only corpus. Free seats only for now -- these cells are not
#: on the gpt-5-nano scale.
#: gemma is NHR@FAU's ``RedHatAI/gemma-4-31B-it-FP8-block``, not InnKube's
#: ``gemma4-31b-it``. InnKube serves this model at 31-47 minutes per essay on the
#: 120-passage oracle prompt -- 45 to 63 hours for one arm -- because it collapses
#: on long contexts while staying fast (33-60 s) on ordinary ones. The whole row
#: moved rather than the oracle alone: an arm on one deployment differenced
#: against a baseline on another measures the host, not the condition.
GENERATORS = [
    ('DeepSeek-V4-Flash', 'deepseek-ai/DeepSeek-V4-Flash',
     'rag_deepseek-ai_DeepSeek-V4-Flash',
     'oracle_deepseek-ai_DeepSeek-V4-Flash_gold_combined.csv'),
    ('Qwen3.6-35B-A3B', 'Qwen/Qwen3.6-35B-A3B-FP8', 'rag_Qwen_Qwen3.6-35B-A3B-FP8',
     'oracle_Qwen_Qwen3.6-35B-A3B-FP8_gold_combined.csv'),
    ('gpt-oss-120B', 'openai/gpt-oss-120b', 'rag_openai_gpt-oss-120b',
     'oracle_openai_gpt-oss-120b_gold_combined.csv'),
    ('Gemma-4-31B-it', 'RedHatAI/gemma-4-31B-it-FP8-block',
     'rag_RedHatAI_gemma-4-31B-it-FP8-block',
     'oracle_RedHatAI_gemma-4-31B-it-FP8-block_gold_combined.csv'),
    ('Qwen3-Next-80B-A3B', 'qwen3-next-80b-a3b-instruct',
     'rag_qwen3-next-80b-a3b-instruct',
     'oracle_qwen3-next-80b-a3b-instruct_gold_combined.csv'),
    # The small models are here to show that the ladder's shape -- search arms
    # inside the noise, oracle outside it -- is not a property of the frontier
    # tier. Their rows were once mostly blank, and a comment here used to say
    # that was by design: "blank is not pending for these five middle columns".
    # It was pending. Eleven of those cells had their essays generated and two
    # of three judge seats scored; the third seat was an empty column that
    # missing_judges() read as filled, so no rejudge ever refilled it. They are
    # populated now, and they did not leave the story alone -- Magistral-Small
    # at RRF turned out to be a second significant negative. Only cite+LS+RRF50
    # for Mistral-Small is genuinely unrun.
    #
    # Gemma-4-E4B was the eighth generator and is gone. It was the one row whose
    # no-retrieval baseline could not come from the main table -- it is not in
    # the 32-model main comparison, so it has no row there -- which forced the
    # ladder to difference it against a separate norag_fresh file and cost the
    # paper a "one of which is outside the 32" caveat in two places. Its ladder
    # said nothing the other two small models did not: search arms inside the
    # noise, oracle outside it. Dropping it makes the ladder's population a
    # strict subset of the main table's, which is worth more than an eighth row.
    ('Mistral-Small-3.2', 'RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8',
     'rag_RedHatAI_Mistral-Small-3.2-24B-Instruct-2506-FP8',
     'oracle_RedHatAI_Mistral-Small-3.2-24B-Instruct-2506-FP8_gold_combined.csv'),
    ('Magistral-Small', 'GaleneAI/Magistral-Small-2509-FP8-Dynamic',
     'rag_GaleneAI_Magistral-Small-2509-FP8-Dynamic',
     'oracle_GaleneAI_Magistral-Small-2509-FP8-Dynamic_gold_combined.csv'),
]

SCALE = pt.SCALES['free3']


#: Which gold set the recall rows are scored against. The battery stores both,
#: and they are different denominators over the same retrieval: ``refex`` is the
#: ~21 norms/case ``legal_ref_sim`` sees, ``gold`` the ~52 that dual section and
#: article extraction finds. Only ``gold`` carries the state-law split, which is
#: the row the corpus argument turns on, so ``gold`` is used for all of them --
#: one denominator down the whole block. (Taking each cell's max across the two,
#: as this function did before, silently gave cite-only its refex number and
#: round-robin its gold one, i.e. compared two different denominators across
#: one row.)
GOLD_SET = 'gold'


def recall_row(field, corpus='combined'):
    """``{column key: percent}`` for one battery metric, each column at its own k."""
    b = pd.read_csv(BATTERY)
    b = b[(b.corpus == corpus) & (b.gold_set == GOLD_SET)]
    out = {}
    for key, _, configs, _, k in COLS:
        if not configs:
            continue
        for c in configs:                      # first config that has been run
            hit = b[(b.config == c) & (b.k == k)]
            if len(hit):
                out[key] = 100 * hit[field].max()
                break
    return out


def oracle_recall():
    """Gold norms the corpus can supply, overall and for state law.

    Read off the oracle arm's own stored context rather than recomputed from the
    knowledge base, so it is the coverage the arm actually had.
    """
    gold = gold_by_case()
    p = f'{pt.B}/oracle_rag/ji2/{ORACLE_CTX}'
    if not os.path.exists(p):
        return {}, {}, {}
    ctx = context_norms(p)
    def frac(pred):
        num = den = 0
        for i, g in gold.items():
            g = {n for n in g if pred(n)}
            num += len(g & ctx.get(i, set()))
            den += len(g)
        return 100 * num / den if den else np.nan
    state = lambda n: refs_mod.jurisdiction(n[0]) == 'state'
    hit = 100 * np.mean([bool(g & ctx.get(i, set())) for i, g in gold.items()])
    return frac(lambda n: True), frac(state), hit


def essay_row(mid, stem, oracle_file, norag):
    """``{column key: per-case score vector}`` for one generator.

    The vector rather than its mean, because the arms and the baseline are the
    same 81 cases in the same order: the difference is paired, and a paired
    standard error is roughly half the unpaired one. Keeping the cases is what
    lets the table say which cells are actually distinguishable from no RAG.
    """
    out = {}
    base = pt.baseline(mid, SCALE, norag)
    if base is not None:
        out['no_rag'] = base
    for key, _, _, d, _k in COLS:
        if not d:
            continue
        p = f'{pt.B}/{d}/ji2/{stem}.csv'
        if os.path.exists(p):
            v = pt.med(pd.read_csv(p).sort_values('index'), SCALE)
            if v is not None:
                out[key] = v
    if oracle_file:
        p = f'{pt.B}/oracle_rag/ji2/{oracle_file}'
        if os.path.exists(p):
            v = pt.med(pd.read_csv(p), SCALE)
            if v is not None:
                out['oracle'] = v
    return out


#: Per-case citation recall is ~90 seconds of refex over every stored essay, so
#: it is cached. Delete the file (or pass --recite) to rebuild it.
CITE_CACHE = 'analysis/out/ladder_citations.csv'


def essay_norms(text):
    """The norms an essay cites, keyed the way ``gold_by_case`` keys them."""
    if not isinstance(text, str):
        return set()
    return {(refs_mod.corpus_key(c.book), c.section.lower())
            for c in refs_mod.canonicalise(refs_mod.extract(text))}


def cite_recall(src, gold):
    """Per-case share of the solution's cited norms that the essay also cites.

    ``src`` is an arm CSV path or an already-loaded ``(index, answer)`` frame --
    the no-RAG row comes from :func:`paper_table.norag_essays`, which assembles
    it from two files.
    """
    d = src if isinstance(src, pd.DataFrame) else \
        pd.read_csv(src, usecols=['index', 'answer']).sort_values('index')
    out = []
    for i, a in zip(d['index'], d['answer']):
        g = gold.get(int(i))
        if g:
            out.append(len(g & essay_norms(a)) / len(g))
    return 100 * np.array(out) if out else None


def cite_row(mid, stem, oracle_file, gold):
    """``{column key: per-case citation recall}`` for one generator.

    The row the score rows cannot supply: whether retrieval changes what the
    essay *cites*. A judge score can move for a dozen reasons; this moves only
    if the norms in the essay changed. It is also the one place the oracle's
    advantage is legible as something other than a number -- the search arms
    hand over 1.5-9.6% of the cited norms and the essays cite the same share
    either way, which is the mechanism behind every null in the block above.
    """
    out = {}
    # the same essays paper_table.baseline() scores, so that the citation row
    # and the score row above it are two views of one no-RAG condition
    d = pt.norag_essays(mid)
    if d is not None:
        v = cite_recall(d, gold)
        if v is not None:
            out['no_rag'] = v
    if 'no_rag' not in out:
        f = mid.replace('/', '_')
        for pat in pt.NORAG_SRC:
            pth = f'{pt.B}/oracle_rag/ji2/{pat.format(f=f)}'
            if os.path.exists(pth):
                v = cite_recall(pth, gold)
                if v is not None:
                    out['no_rag'] = v
                    break
    for key, _, _, d, _k in COLS:
        if not d:
            continue
        pth = f'{pt.B}/{d}/ji2/{stem}.csv'
        if os.path.exists(pth):
            v = cite_recall(pth, gold)
            if v is not None:
                out[key] = v
    if oracle_file:
        pth = f'{pt.B}/oracle_rag/ji2/{oracle_file}'
        if os.path.exists(pth):
            v = cite_recall(pth, gold)
            if v is not None:
                out['oracle'] = v
    return out


def cite_rows(gold, rebuild=False):
    """One citation row per generator, through the cache."""
    if not rebuild and os.path.exists(CITE_CACHE):
        c = pd.read_csv(CITE_CACHE)
        return [(short, {k: g[k].values for k in g.columns if k not in ('model', 'index')
                         and g[k].notna().any()}, '{:.2f}')
                for short, g in ((s, c[c.model == s]) for s, *_ in GENERATORS)
                if len(g)]
    rows, flat = [], []
    for short, mid, stem, orc in GENERATORS:
        vals = cite_row(mid, stem, orc, gold)
        rows.append((short, vals, '{:.2f}'))
        for k, v in vals.items():
            flat += [dict(model=short, index=i, **{k: x}) for i, x in enumerate(v)]
    if flat:
        pd.DataFrame(flat).groupby(['model', 'index']).first().reset_index() \
          .to_csv(CITE_CACHE, index=False)
    return rows


def order_by_no_rag(norag):
    """``GENERATORS``, strongest no-retrieval model first.

    The reading order of every block is the same question -- what does this
    pipeline buy *this* model -- and the answer is easier to see when the rows
    descend by how well the model does without retrieval, because the reader
    can check whether the gains track raw capability. The list is reordered in
    place so the essay block, the citation block and the corpus table cannot
    drift apart: they iterate the same list, and a per-block sort would let one
    of them silently fall out of step with the others.

    Ties are broken by name, so the order is stable across rebuilds rather than
    depending on where a tied model happened to sit before. Two models do tie
    here: Qwen3.6-35B-A3B and Qwen3-Next-80B-A3B are both at 30.62, which is
    the judge grid rather than a coincidence -- an eleven-point scale medianed
    over three seats leaves arm means quantised.
    """
    def key(g):
        b = pt.baseline(g[1], SCALE, norag)
        return (-float(np.mean(b)) if b is not None else 1.0, g[0])
    return sorted(GENERATORS, key=key)


def build(rebuild=False):
    norag = pd.read_csv(f'{pt.B}/without_rag/ji2/no_rag_ji2_result.csv') \
              .sort_values('index')
    GENERATORS[:] = order_by_no_rag(norag)
    gold = gold_by_case()
    o_all, o_state, o_hit = oracle_recall()
    rec = recall_row('recall')
    st = recall_row('state_recall')
    hit = recall_row('cases_with_hit')
    if o_all:
        rec['oracle'], st['oracle'], hit['oracle'] = o_all, o_state, o_hit

    blocks = [
        ('Retrieval quality (\\%)', RETRIEVAL_BASE, 'row', [
            ('gold-norm recall@$k$', rec, '{:.2f}'),
            ('state-law recall@$k$', st, '{:.2f}'),
            ('cases with a hit@$k$', hit, '{:.1f}'),
        ]),
        ('Essay score (0--100)', 'no_rag', 'block', [
            (short, essay_row(mid, stem, orc, norag), '{:.2f}')
            for short, mid, stem, orc in GENERATORS
        ]),
        ('Gold norms cited by the essay (\\%)', 'no_rag', 'block',
         cite_rows(gold, rebuild=rebuild)),
    ]
    return blocks


#: the corpus contrast, at the one pipeline both corpora were run through
CORPUS_COLS = [('federal', 'federal only'), ('combined', 'combined')]


def corpus_blocks(norag):
    """Federal-only against combined, at the fusion the ladder prints.

    A second table rather than more columns in the first: corpus and pipeline
    are orthogonal axes, and the federal corpus was only ever run through one
    pipeline, so folding it into the pipeline columns would give a row that
    is almost entirely dashes. Same rows, two columns, no holes.

    The retrieval rows are measured at RRF, because ``PAPER_COLS`` starts the
    ladder at RRF -- round-robin is in ``COLS`` but is not printed, and a
    corpus contrast measured at a fusion the reader never sees cannot be read
    against the table beside it. The essay rows still come from the
    round-robin arms, and both sides of the contrast are round-robin, so the
    essay block is internally consistent even though it sits at a different
    fusion from the retrieval block. ``rag_titled_federal_rrf`` now holds three
    of the five federal arms at RRF; DeepSeek's never generated and gpt-oss is
    a judge seat short, so moving the block would cost two rows to gain one
    fusion. The caption says which block is which.

    Rows carry the per-case vector rather than its mean. Both corpora are the
    same 81 cases in the same order, so the difference is paired and ``tex``
    can mark the cells that clear twice its standard error -- which, at the
    time of writing, is none of them. That is the result: the Bavarian half
    buys recall and does not convert it.

    The point of the contrast is the state-law row. The federal corpus contains
    no Bavarian law at all, so its state recall is zero by construction while
    its *overall* recall is barely distinguishable -- which is exactly why
    overall recall is the wrong number to report for this corpus.
    """
    b = pd.read_csv(BATTERY)
    b = b[(b.k == K) & (b.gold_set == GOLD_SET)]
    def cell(field, config):
        hit = b[b.config == config]
        return 100 * hit[field].max() if len(hit) else None
    rec_c, st_c = recall_row('recall'), recall_row('state_recall')
    rows = [(f'gold-norm recall@{K}',
             dict(federal=cell('recall', 'corpus_federal_rrf'),
                  combined=rec_c.get('rrf')), '{:.2f}'),
            (f'state-law recall@{K}',
             dict(federal=cell('state_recall', 'corpus_federal_rrf'),
                  combined=st_c.get('rrf')), '{:.2f}')]
    essays = []
    for short, mid, stem, _ in GENERATORS:
        vals = {}
        for corpus, d in (('federal', 'rag_titled_federal'),
                          ('combined', 'rag_titled_combined')):
            pth = f'{pt.B}/{d}/ji2/{stem}.csv'
            if os.path.exists(pth):
                v = pt.med(pd.read_csv(pth).sort_values('index'), SCALE)
                if v is not None:
                    vals[corpus] = v
        # Magistral-Small and Mistral-Small never ran against the federal-only
        # corpus. A row with a dash on one side is not a contrast -- it prints
        # a combined number the ladder already prints, next to nothing to read
        # it against -- so the table is the five generators that ran both.
        if len(vals) == len(CORPUS_COLS):
            essays.append((short, vals, '{:.2f}'))
    # two blocks, as in the main table: percentage points of recall and points
    # of judge score are different units and must not share a colour scale
    return [('Retrieval quality (\\%)', 'federal', 'row', rows),
            ('Essay score (0--100)', 'federal', 'block', essays)]


#: Column head split over two rows. Twelve columns of judge scores is a wide
#: table; breaking the names at the hyphen buys back the width of the longest
#: one without abbreviating anything. Single-word heads sit in the lower row so
#: every head is bottom-aligned against the rule.
#:
#: The budget rides in the second row rather than as a subscript, which is both
#: shorter and more legible than ``cite+LS+RRF$_{50}$``.
HEAD = {
    'no_rag': ('no', 'RAG'),
    'dense': ('', 'dense'),
    'bm25': ('', 'BM25'),
    'roundrobin': ('round-', 'robin'),
    'rrf': ('', 'RRF'),
    'rerank': ('', 're-rank'),
    'cite_only': ('cite-', 'only'),
    'cite_rrf': ('cite+', 'RRF'),
    'rrf50': ('RRF', r'$k{=}50$'),
    'cite_rrf50': ('cite+RRF', r'$k{=}50$'),
    'ls_text': ('LS-text', r'$k{=}10$'),
    'cite_ls_rrf50': ('cite+LS+RRF', r'$k{=}50$'),
    'oracle': ('', 'oracle'),
    'federal': ('federal', 'only'),
    'combined': ('', 'combined'),
}


#: Half-width of the colour scale, in standard deviations of the deltas it is
#: drawn over. The first version of this table used ``max|delta|``, which for
#: the essay block lands at 2.95 sd -- so exactly one cell was fully saturated
#: and the rest read as a fraction of it. That over-saturates: a three-point
#: move rendered as near-black. Six sd puts the largest delta at about half
#: saturation and leaves headroom for arms not yet run, so a colour keeps its
#: meaning when the blank columns fill in.
SIGMA = 6.0


def shade(d, vmax):
    """RdBu centred on zero: blue for improvement, red for loss, white at 0.

    ``vmax`` is the half-width of the scale, and values beyond it clip. Returns
    the ``HTML`` triple and whether the cell needs light text.
    """
    import matplotlib as mpl
    r, g, b, _ = mpl.colormaps['RdBu'](mpl.colors.Normalize(-vmax, vmax)(d))
    # ITU-R BT.709 relative luminance, on the sRGB values as printed
    dark = 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45
    return '{:02X}{:02X}{:02X}'.format(*(round(255 * c) for c in (r, g, b))), dark


def sem(v, base):
    """Standard error of the paired difference, or None if it is not paired."""
    if not isinstance(v, np.ndarray) or not isinstance(base, np.ndarray):
        return None
    k = min(len(v), len(base))
    if k < 2:
        return None
    d = v[:k] - base[:k]
    return float(d.std(ddof=1) / np.sqrt(k))


#: How the tabular is made to fit the text block.
#:
#: Measured natural widths of the twelve-column table (10pt article, T1):
#: 638.7pt at \normalsize, 538.8pt at \scriptsize -- four size steps buy only
#: 16%. ``\tabcolsep`` is an absolute length, so the 13 columns carry
#: 26 x 6pt = 156pt of padding that does not shrink with the font, which is why
#: shrinking the font alone looks like it does nothing. At 3pt it is 460.8pt.
#:
#: IOS-Book-Article's text block is 372pt, so the last step has to be a box that
#: scales: 372/460.8 = 0.81, i.e. an effective 5.7pt body. ``resizebox`` is the
#: default because graphicx is already in the draft's preamble and adjustbox is
#: not installed in this TeX tree; ``adjustbox`` is better behaved (``max
#: width`` only ever shrinks, where resizebox would also enlarge a small table)
#: if the package is available. ``none`` emits the bare tabular.
#:
#: With the three single-strategy legs dropped the nine printed columns measure
#: 381.0pt, so the box now shaves 2% rather than 19% and the body is an
#: effective 6.9pt. Under ``--full`` it is the old 12-column table again and the
#: scaling is back to 0.81.
FIT = {
    'adjustbox': (r'  \begin{adjustbox}{max width=\linewidth}',
                  r'  \end{adjustbox}'),
    'resizebox': (r'  \resizebox{\linewidth}{!}{%', r'  }'),
    'none': (None, None),
}


def tex(blocks, cols=None, label='tab:ladder', caption=None, fit='resizebox'):
    cols = cols or [(k, l) for k, l, *_ in COLS if k in PAPER_COLS]
    keys = [k for k, _ in cols]
    heads = [HEAD.get(k, ('', lbl)) for k, lbl in cols]
    open_fit, close_fit = FIT[fit]
    out = [r'\begin{table}[t]', r'  \centering', r'  \scriptsize',
           # absolute, so it does not shrink with the font -- see FIT
           r'  \setlength{\tabcolsep}{3pt}',
           r'  \renewcommand{\arraystretch}{1.15}']
    if open_fit:
        out.append(open_fit)
    out += [r'  \begin{tabular}{l' + 'r' * len(cols) + '}', r'    \toprule',
           '     & ' + ' & '.join(t for t, _ in heads) + r' \\',
           '     & ' + ' & '.join(b for _, b in heads) + r' \\']
    for title, base_key, scope, rows in blocks:
        out.append(r'    \midrule')
        # The indent sits on the section headline, not on the rows under it.
        # Row labels are the longest strings in column one -- seven model names
        # and three retrieval metrics -- so indenting them costs a \quad of
        # width in every one, and the table is already tight. Indenting the
        # three headlines instead costs it three times and still groups.
        out.append(f'    \\multicolumn{{{len(cols) + 1}}}{{l}}{{\\quad\\emph{{{title}}}, '
                   f'against {dict(cols)[base_key]}}} \\\\')
        # The scale is drawn over the *systems* only. The oracle is a ceiling
        # rather than a pipeline, and at 91% recall against round-robin's 2.64
        # it would set a scale on which every real pipeline renders as the same
        # near-white. It still gets a colour; it just clips.
        #
        # ``scope`` says how wide the scale is drawn. The essay blocks are one
        # scale for the whole block, so a colour reads down them: every row is
        # the same quantity. The retrieval block cannot be -- a share of *norms*
        # and a share of *cases* are different quantities, and putting the
        # hit-rate row's spread on the recall row's scale flattens the recall
        # row to white. Its rows are scaled one at a time.
        num = lambda x: np.nan if x is None else float(np.mean(x))
        def scale_over(rs):
            d = [num(v.get(k)) - num(v.get(base_key))
                 for _, v, _ in rs for k in keys if k != 'oracle']
            d = np.array(d, dtype=float)
            sd = np.nanstd(d, ddof=1) if np.sum(~np.isnan(d)) > 1 else np.nan
            return SIGMA * sd if sd and not np.isnan(sd) and sd > 0 else 1.0
        block_vmax = scale_over(rows)
        for row in rows:
            row_label, vals, fmt = row
            vmax = block_vmax if scope == 'block' else scale_over([row])
            cells = []
            for k in keys:
                raw = vals.get(k)
                v = num(raw)
                if np.isnan(v):
                    cells.append('--')
                    continue
                cell = fmt.format(v)
                raw_b = vals.get(base_key)
                b = num(raw_b)
                if raw_b is not None and k != base_key and not np.isnan(b):
                    # Paired, and only where both cells kept their cases. The
                    # retrieval rows are single numbers per pipeline -- there is
                    # no per-case vector to pair -- so they simply get no star.
                    if sem(raw, raw_b) is not None and abs(v - b) > 2 * sem(raw, raw_b):
                        cell = f'\\textbf{{{cell}}}'
                    rgb, dark = shade(v - b, vmax)
                    cell = (f'\\cellcolor[HTML]{{{rgb}}}'
                            + (f'\\textcolor{{white}}{{{cell}}}' if dark else cell))
                cells.append(cell)
            out.append(f'    {row_label} & ' + ' & '.join(cells) + r' \\')
    out += [r'    \bottomrule', r'  \end{tabular}']
    if close_fit:
        out.append(close_fit)
    out += ['  \\caption{' + (caption or CAPTION) + '}',
            f'  \\label{{{label}}}', r'\end{table}']
    return '\n'.join(out)


#: The $1.9$ in the last sentence is measured, not assumed. Three generators
#: have the same no-retrieval essays scored by the same free3 seats in two
#: separate invocations -- ``MAIN_NORAG`` and the ``NORAG_SRC`` fallback -- and
#: the panel-median means differ by +0.12 (gpt-oss), -0.62 (Gemma-4-31B) and
#: +1.85 (Qwen3-Next). DeepSeek's pair is excluded: its two files are different
#: generations, not one generation judged twice. The earlier $\pm0.75$ came
#: from the DS_ALT seat-name check in paper_table, which measures one seat
#: agreeing with itself under two column names -- a smaller thing than a whole
#: panel re-run, and not the quantity the caption claims.
CAPTION = (
    r'The retrieval ladder over the 81 GPBam cases: what each pipeline finds, '
    r'and what that buys the essay. \emph{RRF} rank-fuses a dense and a '
    r'full-text search; \emph{re-rank} reorders that pool with '
    r'Qwen3-Reranker-4B; \emph{cite-only} resolves by exact key the norms a '
    r'second rewriter pass names as applicable, without searching; '
    r'\emph{cite+RRF} fills the remaining budget behind them; '
    r'\emph{cite+LS+RRF} also resolves the Normenketten of the retrieved '
    r'Leitsaetze; \emph{oracle} supplies the statutes the reference solution '
    r'cites, a bound rather than a system. The last block is the mechanism '
    r"behind the second: the share of the solution's cited norms that the "
    r'essay cites too. Columns are ten passages unless '
    r"marked $k{=}50$, which is also each recall row's budget; blank cells are "
    r'arms not yet run, and the omitted legs -- dense alone, full-text alone, '
    r'and hybrid under a round-robin sub-query merge -- are within noise of '
    r"RRF. Shading is the difference from the block's own baseline (RRF above, "
    r'no RAG below), blue better, so it reads down a block and not across; '
    r'bold is a paired difference beyond twice its standard error, which is '
    r'what separates two cells the judge grid prints as equal. Scores are '
    r'per-case medians over three free judge seats (Qwen3.6-35B-A3B, '
    r'DeepSeek-V4-Flash, gpt-oss-120B); the no-RAG column was judged in a '
    r'separate invocation and the seats are sampled, so a level offset of up '
    r'to $1.9$ points sits under every difference taken against it, which the '
    r'printed standard errors do not carry.')


CORPUS_CAPTION = (
    r'What the Bavarian half of the corpus is worth, over the five generators '
    r'that ran both. The federal corpus contains no state law, so its '
    r'state-law recall is zero by construction while its overall recall is '
    r'barely distinguishable -- overall recall is the wrong number to report '
    r'here. Adding the Bavarian half recovers that recall without converting '
    r'it: no generator moves by twice the standard error of the paired '
    r'difference. The retrieval rows hold the pipeline at RRF, so they read at '
    r'the same setting as Table~\ref{tab:ladder}; the essay rows are the '
    r'round-robin arms on both sides, the only fusion generated over both '
    r'corpora. Rows and shading follow Table~\ref{tab:ladder}; the baseline '
    r'column is the federal corpus.')


def main(rebuild=False, fit='resizebox', full=False):
    norag = pd.read_csv(f'{pt.B}/without_rag/ji2/no_rag_ji2_result.csv') \
              .sort_values('index')
    blocks = build(rebuild=rebuild)
    body = ('% generated by analysis/ladder_table.py -- do not edit by hand.\n'
            '% Requires \\usepackage[table]{xcolor}, \\usepackage{booktabs} and\n'
            '% \\usepackage{graphicx}. Rebuild with --fit adjustbox to use that\n'
            '% package instead, or --fit none for the bare tabular.\n\n'
            + tex(blocks, fit=fit,
                  cols=[(k, l) for k, l, *_ in COLS] if full else None) + '\n\n'
            + tex(corpus_blocks(norag), cols=CORPUS_COLS, label='tab:corpus',
                  caption=CORPUS_CAPTION, fit='none') + '\n')
    with open(OUT, 'w') as fh:
        fh.write(body)
    print(f'-> {OUT}\n')
    for title, base_key, _, rows in blocks:
        print(f'=== {title}  (vs {base_key}) ===')
        print(pd.DataFrame({lbl: {k: np.mean(x) for k, x in v.items()}
                            for lbl, v, _ in rows}).T
              .reindex(columns=[k for k, *_ in COLS]).round(2)
              .to_string(na_rep='  --  '))
        print()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    # The citation block is the only expensive part -- refex over every stored
    # essay, about two minutes -- and it only changes when an arm is added or
    # re-generated, so it is cached and rebuilt on request.
    ap.add_argument('--recite', action='store_true',
                    help=f'recompute {CITE_CACHE} instead of reading it')
    ap.add_argument('--sigma', type=float, default=SIGMA,
                    help='half-width of the colour scale, in sd of the deltas')
    ap.add_argument('--fit', default='resizebox', choices=sorted(FIT),
                    help='how the wide table is made to fit the text block')
    ap.add_argument('--full', action='store_true',
                    help='print every column in COLS, not just PAPER_COLS')
    a = ap.parse_args()
    SIGMA = a.sigma
    main(rebuild=a.recite, fit=a.fit, full=a.full)

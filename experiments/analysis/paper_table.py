"""Shared machinery for the retrieval tables, and the five-model arm summary.

This was the paper's retrieval table -- five models, four arms, two judge
scales, written out as ``rag_table.tex``. That table is gone: ``ladder_table``
covers the same five arms plus seven more, over eight generators instead of
five, on the scale the paper reports, so the two said the same thing and the
ladder said more. What is left here is the layer every retrieval script imports
-- :func:`med`, :func:`baseline`, :func:`norag_essays`, ``SCALES``, ``B`` --
plus a console dump of the arm summary.

All scales come out of the same files. ``free3`` is the paper's scale: three
seats, all on NHR@FAU, all free to re-run. ``free`` is that panel without
gpt-oss, kept because a two-seat median has no tiebreaker and the gap between
the two says how much work the tiebreaker is doing.

``panel`` -- the old scale, with gpt-5-nano in the third seat -- is **retired**:
gpt-oss replaced nano and no table reports it. It stays in ``SCALES`` because
the disagreement is on the record: nano's own RAG deltas average -0.18 where the
free seats average +2.67, so its column compressed every effect by roughly a
third and turned two cells negative. See ``judge_recall_slope.py`` for why.
Nothing new should be judged on it. Its 17 cells cost about $6 and their raw
seat columns are gitignored, so the only durable copy is the archived CSV::

    git show be11a617:experiments/analysis/out/paper_table.csv

Baselines are per model and per serving, keyed on the endpoint identifier, so
InnKube's ``gemma4-31b-it`` and FAU's ``RedHatAI/gemma-4-31B-it-FP8-block``
never share one.

Run from ``experiments/``::

    python analysis/paper_table.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

B = 'zubaers_result/essay_writing'
#: gitignored: the tracked copy is the archive in be11a617, and nothing
#: downstream reads this. Re-running still writes it for inspection.
OUT = 'analysis/out/paper_table.csv'
QWEN = 'score_Judge (Qwen3.6-35B-A3B-FP8)'
#: the same judge under the name its OpenRouter deployment stored
QWEN_ALT = 'score_Judge (qwen3.6-35b-a3b)'
DS = 'score_Judge (DeepSeek-V4-Flash)'
#: the same seat under the name the 2026-08-19/20 no-RAG re-judge stored. The
#: suffix is that run's bookkeeping for "post-swap weights", not a second model:
#: on identical essays the two names agree to +0.74, -0.37 and -0.49 across
#: three generators, inside the run-to-run band of every other seat.
DS_ALT = 'score_Judge (DeepSeek-V4-Flash-0731)'
#: the same seat served through OpenRouter, pinned to DeepInfra, for the days
#: NHR@FAU's deployment is down -- see the fallback block in endpoints.yaml.
#: ``judge_col`` derives the column from the slug, and the OpenRouter slug is
#: lowercase, so this is a third spelling of one seat rather than a new judge.
#: Validated against DS on the one arm that carries both: mean |delta| 6.25 over
#: 8 cases, where the FAU seat re-run against itself scores 6.42 and 8.46.
DS_OR = 'score_Judge (deepseek-v4-flash-0731)'
NANO = 'score_Judge (gpt-5-nano)'
GPTOSS = 'score_Judge (gpt-oss-120b)'
#: seat -> the other names that seat has been written out under. Tried in order
#: after the canonical name, so a file that carries both is read on the
#: canonical one.
SEAT_ALIASES = {QWEN: (QWEN_ALT,), DS: (DS_ALT, DS_OR)}
#: ``free3`` is the scale the paper now reports: three seats, all on NHR@FAU,
#: all free to re-run. It exists because ``free`` is a two-judge median, and a
#: two-judge median is the mean of two disagreeing seats with no tiebreaker --
#: gpt-oss is that tiebreaker. ``panel`` is kept for continuity with the earlier
#: tables but its nano seat cannot be refreshed without spending, and nano is
#: the seat that disagrees about RAG credit in the first place.
SCALES = {'free': [QWEN, DS], 'free3': [QWEN, DS, GPTOSS],
          'panel': [QWEN, DS, NANO]}

#: short name -> (endpoint identifier, arm file stem, oracle file)
#:
#: The oracle files are the ``_gold_combined`` ones, over the same corpus the
#: retrieval arms search. The ``_gold_panel`` files they replace were built over
#: ./my_knowledge_base -- federal-only, patched with a 100-norm sample holding no
#: GO and no BV, two of the four most-cited Bavarian books -- so they supplied
#: 35.0 of 51.9 cited norms per case and 34.4% of the state-law ones, against
#: 47.4 and 94.9% here. An upper bound handicapped on the axis the corpus
#: section argues is binding is not an upper bound.
#:
#: qwen36 names no oracle: its row is InnKube's serving and the combined oracle
#: was run against FAU's. med() drops the cell rather than compare across hosts.
#: gemma is NHR@FAU's ``RedHatAI/gemma-4-31B-it-FP8-block``, not InnKube's
#: ``gemma4-31b-it``. InnKube serves this model at 31-47 minutes per essay on the
#: 120-passage oracle prompt -- 45 to 63 hours for one arm -- because it collapses
#: on long contexts while staying fast (33-60 s) on ordinary ones. The whole row
#: moved rather than the oracle alone: an arm on one deployment differenced
#: against a baseline on another measures the host, not the condition.
MODELS = [
    ('DeepSeek', 'deepseek-ai/DeepSeek-V4-Flash', 'rag_deepseek-ai_DeepSeek-V4-Flash',
     'oracle_deepseek-ai_DeepSeek-V4-Flash_gold_combined.csv'),
    ('gpt-oss', 'openai/gpt-oss-120b', 'rag_openai_gpt-oss-120b',
     'oracle_openai_gpt-oss-120b_gold_combined.csv'),
    ('gemma4', 'RedHatAI/gemma-4-31B-it-FP8-block',
     'rag_RedHatAI_gemma-4-31B-it-FP8-block',
     'oracle_RedHatAI_gemma-4-31B-it-FP8-block_gold_combined.csv'),
    ('qwen3next', 'qwen3-next-80b-a3b-instruct', 'rag_qwen3-next-80b-a3b-instruct',
     'oracle_qwen3-next-80b-a3b-instruct_gold_combined.csv'),
    ('qwen36', 'qwen36-35b', 'rag_qwen36-35b',
     None),
]
ARMS = [('roundrobin', 'rag_titled_combined'), ('rrf', 'rag_titled_combined_rrf'),
        ('cite_rrf', 'rag_titled_combined_cite_rrf')]


def med(df, scale):
    """Per-case median over the scale's seats, or None if one is unfilled."""
    cols = []
    for c in scale:
        for alias in (c,) + SEAT_ALIASES.get(c, ()):
            if alias in df.columns and df[alias].notna().any():
                cols.append(alias)
                break
    return 100 * df[cols].median(axis=1, skipna=True).values if len(cols) == len(scale) else None


#: no-retrieval baselines, in the order they are preferred.
#:
#: The published ``no_rag_ji2_result.csv`` is last, and that ordering is the
#: whole point.  Its DeepSeek column was written in July, before the FAU
#: deployment swapped DeepSeek-V4-Flash for materially better weights; every RAG
#: arm in this table was judged in August, after.  Scoring the baseline with the
#: old seat and the arms with the new one books the seat upgrade as a retrieval
#: effect: on DeepSeek's own essays the same 81 answers score 30.87 under the old
#: seat and 38.89 under the new one, which is larger than the retrieval effect
#: being measured.  The re-judged files are the same essays under the current
#: seats, so they are what the arms may be differenced against.
#: ``norag_fresh_{f}_gold_combined.csv`` is what ``oracle_experiment.py run
#: --fresh-baseline --tag _gold_combined`` writes -- the same essays as the
#: bare ``norag_fresh`` file but generated alongside the combined-corpus oracle,
#: which is the only baseline some models have.
NORAG_SRC = ('rejudged_no_rag_{f}_panel.csv', 'rejudged_no_rag_{f}.csv',
             'norag_fresh_{f}_panel.csv', 'norag_fresh_{f}.csv',
             'norag_fresh_{f}_gold_combined.csv')

#: The no-retrieval run the paper's main results table reports, and now the
#: first baseline every table here tries. One no-RAG number per model across the
#: whole paper was the point: the ladder and the main table were printing
#: DeepSeek at 44.69 and 43.77, Qwen3-Next at 28.77 and 30.62, and a reader has
#: no way to tell which of those a delta was taken against.
#:
#: Two things separate it from the ``NORAG_SRC`` files, and only one of them is
#: a difference in kind.
#:
#: DeepSeek is the one in kind. Its row here is the ``-0731`` *re-generation*,
#: written after the 2026-08-01 weight swap, and so is every DeepSeek RAG arm in
#: these tables. The ``NORAG_SRC`` baseline was the pre-swap generation
#: re-judged, which differenced August arms against July essays and booked the
#: checkpoint upgrade as a retrieval null: on the current seats the two
#: generations score 44.44 and 43.77 for +0.67 of pure generator.
#:
#: The other is that this is a separate invocation of the same three seats on
#: the same essays with the same stored ``ji2`` prompt. The seats are sampled,
#: so that is not free: only about a third of cases come back with the same
#: score, per-case r between runs is 0.41-0.76, and a seat mean wanders by up to
#: 2.5 points in either direction. Within one run that noise is shared between
#: an arm and its baseline and cancels in the paired difference; across runs it
#: does not, so every paired SE in these tables is larger than it was. That is
#: the price of the shared baseline and it is paid knowingly.
MAIN_NORAG = f'{B}/without_rag_0731/no_rag_ji2_rejudged.csv'

#: endpoint identifier -> the row that run stores it under, where they differ.
#: Only DeepSeek differs, and see above for why it is the ``-0731`` row.
MAIN_NORAG_ID = {'deepseek-ai/DeepSeek-V4-Flash': 'deepseek-ai/DeepSeek-V4-Flash-0731'}

#: ...and the file the answers themselves live in, for the rows that need the
#: essay text rather than its score. ``MAIN_NORAG`` carries scores only.
#: The ``.csv`` twin of this file was deleted on 2026-08-23 (8298701f) as
#: subsumed by the ``.jsonl``, but this pointer kept naming the ``.csv``. From
#: then on ``norag_essays`` fell through to the published July run, so DeepSeek's
#: citation rows read pre-swap essays (29.05% gold norms cited) where the ladder
#: prints 25.86 from these 08-20 essays -- a silent change of no-RAG essays under
#: a printed baseline.
MAIN_NORAG_ESSAYS = {'deepseek-ai/DeepSeek-V4-Flash':
                     f'{B}/without_rag_0731/ds_v4_flash_0731_high.jsonl'}
PUB_NORAG = f'{B}/without_rag/ji2/no_rag_ji2_result.csv'

_main_norag = None
_pub_answers = None


def pub_answers():
    """``(index, answer, model)`` of the published no-RAG run, read once.

    The file is ~900 MB with the judgement text in it; three columns of it is
    small, but the parse is not, and every generator asks for a slice.
    """
    global _pub_answers
    if _pub_answers is None:
        _pub_answers = pd.read_csv(PUB_NORAG, usecols=['index', 'answer', 'model'])
    return _pub_answers


def main_norag():
    """The main table's no-RAG scores, read once."""
    global _main_norag
    if _main_norag is None:
        _main_norag = pd.read_csv(MAIN_NORAG) if os.path.exists(MAIN_NORAG) \
            else pd.DataFrame(columns=['model', 'index'])
    return _main_norag


def norag_essays(mid):
    """``(index, answer)`` for the essays :func:`baseline` scores, or None.

    The counterpart to :func:`baseline` for rows built from the essay text --
    citation recall, reference similarity -- so that they and the score rows
    describe the same 81 answers. Falls back to the published run for every
    model whose baseline is not a re-generation.
    """
    p = MAIN_NORAG_ESSAYS.get(mid)
    if p and os.path.exists(p):
        d = (pd.read_json(p, lines=True)[['index', 'answer']] if p.endswith('.jsonl')
             else pd.read_csv(p, usecols=['index', 'answer']))
        return d.sort_values('index')
    if (main_norag().model == MAIN_NORAG_ID.get(mid, mid)).any() \
            and os.path.exists(PUB_NORAG):
        d = pub_answers()
        d = d[d.model == mid]
        if len(d):
            return d.sort_values('index')
    return None


def baseline(mid, scale, norag):
    """Same-scale no-retrieval baseline for ``mid``, or None if no file fills it.

    ``MAIN_NORAG`` first, so every table in the paper differences against the
    number its main results table prints. It cannot serve the ``panel`` scale --
    it has no nano seat -- and it does not carry every model, so the older
    per-model files remain the fallback and a few rows still come from them.
    """
    row = main_norag()
    row = row[row.model == MAIN_NORAG_ID.get(mid, mid)]
    if len(row):
        b = med(row.sort_values('index'), scale)
        if b is not None:
            return b
    f = mid.replace('/', '_')
    for pat in NORAG_SRC:
        alt = f'{B}/oracle_rag/ji2/{pat.format(f=f)}'
        if os.path.exists(alt):
            b = med(pd.read_csv(alt), scale)
            if b is not None:
                return b
    return med(norag[norag.model == mid], scale) \
        if (norag.model == mid).any() else None


def build():
    norag = pd.read_csv(f'{B}/without_rag/ji2/no_rag_ji2_result.csv')
    rows = []
    for scale_name, scale in SCALES.items():
        for short, mid, stem, orc in MODELS:
            base = baseline(mid, scale, norag)
            cells = {'no_rag': base}
            for arm, d in ARMS:
                p = f'{B}/{d}/ji2/{stem}.csv'
                cells[arm] = med(pd.read_csv(p).sort_values('index'), scale) \
                    if os.path.exists(p) else None
            po = f'{B}/oracle_rag/ji2/{orc}' if orc else None
            cells['oracle'] = med(pd.read_csv(po), scale) \
                if po and os.path.exists(po) else None
            for arm, v in cells.items():
                if v is None:
                    continue
                r = dict(scale=scale_name, model=short, arm=arm, n=len(v), score=v.mean())
                if base is not None and arm != 'no_rag':
                    k = min(len(v), len(base))
                    d_ = v[:k] - base[:k]
                    r['delta'] = d_.mean()
                    r['delta_sem'] = d_.std(ddof=1) / np.sqrt(k)
                rows.append(r)
    return pd.DataFrame(rows)


#: endpoint identifier -> the name the paper uses
#: the arms of the retired table, still the order the console dump prints
COLS = ['no_rag', 'roundrobin', 'rrf', 'cite_rrf', 'oracle']


def main():
    t = build()
    t.to_csv(OUT, index=False)
    print(f'{len(t)} rows -> {OUT}\n')
    for scale in SCALES:
        s = t[t.scale == scale]
        print(f'=== {scale} ===')
        print(s.pivot(index='model', columns='arm', values='score')
              .reindex(columns=COLS)
              .round(2).to_string(na_rep='  --  '))
        print(s.pivot(index='model', columns='arm', values='delta')
              .reindex(columns=COLS[1:])
              .round(2).to_string(na_rep='  --  '), '\n')


if __name__ == '__main__':
    main()

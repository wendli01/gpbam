"""Pretraining-contamination probe: verbatim spans shared with the reference solution.

If a model had seen a GPBam *Lösung* during pretraining, its essay for that case
should reproduce long exact spans of it.  Coincidental overlap between two pieces
of German legal prose is short; a long exact match is not coincidence.

The confound is that both texts legitimately quote the same statutes.  We control
for it by measuring the same quantity against *other* cases' reference solutions:
those share the genre, the Gutachtenstil boilerplate and much statutory wording,
but nothing case-specific.  The contamination signal is the excess of the matched
pair over the control pairs, not the raw match length.

    L_own    longest exact word-span shared with this case's reference solution
    L_ctrl   the same, against CONTROLS other reference solutions (max)
    excess   L_own - L_ctrl

Model set: the main table's, read from ``panel.roster`` rather than restated
here.  Four of those models wrote their essays outside the published CSV -- the
0731 re-generation and the three frontier runs -- and are read from their own
JSONL, exactly as ``deepseek_rerun/scripts/recompute_legal_ref.py`` reads them.
Any model on the roster whose essays no source below supplies is named on
stdout, so a missing generator is a message rather than a smaller *n*.

Usage:  python analysis/verbatim_leakage.py
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

import panel

B = os.environ.get('GPBAM_EXPERIMENTS') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = f'{B}/zubaers_result/essay_writing/without_rag/ji2/no_rag_ji2_result.csv'

# Generations that live outside the published CSV, each as (jsonl, model id).
# The same four rows recompute_legal_ref.py adds, for the same reason: the model
# is in the main table, its essays are not in the published CSV.
FRONTIER = f'{B}/zubaers_result/essay_writing/frontier'
EXTRA = [
    (f'{B}/zubaers_result/essay_writing/without_rag_0731/ds_v4_flash_0731_high.jsonl',
     'deepseek-ai/DeepSeek-V4-Flash-0731'),
    (f'{FRONTIER}/gemini-3.7-flash_essays.jsonl', 'google/gemini-3.7-flash'),
    # `answer` here is already the text after `</think>` -- run_model.py splits
    # the trace off, so nothing below ever scores a model's own monologue.
    (f'{FRONTIER}/soofi-s-isar-preview_essays.jsonl', 'soofi-s-isar-preview'),
    # Written through the Claude Code agent harness rather than the API, in five
    # batches; all 81 cases, see additional_models/opus5_agent/README.md.
    (f'{B}/analysis/additional_models/opus5_agent/opus5_agent.jsonl',
     'anthropic/claude-opus-5'),
]

CONTROLS = 4
KMAX = 400
WORD = re.compile(r"[\wÄÖÜäöüß§]+", re.U)


def toks(s):
    return WORD.findall(str(s).lower())


def grams(t, k):
    return {hash(tuple(t[i:i + k])) for i in range(len(t) - k + 1)} if len(t) >= k else set()


def longest_match(a, b, kmax=KMAX):
    """Longest k with a shared k-gram, by binary search on k."""
    if not grams(a, 1) & grams(b, 1):
        return 0
    lo, hi = 1, min(kmax, len(a), len(b))
    while lo < hi:                              # invariant: lo matches, hi+1 does not
        mid = (lo + hi + 1) // 2
        if grams(a, mid) & grams(b, mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def essays():
    """Every no-RAG essay in the main table's model set, one row per essay."""
    pub = pd.read_csv(SRC, usecols=['index', 'model', 'answer'])
    frames = [pub]
    for path, mid in EXTRA:
        if not os.path.exists(path):
            print(f'skipping {mid}: no {path}')
            continue
        g = pd.DataFrame([json.loads(l) for l in open(path)]).drop_duplicates('index')
        g = g[g.answer.notna()]
        frames.append(g.assign(model=mid)[['index', 'model', 'answer']])
        print(f'{mid}: {len(g)} essays from {os.path.basename(path)}')
    d = pd.concat(frames, ignore_index=True)
    d['key'] = d.model.map(panel.key)
    panel.check(d.key.unique(), 'essays')
    d = d[d.key.isin(panel.roster().key)]
    return d.sort_values(['model', 'index'])


def main():
    d = essays()
    from oracle_experiment import cases
    _, s = cases()
    sols = pd.Series(s, index=range(len(s)))
    sol_toks = {i: toks(v) for i, v in sols.items()}
    cases_ = sorted(sol_toks)
    rng = np.random.default_rng(0)

    rows = []
    for n, (_, r) in enumerate(d.iterrows(), 1):
        e = toks(r['answer'])
        if len(e) < 50:
            continue
        i = r['index']
        own = longest_match(e, sol_toks[i])
        ctrl = [longest_match(e, sol_toks[j]) for j in
                rng.choice([c for c in cases_ if c != i], CONTROLS, replace=False)]
        rows.append(dict(model=r['model'], case=i, words=len(e),
                         L_own=own, L_ctrl=max(ctrl), excess=own - max(ctrl)))
        if n % 250 == 0:
            print(f'  {n}/{len(d)}', flush=True)
    out = pd.DataFrame(rows)
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out',
                       'verbatim_leakage.csv')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    out.to_csv(dst, index=False)

    print(f'\n{len(out)} essays, {out.model.nunique()} models\n')
    print('longest exact shared word-span (own reference vs control references)')
    print(f'  own    : median {out.L_own.median():.0f}  p95 {out.L_own.quantile(.95):.0f}  max {out.L_own.max()}')
    print(f'  control: median {out.L_ctrl.median():.0f}  p95 {out.L_ctrl.quantile(.95):.0f}  max {out.L_ctrl.max()}')
    print(f'  excess : median {out.excess.median():+.0f}  p95 {out.excess.quantile(.95):+.0f}  max {out.excess.max():+d}')
    print(f'\nessays with excess > 20 words: {(out.excess > 20).sum()}'
          f'  >= 20: {(out.excess >= 20).sum()}  >= 40: {(out.excess >= 40).sum()}')
    g = out.groupby('model').agg(n=('excess', 'size'),
                                 med_own=('L_own', 'median'), med_ctrl=('L_ctrl', 'median'),
                                 med_excess=('excess', 'median'), max_excess=('excess', 'max'))
    print('\nby model, worst 10 by median excess:')
    print(g.sort_values('med_excess', ascending=False).head(10).to_string())
    print('\nfull per-model table:')
    print(g.sort_values('max_excess', ascending=False).to_string())


if __name__ == '__main__':
    main()

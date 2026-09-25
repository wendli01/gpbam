"""Does citing the gold norms explain the judge score, condition by condition?

Two questions, and they have different answers.

**Levels.** Within one condition, do the essays that cite more of the reference
solution's norms score better? Yes, modestly: r = 0.46 pooled over models, but
that pools between-model differences (a stronger model both cites more and
scores higher). The honest figure is the mean within-model correlation, 0.19.

**Changes.** When a retrieval arm makes a model cite more gold norms *in a given
case*, does that case score better? Barely. The per-case delta correlations run
0.03 to 0.22, and the slope says a 10-point gain in citation recall buys 1-4
points of judge score. The oracle moves citation recall by 8 to 26 points, which
on that slope predicts 1-3 points; it actually gains 3.2 to 6.0. So most of the
oracle's benefit is not the naming of the norms -- consistent with the
``_gold_cites`` ablation, where the norms' identifiers without their text bought
nothing at all.

Read the delta correlations against the per-case reliability of the judge, not
against 1.0: re-generating the same arm and re-scoring it gives per-case r =
0.26, so the ceiling for any per-case correlation here is about sqrt(0.26) =
0.51. The oracle's 0.121 corrects to roughly 0.24 -- still a minority of the
effect.

Run from ``experiments/``::

    PYTHONPATH=analysis python analysis/cite_score_link.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ladder_table as lt
import paper_table as pt

CITE = 'analysis/out/ladder_citations.csv'
OUT = 'analysis/out/cite_score_link.csv'
#: per-case reliability of the judge, from the stored re-generation of
#: rag_titled_combined on Qwen3.6 (see the ladder table's noise-floor note)
RELIABILITY = 0.26


def load():
    cit = pd.read_csv(CITE)
    norag = pd.read_csv(f'{pt.B}/without_rag/ji2/no_rag_ji2_result.csv').sort_values('index')
    scores = {s: lt.essay_row(m, st, o, norag) for s, m, st, o in lt.GENERATORS}
    return cit, scores


def pairs(cit, scores, arm, delta):
    """Per-case (citation recall, judge score) pairs, pooled over models."""
    cs, ss, per_model = [], [], []
    for short, g in cit.groupby('model'):
        sc = scores.get(short, {})
        if arm not in sc or arm not in g or g[arm].isna().all():
            continue
        g = g.sort_values('index')
        n = min(len(sc[arm]), len(g))
        c, s = g[arm].values[:n], np.asarray(sc[arm])[:n]
        if delta:
            if 'no_rag' not in sc or 'no_rag' not in g:
                continue
            c = c - g['no_rag'].values[:n]
            s = s - np.asarray(sc['no_rag'])[:n]
        ok = ~(np.isnan(c) | np.isnan(s))
        if ok.sum() < 10:
            continue
        cs.append(c[ok])
        ss.append(s[ok])
        per_model.append((short, stats.pearsonr(c[ok], s[ok])[0]))
    if not cs:
        return None
    return np.concatenate(cs), np.concatenate(ss), per_model


def main():
    cit, scores = load()
    rows = []
    for arm, *_ in lt.COLS:
        for delta in (False, True):
            if delta and arm == 'no_rag':
                continue
            got = pairs(cit, scores, arm, delta)
            if got is None:
                continue
            c, s, per_model = got
            r, p = stats.pearsonr(c, s)
            sl, _, _, _, se = stats.linregress(c, s)
            rows.append(dict(arm=arm, kind='delta' if delta else 'level',
                             models=len(per_model), n=len(c), r=r, p=p,
                             r_within_mean=np.mean([x for _, x in per_model]),
                             slope=sl, slope_sem=se,
                             r_corrected=r / np.sqrt(RELIABILITY)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT, index=False)
    print(f'{len(t)} rows -> {OUT}\n')
    for kind, title in (('level', 'Within condition (levels)'),
                        ('delta', 'Change against no RAG (per case)')):
        s = t[t.kind == kind].set_index('arm')
        print(f'=== {title} ===')
        print(s[['models', 'n', 'r', 'p', 'r_within_mean', 'slope', 'slope_sem',
                 'r_corrected']].round(3).to_string())
        print()


if __name__ == '__main__':
    main()

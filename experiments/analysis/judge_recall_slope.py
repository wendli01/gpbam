"""Does a judge reward retrieval *more* when retrieval actually worked?

The panel and free scales disagree about RAG because gpt-5-nano barely moves
for it: mean delta +0.12 against +1.23 and +2.98 for the two free seats. Two
readings fit that equally well from the means alone -- nano is accurate and RAG
genuinely does little, or nano cannot verify a retrieved statute and ignores it.

Recitation ranks the seats in the predicted order (DeepSeek 0.337, Qwen 0.167,
nano 0.104 ROUGE-L on the GPBam statutes), but with three seats that is a rank
correlation on three points, and recitation is confounded with being a stronger
model in general.

This is the within-judge version, which does not have that confound. Retrieval
quality varies case to case: on some of the 81 the retriever hands over a good
share of the norms the reference solution cites, on others almost none. So for
one seat and one arm, regress

    y_i = score(arm, case i) - score(no RAG, case i)        [that seat only]
    x_i = |gold norms of case i that the arm's context contained| / |gold|

A judge that can tell a correct retrieved norm from a wrong one should score
the high-recall cases higher: positive slope. A judge that ignores retrieved
material regardless has a flat slope even if the arm's *mean* delta is positive
for unrelated reasons -- length, structure, confidence. The judge is held fixed
across the 81 points, so general capability cancels, and y is already a
within-case difference, so case difficulty differences out.

The context is read back out of the stored ``qa_prompt``, so nothing is
retrieved or generated again. It is byte-identical across generator models for
a given arm -- same retriever, same query -- which is asserted below rather than
assumed, so the five models are five replicates of one x.

Cells are (model, arm). Both variables are demeaned within cell before pooling,
which removes every between-cell mean difference and leaves only the question
asked: within a cell, do the better-retrieved cases gain more? Standard errors
cluster on case, because the same 81 cases recur in all 15 cells.

Run from ``experiments/``::

    PYTHONPATH=analysis python analysis/judge_recall_slope.py
"""

import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import refs as refs_mod
from oracle_experiment import cases
import paper_table as pt

OUT = 'analysis/out/judge_recall_slope.csv'
SEATS = {'DeepSeek-V4-Flash': [pt.DS], 'Qwen3.6-35B-A3B': [pt.QWEN],
         'gpt-5-nano': [pt.NANO]}
#: recitation of the GPBam statutes, ROUGE-L F1, for the read-out
RECITATION = {'DeepSeek-V4-Flash': 0.337, 'Qwen3.6-35B-A3B': 0.167,
              'gpt-5-nano': 0.104}
NORM_RE = re.compile(r'<norm zitat="([^"]*)"')
SOURCE_RE = re.compile(r'\[SOURCE \d+: (.*?) \(')


def gold_by_case():
    """``{case index: {(book, section)}}`` from the reference solutions."""
    _, sols = cases()
    # corpus_key, not book.lower(): the corpus stores several books under names
    # the solutions do not use -- BBauG for BauGB among them -- and
    # oracle_experiment.gold_sets keys on corpus_key for exactly that reason.
    # Keying gold one way and the retrieved context the other silently
    # undercounts every intersection.
    return {i: {(refs_mod.corpus_key(c.book), c.section.lower())
                for c in refs_mod.canonicalise(refs_mod.extract(s))}
            for i, s in enumerate(sols)}


def context_norms(path):
    """``{case index: {(book, section)}}`` actually supplied to the generator."""
    d = pd.read_csv(path, usecols=['index', 'qa_prompt']).sort_values('index')
    out = {}
    for i, q in zip(d['index'], d['qa_prompt']):
        q = str(q)
        cites = NORM_RE.findall(q) or SOURCE_RE.findall(q)
        keys = set()
        for cite in cites:
            # Two context formats, and they are written in opposite orders. The
            # <norm zitat="..."> arms use the citation form a Gutachten uses --
            # "§ 34 BauGB" -- which extract parses. The [SOURCE n: ...] oracle
            # arms label passages book-first -- "BAYVWVFG Art. 48" -- which
            # extract deliberately does not parse, because that is the
            # Normenkette order. chain_norms reads it. Trying extract first and
            # falling back keeps both readable without sniffing the format.
            hit = {(refs_mod.corpus_key(c.book), c.section.lower())
                   for c in refs_mod.canonicalise(refs_mod.extract(cite))}
            keys |= hit or refs_mod.chain_norms(cite)
        out[int(i)] = keys
    if not any(out.values()):
        raise SystemExit(f'{path}: no citations parsed out of qa_prompt')
    return out


def recall_by_arm(gold):
    """Per-case gold recall of each arm's context, and the raw hit count."""
    out = {}
    for arm, d in pt.ARMS:
        ctx, seen = None, None
        for _, _, stem, _ in pt.MODELS:
            p = f'{pt.B}/{d}/ji2/{stem}.csv'
            if not os.path.exists(p):
                continue
            c = context_norms(p)
            if ctx is None:
                ctx, seen = c, stem
            elif c != ctx:
                raise SystemExit(f'{arm}: context differs between {seen} and '
                                 f'{stem}; the five models are not replicates')
        if ctx is None:
            continue
        out[arm] = pd.DataFrame([
            dict(case=i, hits=len(gold[i] & ctx[i]), n_gold=len(gold[i]),
                 recall=len(gold[i] & ctx[i]) / len(gold[i]) if gold[i] else np.nan)
            for i in sorted(ctx)])
    return out


def panel(gold):
    """One row per (seat, model, arm, case): recall and that seat's delta."""
    rec = recall_by_arm(gold)
    norag = pd.read_csv(f'{pt.B}/without_rag/ji2/no_rag_ji2_result.csv') \
              .sort_values('index')
    rows = []
    for seat, scale in SEATS.items():
        for short, mid, stem, _ in pt.MODELS:
            base = pt.baseline(mid, scale, norag)
            if base is None:
                continue
            for arm, d in pt.ARMS:
                p = f'{pt.B}/{d}/ji2/{stem}.csv'
                if not os.path.exists(p) or arm not in rec:
                    continue
                v = pt.med(pd.read_csv(p).sort_values('index'), scale)
                if v is None:
                    continue
                k = min(len(v), len(base))
                r = rec[arm].iloc[:k]
                rows.append(pd.DataFrame(dict(
                    seat=seat, model=short, arm=arm, case=r['case'].values,
                    recall=r['recall'].values, hits=r['hits'].values,
                    n_gold=r['n_gold'].values,
                    delta=v[:k] - base[:k])))
    return pd.concat(rows, ignore_index=True).dropna(subset=['delta', 'recall'])


def slope(df, xcol='recall'):
    """Within-cell OLS of delta on *xcol*, SE clustered on case."""
    g = df.groupby(['model', 'arm'])
    x = (df[xcol] - g[xcol].transform('mean')).values
    y = (df['delta'] - g['delta'].transform('mean')).values
    xx = x @ x
    if xx == 0:
        return np.nan, np.nan, len(df)
    b = (x @ y) / xx
    e = y - b * x
    cl = pd.DataFrame({'s': x * e, 'c': df['case'].values})
    meat = (cl.groupby('c')['s'].sum() ** 2).sum()
    n, k, G = len(df), df.groupby(['model', 'arm']).ngroups + 1, df['case'].nunique()
    adj = (G / (G - 1)) * ((n - 1) / (n - k))
    return b, np.sqrt(adj * meat / xx ** 2), n


#: same pipeline, same corpus, different budget -- the only pair on record that
#: varies how much is supplied while holding the retrieval method fixed
SUPPLY_PAIR = ('rag_titled_combined_rrf', 'rag_titled_combined_rrf_k50',
               'rag_deepseek-ai_DeepSeek-V4-Flash')


def supply(gold):
    """Separate the credit for a correct norm from the penalty for a junk one.

    Within any single arm the two are perfectly collinear: every arm supplies a
    fixed k, so junk = k - hits and only their difference is identified. The
    k=10/k=50 RRF pair breaks that, because the budget moves while the pipeline
    does not.

    Difference the two arms case by case. The no-RAG baseline cancels, so this
    needs no baseline at all, and case difficulty differences out twice over::

        y_i = score(k50, i) - score(k10, i)
        x_i = hits(k50, i)  - hits(k10, i)

    Supply rises by exactly 40 on every case, so x and the junk increment are
    collinear *given an intercept*, and that is what identifies both terms:
    forty more passages of which (40 - x) are junk. Hence

        junk per norm = intercept / 40
        gold per norm = slope + intercept / 40

    because raising hits by one lowers junk by one at fixed budget.
    """
    a_dir, b_dir, stem = SUPPLY_PAIR
    ha = context_norms(f'{pt.B}/{a_dir}/ji2/{stem}.csv')
    hb = context_norms(f'{pt.B}/{b_dir}/ji2/{stem}.csv')
    da = pd.read_csv(f'{pt.B}/{a_dir}/ji2/{stem}.csv').sort_values('index')
    db = pd.read_csv(f'{pt.B}/{b_dir}/ji2/{stem}.csv').sort_values('index')
    idx = sorted(set(ha) & set(hb))
    x = np.array([len(gold[i] & hb[i]) - len(gold[i] & ha[i]) for i in idx], float)
    d_supply = np.mean([len(hb[i]) - len(ha[i]) for i in idx])

    print(f'\n{b_dir.split("_")[-1]} vs k10, same pipeline, {stem.split("_")[-1]}: '
          f'{d_supply:.0f} more passages per case, '
          f'{x.mean():+.2f} of them gold (range {x.min():+.0f} to {x.max():+.0f})')
    print(f"{'seat':20s} {'gold/norm':>18s} {'junk/norm':>18s}")
    rows = []
    for seat, scale in SEATS.items():
        va, vb = pt.med(da, scale), pt.med(db, scale)
        if va is None or vb is None:
            print(f'{seat:20s} {"-- seat unfilled --":>18s}')
            continue
        y = (vb - va)[:len(x)]
        m = ~np.isnan(y)
        X = np.column_stack([np.ones(m.sum()), x[m]])
        b, *_ = np.linalg.lstsq(X, y[m], rcond=None)
        e = y[m] - X @ b
        xtx_i = np.linalg.inv(X.T @ X)
        # HC1: one observation per case, so heteroskedasticity is the only worry
        V = xtx_i @ (X.T @ np.diag(e ** 2) @ X) @ xtx_i * len(e) / (len(e) - 2)
        se = np.sqrt(np.diag(V))
        junk, junk_se = b[0] / d_supply, se[0] / d_supply
        g, g_se = b[1] + junk, np.sqrt(V[1, 1] + V[0, 0] / d_supply ** 2
                                       + 2 * V[0, 1] / d_supply)
        print(f'{seat:20s} {g:+9.3f} +/- {g_se:5.3f} {junk:+9.3f} +/- {junk_se:5.3f}')
        rows.append(dict(seat=seat, gold_per_norm=g, gold_se=g_se,
                         junk_per_norm=junk, junk_se=junk_se, n=int(m.sum())))
    print('points of judge score per norm supplied; junk is a norm the reference\n'
          'solution does not cite')
    return pd.DataFrame(rows)

def main():
    gold = gold_by_case()
    d = panel(gold)
    d.to_csv(OUT, index=False)
    print(f'{len(d)} rows -> {OUT}')

    a = d.drop_duplicates(['arm', 'case'])
    print('\nretrieval quality actually varies (per arm, over the 81 cases):')
    print(a.groupby('arm')[['recall', 'hits', 'n_gold']]
           .agg(['mean', 'min', 'max']).round(3).to_string(), '\n')

    print('delta = f(gold recall), within (model, arm), SE clustered on case')
    print(f"{'seat':20s} {'recite':>7s} {'slope/10pp':>12s} {'t':>6s} {'mean d':>8s}  n")
    for seat in SEATS:
        s = d[d.seat == seat]
        b, se, n = slope(s)
        print(f'{seat:20s} {RECITATION[seat]:7.3f} '
              f'{b / 10:+7.2f} +/- {se / 10:4.2f} {b / se:6.2f} '
              f'{s["delta"].mean():+8.2f}  {n}')
    print('\nslope is score points per 10 percentage points of gold recall')
    supply(gold).to_csv(OUT.replace('.csv', '_supply.csv'), index=False)

    # The slope is the recall-*proportional* credit. What is left over is the
    # credit a seat pays for having retrieved material at all, correct or not:
    # each cell's own intercept, averaged over cells.
    print('\ndecomposing the mean delta at the seat level:')
    print(f"{'seat':20s} {'flat':>7s} {'x recall':>9s} {'= mean d':>9s}")
    for seat in SEATS:
        sd = d[d.seat == seat]
        b, _, _ = slope(sd)
        g = sd.groupby(['model', 'arm'])
        flat = (g['delta'].mean() - b * g['recall'].mean()).mean()
        print(f'{seat:20s} {flat:+7.2f} {b * sd["recall"].mean():+9.2f} '
              f'{sd["delta"].mean():+9.2f}')
    print('flat = fitted delta at zero gold recall; the columns are additive only\n'
          'up to the spread of recall across cells')

    print('\nsame slope, one arm at a time (SE clustered on case):')
    print(f"{'seat':20s} " + ' '.join(f'{a:>16s}' for a, _ in pt.ARMS))
    for seat in SEATS:
        cells = []
        for arm, _ in pt.ARMS:
            b, se, n = slope(d[(d.seat == seat) & (d.arm == arm)])
            cells.append(f'{b / 10:+6.2f} +/- {se / 10:4.2f}' if n else f'{"--":>16s}')
        print(f'{seat:20s} ' + ' '.join(cells))


if __name__ == '__main__':
    main()

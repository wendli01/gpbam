"""Is retrieving the right norms possible without first solving the case?

HyDE retrieves with a hypothetical answer rather than the question.  Here it costs
nothing to evaluate, because the hypothetical answers already exist: the 2,430
essays from the **no-retrieval** run are exactly what a HyDE step would generate,
one per (model, case), spanning the full quality range from llama-3.1-8b at 4.4
to DeepSeek-V4-Flash at 46.4.

Using them as queries turns a rhetorical question into a measurement.  If
retrieval recall rises with the quality of the draft answer, then retrieval is
not a step that can precede the reasoning -- the pipeline would have to solve the
case to know what to look up.  The reference solution itself gives the ceiling.

Run from ``experiments/``::

    PYTHONPATH=analysis python analysis/hyde_simulation.py             # fts, all models
    PYTHONPATH=analysis python analysis/hyde_simulation.py --hybrid    # + dense channel
"""

import argparse
import json

import numpy as np
import pandas as pd

from retrieval_ablation import Retriever, norm_of, is_structural
from retrieval_diagnostics import refex_norms, GPBAM, OUT

NORAG_CSV = 'zubaers_result/essay_writing/without_rag/ji2/no_rag_ji2_result.csv'
JUDGES = ['score_Judge (gpt-5-nano)', 'score_Judge (DeepSeek-V4-Flash)']
QWEN_FP8, QWEN_API = 'score_Judge (Qwen3.6-35B-A3B-FP8)', 'score_Judge (qwen3.6-35b-a3b)'


def score_of(df):
    q = df[QWEN_FP8].fillna(df[QWEN_API])
    return df[JUDGES].join(q.rename('q')).median(axis=1, skipna=True)


def chunks(text, words=120, max_chunks=12):
    """Split a draft answer into query-sized pieces, spread over the whole document.

    Necessary, not cosmetic: a *Gutachten* opens with a table of contents, so
    truncating to the first N words -- as a single-query search must -- throws away
    the entire *Begründetheit* and tests the wrong thing.  Real HyDE
    implementations chunk the hypothetical document for the same reason.
    """
    w = str(text).split()
    if len(w) <= words:
        return [' '.join(w)]
    parts = [' '.join(w[i:i + words]) for i in range(0, len(w), words)]
    if len(parts) <= max_chunks:
        return parts
    idx = np.linspace(0, len(parts) - 1, max_chunks).round().astype(int)
    return [parts[i] for i in sorted(set(idx))]


def recall_for(ret, queries_by_case, gold, k, mode, drop_structural=True, chunked=True):
    hits = tot = 0
    for case, q in queries_by_case.items():
        g = gold.get(case)
        if not g or not isinstance(q, str) or len(q) < 50:
            continue
        if chunked:
            rows = ret.search_multi(chunks(q), k, mode, drop_structural)
        else:
            rows = ret.search(q, k, mode, drop_structural)
        got = {n for n in (norm_of(r) for r in rows) if n}
        hits += len(got & g)
        tot += len(g)
    return hits / tot if tot else float('nan')


def main(hybrid=False, device='cuda', k=5, models=None):
    d = json.load(open(GPBAM))
    order = sorted(d['solutions'], key=lambda x: int(x))
    gold = {i: refex_norms(d['solutions'][k_]) for i, k_ in enumerate(order)}
    facts = {i: d['facts'][k_] for i, k_ in enumerate(order)}
    sols = {i: d['solutions'][k_] for i, k_ in enumerate(order)}

    nr = pd.read_csv(NORAG_CSV)
    nr['_score'] = score_of(nr)
    quality = nr.groupby('model')._score.mean().mul(100).sort_values()

    modes = ['fts', 'hybrid'] if hybrid else ['fts']
    ret = Retriever(device=device, need_vectors=hybrid)

    rows = []
    for mode in modes:
        base = recall_for(ret, facts, gold, k, mode)
        oracle = recall_for(ret, sols, gold, k, mode)
        print(f'\n=== {mode}, k={k} ===')
        print(f'{"case facts (no HyDE)":<42} essay score    --   recall {100 * base:5.2f}%')
        rows.append(dict(mode=mode, source='case facts', essay_score=np.nan, recall=base))

        subset = quality.index if not models else [m for m in quality.index
                                                   if any(t in m for t in models)]
        for model in subset:
            ans = nr[nr.model == model].set_index('index').answer.to_dict()
            r = recall_for(ret, ans, gold, k, mode)
            rows.append(dict(mode=mode, source=model, essay_score=quality[model], recall=r))
            print(f'HyDE: {model.split("/")[-1]:<36} {quality[model]:5.1f}        recall {100 * r:5.2f}%')

        print(f'{"HyDE oracle: the reference solution":<42} {"100.0":>5}        '
              f'recall {100 * oracle:5.2f}%')
        rows.append(dict(mode=mode, source='reference solution', essay_score=100.0, recall=oracle))

    out = pd.DataFrame(rows)
    out.to_csv(f'{OUT}/hyde_simulation.csv', index=False)

    for mode in modes:
        sub = out[(out['mode'] == mode) & out.essay_score.notna() & (out.essay_score < 99)]
        if len(sub) > 2:
            from scipy.stats import spearmanr
            sub = sub.dropna(subset=['recall'])
            rho, p = spearmanr(sub.essay_score, sub.recall)
            print(f'\n[{mode}] recall vs. the quality of the draft answer used as the query: '
                  f'rho={rho:.2f} (p={p:.4f}, n={len(sub)} models)')
    print(f'\nwritten to {OUT}/hyde_simulation.csv')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--hybrid', action='store_true')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--models', nargs='*', default=None,
                    help='substrings selecting which no-RAG runs to use as HyDE documents')
    main(**vars(ap.parse_args()))

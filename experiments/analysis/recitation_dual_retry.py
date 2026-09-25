"""Re-generate the individual delta answers that came back empty.

``recitation_dual_delta.py`` writes one row per provision whether or not the
call succeeded, so a transport-level drop leaves a NaN answer in an otherwise
complete file -- which the sweep's resume check reads as "done".  This retries
just those cells, in place, and rescores them.

Run from ``experiments/``::

    python analysis/recitation_dual_retry.py --list
    python analysis/recitation_dual_retry.py
"""
import argparse
import os
import sys
from glob import glob
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recitation_dual_delta as dd                          # noqa: E402  (env shim runs here)
from src import prompts, qa, config                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--attempts', type=int, default=2)
    args = ap.parse_args()

    holes = []
    for path in sorted(glob(str(dd.OUT / 'delta_*.csv'))):
        d = pd.read_csv(path)
        bad = d[d.answer.isna()]
        for i, r in bad.iterrows():
            holes.append((path, i, r['model'], r['query']))
    if not holes:
        print('no empty answers')
        return
    for _, _, m, q in holes:
        print(f'  hole  {m:44s} {q}')
    if args.list:
        return

    for path in sorted({p for p, *_ in holes}):
        d = pd.read_csv(path)
        m = d['model'].iloc[0]
        idx = list(d.index[d.answer.isna()])
        kw = {}
        if m in dd.ROUTE_OVERRIDES:
            ep, _ = dd.ROUTE_OVERRIDES[m]
            cfg = config._load()['endpoints'][ep]
            kw = {'inference_endpoint': cfg['base_url'], 'token_var': cfg['token_env']}
        wire = dd.ROUTE_OVERRIDES[m][1] if m in dd.ROUTE_OVERRIDES else m
        ar = qa.AnswerGenerator(prompt=prompts.AR_USER, system_prompt=prompts.AR_SYSTEM,
                                model=wire, max_tokens=None, **kw)
        for attempt in range(args.attempts):
            if not idx:
                break
            pred = ar.predict(d.loc[idx, 'query'].tolist())
            for i, p in zip(list(idx), pred):
                if p:
                    d.loc[i, 'answer'] = p
                    d.loc[i, 'score'] = dd.score_ar([d.loc[i, 'target']], [p])[0]
            idx = [i for i in idx if pd.isna(d.loc[i, 'answer'])]
            print(f'{m}: attempt {attempt + 1}, {len(idx)} still empty')
        d.to_csv(path, index=False)
        print(f'{m}: {d.answer.notna().sum()}/{len(d)} answered, '
              f'mean ROUGE-L {100 * d.score.mean():.2f} -> {Path(path).name}')

    files = sorted(glob(str(dd.OUT / 'delta_*.csv')))
    allrows = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    allrows.to_csv(dd.OUT / 'recitation_dual_delta.csv', index=False)
    print(f'\n{len(allrows)} rows across {allrows.model.nunique()} models rewritten')


if __name__ == '__main__':
    main()

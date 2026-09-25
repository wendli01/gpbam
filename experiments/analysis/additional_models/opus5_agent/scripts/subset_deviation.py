"""How much a partial-coverage row can differ from the full-81 score it stands in for.

The Opus row answers a contiguous subset of the 81 cases, so its mean is over
that subset rather than over the benchmark. This puts a number on what that
costs, empirically rather than by assumption: for every model that answered all
81 cases, draw random k-case subsets and measure how far the subset mean lands
from that model's own full-81 mean.

The answer is an upper bound on the sampling part of the Opus row's error and
says nothing about the harness caveats in README.md, which do not shrink with k.

    python scripts/subset_deviation.py [k ...]      # default 62 72
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
REJUDGED = ROOT / ('experiments/zubaers_result/essay_writing/without_rag_0731/'
                   'no_rag_ji2_rejudged.csv')
SEATS = ['score_Judge (gpt-oss-120b)', 'score_Judge (qwen3.6-35b-a3b)',
         'score_Judge (DeepSeek-V4-Flash-0731)']
DRAWS = 400

ks = [int(a) for a in sys.argv[1:]] or [62, 72]
df = pd.read_csv(REJUDGED)
df['med'] = df[SEATS].median(axis=1) * 100
med = df.dropna(subset=['med'])[['model', 'index', 'med']]
full = med.groupby('model')['index'].nunique()
models = full[full == 81].index
assert len(models), 'no model with all 81 cases'

rng = np.random.default_rng(0)
for k in ks:
    devs = []
    for m in models:
        v = med[med.model == m].sort_values('index')['med'].to_numpy()
        idx = np.array([rng.permutation(81)[:k] for _ in range(DRAWS)])
        devs.append(np.abs(v[idx].mean(axis=1) - v.mean()))
    d = np.concatenate(devs)
    print(f'k={k:3d}: mean abs deviation {d.mean():.2f}, p95 {np.percentile(d, 95):.2f}'
          f'   ({len(models)} models x {DRAWS} draws)')

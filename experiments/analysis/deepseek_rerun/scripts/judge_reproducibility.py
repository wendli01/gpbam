"""Judge reproducibility: the same judge, the same prompts, one month apart.

Qwen3.6-35B-A3B-FP8 was NOT affected by the 2026-08-01 DeepSeek weight swap, so
re-running it measures how much of a GPBam score is reproducible and how much is
judge non-determinism -- at essay, model and rank level -- and doubles as the control
that makes the DeepSeek checkpoint effect attributable.

Also bootstraps over cases to answer "how many cases does a stable model score need?".
Pure resampling: no API calls.

    python judge_reproducibility.py [outdir]
"""
import glob, sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

# Derived, not hardcoded: an absolute path here resolves against whatever
# checkout happens to live there rather than the one this file is in.
ROOT = Path(__file__).resolve().parents[4]
RERUN_SEAT = 'Qwen3.6-35B'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from seats import load_seat  # noqa: E402
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
PUB = ROOT / 'experiments/zubaers_result/essay_writing/without_rag/ji2/no_rag_ji2_result.csv'
QCOLS = ['score_Judge (Qwen3.6-35B-A3B-FP8)', 'score_Judge (qwen3.6-35b-a3b)']
SEED = 20260820

new = load_seat(RERUN_SEAT, ['index', 'model', 'score'])
new['qw_new'] = new.score * 100
old = pd.concat([c for c in pd.read_csv(PUB, usecols=['index', 'model'] + QCOLS, chunksize=20000)])
old['qw_pub'] = old[QCOLS[0]].combine_first(old[QCOLS[1]]) * 100
m = old[['index', 'model', 'qw_pub']].merge(new[['index', 'model', 'qw_new']], on=['index', 'model']).dropna()
d = m.qw_new - m.qw_pub

rows = []
rows.append(('essay', 'n', len(m)))
rows.append(('essay', 'mean delta', round(d.mean(), 2)))
rows.append(('essay', 'sd delta', round(d.std(), 2)))
rows.append(('essay', 'mean |delta|', round(d.abs().mean(), 2)))
rows.append(('essay', 'identical %', round((d == 0).mean() * 100, 1)))
rows.append(('essay', 'within one 10-pt step %', round((d.abs() <= 10).mean() * 100, 1)))
rows.append(('essay', 'pearson r', round(stats.pearsonr(m.qw_pub, m.qw_new)[0], 3)))

g = m.groupby('model').agg(pub=('qw_pub', 'mean'), new=('qw_new', 'mean'))
g['delta'] = g.new - g.pub
tt = stats.ttest_1samp(g.delta, 0)
rows.append(('model', 'n models', len(g)))
rows.append(('model', 'mean delta', round(g.delta.mean(), 2)))
rows.append(('model', 'SEM', round(g.delta.std() / np.sqrt(len(g)), 2)))
rows.append(('model', 't / p', f'{tt.statistic:.2f} / {tt.pvalue:.3f}'))
rows.append(('model', 'mean |delta|', round(g.delta.abs().mean(), 2)))
rows.append(('model', 'max |delta|', round(g.delta.abs().max(), 2)))
rows.append(('model', 'pearson r', round(stats.pearsonr(g.pub, g.new)[0], 4)))
rows.append(('model', 'kendall tau-b', round(stats.kendalltau(g.pub, g.new, variant='b')[0], 4)))

ro, rn = g.pub.rank(ascending=False), g.new.rank(ascending=False)
rows.append(('rank', 'models moving at all', f'{(ro != rn).sum()}/{len(g)}'))
rows.append(('rank', 'moving >=2 places', f'{((rn - ro).abs() >= 2).sum()}/{len(g)}'))
rows.append(('rank', 'top-5 set identical', set(g.pub.nlargest(5).index) == set(g.new.nlargest(5).index)))
summary = pd.DataFrame(rows, columns=['level', 'statistic', 'value'])
summary.to_csv(OUT / 'judge_reproducibility.csv', index=False)
print(summary.to_string(index=False))

# how many cases does a stable model score need?
rng = np.random.default_rng(SEED)
cases = sorted(m['index'].unique())
piv_p = m.pivot_table(index='model', columns='index', values='qw_pub')
piv_n = m.pivot_table(index='model', columns='index', values='qw_new')
curve = []
for k in [5, 10, 20, 30, 40, 50, 60, 70, 81]:
    reps = []
    for _ in range(400):
        sel = rng.choice(cases, size=k, replace=False)
        reps.append((piv_n[sel].mean(axis=1) - piv_p[sel].mean(axis=1)).abs().mean())
    curve.append(dict(n_cases=k, mean_abs_delta=round(float(np.mean(reps)), 2),
                      p95=round(float(np.percentile(reps, 95)), 2)))
cv = pd.DataFrame(curve)
cv.to_csv(OUT / 'cases_needed_curve.csv', index=False)
print('\nHow reproducible is a model score, as a function of cases used:')
print(cv.to_string(index=False))

with open(OUT / 'judge_reproducibility.tex', 'w') as f:
    f.write('% Qwen3.6-35B re-run: same model, same prompts, ~1 month apart\n')
    f.write('\\begin{tabular}{llr}\n\\toprule\nLevel & Statistic & Value \\\\\n\\midrule\n')
    for _, r in summary.iterrows():
        f.write(f'{r.level} & {r.statistic} & {r.value} \\\\\n')
    f.write('\\bottomrule\n\\end{tabular}\n')
print(f'\nwrote judge_reproducibility.{{csv,tex}} and cases_needed_curve.csv to {OUT}')

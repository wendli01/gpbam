"""Is the bimodal per-case score distribution peculiar to the Opus row?

The Opus write-up notes that its per-case judge medians pile up at 90-100 with a
second lump at 30-40, and that adding cases did not smooth it. That is only
interesting if strong models normally produce a single hump. This measures the
shape for every row on the same footing.

Per case, the median over the three re-judged seats -- the same statistic the
main table averages. The Opus row is taken from opus5_agent/scores.csv rather
than from the CSV. The two now agree case for case -- checked, all 81 -- since
make_rejudged_result.py carries the same judged/ output into the CSV; reading
the source of that row keeps this script correct if the CSV is ever rebuilt
from a different pass.

Two readings, because neither alone is honest:

  * The share at each end. Cheap and directly comparable, but confounded with
    the mean: a model averaging 20 has mass near the floor by construction, and
    that is not bimodality.
  * Sarle's bimodality coefficient, BC = (skew^2 + 1) / corrected kurtosis, with
    the usual 5/9 reference. It is scale-free, so it is not confounded with the
    mean the way the tail shares are -- but it rises for any low-kurtosis shape,
    including a flat one, so a high BC is "not one tight hump", not "two humps".
    The histogram is printed so the distinction is visible rather than asserted.
"""
import pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[4]
CSV = ROOT / 'experiments/zubaers_result/essay_writing/without_rag_0731/no_rag_ji2_rejudged.csv'
OPUS = ROOT / 'experiments/analysis/additional_models/opus5_agent/scores.csv'

df = pd.read_csv(CSV)
seats = [c for c in df.columns if c.startswith('score_')]
df['med'] = df[seats].median(axis=1) * 100

series = {m: g['med'].dropna().values for m, g in df.groupby('model')}
# scores.csv is where the Opus judging is produced; the CSV copies it.
series = {m: v for m, v in series.items() if 'opus' not in m.lower()}
series['anthropic/claude-opus-5'] = pd.read_csv(OPUS)['median'].values


def bc(x):
    """Sarle's bimodality coefficient with the sample correction."""
    n = len(x)
    m = x.mean(); s = x.std(ddof=1)
    if s == 0:
        return np.nan
    z = (x - m) / s
    g1 = n / ((n - 1) * (n - 2)) * (z ** 3).sum()
    g2 = ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3)) * (z ** 4).sum()
          - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    return (g1 ** 2 + 1) / (g2 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


rows = []
for m, v in series.items():
    rows.append(dict(model=m.split('/')[-1], n=len(v), mean=v.mean(),
                     hi=(v >= 90).mean() * 100, lo=(v <= 40).mean() * 100,
                     bc=bc(v)))
out = pd.DataFrame(rows).sort_values('mean', ascending=False).reset_index(drop=True)

pd.set_option('display.width', 200)
print('per-case judge median, shape by model (sorted by mean)\n')
print(out.to_string(index=False, float_format=lambda x: f'{x:6.2f}'))
print('\nBC > 0.556 is the conventional flag for "not a single tight hump".')

print('\nhistograms of the per-case median, top 8 by mean (bin width 10)\n')
edges = np.arange(-5, 106, 10)
for _, r in out.head(8).iterrows():
    v = series[[k for k in series if k.split('/')[-1] == r['model']][0]]
    h, _ = np.histogram(v, bins=edges)
    bar = ' '.join(f'{c:2d}' for c in h)
    print(f"{r['model'][:34]:34s} {bar}   mean {r['mean']:5.1f}")
print(f"{'':34s} " + ' '.join(f'{b:2d}' for b in range(0, 101, 10)))

out.to_csv(pathlib.Path(__file__).resolve().parents[1] / 'results/score_shape.csv', index=False)

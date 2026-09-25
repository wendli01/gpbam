"""Rank the Opus agent run against every benchmarked model, both datasets, full 100.

Also reports the corpus-ordering effect that mattered while only half of GPBam
existed: `gp_laws.csv` is sorted by citation count descending and the most-cited
norms are the longer ones, so rows 0-49 are the *harder* half. On the published
models the first-50 mean sits ~2.5 points below the full-100 mean. With all 100
articles now run this is no longer a caveat on the Opus row, but it is a real
property of the corpus and is printed so nobody re-derives it.
"""
import csv, pathlib
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parents[1]
ROOT = HERE.parents[4]
N_BOOT, SEED = 10_000, 0
AR = ROOT / 'experiments/zubaers_result/article_recitation'
csv.field_size_limit(10 ** 9)


def boot_sem(a, seed=SEED):
    rng = np.random.default_rng(seed)
    return rng.choice(a, size=(N_BOOT, a.size), replace=True).mean(axis=1).std(ddof=0)


order = {}
for dataset, name in [('GPBam Laws', 'gp_laws.csv'), ('Most cited Laws', 'most_cited_laws.csv')]:
    laws = pd.read_csv(ROOT / 'experiments/data' / name)
    order[dataset] = {f'{r.law_book} {r.article}': i for i, r in enumerate(laws.itertuples())}

pub = [pd.read_csv(AR / 'article_recitation.csv', engine='python')]
for f in ('recitation_0731.csv', 'recitation_gemma4-31b.csv', 'recitation_phi4-mini.csv'):
    if (AR / f).exists():
        pub.append(pd.read_csv(AR / f, engine='python'))
pub = pd.concat(pub, ignore_index=True)
pub = pd.concat([pub, pd.read_csv(HERE / 'recitation_opus5_agent.csv', engine='python')],
                ignore_index=True)
pub['i'] = [order.get(d, {}).get(q, np.nan) for d, q in zip(pub.dataset, pub['query'])]
pub = pub.dropna(subset=['i'])

tables = {}
for dataset in order:
    rows = []
    for m, sub in pub[pub.dataset == dataset].groupby('model'):
        sub = sub.drop_duplicates('i')
        if len(sub) < 100:
            continue
        s = sub.score.to_numpy() * 100
        rows.append({'model': m.split('/')[-1], 'score': s.mean(), 'sem': boot_sem(s),
                     'std': s.std(ddof=1), 'ge90': int((s >= 90).sum())})
    tables[dataset] = (pd.DataFrame(rows).sort_values('score', ascending=False)
                       .reset_index(drop=True))

pd.set_option('display.width', 200)
for dataset, t in tables.items():
    t.insert(0, 'rank', t.index + 1)
    print(f'\n=== {dataset}, all 100 articles ===')
    print(t.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

merged = tables['GPBam Laws'][['model', 'score']].merge(
    tables['Most cited Laws'][['model', 'score']], on='model', suffixes=('_gpbam', '_cited'))
merged.to_csv(HERE / 'model_comparison.csv', index=False)
print(f"\nSpearman(GPBam, Most cited) across {len(merged)} models: "
      f"{merged.score_gpbam.corr(merged.score_cited, method='spearman'):.4f}")

g = pub[pub.dataset == 'GPBam Laws']
d = [(sub[sub.i < 50].score.mean() - sub.score.mean()) * 100
     for _, sub in g.groupby('model') if len(sub.drop_duplicates('i')) >= 100]
print(f'corpus-ordering effect, first-50 minus full-100 over {len(d)} models: '
      f'mean {np.mean(d):+.2f}, {sum(x < 0 for x in d)}/{len(d)} negative')
print(f'wrote {HERE / "model_comparison.csv"}')

"""Within-model test of the knowledge mechanism.

The paper's model-level claim -- essay quality tracks statutory knowledge
(rho=0.88 over 32 models) -- is confounded: a model that recites well also
scores high on general capability, cites more norms and writes at length.

This asks the same question inside a model, where those confounds are constant:
for a given model, does it write better essays on the cases whose governing
norms it happens to know?

K[m,i] is model m's mean Article Recitation ROUGE-L over the norms case i's
reference solution cites, restricted to the 100 GPBam-Laws provisions for which
a recitation score exists. S[m,i] is the essay score.

Two-way demeaning removes case difficulty AND model ability, so the residual
association is the interaction: does model m do relatively better on the cases
whose law model m relatively knows?
"""
import argparse, numpy as np, os, pandas as pd, glob
from scipy.stats import spearmanr

import panel

B = os.environ.get('GPBAM_EXPERIMENTS') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_NORMS = 3          # cases with fewer overlapping provisions carry no signal
SEATS = ['score_Judge (gpt-oss-120b)', 'score_Judge (qwen3.6-35b-a3b)',
         'score_Judge (DeepSeek-V4-Flash-0731)']

def load(extractor='refex'):
    """Both sides of the test read the same citation extractor.

    ``refex`` is the published pairing: the gold norms the scoring pipeline's
    extractor resolves, against the battery of provisions those citations make
    most-cited.  ``dual`` is the same construction under ``analysis/refs.py``,
    which also reads ``Art.`` citations -- a wider gold set (52 norms per case
    against 21) and a battery in which 27 of the 100 provisions differ.
    """
    if extractor == 'dual':
        # 73 published provisions + the 27 recitation_dual_delta.py generated
        recit = pd.read_csv(f'{B}/zubaers_result/article_recitation/dual/'
                            'recitation_dual_gpbam_laws.csv')
    else:
        # Models recited after the published sweep live in sibling recitation_*.csv
        # files: the 0731 re-generation, the two late FAU-served models, and the
        # three frontier runs. Same 200 prompts, same ROUGE-L scoring.
        recit = pd.concat([pd.read_csv(f'{B}/zubaers_result/article_recitation/article_recitation.csv')]
                          + [pd.read_csv(f, usecols=['model', 'query', 'dataset', 'score'])
                             for f in sorted(glob.glob(
                                 f'{B}/zubaers_result/article_recitation/recitation_*.csv'))])
    recit = recit[recit.dataset == 'GPBam Laws'].copy()
    recit = recit[recit.score.notna()]
    recit[['book', 'section']] = recit['query'].str.split(' ', n=1, expand=True)
    recit['model'] = recit.model.map(panel.key)

    gold = pd.read_csv('out/gold_norm_buckets.csv' if extractor == 'dual'
                       else 'out/gold_norm_buckets_refex.csv')
    gold = gold[['case', 'book', 'section']].astype({'section': str}).drop_duplicates()

    essay = pd.read_csv(f'{B}/zubaers_result/essay_writing/without_rag_0731/no_rag_ji2_rejudged.csv')
    essay['model'] = essay.model.map(panel.key)
    essay['score'] = essay[SEATS].median(axis=1) * 100

    # Both sides restricted to the main table's models, so this test and the
    # leaderboard cannot disagree about who is in the panel.
    keep = set(panel.roster().key)
    panel.check(set(recit.model) & set(essay.model), 'recitation x essays')
    return (recit[recit.model.isin(keep)], gold,
            essay[essay.model.isin(keep)].copy())


def build(recit, gold, essay):
    # K[m,i]: mean recitation over the case's gold norms we have a score for
    j = gold.merge(recit[['model', 'book', 'section', 'score']], on=['book', 'section'])
    K = (j.groupby(['model', 'case'])
           .agg(K=('score', 'mean'), n_norms=('score', 'size')).reset_index())
    K = K[K.n_norms >= MIN_NORMS]
    df = K.merge(essay[['model', 'index', 'score']].rename(
        columns={'index': 'case', 'score': 'S'}), on=['model', 'case'])
    return df


def demean(df, cols=('S', 'K')):
    d = df.copy()
    for c in cols:                       # two-way within transform
        d[c + '_r'] = (d[c] - d.groupby('case')[c].transform('mean')
                            - d.groupby('model')[c].transform('mean') + d[c].mean())
    return d


def boot_ci(d, n=10000, seed=0):
    rng, cases = np.random.default_rng(seed), d['case'].unique()
    out = []
    for _ in range(n):                   # cluster bootstrap on cases
        pick = rng.choice(cases, len(cases), replace=True)
        s = pd.concat([d[d.case == c] for c in pick])
        r = spearmanr(s.S_r, s.K_r)[0]
        if np.isfinite(r):
            out.append(r)
    return np.percentile(out, [2.5, 97.5])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--extractor', choices=['refex', 'dual'], default='refex',
                    help='citation extractor for BOTH the gold norms and the battery')
    a = ap.parse_args()

    df = build(*load(a.extractor))
    print(f'extractor: {a.extractor}')
    print(f'panel: {df.model.nunique()} models x {df.case.nunique()} cases '
          f'= {len(df)} model-case pairs')
    print(f'overlapping gold norms per case: median {df.n_norms.median():.0f}, '
          f'range {df.n_norms.min()}-{df.n_norms.max()}')

    naive = spearmanr(df.S, df.K)[0]
    per = df.groupby('model').apply(
        lambda g: spearmanr(g.S, g.K)[0], include_groups=False).dropna()
    d = demean(df)
    resid = spearmanr(d.S_r, d.K_r)[0]

    print(f'\npooled, no adjustment            rho = {naive:+.3f}')
    print(f'within model (mean of {len(per)} models)  rho = {per.mean():+.3f}  '
          f'median {per.median():+.3f}  [{per.min():+.2f}, {per.max():+.2f}]  '
          f'positive in {(per > 0).sum()}/{len(per)}')
    print(f'two-way demeaned (case + model)  rho = {resid:+.3f}  '
          f'95% CI {np.round(boot_ci(d, 2000), 3)}')

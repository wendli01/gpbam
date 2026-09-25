"""Does a judge favour essays written by itself, or by its own vendor?

Three of the models in the generator zoo *are* the three judges (``gpt-5-nano``,
``qwen3.6-35b-a3b``, ``DeepSeek-V4-Flash``) and ten more share a vendor with one
of them. The reported Score is a median over that panel, so any self-preference
propagates straight into the leaderboard. This script measures it from the
stored essay-scoring results -- no new LLM calls, every number comes from the
``score_Judge (...)`` columns already in the two result CSVs.

Relatedness is graded in three severities. Only two of them are estimable with
the current panel:

``self``      generator and judge are the same model (all 3 judges)
``vendor``    same vendor, different model (all 3 judges, 2-5 generators each)
``family*``   same vendor *and* same version series. Exactly one pair qualifies
              -- ``gpt-5-mini`` judged by ``gpt-5-nano`` -- so it is given its
              own dummy (keeping it out of the ``vendor`` estimate) and reported
              as a marked footnote rather than a tier of its own.

Why the naive estimator does not work
-------------------------------------
"Judge score minus the mean of the other judges" is confounded twice over:

1. The judges are on different scales. Panel means are 33.4 / 29.4 / 17.7 with
   SDs 20.6 / 17.0 / 12.1, and DeepSeek-V4-Flash scores exactly 0 on 14% of
   essays -- its scale is compressed.
2. Models related to a judge are *good* models. Mean consensus score is 33 for
   related pairs against 26 for unrelated ones.

A compressed judge therefore looks hostile to good essays. Uncorrected, this
makes DeepSeek-V4-Flash appear to penalise its own model by -5.2 points; the
corrected estimate for the same pair is +7.9. The sign flip is pure artifact,
so the correction below is load-bearing rather than cosmetic.

The corrected estimator, in four steps:

1. Quantile-normalise within judge (``q = rank(pct) * 100``), which equalises
   the three marginal distributions by construction and removes confound 1.
2. Build ``cons``, the leave-one-out mean of the *other* judges' ``q`` on the
   same essay -- a judge-independent index of essay quality.
3. Fit ``q ~ C(judge) * bs(cons, df=5) + C(tier)``. The judge-specific spline is
   that judge's calibration curve against consensus quality, so the ``C(tier)``
   coefficients are identified from deviation at *matched quality*, removing
   confound 2.
4. Coefficients read as consensus-percentile points.

Inference. Tier is constant within a generator, so generator-level clustering is
what generalises -- but there are only 1-5 related generators per judge. Pooled
rows therefore carry SEs clustered by generator plus a generator-label
permutation test as the headline p-value; per-judge rows carry SEs clustered by
case and are conditional on those particular generators.

Run from ``experiments/``::

    PYTHONPATH=analysis python analysis/judge_family_bias.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.notebook_eval.essay_evaluation import _logical_judge_scores  # noqa: E402

CONDITIONS = [
    ('no_rag', 'No RAG', 'zubaers_result/essay_writing/without_rag/ji2/no_rag_ji2_result.csv'),
    ('law_rag', 'Law RAG', 'zubaers_result/essay_writing/with_rag/ji2/with_rag_ji2_result.csv'),
]
OUT_TEX_MATRIX = '../literature/paper/judge_score_matrix.tex'
MATRIX_COND = 'no_rag'  # the law-RAG run is being redone; keep the matrix on no-RAG
OUT_CSV = 'analysis/out/judge_family_bias.csv'
OUT_CSV_MODELS = 'analysis/out/judge_family_bias_models.csv'

N_PERM = 2000
PERM_SEED = 0
SPLINE_DF = 5

# Generator id -> (vendor, version series, model identity). `model` is what makes
# two entries the same weights: the two Qwen3.6 rows are the same model served by
# two different endpoints, and the FP8 / block-quantised repackagings are the same
# model as their upstream release. `family` is the version series, in the sense of
# the user-facing model number (qwen3.5-122b and qwen3.5-397b are one family).
TAX = {
    'GaleneAI/Magistral-Small-2509-FP8-Dynamic':        ('Mistral', 'magistral', 'magistral-small-2509'),
    'Microsoft/Phi-4-mini-instruct':                    ('Microsoft', 'phi-4', 'phi-4-mini'),
    'Qwen/Qwen3.6-35B-A3B-FP8':                         ('Qwen', 'qwen3.6', 'qwen3.6-35b-a3b'),
    'qwen/qwen3.6-35b-a3b':                             ('Qwen', 'qwen3.6', 'qwen3.6-35b-a3b'),
    'RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8':  ('Mistral', 'mistral-small-3', 'mistral-small-3.2-24b'),
    'RedHatAI/gemma-4-31B-it-FP8-block':                ('Google', 'gemma-4', 'gemma-4-31b'),
    'anthropic/claude-haiku-4.5':                       ('Anthropic', 'claude-4.5', 'claude-haiku-4.5'),
    'deepseek-ai/DeepSeek-V4-Flash':                    ('DeepSeek', 'deepseek-v4', 'deepseek-v4-flash'),
    'deepseek/deepseek-chat-v3-0324':                   ('DeepSeek', 'deepseek-v3', 'deepseek-v3-0324'),
    'deepseek/deepseek-r1-0528':                        ('DeepSeek', 'deepseek-r1', 'deepseek-r1-0528'),
    'deepseek/deepseek-v3.2':                           ('DeepSeek', 'deepseek-v3', 'deepseek-v3.2'),
    'google/gemini-2.5-flash-lite':                     ('Google', 'gemini-2.5', 'gemini-2.5-flash-lite'),
    'ibm-granite/granite-4.1-3b':                       ('IBM', 'granite-4', 'granite-4.1-3b'),
    'meta-llama/llama-3.1-8b-instruct':                 ('Meta', 'llama-3', 'llama-3.1-8b'),
    'meta-llama/llama-3.3-70b-instruct':                ('Meta', 'llama-3', 'llama-3.3-70b'),
    'meta-llama/llama-4-maverick':                      ('Meta', 'llama-4', 'llama-4-maverick'),
    'mistralai/Ministral-3-14B-Reasoning-2512':         ('Mistral', 'ministral-3', 'ministral-3-14b'),
    'mistralai/Mistral-Medium-3.5-128B':                ('Mistral', 'mistral-medium-3', 'mistral-medium-3.5'),
    'mistralai/mistral-large-2512':                     ('Mistral', 'mistral-large', 'mistral-large-2512'),
    'mistralai/mistral-small-3.1-24b-instruct':         ('Mistral', 'mistral-small-3', 'mistral-small-3.1-24b'),
    'openai/gpt-4o-mini-2024-07-18':                    ('OpenAI', 'gpt-4o', 'gpt-4o-mini'),
    'openai/gpt-5-mini':                                ('OpenAI', 'gpt-5', 'gpt-5-mini'),
    'openai/gpt-5-nano':                                ('OpenAI', 'gpt-5', 'gpt-5-nano'),
    'openai/gpt-oss-120b':                              ('OpenAI', 'gpt-oss', 'gpt-oss-120b'),
    'qwen/qwen3-235b-a22b-thinking-2507':               ('Qwen', 'qwen3', 'qwen3-235b'),
    'qwen/qwen3-32b':                                   ('Qwen', 'qwen3', 'qwen3-32b'),
    'qwen/qwen3.5-122b-a10b':                           ('Qwen', 'qwen3.5', 'qwen3.5-122b'),
    'qwen/qwen3.5-397b-a17b':                           ('Qwen', 'qwen3.5', 'qwen3.5-397b'),
    'qwen3-next-80b-a3b-instruct':                      ('Qwen', 'qwen3-next', 'qwen3-next-80b'),
    'utter-project/EuroLLM-22B-Instruct-2512':          ('UTTER', 'eurollm', 'eurollm-22b'),
    'windprak/open_steuerllm':                          ('other', 'steuerllm', 'open-steuerllm'),
}

# Judge label (as it appears in `score_Judge (<label>)`, after the two Qwen
# endpoint columns are coalesced) -> the same triple.
JUDGES = {
    'gpt-5-nano':        ('OpenAI', 'gpt-5', 'gpt-5-nano'),
    'qwen3.6-35b-a3b':   ('Qwen', 'qwen3.6', 'qwen3.6-35b-a3b'),
    'DeepSeek-V4-Flash': ('DeepSeek', 'deepseek-v4', 'deepseek-v4-flash'),
}

# Column headers for the score matrix: the full judge ids set the column width
# there, and the numbers in them are only four characters wide.
JUDGE_SHORT = {
    'gpt-5-nano': 'gpt-5-nano',
    'qwen3.6-35b-a3b': 'qwen3.6-35b',
    'DeepSeek-V4-Flash': 'DeepSeek-V4',
}

TIERS = ['unrelated', 'vendor', 'family*', 'self']
REPORT_TIERS = ['self', 'vendor']
TIER_LABEL = {'self': 'Same model', 'vendor': 'Same vendor', 'family*': 'Same family'}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_long():
    """Long-format panel: one row per (condition, case, generator, judge)."""
    frames = []
    for cond, _, path in CONDITIONS:
        df = pd.read_csv(path)

        # The two Qwen judge columns are one logical judge served by two
        # endpoints; _logical_judge_scores coalesces them. Assert the premise.
        qcols = [c for c in df.columns if c.startswith('score_Judge') and 'qwen' in c.lower()]
        if len(qcols) > 1:
            both = df[qcols].notna().sum(axis=1) > 1
            assert not both.any(), f'{path}: {both.sum()} rows carry >1 Qwen judge score'

        for label, s in _logical_judge_scores(df).items():
            frames.append(pd.DataFrame({'cond': cond, 'case': df['index'],
                                        'gen': df['model'], 'judge': label,
                                        'score': s * 100}))

    long = pd.concat(frames, ignore_index=True).dropna(subset=['score'])

    unmapped = set(long['gen']) - set(TAX)
    assert not unmapped, f'generators missing from TAX: {sorted(unmapped)}'
    assert set(long['judge']) == set(JUDGES), \
        f'judge labels {sorted(set(long["judge"]))} != {sorted(JUDGES)}'

    long[['vendor', 'family', 'model']] = long['gen'].map(TAX).apply(pd.Series)
    long[['j_vendor', 'j_family', 'j_model']] = long['judge'].map(JUDGES).apply(pd.Series)
    long['tier'] = pd.Categorical(
        [tier_of(r) for r in long.itertuples()], categories=TIERS)
    long['essay'] = long['cond'] + '|' + long['case'].astype(str) + '|' + long['gen']
    return long


def tier_of(r):
    if r.model == r.j_model:
        return 'self'
    if r.vendor == r.j_vendor and r.family == r.j_family:
        return 'family*'
    if r.vendor == r.j_vendor:
        return 'vendor'
    return 'unrelated'


def add_normalised(long):
    """Per-judge quantile scale `q`, and the leave-one-out consensus `cons`."""
    long = long.copy()
    long['q'] = long.groupby('judge')['score'].rank(pct=True) * 100
    g = long.groupby('essay')['q']
    n, total = g.transform('count'), g.transform('sum')
    # An essay judged by a single judge carries no consensus; there are none in
    # practice (min panel size is 2), but guard rather than divide by zero.
    long['cons'] = np.where(n > 1, (total - long['q']) / (n - 1), np.nan)
    return long.dropna(subset=['cons'])


# --------------------------------------------------------------------------- #
# Estimation
# --------------------------------------------------------------------------- #
def fit_tiers(df, cluster, by_judge=False):
    """Tier coefficients from the quality-controlled model.

    Returns {tier: (coef, se, p)}. With `by_judge` the judge interaction is
    dropped (the frame already holds a single judge).
    """
    sub = df.copy()
    sub['tier'] = sub['tier'].cat.remove_unused_categories()
    rhs = f'bs(cons, df={SPLINE_DF})' if by_judge else f'C(judge)*bs(cons, df={SPLINE_DF})'
    m = smf.ols(f'q ~ {rhs} + C(tier)', data=sub).fit(
        cov_type='cluster', cov_kwds={'groups': sub[cluster]})
    out = {}
    for k in m.params.index:
        if k.startswith('C(tier)[T.'):
            out[k[len('C(tier)[T.'):-1]] = (m.params[k], m.bse[k], m.pvalues[k])
    return out


def permutation_p(df, n_perm=N_PERM, seed=PERM_SEED):
    """Generator-label permutation test for the pooled tier effects.

    The null is "relatedness is assigned to generators at random". Each draw
    permutes which generator carries which taxonomy entry and recomputes the
    tier contrast on residuals from the quality model *without* tier terms --
    refitting the spline 2000 times would be needlessly slow and changes
    nothing, since the residualisation does not depend on the tier labels.

    Returns {tier: (observed_contrast, p)}. The observed contrast is computed
    with the same residual statistic as the null, so the p-value is coherent;
    the point estimates in the table come from `fit_tiers`.
    """
    base = smf.ols(f'q ~ C(judge)*bs(cons, df={SPLINE_DF})', data=df).fit()
    resid = (df['q'] - base.predict(df)).to_numpy()

    vendor = df['gen'].map({k: v[0] for k, v in TAX.items()}).to_numpy()
    family = df['gen'].map({k: v[1] for k, v in TAX.items()}).to_numpy()
    model = df['gen'].map({k: v[2] for k, v in TAX.items()}).to_numpy()
    j_vendor, j_family, j_model = (df[c].to_numpy() for c in ('j_vendor', 'j_family', 'j_model'))

    def contrast(v, f, m):
        t = np.where(m == j_model, 'self',
                     np.where((v == j_vendor) & (f == j_family), 'family*',
                              np.where(v == j_vendor, 'vendor', 'unrelated')))
        ref = resid[t == 'unrelated'].mean()
        return {k: resid[t == k].mean() - ref for k in REPORT_TIERS + ['family*']
                if (t == k).any()}

    obs = contrast(vendor, family, model)
    rng = np.random.default_rng(seed)
    gens = np.array(sorted(TAX))
    lut = {g: i for i, g in enumerate(gens)}
    idx = df['gen'].map(lut).to_numpy()

    hits = {k: 0 for k in obs}
    draws = {k: 0 for k in obs}
    v_all = np.array([TAX[g][0] for g in gens])
    f_all = np.array([TAX[g][1] for g in gens])
    m_all = np.array([TAX[g][2] for g in gens])
    for _ in range(n_perm):
        p = rng.permutation(len(gens))
        null = contrast(v_all[p][idx], f_all[p][idx], m_all[p][idx])
        for k in obs:
            if k in null:
                draws[k] += 1
                hits[k] += abs(null[k]) >= abs(obs[k])
    return {k: (obs[k], (hits[k] + 1) / (draws[k] + 1), draws[k]) for k in obs}


# --------------------------------------------------------------------------- #
# Debiased leaderboard
# --------------------------------------------------------------------------- #
def debias(long):
    """Add `score_adj`: the score each judge would have given without the tier effect.

    The correction is fitted in quantile space and mapped back through the
    judge's own empirical quantile function *as a difference*::

        score_adj = score + Q_j(q - gamma) - Q_j(q)

    Mapping back absolutely instead (`score_adj = Q_j(q - gamma)`) leaves a
    ~0.12-point round-trip error, because `rank(pct=True)` with ties is not the
    exact inverse of the empirical quantile function. That error would land on
    every row including the unrelated ones. The differential form is exact at
    gamma = 0, which the assertion below pins down.
    """
    m = smf.ols(f'q ~ C(judge)*bs(cons, df={SPLINE_DF}) + C(tier)', data=long).fit()
    gamma = {t: m.params.get(f'C(tier)[T.{t}]', 0.0) for t in TIERS}

    out = long.copy()
    out['q_adj'] = (out['q'] - out['tier'].astype(str).map(gamma)).clip(0.01, 100)
    out['score_adj'] = np.nan
    for _, sub in out.groupby('judge'):
        emp = np.sort(sub['score'].to_numpy())
        Q = lambda x: np.quantile(emp, np.asarray(x) / 100, method='linear')  # noqa: E731
        out.loc[sub.index, 'score_adj'] = (
            sub['score'].to_numpy() + Q(sub['q_adj'].to_numpy()) - Q(sub['q'].to_numpy()))

    unrel = out['tier'] == 'unrelated'
    assert np.allclose(out.loc[unrel, 'score_adj'], out.loc[unrel, 'score'], atol=1e-9), \
        'debiasing moved unrelated rows -- the quantile round-trip is not differential'
    return out, gamma


def leaderboard(long):
    """Reported vs debiased Score per model, per condition.

    Score is recomputed exactly as `make_summary` does it: per-essay median over
    the judge panel (x100 already applied), then mean over the model's cases.
    """
    rows = []
    for cond, cond_label, _ in CONDITIONS:
        sub = long[long['cond'] == cond]
        rep = sub.groupby(['case', 'gen'])['score'].median().groupby('gen').mean()
        adj = sub.groupby(['case', 'gen'])['score_adj'].median().groupby('gen').mean()
        rel = (sub[sub['tier'] != 'unrelated']
               .groupby('gen')['tier'].agg(lambda s: '+'.join(sorted(set(s.astype(str))))))
        t = pd.DataFrame({'cond': cond, 'cond_label': cond_label,
                          'score': rep, 'score_debiased': adj})
        t['delta'] = t['score_debiased'] - t['score']
        t['rank'] = t['score'].rank(ascending=False, method='min').astype(int)
        t['rank_debiased'] = t['score_debiased'].rank(ascending=False, method='min').astype(int)
        t['relation'] = rel
        rows.append(t.reset_index())
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #
# Sequential ramps for the cell shading, low -> high. Magnitude is a sequential
# job, so every option is monotonic in lightness rather than a rainbow, and each
# degrades to a readable grey ramp in black-and-white print. Every shaded cell
# also prints its number, so colour is redundant encoding, never the only channel.
#
# The ramps run their FULL luminance range and the body text flips to white once
# black would fall below 4.5:1 (see `tint`). Stopping the ramp early to keep the
# text black is what makes a heatmap look washed out -- it throws away most of
# the dynamic range to avoid a problem that one \textcolor solves.
CMAPS = {
    # single hue, white -> dark blue. The most conservative, and the only one
    # whose lightest steps are genuinely paper-white.
    'blues': ['FFFFFF', 'CDE2FB', 'B7D3F6', '9EC5F4', '86B6EF', '6DA7EC',
              '5598E7', '3987E5', '256ABF', '184F95', '0D366B'],
    # multi-hue sequential, pale yellow -> deep blue. Much wider perceptual range
    # than one hue can give, still monotonic in lightness and CVD-safe.
    'ylgnbu': ['FFFFD9', 'EDF8B1', 'C6E9B4', '7ECDBB', '40B5C4', '1D90C0',
               '225DA8', '243392', '081D58'],
    # viridis, reversed so that low values are the light end. Perceptually uniform
    # and the widest contrast of the three, but rejected as the default: its light
    # end is a saturated yellow, so on paper the weakest models shout the loudest.
    'viridis': ['FDE725', 'ADDC30', '5EC962', '28AE80', '21918C', '2C728E',
                '3B528B', '472D7B', '440154'],
}
CMAP = 'ylgnbu'


def _luminance(rgb):
    c = [x / 255 for x in rgb]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def tint(value, vmax, body, cmap=None):
    """`\\cellcolor` for `value` on a 0..vmax ramp, wrapped around `body`.

    Flips the text to white where black would drop below 4.5:1 on the fill, which
    is what lets the ramp use its whole range instead of stopping at the last
    step that black text survives.
    """
    ramp = CMAPS[cmap or CMAP]
    if value is None or not np.isfinite(value) or vmax <= 0:
        return body
    pos = min(abs(value) / vmax, 1.0) * (len(ramp) - 1)
    lo, frac = int(pos), pos - int(pos)
    hi = min(lo + 1, len(ramp) - 1)
    rgb = [round(int(ramp[lo][i:i + 2], 16) * (1 - frac) + int(ramp[hi][i:i + 2], 16) * frac)
           for i in (0, 2, 4)]
    fill = r'\cellcolor[HTML]{' + ''.join(f'{c:02X}' for c in rgb) + '}'
    if (_luminance(rgb) + 0.05) / 0.05 < 4.5:
        body = r'\textcolor{white}{' + body + '}'
    return fill + body


def esc(s):
    return s.replace('_', r'\_')


def texttt(s):
    return r'\texttt{' + esc(s) + '}'


def taxonomy_comments(long):
    """The tier assignment, as %-comments, so the table is auditable from source."""
    lines = ['%', '% Tier assignment (generator -> judge), from TAX/JUDGES in',
             '% experiments/analysis/judge_family_bias.py:']
    rel = long[long['tier'] != 'unrelated'][['judge', 'tier', 'gen']].drop_duplicates()
    for judge in JUDGES:
        lines.append(f'%   judge {judge}:')
        for t in ['self', 'family*', 'vendor']:
            gens = sorted(rel[(rel['judge'] == judge) & (rel['tier'] == t)]['gen'])
            if gens:
                lines.append(f'%     {t:<8s} {", ".join(gens)}')
    related = set(long[long['tier'] != 'unrelated']['gen'])
    lines += [f'%   generators unrelated to every judge: '
              f'{long["gen"].nunique() - len(related)}', '%']
    return lines


def n_models(long, tier, judge=None):
    """Distinct model *identities* at a tier -- not endpoint ids.

    The two Qwen3.6 entries are one model served by two endpoints (one per
    condition), so counting `gen` would report the Qwen judge as having two
    same-model generators.
    """
    sel = long['tier'] == tier
    if judge is not None:
        sel &= long['judge'] == judge
    return long.loc[sel, 'model'].nunique()


def build_matrix(long, gamma=None, cond='no_rag'):
    """Full score matrix: every model x every judge, with related cells marked.

    Fill encodes the printed score on one shared scale. Relatedness cannot also
    be a fill -- one cell, one fill -- so it goes on a separate, non-colour
    channel: the number goes bold and picks up a superscript symbol keyed to the
    tier. That also keeps the marking alive in greyscale print and for readers
    who cannot separate the hues.
    """
    sub = long[long['cond'] == cond]
    judges = list(JUDGES)
    per_judge = (sub.groupby(['gen', 'judge'])['score'].mean().unstack()[judges])
    score = sub.groupby(['case', 'gen'])['score'].median().groupby('gen').mean()
    debiased = sub.groupby(['case', 'gen'])['score_adj'].median().groupby('gen').mean()
    tiers = (sub[sub['tier'] != 'unrelated']
             .groupby(['gen', 'judge'])['tier'].agg(lambda s: s.astype(str).iloc[0]))
    related = set(sub.loc[sub['tier'] != 'unrelated', 'gen'])

    table = (per_judge.assign(Score=score, Debiased=debiased)
             .sort_values('Score', ascending=False))
    vmax = table.max().max()
    cond_label = dict((c, l) for c, l, _ in CONDITIONS)[cond]

    lines = [
        r'% Generated by experiments/analysis/judge_family_bias.py -- do not hand-edit.',
        r'% Needs \usepackage[table]{xcolor} (or colortbl) for \cellcolor.',
        # This is the only table, so it carries the tier assignment behind the
        # M/F/V marks -- otherwise nothing in the paper source records it.
        *taxonomy_comments(sub),
        # ...and the coefficients the Debiased column was built from.
        '%', '% Tier effects subtracted for the Debiased column '
             '(consensus-percentile points):',
        *[f'%   {t:<8s} {g:+.2f}' for t, g in (gamma or {}).items() if g],
        '%',
        r'\begin{table}[htbp]',
        r'\centering',
        # One sentence per line, so a hand-edit to the caption produces a readable
        # diff rather than one enormous changed line.
        r'\caption{\textbf{The full judge-by-model score matrix, with every judgement a model '
        r'received from a related judge marked.}',
        r'Each cell is that judge\textquotesingle s mean over the '
        rf'{sub["case"].nunique()} cases, in percent, for the {cond_label} condition.',
        r'\emph{Score} is the reported metric, the per-essay median over the panel; '
        r'\emph{Debiased} is the same median after each related judge\textquotesingle s tier '
        r'effect is subtracted and mapped back onto its own scale, and $\Delta$ is the gap '
        r'between them.',
        r'$\Delta$ is blank where no judge was related to the model at all, which is a different '
        r'statement from the $0.0$ of a model that was corrected but whose median did not move.',
        r'Both ensemble score columns are shaded on the same scale as the judge columns, so the '
        r'correction is visible as a step in colour;',
        r'$\Delta$ is left unshaded rather than given a second scale of its own.',
        r'\emph{Bias} gives the closest relation the model has to any judge on the panel, '
        r'most severe first where there is more than one: '
        r'$\mathrm{M}$ same model, $\mathrm{F}$ same family (same vendor and version series), '
        r'$\mathrm{V}$ same vendor; blank means related to none of them.',
        r'The bold cell with a matching superscript in the judge columns is the judgement '
        r'that relation concerns, so the column says \emph{whether} and the superscript says '
        r'\emph{where}.',
        r'}',
        rf'\label{{tab:judge_score_matrix_{cond}}}',
        r'\scriptsize',
        r'\setlength{\tabcolsep}{4pt}',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{lcrrr@{\hspace{2em}}' + 'r' * len(judges) + '}',
        r'\toprule',
        (r'& & \multicolumn{3}{c}{Ensemble} & '
         rf'\multicolumn{{{len(judges)}}}{{c}}{{Individual judge}} \\'),
        rf'\cmidrule(lr){{3-5}}\cmidrule(lr){{6-{5 + len(judges)}}}',
        ' & '.join(['Model', 'Bias', 'Score', 'Debiased', r'$\Delta$']
                   + [texttt(JUDGE_SHORT[j]) for j in judges]) + r' \\',
        r'\midrule',
    ]
    # Letters, not shapes: at \scriptsize a \diamond and a \circ are the same blob,
    # and these three levels have to stay apart.
    sym = {'self': r'\mathrm{M}', 'family*': r'\mathrm{F}', 'vendor': r'\mathrm{V}'}

    def relation(gen):
        """The model's relatedness letters, most severe first.

        A model can be related to more than one judge -- and to different degrees --
        so this is a set, not a single letter. Ordered by severity rather than
        alphabetically so the eye lands on the worst case first.
        """
        ts = {t for (g, _), t in tiers.items() if g == gen}
        return '/'.join(f'${sym[t]}$' for t in TIERS[::-1] if t in ts)
    for gen, row in table.iterrows():
        d = row['Debiased'] - row['Score']
        # Blank, not 0.0, where no judge was related: that is "nothing to correct",
        # which is a different statement from "corrected, and it did not move the
        # median" -- several related models land on exactly 0.0.
        delta = '' if gen not in related else (r'$0.0$' if abs(d) < 0.05 else f'${d:+.1f}$')
        cells = [texttt(gen.split('/')[-1]),
                 relation(gen),
                 tint(row['Score'], vmax, f"{row['Score']:.1f}"),
                 tint(row['Debiased'], vmax, f"{row['Debiased']:.1f}"),
                 delta]
        for j in judges:
            t = tiers.get((gen, j))
            body = f'{row[j]:.1f}' if pd.isna(t) else rf'\textbf{{{row[j]:.1f}}}$^{{{sym[t]}}}$'
            cells.append(tint(row[j], vmax, body))
        lines.append(' & '.join(cells) + r' \\')
    lines += [
        r'\midrule',
        ' & '.join([r'\emph{Judge mean}', '', '', '', '']
                   + [rf'\emph{{{table[j].mean():.1f}}}' for j in judges]) + r' \\',
        r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    return lines


# --------------------------------------------------------------------------- #
def main(out_tex_matrix=OUT_TEX_MATRIX,
         out_csv=OUT_CSV, out_csv_models=OUT_CSV_MODELS, n_perm=N_PERM):
    long = add_normalised(load_long())
    print(f'{len(long)} judgements | {long["gen"].nunique()} generators | '
          f'{long["judge"].nunique()} judges | {long["case"].nunique()} cases')
    print(long.groupby(['tier', 'judge'], observed=True)['gen'].nunique()
          .unstack().fillna(0).astype(int).to_string(), '\n')

    pooled = {'both': fit_tiers(long, cluster='gen')}
    per_judge = {}
    for cond, _, _ in CONDITIONS:
        sub = long[long['cond'] == cond]
        pooled[cond] = fit_tiers(sub, cluster='gen')
        per_judge[cond] = {j: fit_tiers(sub[sub['judge'] == j], cluster='case', by_judge=True)
                           for j in JUDGES}

    perm = permutation_p(long, n_perm=n_perm)

    print('pooled tier effects (consensus-percentile points, SE clustered by generator)')
    for t in REPORT_TIERS + ['family*']:
        row = [f'{t:9s}']
        for k in ['both'] + [c for c, _, _ in CONDITIONS]:
            e = pooled[k].get(t)
            row.append(f'{k}={e[0]:+6.2f}({e[1]:.2f})' if e else f'{k}=--')
        o, p, n = perm.get(t, (np.nan, np.nan, 0))
        print('  ' + '  '.join(row) + f'  perm p={p:.4f} (n={n})')

    print('\nper-judge (SE clustered by case)')
    for j in JUDGES:
        for cond, label, _ in CONDITIONS:
            parts = [f'{t}={per_judge[cond][j][t][0]:+6.2f}({per_judge[cond][j][t][1]:.2f})'
                     for t in REPORT_TIERS if t in per_judge[cond][j]]
            print(f'  {j:20s} {label:8s} ' + '  '.join(parts))

    selfs = [per_judge[c][j].get('self') for c, _, _ in CONDITIONS for j in JUDGES]
    assert all(e and e[0] > 0 for e in selfs), \
        'a judge shows negative self-preference -- check the scale correction'

    long, gamma = debias(long)
    board = leaderboard(long)
    print('\nfitted tier effects used for debiasing:',
          {k: round(v, 2) for k, v in gamma.items() if v})

    # --- write ------------------------------------------------------------ #
    scopes = [(None, 'pooled', pooled)]
    scopes += [(j, f'judge:{j}', {c: per_judge[c][j] for c, _, _ in CONDITIONS}) for j in JUDGES]
    tidy = []
    for judge, scope, table in scopes:
        for cond, est in table.items():
            for t, (coef, se, p) in est.items():
                tidy.append({'scope': scope, 'condition': cond, 'tier': t,
                             'coef': coef, 'se': se, 'p': p,
                             'cluster': 'generator' if judge is None else 'case',
                             'perm_p': perm[t][1] if judge is None and t in perm else np.nan,
                             'n_models': n_models(long, t, judge)})
    for path, frame in [(out_csv, pd.DataFrame(tidy)), (out_csv_models, board)]:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        frame.to_csv(path, index=False)
        print(f'written {path}')

    lines = build_matrix(long, gamma, MATRIX_COND)
    os.makedirs(os.path.dirname(os.path.abspath(out_tex_matrix)), exist_ok=True)
    with open(out_tex_matrix, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'written {out_tex_matrix}\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-matrix', default=OUT_TEX_MATRIX)
    ap.add_argument('--csv', default=OUT_CSV)
    ap.add_argument('--csv-models', default=OUT_CSV_MODELS)
    ap.add_argument('--n-perm', type=int, default=N_PERM)
    ap.add_argument('--cmap', default=CMAP, choices=sorted(CMAPS),
                    help='sequential ramp for the cell shading')
    a = ap.parse_args()
    CMAP = a.cmap
    main(a.out_matrix, a.csv, a.csv_models, a.n_perm)

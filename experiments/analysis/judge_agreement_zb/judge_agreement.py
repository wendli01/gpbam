"""Inter-judge agreement for the panel that Table 5 actually reports.

The paper carried this analysis twice already -- ``Table 15`` in long form and
``Table 16`` as its compact restatement -- and both described a panel the study
no longer uses: gpt-5-nano in place of gpt-oss-120b, 26 generation models rather
than 32, ~2100 essays rather than ~2590. A validation of a panel that did not
produce the scores is worse than no validation, so both are replaced by the one
table this script writes.

The row set is not chosen here. It is read off ``main_essay_table.csv``, which is
the same parent ``merged_results_table.py`` reads to build Table 5, so the two
tables cannot come to describe different experiments. Three assertions hold that
claim to something stronger than a shared filename: the model set must match
exactly, the per-judge means rebuilt from the raw judgements must reproduce that
file's ``gpt_oss``/``qwen36``/``dsv4`` columns, and the per-case medians must
reproduce its ``essay`` column. The last is the one that matters -- means agree
whenever the marginals do, but the median column only reproduces if every
``(model, index)`` pair lines up, which is what the essay level is built on.

Two levels, because they answer different questions
---------------------------------------------------
At the *model* level one observation is a generation model and a judge's value is
its mean over that model's essays. This is the level the leaderboard lives at, so
it is the level at which agreement has to hold for Table 5 to mean anything.

At the *essay* level one observation is a single essay. This is the harder test
and the lower number: judges that rank 32 models identically can still disagree
about which of two essays by the same model is better.

Three statistics, because they fail differently
------------------------------------------------
Spearman's rho is rank agreement, and it is invariant to a judge's overall level.
That invariance is the problem. This panel's judges sit at visibly different
heights -- mean score 28.4 (gpt-oss), 33.4 (Qwen3.6), 24.8 (DS-V4) over the 32
rows -- and a reader who takes a judge's number at face value is affected by
exactly the offset rho discards.

Pearson's r was reported here at one point and has been dropped. On this data it
tracked rho closely enough to add a column without adding a question: it answers
the same "do they move together" that rho answers, less robustly on a bounded
11-level scale, and it is level-invariant in the same way. Two statistics that
fail differently are worth three columns; three statistics where two fail the
same way are not.

Quadratic-weighted kappa is the statistic that does not discard it:

    kappa_w = 1 - sum(W * O) / sum(W * E),   W_ij = (i - j)^2 / (K - 1)^2

with ``E`` the outer product of the two judges' *own* marginals -- Cohen's chance
model, not a pooled one. A judge sitting nine points low keeps its r and loses
kappa_w. The sharpest case in this run is Qwen3.6 x DS-V4, which has the highest
essay-level r of the three pairs and the lowest kappa_w.

At the essay level the scores are already ordinal: the judges emit a 0-1 grid in
0.1 steps, so x100 gives K = 11 fixed levels, 0..100. The level set is pinned to
all eleven rather than to the ones a pair happens to use, or the index spacing --
and so the weights -- would mean something different in each pair. 15 of 7775
judgements land off that grid (0.35, 0.37, 0.45, 0.55, 0.75, 0.85, all parser
strays); they are snapped to the nearest step. ``--self-test`` prints the
count, so a future run that starts snapping in bulk says so.

At the model level the values are means, not grid points, so there is nothing to
tabulate. Rather than bin them to a grid built for individual essays -- which is
lossy here, DS-V4's 32 means occupy 6 of the 11 bins with 12 models in one --
kappa_w is taken in its bin-width-to-zero limit, which is the same formula with
the sums replaced by their expectations:

    kappa_w = 1 - E[(a - b)^2] / (E[a^2] - 2 E[a] E[b] + E[b^2])

the denominator being E[(a_i - b_j)^2] under independent pairing. It is the same
statistic: where both forms are available, at the essay level, they agree to
within 0.001. The ordinal form is checked against
``sklearn.metrics.cohen_kappa_score(weights='quadratic')`` in ``--self-test``.

Intervals
---------
Percentile bootstrap, 10,000 resamples, fixed seed.

The essay-level bootstrap resamples whole generation models, not essays. Essays
are clustered -- 81 per model, sharing a generator -- so resampling them
independently treats 2592 observations as if they were 2592 independent draws.
Measured on the three kappa_w intervals, the naive version comes out 3.6x, 5.7x
and 5.8x narrower than the clustered one; on Qwen3.6 x DS-V4 that is
[0.727, 0.767] pretending against [0.587, 0.822] honest. The point estimates are
identical either way; only the intervals move.

Model-level p-values are analytic for rho, and permutation for kappa_w, which has
no closed form. All six come back < 0.001. Essay-level p-values are
omitted rather than computed: the same clustering that widens the intervals
invalidates the null those tests assume, and a permutation over clusters would be
answering a question nobody asked (whether judges agree *at all*) at a level where
the answer is not in doubt.

Usage
-----
    python experiments/analysis/judge_agreement_zb/judge_agreement.py
    python experiments/analysis/judge_agreement_zb/judge_agreement.py --self-test
    python experiments/analysis/judge_agreement_zb/judge_agreement.py --boot 1000

Writes ``judge_agreement_zb.{tex,csv,pdf,png}`` beside itself, plus
``judge_agreement_zb_figure_caption.txt``; the table's caption is emitted inside
the ``.tex``. Both captions are generated rather than stored, so neither can drift
from the numbers it describes. Its inputs stay in
``experiments/analysis/deepseek_rerun/``, which owns the re-judged panel; this
directory owns the agreement analysis and nothing else.
"""
import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
# Inputs belong to the re-judged panel and stay there. This directory owns its
# outputs and nothing else -- a local copy of main_essay_table.csv or the seat
# files is exactly the drift that left the old Tables 15/16 describing a panel
# the study had stopped using.
PANEL = ROOT / 'experiments/analysis/deepseek_rerun'
RESULTS = PANEL / 'results'
OUT = Path(__file__).resolve().parent

# Deliberate cross-directory import: load_seat owns the seat-file format and its
# sort order, so a second copy here would be free to drift from the one every
# other consumer uses.
sys.path.insert(0, str(PANEL / 'scripts'))
from seats import load_seat  # noqa: E402
G = ROOT / 'experiments/zubaers_result/essay_writing/without_rag_0731'
F = ROOT / 'experiments/zubaers_result/essay_writing/frontier'
O = ROOT / 'experiments/analysis/additional_models/opus5_agent'

# (seat file stem, short key, display name). The order is the panel's order in
# Table 5, so the pairs below come out in the same order a reader met them there.
JUDGES = [('gpt-oss-120b', 'oss', 'GPT-oss-120b'),
          ('Qwen3.6-35B', 'qwen', 'Qwen3.6-35B'),
          ('DeepSeek-V4-Flash-0731', 'ds', 'DS-V4-Flash')]
KEYS = [k for _, k, _ in JUDGES]
PAIRS = [('oss', 'qwen'), ('oss', 'ds'), ('qwen', 'ds')]
NAME = {k: n for _, k, n in JUDGES}

# Generations judged on the same three seats but outside the published sweep, so
# their essays live beside their own run rather than in the seat files. Same list
# as rebuild_tables.py's EXTRA_RUNS -- if that grows, this grows, and the row-set
# assertion below is what says so out loud rather than silently dropping a model.
EXTRA_RUNS = [('deepseek-ai/DeepSeek-V4-Flash-0731', G / 'judged'),
              ('google/gemini-3.7-flash', F / 'judged_gemini-3.7-flash'),
              ('soofi-s-isar-preview', F / 'judged_soofi-s-isar-preview'),
              ('anthropic/claude-opus-5', O / 'judged')]

K = 11              # 0, 10, ..., 100
N_BOOT, SEED = 10_000, 0


# ------------------------------------------------------------------ statistics
def levels(s):
    """0-100 score to its index on the judges' own 11-point grid.

    np.rint breaks ties to even, so the handful of scores landing exactly between
    two steps (0.35 -> 40, 0.45 -> 40) round in different directions. The
    alternative tie-break moves the essay-level kappas by at most 0.0006, well
    inside the two decimals reported, so which one is used does not matter -- but
    it is a choice rather than an inevitability, and it is made here.
    """
    return np.clip(np.rint(np.asarray(s, float) / 10.0), 0, K - 1).astype(int)


def qwk_ordinal(a, b):
    """Cohen's quadratic-weighted kappa on the fixed 11-level grid."""
    a, b = levels(a), levels(b)
    obs = np.zeros((K, K))
    np.add.at(obs, (a, b), 1)
    exp = np.outer(np.bincount(a, minlength=K), np.bincount(b, minlength=K)) / len(a)
    i = np.arange(K)
    w = (i[:, None] - i[None, :]) ** 2 / (K - 1) ** 2
    return 1 - (w * obs).sum() / (w * exp).sum()


def qwk_continuous(a, b):
    """The same kappa with the bin width taken to zero, for unbinnable means."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    indep = (a ** 2).mean() - 2 * a.mean() * b.mean() + (b ** 2).mean()
    return 1 - ((a - b) ** 2).mean() / indep


def rho(a, b):
    return spearmanr(a, b).statistic


def ci_iid(f, a, b, n_boot, seed=SEED):
    """Percentile CI resampling observations -- correct where one row is one unit."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    v = [f(a[i], b[i]) for i in rng.integers(0, len(a), (n_boot, len(a)))]
    return tuple(np.percentile(v, [2.5, 97.5]))


def ci_cluster(f, d, ka, kb, n_boot, seed=SEED):
    """Percentile CI resampling whole generation models.

    Essays are not exchangeable: 81 of them share a generator, and an essay-wise
    resample would report an interval the design does not support.
    """
    rng = np.random.default_rng(seed)
    groups = [g[[ka, kb]].dropna().to_numpy() for _, g in d.groupby('model', sort=True)]
    v = []
    for row in rng.integers(0, len(groups), (n_boot, len(groups))):
        s = np.vstack([groups[i] for i in row])
        v.append(f(s[:, 0], s[:, 1]))
    return tuple(np.percentile(v, [2.5, 97.5]))


def perm_p(f, a, b, n_boot, seed=SEED):
    """Two-sample permutation p for a statistic with no analytic null."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    obs = f(a, b)
    ge = sum(f(a, rng.permutation(b)) >= obs for _ in range(n_boot))
    return (ge + 1) / (n_boot + 1)


# ------------------------------------------------------------------------ data
def _seat_sweep(results):
    d = None
    for stem, key, _ in JUDGES:
        x = load_seat(stem, ['index', 'model', 'score'], results=results) \
            .rename(columns={'score': key})
        d = x if d is None else d.merge(x, on=['index', 'model'])
    return d


def _extra_run(model_id, judged_dir):
    cols = {}
    for stem, key, _ in JUDGES:
        j = pd.DataFrame([json.loads(l) for l in open(judged_dir / f'{stem}.jsonl')])
        cols[key] = j.drop_duplicates('index').set_index('index').score
    return pd.DataFrame(cols).reset_index().assign(model=model_id)


def load(results):
    """Per-essay judge scores for exactly the rows Table 5 reports.

    Returns ``(essays, model_means, reference_table)``. Raises rather than
    quietly disagreeing with the table it is supposed to validate.
    """
    ref = pd.read_csv(results / 'main_essay_table.csv', index_col='model')
    ess = pd.concat([_seat_sweep(results)]
                    + [_extra_run(m, p) for m, p in EXTRA_RUNS], ignore_index=True)
    ess = ess[ess.model.isin(ref.index)].reset_index(drop=True)
    for k in KEYS:
        ess[k] = ess[k] * 100

    missing = set(ref.index) - set(ess.model)
    assert not missing, (
        f'{len(missing)} models are in main_essay_table.csv but have no per-essay '
        f'judgements here: {sorted(missing)}. A generation judged outside the '
        f'published sweep needs its judged/ directory listed in EXTRA_RUNS.')

    mdl = ess.groupby('model')[KEYS].mean().reindex(ref.index)
    d_mean = (mdl - ref[['gpt_oss', 'qwen36', 'dsv4']].set_axis(KEYS, axis=1)).abs().max().max()
    d_med = (ess.assign(p=ess[KEYS].median(axis=1)).groupby('model').p.mean()
             .reindex(ref.index) - ref.essay).abs().max()
    assert d_mean < 1e-9 and d_med < 1e-9, (
        f'rebuilt judgements do not reproduce main_essay_table.csv (per-judge means '
        f'off by {d_mean:.2e}, per-case medians by {d_med:.2e}) -- one of the two is '
        f'stale, rebuild before reporting agreement on it')
    return ess, mdl, ref


# ------------------------------------------------------------------- reporting

# The figure's caption is generated rather than kept in a static file, for the same
# reason the table's is: it quotes kappas and offsets, and a caption that can drift
# from the figure it sits under is worse than no caption. Plain text, because it is
# pasted into whatever float the document uses.
FIG_CAPTION = """Where the three judges disagree, GPBam essay writing, no
retrieval. One column per judge pair, over the %(nmod)d generation models of the
main results table.

Top row, essay level: the joint distribution of the two judges' scores on their
own fixed %(K)d-point grid -- the matrix from which quadratic-weighted kappa is
computed, so the off-diagonal mass the statistic charges for is visible directly.
Colour is the percentage of essays in a cell, on a power scale so the sparse
high-scoring corner stays legible next to the mode. The dashed line is exact
agreement. n = %(nmin)s-%(nmax)s essays per pair (pairwise-complete);
kappa_w = %(ke1).2f, %(ke2).2f, %(ke3).2f left to right.

Bottom row, model level: one point per generation model, each judge's mean over
that model's essays, against the same identity line. rho = %(rm1).2f, %(rm2).2f,
%(rm3).2f and kappa_w = %(km1).2f, %(km2).2f, %(km3).2f.

The two rows show one fact at two resolutions. The off-diagonal mass above is not
symmetric about the identity line, and by the row below it has resolved into a
clean vertical offset -- %(o1)+.1f, %(o2)+.1f and %(o3)+.1f points of mean
difference. This is what separates rho from kappa_w: the judges order essays
alike while scoring them at different heights, and only kappa_w is sensitive to
the second half of that sentence. The rightmost pair is the clearest case, with
every model above the line."""


# The caption travels with the table, so it states the design rather than pointing
# at prose the artefact does not control. Written to survive being copied into the
# paper directory on its own.
CAPTION = r"""Inter-judge agreement in the GPBam essay-writing experiment (no
retrieval). Agreement is evaluated pairwise for the three LLM judges that produce
the reported score. At the \emph{model} level, one observation is a generation
model and each judge's value is its mean score over that model's essays; at the
\emph{essay} level, one observation is an individual essay for which both judges
supplied a score, and $n$ is pairwise-complete. Spearman's $\rho$ measures rank
agreement; quadratic-weighted kappa ($\kappa_w$) measures agreement in absolute
score levels, and is the only one of the two that is sensitive to a judge's
overall calibration -- mean score over these rows is %s. Scores lie on
$0$--$1$ and are placed on the equivalent fixed %d-point ordinal grid for
$\kappa_w$ at the essay level; at the model level the values are means rather
than grid points, so the bin-width-to-zero form of the same statistic is used.
Entries are coefficients with percentile bootstrap $95\%%$ confidence intervals
from %s resamples, rounded to two decimals, so an interval printed as $1.00$ is a
rounded $0.99$ rather than perfect agreement. Model-level intervals resample
models; essay-level intervals resample complete generation-model clusters,
because essays from one model are not independent. All model-level coefficients
have $p < 0.001$ (analytic for $\rho$, permutation for $\kappa_w$); essay-level
$p$-values are omitted for the same dependence reason. Higher is stronger
agreement; $\kappa_w = 1$ under perfect agreement."""

MEASURES = [('Spearman rho', rho)]


def compute(ess, mdl, n_boot):
    """One row per (level, pair, measure)."""
    out = []
    for ka, kb in PAIRS:
        x = mdl[[ka, kb]].dropna()
        for label, f in MEASURES:
            lo, hi = ci_iid(f, x[ka], x[kb], n_boot)
            p = spearmanr(x[ka], x[kb]).pvalue
            out.append(dict(level='Model', pair=f'{NAME[ka]} / {NAME[kb]}', n=len(x),
                            clusters=np.nan, measure=label, coef=f(x[ka], x[kb]),
                            lo=lo, hi=hi, p=p))
        lo, hi = ci_iid(qwk_continuous, x[ka], x[kb], n_boot)
        out.append(dict(level='Model', pair=f'{NAME[ka]} / {NAME[kb]}', n=len(x),
                        clusters=np.nan, measure='QWK kappa_w',
                        coef=qwk_continuous(x[ka], x[kb]), lo=lo, hi=hi,
                        p=perm_p(qwk_continuous, x[ka], x[kb], n_boot)))

    for ka, kb in PAIRS:
        x = ess[['model', ka, kb]].dropna()
        for label, f in MEASURES + [('QWK kappa_w', qwk_ordinal)]:
            lo, hi = ci_cluster(f, x, ka, kb, n_boot)
            out.append(dict(level='Essay', pair=f'{NAME[ka]} / {NAME[kb]}', n=len(x),
                            clusters=x.model.nunique(), measure=label,
                            coef=f(x[ka], x[kb]), lo=lo, hi=hi, p=np.nan))
    return pd.DataFrame(out)


def figure_caption(res, ess):
    """The figure's caption, with its numbers taken from the same frame it plots."""
    def get(level, pair, measure):
        return res[(res.level == level) & (res.pair == f'{NAME[pair[0]]} / {NAME[pair[1]]}')
                   & (res.measure == measure)].iloc[0].coef
    ns = [len(ess[[a, b]].dropna()) for a, b in PAIRS]
    mdl = ess.groupby('model')[KEYS].mean()
    d = dict(nmod=ess.model.nunique(), K=K, nmin=f'{min(ns):,}', nmax=f'{max(ns):,}')
    for i, pr in enumerate(PAIRS, 1):
        d[f'ke{i}'] = get('Essay', pr, 'QWK kappa_w')
        d[f'rm{i}'] = get('Model', pr, 'Spearman rho')
        d[f'km{i}'] = get('Model', pr, 'QWK kappa_w')
        d[f'o{i}'] = (mdl[pr[0]] - mdl[pr[1]]).mean()
    # Re-wrapped after substitution: the numbers vary in width, so wrapping the
    # template instead leaves the breaks landing mid-clause.
    return '\n\n'.join(textwrap.fill(' '.join(para.split()), 78)
                       for para in (FIG_CAPTION % d).split('\n\n'))


def _cell(row):
    """Coefficient and its interval, two decimals throughout.

    Two decimals is all these numbers support: the intervals are 10,000-resample
    bootstrap percentiles, so the third digit moves with the seed. One casualty is
    honest rather than tidy -- the Qwen3.6/DS-V4 model-level rho interval prints
    [0.98, 1.00], and that 1.00 is a rounded 0.999, not a claim of perfect
    agreement. The table note says so.
    """
    return f'{row.coef:.2f} & {{\\tiny$[{row.lo:.2f},\\,{row.hi:.2f}]$}}'


def latex(res, mdl, ess, n_boot):
    """One captioned table, replacing the former Tables 15 and 16."""
    means = ', '.join(f'{NAME[k]} {mdl[k].mean():.1f}' for k in KEYS)
    L = [
        r'% Generated by experiments/analysis/judge_agreement_zb/judge_agreement.py',
        r'% Inter-judge agreement, no retrieval, on the panel Table 5 reports.',
        r'% Replaces the former Table 15 (long) and Table 16 (compact), which described',
        r'%   an older panel -- gpt-5-nano rather than gpt-oss-120b, 26 models, ~2100',
        r'%   essays -- and so validated judges that did not produce the reported scores.',
        r'%',
        r'% Rows are the row set of main_essay_table.csv, the same parent',
        f'%   merged_results_table.py reads for Table 5: {len(mdl)} models, {len(ess)} essays.',
        r'%   The script asserts that the per-judge means and the per-case medians it',
        r'%   rebuilds reproduce that file, so the two tables cannot drift apart.',
        r'%',
        r'% Model level: one observation is a generation model, each judge scored by its',
        "%   mean over that model's essays. Essay level: one observation is one essay",
        r'%   both judges scored. n is pairwise-complete, hence the odd essay difference.',
        r'%',
        "% rho is invariant to a judge's overall level; kappa_w is not, which is",
        r'%   why it is here. Mean score over the rows: ' + means + '.',
        f'% QWK: quadratic weights on the judges\' own fixed {K}-point grid (0..100) at the',
        r'%   essay level; at the model level the values are means rather than grid points,',
        r'%   so the bin-width-to-zero form of the same statistic is used. Where both are',
        r'%   defined they agree to 0.001.',
        r'%',
        f'% CIs are percentile bootstrap, {n_boot:,} resamples, seed {SEED}. Essay-level',
        r'%   intervals resample whole generation models: 81 essays share a generator,',
        r'%   and resampling essays instead reports intervals 3.6-5.8x too narrow.',
        r'% All model-level coefficients have p < 0.001 (analytic for rho, permutation',
        r'%   for kappa_w), so the column is stated here rather than printed.',
        r'%   Essay-level p-values are omitted: the clustering invalidates their null.',
        r'%',
        r'% tabcolsep is set locally and restored by the group the table environment',
        r'%   opens, so it cannot leak into the next table in the document.',
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{' + (CAPTION % (means, K, f'{n_boot:,}')).strip() + '}',
        r'\label{tab:judge_agreement_norag}',
        r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{l l r r@{\hspace{2pt}}l r@{\hspace{2pt}}l}',
        r'\toprule',
        r'Level & Judge pair & $n$ & \multicolumn{2}{c}{Spearman $\rho$}'
        r' & \multicolumn{2}{c}{QWK $\kappa_w$} \\',
        r'\midrule',
    ]
    for level in ['Model', 'Essay']:
        block = res[res.level == level]
        for i, pair in enumerate(dict.fromkeys(block.pair)):
            g = block[block.pair == pair].set_index('measure')
            head = level if i == 0 else ''
            # The pair is stacked rather than run together: two full endpoint names
            # on one line is the widest thing in the table by a distance, and the
            # column then sets the width of every gutter beside it.
            a, b = pair.split(' / ')
            cell = r'\shortstack[l]{' + a + r' /\\ ' + b + '}'
            L.append(f'{head} & {cell} & {int(g.iloc[0].n)} & '
                     + _cell(g.loc['Spearman rho']) + ' & '
                     + _cell(g.loc['QWK kappa_w']) + r' \\')
            if i < 2:
                L.append(r'\addlinespace[5pt]')
        if level == 'Model':
            L.append(r'\addlinespace[8pt]')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return L


# ---------------------------------------------------------------------- figure
def figure(ess, mdl, res, path):
    """Where the disagreement lives, which is the one thing the table cannot show.

    Top row is the essay-level joint distribution on the judges' own grid -- the
    matrix kappa_w is computed from, so a reader sees the off-diagonal mass the
    statistic is charging for, and sees that it sits on one side. Bottom row is
    the same pair at the model level against the identity line, where that same
    one-sidedness has become a clean vertical offset.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import PowerNorm

    # rocket over a matplotlib default: it is the ramp figure_common already uses
    # elsewhere, and it stays monotone in luminance when the appendix prints grey.
    heat = sns.color_palette('rocket_r', as_cmap=True)

    plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 11,
                         'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
                         'axes.edgecolor': '#4A4A4A', 'axes.linewidth': .8})
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2))

    def stats(level, ka, kb):
        g = res[(res.level == level) & (res.pair == f'{NAME[ka]} / {NAME[kb]}')] \
            .set_index('measure')
        return g.loc['Spearman rho'].coef, g.loc['QWK kappa_w'].coef

    for j, (ka, kb) in enumerate(PAIRS):
        # --- essay level: the 11x11 the ordinal kappa is built on
        ax = axes[0, j]
        x = ess[[ka, kb]].dropna()
        a, b = levels(x[ka]), levels(x[kb])
        m = np.zeros((K, K))
        np.add.at(m, (a, b), 1)
        m = m / m.sum() * 100
        im = ax.imshow(m, origin='lower', cmap=heat, norm=PowerNorm(0.45),
                       extent=(-.5, K - .5, -.5, K - .5))
        ax.plot([-.5, K - .5], [-.5, K - .5], color='#1F4E79', lw=1.1, ls='--', alpha=.85)
        _, kw = stats('Essay', ka, kb)
        ax.set_title(f'{NAME[ka]} vs {NAME[kb]}', pad=7)
        # Every panel is labelled on both axes. The third column pairs a different
        # judge on y than the first two, so a shared row label would be a lie.
        ax.set_xlabel(f'{NAME[kb]} score'), ax.set_ylabel(f'{NAME[ka]} score')
        ax.set_xticks(range(0, K, 2)), ax.set_yticks(range(0, K, 2))
        ax.set_xticklabels(range(0, 100 + 1, 20)), ax.set_yticklabels(range(0, 100 + 1, 20))
        ax.text(.04, .96, f'$\\kappa_w$ = {kw:.3f}\nn = {len(x):,}', transform=ax.transAxes,
                va='top', ha='left', fontsize=8.5,
                bbox=dict(fc='white', ec='#B0B0B0', lw=.6, alpha=.92, pad=2.5))
        ax.grid(False)

        # --- model level: the same pair, one point per generation model
        ax = axes[1, j]
        lim = (0, max(mdl[ka].max(), mdl[kb].max()) * 1.08)
        ax.plot(lim, lim, color='#1F4E79', lw=1.1, ls='--', alpha=.85, zorder=1)
        ax.scatter(mdl[kb], mdl[ka], s=34, color='#C0504D', ec='white', lw=.7, zorder=3)
        sr, kw = stats('Model', ka, kb)
        off = (mdl[ka] - mdl[kb]).mean()
        ax.set_xlim(lim), ax.set_ylim(lim), ax.set_aspect('equal')
        ax.set_xlabel(f'{NAME[kb]} mean'), ax.set_ylabel(f'{NAME[ka]} mean')
        ax.text(.04, .96, f'$\\rho$ = {sr:.3f}\n$\\kappa_w$ = {kw:.3f}',
                transform=ax.transAxes, va='top', ha='left', fontsize=8.5,
                bbox=dict(fc='white', ec='#B0B0B0', lw=.6, alpha=.92, pad=2.5))
        ax.text(.96, .06, f'mean offset {off:+.1f}', transform=ax.transAxes,
                va='bottom', ha='right', fontsize=8.5, color='#5A5A5A')
        ax.grid(True, color='#E4E4E4', lw=.6)
        ax.set_axisbelow(True)

    fig.subplots_adjust(left=.10, right=.885, top=.945, bottom=.075, hspace=.36, wspace=.34)
    # Placed after the layout, in figure coordinates, so it cannot land on the
    # third panel's tick labels the way an ax-anchored colorbar does.
    cax = fig.add_axes([.905, axes[0, 0].get_position().y0, .014,
                        axes[0, 0].get_position().height])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('% of essays', fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.text(.012, .74, 'Essay level', rotation=90, va='center', fontsize=11.5, weight='bold')
    fig.text(.012, .27, 'Model level', rotation=90, va='center', fontsize=11.5, weight='bold')
    for ext in ('pdf', 'png'):
        fig.savefig(path.with_suffix('.' + ext), dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------- self-test
def self_test(ess, mdl):
    """The ordinal kappa against sklearn, and the two kappa forms against each other."""
    from sklearn.metrics import cohen_kappa_score
    worst_sk = worst_form = 0.0
    for ka, kb in PAIRS:
        x = ess[[ka, kb]].dropna()
        mine = qwk_ordinal(x[ka], x[kb])
        theirs = cohen_kappa_score(levels(x[ka]), levels(x[kb]),
                                   weights='quadratic', labels=np.arange(K))
        worst_sk = max(worst_sk, abs(mine - theirs))
        worst_form = max(worst_form, abs(mine - qwk_continuous(x[ka], x[kb])))
    print(f'  ordinal QWK vs sklearn cohen_kappa_score: max |diff| {worst_sk:.2e}')
    print(f'  ordinal vs continuous QWK, essay level:   max |diff| {worst_form:.4f}')
    assert worst_sk < 1e-12, 'the ordinal kappa no longer matches sklearn'
    assert worst_form < 0.005, 'the two kappa forms have stopped being the same statistic'
    grid = sum(int(((ess[k].dropna() / 10) % 1 != 0).sum()) for k in KEYS)
    total = sum(int(ess[k].notna().sum()) for k in KEYS)
    print(f'  judgements off the {K}-point grid, snapped: {grid} of {total}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out_dir', nargs='?', default=str(OUT),
                    help='where to write judge_agreement_zb.{tex,csv,pdf,png} '
                         '(default: beside this script)')
    ap.add_argument('--results', default=str(RESULTS),
                    help='where to read main_essay_table.csv and the seat files from '
                         '(default: the deepseek_rerun panel that produced them)')
    ap.add_argument('--boot', type=int, default=N_BOOT,
                    help=f'bootstrap and permutation resamples (default {N_BOOT})')
    ap.add_argument('--self-test', action='store_true',
                    help='check the kappa implementation and exit')
    ap.add_argument('--no-figure', action='store_true', help='tables only')
    a = ap.parse_args()

    ess, mdl, ref = load(Path(a.results))
    print(f'{len(ref)} models, {len(ess)} essays, {ess.model.nunique()} clusters '
          f'-- reproduces main_essay_table.csv')
    if a.self_test:
        self_test(ess, mdl)
        return

    res = compute(ess, mdl, a.boot)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # _zb on the artefacts, not on the script: the tex is copied into the paper
    # directory and the pdf is \includegraphics'd, so once they leave here the
    # filename is the only provenance they still carry.
    (out / 'judge_agreement_zb.tex').write_text('\n'.join(latex(res, mdl, ess, a.boot)) + '\n')
    res.to_csv(out / 'judge_agreement_zb.csv', index=False)
    if not a.no_figure:
        figure(ess, mdl, res, out / 'judge_agreement_zb.pdf')
        (out / 'judge_agreement_zb_figure_caption.txt').write_text(
            figure_caption(res, ess) + '\n')

    for level in ['Model', 'Essay']:
        block = res[res.level == level]
        print(f'\n{level} level')
        for pair in dict.fromkeys(block.pair):
            g = block[block.pair == pair].set_index('measure')
            cells = '  '.join(
                f'{m.split()[0]:8s} {g.loc[m].coef:6.3f} [{g.loc[m].lo:6.3f},{g.loc[m].hi:6.3f}]'
                for m in ['Spearman rho', 'QWK kappa_w'])
            print(f'  {pair:30s} n={int(g.iloc[0].n):5d}  {cells}')
    print('\nmean score over the rows: '
          + '  '.join(f'{NAME[k]}={mdl[k].mean():.1f}' for k in KEYS))
    made = '{tex,csv}' if a.no_figure else '{tex,csv,pdf,png}, and the figure caption'
    print(f'wrote judge_agreement_zb.{made} to {out}')


if __name__ == '__main__':
    main()

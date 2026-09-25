"""Rebuild the main essay-writing table, the per-judge table and the associations
table on the re-judged panel.

Main table: no-RAG essay quality on the three seats re-run in one window
(gpt-oss-120b / Qwen3.6-35B / DeepSeek-V4-Flash-0731), with both article-recitation
scores folded in as columns, and DeepSeek carried as two rows -- the published run on
the retired checkpoint and the 0731 re-generation.

Associations table: what predicts essay quality. Mirrors the predictor set of
tab:essay_corr_combined in the paper draft (Knowledge / Length / Cost), and adds
the second recitation column and the Artificial Analysis index that the published
version was missing.

Where each predictor comes from, and why it matters
---------------------------------------------------
  Legal Ref. Sim.   generation property, recomputed for every row by
                    recompute_legal_ref.py with the currently installed refex --
                    which does NOT reproduce the published column. One extractor
                    across all 31 rows, rather than a published column of thirty
                    and one row computed differently. The published column is kept
                    for the sensitivity check the script prints.
  Answer Tokens     generation property, from the published run / the 0731 JSONL.
  Gen. Cost         generation property. Reported by the endpoint where there is
                    one; imputed from OpenRouter list price (OR_PRICE below) for
                    the self-hosted models, which either report nothing or report
                    a literal zero. n rises from 18 to 25.
  Judge Tokens      PANEL property -- summed over the three NEW seats, not carried
                    over from the retired panel.

Judge Cost, which the published table carries, is dropped: every seat on the new
panel is self-hosted, so the column is identically zero and the correlation it
reported was never measuring anything.

RAG columns are deliberately absent: the RAG arms have not been re-judged on these
seats, so a cross-arm delta would compare two judge versions.
"""
import glob, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seats import load_seat                                   # noqa: E402

# Model ids, defined before OR_PRICE because it keys on one of them.
DS0731 = 'deepseek-ai/DeepSeek-V4-Flash-0731'
GEMINI = 'google/gemini-3.7-flash'
OPUS = 'anthropic/claude-opus-5'
SOOFI = 'soofi-s-isar-preview'

# OpenRouter list price, USD per million tokens, (prompt, completion), fetched from
# https://openrouter.ai/api/v1/models on 2026-08-20. Used only for models we served
# ourselves, which have no price of their own: FAU reports either nothing at all or a
# literal 0.0, and a zero is not a price -- it would have anchored the bottom of the
# Gen. Cost row with seven models that are not actually free to run.
#
# How good is the imputation? Reconstructing the *reported* charge from list price for
# the 16 models that really were billed through OpenRouter is exact (ratio 1.000) for
# every first-party endpoint -- Anthropic, OpenAI, Google, DeepSeek -- and scatters
# 0.63x-1.95x for open-weight models, where OpenRouter's headline price belongs to one
# provider and the run may have been routed to another, and where prices have moved
# since July. So treat this row as an order-of-magnitude, rank-preserving estimate for
# the self-hosted models, not as a bill.
OR_PRICE = {
    'deepseek-ai/DeepSeek-V4-Flash':      ('deepseek/deepseek-v4-flash',      0.0812, 0.1624),
    'deepseek-ai/DeepSeek-V4-Flash-0731': ('deepseek/deepseek-v4-flash-0731', 0.1400, 0.2800),
    'Qwen/Qwen3.6-35B-A3B-FP8':           ('qwen/qwen3.6-35b-a3b',            0.1400, 1.0000),
    'openai/gpt-oss-120b':                ('openai/gpt-oss-120b',             0.0300, 0.1700),
    'RedHatAI/gemma-4-31B-it-FP8-block':  ('google/gemma-4-31b-it',           0.0900, 0.3400),
    'RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8':
                                          ('mistralai/mistral-small-3.2-24b-instruct', 0.09375, 0.2500),
    'mistralai/Mistral-Medium-3.5-128B':  ('mistralai/mistral-medium-3-5',    1.5000, 7.5000),
    'mistralai/Ministral-3-14B-Reasoning-2512':
                                          ('mistralai/ministral-14b-2512',    0.2000, 0.2000),
    'qwen3-next-80b-a3b-instruct':        ('qwen/qwen3-next-80b-a3b-instruct', 0.0900, 1.1000),
    # A closed-beta preview with no listing of its own. Priced by the nearest
    # listed sibling in size and shape: 30B total / 3.5B active against
    # Qwen3.6's 35B / 3B.
    SOOFI:                                ('proxy: qwen/qwen3.6-35b-a3b',     0.1400, 1.0000),
}
# Deliberately absent, and therefore still unpriced:
#   GaleneAI/Magistral-Small-2509-FP8-Dynamic  no Magistral listing on OpenRouter
#   utter-project/EuroLLM-22B-Instruct-2512    no EuroLLM listing
#   windprak/open_steuerllm                    local fine-tune, never published
#   Microsoft/Phi-4-mini-instruct              only the 14B phi-4 is listed, not mini
#   ibm-granite/granite-4.1-3b                 only granite-4.1-8b is listed
#   mistralai/mistral-small-3.1-24b-instruct   listed, but all 81 calls errored -> no tokens

# Derived, not hardcoded: an absolute path here resolves against whatever
# checkout happens to live there rather than the one this file is in.
ROOT = Path(__file__).resolve().parents[4]
R = ROOT / 'experiments/analysis/deepseek_rerun/results'
G = ROOT / 'experiments/zubaers_result/essay_writing/without_rag_0731'
PUB = ROOT / 'experiments/zubaers_result/essay_writing/without_rag/ji2/no_rag_ji2_result.csv'
REC = ROOT / 'experiments/zubaers_result/article_recitation'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
F = ROOT / 'experiments/zubaers_result/essay_writing/frontier'
O = ROOT / 'experiments/analysis/additional_models/opus5_agent'
# Opus answered all 81 cases, but was written through the agent harness rather than
# the API path every other row used, so its row is marked in the table and the
# caveat is spelled out in the header.
OPUS_N = 81

# Rows carried through the pipeline but not reported. Both are dropped at the
# presentation layer only -- they stay in the re-judged CSV and in
# legal_ref_recomputed.csv, so nothing upstream has to be re-run to bring them
# back, and the recomputation that cites them still describes what it did.
#   mistral-small-3.1-24b-instruct  all 81 calls errored; the "essays" are
#     25-character stubs and every judge scored them 0.0. An incomplete run, not
#     a measurement of the model.
#   DeepSeek-V4-Flash  the checkpoint NHR@FAU retired on 2026-08-01, superseded
#     by the -0731 re-generation that is reported in its place. Keeping both put
#     one model in the table twice, on two different sets of weights.
RETIRED = ['mistralai/mistral-small-3.1-24b-instruct', 'deepseek-ai/DeepSeek-V4-Flash']

# Models that are queued but not run. Nothing here is a measurement. The AA index is
# the one real number, because it is external to this work; every column this study
# produces renders as "--" and the row is daggered and italicised, because a
# placeholder that looks like a result is worse than no row at all.
#
# Their position is where they are *expected* to land, not where they were measured
# to land, and the table says so in its header comment.
PENDING = []

# --------------------------------------------------------- essay scores + judge cost
def seat(sub, tag):
    d = load_seat(sub, ['index', 'model', 'score', 'completion_tokens'], results=R)
    return d.rename(columns={'score': f's_{tag}', 'completion_tokens': f't_{tag}'})

ess = seat('gpt-oss-120b', 'oss').merge(seat('Qwen3.6-35B', 'qwen'), on=['index', 'model']) \
        .merge(seat('DeepSeek-V4-Flash-0731', 'ds'), on=['index', 'model'])
SC = ['s_oss', 's_qwen', 's_ds']
ess['panel'] = ess[SC].median(axis=1) * 100
ess['judge_tok'] = ess[['t_oss', 't_qwen', 't_ds']].sum(axis=1, min_count=1)

tab = ess.groupby('model').agg(
    essay=('panel', 'mean'), essay_sem=('panel', lambda s: s.std() / np.sqrt(len(s))),
    gpt_oss=('s_oss', lambda s: 100 * s.mean()), qwen36=('s_qwen', lambda s: 100 * s.mean()),
    dsv4=('s_ds', lambda s: 100 * s.mean()), judge_tok=('judge_tok', 'mean'))

# Generations that are not in the published sweep, each judged on the same three
# seats and so each turned into a row the same way. `expect` asserts a coverage
# the caller already knows -- only Opus carries one, now all 81.
EXTRA_RUNS = [
    (DS0731, G / 'judged', None),                       # the 0731 re-generation
    (GEMINI, F / f'judged_{GEMINI.split("/")[-1]}', None),
    (SOOFI, F / f'judged_{SOOFI}', None),
    (OPUS, O / 'judged', OPUS_N),                       # agent harness, marked in the table
]


def judged_row(judged_dir, expect=None):
    """One table row from a judged/ directory holding one JSONL per seat."""
    cols = {}
    for f, c in [('gpt-oss-120b', 'oss'), ('Qwen3.6-35B', 'qwen'),
                 ('DeepSeek-V4-Flash-0731', 'ds')]:
        j = pd.DataFrame([json.loads(l) for l in open(judged_dir / f'{f}.jsonl')]) \
              .drop_duplicates('index')
        cols[f's_{c}'] = j.set_index('index').score * 100
        cols[f't_{c}'] = j.set_index('index').completion_tokens
    d = pd.DataFrame(cols)
    if expect is not None:
        assert len(d) == expect, f'{judged_dir}: {len(d)} judged, expected {expect}'
    med = d[SC].median(axis=1)
    return dict(essay=med.mean(), essay_sem=med.std() / np.sqrt(len(d)),
                gpt_oss=d.s_oss.mean(), qwen36=d.s_qwen.mean(), dsv4=d.s_ds.mean(),
                judge_tok=d[['t_oss', 't_qwen', 't_ds']].sum(axis=1).mean())


for _mid, _dir, _expect in EXTRA_RUNS:
    tab.loc[_mid] = judged_row(_dir, _expect)

# ------------------------------------------------------- generation-side predictors
pub = pd.read_csv(PUB, usecols=['model', 'legal_ref_sim', 'prompt_tokens',
                                'completion_tokens', 'total_cost'])
gen = pub.groupby('model').agg(legal_ref=('legal_ref_sim', 'mean'),
                               prm_tok=('prompt_tokens', 'mean'),
                               ans_tok=('completion_tokens', 'mean'),
                               gen_cost=('total_cost', 'mean'))
# One extractor for every row: see recompute_legal_ref.py for why the published
# column cannot be used, and for the two corrections applied while recomputing.
RX = ROOT / 'experiments/analysis/deepseek_rerun/results/legal_ref_recomputed.csv'
assert RX.exists(), f'run recompute_legal_ref.py first -- no {RX}'
rx = pd.read_csv(RX)
gen['legal_ref_published'] = gen['legal_ref'] * 100
gen['legal_ref'] = rx.groupby('model').refex_now.mean() * 100
# The same three generations again, on the generation side. legal_ref comes from
# the one recomputation that covers every row; legal_ref_published stays NaN
# because these postdate the published sweep and so have no published column.
# A cost only where the endpoint billed: gemini went through OpenRouter and its
# reported_cost is a real charge, while 0731 (NHR@FAU) and soofi (InnKube) are
# self-hosted and get list-price imputation below, like every other such row.
EXTRA_GEN = [
    (DS0731, G / 'ds_v4_flash_0731_high.jsonl'),
    (GEMINI, F / f'{GEMINI.split("/")[-1]}_essays.jsonl'),
    (SOOFI, F / f'{SOOFI}_essays.jsonl'),
]
for _mid, _path in EXTRA_GEN:
    _g = pd.DataFrame([json.loads(l) for l in open(_path)]).drop_duplicates('index')
    gen.loc[_mid] = dict(legal_ref=rx[rx.model == _mid].refex_now.mean() * 100,
                         legal_ref_published=np.nan,
                         prm_tok=_g.prompt_tokens.mean(),
                         ans_tok=_g.completion_tokens.mean(),
                         gen_cost=_g.reported_cost.mean() if 'reported_cost' in _g else np.nan)

# The agent harness reports no usage record at all, so Opus is the one row whose
# tokens are estimated rather than recorded -- prompt tokens looked up from the
# claude-haiku-4.5 run on the same prompts, visible output measured from the essays,
# thinking carried over from haiku on the same case. estimate_tokens.py argues each
# step. Cost follows from those tokens at OpenRouter list, so it counts as imputed.
oest = pd.read_csv(O / 'token_estimate.csv')
gen.loc[OPUS] = dict(legal_ref=rx[rx.model == OPUS].refex_now.mean() * 100,
                     legal_ref_published=np.nan,
                     prm_tok=oest.prompt_tokens.mean(),
                     ans_tok=oest.completion_tokens.mean(),
                     gen_cost=oest.cost_usd.mean())

# Price the self-hosted rows at OpenRouter list. A reported 0.0 counts as unpriced.
gen['cost_imputed'] = False
for m, (_, p_in, p_out) in OR_PRICE.items():
    if m not in gen.index:
        continue
    if not (pd.isna(gen.at[m, 'gen_cost']) or gen.at[m, 'gen_cost'] == 0):
        continue                                          # a real charge -- leave it alone
    gen.loc[m, ['gen_cost', 'cost_imputed']] = [
        (gen.at[m, 'prm_tok'] * p_in + gen.at[m, 'ans_tok'] * p_out) / 1e6, True]
gen.loc[OPUS, 'cost_imputed'] = True          # priced at list, from estimated tokens
gen = gen.drop(columns='prm_tok')
tab = tab.join(gen, how='left')

# ------------------------------------------------------------- article recitation
rec = pd.read_csv(REC / 'article_recitation.csv', usecols=['model', 'score', 'dataset'])
for f in sorted(REC.glob('recitation_*.csv')):            # 0731 / gemma / phi, as they land
    e = pd.read_csv(f, usecols=['model', 'score', 'dataset'])
    rec = pd.concat([rec, e[e.score.notna()]], ignore_index=True)
rec = rec.pivot_table(index='model', columns='dataset', values='score', aggfunc='mean') * 100
rec.columns = ['rec_gpbam', 'rec_mostcited']
ALIAS = {'qwen/qwen3.6-35b-a3b': 'Qwen/Qwen3.6-35B-A3B-FP8'}   # innkube vs FAU serving
rec.index = [ALIAS.get(i, i) for i in rec.index]
tab = tab.join(rec, how='left')

# ------------------------------------------ Artificial Analysis intelligence index
src = (ROOT / 'experiments/analysis/plot_aa_index_vs_essay.py').read_text()
blk = src[src.index('AA_INDEX = {'):]
blk = blk[:blk.index('\n}')]
aa = {m: int(i) for m, i in re.findall(r'"([^"]+)":\s*\(\s*"[^"]*",\s*"[^"]*",\s*(\d+)', blk)}
tab['aa_index'] = [aa.get(m, np.nan) for m in tab.index]

tab = tab.drop(index=[m for m in RETIRED if m in tab.index])
tab = tab.sort_values('essay', ascending=False)
for m, _, aa, _ in PENDING:                       # unmeasured: only the AA index
    tab.loc[m] = {c: np.nan for c in tab.columns}
    tab.loc[m, 'aa_index'] = aa
    tab.loc[m, 'cost_imputed'] = False             # a bool column, and NaN is not one
PENDING_IDS = [m for m, _, _, _ in PENDING]
tab = tab.reindex(PENDING_IDS + [i for i in tab.index if i not in PENDING_IDS])
# gen_cost spans four orders of magnitude and the cheap end is where the ties would
# be manufactured, so it keeps six decimals rather than the display precision.
# Full precision, deliberately. merged_results_table.py reads this file and
# formats to 2 dp; rounding to 3 here made it round twice, and 77.884615 was
# reaching the merged table as 77.885 and printing as 77.89 against the 77.88
# of this table -- two numbers for one cell in the same paper.
tab.to_csv(OUT / 'main_essay_table.csv')
n_imp = int(tab.cost_imputed.sum())
print(f'=== Gen. Cost: {n_imp} self-hosted rows priced at OpenRouter list (2026-08-20), '
      f'{tab.gen_cost.notna().sum() - n_imp} as reported, '
      f'{tab.gen_cost.isna().sum()} still unpriced')
print(tab.loc[tab.cost_imputed, ['ans_tok', 'gen_cost']].round(4).to_string())

print('\n=== MAIN ESSAY-WRITING TABLE (no-RAG, three seats re-run in one window)')
print(tab[['essay', 'essay_sem', 'gpt_oss', 'qwen36', 'dsv4',
           'rec_gpbam', 'rec_mostcited', 'aa_index']].round(2).to_string())

# --------------------------------------------------------------------- associations
# Legal Ref. Sim. sits in its own group at the bottom, not among the Knowledge
# predictors. It is a second metric of the essay-writing task, computed from the
# same 81 essays the judges grade, so its association with the judge median is
# partly built in and is not evidence of the same kind as recitation or the AA
# index, both of which are measured on something else. It is reported for
# continuity with the published table and because it is the strongest number in
# the table -- which is the point.
#
# Spearman only in the table. Pearson measures skew rather than association on the
# rows whose predictor spans orders of magnitude -- Gen. Cost runs 0.0002 to 0.124,
# where r=0.32 against rho=0.59, the gap being two models far out on the x-axis --
# and the outcome is a judge-median quality index over a deliberately heterogeneous
# model set, so rank is the honest level of measurement. r is still written to the
# CSV, with its own bootstrap CI, and the divergence is printed below.
SPECS = [('Knowledge', 'Article Recitation, GPBam',      'rec_gpbam'),
         ('Knowledge', 'Article Recitation, Most cited', 'rec_mostcited'),
         ('Knowledge', 'Artificial Analysis index',      'aa_index'),
         ('Length',    'Answer Tokens (k)',              'ans_tok'),
         ('Length',    'Judge Tokens (k)',               'judge_tok'),
         ('Cost',      'Gen. Cost (\\$)',                'gen_cost'),
         ('Same essay', 'Legal Ref. Sim.',               'legal_ref')]

def boot_ci(x, y, fn, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n, x.size))
    v = [fn(x[i], y[i]).statistic for i in idx
         if np.unique(x[i]).size > 2 and np.unique(y[i]).size > 2]
    return (np.nan, np.nan) if not v else tuple(np.percentile(v, [2.5, 97.5]))

rows = []
for group, name, col in SPECS:
    s = tab[[col, 'essay']].dropna()
    if len(s) < 3 or s[col].nunique() < 2:
        rows.append(dict(group=group, predictor=name, n=len(s), pearson_r=np.nan,
                         spearman_rho=np.nan, ci_low=np.nan, ci_high=np.nan, p_rho=np.nan))
        continue
    x, y = s[col].to_numpy(float), s.essay.to_numpy(float)
    lo, hi = boot_ci(x, y, stats.spearmanr)
    rho, p = stats.spearmanr(x, y)
    # Pearson is kept in the CSV only -- see the note above SPECS for why it is
    # not in the table. Its CI is bootstrapped the same way so the CSV is not
    # half-quantified.
    r = stats.pearsonr(x, y)[0]
    rlo, rhi = boot_ci(x, y, stats.pearsonr)
    rows.append(dict(group=group, predictor=name, n=len(s), spearman_rho=round(rho, 3),
                     ci_low=round(lo, 3), ci_high=round(hi, 3), p_rho=float(f'{p:.2g}'),
                     pearson_r=round(r, 3), pearson_lo=round(rlo, 3), pearson_hi=round(rhi, 3)))
assoc = pd.DataFrame(rows)
assoc.to_csv(OUT / 'associations_essay_quality.csv', index=False)
print('\n=== ASSOCIATIONS WITH ESSAY QUALITY (no retrieval, re-judged panel)')
print(assoc.to_string(index=False))
div = assoc.dropna(subset=['spearman_rho']).assign(gap=lambda d: (d.spearman_rho - d.pearson_r).abs())
div = div[div.gap > 0.1]
if len(div):
    print('\n  r and rho diverge by more than 0.10 on: '
          + '; '.join(f'{r.predictor} (r={r.pearson_r:.2f}, rho={r.spearman_rho:.2f})'
                      for _, r in div.iterrows())
          + '\n  -- nonlinearity, not disagreement about direction.')

# Sensitivity 1: does swapping the extractor change what Legal Ref. Sim. says?
a = tab[['legal_ref', 'essay']].dropna()                    # recomputed, the one reported
b = tab[['legal_ref_published', 'essay']].dropna()          # the published column
print(f'\n  Legal Ref. Sim. by extractor: recomputed rho='
      f'{stats.spearmanr(a.legal_ref, a.essay).statistic:+.3f} (n={len(a)}, reported)  vs  '
      f'published rho={stats.spearmanr(b.legal_ref_published, b.essay).statistic:+.3f} (n={len(b)})')
j = tab[['legal_ref', 'legal_ref_published']].dropna()
print(f'  the two columns agree at r={stats.pearsonr(j.legal_ref, j.legal_ref_published)[0]:.4f}, '
      f'tau-b={stats.kendalltau(j.legal_ref, j.legal_ref_published).statistic:.3f}; '
      f'largest single move {(j.legal_ref - j.legal_ref_published).abs().max():.2f} points')

# Sensitivity 2 used to drop the degenerate mistral-small-3.1 row here. It is now
# out of the table entirely (see RETIRED), so there is nothing left to drop.

s = tab[['rec_gpbam', 'rec_mostcited', 'aa_index', 'essay']].dropna()
X = np.column_stack([np.ones(len(s)), s.rec_gpbam, s.aa_index])
pred = X @ np.linalg.lstsq(X, s.essay, rcond=None)[0]
r2 = 1 - ((s.essay - pred) ** 2).sum() / ((s.essay - s.essay.mean()) ** 2).sum()
print(f'\n  joint recitation(GPBam) + AA index: R^2 = {r2:.3f} (n={len(s)})')
print(f'  the two recitation scores correlate with each other at r = '
      f'{stats.pearsonr(s.rec_gpbam, s.rec_mostcited)[0]:.3f}')

# What the estimated Opus cost is worth to the Cost association, quoted in the
# associations header: it is the one predictor value in the table that no provider
# ever reported, and it is also the largest, so the reader is owed the delta.
_g = assoc.dropna(subset=['spearman_rho']).assign(
    gap=lambda d: (d.spearman_rho - d.pearson_r).abs()).sort_values('gap', ascending=False).iloc[0]
_wide = f'{_g.predictor} (r={_g.pearson_r:.2f} vs rho={_g.spearman_rho:.2f})'

_c = tab[['gen_cost', 'essay']].dropna()
n_cost = len(_c)
rho_cost = stats.spearmanr(_c.gen_cost, _c.essay).statistic
_c2 = _c.drop(index=OPUS, errors='ignore')
rho_cost_noopus = stats.spearmanr(_c2.gen_cost, _c2.essay).statistic

# ------------------------------------------------------------------- LaTeX output
NL, B = chr(92) * 2, chr(92)

def short(m):
    """Compact display name: drop the vendor prefix and the quantisation suffix."""
    n = re.sub(r'-FP8(-block|-Dynamic)?$', '', m.split('/')[-1], flags=re.I)
    return n.replace('_', B + '_')

def num(v, d=2):
    return '--' if pd.isna(v) else f'{v:.{d}f}'

with open(OUT / 'main_essay_table.tex', 'w') as f:
    f.write('% no-RAG. Judge median over three seats re-judged in one window (2026-08-19/20):\n')
    f.write('% gpt-oss-120b, Qwen3.6-35B-A3B, DeepSeek-V4-Flash-0731. Row-wise median of the\n')
    part = f' ({OPUS_N} for the $^' + B + 'ddagger$ row)' if OPUS_N < 81 else ''
    f.write(f'% three, averaged over the 81 cases{part},\n')
    f.write('% bootstrap SE in the subscript.\n')
    f.write('% Legal ref. sim. is the second essay-writing metric, computed from the same\n')
    f.write('% essays. Recomputed for every row by recompute_legal_ref.py: the published\n')
    f.write(f'% column cannot be reproduced by the installed refex, so one extractor across\n')
    f.write(f'% all {len(tab)} rows replaces a published column of 30 plus the rows added since.\n')
    f.write('% One DeepSeek row: the -0731 re-generation. The published row ran on the\n')
    f.write('% checkpoint NHR@FAU retired on 2026-08-01 and is not reported.\n')
    f.write('% mistral-small-3.1-24b-instruct is not reported either: all 81 of its calls\n')
    f.write('% errored, leaving 25-character stubs that every judge scored 0.0.\n')
    f.write('%\n% $^' + B + 'ddagger$ Claude-Opus-5 is measured, but not on equal terms with\n')
    f.write(f'% the rest of the table, and the row should not be read as one more benchmarked\n')
    if OPUS_N < 81:
        f.write(f'% model. Two differences, both of which favour it:\n')
        f.write(f'%   1. Coverage, essay columns only. {OPUS_N} of the 81 cases -- a contiguous\n')
        f.write(f'%      0-{OPUS_N - 1}, not a sample. Judge median is the mean over those, so the SE is\n')
        f.write('%      over ' + str(OPUS_N) + ', not 81. The recitation columns are NOT affected: all 200\n')
        f.write('%      recitation prompts were answered, so those two cells are full coverage.\n')
        f.write('%   2. Harness. Written through the Claude Code agent loop on the benchmark\n')
        f.write('%      prompt, not through the API path every other row used. The prompt is\n')
        f.write('%      byte-identical; the surrounding system prompt and tool loop are not.\n')
        f.write('%      See experiments/analysis/additional_models/opus5_agent/README.md.\n')
    else:
        f.write('% model. Coverage is no longer one of the reasons: all 81 essay cases are\n')
        f.write('% answered, so every cell in the row averages the same 81 cases as every\n')
        f.write('% other row, and the recitation cells all 200 prompts of both datasets.\n')
        f.write('% One difference remains, and it favours the row:\n')
        f.write('%   Harness. Written through the Claude Code agent loop on the benchmark\n')
        f.write('%   prompt, not through the API path every other row used. The prompt is\n')
        f.write('%   byte-identical; the surrounding system prompt and tool loop are not.\n')
        f.write('%   See experiments/analysis/additional_models/opus5_agent/README.md.\n')
    f.write('% Judging is on the same three seats as every other row, and the three seats\n')
    f.write('% disagree about these essays by 21 points -- wider than on any measured row --\n')
    f.write('% so the median is adjudicating a real disagreement, not smoothing noise.\n')
    f.write('% Recitation is measured, on all 200 prompts of both datasets, through the same\n')
    f.write('% agent harness and scored by the same ROUGE-L call as every other row. Every\n')
    f.write('% transcript was audited: 200 agents, 200 tool calls, all of them the Write of\n')
    f.write('% the answer, no read of the reference corpora that sit in this repo. See\n')
    f.write('% additional_models/opus5_agent/recitation/README.md. The harness caveat above\n')
    f.write('% still applies to these cells, which is why the row keeps its marker.\n')
    if PENDING:
        f.write('%\n% $^' + B + 'dagger$ rows are PLACEHOLDERS: queued, not run. Every column\n')
        f.write('% this study produces is "--" for them; the AA index is external and real.\n')
        f.write('% They sit where they are EXPECTED to land, not where they were measured to.\n')
        for m, name, aa, note in PENDING:
            f.write(f'%   {name}: AA {aa} -- {note}\n')
    f.write('#BEGIN#{lrrrrr}\n#TOP#\n')
    f.write(' & #MC2#{c}{Essay writing} & #MC2#{c}{Article recitation} & ' + NL + '\n')
    f.write('#CMIDA# #CMIDB#\n')
    f.write('Model & Judge median & Legal ref. sim. & GPBam & Most cited & AA '
            + NL + '\n#MID#\n')
    for m, r in tab.iterrows():
        pend = next((p for p in PENDING if p[0] == m), None)
        if pend:
            name = B + 'textit{' + pend[1] + '}$^{' + B + 'dagger}$'
            essay = '--'
        else:
            name, essay = short(m), f'{r.essay:.2f}$_{{{B}pm{r.essay_sem:.2f}}}$'
            if m == OPUS:
                name += '$^{' + B + 'ddagger}$'
        f.write(' & '.join([name, essay, num(r.legal_ref), num(r.rec_gpbam),
                            num(r.rec_mostcited), num(r.aa_index, 0)]) + ' ' + NL + '\n')
    f.write('#BOT#\n#END#\n')

with open(OUT / 'per_judge_scores.tex', 'w') as f:
    f.write('% Per-seat scores behind the judge median in the main table.\n')
    f.write('#BEGIN#{lrrr}\n#TOP#\n')
    f.write('Model & gpt-oss-120b & Qwen3.6-35B & DeepSeek-V4-Flash-0731 ' + NL + '\n#MID#\n')
    for m, r in tab.iterrows():
        if any(p[0] == m for p in PENDING):
            continue
        f.write(' & '.join([short(m), num(r.gpt_oss), num(r.qwen36), num(r.dsv4)]) + ' ' + NL + '\n')
    f.write('#BOT#\n#END#\n')

def cell(r):
    if pd.isna(r.spearman_rho):
        return [str(int(r.n)), '--', '']
    return [str(int(r.n)), f'{r.spearman_rho:.2f}',
            '{' + B + 'tiny$[' + f'{r.ci_low:.2f}' + ',' + B + ',' + f'{r.ci_high:.2f}' + ']$}']

lines = ['#BEGIN#{l r r@{' + B + 'hspace{3pt}}l}', '#TOP#',
         'Predictor & $n$ & #MC2#{c}{$' + B + 'rho$ {'
         + B + 'tiny[95' + B + '% CI]}} ' + NL, '#MID#']
last = None
for _, r in assoc.iterrows():
    if r.group != last:
        if last is not None:
            lines.append('#ADDLINE#')
        lines.append('#MC4#{l}{' + B + 'emph{' + r.group + '}} ' + NL)
        last = r.group
    lines.append(' & '.join([r.predictor] + cell(r)) + ' ' + NL)
lines += ['#BOT#', '#END#']
(OUT / 'associations_essay_quality.tex').write_text(
    '% Associations with essay quality, no retrieval, re-judged panel (2026-08-19/20).\n'
    '% Spearman only. Pearson is in the companion CSV with its own bootstrap CI; it\n'
    '% is not tabled because on the skewed predictors it measures skew rather than\n'
    f'% association. Widest |r - rho| in this run: {_wide}.\n'
    '% Legal Ref. Sim. is recomputed with one extractor across every row.\n'
    '% Judge Tokens is a property of the NEW panel, recomputed from the seats.\n'
    '% The published table\'s Judge Cost row is dropped: every seat is self-hosted,\n'
    '% so that column is identically zero.\n'
    '% Token rows are correlated on raw counts; the (k) in the label is display-only\n'
    '% and rank-neutral.\n'
    '% Gen. Cost: reported where the endpoint bills, imputed from OpenRouter list\n'
    '% price on 2026-08-20 otherwise (see OR_PRICE). Models with no OpenRouter\n'
    '% listing at their size stay out, which is where the shortfall in n comes from.\n'
    '% One caveat specific to this row: claude-opus-5 has no usage record at all, so\n'
    '% both its tokens and its price are estimated (estimate_tokens.py), and it is\n'
    '% the largest x in the column by a factor of six. Dropping it moves rho from\n'
    f'% {rho_cost:+.3f} to {rho_cost_noopus:+.3f} (n {n_cost} -> {n_cost - 1}).\n'
    '% Answer Tokens carries the same estimate, from the same place.\n' + '\n'.join(lines) + '\n')

REP = {'#BEGIN#': B + 'begin{tabular}', '#END#': B + 'end{tabular}', '#TOP#': B + 'toprule',
       '#MID#': B + 'midrule', '#BOT#': B + 'bottomrule', '#MC2#': B + 'multicolumn{2}',
       '#MC4#': B + 'multicolumn{4}',
       '#ADDLINE#': B + 'addlinespace[2pt]',
       '#CMIDA#': B + 'cmidrule(lr){2-3}', '#CMIDB#': B + 'cmidrule(lr){4-5}'}
for name in ['main_essay_table.tex', 'per_judge_scores.tex', 'associations_essay_quality.tex']:
    q = OUT / name
    s = q.read_text()
    for k, v in REP.items():
        s = s.replace(k, v)
    q.write_text(s)
print(f'\nwrote main_essay_table.{{csv,tex}}, per_judge_scores.tex, '
      f'associations_essay_quality.{{csv,tex}} to {OUT}')

"""Recompute `legal_ref_sim` with the dual §/Art. extractor instead of refex.

Why
---
`src.scoring.legal_ref_similarity` uses refex, which resolves a citation only for
books in its 1104-entry gazetteer plus a `§`-only fallback.  `GG` is in neither,
and the fallback does not fire on the `Art.` form, so every `Art. 12 Abs. 1 GG`,
`Art. 11 PAG`, `Art. 21 BayVwVfG` is dropped silently -- 2,308 of the 4,206
distinct case-level norms in the reference solutions (`analysis/out/refex_coverage.csv`).
`analysis/refs.py` sees both forms and is what every retrieval number already
uses; this script puts the essay-side metric on the same extractor.

The two are not nested.  refex resolves 228 norms `refs.py` misses, so this is a
swap rather than a widening, and both columns are written out for comparison.

Conventions carried over from `deepseek_rerun/scripts/recompute_legal_ref.py`
-----------------------------------------------------------------------------
Same essays, same sources, same undefined-case rule: a case whose *reference
solution* yields no extractable norm carries no signal about citation agreement
and is NaN rather than scored.  Under refex four of the 81 are undefined; under
the dual extractor none are, because every solution cites something in one form
or the other.  Unicode spacing is normalised before extraction either way.

Run from ``experiments/``::

    PYTHONPATH=analysis python analysis/legal_ref_dual.py
"""
import json
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refs as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PUB = ROOT / 'experiments/zubaers_result/essay_writing/without_rag/ji2/no_rag_ji2_result.csv'
REJUDGED = ROOT / 'experiments/zubaers_result/essay_writing/without_rag_0731/no_rag_ji2_rejudged.csv'
GEN0731 = ROOT / 'experiments/zubaers_result/essay_writing/without_rag_0731/ds_v4_flash_0731_high.jsonl'
FRONTIER = ROOT / 'experiments/zubaers_result/essay_writing/frontier'
OPUS = ROOT / 'experiments/analysis/additional_models/opus5_agent/opus5_agent.jsonl'
RECOMPUTED = ROOT / 'experiments/analysis/deepseek_rerun/results/legal_ref_recomputed.csv'
OUT = ROOT / 'experiments/analysis/out/legal_ref_dual.csv'
OUT_MODEL = ROOT / 'experiments/analysis/out/legal_ref_dual_by_model.csv'

EXTRA = [
    (GEN0731, 'deepseek-ai/DeepSeek-V4-Flash-0731'),
    (FRONTIER / 'gemini-3.7-flash_essays.jsonl', 'google/gemini-3.7-flash'),
    (FRONTIER / 'soofi-s-isar-preview_essays.jsonl', 'soofi-s-isar-preview'),
    (OPUS, 'anthropic/claude-opus-5'),
]

N_BOOT = 10_000
SEED = 20260909

_SPACES = {c: ' ' for c in range(0x110000)
           if chr(c) != ' ' and unicodedata.category(chr(c)) == 'Zs'}
_SPACES.update({0x200b: None, 0x200c: None, 0x200d: None, 0xfeff: None})


def _normalise(text):
    return str(text).translate(_SPACES)


def _norms(text):
    """(book, section) pairs, both citation forms, PDF artefacts repaired."""
    return {(c.book.lower(), c.section.lower())
            for c in R.canonicalise(R.extract(_normalise(text)))}


def _jaccard(a, b):
    if not a:
        return np.nan          # undefined: the solution cites nothing
    if not b:
        return 0.0
    return len(a & b) / len(a | b)


# --- solutions -------------------------------------------------------------
sol = {int(k): v for k, v in json.load(open(ROOT / 'experiments/data/gpbam.json'))['solutions'].items()}
sol_norms = {i: _norms(v) for i, v in sol.items()}
undefined = {i for i, n in sol_norms.items() if not n}
print(f'{len(sol)} reference solutions, '
      f'{np.mean([len(n) for n in sol_norms.values()]):.1f} norms each; '
      f'undefined cases: {sorted(undefined) or "none"}')

# --- essays ----------------------------------------------------------------
pub = pd.read_csv(PUB, usecols=['index', 'model', 'answer', 'legal_ref_sim'])
frames = [pub]
for path, mid in EXTRA:
    if not path.exists():
        print(f'skipping {mid}: no {path}')
        continue
    g = pd.DataFrame([json.loads(l) for l in open(path)]).drop_duplicates('index')
    g = g[g.answer.notna()]
    frames.append(g.assign(model=mid, legal_ref_sim=np.nan)[['index', 'model', 'answer', 'legal_ref_sim']])
    print(f'{mid}: {len(g)} essays from {path.name}')
d = pd.concat(frames, ignore_index=True)

t0 = time.time()
# Keep the raw set sizes as well: Jaccard conflates coverage with restraint, and
# the precision/recall split reported alongside it must come from this same
# extraction rather than a second, separately-parameterised one.
cells = [(len(sol_norms[i]), len(n), len(sol_norms[i] & n))
         for n, i in ((_norms(a), i) for a, i in zip(d.answer, d['index']))]
d['n_sol'], d['n_essay'], d['n_hit'] = (list(c) for c in zip(*cells))
d['dual'] = [np.nan if s_ == 0 else (0.0 if e == 0 else h / (s_ + e - h))
             for s_, e, h in cells]
d['precision'] = [np.nan if e == 0 else h / e for _, e, h in cells]
d['recall'] = [np.nan if s_ == 0 else h / s_ for s_, _, h in cells]
print(f'{len(d)} essays in {time.time() - t0:.0f}s')

# The refex column the reported table is built on, for the side-by-side.
if RECOMPUTED.exists():
    d = d.merge(pd.read_csv(RECOMPUTED)[['index', 'model', 'refex_now']],
                on=['index', 'model'], how='left')
else:
    print(f'note: {RECOMPUTED} missing -- comparing against the published column instead')
    d['refex_now'] = d.legal_ref_sim

d[['index', 'model', 'legal_ref_sim', 'refex_now', 'dual',
   'n_sol', 'n_essay', 'n_hit', 'precision', 'recall']].to_csv(OUT, index=False)


# --- model level, with the bootstrap SE the table reports ------------------
def boot_se(values, rng):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return np.nan
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    return float(v[idx].mean(axis=1).std(ddof=1))


rng = np.random.default_rng(SEED)
rows = []
for model, g in d.groupby('model'):
    rows.append({'model': model, 'n': int(g.dual.notna().sum()),
                 'precision': g.precision.mean(), 'recall': g.recall.mean(),
                 'cited': g.n_essay.mean(),
                 'refex': g.refex_now.mean() * 100, 'refex_se': boot_se(g.refex_now, rng) * 100,
                 'dual': g.dual.mean() * 100, 'dual_se': boot_se(g.dual, rng) * 100})
m = pd.DataFrame(rows).set_index('model')

# essay score from the reported panel, so the correlation matches the paper's
rej = pd.read_csv(REJUDGED)
JUDGES = ['score_Judge (gpt-oss-120b)', 'score_Judge (qwen3.6-35b-a3b)',
          'score_Judge (DeepSeek-V4-Flash-0731)']
rej['essay'] = rej[JUDGES].median(axis=1)
m['essay'] = rej.groupby('model').essay.mean() * 100
m['delta'] = m.dual - m.refex
m = m.sort_values('essay', ascending=False)
m.round(2).to_csv(OUT_MODEL)
print('\n' + m[['n', 'essay', 'refex', 'refex_se', 'dual', 'dual_se', 'delta',
                'precision', 'recall', 'cited']].round(2).to_string())

ok = m.dropna(subset=['refex', 'dual'])
print(f'\nRef. column: mean delta {ok.delta.mean():+.2f}, max |delta| {ok.delta.abs().max():.2f}, '
      f'spearman {stats.spearmanr(ok.refex, ok.dual).statistic:.3f}, '
      f'tau-b {stats.kendalltau(ok.refex, ok.dual).statistic:.3f}')
shift = (ok.refex.rank(ascending=False) - ok.dual.rank(ascending=False)).abs()
print(f'rank shift: mean {shift.mean():.2f}, max {int(shift.max())}, '
      f'{int((shift > 0).sum())}/{len(ok)} models move')

s = m.dropna(subset=['essay'])
for col in ('refex', 'dual'):
    sub = s.dropna(subset=[col])
    r = stats.spearmanr(sub.essay, sub[col])
    print(f'spearman(essay, {col:5s}) = {r.statistic:.3f}  n={len(sub)}  p={r.pvalue:.1e}')
# The precision/recall split the results section reports beside the Jaccard.
rep = m.drop(index=[i for i in ('deepseek-ai/DeepSeek-V4-Flash',
                                'mistralai/mistral-small-3.1-24b-instruct') if i in m.index])
e = d[d.model.isin(rep.index)]
print(f'\nreported panel: {len(rep)} models, {len(e)} essays')
print(f'  solution norms {d.n_sol.mean():.1f} +- {d.groupby("index").n_sol.first().std(ddof=1):.0f}, '
      f'essay norms {e.n_essay.mean():.1f}')
print(f'  mean precision {e.precision.mean():.2f}, mean recall {e.recall.mean():.2f}')
for half in ('recall', 'precision'):
    print(f'  spearman(essay, {half:9s}) = '
          f'{stats.spearmanr(rep.essay, rep[half]).statistic:.2f}')
best = rep.precision.idxmax()
print(f'  highest precision: {best} at {rep.precision.max():.2f} '
      f'on {rep.cited[best]:.1f} norms/essay, essay {rep.essay[best]:.2f}')

print(f'\nwrote {OUT}\n      {OUT_MODEL}')

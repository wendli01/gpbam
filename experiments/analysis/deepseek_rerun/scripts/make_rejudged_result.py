"""Re-judged drop-in for ``no_rag_ji2_result.csv``.

Every figure with essay quality on an axis ultimately reads that one file, and the
three ``plot_*`` scripts all chain off ``plot_cost_performance.build_table()``, whose
score is ``df.filter(regex='^score_Judge').median(axis=1)``. So the cheapest correct
way to move the figures onto the re-judged panel is not to fork four plotting scripts
but to hand them the same file with the judge columns replaced.

What this writes is the published result CSV with:

  * the four published ``score_Judge`` columns dropped -- ``gpt-5-nano``, both Qwen
    endpoint variants, and ``DeepSeek-V4-Flash`` on the checkpoint NHR@FAU retired on
    2026-08-01 -- and the three re-judged seats put in their place;
  * ``judgement_*`` columns rebuilt from the seats, so token and cost columns stay
    consistent with the scores beside them rather than describing the old panel;
  * three extra blocks of rows for generations that are not in the published sweep
    but were judged on the same three seats -- see ``EXTRA``.

Everything the judges did not touch -- ``legal_ref_sim``, ``prompt_tokens``,
``completion_tokens``, ``total_cost`` -- is carried through unchanged, because
re-judging does not alter the generation. The essay text is not: see ``CARRY``.

The three extra generations
---------------------------
``DeepSeek-V4-Flash-0731``  81 essays, the re-generation. Self-hosted at NHR@FAU,
    exactly like the published DeepSeek row: real tokens, no price reported.
``google/gemini-3.7-flash``  81 essays through OpenRouter: real tokens and a real
    bill, so nothing about this row is imputed.
``soofi-s-isar-preview``  81 essays, self-hosted on InnKube: real tokens, no
    price reported, so it is imputed at list downstream like the FAU rows.
``anthropic/claude-opus-5``  all 81 essays, written through the agent harness rather
    than the API (see ``additional_models/opus5_agent/README.md``). It is the one
    block whose tokens are estimated rather than recorded, because the harness
    reports no usage; ``estimate_tokens.py`` documents how, and every consumer
    flags the row. Its 48 rows are 48, not 81 padded out -- the plotting code
    normalises per essay and says which rows are short.

    python make_rejudged_result.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / 'experiments/analysis/deepseek_rerun/results'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from seats import load_seat  # noqa: E402
GEN0731 = ROOT / 'experiments/zubaers_result/essay_writing/without_rag_0731'
FRONTIER = ROOT / 'experiments/zubaers_result/essay_writing/frontier'
OPUS_DIR = ROOT / 'experiments/analysis/additional_models/opus5_agent'
PUB = ROOT / 'experiments/zubaers_result/essay_writing/without_rag/ji2/no_rag_ji2_result.csv'
OUT = GEN0731 / 'no_rag_ji2_rejudged.csv'
DS0731 = 'deepseek-ai/DeepSeek-V4-Flash-0731'
GEMINI = 'google/gemini-3.7-flash'
OPUS = 'anthropic/claude-opus-5'
SOOFI = 'soofi-s-isar-preview'

SEATS = {                    # judge label -> (published seat dir, name in every judged/)
    'gpt-oss-120b':           ('gpt-oss-120b',           'gpt-oss-120b.jsonl'),
    'qwen3.6-35b-a3b':        ('Qwen3.6-35B',            'Qwen3.6-35B.jsonl'),
    'DeepSeek-V4-Flash-0731': ('DeepSeek-V4-Flash-0731', 'DeepSeek-V4-Flash-0731.jsonl'),
}

# (model id, essay jsonl, judged dir, token source). "recorded" reads prompt_tokens /
# completion_tokens straight off the generation; "estimated" reads them from the CSV
# estimate_tokens.py writes, which is the only place in this pipeline where a token
# count is not something a provider reported.
EXTRA = [
    (DS0731, GEN0731 / 'ds_v4_flash_0731_high.jsonl', GEN0731 / 'judged', 'recorded'),
    (GEMINI, FRONTIER / 'gemini-3.7-flash_essays.jsonl',
     FRONTIER / 'judged_gemini-3.7-flash', 'recorded'),
    (SOOFI, FRONTIER / f'{SOOFI}_essays.jsonl', FRONTIER / f'judged_{SOOFI}', 'recorded'),
    (OPUS, OPUS_DIR / 'opus5_agent.jsonl', OPUS_DIR / 'judged', 'estimated'),
]

# Deliberately without `answer`, `message` and `qa_prompt`: 83 MB of the 84 MB is
# those three, nothing downstream of this file reads them, and the essay text is
# already tracked twice -- in the published result CSV and, for the extra rows, in
# the generation jsonl each block is built from. A derived index should not carry a
# third copy.
CARRY = ['index', 'model', 'legal_ref_sim', 'finish_reason', 'prompt_tokens', 'time',
         'completion_tokens', 'total_tokens', 'total_cost', 'temperature', 'top_p']


def seat_frames():
    """Per-seat score and judge token count, keyed by (index, model)."""
    out = {}
    for label, (sub, jsonl) in SEATS.items():
        frames = [load_seat(sub, ['index', 'model', 'score',
                                  'prompt_tokens', 'completion_tokens'],
                            results=RESULTS)]
        for mid, _, judged, _ in EXTRA:
            j = pd.DataFrame([json.loads(l) for l in open(judged / jsonl)]
                             ).drop_duplicates('index')
            frames.append(j.assign(model=mid)[['index', 'model', 'score',
                                               'prompt_tokens', 'completion_tokens']])
        out[label] = pd.concat(frames, ignore_index=True)
    return out


def gen_block(mid, path, tokens, carry, rx):
    """One generation as rows in the published CSV's shape."""
    g = pd.DataFrame([json.loads(l) for l in open(path)]).drop_duplicates('index')
    if 'error' in g:
        g = g[g.error.isna()]
    block = pd.DataFrame({c: np.nan for c in carry}, index=range(len(g)))
    block['index'] = g['index'].to_numpy()
    block['model'] = mid
    if 'finish_reason' in g and 'finish_reason' in block:
        block['finish_reason'] = g.finish_reason.to_numpy()

    if tokens == 'recorded':
        block['prompt_tokens'] = g.prompt_tokens.to_numpy()
        block['completion_tokens'] = g.completion_tokens.to_numpy()
    else:
        est = pd.read_csv(OPUS_DIR / 'token_estimate.csv').set_index('index')
        assert set(g['index']) <= set(est.index), 'estimate_tokens.py is stale'
        block['prompt_tokens'] = est.prompt_tokens.reindex(block['index']).to_numpy()
        block['completion_tokens'] = est.completion_tokens.reindex(block['index']).to_numpy()
    block['total_tokens'] = block.prompt_tokens + block.completion_tokens
    # A price only where the provider charged one. Self-hosted rows and the agent
    # run are left NaN and priced at list downstream, where the imputation is
    # documented and flagged in the figure.
    block['total_cost'] = g.reported_cost.to_numpy() if 'reported_cost' in g else np.nan

    return block.drop(columns=['legal_ref_sim']).merge(
        rx[rx.model == mid].rename(columns={'refex_now': 'legal_ref_sim'}),
        on=['index', 'model'], how='left')


pub = pd.read_csv(PUB)
carry = [c for c in CARRY if c in pub.columns]
base = pub[carry].copy()

# legal_ref_sim is replaced, not carried: quality_vs_legal_reference_similarity plots
# it, and it has to be the same column the tables report. See recompute_legal_ref.py.
RX = ROOT / 'experiments/analysis/deepseek_rerun/results/legal_ref_recomputed.csv'
assert RX.exists(), f'run recompute_legal_ref.py first -- no {RX}'
rx = pd.read_csv(RX)[['index', 'model', 'refex_now']]
base = base.drop(columns=['legal_ref_sim']).merge(rx, on=['index', 'model'], how='left') \
           .rename(columns={'refex_now': 'legal_ref_sim'})

blocks = [gen_block(mid, path, tokens, carry, rx) for mid, path, _, tokens in EXTRA]
for mid, block in zip([e[0] for e in EXTRA], blocks):
    print(f'{mid}: {len(block)} rows')
out = pd.concat([base] + blocks, ignore_index=True)

for label, d in seat_frames().items():
    d = d.rename(columns={'score': f'score_Judge ({label})'})
    d[f'judgement_Judge ({label})'] = [
        str({'prompt_tokens': p, 'completion_tokens': c, 'total_cost': 0.0})
        for p, c in zip(d.prompt_tokens, d.completion_tokens)]
    out = out.merge(d[['index', 'model', f'score_Judge ({label})',
                       f'judgement_Judge ({label})']], on=['index', 'model'], how='left')

sc = out.filter(regex=r'^score_Judge')
assert len(sc.columns) == 3, sc.columns.tolist()
n_missing = int(sc.isna().any(axis=1).sum())
print(f'{len(out)} rows, {out.model.nunique()} models, {len(sc.columns)} judge columns')
print(f'rows missing a judge score: {n_missing}')
print(f'panel median mean: {(sc.median(axis=1) * 100).mean():.2f}')
out.to_csv(OUT, index=False)
print(f'wrote {OUT}')

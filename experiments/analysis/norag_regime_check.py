"""Does a freshly generated DS4 no-RAG arm score where the banked one does?

Every cell of the DeepSeek oracle ablation is (arm - no-RAG baseline), and the
baseline is the main table's run. If DS4's generation shifted between that run
and the 2026-09-12 re-generation, every re-run cell carries the shift on top of
its condition effect -- and the pattern says it does: the five regenerated cells
moved +5.6 to +8.7 while `_gold_top10`, whose essays were *not* regenerated,
moved exactly 0.00.

This regenerates the no-RAG arm and compares it against the banked one on the
**gpt-oss seat**, the only judge that is NHR@FAU on both sides, so the comparison
is generation-vs-generation with nothing substituted downstream. Judging is
therefore free; only the 81 generations cost anything.

CONFOUND, stated up front: NHR@FAU's DeepSeek deployment is down, so this
generates through the OpenRouter fallback. A shift measured here is
"fresh generation vs banked generation" and cannot be attributed to an FAU
redeployment rather than to DeepInfra-vs-FAU. What it *can* settle is whether
the banked baseline sits low relative to any fresh draw -- which is what decides
whether the re-run's +11.60 oracle delta is a condition effect or a baseline
artefact.

    PYTHONPATH=analysis python -u analysis/norag_regime_check.py          # run it
    PYTHONPATH=analysis python -u analysis/norag_regime_check.py --compare  # report only

``--effort=low`` generates at DeepSeek's `low` (no effort prefix), the regime
NHR@FAU actually served in August, into ``norag_regime_..._low.csv``.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath('..'))

import oracle_experiment as oe
import paper_table as pt

MODEL = 'deepseek/deepseek-v4-flash-0731'          # the endpoint
STORED = 'deepseek-ai/DeepSeek-V4-Flash'           # the identity, as the arms store it
SEAT = 'openai/gpt-oss-120b'                       # free, NHR@FAU, same on both sides
OUT = f'{oe.D}/norag_regime_deepseek-ai_DeepSeek-V4-Flash.csv'
PROVIDER = {'order': ['DeepInfra'], 'allow_fallbacks': False,
            'data_collection': 'deny'}
EFFORT = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--effort=')), 'high')
if EFFORT != 'high':
    OUT = OUT.replace('.csv', f'_{EFFORT}.csv')


def banked_gptoss():
    """The banked no-RAG arm's gpt-oss seat, per case."""
    row = pt.main_norag()
    row = row[row.model == pt.MAIN_NORAG_ID.get(STORED, STORED)].sort_values('index')
    return 100 * row['score_Judge (gpt-oss-120b)'].to_numpy(dtype=float)


def compare():
    if not os.path.exists(OUT):
        print(f'{OUT} not generated yet')
        return
    d = pd.read_csv(OUT).sort_values('index')
    col = 'score_Judge (gpt-oss-120b)'
    if col not in d.columns:
        print(f'no gpt-oss seat in {OUT} yet'); return
    new = 100 * d[col].to_numpy(dtype=float)
    old = banked_gptoss()
    k = min(len(new), len(old))
    dd = pd.Series(new[:k] - old[:k]).dropna()
    s = dd.std(ddof=1) / np.sqrt(len(dd))
    print(f'\n=== gpt-oss seat (NHR@FAU on both sides), n={len(dd)}')
    print(f'banked no-RAG arm   : {np.nanmean(old[:k]):.2f}')
    print(f'fresh  no-RAG arm   : {np.nanmean(new[:k]):.2f}')
    print(f'shift               : {dd.mean():+.2f}   2SEM {2 * s:.2f}')
    print(f'\nthe oracle arm on this same seat reads 53.15, i.e. +11.60 over the')
    print(f'banked baseline. Against a fresh baseline it would read '
          f'{53.15 - np.nanmean(new[:k]):+.2f}.')
    print('\nessay length, mean chars:  banked n/a here, fresh '
          f'{d.answer.dropna().str.len().mean():.0f}')
    print(f'completion tokens, median: fresh {d.completion_tokens.median():.0f} '
          f'(effort {EFFORT}; banked FAU-August 10,685)')
    if abs(dd.mean()) > 2 * s:
        print('\n-> the banked baseline does NOT match a fresh draw; the re-run'
              '\n   deltas are inflated by roughly this shift.')
    else:
        print('\n-> a fresh draw reproduces the banked baseline; the re-run'
              '\n   oracle delta is a condition effect, not a baseline artefact.')


def main():
    if '--compare' in sys.argv:
        compare(); return
    oe.api_keys()
    facts, sols = oe.cases()
    from src import qa, scoring, prompts
    qa_plain, _ = prompts.QA_PROMPTS_BY_NAME['qa1']
    instruction = prompts.JUDGE_INSTRUCTIONS_BY_NAME['ji2']
    judge = scoring.JudgeEnsemble(
        [scoring.Judge(model=SEAT, prompt=prompts.build_judge_user(instruction))],
        verbose=True)
    gen = qa.AnswerGenerator(model=MODEL, prompt=qa_plain, provider=PROVIDER,
                             max_concurrency=16, timeout=oe.GEN_TIMEOUT,
                             reasoning_effort=EFFORT)
    print(f'generating {len(facts)} no-RAG essays with {gen.model} on '
          f'{gen.inference_endpoint} via {PROVIDER["order"]} at effort {EFFORT} -> {OUT}',
          flush=True)
    print(f'judging inline with {SEAT} (free, NHR@FAU)', flush=True)
    oe.generate_chunked(gen, facts, sols, judge, STORED, OUT, chunk=20)
    compare()


if __name__ == '__main__':
    main()

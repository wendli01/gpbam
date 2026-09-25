"""DeepSeek judge seat at `low` for existing arms, into side files.

``--endpoint deepinfra`` (OpenRouter pinned to DeepInfra, ~$0.002 per call) or
``gwdg`` (free; 30/min, 200/h, 1000/day, and reserved for generation). The two
agree at `low` (DS4-ablation-rerun.md, "GWDG as the DeepSeek judge"). Only rows
with a real answer are sent -- a blank answer is not an essay -- rate limits are
waited out via the Generator's retry budget, and rows whose score does not parse
are retried once at the end.

    python analysis/ds4_low/judge_low_arms.py --endpoint deepinfra <arm.csv> <side.csv> [...]

``--seat gpt-oss`` or ``--seat qwen`` fills the NHR@FAU seats the same way (live rows
only, retried, one side file per seat), so a supervisor can re-launch any seat
that is incomplete without touching the others.
"""
import os, sys, time
import numpy as np, pandas as pd
R = os.environ.get('GPBAM_ROOT') or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root: set GPBAM_ROOT to override
os.chdir(f'{R}/experiments'); sys.path[:0] = [f'{R}/experiments/analysis', R]
import oracle_experiment as oe
oe.api_keys()
facts, sols = oe.cases()
from src import scoring, prompts, evaluate
args = sys.argv[1:]
ENDPOINT, SEAT_KEY = 'gwdg', 'deepseek-low'
while args[:1] in (['--endpoint'], ['--seat']):
    if args[0] == '--endpoint':
        ENDPOINT = args[1]
    else:
        SEAT_KEY = args[1]
    args = args[2:]
assert ENDPOINT in ('deepinfra', 'gwdg'), ENDPOINT
FAU_SEATS = {'gpt-oss': 'openai/gpt-oss-120b', 'qwen': 'Qwen/Qwen3.6-35B-A3B-FP8'}
assert SEAT_KEY in ('deepseek-low', *FAU_SEATS), SEAT_KEY
SEAT = (f"Judge ({FAU_SEATS[SEAT_KEY].split('/')[-1]})" if SEAT_KEY in FAU_SEATS
        else 'Judge (deepseek-v4-flash-0731)')
DEEPINFRA = {'order': ['DeepInfra'], 'allow_fallbacks': False, 'data_collection': 'deny'}


def judge_rows(answers, idx):
    instr = prompts.build_judge_user(prompts.JUDGE_INSTRUCTIONS_BY_NAME['ji2'])
    if SEAT_KEY in FAU_SEATS:
        j = scoring.Judge(model=FAU_SEATS[SEAT_KEY], prompt=instr, max_concurrency=12, max_retries=60)
    elif ENDPOINT == 'gwdg':
        j = scoring.Judge(model='deepseek-v4-flash-0731', prompt=instr, inference_endpoint=os.environ['ENDPOINT_AC'],
                          token_var='API_KEY_AC', reasoning_effort='low', stream=True, max_concurrency=6,
                          max_retries=80, max_backoff=120)
    else:
        j = scoring.Judge(model='deepseek/deepseek-v4-flash-0731', prompt=instr, provider=DEEPINFRA,
                          reasoning_effort='low', max_concurrency=12)
    ens = scoring.JudgeEnsemble([j], verbose=True)
    return evaluate.rejudge_model(answers, [facts[i] for i in idx], [sols[i] for i in idx], judge=ens,
                                  model_name='deepseek-ai/DeepSeek-V4-Flash', verbose=True)


for arm_p, side_p in zip(args[0::2], args[1::2]):
    arm = pd.read_csv(arm_p, low_memory=False).sort_values('index').reset_index(drop=True)
    live = [i for i in range(len(arm)) if isinstance(arm.answer[i], str) and arm.answer[i].strip()]
    if os.path.exists(side_p):
        side = pd.read_csv(side_p, low_memory=False).sort_values('index').reset_index(drop=True)
    else:
        side = pd.DataFrame({'index': arm['index'].values})
    col = f'score_{SEAT}'
    todo = [i for i in live if col not in side or pd.isna(side.loc[i, col])]
    print(f'### {time.strftime("%T")} {os.path.basename(arm_p)}: {len(live)} live, {len(todo)} to judge -> {side_p}', flush=True)
    for rnd in (1, 2):
        if not todo:
            break
        t0 = time.time()
        out = judge_rows([arm.answer[i] for i in todo], todo)
        for c in [c for c in out.columns if c.endswith(SEAT)]:
            if c not in side:
                side[c] = np.nan
            side[c] = side[c].astype(object)
            for k, i in enumerate(todo):
                side.at[i, c] = out[c].iloc[k]
        if SEAT_KEY in FAU_SEATS:
            side['judge_endpoint'], side['judge_reasoning_effort'] = 'nhr_fau', 'high'
        else:
            side['judge_endpoint'] = 'gwdg' if ENDPOINT == 'gwdg' else 'openrouter:DeepInfra'
            side['judge_reasoning_effort'] = 'low'
        side.to_csv(side_p + '.tmp', index=False); os.replace(side_p + '.tmp', side_p)
        todo = [i for i in live if pd.isna(side.loc[i, col])]
        print(f'    round {rnd}: {time.time()-t0:.0f}s, {len(todo)} unscored left', flush=True)
    s = pd.to_numeric(side[col], errors='coerce')
    print(f'### {os.path.basename(side_p)}: n={int(s.notna().sum())} mean {100*s.mean():.2f}', flush=True)

"""Fill null rows in the low-effort DeepSeek judge side files (retry only those rows)."""
import os, sys
import pandas as pd
R = os.environ.get('GPBAM_ROOT') or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root: set GPBAM_ROOT to override
os.chdir(f'{R}/experiments'); sys.path[:0] = [f'{R}/experiments/analysis', R]
import oracle_experiment as oe
oe.api_keys()
facts, sols = oe.cases()
from src import scoring, prompts, evaluate
C = 'score_Judge (deepseek-v4-flash-0731)'
PROVIDER = {'order': ['DeepInfra'], 'allow_fallbacks': False, 'data_collection': 'deny'}
for arm_p, side_p in zip(sys.argv[1::2], sys.argv[2::2]):
    arm = pd.read_csv(arm_p, low_memory=False).sort_values('index').reset_index(drop=True)
    side = pd.read_csv(side_p, low_memory=False).sort_values('index').reset_index(drop=True)
    holes = [i for i in range(len(side)) if pd.isna(side.loc[i, C]) and isinstance(arm.loc[i, 'answer'], str) and arm.loc[i, 'answer'].strip()]
    blank = int(arm.answer.isna().sum())
    print(f'### {os.path.basename(side_p)}: {len(holes)} null judgement(s) on real answers, {blank} blank answer(s)', flush=True)
    for i in holes:
        fr = side.loc[i, 'judgement_finish_reason_Judge (deepseek-v4-flash-0731)'] if 'judgement_finish_reason_Judge (deepseek-v4-flash-0731)' in side else None
        txt = str(side.loc[i, 'judgement_text_Judge (deepseek-v4-flash-0731)'])[:160].replace('\n', ' ') if 'judgement_text_Judge (deepseek-v4-flash-0731)' in side else ''
        print(f'   case {int(side.loc[i, "index"])}: finish={fr} text={txt!r}', flush=True)
    if not holes:
        continue
    ens = scoring.JudgeEnsemble([scoring.Judge(model='deepseek/deepseek-v4-flash-0731',
                                               prompt=prompts.build_judge_user(prompts.JUDGE_INSTRUCTIONS_BY_NAME['ji2']),
                                               provider=PROVIDER, max_concurrency=8, reasoning_effort='low')], verbose=False)
    out = evaluate.rejudge_model([arm.loc[i, 'answer'] for i in holes], [facts[i] for i in holes], [sols[i] for i in holes],
                                 judge=ens, model_name='deepseek-ai/DeepSeek-V4-Flash', verbose=False)
    for k, i in enumerate(holes):
        for col in out.columns:
            if col.endswith('Judge (deepseek-v4-flash-0731)') or col == 'judging_cost':
                side[col] = side[col].astype(object) if col not in ('judging_cost',) and col in side else side.get(col)
                side.at[i, col] = out[col].iloc[k]
    side.to_csv(side_p + '.tmp', index=False); os.replace(side_p + '.tmp', side_p)
    print(f'   filled: n={int(side[C].notna().sum())} mean {100*side[C].mean():.2f}', flush=True)

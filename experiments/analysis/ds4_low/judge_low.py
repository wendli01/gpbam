"""Re-judge the DS4-ablation-low _gold_combined arm with the DeepSeek seat at `low` on DeepInfra.

The August DeepSeek judge seat (NHR@FAU) sent `high` without DeepSeek's effort prefix,
i.e. judged at `low`; the DeepInfra seat honours `high`. Separate file, nothing overwritten."""
import os, sys, time
import pandas as pd
R = os.environ.get('GPBAM_ROOT') or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root: set GPBAM_ROOT to override
os.chdir(f'{R}/experiments'); sys.path[:0] = [f'{R}/experiments/analysis', R]
import oracle_experiment as oe
oe.api_keys()
facts, sols = oe.cases()
from src import scoring, prompts, evaluate
src, out_p = sys.argv[1], sys.argv[2]
d = pd.read_csv(src, low_memory=False).sort_values('index').reset_index(drop=True)
PROVIDER = {'order': ['DeepInfra'], 'allow_fallbacks': False, 'data_collection': 'deny'}
ens = scoring.JudgeEnsemble([scoring.Judge(model='deepseek/deepseek-v4-flash-0731',
                                           prompt=prompts.build_judge_user(prompts.JUDGE_INSTRUCTIONS_BY_NAME['ji2']),
                                           provider=PROVIDER, max_concurrency=16, reasoning_effort='low')], verbose=True)
t0 = time.time()
out = evaluate.rejudge_model(d.answer.tolist(), facts, sols, judge=ens, model_name='deepseek-ai/DeepSeek-V4-Flash', verbose=True)
keep = ['index'] if 'index' in out else []
out.insert(0, 'index', d['index'].values) if 'index' not in out else None
out['judge_reasoning_effort'] = 'low'
out.to_csv(out_p, index=False)
c = 'score_Judge (deepseek-v4-flash-0731)'
print(f'### low-effort DeepSeek seat: n={int(out[c].notna().sum())} mean {100*out[c].mean():.2f} '
      f'(high-effort seat {100*d[c].mean():.2f}); cost ${out.judging_cost.sum():.2f}; {time.time()-t0:.0f}s -> {out_p}', flush=True)

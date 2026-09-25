"""Is GWDG's DeepSeek judge at `low` interchangeable with DeepInfra's? Same essays, paired."""
import os, sys, time
import pandas as pd
R = os.environ.get('GPBAM_ROOT') or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root: set GPBAM_ROOT to override
os.chdir(f'{R}/experiments'); sys.path[:0] = [f'{R}/experiments/analysis', R]
import oracle_experiment as oe
oe.api_keys()
facts, sols = oe.cases()
from src import scoring, prompts, evaluate
O = 'zubaers_result/essay_writing/oracle_rag/DS4-ablation-low'
arm = pd.read_csv(f'{O}/oracle_deepseek-ai_DeepSeek-V4-Flash_gold_combined.csv', low_memory=False).sort_values('index').reset_index(drop=True)
out_p = f'{O}/judge_gwdg_low_deepseek_gold_combined.csv'
ens = scoring.JudgeEnsemble([scoring.Judge(model='deepseek-v4-flash-0731',
                                           prompt=prompts.build_judge_user(prompts.JUDGE_INSTRUCTIONS_BY_NAME['ji2']),
                                           inference_endpoint=os.environ['ENDPOINT_AC'], token_var='API_KEY_AC',
                                           reasoning_effort='low', stream=True, max_concurrency=8)], verbose=True)
t0 = time.time()
out = evaluate.rejudge_model(arm.answer.tolist(), facts, sols, judge=ens, model_name='deepseek-ai/DeepSeek-V4-Flash', verbose=True)
out.insert(0, 'index', arm['index'].values) if 'index' not in out else None
out['judge_endpoint'], out['judge_reasoning_effort'] = 'gwdg', 'low'
out.to_csv(out_p, index=False)
print(f'### GWDG low DeepSeek seat done in {time.time()-t0:.0f}s -> {out_p}', flush=True)

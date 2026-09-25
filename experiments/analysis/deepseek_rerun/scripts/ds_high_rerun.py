"""Extra arm: DeepSeek-V4-Flash no-RAG essays at reasoning_effort='max'.

Identical to the published no-RAG run in every respect except the effort knob:
same qa.AnswerGenerator, same QA_USER prompt, same empty context, same tasks.
The published run uses src/llm.py's default 'high'; AA scores the 0731 checkpoint
only at 'max', which is the mismatch this arm sizes.

Generation only -- judging is a separate step, so nothing published is touched.
"""
import os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(ROOT / '.env', override=True)

# endpoints.yaml expects nhr_fau_api / openrouter_api; this checkout's .env uses
# *_FAU / *_OR. Same aliasing the analysis scripts do (oracle_experiment.api_keys).
if 'API_KEY_OR' in os.environ:
    os.environ['openrouter_api'] = os.environ['API_KEY_OR']
for env_name, yaml_name in (('API_KEY_UP', 'innkube_api'), ('API_KEY_FAU', 'nhr_fau_api')):
    if env_name in os.environ:
        os.environ[yaml_name] = os.environ[env_name]

from src import qa

MODEL = 'deepseek-ai/DeepSeek-V4-Flash'
EFFORT = 'high'
OUT = 'experiments/zubaers_result/essay_writing/without_rag_0731/ds_v4_flash_0731_high.csv'

gpbam = pd.read_json('experiments/data/gpbam.json')
tasks = gpbam.facts.values
print(f'{MODEL} @ effort={EFFORT}  n={len(tasks)}', flush=True)

gen = qa.AnswerGenerator(model=MODEL, reasoning_effort=EFFORT,
                         max_concurrency=4, timeout=3600, max_retries=8)
print(f'endpoint={gen.inference_endpoint}  resolved_model={gen.model}  '
      f'effort={gen.reasoning_effort}  reasoning={gen.reasoning}', flush=True)

t0 = time.time()
answers, info = gen.predict(tasks, return_raw=True)
el = time.time() - t0

rows = []
for i, (a, inf) in enumerate(zip(answers, info)):
    r = {'index': i, 'model': MODEL, 'reasoning_effort': EFFORT,
         'answer': a, 'chars': len(a) if a else 0}
    if inf:
        for k in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'total_cost',
                  'finish_reason', 'time'):
            r[k] = inf.get(k)
    rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
ok = df.answer.notna().sum()
print(f'\ndone in {el/60:.1f} min -> {OUT}')
print(f'{ok}/{len(df)} generated, mean {df.chars.mean():.0f} chars, '
      f'median {df.chars.median():.0f}, min {df.chars.min()}, max {df.chars.max()}')
print(f'completion tokens: mean {df.completion_tokens.mean():.0f}' if df.completion_tokens.notna().any() else '')

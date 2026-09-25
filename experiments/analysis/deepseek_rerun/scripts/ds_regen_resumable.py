"""DeepSeek-V4-Flash no-RAG essays on the 0731 weights, effort=high.

Resumable: one row appended per essay as it completes, so a crash or a kill costs
at most the in-flight items. Retries per essay on the gateway's 600s LiteLLM
timeout (which is tighter than the ~1800s proxy cap).
"""
import os, sys, time, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, openai
from dotenv import load_dotenv
load_dotenv(ROOT / '.env', override=True)
from src.prompts import QA_USER

OUT = 'experiments/zubaers_result/essay_writing/without_rag_0731/ds_v4_flash_0731_high.jsonl'
MODEL, EFFORT, WORKERS = 'deepseek-ai/DeepSeek-V4-Flash', 'high', 3
os.makedirs(os.path.dirname(OUT), exist_ok=True)
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: done.add(json.loads(l)['index'])
        except Exception: pass
print(f'{MODEL} effort={EFFORT} workers={WORKERS}; {len(done)} already done', flush=True)

gpbam = pd.read_json('experiments/data/gpbam.json')
client = openai.OpenAI(api_key=os.environ['API_KEY_FAU'],
                       base_url='https://hub.nhr.fau.de/api/llmgw/v1', timeout=900)
todo = [i for i in range(len(gpbam)) if i not in done]
import threading; lock = threading.Lock()

def one(i):
    msgs = [{'role': 'system', 'content': ''},
            {'role': 'user', 'content': QA_USER.format('', gpbam.facts.values[i])}]
    for a in range(6):
        try:
            t0 = time.time()
            r = client.chat.completions.create(model=MODEL, messages=msgs, reasoning_effort=EFFORT)
            m = r.choices[0].message
            rec = dict(index=i, model=MODEL, reasoning_effort=EFFORT, answer=m.content,
                       chars=len(m.content or ''), prompt_tokens=r.usage.prompt_tokens,
                       completion_tokens=r.usage.completion_tokens,
                       seconds=round(time.time()-t0,1), finish_reason=r.choices[0].finish_reason,
                       attempt=a+1, error=None)
            with lock:
                with open(OUT,'a') as f: f.write(json.dumps(rec, ensure_ascii=False)+'\n')
            print(f'  {i:>3} ok {rec["seconds"]:6.1f}s {rec["chars"]:>6}ch attempt {a+1}', flush=True)
            return
        except Exception as e:
            if a == 5:
                with lock:
                    with open(OUT,'a') as f:
                        f.write(json.dumps(dict(index=i, model=MODEL, answer=None, chars=0,
                                error=f'{type(e).__name__}: {e}'[:300]), ensure_ascii=False)+'\n')
                print(f'  {i:>3} FAILED: {type(e).__name__}', flush=True); return
            time.sleep(5*(a+1))

with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    list(ex.map(one, todo))
rows=[json.loads(l) for l in open(OUT)]
df=pd.DataFrame(rows).drop_duplicates('index').sort_values('index')
df.to_csv(OUT.replace('.jsonl','.csv'), index=False)
ok=df.answer.notna().sum()
print(f'\ndone: {ok}/{len(df)} generated, mean {df[df.answer.notna()].chars.mean():.0f} chars', flush=True)

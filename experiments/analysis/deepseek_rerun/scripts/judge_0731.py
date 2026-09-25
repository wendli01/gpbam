"""Judge the 0731 re-generated essays with the same three seats, so the table can
carry both rows: DeepSeek-V4-Flash (published, retired checkpoint) and -0731.

Waits for the re-generation to finish, then builds the ji2 prompt exactly as
src/scoring.Judge does: prompt.format(task, solution, answer).
"""
import json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import pandas as pd, openai
from dotenv import load_dotenv


sys.path.insert(0, str(Path(__file__).resolve().parent))
from repo_env import dotenv_path  # noqa: E402

load_dotenv(dotenv_path(ROOT), override=True)
from src.prompts import build_judge_user, JUDGE_INSTRUCTION_V2, JUDGE_SYSTEM

# Parameterised so any generation can be judged by the same three seats -- the
# 0731 re-generation it was written for stays the default.
GEN = os.environ.get(
    'JUDGE_GEN',
    'experiments/zubaers_result/essay_writing/without_rag_0731/ds_v4_flash_0731_high.jsonl')
OUTD = os.environ.get(
    'JUDGE_OUTD', 'experiments/zubaers_result/essay_writing/without_rag_0731/judged')
WAIT = os.environ.get('JUDGE_WAIT', '1') == '1'   # 0 to fail fast if the run is short
SEATS=[('gpt-oss-120b','gpt-oss-120b'),
       ('DeepSeek-V4-Flash-0731','deepseek-ai/DeepSeek-V4-Flash'),
       ('Qwen3.6-35B','Qwen/Qwen3.6-35B-A3B-FP8')]
JU=build_judge_user(JUDGE_INSTRUCTION_V2); SCORE=re.compile(r"\[\[(\d+(?:[.,]\d+)?)\]\]")
os.makedirs(OUTD, exist_ok=True)
client=openai.OpenAI(api_key=os.environ['API_KEY_FAU'],
                     base_url='https://hub.nhr.fau.de/api/llmgw/v1', timeout=900)
gpbam=pd.read_json('experiments/data/gpbam.json')

print(f'waiting for {GEN} to reach 81 essays...', flush=True)
while True:
    n = sum(1 for _ in open(GEN)) if os.path.exists(GEN) else 0
    if n >= 81 or not WAIT:
        break
    time.sleep(120)
gen = {json.loads(l)['index']: json.loads(l) for l in open(GEN)}
gen = {k: v for k, v in gen.items() if v.get('answer')}
print(f'generation complete: {len(gen)}/81 usable essays', flush=True)

lock=threading.Lock()
for label, model in SEATS:
    out=f'{OUTD}/{label}.jsonl'
    done={json.loads(l)['index'] for l in open(out)} if os.path.exists(out) else set()
    todo=[i for i in sorted(gen) if i not in done]
    print(f'== seat {label}: {len(todo)} to judge', flush=True)
    def one(i, model=model, out=out):
        msgs=[{'role':'system','content':JUDGE_SYSTEM},
              {'role':'user','content':JU.format(gpbam.facts.values[i], gpbam.solutions.values[i], gen[i]['answer'])}]
        for a in range(5):
            try:
                r=client.chat.completions.create(model=model, messages=msgs, reasoning_effort='high')
                txt=r.choices[0].message.content or ''; h=SCORE.findall(txt)
                rec=dict(index=i, seat=label, score=float(h[-1].replace(',','.')) if h else None,
                         judgement=txt, prompt_tokens=r.usage.prompt_tokens,
                         completion_tokens=r.usage.completion_tokens, error=None)
                with lock, open(out,'a') as f: f.write(json.dumps(rec, ensure_ascii=False)+'\n')
                return
            except Exception as e:
                if a==4:
                    with lock, open(out,'a') as f:
                        f.write(json.dumps(dict(index=i, seat=label, score=None,
                                error=f'{type(e).__name__}: {e}'[:200]), ensure_ascii=False)+'\n')
                    return
                time.sleep(5*(a+1))
    with ThreadPoolExecutor(max_workers=4) as ex: list(ex.map(one, todo))
    d=pd.DataFrame([json.loads(l) for l in open(out)]).drop_duplicates('index')
    print(f'   {label}: {d.score.notna().sum()}/{len(d)} parsed, mean {d.score.mean()*100:.2f}', flush=True)
print('ALL SEATS DONE', flush=True)

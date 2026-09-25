"""Cost and verbosity pilot for a frontier generator, before committing to a full run.

The price of adding a model is dominated by completion tokens, not by the headline
rate, and completion tokens are the part that cannot be looked up: across the
published run they span 1,091 (EuroLLM) to 20,195 (gpt-5-mini) per essay, and
DeepSeek's own re-generation produced 75% more than its published one on identical
prompts. A handful of real essays pins that down for a few cents.

Runs the actual no-RAG prompt on the actual cases, so the tokens measured are the
tokens a full run would bill. Reports observed cost per essay, and extrapolates to
81 essays plus the 200-item recitation sweep at the same verbosity.

    PILOT_MODEL=google/gemini-3.7-flash PILOT_N=5 python frontier_pilot.py
"""
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import openai                                                       # noqa: E402
import pandas as pd                                                 # noqa: E402
from dotenv import load_dotenv                                      # noqa: E402

# .env is untracked, so it exists only in the primary checkout. Inside a linked
# worktree ROOT/.env does not exist, and the failure is a KeyError on the API key
# thirty lines later rather than anything that names the cause.
def _dotenv():
    here = ROOT / '.env'
    if here.exists():
        return here
    common = subprocess.run(['git', 'rev-parse', '--path-format=absolute',
                             '--git-common-dir'], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    main = Path(common).parent / '.env' if common else None
    if main and main.exists():
        print(f'note: .env from the primary checkout, {main}')
        return main
    raise SystemExit(f'no .env at {here} and none in the primary checkout')


load_dotenv(_dotenv(), override=True)
from src.prompts import AR_SYSTEM, AR_USER, QA_USER                 # noqa: E402

MODEL = os.environ.get('PILOT_MODEL', 'google/gemini-3.7-flash')
N = int(os.environ.get('PILOT_N', '5'))
N_REC = int(os.environ.get('PILOT_N_REC', '5'))
EFFORT = os.environ.get('PILOT_EFFORT', 'high')
# 3 concurrent requests to Google through OpenRouter produced finish_reason
# ='error' on 8 of 10 calls; 1 is reliable and a pilot is 10 requests.
WORKERS = int(os.environ.get('PILOT_WORKERS', '1'))
OUT = ROOT / f'experiments/analysis/deepseek_rerun/results/pilot_{MODEL.split("/")[-1]}.jsonl'

# OpenRouter list price, USD per million tokens, as of 2026-08-20.
PRICE = {
    'google/gemini-3.7-flash': (0.375, 1.875),
    'z-ai/glm-5.3':            (1.40, 4.40),
    'openai/gpt-5.6-sol':      (2.50, 15.00),
    'anthropic/claude-fable-5': (10.00, 50.00),
}

client = openai.OpenAI(api_key=os.environ['API_KEY_OR'],
                       base_url='https://openrouter.ai/api/v1', timeout=900)
gpbam = pd.read_json('experiments/data/gpbam.json')
rec_src = pd.read_csv('experiments/zubaers_result/article_recitation/article_recitation.csv',
                      usecols=['model', 'query', 'dataset'])
rec_src = rec_src[rec_src.model == 'deepseek-ai/DeepSeek-V4-Flash'].reset_index(drop=True)

lock = threading.Lock()
rows = []


def call(kind, i, msgs):
    for attempt in range(4):
        try:
            t0 = time.time()
            kw = {}
            if EFFORT:
                kw['reasoning_effort'] = EFFORT
            r = client.chat.completions.create(
                model=MODEL, messages=msgs,
                extra_body={'usage': {'include': True}}, **kw)
            u, m = r.usage, r.choices[0].message
            # OpenRouter reports an upstream failure as HTTP 200 with
            # finish_reason='error', partial content and zero usage. Taken at face
            # value that is a successful call costing nothing, which is how a
            # 3-worker pilot came back claiming this model bills $0 per essay.
            if r.choices[0].finish_reason == 'error' or not u.completion_tokens:
                raise RuntimeError(
                    f'upstream error: finish_reason={r.choices[0].finish_reason}, '
                    f'completion_tokens={u.completion_tokens}, chars={len(m.content or "")}')
            rec = dict(kind=kind, i=i, model=MODEL, effort=EFFORT,
                       chars=len(m.content or ''), prompt_tokens=u.prompt_tokens,
                       completion_tokens=u.completion_tokens,
                       reported_cost=getattr(u, 'cost', None),
                       reasoning_tokens=getattr(getattr(u, 'completion_tokens_details', None),
                                                'reasoning_tokens', None),
                       provider=getattr(r, 'provider', None),
                       seconds=round(time.time() - t0, 1),
                       finish_reason=r.choices[0].finish_reason, attempt=attempt + 1,
                       error=None)
            with lock:
                rows.append(rec)
                with open(OUT, 'a') as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            print(f'  {kind:10s} {i:>3} ok {rec["seconds"]:6.1f}s  '
                  f'{u.prompt_tokens:>6}p {u.completion_tokens:>6}c  {rec["chars"]:>6}ch',
                  flush=True)
            return
        except Exception as e:
            if attempt == 3:
                print(f'  {kind:10s} {i:>3} FAILED {type(e).__name__}: {e}'[:200], flush=True)
                with lock:
                    rows.append(dict(kind=kind, i=i, error=f'{type(e).__name__}: {e}'[:200],
                                     prompt_tokens=None, completion_tokens=None))
                return
            time.sleep(10 * (attempt + 1))


print(f'{MODEL} effort={EFFORT}: {N} essays + {N_REC} recitation items', flush=True)
jobs = [('essay', i, [{'role': 'system', 'content': ''},
                      {'role': 'user', 'content': QA_USER.format('', gpbam.facts.values[i])}])
        for i in range(N)]
jobs += [('recitation', i, [{'role': 'system', 'content': AR_SYSTEM},
                            {'role': 'user', 'content': AR_USER.format('', rec_src['query'][i])}])
         for i in range(N_REC)]
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    list(ex.map(lambda a: call(*a), jobs))

d = pd.DataFrame(rows)
ok = d[d.error.isna()] if 'error' in d else d
if ok.empty:
    sys.exit('every call failed')
p_in, p_out = PRICE.get(MODEL, (float('nan'),) * 2)
print(f'\n=== {MODEL} @ {EFFORT}   list price {p_in}/{p_out} per M tokens')
for kind, n_full in (('essay', 81), ('recitation', 200)):
    s = ok[ok.kind == kind]
    if s.empty:
        continue
    per = (s.prompt_tokens * p_in + s.completion_tokens * p_out).mean() / 1e6
    print(f'  {kind:10s} n={len(s):<3} prompt {s.prompt_tokens.mean():8.0f}  '
          f'completion {s.completion_tokens.mean():8.0f}  ${per:.4f}/item  '
          f'-> ${per * n_full:7.2f} for {n_full}')
    if s.reported_cost.notna().any():
        print(f'{"":13s}OpenRouter-reported: ${s.reported_cost.mean():.5f}/item')
tot = sum((ok[ok.kind == k].prompt_tokens * p_in + ok[ok.kind == k].completion_tokens * p_out
           ).mean() / 1e6 * n for k, n in (('essay', 81), ('recitation', 200))
          if not ok[ok.kind == k].empty)
print(f'  {"TOTAL":10s} full benchmark at this verbosity: ${tot:.2f}')
print(f'\nwrote {OUT}')

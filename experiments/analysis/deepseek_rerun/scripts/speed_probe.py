"""Measure a model's real throughput on GPBam's own prompts, then project a full run.

A tokens/second number from a toy prompt does not project: GPBam essay prompts are
~9k characters and the answers are long, so the run is decode-bound in a way a
one-line probe never shows. This sends the actual prompts, at the concurrency the
full run would use, and reports what the full run would therefore cost in wall time.

    RUN_MODEL=soofi-s-isar-preview RUN_ENDPOINT=innkube python speed_probe.py

  RUN_MODEL      model id as the endpoint names it
  RUN_ENDPOINT   innkube | fau | openrouter                    default innkube
  RUN_EFFORT     reasoning_effort, or empty to omit the field  default '' (omit)
  PROBE_ESSAYS   how many essay prompts to time                default 3
  PROBE_RECIT    how many recitation prompts to time           default 6
  PROBE_WORKERS  comma-separated concurrency levels to compare default 1,4
  PROBE_OFFSET   first item index to time                      default 0

The gateway caches an identical request, so re-probing the same indices returns
in milliseconds and measures the cache. Each concurrency level therefore gets a
fresh slice of items, and PROBE_OFFSET moves the whole window when the script is
run twice against the same endpoint.

Nothing is written to the result tree: this is a measurement of the endpoint, not
a partial run, and a partial run left where a full one belongs gets reused.
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import json                                                          # noqa: E402
import openai                                                        # noqa: E402
from dotenv import load_dotenv                                       # noqa: E402


def _dotenv():
    here = ROOT / '.env'
    if here.exists():
        return here
    common = subprocess.run(['git', 'rev-parse', '--path-format=absolute',
                             '--git-common-dir'], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    main = Path(common).parent / '.env' if common else None
    if main and main.exists():
        return main
    raise SystemExit('no .env found')


load_dotenv(_dotenv(), override=True)
from src.prompts import AR_SYSTEM, AR_USER, QA_USER                  # noqa: E402

MODEL = os.environ['RUN_MODEL']
ENDPOINT = os.environ.get('RUN_ENDPOINT', 'innkube')
EFFORT = os.environ.get('RUN_EFFORT', '')
N_ESSAY = int(os.environ.get('PROBE_ESSAYS', '3'))
N_RECIT = int(os.environ.get('PROBE_RECIT', '6'))
LEVELS = [int(w) for w in os.environ.get('PROBE_WORKERS', '1,4').split(',')]
OFFSET = int(os.environ.get('PROBE_OFFSET', '0'))

ENDPOINTS = {'openrouter': ('API_KEY_OR', 'https://openrouter.ai/api/v1'),
             'fau': ('API_KEY_FAU', 'https://hub.nhr.fau.de/api/llmgw/v1'),
             'innkube': ('API_KEY_UP', 'https://llms.innkube.fim.uni-passau.de/v1')}
key_env, base_url = ENDPOINTS[ENDPOINT]
client = openai.OpenAI(api_key=os.environ[key_env], base_url=base_url, timeout=1800)

import pandas as pd                                                # noqa: E402

# Exactly the item sets run_model.py uses, so the per-item timing projects onto
# the run it is projecting onto.
FACTS = pd.read_json(ROOT / 'experiments/data/gpbam.json').facts.values
REC = pd.read_csv(ROOT / 'experiments/zubaers_result/article_recitation/article_recitation.csv',
                  usecols=['model', 'query'])
REC = REC[REC.model == 'deepseek-ai/DeepSeek-V4-Flash'].reset_index(drop=True)

TOTAL = {'essay': len(FACTS), 'recitation': len(REC)}


def essay_msgs(i):
    return [{'role': 'system', 'content': ''},
            {'role': 'user', 'content': QA_USER.format('', FACTS[i])}]


def recit_msgs(i):
    return [{'role': 'system', 'content': AR_SYSTEM},
            {'role': 'user', 'content': AR_USER.format('', REC['query'][i])}]


def one(msgs):
    kw = {'reasoning_effort': EFFORT} if EFFORT else {}
    t0 = time.time()
    r = client.chat.completions.create(model=MODEL, messages=msgs, **kw)
    u = r.usage
    txt = r.choices[0].message.content or ''
    # A reasoning model whose server has no reasoning parser leaves the trace in
    # `content`, closed by </think>. Measure the answer, not the monologue.
    answer = txt.split('</think>')[-1] if '</think>' in txt else txt
    return dict(seconds=time.time() - t0, prompt_tokens=u.prompt_tokens,
                completion_tokens=u.completion_tokens, chars=len(txt),
                answer_chars=len(answer.strip()),
                thinking_chars=len(txt) - len(answer),
                finish=r.choices[0].finish_reason)


def probe(kind, build, n, workers, offset):
    # A fresh slice of items per concurrency level. The gateway caches an
    # identical request, so re-timing the same indices at the next level
    # returned 4 essays in 0.1s -- a measurement of the cache, not the model.
    items = [(offset + i) % TOTAL[kind] for i in range(n)]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(lambda i: one(build(i)), items))
    wall = time.time() - t0
    ct = sum(r['completion_tokens'] for r in rows)
    per_item = wall / n
    print(f'  {kind:11s} w={workers}  {n} items in {wall:6.1f}s  '
          f'{per_item:6.1f}s/item  {ct / wall:6.1f} tok/s aggregate  '
          f'{ct / n:7.0f} completion tok/item  '
          f'{sum(r["answer_chars"] for r in rows) / n:7.0f} answer chars')
    if any(r['thinking_chars'] for r in rows):
        share = sum(r['thinking_chars'] for r in rows) / sum(r['chars'] for r in rows)
        print(f'              thinking trace is in `content`: {share:.0%} of characters')
    bad = [r['finish'] for r in rows if r['finish'] != 'stop']
    if bad:
        print(f'              !! finish_reason not stop on {len(bad)}: {set(bad)}')
    return per_item


print(f'{MODEL} @ {ENDPOINT}  effort={EFFORT or "(omitted)"}', flush=True)
best = {}
offset = {'essay': OFFSET, 'recitation': OFFSET}
for w in LEVELS:
    print(f'-- concurrency {w}', flush=True)
    for kind, build, n in [('essay', essay_msgs, N_ESSAY), ('recitation', recit_msgs, N_RECIT)]:
        if n:
            best[(kind, w)] = probe(kind, build, n, w, offset[kind])
            offset[kind] += n

print('\n=== projected wall time for the full run')
for w in LEVELS:
    total = sum(best[(k, w)] * TOTAL[k] for k in TOTAL if (k, w) in best)
    parts = ' + '.join(f'{k} {best[(k, w)] * TOTAL[k] / 60:.0f}min'
                       for k in TOTAL if (k, w) in best)
    print(f'  w={w}: {parts} = {total / 60:.0f} min ({total / 3600:.1f} h)')
print('\nProjection assumes the probe items are typical and the endpoint stays as '
      'loaded as it is now. Both are optimistic; treat as a lower bound.')

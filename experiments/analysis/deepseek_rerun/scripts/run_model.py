"""Run one generator over both GPBam tasks: 81 no-RAG essays and 200 recitation items.

Env-parameterised so a new model needs no new script, and resumable at item
granularity -- one JSONL row appended per completion, so a crash, a kill or a
rate-limit storm costs at most the in-flight items.

    RUN_MODEL=google/gemini-3.7-flash RUN_TASKS=essay,recitation python run_model.py

  RUN_MODEL     model id as the endpoint names it
  RUN_ENDPOINT  'openrouter' (billed), 'fau' or 'innkube'      default openrouter
  RUN_EFFORT    reasoning_effort, or empty to omit the field   default high
  RUN_WORKERS   concurrent requests                            default 2
  RUN_TASKS     comma-separated subset of essay,recitation     default both
  RUN_SLUG      output filename stem                           default the model's last path segment
  RUN_PREFIX    '', 'high' or 'max': DeepSeek's own effort prefix, prepended to the system
                message (see EFFORT_PREFIX)                    default ''
  RUN_REC_DIR   directory for the recitation output            default article_recitation/
  RUN_ESSAY_DIR directory for the essay output                 default essay_writing/frontier/
  RUN_TIMEOUT   per-request timeout in seconds                  default 900

OpenRouter reports an upstream failure as HTTP 200 with finish_reason='error',
partial content and zero usage. That is indistinguishable from a cheap success
unless it is checked for, so it is, and it retries. Three concurrent requests to
Google produced it on 8 of 10 calls; two is the default for that reason.

Thinking traces
---------------
A reasoning model served without a reasoning parser puts its chain of thought in
`content` rather than in a separate field, closed by `</think>` -- the opening tag
is eaten by the chat template, so there is nothing to match on but the close.
`soofi-s-isar-preview` on InnKube does exactly this, and the trace is 41-57% of an
essay's characters and over 90% of a recitation answer's. Left alone it would be
judged as part of the essay and scanned for citations. So `answer` is the text
after the last `</think>`, and the trace is kept beside it in `thinking` -- kept
rather than discarded, because it is the only record that the split happened.
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
import openai                                                        # noqa: E402
import pandas as pd                                                  # noqa: E402
from dotenv import load_dotenv                                       # noqa: E402


def _dotenv():
    """.env is untracked, so in a linked worktree it lives in the primary checkout."""
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
ENDPOINT = os.environ.get('RUN_ENDPOINT', 'openrouter')
EFFORT = os.environ.get('RUN_EFFORT', 'high')
WORKERS = int(os.environ.get('RUN_WORKERS', '2'))
TASKS = [t.strip() for t in os.environ.get('RUN_TASKS', 'essay,recitation').split(',') if t.strip()]
SLUG = os.environ.get('RUN_SLUG', MODEL.split('/')[-1])

PREFIX = os.environ.get('RUN_PREFIX', '')
TIMEOUT = float(os.environ.get('RUN_TIMEOUT', '900'))

#: DeepSeek-V4's reasoning effort is only a text prefix at the very start of the
#: prompt (encoding/encoding_dsv4.py in the Hugging Face repo, REASONING_EFFORT_PROMPTS;
#: `low` adds nothing, and the system template is plain "{content}", so prefix +
#: system text in one system message is the encoding verbatim). GWDG, and NHR@FAU
#: in August, send `high` without it; RUN_EFFORT=low RUN_PREFIX=high is DeepSeek's
#: real `high` there. See DS4-ablation-rerun.md, "Reasoning budget".
EFFORT_PREFIX = {
    'high': ("Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
             "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
             "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n"),
    'max': ("Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n"
            "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\n"
            "Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n"),
}

ENDPOINTS = {'openrouter': ('API_KEY_OR', 'https://openrouter.ai/api/v1'),
             'fau': ('API_KEY_FAU', 'https://hub.nhr.fau.de/api/llmgw/v1'),
             'innkube': ('API_KEY_UP', 'https://llms.innkube.fim.uni-passau.de/v1'),
             # GWDG academic cloud: 30/min, 200/h, 1000/day per key, every request counted
             'gwdg': ('API_KEY_AC', os.environ.get('ENDPOINT_AC', 'https://chat-ai.academiccloud.de/v1'))}
key_env, base_url = ENDPOINTS[ENDPOINT]
client = openai.OpenAI(api_key=os.environ[key_env], base_url=base_url, timeout=TIMEOUT, max_retries=0)

ESSAY_OUT = (Path(os.environ['RUN_ESSAY_DIR']) if os.environ.get('RUN_ESSAY_DIR')
             else ROOT / 'experiments/zubaers_result/essay_writing/frontier') / f'{SLUG}_essays.jsonl'
REC_OUT = (Path(os.environ['RUN_REC_DIR']) if os.environ.get('RUN_REC_DIR')
           else ROOT / 'experiments/zubaers_result/article_recitation') / f'recitation_{SLUG}.jsonl'
lock = threading.Lock()


def load_done(path):
    if not path.exists():
        return set()
    out = set()
    for line in open(path):
        try:
            r = json.loads(line)
            if r.get('error') is None:
                out.add(r['index'])
        except Exception:
            pass
    return out


def call(msgs, tries=5):
    last = None
    attempt, waited = 0, 0
    while attempt < tries:
        try:
            kw = {'reasoning_effort': EFFORT} if EFFORT else {}
            t0 = time.time()
            r = client.chat.completions.create(
                model=MODEL, messages=msgs, extra_body={'usage': {'include': True}}, **kw)
            u, m = r.usage, r.choices[0].message
            if r.choices[0].finish_reason == 'error' or not u.completion_tokens:
                raise RuntimeError(f'upstream error: finish_reason='
                                   f'{r.choices[0].finish_reason}, tokens={u.completion_tokens}')
            text = m.content or ''
            thinking, _, tail = text.rpartition('</think>')
            answer = tail.strip() if thinking else text
            return dict(answer=answer, thinking=thinking or None,
                        prompt_tokens=u.prompt_tokens,
                        completion_tokens=u.completion_tokens,
                        reasoning_tokens=getattr(getattr(u, 'completion_tokens_details', None),
                                                 'reasoning_tokens', None),
                        reported_cost=getattr(u, 'cost', None),
                        finish_reason=r.choices[0].finish_reason,
                        seconds=round(time.time() - t0, 1), attempt=attempt + 1, error=None)
        except Exception as e:
            last = f'{type(e).__name__}: {e}'[:200]
            # a rate limit (GWDG's per-hour cap) is waited out, not counted as a try
            if getattr(e, 'status_code', None) == 429 and waited < 7200:
                time.sleep(60); waited += 60
                continue
            attempt += 1
            if attempt < tries:
                time.sleep(10 * attempt)
    return dict(answer=None, thinking=None, prompt_tokens=None, completion_tokens=None,
                reasoning_tokens=None,
                reported_cost=None, finish_reason=None, seconds=None, attempt=tries, error=last)


def run(kind, items, out_path, build):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    todo = [i for i in range(len(items)) if i not in done]
    print(f'== {kind}: {len(todo)} to do, {len(done)} already done -> {out_path.name}', flush=True)

    def one(i):
        rec = dict(index=i, model=MODEL, kind=kind, reasoning_effort=EFFORT, effort_prefix=PREFIX or None,
                   endpoint=ENDPOINT, **build(i))
        msgs = rec.pop('msgs')
        if PREFIX:
            assert msgs[0]['role'] == 'system'
            msgs = [{'role': 'system', 'content': EFFORT_PREFIX[PREFIX] + msgs[0]['content']}] + msgs[1:]
        r = call(msgs)
        rec.update(r, chars=len(r['answer'] or ''))
        with lock, open(out_path, 'a') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        flag = 'ok ' if r['error'] is None else 'ERR'
        print(f'  {kind[:5]} {i:>3} {flag} {str(r["seconds"]):>6}s '
              f'{str(r["completion_tokens"]):>6}c {rec["chars"]:>6}ch a{r["attempt"]}', flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(one, todo))
    n_ok = len(load_done(out_path))
    print(f'== {kind}: {n_ok}/{len(items)} complete', flush=True)
    return n_ok


print(f'{MODEL} @ {ENDPOINT} effort={EFFORT} prefix={PREFIX or None} workers={WORKERS} tasks={TASKS} -> {REC_OUT}', flush=True)

if 'essay' in TASKS:
    gpbam = pd.read_json('experiments/data/gpbam.json')
    run('essay', gpbam.facts.values, ESSAY_OUT,
        lambda i: dict(msgs=[{'role': 'system', 'content': ''},
                             {'role': 'user',
                              'content': QA_USER.format('', gpbam.facts.values[i])}]))

if 'recitation' in TASKS:
    src = pd.read_csv('experiments/zubaers_result/article_recitation/article_recitation.csv',
                      usecols=['model', 'query', 'target', 'dataset'])
    src = src[src.model == 'deepseek-ai/DeepSeek-V4-Flash'].reset_index(drop=True)
    run('recitation', src, REC_OUT,
        lambda i: dict(query=src['query'][i], target=src['target'][i], dataset=src['dataset'][i],
                       msgs=[{'role': 'system', 'content': AR_SYSTEM},
                             {'role': 'user', 'content': AR_USER.format('', src['query'][i])}]))

    # score with the same ROUGE-L F1 the published sweep used
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    # keep='last': a resumed run appends a fresh row for every index that
    # errored, so the retry is the LAST row for that index. Keeping the first
    # would keep the original failure and then drop it as an error row, which
    # silently discards every successful retry.
    d = pd.DataFrame([json.loads(l) for l in open(REC_OUT)]).drop_duplicates('index', keep='last')
    d = d[d.error.isna()]
    d['score'] = [sc.score(str(t), str(a))['rougeL'].fmeasure for t, a in zip(d.target, d.answer)]
    d['model'] = MODEL
    d.to_csv(str(REC_OUT).replace('.jsonl', '.csv'), index=False)
    print(d.groupby('dataset').score.mean().mul(100).round(2).to_string(), flush=True)

print('ALL TASKS DONE', flush=True)

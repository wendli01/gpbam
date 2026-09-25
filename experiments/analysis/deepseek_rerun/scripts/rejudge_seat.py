"""Re-judge the published no-RAG essays with gpt-oss-120b (free seat, NHR@FAU).

Sends the byte-identical ji2 message pair from the ``judge_prompt`` column, so the
new seat is directly comparable to nano / Qwen3.6 / DeepSeek-V4-Flash.

Models are processed best-first on the existing panel median. The seat writes ONE
file -- ``results/<seat>.jsonl``, appended a model at a time -- because a seat is
one experiment, and a directory of thirty per-model CSVs is thirty files in every
diff for no analytical gain. Resumption reads back the (model, index) pairs the
file already holds, so a stopped run picks up exactly where it left off, and a
line torn by a kill is simply re-judged.

  python rejudge.py extract   # one pass over the 909MB CSV -> per-model prompt pickles
  python rejudge.py judge     # judge in ranked order, skipping finished models
"""
import ast, os, pickle, re, sys, time
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

import json
import numpy as np
import pandas as pd
import openai
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seats import done_pairs, seat_path            # noqa: E402
from repo_env import dotenv_path                   # noqa: E402

# Derived, not hardcoded: an absolute path here resolves against whatever
# checkout happens to live there rather than the one this file is in.
ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / 'experiments/zubaers_result/essay_writing/without_rag/ji2/no_rag_ji2_result.csv'
# The re-judging run wrote its seats here. They are vendored into the repo under
# results/<seat>/, which is what every analysis script reads, so this default lets a
# re-run land beside them; point REJUDGE_DIR elsewhere to keep a run separate.
# `prompts/` holds the per-model judge prompts pickled out of SRC and is not vendored
# -- it is regenerated from SRC's judge_prompt column by rejudge.py.
D = Path(os.environ.get('REJUDGE_DIR',
                        ROOT / 'experiments/analysis/deepseek_rerun/results'))
PROMPTS = D / 'prompts'
JC = ['score_Judge (gpt-5-nano)', 'score_Judge (Qwen3.6-35B-A3B-FP8)',
      'score_Judge (qwen3.6-35b-a3b)', 'score_Judge (DeepSeek-V4-Flash)']
SCORE = re.compile(r"\[\[(\d+(?:[.,]\d+)?)\]\]")
import argparse
MODEL = os.environ['SEAT']
EFFORT = os.environ.get('SEAT_EFFORT', 'high')
WORKERS = int(os.environ.get('SEAT_WORKERS', '6'))
# SEAT_OUT names the seat, not a directory: results/<SEAT_OUT>.jsonl.
OUT = seat_path(os.environ['SEAT_OUT'], results=D)
OUT.parent.mkdir(parents=True, exist_ok=True)
slug = lambda m: m.replace('/', '_')


def extract():
    parts = []
    for ch in pd.read_csv(SRC, usecols=['index', 'model', 'judge_prompt'] + JC, chunksize=250):
        parts.append(ch)
    df = pd.concat(parts, ignore_index=True)
    df['qwen'] = df[JC[1]].combine_first(df[JC[2]])
    df['panel_med'] = df[[JC[0], 'qwen', JC[3]]].median(axis=1, skipna=True) * 100
    rank = df.groupby('model').panel_med.mean().sort_values(ascending=False)
    rank.to_csv(D / 'model_rank.csv', header=['panel_mean'])
    for m, g in df.groupby('model'):
        recs = [(int(r['index']), r['judge_prompt']) for _, r in g.iterrows()]
        with open(PROMPTS / f'{slug(m)}.pkl', 'wb') as f:
            pickle.dump(recs, f)
    print(f'extracted {len(df)} rows, {df.model.nunique()} models')
    print(rank.round(2).to_string())


def judge():
    print(f'SEAT={MODEL} effort={EFFORT} workers={WORKERS} out={OUT}', flush=True)
    load_dotenv(dotenv_path(ROOT))
    client = openai.OpenAI(api_key=os.environ['API_KEY_FAU'],
                           base_url='https://hub.nhr.fau.de/api/llmgw/v1', timeout=1800)
    rank = pd.read_csv(D / 'model_rank.csv', index_col=0)['panel_mean']

    def one(rec):
        idx, jp = rec
        msgs = ast.literal_eval(jp)
        for attempt in range(5):
            try:
                t0 = time.time()
                r = client.chat.completions.create(model=MODEL, messages=msgs,
                                                   reasoning_effort=EFFORT)
                txt = r.choices[0].message.content or ''
                hits = SCORE.findall(txt)
                u = r.usage
                return dict(index=idx, score=float(hits[-1].replace(',', '.')) if hits else None,
                            n_matches=len(hits), judgement=txt,
                            prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
                            seconds=round(time.time() - t0, 1),
                            finish_reason=r.choices[0].finish_reason, error=None)
            except Exception as e:
                if attempt == 4:
                    return dict(index=idx, score=None, n_matches=0, judgement=None,
                                prompt_tokens=None, completion_tokens=None, seconds=None,
                                finish_reason=None, error=f'{type(e).__name__}: {e}'[:300])
                time.sleep(2 ** attempt * 4)

    done = done_pairs(os.environ['SEAT_OUT'], results=D)
    if done:
        print(f'resuming: {len(done)} (model, case) pairs already in {OUT.name}', flush=True)

    for m, mean in rank.items():
        recs = pickle.load(open(PROMPTS / f'{slug(m)}.pkl', 'rb'))
        todo = [r for r in recs if (m, int(r[0])) not in done]
        if not todo:
            print(f'skip  {m:<50} (done)', flush=True); continue
        t0 = time.time()
        print(f'START {m:<50} panel {mean:5.2f}  n={len(todo)}'
              f'{"" if len(todo) == len(recs) else f" of {len(recs)}"}', flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            rows = list(ex.map(one, todo))
        out = pd.DataFrame(rows).sort_values('index')
        out.insert(1, 'model', m)
        # Appended, not rewritten: the file is the run's own progress record, and
        # a model's block lands only once its judging finished.
        with OUT.open('a', encoding='utf-8') as fh:
            for rec in out.to_dict('records'):
                fh.write(json.dumps(
                    {k: (None if pd.isna(v) else v) for k, v in rec.items()},
                    ensure_ascii=False) + '\n')
        ok = out.score.notna().sum()
        print(f'DONE  {m:<50} {ok}/{len(out)} parsed, mean {out.score.mean()*100:5.1f}, '
              f'{out.error.notna().sum()} err, {time.time()-t0:6.0f}s', flush=True)


if __name__ == '__main__':
    {'extract': extract, 'judge': judge}[sys.argv[1]]()

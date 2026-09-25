"""Which OpenRouter providers can serve the DeepSeek oracle arms?

The unpinned ``_gold_combined`` run on OpenRouter came back broken: 14 of 40
essays claimed the Sachverhalt was missing, and the arm scored 0.22 against
0.42 for top-10 (see oracle_rag/ji2/broken_openrouter/README.md). OpenRouter
routed those requests to whichever provider it picked, and the CSV does not
record which one.

This sends the same prompts that failed -- the largest ``_gold_combined``
contexts, all of them refused or scoring 0.0-0.1 -- to each provider in turn,
pinned with ``allow_fallbacks=False``, and scores the answers with the run's
own Qwen3.6 judge on NHR@FAU. A provider that serves the model faithfully
should neither refuse nor score near zero on these.

Billed on OpenRouter: roughly $0.015 per call at 25k completion tokens, so a
dozen providers x 4 cases is about $1. Run from ``experiments/``::

    PYTHONPATH=analysis python analysis/or_provider_probe.py
"""
import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

import oracle_experiment as ox

MODEL = 'deepseek/deepseek-v4-flash-0731'
JUDGE = 'Qwen/Qwen3.6-35B-A3B-FP8'
#: the four largest refused / near-zero cases of the broken combined arm
CASES = (13, 11, 22, 25)
#: blog's GPQA/TAU leaders for V4 Flash 0731 (NextBit, SiliconFlow, Novita,
#: Alibaba), its worst (DigitalOcean), and the cheap end of the endpoint list,
#: where an unpinned request most likely landed
PROVIDERS = ('open-inference', 'deepinfra', 'streamlake', 'baseten', 'parasail',
             'together', 'morph', 'alibaba', 'siliconflow', 'nextbit', 'novita',
             'digitalocean')
REFUSAL = re.compile(
    r'mangels (?:eines )?Sachverhalt|kein(?:en)? (?:konkreten |vollständigen )?Sachverhalt'
    r'|Sachverhalt.{0,60}(?:fehlt|nicht (?:mit|über|ent)|unvollständig)|Fallmaterial'
    r'|ohne (?:den|einen|konkreten) Sachverhalt|Fall(?:schilderung|angaben).{0,40}fehl'
    r'|keine (?:konkreten )?(?:Tatsachen|Sachverhaltsangaben)')
OUT = Path('zubaers_result/probes/or_provider_probe.csv')


def pinned_generator(slug, prompt):
    from src import qa
    gen = qa.AnswerGenerator(model=MODEL, prompt=prompt, max_retries=3)
    create = gen.client.chat.completions.create
    gen.served_by = []

    def pinned(**kw):
        kw.setdefault('extra_body', {})['provider'] = {'only': [slug],
                                                       'allow_fallbacks': False}
        resp = create(**kw)
        gen.served_by.append(getattr(resp, 'provider', None))
        return resp

    gen.client.chat.completions.create = pinned
    return gen


def main(providers, case_ids, max_passages=120):
    ox.api_keys()
    from src import prompts, scoring
    facts, sols = ox.cases()
    gold, ranked = ox.gold_sets(sols)
    ret = ox.OracleRetriever(facts, gold, max_passages=max_passages, ranked=ranked)
    _, qa_rag = prompts.QA_PROMPTS_BY_NAME['qa1']
    gens = {p: pinned_generator(p, qa_rag) for p in providers}
    any_gen = next(iter(gens.values()))
    msgs = {}
    for i in case_ids:
        ctx = f'<context>\n{any_gen.format_context(ret.predict([facts[i]])[0])}\n</context>'
        msgs[i] = any_gen.build_messages(ctx, facts[i])
        assert facts[i].strip()[:200] in msgs[i][-1]['content'], f'case {i}: facts missing'

    jobs = [(p, i) for p in providers for i in case_ids]
    print(f'{len(jobs)} calls: {len(providers)} providers x {len(case_ids)} cases', flush=True)

    def call(job):
        p, i = job
        t = time.time()
        try:
            res = gens[p]._call_with_retry(msgs[i])
        except Exception as e:           # a provider that 404s the pin, say
            res, err = None, f'{type(e).__name__}: {str(e)[:120]}'
        else:
            err = None if res else 'no response'
        print(f'  {time.strftime("%T")}  {p:15} case {i:2}  '
              f'{"ok" if res else err}  {time.time() - t:5.0f}s', flush=True)
        return {'provider': p, 'index': i, 'error': err, **(res or {})}

    with ThreadPoolExecutor(len(jobs)) as ex:
        rows = list(ex.map(call, jobs))
    df = pd.DataFrame(rows)
    df['served_by'] = df.provider.map(lambda p: ','.join(sorted({str(s) for s in gens[p].served_by})))
    df['refusal'] = df.message.fillna('').str.contains(REFUSAL)
    df['tok_s'] = df.completion_tokens / df.time

    ok = df.message.notna()
    instruction = prompts.JUDGE_INSTRUCTIONS_BY_NAME['ji2']
    judge = scoring.Judge(model=JUDGE, prompt=prompts.build_judge_user(instruction))
    print(f'judging {ok.sum()} answers with {JUDGE}', flush=True)
    _, scores = judge.predict([facts[i] for i in df.index[ok].map(df['index'])],
                              list(df.message[ok]),
                              [sols[i] for i in df.index[ok].map(df['index'])],
                              return_raw=True)
    df.loc[ok, 'score'] = scores
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    summary = df.groupby('provider').agg(
        n=('index', 'size'), ok=('message', lambda s: s.notna().sum()),
        refusals=('refusal', 'sum'), score=('score', 'mean'),
        out_tok=('completion_tokens', 'mean'), tok_s=('tok_s', 'mean'),
        secs=('time', 'max'), cost=('total_cost', 'sum'),
        served_by=('served_by', 'first')).sort_values('score', ascending=False)
    print(summary.round(3).to_string())
    print(f'total billed ${df.total_cost.sum():.3f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--providers', nargs='+', default=PROVIDERS)
    ap.add_argument('--cases', nargs='+', type=int, default=CASES)
    a = ap.parse_args()
    main(a.providers, a.cases)

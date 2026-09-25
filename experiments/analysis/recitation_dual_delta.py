"""Generate the recitation answers the dual-extractor battery adds.

``analysis/build_gp_laws_dual.py`` rebuilds the *GPBam laws* set with the
``§``/``Art.`` extractor `analysis/refs.py` uses, which keeps 73 of the published
100 provisions and replaces 27 -- 24 of them GG articles the refex gazetteer made
invisible.  The 73 kept provisions already have answers from every model, so only
the 27 newcomers need generating.

This runs those 27 prompts, with the published prompt pair (``AR_SYSTEM`` /
``AR_USER``) and the published scorer (ROUGE-L F1, no stemmer), for every model
whose serving route costs nothing: NHR@FAU, InnKube and a local vLLM.  Billed
routes are listed and skipped.

Answers land in ``zubaers_result/article_recitation/dual/`` -- a subdirectory, so
the ``recitation_*.csv`` glob in ``within_model_knowledge.py`` does not pick them
up and double-count the battery before the swap is deliberate.

The scorer is verified against the published column before anything is
generated: if ROUGE-L here does not reproduce the stored scores on the kept
provisions, the new numbers would not be comparable with the old ones.

Run from ``experiments/``::

    python analysis/recitation_dual_delta.py --list      # what would run, no calls
    python analysis/recitation_dual_delta.py
    python analysis/recitation_dual_delta.py --models openai/gpt-oss-120b
"""
import argparse
import os
import sys
import time
from glob import glob
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / '.env')

# `endpoints.yaml` names the key variables one way and `.env` another; alias
# rather than edit either, so this script routes exactly as production does.
for _yaml_name, _env_name in [('nhr_fau_api', 'API_KEY_FAU'), ('innkube_api', 'API_KEY_UP'),
                              ('openrouter_api', 'API_KEY_OR'), ('VLLM_API_KEY', 'API_KEY_')]:
    if not os.getenv(_yaml_name) and os.getenv(_env_name):
        os.environ[_yaml_name] = os.environ[_env_name]
os.environ.setdefault('VLLM_API_KEY', 'EMPTY')

from rouge_score import rouge_scorer                       # noqa: E402
from src import config, prompts, qa                        # noqa: E402

AR = ROOT / 'experiments/zubaers_result/article_recitation'
OUT = AR / 'dual'
DATASET = 'GPBam Laws'
FREE_ENDPOINTS = {'nhr_fau', 'innkube', 'local_vllm'}

# Models whose published run used a route `endpoints.yaml` does not name, because
# the key there is the serving alias rather than the row label. Without these the
# resolver would fall through to OpenRouter and bill a run that was free.
ROUTE_OVERRIDES = {
    # the -0731 checkpoint is served under its own name at NHR@FAU; the yaml
    # entry points the *plain* key at it, so the explicit name resolves nowhere
    'deepseek-ai/DeepSeek-V4-Flash-0731': ('nhr_fau', 'deepseek-ai/DeepSeek-V4-Flash-0731'),
    'soofi-s-isar-preview': ('innkube', 'soofi-s-isar-preview'),
}

# Rows retired from the reported table: their recitation numbers are not read by
# anything, so regenerating them would only cost time.
RETIRED = {'deepseek-ai/DeepSeek-V4-Flash', 'mistralai/mistral-small-3.1-24b-instruct'}

_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)


def score_ar(targets, preds):
    return [_scorer.score(t, p or '')['rougeL'].fmeasure for t, p in zip(targets, preds)]


# --------------------------------------------------------------------------
def published():
    """Every published GPBam-laws answer, from the main CSV and the per-model ones."""
    frames = []
    main = pd.read_csv(AR / 'article_recitation.csv',
                       usecols=['model', 'dataset', 'query', 'target', 'answer', 'score'])
    frames.append(main[main.dataset == DATASET])
    for path in sorted(glob(str(AR / 'recitation_*.csv'))):
        d = pd.read_csv(path, usecols=lambda c: c in
                        {'model', 'dataset', 'query', 'target', 'answer', 'score'})
        frames.append(d[d.dataset == DATASET])
    return pd.concat(frames, ignore_index=True)


def verify_scorer(pub, n=400):
    """ROUGE-L here must reproduce the published column, or nothing is comparable."""
    s = pub.dropna(subset=['answer', 'target', 'score']).head(n)
    got = score_ar(s.target, s.answer)
    worst = max(abs(a - b) for a, b in zip(got, s.score))
    print(f'scorer check: {len(s)} published answers, max |delta| {worst:.2e}')
    if worst > 1e-9:
        raise SystemExit('ROUGE-L does not reproduce the published scores -- stopping.')


def route(model):
    """(endpoint name, free?) for a model, as `endpoints.yaml` resolves it."""
    if model in ROUTE_OVERRIDES:
        name = ROUTE_OVERRIDES[model][0]
        return name, name in FREE_ENDPOINTS
    cfg = config._load()
    entry = (cfg.get('models') or {}).get(model, {})
    name = entry.get('endpoint', cfg.get('default_endpoint', 'openrouter'))
    return name, name in FREE_ENDPOINTS


def reachable(endpoint_name):
    cfg = config._load()
    ep = cfg['endpoints'][endpoint_name]
    key = os.getenv(ep['token_env']) or 'EMPTY'
    try:
        r = requests.get(ep['base_url'].rstrip('/') + '/models',
                         headers={'Authorization': f'Bearer {key}'}, timeout=15)
        return r.status_code == 200
    except requests.RequestException:
        return False


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', help='restrict to these model ids')
    ap.add_argument('--list', action='store_true', help='print the plan and exit')
    ap.add_argument('--include-billed', action='store_true',
                    help='also run models on paid routes (off by default)')
    args = ap.parse_args()

    new = pd.read_csv(ROOT / 'experiments/data/gp_laws_dual.csv', dtype={'article': str})
    old = pd.read_csv(ROOT / 'experiments/data/gp_laws.csv', dtype={'article': str})
    old_keys = {(str(b).lower(), str(a).lower()) for b, a in zip(old.law_book, old.article)}
    delta = new[[(str(b).lower(), str(a).lower()) not in old_keys
                 for b, a in zip(new.law_book, new.article)]].reset_index(drop=True)
    queries = (delta.law_book + ' ' + delta.article).tolist()
    targets = delta.content.tolist()
    print(f'{len(delta)} provisions enter the battery: '
          + ', '.join(queries[:6]) + f' ... (+{len(queries) - 6})')

    pub = published()
    verify_scorer(pub)

    models = args.models or sorted(pub.model.unique())
    plan, skipped = [], []
    checked = {}
    for m in models:
        if m in RETIRED and not args.models:
            skipped.append((m, '-', 'retired from the reported table'))
            continue
        name, free = route(m)
        if not free and not args.include_billed:
            skipped.append((m, name, 'billed route'))
            continue
        if name not in checked:
            checked[name] = reachable(name)
        if not checked[name]:
            skipped.append((m, name, 'endpoint unreachable'))
            continue
        plan.append((m, name))

    print(f'\nwill run {len(plan)} models x {len(delta)} prompts = {len(plan) * len(delta)} generations')
    for m, name in plan:
        print(f'  run   {m:52s} {name}')
    for m, name, why in skipped:
        print(f'  skip  {m:52s} {name:11s} {why}')
    if args.list:
        return

    OUT.mkdir(parents=True, exist_ok=True)
    for m, name in plan:
        tag = m.split('/')[-1]
        path = OUT / f'delta_{tag}.csv'
        if path.exists():
            done = pd.read_csv(path)
            if set(done['query']) >= set(queries):
                print(f'\n{m}: complete ({len(done)} rows) -- skipping')
                continue
        t0 = time.time()
        try:
            kw = {}
            if m in ROUTE_OVERRIDES:
                ep, wire_id = ROUTE_OVERRIDES[m]
                cfg = config._load()['endpoints'][ep]
                kw = {'inference_endpoint': cfg['base_url'], 'token_var': cfg['token_env']}
            ar = qa.AnswerGenerator(prompt=prompts.AR_USER, system_prompt=prompts.AR_SYSTEM,
                                    model=ROUTE_OVERRIDES[m][1] if m in ROUTE_OVERRIDES else m,
                                    max_tokens=None, **kw)
            pred, info = ar.predict(queries, return_raw=True)
        except Exception as e:                    # one dead model must not stop the sweep
            print(f'\n{m}: FAILED  {type(e).__name__}: {str(e)[:140]}')
            continue
        scores = score_ar(targets, pred)
        info_df = pd.DataFrame([i if i is not None else {} for i in info])
        df = pd.DataFrame({'model': m, 'endpoint': name, 'dataset': DATASET,
                           'law_book': delta.law_book, 'article': delta.article,
                           'query': queries, 'target': targets,
                           'answer': pred, 'score': scores})
        for col in ('completion_tokens', 'prompt_tokens', 'total_cost', 'finish_reason', 'time'):
            if col in info_df:
                df[col] = info_df[col].values
        df.to_csv(path, index=False)
        ok = sum(p is not None for p in pred)
        print(f'\n{m}: {ok}/{len(queries)} answered, mean ROUGE-L {100 * df.score.mean():.2f}, '
              f'{time.time() - t0:.0f}s -> {path.name}')

    files = sorted(glob(str(OUT / 'delta_*.csv')))
    if files:
        allrows = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        allrows.to_csv(OUT / 'recitation_dual_delta.csv', index=False)
        print(f'\n{len(allrows)} rows across {allrows.model.nunique()} models '
              f'-> {OUT / "recitation_dual_delta.csv"}')
        if 'total_cost' in allrows:
            print(f'total generation cost: ${allrows.total_cost.fillna(0).sum():.4f}')


if __name__ == '__main__':
    main()

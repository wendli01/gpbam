"""Does a newly served checkpoint reproduce the published recitation run?

Three models -- ``open_steuerllm``, ``EuroLLM-22B-Instruct-2512`` and
``Ministral-3-14B-Reasoning-2512`` -- were served on a local vLLM that no longer
exists.  Their *GPBam laws* cell in Table 1 is therefore a mean over the 73
provisions the dual-extractor swap kept, not over 100, and carries an asterisk
for it (see ``RECITATION_RERUN_HANDOFF.md``).

Before generating the 27 missing provisions on a *new* host, that host has to be
shown to reproduce the old one.  A fresh serving can differ in quantisation,
chat template, sampling defaults or context ceiling, and any of those would make
the 27 new answers incomparable with the 73 kept ones -- silently, because the
number that comes out is a plausible-looking ROUGE-L either way.

This re-generates provisions that **already have published answers** and asks
whether the new serving lands inside the old one's own sampling noise:

    published   the stored answers, scored by the stored `score` column
    run_a       the same prompts, re-generated on the new host
    run_b       the same prompts again -- the sampling floor

``temperature`` is never set by ``src.llm`` (the field is omitted, so the server
picks), and the published runs were sampled rather than greedy.  Per-item
equality is therefore *not* expected and is not the test.  The test is that
``|run_a - published|`` does not exceed ``|run_b - run_a|``: a difference inside
the floor is sampling, a difference outside it is a different model.

Two item sets, both with a published number for all three models:

    kept73      the 73 GPBam-laws provisions that survive the extractor swap --
                exactly the cells that are starred in the paper
    mostcited   the 100 Most-cited-laws provisions -- untouched by the swap, so
                an independent check on the serving

Run from ``experiments/``::

    python analysis/recitation_repro_probe.py --list
    python analysis/recitation_repro_probe.py --dataset mostcited --n 30
    python analysis/recitation_repro_probe.py --dataset kept73          # all 73

Resumable: each completed pass is written to ``--out`` and is not re-run, so a
kill costs one pass.  Generation is free on a local route.
"""
import argparse
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

# One source of truth for routing, scoring and the env aliasing: importing the
# delta script gives us exactly the plumbing that produced the 29 complete rows.
from recitation_dual_delta import (AR, route, reachable, score_ar,   # noqa: E402
                                   verify_scorer)
from src import prompts, qa                                          # noqa: E402

LOCAL = ['windprak/open_steuerllm',
         'utter-project/EuroLLM-22B-Instruct-2512',
         'mistralai/Ministral-3-14B-Reasoning-2512']
COLS = ['model', 'dataset', 'query', 'target', 'answer', 'score']


def published(dataset):
    """Every published answer for one dataset, from the main CSV and per-model ones."""
    frames = [pd.read_csv(AR / 'article_recitation.csv', usecols=COLS)]
    for path in sorted(glob(str(AR / 'recitation_*.csv'))):
        frames.append(pd.read_csv(path, usecols=lambda c: c in set(COLS)))
    d = pd.concat(frames, ignore_index=True)
    return d[d.dataset == dataset].reset_index(drop=True)


def kept_keys():
    """The 73 provisions the extractor swap keeps: published set AND dual set."""
    new = pd.read_csv(ROOT / 'experiments/data/gp_laws_dual.csv', dtype={'article': str})
    old = pd.read_csv(ROOT / 'experiments/data/gp_laws.csv', dtype={'article': str})
    key = lambda b, a: f'{str(b).strip().lower()} {str(a).strip().lower()}'
    return ({key(b, a) for b, a in zip(new.law_book, new.article)} &
            {key(b, a) for b, a in zip(old.law_book, old.article)})


def items(dataset, n, seed):
    """(dataset name, frame of query/target) for the chosen item set."""
    name = 'GPBam Laws' if dataset == 'kept73' else 'Most cited Laws'
    pub = published(name)
    if dataset == 'kept73':
        keep = kept_keys()
        pub = pub[pub['query'].str.strip().str.lower().isin(keep)]
    if n:
        qs = sorted(pub['query'].unique())
        rng = np.random.default_rng(seed)
        pick = set(rng.choice(qs, size=min(n, len(qs)), replace=False))
        pub = pub[pub['query'].isin(pick)]
    return name, pub.reset_index(drop=True)


def generate(model, queries, targets):
    ar = qa.AnswerGenerator(prompt=prompts.AR_USER, system_prompt=prompts.AR_SYSTEM,
                            model=model, max_tokens=None)
    pred, _ = ar.predict(queries, return_raw=True)
    return pred, np.array(score_ar(targets, pred)) * 100


def report(tag, model, pub_s, runs):
    """Effect against noise floor, in ROUGE-L points."""
    a = runs[0]
    eff = a - pub_s
    print(f'\n  {model}  [{tag}, n={len(pub_s)}]')
    print(f'    published {pub_s.mean():6.2f}   run_a {a.mean():6.2f}'
          + ''.join(f'   run_{chr(98 + i)} {r.mean():6.2f}' for i, r in enumerate(runs[1:])))
    print(f'    effect (run_a - published): mean {eff.mean():+6.2f}  '
          f'mean|d| {np.abs(eff).mean():5.2f}  max|d| {np.abs(eff).max():5.2f}')
    if len(runs) < 2:
        print('    no second pass -- no noise floor, so nothing is decided')
        return
    noise = runs[1] - a
    print(f'    noise  (run_b - run_a):     mean {noise.mean():+6.2f}  '
          f'mean|d| {np.abs(noise).mean():5.2f}  max|d| {np.abs(noise).max():5.2f}')
    from scipy import stats
    if np.abs(eff).sum() + np.abs(noise).sum() == 0:
        print('    both differences are identically zero -- greedy decoding, exact match')
        return
    p = stats.wilcoxon(np.abs(eff), np.abs(noise)).pvalue
    verdict = ('reproduces: the gap to published is inside this host\'s own noise'
               if p > .05 else
               'DOES NOT reproduce: the gap exceeds the sampling floor -- do not '
               'generate the 27 on this serving')
    print(f'    Wilcoxon(|effect| vs |noise|): p={p:.3f}  ->  {verdict}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', default=LOCAL)
    ap.add_argument('--dataset', choices=['kept73', 'mostcited'], default='kept73')
    ap.add_argument('--n', type=int, default=0, help='sample this many items (0 = all)')
    ap.add_argument('--reps', type=int, default=1, help='extra passes for the noise floor')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='analysis/out/recitation_repro_probe.csv')
    ap.add_argument('--list', action='store_true', help='print the plan and exit')
    a = ap.parse_args()

    name, pub = items(a.dataset, a.n, a.seed)
    queries = sorted(pub['query'].unique())
    print(f'{name}: {len(queries)} provisions'
          + (f' (sampled from {len(kept_keys()) if a.dataset == "kept73" else 100}, '
             f'seed {a.seed})' if a.n else ''))

    plan, skipped = [], []
    for m in a.models:
        have = pub[pub.model == m]
        if have.empty:
            skipped.append((m, '-', 'no published answers for this dataset'))
            continue
        ep, _ = route(m)
        if not reachable(ep):
            skipped.append((m, ep, 'endpoint unreachable -- is the host up?'))
            continue
        plan.append((m, ep))
    passes = 1 + max(0, a.reps)
    print(f'\nwill run {len(plan)} models x {len(queries)} prompts x {passes} passes '
          f'= {len(plan) * len(queries) * passes} generations')
    for m, ep in plan:
        print(f'  run   {m:52s} {ep}')
    for m, ep, why in skipped:
        print(f'  skip  {m:52s} {ep:11s} {why}')
    if a.list or not plan:
        return

    verify_scorer(published('GPBam Laws'))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()

    rows = []
    for m, ep in plan:
        have = pub[(pub.model == m) & (pub['query'].isin(queries))]
        have = have.drop_duplicates('query').set_index('query').loc[queries]
        pub_s = have.score.values * 100
        targets = have.target.tolist()
        runs = []
        for p in range(passes):
            tag = f'run_{chr(97 + p)}'
            prev = done[(done.model == m) & (done.pass_ == tag)] if len(done) else pd.DataFrame()
            if len(prev) == len(queries):
                print(f'  resuming: {m} {tag} already on disk')
                runs.append(prev.set_index('query').loc[queries].score.values)
                continue
            t0 = time.time()
            try:
                pred, sc = generate(m, queries, targets)
            except Exception as e:
                print(f'\n{m}: FAILED  {type(e).__name__}: {str(e)[:140]}')
                break
            ok = sum(x is not None for x in pred)
            print(f'  {m} {tag}: {ok}/{len(queries)} answered, '
                  f'mean {sc.mean():.2f}, {time.time() - t0:.0f}s')
            rows.append(pd.DataFrame({'model': m, 'endpoint': ep, 'dataset': name,
                                      'pass_': tag, 'query': queries, 'target': targets,
                                      'answer': pred, 'score': sc,
                                      'published_score': pub_s}))
            pd.concat([done] + rows, ignore_index=True).to_csv(out, index=False)
            runs.append(sc)
        if runs:
            report(a.dataset, m, pub_s, runs)

    print(f'\nwritten to {out}')


if __name__ == '__main__':
    main()

"""Does adding state law to the corpus actually make the norms findable?

The oracle condition showed that supplying the governing norms is worth ~5-7
points. It did not show that a retriever can *find* them, because the oracle
does not search -- it is handed the answer's own citations.

This is the retrieval-side test: identical pipeline, identical queries, two
corpora. If recall barely moves, the corpus gap was not the binding constraint
and the retrieval story stays a retriever story. If it jumps, the missing
Landesrecht was the constraint.

Queries come from the pipeline's own ``qa.ReWriter`` (3-5 standalone legal search
queries per case), computed once and cached so both corpora see identical input.
``--queries sentences`` substitutes sentence groups of the Sachverhalt instead,
which needs no LLM but measures a degraded pipeline: narrative prose is not what
the retriever was tuned for, and the round-robin merge has nothing to diversify.

Run from ``experiments/``::

    PYTHONPATH=analysis python analysis/corpus_comparison.py
"""

import argparse
import json
import re

import numpy as np
import pandas as pd

import os, sys
sys.path.insert(0, os.path.abspath(".."))

import refs as refs_mod
from retrieval_diagnostics import refex_norms, GPBAM, OUT

FEDERAL_KB = './my_knowledge_base'
BAYERN_KB = './my_knowledge_base_bayern'
# Same model family as the InnKube default the pipeline shipped with, but the
# NHR@FAU endpoint measured 6.6 s/case against InnKube's 11.6 s/case.
REWRITER_MODEL = 'Qwen/Qwen3.6-35B-A3B-FP8'
SECTION_RE = re.compile(r'(?:§+|Art\.?)\s*(\d+\s?[a-z]?)')


def gold_sets(gold='full'):
    d = json.load(open(GPBAM))
    order = sorted(d['solutions'], key=lambda x: int(x))
    facts = [d['facts'][k] for k in order]
    out = {}
    for i, k in enumerate(order):
        sol = d['solutions'][k]
        if gold == 'refex':
            out[i] = refex_norms(sol)
        else:
            out[i] = {(refs_mod.corpus_key(c.book), c.section.lower())
                      for c in refs_mod.canonicalise(refs_mod.extract(sol))}
    return facts, out


def subqueries(facts, n=6):
    """Turn a Sachverhalt into a handful of coherent sub-queries.

    ``LawRetriever`` splits its input on *every* newline, and the GPBam facts are
    a single paragraph hard-wrapped at ~76 columns.  Feeding them in raw issues
    seventy queries cut mid-word (``kreisange`` / ``hoerigen Gemeinde G.``), and
    the round-robin merge in ``_search_case`` then fills the whole budget with
    rank-0 hits for those fragments -- which is how the first run came back with
    coffee-tax and milk-quota statutes.  Unwrap first, then group sentences into
    ``n`` chunks, roughly the granularity the rewriter emits online.
    """
    text = re.sub(r'\s*\n\s*', ' ', facts)
    text = re.sub(r'\s+', ' ', text).strip()
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ])', text)
             if len(s.strip()) > 20]
    if not sents:
        return text
    per = max(1, len(sents) // n)
    return '\n'.join(' '.join(sents[i:i + per]) for i in range(0, len(sents), per))


def norm_of(row):
    m = SECTION_RE.search(str(row.get('paragraph', '')))
    if not m:
        return None
    return (refs_mod.corpus_key(row.get('law_book', '')),
            re.sub(r'\s+', '', m.group(1)).lower())


def rewritten_queries(facts, model=REWRITER_MODEL, path=None):
    """The pipeline's own query rewrites, computed once and cached to disk.

    ``qa.ReWriter`` asks for the 3-5 governing legal questions as standalone
    search queries -- genuine diversification, which slicing the narrative into
    sentence groups is not.  It samples, so both corpora have to see the *same*
    rewrites or the run stops being a corpus comparison; hence the cache.
    """
    path = path or f"{OUT}/rewrites_{model.split('/')[-1].lower()}.json"
    if os.path.exists(path):
        cached = json.load(open(path))
        if len(cached) == len(facts):
            print(f'rewrites: {len(cached)} cached queries from {path}', flush=True)
            return cached
    from dotenv import load_dotenv
    load_dotenv('../.env', override=True)
    for env_name, yaml_name in (('API_KEY_UP', 'innkube_api'),
                                ('API_KEY_FAU', 'nhr_fau_api'),
                                ('API_KEY', 'openrouter_api')):
        if env_name in os.environ:
            os.environ.setdefault(yaml_name, os.environ[env_name])
    from src import qa
    print(f'rewriting {len(facts)} cases with {model} ...', flush=True)
    out = qa.ReWriter(model=model).predict(list(facts))
    out = [o or '' for o in out]
    json.dump(out, open(path, 'w'), ensure_ascii=False, indent=1)
    n = [len([l for l in o.splitlines() if len(l.strip()) > 12]) for o in out]
    print(f'  -> {path}: {np.mean(n):.1f} sub-queries per case '
          f'(min {min(n)}, max {max(n)}), {sum(1 for o in out if not o)} failed', flush=True)
    return out


def check_corpus(kb, expect_titled=None):
    """Fail loudly on a corpus the retriever cannot actually query.

    ``LawRetriever`` now defaults to jina-v5, and the original corpora were
    embedded with bge-large-en-v1.5.  Both produce 1024-dimensional vectors, so
    a mismatched pair does not raise -- it silently returns noise, which is what
    a 2.3% recall@50 looks like.  The rebuilt tables are recognisable by the
    ``embed_text`` column ``rebuild_kb_with_titles.py`` adds, so require it
    explicitly rather than trusting a directory name.
    """
    import lancedb
    tbl = lancedb.connect(kb).open_table('documents')
    titled = 'embed_text' in tbl.schema.names
    if expect_titled is not None and titled != expect_titled:
        raise SystemExit(
            f'{kb}: embed_text {"present" if titled else "missing"}, expected the '
            f'{"rebuilt" if expect_titled else "original"} corpus. Refusing to run: '
            'jina and bge vectors are both 1024-d, so the mismatch would not raise, '
            'it would just return noise.')
    return tbl.count_rows(), titled


def evaluate(kb, queries, gold, top_k, device='cuda'):
    from src.rag import LawRetriever
    ret = LawRetriever(kb_name=kb, top_k=top_k, device=device, verbose=0)
    import time
    hits = tot = state_hits = state_tot = 0
    per_case = []
    t0 = time.time()
    for i, q in enumerate(queries):
        g = gold.get(i)
        if not g:
            continue
        # queries already carry one sub-query per line; LawRetriever splits them
        rows = ret.predict([q])[0]
        got = {n for n in (norm_of(r) for r in rows) if n}
        h = len(got & g)
        hits += h
        tot += len(g)
        per_case.append(h > 0)
        st = {n for n in g if refs_mod.jurisdiction(n[0]) == 'state'}
        if st:
            state_hits += len(got & st)
            state_tot += len(st)
        if (i + 1) % 5 == 0 or i + 1 == len(queries):
            el = time.time() - t0
            print(f'    [{i + 1:>2}/{len(queries)}] recall {100 * hits / max(1, tot):5.2f}%  '
                  f'state {100 * state_hits / max(1, state_tot):5.2f}%  '
                  f'{el / (i + 1):.1f}s/case, ~{el / (i + 1) * (len(queries) - i - 1) / 60:.0f} min left',
                  flush=True)
    return dict(recall=hits / tot, cases_with_hit=float(np.mean(per_case)),
                hits=hits, gold=tot,
                state_recall=(state_hits / state_tot) if state_tot else float('nan'),
                state_gold=state_tot)


def main(top_k=10, gold='full', device='cuda', queries='rewriter',
         kb_federal=FEDERAL_KB, kb_bayern=BAYERN_KB, titled=None, tag=''):
    facts, g = gold_sets(gold)
    print(f'gold set: {gold} -- {sum(len(v) for v in g.values()) / len(g):.1f} norms per case')
    print(f'queries:  {queries}\n')
    pairs = (('federal only', kb_federal), ('federal + Bavarian', kb_bayern))
    for name, kb in pairs:
        n, is_titled = check_corpus(kb, titled)
        print(f'  {name:<22} {kb}  {n} rows  '
              f'({"rebuilt/jina" if is_titled else "original/bge"})', flush=True)
    print()
    qs = rewritten_queries(facts) if queries == 'rewriter' else [subqueries(f) for f in facts]
    rows = []
    for name, kb in pairs:
        r = evaluate(kb, qs, g, top_k, device)
        r['corpus'] = name
        r['kb'] = kb
        rows.append(r)
        print(f'{name:<22} recall@{top_k} {100 * r["recall"]:5.2f}%   '
              f'cases with a hit {100 * r["cases_with_hit"]:5.1f}%   '
              f'state-law recall {100 * r["state_recall"]:5.2f}% of {r["state_gold"]}')
    df = pd.DataFrame(rows)
    df['queries'] = queries
    out = f'{OUT}/corpus_comparison_k{top_k}_{gold}_{queries}{tag}.csv'
    df.to_csv(out, index=False)
    print(f'\nwritten to {out}')
    if len(rows) == 2 and rows[0]['recall']:
        print(f'factor: {rows[1]["recall"] / rows[0]["recall"]:.2f}x')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--top-k', type=int, default=10)
    ap.add_argument('--gold', choices=['full', 'refex'], default='full')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--queries', choices=['rewriter', 'sentences'], default='rewriter',
                    help="'rewriter' runs the pipeline's own query rewriter (cached)")
    ap.add_argument('--kb-federal', default=FEDERAL_KB)
    ap.add_argument('--kb-bayern', default=BAYERN_KB,
                    help='the combined corpus: federal law plus BAYERN.RECHT')
    ap.add_argument('--titled', dest='titled', action='store_true', default=None,
                    help='require both corpora to be rebuild_kb_with_titles.py output')
    ap.add_argument('--original', dest='titled', action='store_false',
                    help='require both corpora to be the original bge-embedded tables')
    ap.add_argument('--tag', default='', help='suffix for the output CSV name')
    a = ap.parse_args()
    main(a.top_k, a.gold, a.device, a.queries, a.kb_federal, a.kb_bayern, a.titled, a.tag)

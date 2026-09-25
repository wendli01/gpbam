"""What would have to change for retrieval to work.

Runs directly against the existing LanceDB knowledge base
(``experiments/my_knowledge_base``, 106,354 indexed norms) and measures recall of
the reference solutions' citations under different retrieval settings.  Only the
embedding model runs locally -- **no LLM is called**, so this costs nothing and
needs no API key.

Gold citations come from refex, i.e. the same universe as ``legal_ref_sim`` and
as ``failure_mode_refex.py``, so the numbers are comparable with the paper.

One caveat on the baseline: the deployed pipeline searched with the *rewriter's*
output, which is not stored anywhere and cannot be regenerated without an LLM
call.  The ``facts`` query variants below use the raw case facts instead.  The
5-passage hybrid row is therefore a proxy for the deployed configuration, not a
replica -- it is included so that every other row can be read as a delta against
a baseline produced by the same code path.

Run from ``experiments/``::

    PYTHONPATH=analysis python analysis/retrieval_ablation.py            # default sweep
    PYTHONPATH=analysis python analysis/retrieval_ablation.py --quick    # fts only, no GPU
"""

import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd

import refs as refs_mod
from retrieval_diagnostics import refex_norms, GPBAM, OUT

KB = './my_knowledge_base'
EMBEDDER = 'jinaai/jina-embeddings-v5-text-small-retrieval'

STRUCTURAL_RE = re.compile(
    r'Eingangsformel|Inhalts(?:verzeichnis|übersicht)|Schlussformel|Anlage|Anhang|Präambel',
    re.IGNORECASE)
SECTION_RE = re.compile(r'(?:§+|Art\.?)\s*(\d+\s?[a-z]?)')


def norm_of(row):
    """A retrieved LanceDB row -> ``(book, section)`` in the gold set's key space."""
    m = SECTION_RE.search(str(row['paragraph']))
    if not m:
        return None
    return (refs_mod.corpus_key(row['law_book']), re.sub(r'\s+', '', m.group(1)).lower())


def is_structural(row):
    return bool(STRUCTURAL_RE.search(str(row['paragraph'])))


# --- query construction ----------------------------------------------------

def q_full(facts):
    """The whole Sachverhalt as one string -- the shape the deployed pipeline used."""
    return [re.sub(r'\s+', ' ', facts)]


def q_paragraphs(facts, min_chars=120):
    """One query per paragraph of the facts, issued separately and merged.

    This is the cheap version of "issue the rewriter's queries separately": it
    needs no LLM, and it tests whether the concatenation itself is the problem.
    """
    parts = [re.sub(r'\s+', ' ', p).strip() for p in re.split(r'\n\s*\n', facts)]
    parts = [p for p in parts if len(p) >= min_chars]
    return parts or q_full(facts)


def q_sentences(facts, max_q=8, min_chars=60):
    sents = [re.sub(r'\s+', ' ', s).strip() for s in re.split(r'(?<=[.!?])\s+', facts)]
    sents = [s for s in sents if len(s) >= min_chars]
    if not sents:
        return q_full(facts)
    # longest sentences carry the most legally salient detail
    return sorted(sents, key=len, reverse=True)[:max_q]


def q_oracle(facts, solution=None):
    """Upper bound: query with the reference solution itself.

    The solution names the governing norms in full. If retrieval cannot find them
    even from this, the problem is the index or the matching, not the query.
    """
    return [re.sub(r'\s+', ' ', solution or facts)]


QUERY_BUILDERS = {'facts': q_full, 'paragraphs': q_paragraphs, 'sentences': q_sentences,
                  'oracle_solution': q_oracle}


# --- retrieval -------------------------------------------------------------

class Retriever:
    def __init__(self, kb=KB, embedder=EMBEDDER, device='cuda', need_vectors=True):
        import lancedb
        self.db = lancedb.connect(kb)
        self.table = self.db.open_table('documents')
        self.model = None
        if need_vectors:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(embedder, device=device, trust_remote_code=True)

    @staticmethod
    def _clean(query, max_words=350):
        """Make a string safe for the tantivy query parser.

        The deployed ``_perform_search`` strips only quotes and newlines, which is
        enough for the rewriter's short outputs but not for raw case text --
        parentheses, colons and quotation marks are query-language syntax and
        raise before any search happens.
        """
        # lower-cased because tantivy reads bare AND / OR / NOT / IN as operators,
        # and the index tokenizer lower-cases anyway
        q = re.sub(r'[^\wäöüÄÖÜß§ ]+', ' ', query).lower()
        q = re.sub(r'\s+', ' ', q).strip()
        return ' '.join(q.split()[:max_words])

    def search(self, query, k, mode='hybrid', drop_structural=False):
        over = k * 4 if drop_structural else k
        sanitized = self._clean(query)
        if mode == 'fts':
            res = self.table.search(sanitized, query_type='fts').limit(over)
        elif mode == 'vector':
            v = self.model.encode([sanitized])[0]
            res = self.table.search(v).limit(over)
        else:
            res = self.table.search(sanitized, query_type='hybrid').limit(over)
        rows = res.to_pandas().to_dict('records')
        if drop_structural:
            rows = [r for r in rows if not is_structural(r)]
        return rows[:k]

    def search_multi(self, queries, k, mode='hybrid', drop_structural=False):
        """Issue each query separately, merge, dedupe, keep the first k distinct norms.

        Round-robin over the per-query result lists, so one query cannot crowd
        out the others -- the failure mode the concatenated query has.
        """
        lists = [self.search(q, k, mode, drop_structural) for q in queries]
        merged, seen = [], set()
        for rank in range(k):
            for lst in lists:
                if rank < len(lst):
                    key = (lst[rank]['law_book'], lst[rank]['paragraph'])
                    if key not in seen:
                        seen.add(key)
                        merged.append(lst[rank])
        return merged[:k]


# --- evaluation ------------------------------------------------------------

def evaluate(ret, facts, gold, query_kind, k, mode, drop_structural, separate, sols=None):
    hits, golds, structural, per_case = 0, 0, 0, []
    for i, f in enumerate(facts):
        g = gold[i]
        if not g:
            continue
        if query_kind == 'oracle_solution':
            queries = q_oracle(f, sols[i])
        else:
            queries = QUERY_BUILDERS[query_kind](f)
        if separate and len(queries) > 1:
            rows = ret.search_multi(queries, k, mode, drop_structural)
        else:
            rows = ret.search(' '.join(queries), k, mode, drop_structural)
        got = {n for n in (norm_of(r) for r in rows) if n}
        h = len(got & g)
        hits += h
        golds += len(g)
        structural += sum(is_structural(r) for r in rows)
        per_case.append(h)
    return dict(recall=hits / golds, hits=hits, gold=golds,
                cases_with_hit=float(np.mean([h > 0 for h in per_case])),
                structural=structural / max(1, k * len(per_case)))


def main(quick=False, device='cuda', k_sweep=(5, 20, 50)):
    d = json.load(open(GPBAM))
    order = sorted(d['facts'], key=lambda x: int(x))
    facts = [d['facts'][k] for k in order]
    sols = [d['solutions'][k] for k in order]
    gold = {i: refex_norms(s) for i, s in enumerate(sols)}
    print(f'{len(facts)} cases, {sum(len(g) for g in gold.values())} gold norms '
          f'({np.mean([len(g) for g in gold.values()]):.1f} per case)\n')

    modes = ['fts'] if quick else ['fts', 'hybrid']
    ret = Retriever(device=device, need_vectors=not quick)

    configs = []
    for mode in modes:
        configs.append(('facts', 5, mode, False, False))            # deployed-shape proxy
        configs.append(('facts', 5, mode, True, False))             # + drop structural text
        configs.append(('paragraphs', 5, mode, True, True))         # + issue queries separately
        for k in k_sweep:
            if k == 5:
                continue
            configs.append(('paragraphs', k, mode, True, True))     # + more passages
        configs.append(('sentences', max(k_sweep), mode, True, True))
        configs.append(('oracle_solution', 5, mode, True, False))
        configs.append(('oracle_solution', max(k_sweep), mode, True, False))

    rows = []
    for query_kind, k, mode, drop_struct, separate in configs:
        t0 = time.time()
        r = evaluate(ret, facts, gold, query_kind, k, mode, drop_struct, separate, sols)
        r.update(query=query_kind, k=k, mode=mode, no_structural=drop_struct,
                 separate_queries=separate, seconds=round(time.time() - t0, 1))
        rows.append(r)
        print(f'{mode:<7} {query_kind:<11} k={k:<3} '
              f'{"filtered" if drop_struct else "raw     "} '
              f'{"split" if separate else "single"}  ->  '
              f'recall {100 * r["recall"]:5.2f}%   cases with a hit '
              f'{100 * r["cases_with_hit"]:5.1f}%   structural {100 * r["structural"]:4.1f}%  '
              f'({r["seconds"]}s)')

    out = pd.DataFrame(rows)[['mode', 'query', 'k', 'no_structural', 'separate_queries',
                              'recall', 'cases_with_hit', 'structural', 'hits', 'gold', 'seconds']]
    out.to_csv(f'{OUT}/retrieval_ablation.csv', index=False)
    print(f'\nwritten to {OUT}/retrieval_ablation.csv')
    per_case_gold = np.mean([len(g) for g in gold.values() if g])
    print(f'\nreference points: the deployed run scored 0.96% recall@5 using the rewriter\'s '
          f'concatenated query.\nCeiling on recall is {100 * 5 / per_case_gold:.1f}% at k=5 and '
          f'{min(100, 100 * max(k_sweep) / per_case_gold):.1f}% at k={max(k_sweep)}.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quick', action='store_true', help='BM25 only, no embedding model')
    ap.add_argument('--device', default='cuda')
    main(**vars(ap.parse_args()))

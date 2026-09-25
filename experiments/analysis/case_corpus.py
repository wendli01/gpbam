"""Embed the Bavarian court-decision corpus into a LanceDB table.

``build_case_corpus.py`` acquires, normalises, de-duplicates and chunks the
decisions and stops at ``data/case_corpus/chunks.parquet`` -- deliberately, so
the slow GPU half is a separate, restartable job.  This is that half.

Why a case corpus at all: the statute corpora answer "what does § 34 BauGB
say"; they cannot answer "which norms govern a neighbour's action against a
building permit", because a statute never names the situation it applies to.  A
decision does.  That is the vocabulary gap ``recall_battery.py`` kept measuring
from the wrong side -- every query-side variant (rubric, solution,
statute-style, citations-as-queries) scored *below* the plain rewriter baseline,
which is what a corpus-side problem looks like when you attack it with queries.

Tiers
-----
335k chunks of ~1.3k characters is ~450 M characters, and this card embeds
roughly 5k characters a second: a full pass is about a day of GPU time, which
is too much to spend before knowing whether decisions help at all.

So rows are ordered by ``chunk_ix`` first and length second, and ``--tiers N``
stops after the Nth chunk of every decision.  Tier 0 alone is one chunk from
each of the 21,957 decisions -- the tenor and the opening of the reasoning,
which is where a decision states what it is about -- and takes under two hours.
Because the tier order is deterministic, a shallow run is an exact prefix of a
deeper one, so ``open_or_resume`` extends it later without re-embedding
anything.

Schema
------
Deliberately the columns ``LawRetriever`` already reads, so the same retriever
queries this table unchanged: ``law_book`` holds the court and ``paragraph`` the
docket number plus chunk index.  ``norms`` carries the decision's norm chain
verbatim -- it is not folded into the embedded text, so it stays usable as an
exact-lookup key for the two-hop step (find a decision by its doctrine, follow
its citations to the norms) that ``cite_then_search`` already does for the
norms a model names from memory.

Run from ``experiments/``::

    python analysis/case_corpus.py --tiers 1     # heads only, ~2 h
    python analysis/case_corpus.py --tiers 4     # extends the same table
"""

import os
import sys
import time

import lancedb
import lancedb.pydantic
import pandas as pd
from lancedb.embeddings import get_registry
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath('..'))

from rebuild_kb_with_titles import (MAX_CHARS, batches, encode, open_or_resume,
                                    peak_rss_gb, rss_gb)

MODEL = 'jinaai/jina-embeddings-v5-text-small-retrieval'
CHUNKS = 'data/case_corpus/chunks.parquet'
DST = './my_case_corpus_bayern'


def title_of(r):
    """What identifies the decision, in the form a citation to it takes.

    The norm chain rides along in the title because it is the one part of the
    record written in the same notation as a query naming a norm, and it is
    short enough to survive truncation -- which the reasoning is not.
    """
    ident = f"{r.court}, {r.doktyp} v. {r.date} - {r.aktenzeichen}".strip()
    chain = ' | '.join(r.norm_chain) if len(r.norm_chain) else ''
    return f'{ident} [{chain}]' if chain else ident


def load_rows(path=CHUNKS, tiers=None):
    df = pd.read_parquet(path)
    if tiers is not None:
        df = df[df.chunk_ix < tiers]
    rows = []
    for r in df.itertuples(index=False):
        chain = ' | '.join(r.norm_chain) if len(r.norm_chain) else ''
        title = title_of(r)
        rows.append(dict(
            text=str(r.text), title=title, filename=str(r.doc_id),
            law_book=str(r.court), paragraph=f'{r.aktenzeichen} #{r.chunk_ix}',
            source_path=str(r.chunk_id), court=str(r.court), date=str(r.date),
            doktyp=str(r.doktyp), norms=chain, source=str(r.source),
            tier=int(r.chunk_ix),
            embed_text=f'{title}\n{r.text}'[:MAX_CHARS]))
    # Tier first so a shallow run is a prefix of a deep one; length second so
    # GPU batches stay homogeneous and a 3k-char row does not pad seven short
    # ones up to its own size.
    rows.sort(key=lambda x: (x['tier'], len(x['embed_text'])))
    return rows


def build(chunks=CHUNKS, dst=DST, model_name=MODEL, device='cuda', batch_size=64,
          char_budget=12000, write_batch=1000, resume=True, tiers=None, limit=None):
    rows = load_rows(chunks, tiers)
    if limit:
        rows = rows[:limit]
    chars = sum(len(r['embed_text']) for r in rows)
    print(f'{len(rows)} chunks, {chars / 1e6:.0f}M characters to embed '
          f'(tiers={tiers or "all"})', flush=True)

    func = get_registry().get('sentence-transformers').create(name=model_name, device=device)
    ndims = func.ndims()

    class Docs(lancedb.pydantic.LanceModel):
        embed_text: str = func.SourceField()
        vector: lancedb.pydantic.Vector(ndims) = func.VectorField()

        text: str
        title: str
        filename: str
        law_book: str
        paragraph: str
        source_path: str
        court: str
        date: str
        doktyp: str
        norms: str
        source: str
        tier: int

    db = lancedb.connect(dst)
    tbl, start_at = open_or_resume(db, Docs, dst, rows, resume)
    if start_at >= len(rows):
        print(f'  {tbl.count_rows()} rows already embedded; building the index only',
              flush=True)
        tbl.create_fts_index(['title', 'text'], replace=True)
        return
    if start_at:
        del rows[:start_at]

    t0 = time.time()
    plan = list(batches(rows, batch_size, char_budget))
    bar = tqdm(plan, desc=os.path.basename(dst))
    buf, done = [], 0
    for start, end in bar:
        chunk = rows[start:end]
        for r, v in zip(chunk, encode(func, [r['embed_text'] for r in chunk])):
            r['vector'] = [float(x) for x in v]
        buf.extend(chunk)
        if len(buf) >= write_batch:
            tbl.add(buf)
            for j in range(done, end):
                rows[j] = None
            done = end
            buf = []
            bar.set_postfix_str(f'rss {rss_gb():.1f}G')
    if buf:
        tbl.add(buf)
    rows.clear()
    print(f'  embedded in {(time.time() - t0) / 60:.1f} min, peak rss {peak_rss_gb():.1f} GB',
          flush=True)
    tbl.create_fts_index(['title', 'text'], replace=True)
    print(f'  {tbl.count_rows()} rows, fts(title,text) built -> {dst}', flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--chunks', default=CHUNKS)
    p.add_argument('--dst', default=DST)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=64)
    # 12k rather than the statute rebuild's 24k: this runs beside a generation
    # job that holds ~3 GB of the same 8 GB card.
    p.add_argument('--char-budget', type=int, default=12000)
    p.add_argument('--write-batch', type=int, default=1000)
    p.add_argument('--tiers', type=int, default=None,
                   help='embed only the first N chunks of each decision')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--fresh', action='store_true', help='discard any partial table')
    a = p.parse_args()
    build(a.chunks, a.dst, device=a.device, batch_size=a.batch_size,
          char_budget=a.char_budget, write_batch=a.write_batch,
          resume=not a.fresh, tiers=a.tiers, limit=a.limit)

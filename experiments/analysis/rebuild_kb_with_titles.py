"""Re-embed a knowledge base so the norm's own citation is inside the vector.

``LawRetriever`` puts ``SourceField()`` on ``text`` alone, and ``text`` is the
bare statute body: ``StPO § 97`` does not contain the string "StPO" or "§ 97",
because German statutes do not restate their own citation.  The identifier lives
in ``title`` (``StPO § 97: Beschlagnahmeverbot``), which was neither embedded nor
indexed.  A query naming a norm the way a lawyer writes it therefore had nothing
to match.  Adding ``title`` to the FTS index fixed the lexical half; this fixes
the dense half.

Rows are copied straight out of the existing table rather than re-parsed from
``gesetze-im-internet`` and BAYERN.RECHT, so the corpus is identical by
construction and only the vector changes.  Writes to a new directory, leaving the
current knowledge base intact for the A/B.

Run from ``experiments/``::

    python analysis/rebuild_kb_with_titles.py                  # both corpora
    python analysis/rebuild_kb_with_titles.py --only bayern
"""

import argparse
import os
import resource
from collections import Counter
import sys
import time

# Fragmentation is what turns a recoverable OOM into a fatal one on a small
# card: the retry needs a contiguous block and the allocator has none left.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import lancedb
import lancedb.pydantic
import torch
from lancedb.embeddings import get_registry
from tqdm import tqdm

sys.path.insert(0, os.path.abspath('..'))

MODEL = 'jinaai/jina-embeddings-v5-text-small-retrieval'
COLUMNS = ['text', 'title', 'filename', 'law_book', 'paragraph', 'source_path']
PAIRS = {
    'federal': ('./my_knowledge_base', './my_knowledge_base_titled'),
    'bayern': ('./my_knowledge_base_bayern', './my_knowledge_base_bayern_titled'),
}


#: jina-v5 advertises max_seq_length 32768 and the corpus holds rows up to
#: 652k characters, so a single monster row allocates ~7 GiB and OOMs an 8 GB
#: card whatever the batch size.  Only the vector input is capped; the ``text``
#: column keeps the full statute, so the answer context is unaffected, and the
#: citation this rebuild is about sits in the first line either way.
MAX_CHARS = 4000

#: Characters per GPU batch.  Rows are embedded shortest-first, so a fixed row
#: count makes every tail batch a batch of maximum-length rows -- which is how a
#: run gets to 47% and then dies.  Budgeting by characters lets the short head
#: run wide and the long tail narrow automatically.  8 rows x 4000 chars was
#: measured at 5.25 GiB on this card, so 24k characters targets roughly 4 GiB
#: and leaves room for the desktop's ~0.4 GiB.
CHAR_BUDGET = 24000


def rss_gb():
    """Resident set size right now, in GB."""
    with open('/proc/self/statm') as f:
        return int(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 1e9


def peak_rss_gb():
    """High-water RSS for this process, in GB -- what the OOM killer reacts to."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def batches(rows, max_rows, char_budget):
    """``(start, end)`` index ranges sized by total characters, not row count."""
    start = total = 0
    for i, r in enumerate(rows):
        n = len(r['embed_text'])
        if i > start and (total + n > char_budget or i - start >= max_rows):
            yield start, i
            start, total = i, 0
        total += n
    if start < len(rows):
        yield start, len(rows)


def encode(func, texts):
    """Embed a batch, halving it on OOM rather than dying.

    Supplying vectors explicitly is what made the write path fast, but it also
    took this out from under LanceDB's retry wrapper: an OOM used to be retried
    and now propagates.

    The retry has to happen *outside* the ``except`` block.  While an exception
    is being handled its traceback holds every frame of the forward pass, and
    those frames hold the activation tensors -- so ``empty_cache()`` called in
    the handler frees nothing, and the halved retry runs against a GPU that is
    still full.  That is how a run dies asking for 2 MiB with 7.43 GiB still
    *allocated* (not merely reserved) after the halving supposedly recovered.
    """
    oom = False
    try:
        return list(func.generate_embeddings(texts))
    except torch.OutOfMemoryError:
        oom = True
    # the handler has exited here, so the traceback -- and the activations it
    # was pinning -- are gone and the cache drop can actually reclaim them
    if oom:
        torch.cuda.empty_cache()
        if len(texts) == 1:
            raise RuntimeError(
                f'single {len(texts[0])}-char row does not fit on the GPU; '
                f'lower --max-chars (currently capped at {MAX_CHARS})')
        half = len(texts) // 2
        return encode(func, texts[:half]) + encode(func, texts[half:])


def row_key(r):
    """Identifies a corpus row independently of how it was batched or stored."""
    return (str(r.get('source_path') or ''), str(r.get('paragraph') or ''),
            str(r.get('law_book') or ''))


def open_or_resume(db, schema, dst, rows, resume=True):
    """Continue a partial table if one is there, else start a fresh one.

    This pass takes hours, and it has now been killed twice -- once by the host
    running out of memory, once by the GPU.  Overwriting on every start threw
    away 47% of a completed pass both times, which is pure waste: the rows are
    written in a deterministic order (a stable sort on ``embed_text`` length over
    a fixed source table), so whatever is already stored is exactly the prefix
    of what this run would produce.

    Resuming is only safe if the stored rows really are that prefix, so it is
    checked rather than assumed -- by key, not by position, since a scan is not
    contractually in insertion order.  Anything unexpected falls back to a clean
    rebuild instead of silently welding two different corpora together.
    """
    if 'documents' not in db.table_names():
        return db.create_table('documents', schema=schema, mode='overwrite'), 0
    fresh = lambda: (db.create_table('documents', schema=schema, mode='overwrite'), 0)
    if not resume:
        print('  --fresh: discarding the existing table', flush=True)
        return fresh()
    tbl = db.open_table('documents')
    n = tbl.count_rows()
    if n == 0:
        return fresh()
    if n > len(rows):
        print(f'  existing table has {n} rows but the source has {len(rows)}; rebuilding',
              flush=True)
        return fresh()
    stored = tbl.to_lance().to_table(columns=['source_path', 'paragraph', 'law_book'])
    # Counter, not set: the corpus does repeat these keys, and a set would call
    # two different row multisets equal as long as they drew on the same norms.
    if (Counter(row_key(r) for r in stored.to_pylist())
            != Counter(row_key(r) for r in rows[:n])):
        print(f'  existing {n} rows are not the prefix this run would write; rebuilding',
              flush=True)
        return fresh()
    print(f'  resuming: {n}/{len(rows)} rows already embedded', flush=True)
    return tbl, n


def build(src, dst, model_name=MODEL, device='cuda', batch_size=256, max_chars=MAX_CHARS,
          write_batch=1000, char_budget=CHAR_BUDGET, resume=True):
    func = get_registry().get('sentence-transformers').create(name=model_name, device=device)
    ndims = func.ndims()

    class Docs(lancedb.pydantic.LanceModel):
        # what the vector is built from: citation line first, then the body
        embed_text: str = func.SourceField()
        vector: lancedb.pydantic.Vector(ndims) = func.VectorField()

        text: str
        title: str
        filename: str
        law_book: str
        paragraph: str
        source_path: str

    src_tbl = lancedb.connect(src).open_table('documents')
    rows = src_tbl.to_lance().to_table(columns=COLUMNS).to_pylist()
    print(f'{src}: {len(rows)} rows -> {dst}', flush=True)

    for r in rows:
        title = str(r.get('title') or '').strip()
        body = (r.get('text') or '')[:max_chars]
        r['embed_text'] = f"{title}\n{body}" if title else body

    # Batch homogeneously by length.  Row order in a vector table carries no
    # meaning, and mixing a 4k-char row with seven short ones pads the whole
    # batch to the longest, which is what kept blowing up attention.
    rows.sort(key=lambda r: len(r['embed_text']))

    db = lancedb.connect(dst)
    tbl, start_at = open_or_resume(db, Docs, dst, rows, resume)
    if start_at >= len(rows):
        print(f'  {tbl.count_rows()} rows already embedded; building the index only',
              flush=True)
        tbl.create_fts_index(['title', 'text'], replace=True)
        print(f'  {tbl.count_rows()} rows, fts(title,text) built -> {dst}', flush=True)
        return
    if start_at:
        # dropping the stored prefix frees it as well, so the resumed run starts
        # from the same memory profile as a fresh one
        del rows[:start_at]
    t0 = time.time()
    # Two different limits, so two different batch sizes.  The GPU batch is
    # bounded by ``char_budget`` (a 596M fp32 model on an 8 GB card peaks at
    # 5.25 GiB for 8 rows of 4k chars, so ~32k characters is the ceiling and the
    # default sits below it); ``write_batch`` is bounded by nothing much, and a
    # per-8-row ``add()`` left the GPU idle 88% of the time writing 13k separate
    # fragments.  LanceDB keeps vectors supplied explicitly instead of recomputing
    # them, and the registered function still embeds queries at search time.
    #
    # Hand the rows over to LanceDB and then let go of them.  ``buf`` used to be
    # cleared after each add(), but the dicts it held were the same objects
    # ``rows`` still referenced, so nothing was actually freed: a 1024-float
    # Python list costs 33 kB per row, and keeping one for all 106k rows is
    # 3.6 GB that grows monotonically to the end of the corpus.  Dropping the
    # slice out of ``rows`` as it is written keeps the peak proportional to
    # ``write_batch`` instead of to the corpus.
    plan = list(batches(rows, batch_size, char_budget))
    bar = tqdm(plan, desc=f'embedding {os.path.basename(dst)}')
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


def main(only='both', device='cuda', batch_size=256, max_chars=MAX_CHARS, write_batch=1000,
         char_budget=CHAR_BUDGET, resume=True):
    todo = PAIRS if only == 'both' else {only: PAIRS[only]}
    for name, (src, dst) in todo.items():
        print(f'=== {name} (max_chars={max_chars}, batch<={batch_size} rows / '
              f'{char_budget} chars) ===', flush=True)
        build(src, dst, device=device, batch_size=batch_size, max_chars=max_chars,
              write_batch=write_batch, char_budget=char_budget, resume=resume)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--only', choices=['both', 'federal', 'bayern'], default='both')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch-size', type=int, default=256,
                    help='upper bound on rows per batch; --char-budget usually binds first')
    ap.add_argument('--char-budget', type=int, default=CHAR_BUDGET,
                    help='characters per GPU batch -- the real limit on an 8 GB card')
    ap.add_argument('--max-chars', type=int, default=MAX_CHARS,
                    help='cap on the text handed to the embedder (not on the stored text)')
    ap.add_argument('--write-batch', type=int, default=1000,
                    help='rows per LanceDB add(); independent of the GPU batch')
    ap.add_argument('--fresh', action='store_true',
                    help='discard a partial table instead of continuing it')
    a = ap.parse_args()
    main(a.only, a.device, a.batch_size, a.max_chars, a.write_batch, a.char_budget,
         resume=not a.fresh)

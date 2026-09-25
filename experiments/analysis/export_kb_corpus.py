"""Export the retrieval corpus out of the knowledge base, for LFS.

The knowledge base has two halves and they are not equally replaceable:

  * **federal**, 106,354 provisions, parsed from the QuantLaw
    ``gesetze-im-internet`` dump. That lives in ``experiments/data/`` as a
    *nested git clone* pinned at 67e0567ac -- the revision the paper cites --
    so it cannot be committed into this repository as-is without nesting a
    .git inside a .git. It is reproducible while GitHub and QuantLaw are up.
  * **Bavarian**, 7,008 provisions over 720 statutes, crawled from
    BAYERN.RECHT. There is no revision to pin: the site changes underneath the
    paper, and seven of the norms the reference solutions cite were repealed in
    2014, so a crawl today already cannot reproduce the 31-08-2026 one.

This writes both halves as a single parquet -- the text content only, no
vectors, since those are re-derivable by re-embedding and are what makes the
LanceDB directories multi-gigabyte. 113,362 rows at ~52 MB under zstd, against
1.2 GB for the nested clone and 2.2 GB for the two LanceDB stores.

It exists because a `git clean -fd` on 2026-09-11 deleted every untracked,
non-ignored file under experiments/. The knowledge bases survived only because
.gitignore happened to cover them. This is the same corpus in a form small
enough to commit, so the next accident is recoverable from the repository
rather than from a re-crawl that no longer reproduces.

Run from ``experiments/``::

    PYTHONPATH=analysis python analysis/export_kb_corpus.py
"""

import os
import sys

KB = './my_knowledge_base_bayern_titled'
OUT = 'data/kb_corpus.parquet'
COLUMNS = ['text', 'title', 'filename', 'law_book', 'paragraph', 'source_path']
HOST = 'gesetze-bayern.de'


def export(kb=KB, out=OUT):
    import lancedb
    t = lancedb.connect(kb).open_table('documents')
    df = t.to_lance().to_table(columns=COLUMNS).to_pandas()
    df.to_parquet(out, compression='zstd', index=False)
    bav = df.source_path.astype(str).str.contains(HOST, na=False).sum()
    print(f'-> {out}: {len(df):,} rows '
          f'({bav:,} Bavarian, {len(df) - bav:,} federal), '
          f'{os.path.getsize(out) / 1e6:.1f} MB')
    return df


if __name__ == '__main__':
    export(*(sys.argv[1:3] or []))

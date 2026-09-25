"""Reconstruct ``landesrecht_corpus.json`` from the knowledge base that used it.

``build_landesrecht_corpus.py`` crawls BAYERN.RECHT into
``analysis/out/landesrecht_corpus.json``, and ``rebuild_kb_with_titles.py``
embeds that into ``my_knowledge_base_bayern*``. A ``git clean -fd`` on
2026-09-11 removed the JSON -- it was untracked and, unlike the knowledge
bases, not covered by .gitignore. The knowledge base survived, and it holds
every field the JSON did, so the source can be read back out of the artefact
built from it rather than re-crawled: 7,008 provisions over 720 Bavarian
statutes, matching the figures the paper reports.

Two things do not survive the round trip, both harmless:

  * the negative cache. The crawler stores ``None`` for article numbers that
    404, because statute ranges are not dense and it must not retry them. Those
    keys never reach the knowledge base. A future crawl re-probes them, costing
    requests but changing no output.
  * ``title`` is stored joined as ``"{law_book} {paragraph}: {heading}"``. The
    prefix is stripped back off here, which is exact whenever the heading does
    not itself contain that prefix -- true for every row, since headings are
    of the form "Art. 76 Beseitigung von Anlagen".

Run from ``experiments/``::

    PYTHONPATH=analysis python analysis/export_landesrecht_corpus.py
"""

import json
import os
import re
import sys

KB = './my_knowledge_base_bayern_titled'
OUT = 'analysis/out/landesrecht_corpus.json'
HOST = 'gesetze-bayern.de'


def rows(kb=KB):
    import lancedb
    t = lancedb.connect(kb).open_table('documents')
    df = t.to_lance().to_table(
        columns=['text', 'title', 'law_book', 'paragraph', 'source_path']
    ).to_pandas()
    return df[df.source_path.astype(str).str.contains(HOST, na=False)]


def export(kb=KB, out=OUT):
    df = rows(kb)
    cache, skipped = {}, 0
    for r in df.itertuples():
        m = re.search(r'(\d+\s*[a-z]?)\s*$', str(r.paragraph))
        if not m:
            skipped += 1
            continue
        n = re.sub(r'\s+', '', m.group(1))
        heading = str(r.title)
        prefix = f'{r.law_book} {r.paragraph}: '
        if heading.startswith(prefix):
            heading = heading[len(prefix):]
        cache[f'{r.law_book}|{n}'] = dict(
            law_book=str(r.law_book), paragraph=str(r.paragraph),
            title=heading, text=str(r.text), source=str(r.source_path))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as fh:
        json.dump(cache, fh, ensure_ascii=False, sort_keys=True)
    books = len({k.split('|')[0] for k in cache})
    print(f'-> {out}: {len(cache)} provisions over {books} statutes '
          f'({os.path.getsize(out) / 1e6:.1f} MB), {skipped} unparsable')
    return cache


if __name__ == '__main__':
    export(*(sys.argv[1:3] or []))

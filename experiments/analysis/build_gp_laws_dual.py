"""Rebuild the *GPBam laws* recitation battery with the dual §/Art. extractor.

Why
---
``data/gp_laws.csv`` -- the 100 provisions the Article Recitation task probes --
was built by running refex over the 81 reference solutions and keeping the most
cited ones whose text the pinned ``gesetze-im-internet`` snapshot can supply.
refex resolves no ``Art.``-form citation, so the set contains **no constitutional
provision at all**, although Art. 2 GG is cited 169 times across the solutions --
the third-most-cited norm in the corpus, behind only VwGO § 80 and BauGB § 35.

This rebuilds the same set with ``analysis/refs.py``, which sees both citation
forms and is what every retrieval number already uses.  Selection rule, snapshot
and output columns are unchanged; only the extractor differs.

Writes ``data/gp_laws_dual.csv`` (100 rows, same columns as ``gp_laws.csv``) and
``analysis/out/gp_laws_dual_delta.csv`` (what enters and what leaves).

Run from ``experiments/``::

    PYTHONPATH=analysis python analysis/build_gp_laws_dual.py
"""
import collections
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import refs as R  # noqa: E402

SNAPSHOT = 'data/gesetze-im-internet/data/items'
GPBAM = 'data/gpbam_w_rubric.json'
OLD = 'data/gp_laws.csv'
OUT = 'data/gp_laws_dual.csv'
DELTA = 'analysis/out/gp_laws_dual_delta.csv'
N = 100

SECTION_RE = re.compile(r'^(?:§+|Art\.?|Artikel)\s*(\d+\s?[a-z]?)', re.IGNORECASE)


def snapshot_text():
    """``(corpus_key, section) -> official text`` from the pinned snapshot.

    First occurrence wins: a norm number appears once per law book, and the
    duplicates that do occur are the annex/appendix variants the recitation task
    is not asking about.
    """
    out = {}
    parser = etree.XMLParser(recover=True)
    files = [os.path.join(r, f) for r, _, fs in os.walk(SNAPSHOT)
             for f in fs if f.endswith('.xml')]
    for path in files:
        try:
            tree = etree.parse(path, parser=parser)
        except etree.XMLSyntaxError:
            continue
        for norm in tree.xpath('//norm'):
            jurabk = norm.xpath('.//jurabk/text()')
            enbez = norm.xpath('.//enbez/text()')
            if not jurabk or not enbez:
                continue
            m = SECTION_RE.match(enbez[0].strip())
            if not m:
                continue
            content = ' '.join(norm.xpath('.//text//text()')).strip()
            if not content:
                continue
            key = (R.corpus_key(jurabk[0].strip()), re.sub(r'\s+', '', m.group(1)).lower())
            out.setdefault(key, content)
    print(f'{len(files)} XML files, {len(out)} provisions with text')
    return out


def cited_counts():
    """``(cited book, section) -> citation count`` over the 81 solutions."""
    sol = json.load(open(GPBAM))['solutions']
    cnt = collections.Counter()
    for k in sorted(sol, key=int):
        for c in R.canonicalise(R.extract(sol[k])):
            cnt[(c.book.lower(), c.section.lower())] += 1
    print(f'{len(cnt)} distinct norms cited across {len(sol)} reference solutions')
    return cnt


def main():
    text = snapshot_text()
    cnt = cited_counts()

    rows = []
    for (book, section), n in cnt.most_common():
        # the cited abbreviation is what the query says ("baugb 35"); the corpus
        # files it under its own jurabk ("BBauG"), so look up through the alias
        content = text.get((R.corpus_key(book), section))
        if content is None:
            continue
        rows.append({'law_book': book, 'article': section, 'count': n, 'content': content})
        if len(rows) == N:
            break
    new = pd.DataFrame(rows)
    new.to_csv(OUT, index=False)
    print(f'\nwrote {OUT}: {len(new)} provisions from {new.law_book.nunique()} law books')
    print('  ' + ', '.join(f'{b}:{c}' for b, c in
                           collections.Counter(new.law_book).most_common(10)))

    old = pd.read_csv(OLD, usecols=['law_book', 'article', 'count'], dtype={'article': str})
    old_keys = {(str(b).lower(), str(a).lower()) for b, a in zip(old.law_book, old.article)}
    new_keys = {(b, a) for b, a in zip(new.law_book, new.article)}

    d = []
    for b, a in sorted(new_keys - old_keys, key=lambda k: -cnt[k]):
        d.append({'change': 'enters', 'law_book': b, 'article': a, 'count': cnt[(b, a)]})
    for b, a in sorted(old_keys - new_keys,
                       key=lambda k: -int(old.loc[(old.law_book.str.lower() == k[0]) &
                                                  (old.article.str.lower() == k[1]), 'count'].iloc[0])):
        n = int(old.loc[(old.law_book.str.lower() == b) & (old.article.str.lower() == a), 'count'].iloc[0])
        d.append({'change': 'leaves', 'law_book': b, 'article': a, 'count': n})
    delta = pd.DataFrame(d)
    delta.to_csv(DELTA, index=False)
    kept = len(new_keys & old_keys)
    print(f'\n{kept}/100 kept, {len(new_keys - old_keys)} enter, {len(old_keys - new_keys)} leave')
    print(f'wrote {DELTA}')
    print('\nentering:')
    print(delta[delta.change == 'enters'].to_string(index=False))


if __name__ == '__main__':
    main()

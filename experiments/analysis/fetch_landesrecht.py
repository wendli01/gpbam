"""Fetch the state-law norms GPBam cites from BAYERN.RECHT.

The federal ``gesetze-im-internet`` corpus contains no *Landesrecht*, so roughly
half of every case's cited norms have no text in the knowledge base -- and German
public-law examinations are argued out of exactly those state codes.  This
fetches the specific norms the 81 reference solutions cite, not the collection:
418 norms across 35 books at the time of writing.

**Terms.** BAYERN.RECHT states that its texts "können örtlich, zeitlich und
inhaltlich uneingeschränkt genutzt und weiterverwendet werden", explicitly
including reproduction and further processing; database protection under
§§ 87a ff. UrhG is preserved, which is why this takes a targeted subset rather
than crawling the collection, and § 60d UrhG privileges text-and-data-mining
reproductions for scientific research.  Requests are rate-limited to one per
second and identify themselves.  Note the site's own caveat that these are
*nichtamtliche Fassungen* in their current version -- a norm may have been
amended since the examination was set.

Run from ``experiments/``::

    PYTHONPATH=analysis python analysis/fetch_landesrecht.py           # fetch what is missing
    PYTHONPATH=analysis python analysis/fetch_landesrecht.py --list    # show the work, fetch nothing
"""

import argparse
import json
import os
import re
import time

import pandas as pd
import requests
import lxml.html as LH

import refs as refs_mod
from retrieval_diagnostics import refex_norms, GPBAM, OUT  # noqa: F401

BASE = 'https://www.gesetze-bayern.de/Content/Document/'
UA = ('GPBam-research/0.1 (academic legal-NLP benchmark; '
      'contact lorenz.wendlinger@th-deg.de)')
CACHE = f'{OUT}/landesrecht.json'
CORPUS_INDEX = f'{OUT}/corpus_norms.csv'
DELAY = 1.0

# Site abbreviations differ from the ones lawyers cite. Verified by probe; the
# fallback covers the rest, since the site prefixes most Bavarian codes with Bay.
SITE_ABBR = {
    'bv': 'BayVerf', 'pag': 'BayPAG', 'go': 'BayGO', 'lstvg': 'BayLStVG',
    'agvwgo': 'BayAGVwGO', 'vfghg': 'BayVfGHG', 'lkro': 'BayLKrO',
    'vwzvg': 'BayVwZVG', 'kommzg': 'BayKommZG', 'kag': 'BayKAG',
    'baybo': 'BayBO', 'bayvwvfg': 'BayVwVfG', 'bayversg': 'BayVersG',
    'bayfwg': 'BayFwG', 'bezo': 'BayBezO', 'baynatschg': 'BayNatSchG',
    'baywg': 'BayWG', 'bayeug': 'BayEUG', 'baydsg': 'BayDSG',
}


def canon_book(book):
    """Fold the Bay-prefixed and bare spellings of the same code together.

    The solutions cite both 'AGVwGO' and 'BayAGVwGO' for one law; without this
    they are fetched twice and stored under two keys.
    """
    b = book.lower()
    if b.startswith('bay') and b[3:] in SITE_ABBR:
        return b[3:]
    return b


def site_candidates(book):
    """Candidate site abbreviations for one of our book keys, best first."""
    b = canon_book(book)
    if b in SITE_ABBR:
        return [SITE_ABBR[b]]
    stem = book.upper()
    if b.startswith('bay'):
        # preserve the site's mixed casing: BayAGVwGO, not BayAGVWGO
        return [book[:3].title() + book[3:], book[:3].title() + book[3:].upper(), stem]
    return ['Bay' + book, 'Bay' + stem, stem]


def needed_norms(corpus_index=CORPUS_INDEX):
    """State-law (book, section) pairs cited by the solutions and absent from the corpus."""
    corpus = pd.read_csv(corpus_index, dtype=str).fillna('')
    corpus['key'] = corpus.law_book.map(refs_mod.corpus_key)
    have = {(r.key, r.section.lower()) for r in corpus.itertuples() if r.section}
    have_books = {b for b, _ in have}

    d = json.load(open(GPBAM))
    out = set()
    for k in sorted(d['solutions'], key=lambda x: int(x)):
        for c in refs_mod.canonicalise(refs_mod.extract(d['solutions'][k])):
            n = (canon_book(refs_mod.corpus_key(c.book)), c.section.lower())
            if n in have or n in out:
                continue
            if refs_mod.jurisdiction(c.book, have_books) == 'state':
                out.add(n)
    return sorted(out)


def parse(html):
    """(heading, normtext) from a BAYERN.RECHT document page."""
    doc = LH.fromstring(html)
    head = doc.xpath('//*[@class="paraheading"]')
    body = doc.xpath('//*[@class="cont"]')
    if not body:
        return None, None
    heading = re.sub(r'\s+', ' ', head[0].text_content()).strip() if head else ''
    text = re.sub(r'\s+', ' ', body[0].text_content()).strip()
    if heading and text.startswith(heading):
        text = text[len(heading):].strip()
    return heading, text


def main(list_only=False, limit=None, delay=DELAY):
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    todo = [n for n in needed_norms() if f'{n[0]}|{n[1]}' not in cache]
    print(f'{len(cache)} norms cached, {len(todo)} to fetch')
    if list_only:
        import collections
        for b, c in collections.Counter(b for b, _ in todo).most_common():
            print(f'  {b:<12} {c:>3}')
        return
    if limit:
        todo = todo[:limit]

    s = requests.Session()
    s.headers.update({'User-Agent': UA})
    ok = fail = 0
    for i, (book, sect) in enumerate(todo, 1):
        got = False
        for abbr in site_candidates(book):
            url = f'{BASE}{abbr}-{sect}'
            try:
                r = s.get(url, timeout=30)
            except Exception as e:
                print(f'  {url}: {type(e).__name__}')
                time.sleep(delay)
                continue
            time.sleep(delay)
            if r.status_code != 200:
                continue
            heading, text = parse(r.text)
            if text and len(text) > 40:
                cache[f'{book}|{sect}'] = dict(
                    law_book=book.upper(), paragraph=f'Art. {sect}', title=heading,
                    text=text, source=url)
                ok += 1
                got = True
                break
        if not got:
            fail += 1
            cache[f'{book}|{sect}'] = None          # remember the miss, don't refetch
        if i % 25 == 0 or i == len(todo):
            json.dump(cache, open(CACHE, 'w'), ensure_ascii=False, indent=1)
            print(f'  {i}/{len(todo)}  ok={ok} missing={fail}')
    json.dump(cache, open(CACHE, 'w'), ensure_ascii=False, indent=1)

    have = {k: v for k, v in cache.items() if v}
    print(f'\n{len(have)} norms with text, {len(cache) - len(have)} unresolved')
    if have:
        chars = sum(len(v['text']) for v in have.values()) / len(have)
        print(f'mean {chars:.0f} chars per norm; written to {CACHE}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--list', dest='list_only', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--delay', type=float, default=DELAY)
    a = ap.parse_args()
    main(a.list_only, a.limit, a.delay)

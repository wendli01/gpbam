"""Build a retrievable *Landesrecht* corpus, not just the cited norms.

``fetch_landesrecht.py`` fetches the 330 state-law norms the reference solutions
happen to cite -- enough for an oracle condition, useless as a retrieval corpus,
because a retriever must be able to find norms nobody told it about.  This
fetches whole statutes.

Each law's landing page states its article range in the title ("BayBO ... Art.
1--84"), so enumeration needs no tree-walking: read the range, then request every
article in it.  Missing numbers simply 404 (ranges are not dense) and are
recorded so they are not retried.

Same terms as the targeted fetch: BAYERN.RECHT grants unrestricted use and
reuse; the database right under §§ 87a ff. UrhG is preserved, so this stays a
per-statute fetch of public norm texts rather than a copy of the collection, and
§ 60d UrhG privileges TDM for research.  One request per second, identifying
User-Agent, resumable.

Run from ``experiments/``::

    PYTHONPATH=analysis python analysis/build_landesrecht_corpus.py --plan
    PYTHONPATH=analysis python analysis/build_landesrecht_corpus.py
"""

import argparse
import json
import os
import re
import time

import requests
import lxml.html as LH

from fetch_landesrecht import BASE, UA, parse, SITE_ABBR

CACHE = 'analysis/out/landesrecht_corpus.json'
DELAY = 1.0

# The Bavarian codes German public-law examinations are argued out of. Ordered by
# how much of GPBam's citation mass they carry.
STATUTES = [
    'BayVerf', 'BayGO', 'BayVwVfG', 'BayPAG', 'BayBO', 'BayLStVG', 'BayLKrO',
    'BayVwZVG', 'BayVfGHG', 'BayAGVwGO', 'BayVersG', 'BayKAG', 'BayFwG',
    'BayBezO', 'BayKommZG', 'BayNatSchG', 'BayWG', 'BayStrWG', 'BayDSG',
    'BayEUG', 'BayVwVfGDV', 'BayImSchG',
]

RANGE_RE = re.compile(r'Art\.?\s*(\d+)\s*[-–—bis]+\s*(\d+)')


def article_range(session, abbr):
    """(first, last) article numbers from the statute's landing page title."""
    r = session.get(f'{BASE}{abbr}', timeout=30)
    if r.status_code != 200:
        return None
    doc = LH.fromstring(r.text)
    title = (doc.xpath('//title/text()') or [''])[0]
    head = ' '.join(doc.xpath('//h1//text()')[:3])
    for hay in (title, head, r.text[:6000]):
        m = RANGE_RE.search(hay)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if 0 < lo <= hi <= 400:
                return lo, hi
    return None


def main(plan_only=False, delay=DELAY, statutes=STATUTES, limit=None):
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    s = requests.Session()
    s.headers.update({'User-Agent': UA})

    ranges = {}
    for abbr in statutes:
        rng = article_range(s, abbr)
        time.sleep(delay)
        if rng:
            ranges[abbr] = rng
            print(f'  {abbr:<14} Art. {rng[0]}-{rng[1]}  ({rng[1] - rng[0] + 1} articles)')
        else:
            print(f'  {abbr:<14} no article range found on the landing page')
    total = sum(hi - lo + 1 for lo, hi in ranges.values())
    have = len([k for k, v in cache.items() if v is not None])
    print(f'\n{len(ranges)} statutes, {total} articles to try, {have} already cached '
          f'(~{(total - len(cache)) * delay / 60:.0f} min at {delay:.1f}s/request)')
    if plan_only:
        return

    ok = miss = 0
    for abbr, (lo, hi) in ranges.items():
        for n in range(lo, hi + 1):
            key = f'{abbr}|{n}'
            if key in cache:
                continue
            try:
                r = s.get(f'{BASE}{abbr}-{n}', timeout=30)
            except Exception as e:
                print(f'  {key}: {type(e).__name__}')
                time.sleep(delay)
                continue
            time.sleep(delay)
            heading, text = (parse(r.text) if r.status_code == 200 else (None, None))
            if text and len(text) > 40:
                cache[key] = dict(law_book=abbr, paragraph=f'Art. {n}', title=heading,
                                  text=text, source=f'{BASE}{abbr}-{n}')
                ok += 1
            else:
                cache[key] = None
                miss += 1
            if (ok + miss) % 50 == 0:
                json.dump(cache, open(CACHE, 'w'), ensure_ascii=False)
                print(f'  {abbr}: {ok} fetched, {miss} absent')
            if limit and ok >= limit:
                break
        json.dump(cache, open(CACHE, 'w'), ensure_ascii=False)
        if limit and ok >= limit:
            break

    json.dump(cache, open(CACHE, 'w'), ensure_ascii=False)
    kept = {k: v for k, v in cache.items() if v}
    print(f'\n{len(kept)} articles with text, {len(cache) - len(kept)} absent')
    if kept:
        print(f'mean {sum(len(v["text"]) for v in kept.values()) / len(kept):.0f} chars; '
              f'written to {CACHE}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', dest='plan_only', action='store_true')
    ap.add_argument('--delay', type=float, default=DELAY)
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    main(a.plan_only, a.delay, limit=a.limit)

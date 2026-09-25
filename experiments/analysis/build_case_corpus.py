"""Build the Bavarian court-decision corpus that sits beside the statute corpus.

Why a second corpus at all: the statute knowledge base cannot supply the words
lawyers actually reason with.  ``Rücksichtnahmegebot`` and ``Störerauswahl``
occur *zero* times in 163 M characters of statute text, while 38.5% of court
decisions contain such doctrinal vocabulary against 0.39% of statute passages.
Decisions also carry an explicit norm chain, which makes a later two-hop
retrieval step possible -- find a decision by its doctrine, then follow its
citations to the norms -- so the norm chain is preserved verbatim rather than
folded into the text.

Two sources, chosen because they abut in time.  The second is a snapshot of the
first's database from years it has since dropped:

*Urchs, Mitrović & Granitzer (ICAART 2021)*, Zenodo ``10.5281/zenodo.3936726``,
CC-BY-4.0 -- 32,748 Bavarian decisions from 2015-2020, of which 22,155 sit in
the seven courts the 81 GPBam cases come from.  One download, no crawling.

*BAYERN.RECHT / gesetze-bayern.de*, the Bayerische Staatskanzlei portal --
everything from 2021-01-01, which is the portal's hard floor.  ``robots.txt`` is
``Allow: /``.  The portal serves each decision as structured XML at
``/Content/Zip/{doknr}``; HTML scraping is never necessary and is not done.

**Leitsätze are separated, not discarded.**  A Leitsatz is a ~320-character
abstract rule statement attached to an explicit Normenkette -- the only
doctrinal-register text either source contains, and the register the statute
corpus and the raw judgment body both lack.  So every Leitsatz becomes its own
row in ``leitsaetze.parquet``, classified by authorship, and *none* of them is
folded into ``text`` or into the body chunks.  That keeps the body index clean
and the Leitsatz index independently licensable.

**Terms.** BAYERN.RECHT states that its "Vorschriften und Entscheidungen"
"können örtlich, zeitlich und inhaltlich uneingeschränkt genutzt und
weiterverwendet werden".  That grant does *not* cover the C.H. Beck *redaktionelle
Leitsätze*.  The distinction matters for redistribution, not for use:

* ``amtlich`` -- § 5 Abs. 1 UrhG puts "amtlich verfasste Leitsätze zu
  Entscheidungen" outside copyright entirely, so these rows are free to
  redistribute.
* ``redaktionell`` -- C.H. Beck's editorial work, often personally signed.
  Usable for our own text and data mining under § 60d UrhG as a research
  organisation on lawfully accessed material, but **not redistributable**.  The
  Zenodo deposit's CC-BY-4.0 does not cure this: the depositors could not
  license Beck's rights.
* ``unmarked`` -- no trailing marker; authorship undetermined, treat as
  non-redistributable.

Hence: tag them, use them, and keep the ``redaktionell`` and ``unmarked`` rows
out of any release artifact.  The editorial ``titelzeile`` headline is dropped
outright; it is not part of the output schema.
Requests are one per second, single-threaded, and identify themselves.

Nothing here loads a model, touches a GPU, or embeds anything.  The output is
plain parquet, ready for whatever embedder is chosen later.

Run from ``experiments/``::

    PYTHONPATH=analysis python analysis/build_case_corpus.py all
    PYTHONPATH=analysis python analysis/build_case_corpus.py zenodo     # download + extract
    PYTHONPATH=analysis python analysis/build_case_corpus.py harvest    # crawl BAYERN.RECHT
    PYTHONPATH=analysis python analysis/build_case_corpus.py normalise  # -> decisions.parquet
    PYTHONPATH=analysis python analysis/build_case_corpus.py chunk      # -> chunks.parquet
    PYTHONPATH=analysis python analysis/build_case_corpus.py coverage   # gold-norm coverage

Every stage is resumable and re-entrant.  ``harvest`` records each fetched
``doknr`` on disk, so an interruption costs only the request in flight; re-running
it picks up exactly where it stopped.
"""

import argparse
import datetime as dt
import gzip
import io
import json
import os
import re
import struct
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zlib
from collections import Counter, defaultdict
from http.cookiejar import CookieJar
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refs as refs_mod

# --------------------------------------------------------------------------
# paths and constants

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
OUT = EXPERIMENTS / 'data' / 'case_corpus'
GPBAM = EXPERIMENTS / 'data' / 'gpbam_w_rubric.json'

# Raw material lives outside the repo: a few hundred MB of zips and JSON that
# nothing downstream reads once the parquet files exist.
SCRATCH = Path(os.environ.get(
    'GPBAM_CORPUS_SCRATCH',
    Path(tempfile.gettempdir()) / 'gpbam-corpus-scratch'))

ZENODO_DIR = SCRATCH / 'zenodo'
ZENODO_URL = ('https://zenodo.org/api/records/3936726/files/'
              'German_Decision_Corpus.zip/content')
BR_DIR = SCRATCH / 'br'
BR_DOCS = BR_DIR / 'docs'
BR_HITS = BR_DIR / 'hitlists'

BASE = 'https://www.gesetze-bayern.de'
UA = ('GPBam-research/0.1 (academic legal-NLP benchmark; '
      'contact lorenz.wendlinger@th-deg.de)')
RATE = 1.05          # seconds between requests; the brief allows at most 1/s
BR_FIRST_YEAR = 2021  # the portal holds nothing older

# The seven courts the 81 exam cases are drawn from.  Keyed by the label Zenodo
# uses; BAYERN.RECHT is mapped onto the same labels from <gertyp>+<gerort>.
COURTS = {
    'VGH München':    'vgh_muenchen',
    'VG München':     'vg_muenchen',
    'VG Ansbach':     'vg_ansbach',
    'VG Augsburg':    'vg_augsburg',
    'VG Bayreuth':    'vg_bayreuth',
    'VG Regensburg':  'vg_regensburg',
    'VG Würzburg':    'vg_wuerzburg',
}

# --- Leitsatz authorship -------------------------------------------------
# The marker is the *trailing* parenthetical of the Leitsatz text.  Measured
# frequencies over the seven courts in the Zenodo dump: 33,966 "(redaktioneller
# Leitsatz)", 2,768 signed "(red. LS <name>)" across 13 named editors, 689 bare
# "(Rn. N)" back-references, 650 "(amtlicher Leitsatz)", 453 with no marker at
# all.  Matching only the first form -- as an earlier version of this script did
# -- keeps 2,768 signed editorial Leitsätze, which is the copyright-sensitive
# direction of the error, so the signed form is matched explicitly.
LS_RED = re.compile(r'\(\s*red(?:aktionelle[rs])?\.?\s*(?:LS|Leitsatz)\s*([^)]*?)\s*\)\s*$', re.I)
LS_AMT = re.compile(r'\(\s*amtliche[rs]?\s+Leitsatz\s*\)\s*$', re.I)
# A trailing "(Rn. 18 - 51)" with no authorship marker after it is the court's
# own back-reference into its Randnummern; Beck's editors always sign.  Note the
# ordering in classify_leitsatz: an editorial Leitsatz may carry *both*
# ("... (Rn. 18 – 51) (redaktioneller Leitsatz)"), so redaktionell is tested first.
LS_RN = re.compile(r'\(\s*Rn\.\s*[^)]*\)\s*$', re.I)


def classify_leitsatz(text, red_span=False):
    """``(kind, author, text)`` for one Leitsatz.

    *red_span* carries the portal's own ``<span class="redLS">`` flag, which is
    machine-set and therefore trusted over the textual marker.
    """
    text = text.strip()
    m = LS_RED.search(text)
    if m or red_span:
        author = (m.group(1) or '').strip() if m else ''
        # The signature is metadata, not part of the rule statement.
        return 'redaktionell', author, (text[:m.start()].strip() if m else text)
    m = LS_AMT.search(text)
    if m:
        return 'amtlich', '', text[:m.start()].strip()
    if LS_RN.search(text):
        # Kept in the text: the Randnummer range is provenance, not a signature.
        return 'amtlich', '', text
    return 'unmarked', '', text

CHUNK_TARGET = 1500   # characters; the statute passages average 1,440
CHUNK_MAX = 2200      # a unit longer than this is split on sentence ends
CHUNK_MIN = 200       # below this a chunk is merged forward rather than emitted


# --------------------------------------------------------------------------
# a polite, resumable HTTP client


class Portal:
    """One-request-per-second session against BAYERN.RECHT.

    The portal keeps the search state (date range, court facet, page) in a
    session cookie rather than in the URL, so paging is only meaningful inside a
    live session -- hence a cookie jar rather than bare ``urlopen``.  A 429 or a
    403 aborts the run instead of backing off and retrying: the brief is to stop
    and report a block, not to work around one.
    """

    def __init__(self, rate=RATE):
        self.rate = rate
        self.jar = CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar))
        self.opener.addheaders = [('User-Agent', UA),
                                  ('Accept-Language', 'de-DE,de;q=0.9')]
        self._last = 0.0
        self.n = 0

    def _wait(self):
        gap = self.rate - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()

    def open(self, url, data=None, tries=4):
        for attempt in range(tries):
            self._wait()
            try:
                with self.opener.open(url, data, timeout=90) as r:
                    self.n += 1
                    return r.read()
            except urllib.error.HTTPError as e:
                if e.code in (429, 403):
                    raise SystemExit(
                        f'\nBAYERN.RECHT returned HTTP {e.code} on {url}.\n'
                        'Stopping as instructed rather than working around a block.\n'
                        f'{self.n} requests had been made.')
                if e.code >= 500 and attempt < tries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt < tries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise

    def get(self, path):
        return self.open(BASE + path)

    def search(self, datum_von, datum_bis):
        """Start a fresh search.  Returns the total hit count.

        The verification token is per-page and single-use, so it is re-read from
        the landing page for every search rather than cached.
        """
        home = self.get('/').decode('utf-8', 'replace')
        tok = re.search(r'__RequestVerificationToken"[^>]*value="([^"]*)"', home).group(1)
        body = urllib.parse.urlencode({
            '__RequestVerificationToken': tok,
            'SearchFields.Content': '',
            'SearchFields.Norm': '',
            'SearchFields.Aktenzeichen': '',
            'SearchFields.DatumVon': datum_von,
            'SearchFields.DatumBis': datum_bis,
        }).encode()
        return self.open(BASE + '/Search', body).decode('utf-8', 'replace')


def read_zip(blob):
    """Return ``{name: bytes}`` from a portal zip.

    The portal's zips carry a duplicated, inconsistent central directory that
    both ``unzip`` and :mod:`zipfile` reject ("bad magic number for central
    directory").  The local file headers are perfectly well-formed, so walk them
    directly and ignore the central directory entirely.
    """
    out, off = {}, 0
    while blob[off:off + 4] == b'PK\x03\x04':
        _, _, meth, _, _, _, csz, _, nl, el = struct.unpack('<HHHHHIIIHH', blob[off + 4:off + 30])
        name = blob[off + 30:off + 30 + nl].decode('utf-8', 'replace')
        start = off + 30 + nl + el
        raw = blob[start:start + csz]
        out[name] = zlib.decompress(raw, -15) if meth == 8 else raw
        off = start + csz
    return out


# --------------------------------------------------------------------------
# 1. Zenodo


def stage_zenodo(force=False):
    """Download and unpack the Urchs et al. dump.  Idempotent."""
    ZENODO_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = ZENODO_DIR / 'German_Decision_Corpus.zip'
    corpus = ZENODO_DIR / 'extracted' / 'Corpus'
    if corpus.is_dir() and len(os.listdir(corpus)) > 30000 and not force:
        print(f'zenodo: {len(os.listdir(corpus))} files already extracted')
        return corpus
    if not zip_path.exists() or force:
        print(f'zenodo: downloading {ZENODO_URL}')
        req = urllib.request.Request(ZENODO_URL, headers={'User-Agent': UA})
        with urllib.request.urlopen(req, timeout=300) as r, open(zip_path, 'wb') as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
    print('zenodo: extracting')
    # The dump's local filename fields are mis-encoded (cp437 vs utf-8), which
    # makes ``unzip`` exit non-zero; zipfile reads the central directory, where
    # the names are correct, so it is unaffected.
    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(ZENODO_DIR / 'extracted')
    print(f'zenodo: {len(os.listdir(corpus))} files')
    return corpus


# --------------------------------------------------------------------------
# 2. BAYERN.RECHT harvest

HIT_RE = re.compile(
    r'<li class="hitlistItem">.*?/Content/Document/(?P<dok>Y-[A-Za-z0-9.-]+)\?hl=true">'
    r'<b>(?P<title>.*?)</b>.*?<p class="hlSubTitel">\s*(?P<sub>.*?)\s*</p>',
    re.S)
SUB_RE = re.compile(r'^(?P<typ>.+?)\s+vom\s+(?P<datum>\d{2}\.\d{2}\.\d{4})\s*(?:&#x2013;|–|-)\s*(?P<az>.+)$')


def _unescape(s):
    import html
    return html.unescape(re.sub(r'<[^>]+>', '', s)).strip()


def harvest_hitlist(p, year):
    """All Verwaltungsgerichtsbarkeit hits for one calendar year.

    Searched a year at a time for two reasons: paging stays shallow (~400 pages),
    and a crash costs at most one year of hitlist paging.  The court facet is set
    to the whole Verwaltungsgerichtsbarkeit rather than to each of the seven
    courts in turn -- in 2021 the seven account for 3,969 of its 3,996 hits, so
    per-court faceting would save 0.7% of requests at the cost of seven times the
    session juggling.
    """
    BR_HITS.mkdir(parents=True, exist_ok=True)
    path = BR_HITS / f'{year}.jsonl'
    done = BR_HITS / f'{year}.done'
    if done.exists():
        return [json.loads(l) for l in open(path, encoding='utf-8')]

    today = dt.date.today()
    bis = min(dt.date(year, 12, 31), today)
    p.search(f'01.01.{year}', bis.strftime('%d.%m.%Y'))
    html_ = p.get('/Search/Filter/LEVEL1RSPRTREENODE/Verwaltungsgerichtsbarkeit').decode('utf-8', 'replace')
    m = re.search(r'id="readable">\s*([\d.]+)\s*Treffer', html_)
    total = int(m.group(1).replace('.', '')) if m else 0
    pages = (total + 9) // 10
    print(f'  {year}: {total} Verwaltungsgerichtsbarkeit hits -> {pages} pages')

    rows, seen = [], set()
    with open(path, 'w', encoding='utf-8') as f:
        for pg in range(1, pages + 1):
            # The facet response is already page 1, so it is reused rather than
            # re-requested -- one saved request per year, and one fewer chance to
            # desynchronise the session's paging cursor.
            page = (html_ if pg == 1 else
                    p.get(f'/Search/Page/{pg}').decode('utf-8', 'replace'))
            got = 0
            for hit in HIT_RE.finditer(page):
                dok = hit.group('dok')
                got += 1
                if dok in seen:
                    continue
                seen.add(dok)
                sub = _unescape(hit.group('sub'))
                sm = SUB_RE.match(sub)
                row = dict(doknr=dok, title=_unescape(hit.group('title')),
                           doktyp=sm.group('typ') if sm else None,
                           date=sm.group('datum') if sm else None,
                           aktenzeichen=sm.group('az') if sm else None)
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
            if got == 0:
                print(f'    page {pg}: no hits parsed, stopping this year')
                break
            if pg % 50 == 0:
                print(f'    page {pg}/{pages}  ({len(rows)} unique)', flush=True)
    done.write_text(dt.datetime.now().isoformat())
    return rows


def harvest(limit=None, years=None):
    """Crawl the portal.  Safe to interrupt and re-run at any point."""
    BR_DOCS.mkdir(parents=True, exist_ok=True)
    p = Portal()
    today = dt.date.today()
    years = years or list(range(BR_FIRST_YEAR, today.year + 1))

    print('BAYERN.RECHT: collecting hitlists')
    hits = []
    for y in years:
        hits += harvest_hitlist(p, y)
    # The hitlist titles carry the court, which is what lets the seven target
    # courts be selected before spending a request on the document itself.
    want, skipped = [], Counter()
    for h in hits:
        court = h['title'].split(':', 1)[0].strip() if ':' in h['title'] else ''
        if court in COURTS:
            want.append(h)
        else:
            skipped[court] += 1
    print(f'BAYERN.RECHT: {len(hits)} hits, {len(want)} in the seven target courts')
    print('  skipped courts:', dict(skipped.most_common(8)))

    todo = [h for h in want if not (BR_DOCS / f"{h['doknr']}.xml.gz").exists()]
    print(f'BAYERN.RECHT: {len(want) - len(todo)} already on disk, {len(todo)} to fetch')
    if limit:
        todo = todo[:limit]
    t0 = time.time()
    for i, h in enumerate(todo, 1):
        blob = p.open(f"{BASE}/Content/Zip/{h['doknr']}")
        parts = read_zip(blob)
        xml = next((v for k, v in parts.items()
                    if k.endswith('.xml') and 'META-INF' not in k), None)
        if xml is None:
            print(f"  !! no xml in {h['doknr']}")
            continue
        with gzip.open(BR_DOCS / f"{h['doknr']}.xml.gz", 'wb') as f:
            f.write(xml)
        if i % 200 == 0:
            rate = i / (time.time() - t0)
            eta = (len(todo) - i) / rate / 3600
            print(f'  {i}/{len(todo)}  {rate:.2f}/s  ETA {eta:.1f} h', flush=True)
    print(f'BAYERN.RECHT: done, {len(os.listdir(BR_DOCS))} documents on disk')


# --------------------------------------------------------------------------
# 3. normalise both sources onto one schema

WS = re.compile(r'[ \t   ]+')
TAG = re.compile(r'<[^>]+>')


def clean(s):
    if not s:
        return ''
    import html
    s = html.unescape(s)
    s = WS.sub(' ', s)
    return s.strip()


def norm_az(az):
    """Fold an Aktenzeichen for duplicate detection.

    The two sources disagree on separators -- Zenodo's filenames turn ``/`` into
    ``$`` and its ``file_number`` uses ``/`` where the portal uses ``.`` in some
    Senate numbers -- and on spacing.  Case, whitespace and all punctuation are
    therefore removed before comparing; what remains is the alphanumeric core,
    which both sources agree on.
    """
    return re.sub(r'[^0-9a-z]', '', str(az).lower())


def parse_br(path):
    """One BAYERN.RECHT decision XML -> the common record.

    Randnummern (``<rd nr="N"/>``) are preserved as an explicit marker in the
    text so the chunker can split on them; they are the court's own paragraph
    numbering and the only reliable structural boundary in the Gründe.
    """
    x = gzip.open(path, 'rb').read().decode('utf-8-sig', 'replace')

    def meta(tag):
        m = re.search(rf'<{tag}[^>]*>(.*?)</{tag}>', x, re.S)
        return clean(TAG.sub('', m.group(1))) if m else ''

    dok = re.search(r'doknr="([^"]*)"', x)
    gertyp, gerort = meta('gertyp'), meta('gerort')
    court = f'{gertyp} {gerort}'.strip()

    norm_chain = [clean(TAG.sub('', e)) for e in re.findall(r'<enbez>(.*?)</enbez>', x, re.S)]
    norm_chain = [n for n in dict.fromkeys(norm_chain) if n and n.lower() != 'keine']

    # Leitsätze never enter the body: they are lifted out into their own rows.
    leitsaetze = []
    for m in re.finditer(r'<leitsatz>(.*?)</leitsatz>', x, re.S):
        frag = m.group(1)
        body = clean(TAG.sub('', RD.sub('', frag)))
        if not body:
            continue
        kind, author, txt = classify_leitsatz(body, red_span='class="redLS"' in frag)
        if txt:
            leitsaetze.append(dict(ls_kind=kind, ls_author=author, text=txt))

    parts = []
    for tag in ('tenor', 'tatbestand', 'gruende'):
        m = re.search(rf'<{tag}>(.*?)</{tag}>', x, re.S)
        if m:
            parts += _paragraphs(m.group(1))

    return dict(
        doc_id=dok.group(1) if dok else path.stem,
        source='bayern.recht',
        court=court,
        court_slug=COURTS.get(court, ''),
        date=meta('entsch-datum'),
        aktenzeichen=meta('aktenzeichen'),
        ecli=None,                      # the portal's XML carries no ECLI element
        doktyp=meta('doktyp'),
        norm_chain=norm_chain,
        _paras=parts,
        _leitsaetze=leitsaetze,
        has_leitsatz_removed=bool(leitsaetze),
        n_leitsaetze=len(leitsaetze),
    )


RD = re.compile(r'<rd\s+nr="(\d+)"\s*/?>')


def _paragraphs(fragment):
    """``<p>`` elements of an XML section, each tagged with its Randnummer."""
    out = []
    for m in re.finditer(r'<p\b[^>]*>(.*?)</p>', fragment, re.S):
        body = m.group(1)
        rd = RD.search(body)
        txt = clean(TAG.sub('', RD.sub('', body)))
        if txt:
            out.append((int(rd.group(1)) if rd else None, txt))
    if not out:  # a section without <p>, e.g. a bare <div>
        txt = clean(TAG.sub('', RD.sub('', fragment)))
        if txt:
            out.append((None, txt))
    return out


def parse_zenodo(path):
    """One Urchs et al. JSON -> the common record.

    ``legal_facts`` is empty for most records -- a known defect of the dump, not
    of this parser -- so the text is usually Tenor plus Entscheidungsgründe only.
    ``decision_guidelines`` are the Leitsätze; they carry the same trailing
    authorship marker as the portal's and are lifted out the same way.
    """
    j = json.loads(Path(path).read_text(encoding='utf-8'))
    meta, txt = j.get('meta', {}), j.get('decision_text', {})
    court = clean(meta.get('court', ''))

    nc = meta.get('norm_chains') or []
    nc = [clean(n) for n in nc]
    nc = [n for n in dict.fromkeys(nc) if n and n.lower() != 'keine']

    leitsaetze = []
    for g in (meta.get('decision_guidelines') or []):
        g = clean(g)
        if not g:
            continue
        kind, author, body = classify_leitsatz(g)
        if body:
            leitsaetze.append(dict(ls_kind=kind, ls_author=author, text=body))

    parts = []
    for key in ('tenor', 'legal_facts', 'decision_reasons'):
        for para in (txt.get(key) or []):
            para = clean(para)
            if para:
                parts.append((None, para))

    d = meta.get('date', '')
    iso = f'{d[6:10]}-{d[3:5]}-{d[0:2]}' if re.match(r'\d{2}\.\d{2}\.\d{4}$', d or '') else d
    return dict(
        doc_id='zenodo:' + Path(path).stem,
        source='zenodo',
        court=court,
        court_slug=COURTS.get(court, ''),
        date=iso,
        aktenzeichen=clean(meta.get('file_number', '')),
        ecli=None,                      # the dump carries no ECLI field
        doktyp=clean(meta.get('decision_style', '')),
        norm_chain=nc,
        _paras=parts,
        _leitsaetze=leitsaetze,
        has_leitsatz_removed=bool(leitsaetze),
        n_leitsaetze=len(leitsaetze),
    )


def normalise():
    """Both sources -> ``decisions.parquet``, deduplicated on (court, Aktenzeichen)."""
    OUT.mkdir(parents=True, exist_ok=True)
    recs, stats = [], Counter()

    corpus = ZENODO_DIR / 'extracted' / 'Corpus'
    if corpus.is_dir():
        files = [f for f in os.listdir(corpus) if f.endswith('.json')]
        print(f'zenodo: parsing {len(files)} files')
        for i, f in enumerate(files):
            if f.split(',')[0] not in COURTS:      # cheap court filter on the filename
                continue
            r = parse_zenodo(corpus / f)
            if r['court'] not in COURTS:
                continue
            recs.append(r)
            if i % 5000 == 0:
                print(f'  {i}/{len(files)}', flush=True)
    else:
        print('zenodo: not downloaded, skipping')

    if BR_DOCS.is_dir():
        files = sorted(BR_DOCS.glob('*.xml.gz'))
        print(f'bayern.recht: parsing {len(files)} files')
        for i, f in enumerate(files):
            r = parse_br(f)
            if r['court'] not in COURTS:
                stats['br_offcourt'] += 1
                continue
            recs.append(r)
            if i % 2000 == 0:
                print(f'  {i}/{len(files)}', flush=True)
    else:
        print('bayern.recht: not harvested, skipping')

    df = pd.DataFrame(recs)
    if df.empty:
        raise SystemExit('nothing parsed')
    df['text'] = df['_paras'].map(lambda ps: '\n'.join(t for _, t in ps))
    df = df[df.text.str.len() > 0].copy()

    # --- de-duplicate.  Zenodo is 2015-2020 and the portal starts 2021, so a
    # cross-source collision should be rare; it is measured rather than assumed.
    df['az_key'] = df.aktenzeichen.map(norm_az)
    df['dup_key'] = df.court_slug + '|' + df.az_key
    before = len(df)
    within = df.duplicated(['dup_key', 'source']).sum()
    keys_by_src = df.groupby('dup_key').source.nunique()
    overlap = int((keys_by_src > 1).sum())
    # Longest text wins: the portal XML is complete where Zenodo often lost the
    # Tatbestand, and a duplicate inside one source is usually a truncated scrape.
    df['_len'] = df.text.str.len()
    df = (df.sort_values(['dup_key', '_len'], ascending=[True, False])
            .drop_duplicates('dup_key', keep='first'))
    print(f'dedup: {before} -> {len(df)}  '
          f'({within} duplicates within a source, {overlap} keys in both sources)')
    stats['dup_within'] = int(within)
    stats['dup_cross'] = overlap
    stats['dup_removed'] = before - len(df)

    # --- Leitsätze: one row each, carrying the parent's identity and norm chain
    # so a retrieved Leitsatz is traceable to its decision and to its norms
    # without a join.  They are not chunked -- the median is ~320 characters, so
    # a Leitsatz already *is* the retrieval unit.
    # ``itertuples`` renames leading-underscore columns to positional ``_1``,
    # so the parent rows are taken as dicts instead.
    ls_rows = []
    for r in df[['doc_id', 'source', 'court', 'court_slug', 'date',
                 'aktenzeichen', 'doktyp', 'norm_chain',
                 '_leitsaetze']].to_dict('records'):
        for k, ls in enumerate(r['_leitsaetze']):
            ls_rows.append(dict(
                leitsatz_id=f"{r['doc_id']}@ls{k:02d}", doc_id=r['doc_id'], ls_ix=k,
                ls_kind=ls['ls_kind'], ls_author=ls['ls_author'],
                source=r['source'], court=r['court'], court_slug=r['court_slug'],
                date=r['date'], aktenzeichen=r['aktenzeichen'], doktyp=r['doktyp'],
                norm_chain=list(r['norm_chain']), text=ls['text'],
                n_chars=len(ls['text'])))
    ls = pd.DataFrame(ls_rows)
    ls.to_parquet(OUT / 'leitsaetze.parquet', index=False)
    print(f'\nleitsaetze.parquet: {len(ls)} Leitsätze from '
          f'{ls.doc_id.nunique()} decisions, median {ls.n_chars.median():.0f} chars')
    print(ls.groupby(['source', 'ls_kind']).size().unstack(0, fill_value=0))
    named = ls[ls.ls_author != ''].ls_author.value_counts()
    print(f'named editors ({len(named)}):', dict(named.head(15)))
    for k, v in ls.ls_kind.value_counts().items():
        stats[f'leitsatz_{k}'] = int(v)
    stats['leitsatz_total'] = len(ls)

    cols = ['doc_id', 'source', 'court', 'court_slug', 'date', 'aktenzeichen',
            'ecli', 'doktyp', 'norm_chain', 'text', 'has_leitsatz_removed',
            'n_leitsaetze']
    paras = df.set_index('doc_id')['_paras'].to_dict()
    df[cols].to_parquet(OUT / 'decisions.parquet', index=False)
    # The paragraph/Randnummer structure is what the chunker splits on, and it
    # cannot be recovered from the flattened text, so it is carried separately.
    with gzip.open(OUT / '_paragraphs.jsonl.gz', 'wt', encoding='utf-8') as f:
        for k, v in paras.items():
            f.write(json.dumps([k, v], ensure_ascii=False) + '\n')
    (OUT / 'build_stats.json').write_text(json.dumps(dict(stats), indent=2))

    print(f'\nwritten {OUT / "decisions.parquet"}: {len(df)} decisions')
    print(df.groupby(['source', 'court']).size().unstack(0, fill_value=0))
    return df


# --------------------------------------------------------------------------
# 4. chunk

SENT = re.compile(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ„(§])')


def _split_long(text, target=CHUNK_TARGET):
    """Break an over-long paragraph on sentence ends, never mid-sentence."""
    sents, cur, out = SENT.split(text), '', []
    for s in sents:
        if cur and len(cur) + 1 + len(s) > target:
            out.append(cur)
            cur = s
        else:
            cur = f'{cur} {s}'.strip()
    if cur:
        out.append(cur)
    return out


def pack(paras, target=CHUNK_TARGET):
    """Greedily pack (Randnummer, paragraph) units into ~``target``-char chunks.

    Paragraphs are atomic: a chunk boundary only ever falls on a Randnummer or a
    paragraph break, which is why the units arrive pre-split rather than as one
    string.  The single exception is a paragraph longer than ``CHUNK_MAX``, which
    is split on sentence ends -- German judgments occasionally run a single
    Randnummer past 5,000 characters and one such chunk would dominate its own
    embedding.
    """
    units = []
    for rd, t in paras:
        if len(t) > CHUNK_MAX:
            units += [(rd, s) for s in _split_long(t, target)]
        else:
            units.append((rd, t))

    chunks, cur, rds = [], [], []
    for rd, t in units:
        have = sum(len(c) for c in cur) + len(cur)
        # Close the chunk when the next paragraph would overshoot -- unless the
        # chunk is still under half the target, in which case overshooting beats
        # emitting a stub.  German judgments alternate very short procedural
        # paragraphs with very long ones, and the strict rule leaves a long tail
        # of 200-500 character chunks that carry almost no retrievable signal.
        if cur and have + len(t) > target and have >= target // 2:
            chunks.append((cur, rds))
            cur, rds = [], []
        cur.append(t)
        if rd is not None:
            rds.append(rd)
    if cur:
        chunks.append((cur, rds))

    # A short tail chunk carries little signal on its own; fold it back.
    if len(chunks) > 1 and sum(len(c) for c in chunks[-1][0]) < CHUNK_MIN:
        last = chunks.pop()
        chunks[-1] = (chunks[-1][0] + last[0], chunks[-1][1] + last[1])
    return [('\n'.join(c), r) for c, r in chunks]


def chunk():
    df = pd.read_parquet(OUT / 'decisions.parquet')
    paras = {}
    with gzip.open(OUT / '_paragraphs.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            k, v = json.loads(line)
            paras[k] = v

    rows = []
    for r in df.itertuples():
        ps = paras.get(r.doc_id) or [[None, r.text]]
        for i, (text, rds) in enumerate(pack([(p[0], p[1]) for p in ps])):
            rows.append(dict(
                chunk_id=f'{r.doc_id}#{i:03d}', doc_id=r.doc_id, chunk_ix=i,
                source=r.source, court=r.court, court_slug=r.court_slug,
                date=r.date, aktenzeichen=r.aktenzeichen, doktyp=r.doktyp,
                norm_chain=list(r.norm_chain), text=text,
                rd_first=rds[0] if rds else None, rd_last=rds[-1] if rds else None,
                n_chars=len(text)))
    out = pd.DataFrame(rows)
    out.to_parquet(OUT / 'chunks.parquet', index=False)
    q = out.n_chars.describe(percentiles=[.05, .25, .5, .75, .95])
    print(f'\nwritten {OUT / "chunks.parquet"}: {len(out)} chunks from {out.doc_id.nunique()} decisions')
    print(q.to_string())
    print(f'chunks/decision: mean {len(out)/out.doc_id.nunique():.1f}')
    print(f'total characters: {out.n_chars.sum():,}')
    return out


# --------------------------------------------------------------------------
# 5. gold-norm coverage
#
# The number that decides whether embedding this corpus is worth the GPU time.
# The gold norms come from the 81 reference solutions exactly as
# ``oracle_experiment.gold_sets()`` builds them, so the figure is comparable with
# the statute-corpus coverage already reported.

ENBEZ = re.compile(r'^(?P<book>.*?)\s*(?P<rest>(?:§§|§|Art\.?|Artikel)\s*\d.*)$', re.S)
ANCHOR_SPLIT = re.compile(r'(?=(?:§§|§|Art\.)\s*\d)')


def enbez_norms(enbez):
    """``(book, section)`` pairs from one norm-chain entry.

    The portal and the Zenodo dump both publish the chain book-first --
    ``VwGO § 47 Abs. 6``, ``BayLadSchlG Art. 2 Abs. 2`` -- while ``refs.extract``
    expects the citation form lawyers write, book last.  The entry is therefore
    re-ordered into citation form before extraction, so the corpus and the gold
    set go through one and the same parser.  Multi-section entries
    (``CoronaEinreiseV § 2 Nr. 3a, § 4 Abs. 1``) are split on their anchors first,
    because the shared trailing book only chains across a short span.
    """
    enbez = enbez.strip()
    m = ENBEZ.match(enbez)
    # A minority of entries are already written book-last (``§ 78 Abs. 3 AsylG``);
    # those need no reordering and go straight to the extractor.
    if not m or not m.group('book').strip(' ,;'):
        return {(refs_mod.corpus_key(c.book), c.section.lower())
                for c in refs_mod.canonicalise(refs_mod.extract(enbez))}
    book = m.group('book').strip(' ,;')
    out = set()
    for piece in ANCHOR_SPLIT.split(m.group('rest')):
        piece = piece.strip(' ,;')
        if not piece:
            continue
        for c in refs_mod.canonicalise(refs_mod.extract(f'{piece} {book}')):
            out.add((refs_mod.corpus_key(c.book), c.section.lower()))
    return out


def _text_norms(texts):
    """Worker: every norm cited anywhere in a batch of decision bodies."""
    out = set()
    for t in texts:
        for c in refs_mod.canonicalise(refs_mod.extract(t)):
            out.add((refs_mod.corpus_key(c.book), c.section.lower()))
    return out


def coverage(workers=6):
    d = json.loads(GPBAM.read_text(encoding='utf-8'))
    order = sorted(d['solutions'], key=lambda x: int(x))
    sols = [d['solutions'][k] for k in order]
    gold = {i: {(refs_mod.corpus_key(x.book), x.section.lower())
                for x in refs_mod.canonicalise(refs_mod.extract(s))}
            for i, s in enumerate(sols)}

    df = pd.read_parquet(OUT / 'decisions.parquet')
    ls = pd.read_parquet(OUT / 'leitsaetze.parquet')
    # Which decisions carry a Leitsatz at all, and which carry an *amtlich* one.
    # The second set is the redistributable index: whether it reaches the gold
    # norms on its own decides whether a releasable artifact is worth building.
    with_ls = set(ls.doc_id)
    with_amtlich = set(ls.loc[ls.ls_kind == 'amtlich', 'doc_id'])

    chain, in_text = set(), set()
    chain_ls, chain_amt = set(), set()
    chain_docs = defaultdict(int)
    for r in df.itertuples():
        for e in r.norm_chain:
            for n in enbez_norms(e):
                chain.add(n)
                chain_docs[n] += 1
                if r.doc_id in with_ls:
                    chain_ls.add(n)
                if r.doc_id in with_amtlich:
                    chain_amt.add(n)

    # Citations in the bodies are the expensive half -- ``refs.CITATION_RE`` runs
    # at roughly 0.2 MB/s and the bodies are ~600 MB -- so they are extracted in
    # parallel.  Deliberately fewer workers than cores: two training jobs already
    # own the machine.
    texts = df.text.tolist()
    batches = [texts[i::workers * 4] for i in range(workers * 4)]
    print(f'extracting citations from {sum(map(len, texts))/1e6:.0f} MB of decision '
          f'text with {workers} workers', flush=True)
    import multiprocessing as mp
    with mp.Pool(workers) as pool:
        for s in pool.imap_unordered(_text_norms, batches):
            in_text |= s

    all_gold = set().union(*gold.values())
    def split(s):
        st = {n for n in s if refs_mod.jurisdiction(n[0]) == 'state'}
        fed = {n for n in s if refs_mod.jurisdiction(n[0]) == 'federal'}
        unk = s - st - fed
        return st, fed, unk

    print(f'\n{len(df)} decisions, {len(chain)} distinct norms in their norm chains, '
          f'{len(in_text)} distinct norms cited anywhere in their text')
    print(f'{len(all_gold)} distinct gold norms across the 81 cases\n')

    print(f'{len(with_ls)} decisions carry a Leitsatz, {len(with_amtlich)} an amtlich one\n')

    rows = []
    for label, corpus_norms in (
            ('norm_chain', chain),
            ('norm_chain + text', chain | in_text),
            ('norm_chain, Leitsatz decisions only', chain_ls),
            ('norm_chain, amtlich-Leitsatz decisions only', chain_amt)):
        for jur, sel in zip(('all', 'state', 'federal', 'unknown'),
                            (all_gold,) + split(all_gold)):
            hit = sel & corpus_norms
            # Per-case token coverage: what share of a case's gold norms the
            # corpus cites, averaged over cases -- the shape the oracle
            # experiment reports, so the two are directly comparable.
            per = [len({n for n in g if refs_mod.jurisdiction(n[0]) == jur or jur == 'all'}
                       & corpus_norms) /
                   max(1, len({n for n in g if refs_mod.jurisdiction(n[0]) == jur or jur == 'all'}))
                   for g in gold.values()
                   if {n for n in g if refs_mod.jurisdiction(n[0]) == jur or jur == 'all'}]
            rows.append(dict(basis=label, jurisdiction=jur, gold_norms=len(sel),
                             covered=len(hit),
                             pct_distinct=100 * len(hit) / max(1, len(sel)),
                             pct_per_case=100 * float(np.mean(per)) if per else float('nan'),
                             n_cases=len(per)))
    cov = pd.DataFrame(rows)
    cov.to_csv(OUT / 'gold_coverage.csv', index=False)
    print(cov.to_string(index=False, float_format=lambda v: f'{v:.1f}'))

    # Which gold norms the corpus misses is the actionable half of the number.
    missed = sorted(all_gold - chain - in_text)
    Counter(b for b, _ in missed).most_common()
    print('\nmost-cited gold books the corpus never cites:',
          dict(Counter(b for b, _ in missed).most_common(12)))
    top = Counter()
    for i, g in gold.items():
        for n in g:
            top[n] += 1
    print('\ntop gold norms and whether the corpus cites them:')
    for n, c in top.most_common(15):
        print(f'  {n[0]:<12} {n[1]:<6} in {c:2d} cases  '
              f'chain={"y" if n in chain else "n"} text={"y" if n in in_text else "n"} '
              f'({chain_docs.get(n, 0)} decisions cite it in their chain)')
    return cov


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('stage', choices=['zenodo', 'harvest', 'normalise', 'chunk',
                                      'coverage', 'all'])
    ap.add_argument('--limit', type=int, help='harvest: stop after N documents')
    ap.add_argument('--years', type=int, nargs='*', help='harvest: restrict to these years')
    ap.add_argument('--workers', type=int, default=6,
                    help='coverage: parallel citation-extraction workers')
    a = ap.parse_args()
    if a.stage in ('zenodo', 'all'):
        stage_zenodo()
    if a.stage in ('harvest', 'all'):
        harvest(limit=a.limit, years=a.years)
    if a.stage in ('normalise', 'all'):
        normalise()
    if a.stage in ('chunk', 'all'):
        chunk()
    if a.stage in ('coverage', 'all'):
        coverage(workers=a.workers)


if __name__ == '__main__':
    main()

"""Citation extraction for GPBam descriptive statistics and retrieval diagnostics.

Why this exists next to ``src.scoring.legal_ref_similarity``: refex resolves a
citation only for law books in its 1104-entry gazetteer, plus a ``§``-only
fallback for unknown books.  ``GG`` is not in that gazetteer and the fallback
does not fire on the ``Art.`` form, so every ``Art. 12 Abs. 1 GG``,
``Art. 11 PAG``, ``Art. 21 BayVwVfG`` in the corpus is dropped silently.  German
public law cites its central norms exactly that way, so the descriptive
statistics below need an extractor that sees both citation forms.

This module is analysis-only.  It does not touch the scoring pipeline; nothing
in ``src/`` imports it.
"""

import re
from collections import namedtuple

Citation = namedtuple('Citation', 'book section form text start')

# --- citation grammar ------------------------------------------------------
# A citation is an anchor (§ / §§ / Art. / Artikel), a section number, an
# arbitrary run of subdivision qualifiers, and finally the law-book abbreviation.

_ANCHOR = r'(?P<anchor>§§|§|Artikel|Art\.|Art(?=\s))'
# A sub-article letter is a single letter that does not continue a word:
# "Art. 10a" is § 10a, but "Art. 10 oder ..." must not yield section "10o".
_SECT = r'(?P<sect>\d+(?:\s?[a-z](?![a-zäöüß]))?)'
# Subdivision qualifiers: Abs. 1, Satz 2, Nr. 3, Alt. 1, lit. a, roman numerals
# (the "Art. 12 I GG" style used throughout the reference solutions), bare
# numbers and single letters continuing an enumeration.
_QUAL = (r'(?:Abs(?:\.|atz)?|S\.|S(?:ätze|atz)|Halbs(?:\.|atz)?|Hs\.?|Nrn?\.?|Nummer|'
         r'Alt\.?|Var\.?|Fall|lit\.?|Buchst\.?|[IVXL]{1,5}|\d+\s?[a-z]?|[a-z]{1,2}\)?|'
         r'f\.|ff\.|und|bis|sowie|oder|,|;|'
         # connectors that chain several citations onto one trailing book:
         # "Art. 101 i. V. mit Art. 100 BV", "§ 80 V 1 bzw. § 80a VwGO"
         r'bzw\.|i\.\s?V\.\s?m\.|i\.\s?V\.\s?mit|iVm|§§|§|Art\.?)')
# The book abbreviation: capitalised, may carry a Land suffix (GO NRW, ASOG Bln)
# or a version marker the corpus also uses (BauGB 2004).
_LAND = r'(?:\s+(?:Bln|NRW|NW|BW|BY|BE|BB|HB|HH|HE|MV|NI|RP|SL|SN|ST|LSA|SH|TH))?'
# Tokens that look like a book but are subdivision markers.  Excluded inside the
# regex rather than afterwards, so the engine backtracks and keeps searching for
# the real book instead of discarding the whole citation.
_NOT_BOOK = (r'(?!(?:Abs|Absatz|Nrn?|Nummer|S|Satz|Sätze|Halbsatz|Hs|Alt|Var|Fall|'
             r'Rn|Rz|Anm|Ziff|Buchst|Anlage|Art|Artikel|Unterabs|UAbs|[IVXL]{1,5})\b)')
_BOOK = _NOT_BOOK + r'(?P<book>[A-ZÄÖÜ][A-Za-z0-9ÄÖÜäöüß-]{1,24})' + _LAND

# The qualifier run is non-greedy: the first token that can legitimately be a law
# book ends the citation.  Greedy matching would run "Art. 2 I GG in Verbindung
# mit ..." past GG and report "Verbindung" as the law book.
#
# The repetition is *bounded*, and that is not cosmetic.  ``_QUAL`` is an
# alternation whose branches can match the same text in several ways (``9`` is
# both ``\d+\s?[a-z]?`` and part of an enumeration), so an unbounded ``*?``
# around it partitions a long qualifier run exponentially many ways -- and the
# engine explores all of them whenever no law book follows.  Court Normenketten
# are exactly that input, because they are written book-*first*:
#
#     BauNVO §§ 1 III, V, VI, VII, IX, X, 2, 6 II, III, 7 II Nr. 2, 9, 13 | ...
#
# 196 characters of that took **80 seconds** to fail.  MAX_MID_CHARS already
# rejects anything longer than 60 characters, but it does so *after* the match,
# which is far too late.
#
# 25 is measured, not guessed.  A qualifier token can be a single comma, so 60
# characters can hold far more than a first guess suggests -- at a bound of 15,
# ``BauNVO §§ 1 III, V, VI, VII, IX, X, 2, 6 II, III, 7 II Nr. 2, 9, 13`` stops
# resolving.  Over the 81 reference solutions and 4,000 court Normenketten, a
# bound of 25 loses nothing at all against the unbounded pattern (1,064 gold
# norms and 3,366 chain norms, both exact) while costing 2.9 s where 30 costs
# 18.5 s and 40 costs 24.6 s.
CITATION_RE = re.compile(
    _ANCHOR + r'\s*' + _SECT +
    r'(?P<mid>(?:\s*' + _QUAL + r'){0,25}?)' +
    r'\s+' + _BOOK
)

# Tokens that can sit where a book abbreviation is expected but are not books.
NON_BOOKS = {
    'Abs', 'Absatz', 'Nr', 'Nrn', 'Nummer', 'S', 'Satz', 'Sätze', 'Halbsatz', 'Hs',
    'Alt', 'Var', 'Buchst', 'Rn', 'Rz', 'Anm', 'Fall', 'Var', 'Ziff', 'Anlage',
    'Nach', 'Der', 'Die', 'Das', 'Ein', 'Eine', 'Es', 'Im', 'In', 'Bei', 'Da', 'Denn',
    'Dies', 'Diese', 'Dieser', 'Dabei', 'Damit', 'Danach', 'Dann', 'Daher', 'Dazu',
    'Fraglich', 'Somit', 'Also', 'Zudem', 'Zwar', 'Aber', 'Auch', 'Alternativ',
    'Vgl', 'Siehe', 'Ergebnis', 'Hier', 'Hierzu', 'Hieraus', 'Folglich', 'Mithin',
    'Wenn', 'Weil', 'Wie', 'Wird', 'Ist', 'Sind', 'Kann', 'Könnte', 'Muss', 'Soll',
    'Er', 'Sie', 'Und', 'Oder', 'Sowie', 'Jedoch', 'Allerdings', 'Zunächst',
    'Insoweit', 'Insofern', 'Schließlich', 'Ferner', 'Weiterhin', 'Demnach',
    'Vorliegend', 'Fall', 'Sachverhalt', 'Klage', 'Antrag', 'Bescheid', 'Gericht',
}
# Roman numerals are Absatz markers, never books.
_ROMAN_RE = re.compile(r'^[IVXL]{1,5}$')

# The qualifier run may chain citations onto a shared trailing book, but only
# across a short span -- otherwise a stray "§ 5 ... § 900 BGB" 200 characters
# apart would be read as one reference.
MAX_MID_CHARS = 60


def _normalise_book(raw: str) -> str:
    """Canonical form for a law-book abbreviation.

    Case is normalised (``BayBo`` -> ``BayBO`` is not attempted; we lower-case for
    matching and keep a display form), and the Land suffix is folded into the
    abbreviation so ``GO NRW`` and ``GO`` do not merge.
    """
    return re.sub(r'\s+', ' ', raw.strip())


def extract(text: str):
    """Yield :class:`Citation` for every statutory reference in *text*.

    ``§§ 80, 80a VwGO`` yields one citation per section number; the enumeration
    is expanded from the qualifier run so multi-references are not lost.
    """
    if not isinstance(text, str):
        return
    for m in CITATION_RE.finditer(text):
        book = _normalise_book(m.group('book'))
        head = book.split()[0]
        if head in NON_BOOKS or _ROMAN_RE.match(head):
            continue
        # a book abbreviation is never a plain lower-case word
        if not re.match(r'^[A-ZÄÖÜ]', head):
            continue
        mid = m.group('mid') or ''
        if len(mid) > MAX_MID_CHARS:
            continue
        anchor = m.group('anchor')
        form = 'Art' if anchor.startswith('Art') else 'Par'
        sections = [re.sub(r'\s+', '', m.group('sect'))]
        # §§ 80, 80a VwGO -> expand the enumeration
        if anchor == '§§':
            for extra in re.findall(
                    r'(?:,|und|bis|sowie)\s*(\d+\s?[a-z]?)(?=\s*(?:,|und|bis|sowie|$|\s+[A-ZÄÖÜ]))', mid):
                sections.append(re.sub(r'\s+', '', extra))
        # "Art. 101 i. V. mit Art. 100 BV" / "§ 80 V 1 bzw. § 80a VwGO":
        # every anchor inside the qualifier run shares the trailing book.
        for extra in re.findall(r'(?:§§|§|Art\.?)\s*(\d+\s?[a-z]?)', mid):
            sections.append(re.sub(r'\s+', '', extra))
        for sect in dict.fromkeys(sections):
            yield Citation(book=book, section=sect, form=form,
                           text=m.group(0), start=m.start())


def canonicalise(citations):
    """Clean up PDF-extraction artefacts in a list of citations.

    The reference solutions are extracted from PDFs, so two artefacts recur:
    a superscript footnote number glued to the abbreviation (``GG9``,
    ``BayVwVfG3``) and a line-break hyphen (``Kosten-``, ``BVerf-``).  Trailing
    digits are stripped whenever the bare stem is itself an observed book;
    hyphen fragments cannot be repaired and are dropped.
    """
    citations = list(citations)
    vocab = {c.book.lower() for c in citations}
    out = []
    for c in citations:
        book = c.book
        if book.endswith('-'):
            continue
        stem = re.sub(r'\d+$', '', book)
        if stem != book and len(stem) >= 2 and stem.lower() in vocab:
            book = stem
        out.append(c._replace(book=book))
    return out


# --- Normenketten ----------------------------------------------------------
# Court decisions publish their norm chain book-*first* -- ``VwGO § 47 Abs. 6``,
# ``BayLadSchlG Art. 2 Abs. 2`` -- which is the opposite order from the citation
# form a Gutachten uses and the one CITATION_RE parses.  Handing a raw chain to
# ``extract`` therefore yields nothing at all, and on a long chain it yields
# nothing slowly, because there is no trailing book to terminate the qualifier
# run.  So reorder first, and split multi-section entries on their anchors: the
# shared trailing book only chains across MAX_MID_CHARS.

_CHAIN_SPLIT = re.compile(r'^(?P<book>.*?)\s*(?P<rest>(?:§§|§|Art\.?|Artikel)\s*\d.*)$', re.S)
_ANCHOR_SPLIT = re.compile(r'(?=(?:§§|§|Art\.)\s*\d)')


def _chain_pairs(entry: str):
    """Yield ``(corpus_key, section)`` from one chain entry, in written order."""
    entry = str(entry).strip()
    if not entry:
        return
    m = _CHAIN_SPLIT.match(entry)
    book = m.group('book').strip(' ,;|') if m else ''
    if not book:
        # already book-last, or not a citation at all
        for c in canonicalise(extract(entry)):
            yield corpus_key(c.book), c.section.lower()
        return
    for piece in _ANCHOR_SPLIT.split(m.group('rest')):
        piece = piece.strip(' ,;')
        if piece:
            for c in canonicalise(extract(f'{piece} {book}')):
                yield corpus_key(c.book), c.section.lower()


def chain_norms(entry: str):
    """``(corpus_key, section)`` pairs from one book-first norm-chain entry.

    ``BayDG Art. 6 Abs. 1, Art. 11 Abs. 1, Art. 14`` -> ``baydg`` x {6, 11, 14}.
    Entries already written book-last go straight to :func:`extract`.
    """
    return set(_chain_pairs(entry))


def chain_norms_ordered(entries):
    """Norm keys from several chain entries, first occurrence order preserved.

    Order matters where this is used: the keys become a ranked list of statutes
    to fetch, and the court's own ordering within a chain -- which leads with the
    norm the decision turns on -- is a better prior than an arbitrary set
    iteration would be.
    """
    out, seen = [], set()
    for entry in entries:
        for part in str(entry).split('|'):
            for key in _chain_pairs(part):
                if key not in seen:
                    seen.add(key)
                    out.append(key)
    return out


def norms(text: str):
    """Set of ``(book, section)`` pairs cited in *text*, case-folded on the book."""
    return {(c.book.lower(), c.section.lower()) for c in extract(text)}


def books(text: str):
    """Set of law books cited in *text*, case-folded."""
    return {c.book.lower() for c in extract(text)}


# --- jurisdiction ----------------------------------------------------------
# Which law books are state law.  Bavarian books dominate because the GPBam
# cases come from the Bavarian state examination; the rest cover the books the
# reference solutions actually cite.

STATE_BOOKS = {
    # Bavaria
    'bv', 'baybo', 'bayvwvfg', 'bayvwzvg', 'vwzvg', 'pag', 'poG'.lower(), 'lstvg',
    'go', 'baygo', 'lkro', 'bezo', 'agvwgo', 'baywg', 'bayversg', 'bayksg',
    'vfghg', 'baydsg', 'bayeug', 'bayhschg', 'bayhig', 'baynatschg', 'bayimschg',
    'baystrwg', 'bayabfg', 'sgk', 'kag', 'bayrs', 'bayvbl', 'aggvg', 'baygastv',
    'bayifsmv', 'bayltgescho', 'gltrg', 'bayfig', 'bayhg', 'bayrig',
    # other Länder that appear in public-law exam material
    'asog', 'sog', 'lbo', 'lbauo', 'bauo', 'polg', 'pog', 'sogmv', 'nposg',
    'lvwg', 'lvwvfg', 'gonrw', 'go nrw', 'lstvgnrw', 'ordbg', 'lvwvg',
}
FEDERAL_HINT = {
    'gg', 'vwgo', 'vwvfg', 'vwvg', 'baugb', 'baunvo', 'bimschg', 'bgb', 'stgb',
    'stpo', 'zpo', 'gvg', 'owig', 'bverfgg', 'beamtstg', 'bbg', 'gewo', 'gastg',
    'waffg', 'versammlg', 'stvo', 'stvg', 'fev', 'ifsg', 'tierschg', 'aufenthg',
    'rog', 'uvpg', 'tkg', 'telemediengesetz', 'urhg', 'gmbhg', 'brao', 'bnoto',
    'sgb', 'estg', 'ao', 'hgb', 'partg', 'bwahlg', 'aeuv', 'euv', 'grch',
}


# --- matching citations against the retrieval corpus -----------------------
# The corpus keys norms by ``jurabk``, which does not always equal the
# abbreviation lawyers cite.  Two systematic mismatches matter here: a version
# year is appended (``StVO 2013``), and the Baugesetzbuch is still filed under
# its 1960 abbreviation ``BBauG`` -- so a lexical search for "BauGB", the book
# 25 of the 81 cases turn on, cannot match the corpus at all.
CORPUS_ALIASES = {
    'baugb': 'bbaug',
    # BAYERN.RECHT slugs carry a Bay prefix the citations do not: the corpus says
    # BayPAG, every Gutachten says PAG. Folded to the cited form, which is also
    # what the retrieved passages are labelled with in the prompt.
    'baypag': 'pag', 'bayverf': 'bv', 'baygo': 'go', 'baylstvg': 'lstvg',
    'bayagvwgo': 'agvwgo', 'bayvfghg': 'vfghg', 'baylkro': 'lkro',
    'bayvwzvg': 'vwzvg', 'baykommzg': 'kommzg', 'baykag': 'kag',
    'baybezo': 'bezo',
    'bimschg': 'bimschg',
    'versammlg': 'versammlg',
    'bpolg': 'bgsg',
    'vwvg': 'vwvg',
}


def corpus_key(book: str) -> str:
    """Normalise a law-book abbreviation for comparison with corpus ``jurabk``."""
    b = re.sub(r'\s+(?:19|20)\d\d$', '', str(book).strip())
    # refex occasionally returns a book name with an embedded newline or a roman
    # numeral glued on ("vwgo\niii"); collapse all whitespace before matching
    b = re.sub(r'\s+', '', b).lower().replace('.', '')
    return CORPUS_ALIASES.get(b, b)


def jurisdiction(book: str, corpus_books=None) -> str:
    """Classify a law book as ``state``, ``federal`` or ``unknown``.

    When *corpus_books* (the set of ``jurabk`` values in the federal
    gesetze-im-internet dump) is supplied, corpus membership decides; the
    hand-curated sets only break ties for books absent from both.
    """
    b = book.lower()
    if b in STATE_BOOKS or b.startswith('bay'):
        return 'state'
    if corpus_books is not None and corpus_key(b) in corpus_books:
        return 'federal'
    if b in FEDERAL_HINT:
        return 'federal'
    return 'unknown'

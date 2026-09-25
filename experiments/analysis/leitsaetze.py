"""Extract and index the Leitsätze -- the corpus's only doctrinal-register text.

A Leitsatz is one abstracted rule statement, ~320 characters, attached to an
explicit Normenkette::

    Der Bebauungszusammenhang am Ortsrand endet an den letzten, mit den übrigen
    Häusern im Zusammenhang stehenden Baukörpern, es sei denn, topografische
    Besonderheiten rechtfertigen eine Abweichung.   [BauGB § 35]

That is a commentary Randnummer in all but name, and it is the register both
other corpora lack: a statute states a rule without naming the situations it
governs, a decision names the situation but buries the rule in twenty thousand
characters of procedural history.  The Leitsatz is the join, and it is already
keyed to the norms -- which makes it a retrieval unit and a two-hop bridge at
the same time.

Provenance, and why every row carries it
----------------------------------------
Two kinds, and the distinction is legal rather than editorial:

``amtlich``
    The court's own, marked ``(amtlicher Leitsatz)`` or by a ``(Rn. N)``
    back-reference into its own decision.  § 5 Abs. 1 UrhG puts "amtlich
    verfasste Leitsätze zu Entscheidungen" outside copyright entirely, so these
    are freely redistributable.
``redaktionell``
    Written by C.H. Beck's editors, marked ``(redaktioneller Leitsatz)`` or
    signed ``(red. LS <Name>)``.  These are the editors' own work and are
    protected; BAYERN.RECHT's otherwise unrestricted grant carves them out by
    name, and the Zenodo deposit's CC-BY-4.0 does not cure that, because the
    depositors could not license rights they do not hold.

They are ~95% of the material, so dropping them is not an option and neither is
shipping them.  The position this module implements: use them under § 60d UrhG
(text and data mining for scientific research, on lawfully accessed material,
at a research organisation), tag every row, and keep the text out of any
release artifact -- publish norm chains, document ids and this code instead.
Tagging also buys the ablation for free: ``--kinds amtlich`` builds the
redistributable subset, and the gap between the two is the price of the licence.

Run from ``experiments/``::

    python analysis/leitsaetze.py extract --src <GDC Corpus dir>
    python analysis/leitsaetze.py build
"""

import glob
import json
import os
import re
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
OUT = 'data/case_corpus'
#: written here rather than to ``leitsaetze.parquet`` so it cannot collide with
#: the harvester's own output, which will add the BAYERN.RECHT years later
PARQUET = f'{OUT}/leitsaetze_zenodo.parquet'
DST = './my_leitsatz_index'

COURTS = ('VG München', 'VGH München', 'VG Würzburg', 'VG Augsburg',
          'VG Ansbach', 'VG Bayreuth', 'VG Regensburg', 'VerfGH München')

#: The editorial marker, both forms.  The signed variant is the one a regex for
#: "redaktionell" alone misses -- 2,768 of them in this corpus, and missing them
#: errs in the direction that matters, since a signed Leitsatz is the most
#: clearly authored of all.
RED = re.compile(r'\(\s*(?:redaktionelle[rs]?\s+Leitsat[wz][^)]*|red\.?\s*LS\b[^)]*)\)', re.I)
SIGNED = re.compile(r'\(\s*red\.?\s*LS\s+([^)]+?)\s*\)', re.I)
#: The court's own: either said outright, or a back-reference to the paragraph
#: of its own decision that the rule was drawn from.
AMT = re.compile(r'\(\s*(?:amtliche[rs]?\s+Leitsat[wz]|R[dn]n?\.?\s*\d[^)]*)\)', re.I)
#: Any trailing provenance parenthetical -- stripped from the embedded text,
#: because it is metadata about the sentence rather than part of the rule.
TRAIL = re.compile(r'\s*\((?:[^()]{2,60})\)\s*\Z')
#: Leading enumeration ("1. ", "2. ") -- an artefact of listing several
#: Leitsätze under one decision, not part of the rule either.
LEAD = re.compile(r'\A\s*\d{1,2}\s*[.)]\s*')

MIN_CHARS = 25


def classify(text):
    """``(kind, author)`` for one Leitsatz, from its trailing marker."""
    if RED.search(text):
        m = SIGNED.search(text)
        return 'redaktionell', (m.group(1).strip() if m else '')
    if AMT.search(text):
        return 'amtlich', ''
    return 'unmarked', ''


def clean(text):
    """The rule itself: no enumeration, no provenance parenthetical."""
    t = re.sub(r'\s+', ' ', str(text)).strip()
    t = LEAD.sub('', t)
    while True:
        t2 = TRAIL.sub('', t).strip()
        if t2 == t:
            return t
        t = t2


def extract(src, out=PARQUET, courts=COURTS):
    """Every Leitsatz in the dump, one row each, with its parent's metadata."""
    rows = []
    for f in sorted(glob.glob(os.path.join(src, '*.json'))):
        try:
            j = json.load(open(f, encoding='utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        m = j.get('meta') or {}
        if m.get('court') not in courts:
            continue
        chain = [re.sub(r'\s+', ' ', c).strip() for c in (m.get('norm_chains') or [])]
        chain = [c for c in dict.fromkeys(chain) if c and c.lower() != 'keine']
        for i, raw in enumerate(m.get('decision_guidelines') or []):
            if not raw or len(raw.strip()) < MIN_CHARS:
                continue
            kind, author = classify(raw)
            text = clean(raw)
            if len(text) < MIN_CHARS:
                continue
            rows.append(dict(
                ls_id=f'zenodo:{os.path.basename(f)[:-5]}#LS{i}',
                doc_id=f'zenodo:{os.path.basename(f)[:-5]}',
                source='zenodo', court=str(m.get('court') or ''),
                date=str(m.get('date') or ''),
                aktenzeichen=str(m.get('file_number') or ''),
                doktyp=str(m.get('decision_style') or ''),
                titel=str(m.get('title') or ''),
                norm_chain=' | '.join(chain), ls_ix=i, ls_kind=kind,
                ls_author=author, text=text))
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f'{len(df)} Leitsätze from {df.doc_id.nunique()} decisions -> {out}')
    print(df.ls_kind.value_counts().to_string())
    named = df[df.ls_author != '']
    if len(named):
        print(f'  signed by an editor: {len(named)}')
        print(named.ls_author.value_counts().head(8).to_string())
    print(f'  {df.text.str.len().sum() / 1e6:.1f}M characters, '
          f'median {df.text.str.len().median():.0f} per Leitsatz')
    return df


def normalise(df):
    """Accept either this module's extraction or build_case_corpus's.

    The harvester writes the same Leitsätze under a different schema -- and it
    is the one that carries BAYERN.RECHT, so it is the schema that matters going
    forward. Differences: ``leitsatz_id`` for ``ls_id``, ``norm_chain`` as a list
    of entries rather than a joined string, and no ``titel`` at all.
    """
    df = df.copy()
    if 'ls_id' not in df and 'leitsatz_id' in df:
        df['ls_id'] = df['leitsatz_id']
    if 'titel' not in df:
        df['titel'] = ''
    df['norm_chain'] = [' | '.join(x) if not isinstance(x, str) else x
                        for x in df['norm_chain']]
    df['titel'] = df['titel'].fillna('')
    return df


def title_of(r):
    """Identity plus Normenkette -- the part written in citation notation."""
    ident = f'{r.court}, {r.doktyp} v. {r.date} - {r.aktenzeichen}'.strip(' -,')
    if getattr(r, 'titel', ''):
        ident = f'{ident}: {r.titel}'
    return f'{ident} [{r.norm_chain}]' if r.norm_chain else ident


def build(parquet=PARQUET, dst=DST, model_name=MODEL, device='cuda', batch_size=64,
          char_budget=24000, write_batch=1000, resume=True, kinds=None):
    df = normalise(pd.read_parquet(parquet))
    if kinds:
        df = df[df.ls_kind.isin(kinds.split(','))]
    rows = []
    for r in df.itertuples(index=False):
        title = title_of(r)
        rows.append(dict(
            text=str(r.text), title=title, filename=str(r.doc_id),
            law_book=str(r.court), paragraph=f'{r.aktenzeichen} LS{r.ls_ix}',
            source_path=str(r.ls_id), court=str(r.court), date=str(r.date),
            doktyp=str(r.doktyp), norms=str(r.norm_chain), source=str(r.source),
            ls_kind=str(r.ls_kind), ls_author=str(r.ls_author),
            # The Normenkette leads: it is the one part in the same notation as
            # a query naming a norm, and putting it first means it survives the
            # truncation that would otherwise cut it off a long Leitsatz.
            embed_text=f'{title}\n{r.text}'[:MAX_CHARS]))
    rows.sort(key=lambda x: len(x['embed_text']))
    chars = sum(len(r['embed_text']) for r in rows)
    print(f'{len(rows)} Leitsätze, {chars / 1e6:.1f}M characters to embed', flush=True)

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
        ls_kind: str
        ls_author: str

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
    p.add_argument('cmd', choices=('extract', 'build'))
    p.add_argument('--src', help='extracted GDC Corpus/ directory (extract only)')
    p.add_argument('--parquet', default=PARQUET)
    p.add_argument('--dst', default=DST)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--char-budget', type=int, default=24000)
    p.add_argument('--write-batch', type=int, default=1000)
    p.add_argument('--kinds', default=None,
                   help='comma-separated ls_kind filter, e.g. "amtlich"')
    p.add_argument('--fresh', action='store_true')
    a = p.parse_args()
    if a.cmd == 'extract':
        if not a.src:
            raise SystemExit('--src is required for extract')
        extract(a.src, a.parquet)
    else:
        build(a.parquet, a.dst, device=a.device, batch_size=a.batch_size,
              char_budget=a.char_budget, write_batch=a.write_batch,
              resume=not a.fresh, kinds=a.kinds)

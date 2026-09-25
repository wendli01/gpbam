"""Rebuild my_knowledge_base_titled (federal) from my_knowledge_base_bayern_titled (combined).

The federal titled store holds 26,225 of its 106,354 rows: ``rebuild_kb_with_titles.py``
was interrupted on 2026-09-11, and its FTS index predates the last write, so every
federal search dies in LanceDB with "Added column's length must match table's length".
Re-embedding 80k rows would take hours on this card. It is not needed: the combined
store *is* the federal rows (``source_path`` under gesetze-im-internet, exactly 106,354)
plus 7,008 Bavarian ones, embedded by the same script with the same model and the same
``title + body[:4000]`` input.

What this does, in the order ``rebuild_kb_with_titles.py`` would have written it:
rows from the legacy federal store (``my_knowledge_base``), ``embed_text`` rebuilt the
same way, stable-sorted by its length; the vector for each row taken from the combined
store by (embed_text, source_path, paragraph, law_book); the combined table's schema
(which carries the registered embedding function); FTS over (title, text) as the
original. The broken store is moved aside, not deleted.

``--check`` writes nothing: counts, key multisets, and the vectors of the 26,225 rows the
interrupted rebuild did store, against the combined store's vectors for the same rows.

    python analysis/ds4_low/rebuild_federal_from_combined.py --check
    python analysis/ds4_low/rebuild_federal_from_combined.py            # rebuild + verify
"""
import json, os, sys, time
from collections import Counter
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import lancedb

os.chdir(os.path.join(os.environ.get('GPBAM_ROOT') or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'experiments'))
LEGACY, COMBINED, DST = './my_knowledge_base', './my_knowledge_base_bayern_titled', './my_knowledge_base_titled'
MARK = 'logs/high_gwdg_federal_rebuilt.json'
COLUMNS = ['text', 'title', 'filename', 'law_book', 'paragraph', 'source_path']
MAX_CHARS = 4000
CHECK = '--check' in sys.argv


def embed_text(title, text):
    title = str(title or '').strip()
    body = (text or '')[:MAX_CHARS]
    return f'{title}\n{body}' if title else body


def key(e, sp, pa_, lb):
    return (e, str(sp or ''), str(pa_ or ''), str(lb or ''))


def main():
    t0 = time.time()
    comb = lancedb.connect(COMBINED).open_table('documents')
    ct = comb.to_lance().to_table()
    fed_mask = pc.starts_with(ct['source_path'], 'data/gesetze-im-internet')
    cf = ct.filter(fed_mask)
    print(f'combined {ct.num_rows} rows, federal part {cf.num_rows}', flush=True)
    assert cf.num_rows == 106354, cf.num_rows

    legacy = lancedb.connect(LEGACY).open_table('documents').to_lance().to_table(columns=COLUMNS).to_pylist()
    print(f'legacy federal {len(legacy)} rows', flush=True)
    assert len(legacy) == 106354
    for r in legacy:
        r['embed_text'] = embed_text(r['title'], r['text'])
    legacy.sort(key=lambda r: len(r['embed_text']))            # stable, as rebuild_kb_with_titles

    cols = cf.to_pydict()
    vec_by_key = {}
    for i in range(cf.num_rows):
        k = key(cols['embed_text'][i], cols['source_path'][i], cols['paragraph'][i], cols['law_book'][i])
        vec_by_key.setdefault(k, cols['vector'][i])
    lk = Counter(key(r['embed_text'], r['source_path'], r['paragraph'], r['law_book']) for r in legacy)
    ck = Counter(key(cols['embed_text'][i], cols['source_path'][i], cols['paragraph'][i], cols['law_book'][i]) for i in range(cf.num_rows))
    missing = sum((lk - ck).values())
    print(f'row multisets: legacy-only {missing}, combined-only {sum((ck - lk).values())}', flush=True)
    assert missing == 0, 'legacy federal rows without a combined counterpart'

    report = dict(combined_rows=ct.num_rows, federal_rows=cf.num_rows, legacy_rows=len(legacy))
    if os.path.isdir(DST):
        broken = lancedb.connect(DST).open_table('documents')
        bt = broken.to_lance().to_table(columns=['embed_text', 'source_path', 'paragraph', 'law_book', 'vector']).to_pydict()
        nb = len(bt['embed_text'])
        prefix_ok = Counter(key(bt['embed_text'][i], bt['source_path'][i], bt['paragraph'][i], bt['law_book'][i]) for i in range(nb)) == \
            Counter(key(r['embed_text'], r['source_path'], r['paragraph'], r['law_book']) for r in legacy[:nb])
        diffs = np.array([np.abs(np.asarray(bt['vector'][i], np.float32) - np.asarray(vec_by_key[key(bt['embed_text'][i], bt['source_path'][i], bt['paragraph'][i], bt['law_book'][i])], np.float32)).max()
                          for i in range(nb)])
        cos = np.array([float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) for a, b in
                        ((np.asarray(bt['vector'][i], np.float32), np.asarray(vec_by_key[key(bt['embed_text'][i], bt['source_path'][i], bt['paragraph'][i], bt['law_book'][i])], np.float32)) for i in range(0, nb, 25))])
        print(f'interrupted federal store: {nb} rows, prefix of the rebuild order: {prefix_ok}; '
              f'vector |diff| vs combined: max {diffs.max():.2e}, median {np.median(diffs):.2e}; cosine min {cos.min():.6f}', flush=True)
        report.update(interrupted_rows=nb, interrupted_prefix_ok=bool(prefix_ok), vector_maxdiff=float(diffs.max()),
                      vector_mediandiff=float(np.median(diffs)), cosine_min=float(cos.min()))
    if CHECK:
        print(json.dumps(report), flush=True)
        return

    backup = f'{DST}.broken-2026-09-11'
    if os.path.isdir(DST):
        assert not os.path.exists(backup), f'{backup} exists'
        os.rename(DST, backup)
        print(f'moved the interrupted store to {backup}', flush=True)
    db = lancedb.connect(DST)
    tbl = db.create_table('documents', schema=comb.schema, mode='create')
    names = comb.schema.names
    for s in range(0, len(legacy), 5000):
        chunk = legacy[s:s + 5000]
        data = {n: [] for n in names}
        for r in chunk:
            r['vector'] = vec_by_key[key(r['embed_text'], r['source_path'], r['paragraph'], r['law_book'])]
            for n in names:
                data[n].append(r[n])
        tbl.add(pa.Table.from_pydict(data, schema=comb.schema))
    tbl.create_fts_index(['title', 'text'], replace=True)
    n = tbl.count_rows()
    assert n == 106354, n
    hits = tbl.search('Baugenehmigung im Außenbereich § 35 BauGB', query_type='hybrid').limit(40).to_pandas()
    print(f'rebuilt {DST}: {n} rows, fts(title,text); hybrid test query -> {len(hits)} hits, top {hits.title.head(3).tolist()}', flush=True)
    report.update(rebuilt_rows=n, test_hits=len(hits), backup=backup, seconds=round(time.time() - t0))
    json.dump(report, open(MARK, 'w'), indent=1)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

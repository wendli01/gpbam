"""One-off: fold results/<seat>/*.csv into results/<seat>.jsonl.

Kept in the tree for provenance -- it is the only record of how the merged seat
files were produced from the per-model ones, and it is idempotent, so running it
on an already-merged checkout is a no-op that reports as much.

    python merge_seat_files.py            # merge, verify, leave the old dirs
    python merge_seat_files.py --prune    # ... and delete them once verified

Verification is not optional: every merged file is read back through
seats.load_seat and compared against the CSVs cell by cell before the old
directory is eligible for deletion.
"""
import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seats import RESULTS, SEATS, load_seat, seat_path        # noqa: E402

COLS = ['index', 'model', 'score', 'n_matches', 'judgement', 'prompt_tokens',
        'completion_tokens', 'seconds', 'finish_reason', 'error']


def merge(seat, prune=False):
    src = RESULTS / seat
    dest = seat_path(seat)
    files = sorted(glob.glob(str(src / '*.csv')))
    if not files:
        print(f'  {seat}: no per-model CSVs -- already merged' if dest.exists()
              else f'  {seat}: nothing to merge and no {dest.name}')
        return
    old = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    old = old.sort_values(['model', 'index']).reset_index(drop=True)

    tmp = dest.with_suffix('.jsonl.tmp')
    with tmp.open('w', encoding='utf-8') as fh:
        for r in old.to_dict('records'):
            # NaN is not JSON; the CSVs carry it for "no error" and for scores
            # that failed to parse, and None round-trips back to NaN in pandas.
            rec = {c: (None if pd.isna(r.get(c)) else r.get(c)) for c in COLS}
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
    tmp.replace(dest)

    new = load_seat(seat).sort_values(['model', 'index']).reset_index(drop=True)
    assert list(new.columns) == COLS, f'{seat}: column drift {list(new.columns)}'
    bad = []
    for c in COLS:
        a, b = old[c], new[c]
        if not a.isna().equals(b.isna()):
            bad.append(f'{c}: NaN pattern differs')
        elif not (a[a.notna()].astype(str) == b[b.notna()].astype(str)).all():
            bad.append(f'{c}: values differ')
    assert not bad, f'{seat}: {bad}'
    print(f'  {seat}: {len(files)} CSVs, {len(old)} rows -> {dest.name}, verified')

    log = src / 'run.log'
    if log.exists():
        shutil.copy2(log, RESULTS / f'{seat}.run.log')
        print(f'      kept the run log as {seat}.run.log')
    if prune:
        shutil.rmtree(src)
        print(f'      removed {src}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--prune', action='store_true',
                    help='delete results/<seat>/ once the merge verifies')
    a = ap.parse_args()
    print(f'merging seat directories under {RESULTS}')
    for s in SEATS:
        merge(s, a.prune)

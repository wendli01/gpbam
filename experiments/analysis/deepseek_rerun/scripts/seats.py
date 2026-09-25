"""One place that knows where a re-judged seat's output lives, and how to read it.

Each seat used to be a directory of one CSV per model -- 30 files a seat, 90 in
the tree, which is how a merge request came to show 700 changed files with 609 of
them measurement output. A seat is one experiment, so it is now one file:

    results/<seat>.jsonl        one JSON object per (model, case)
    results/<seat>.run.log      that seat's run log, as before

The record is exactly the row the per-model CSV carried, so nothing was dropped
in the merge: index, model, score, n_matches, judgement, prompt_tokens,
completion_tokens, seconds, finish_reason, error.

jsonl rather than one big CSV because the file is appended as the run proceeds --
that is what makes the run resumable -- and a half-written JSON line is
detectable where a half-written CSV row is not. It is also what judge_0731.py
already writes for the generations judged outside the sweep, so the tree now has
one seat format instead of two.
"""
import json
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / 'results'

# Seat label -> the file its judging wrote. The labels are the ones the analysis
# scripts already use; only the storage changed.
SEATS = {
    'gpt-oss-120b': 'gpt-oss-120b',
    'Qwen3.6-35B': 'Qwen3.6-35B',
    'DeepSeek-V4-Flash-0731': 'DeepSeek-V4-Flash-0731',
}


def seat_path(seat, results=None):
    """The jsonl a seat's judging writes."""
    return (Path(results) if results else RESULTS) / f'{seat}.jsonl'


def load_seat(seat, usecols=None, results=None):
    """A seat's judgements as a DataFrame, optionally only some columns.

    `usecols` mirrors ``pd.read_csv(usecols=...)`` at the call sites this
    replaced: the judgement text is by far the largest field, so a caller that
    only wants scores should say so rather than read 15 MB of prose.
    """
    p = seat_path(seat, results)
    if not p.exists():
        raise FileNotFoundError(
            f'no seat file at {p}. If this is an old checkout with '
            f'results/{seat}/*.csv, run scripts/merge_seat_files.py once.')
    keep = set(usecols) | {'model', 'index'} if usecols else None
    rows = []
    with p.open(encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({k: r.get(k) for k in keep} if keep else r)
    d = pd.DataFrame(rows)
    # Sorted on read, so the analysis never depends on the order the run
    # happened to append in. The key is the slugged model name because that is
    # the order the per-model CSVs were globbed in before the merge, which keeps
    # regression output bit-identical across the change rather than merely equal
    # to the digits anyone prints.
    d = (d.assign(_k=d['model'].str.replace('/', '_', regex=False))
           .sort_values(['_k', 'index'], kind='stable')
           .drop(columns='_k')
           .reset_index(drop=True))
    return d[list(usecols)] if usecols else d


def done_pairs(seat, results=None):
    """(model, index) pairs already judged, so a re-run can resume."""
    p = seat_path(seat, results)
    if not p.exists():
        return set()
    done = set()
    with p.open(encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue          # a line torn by a kill; the pair re-runs
            done.add((r['model'], r['index']))
    return done

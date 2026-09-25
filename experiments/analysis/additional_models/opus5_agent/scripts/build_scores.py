"""Collapse the three per-seat judgement files into scores.csv.

scores.csv was originally built by hand, which is why re-judging extra cases
had nothing to re-run. This is that step, written down: read judged/*.jsonl,
pivot to index x seat, take the row-wise median over the three seats, and
report the mean over cases with its standard error.

The median is over exactly the three seats of the re-judged panel; a case
missing any seat is a hard error rather than a two-seat median, because a
two-seat median is a mean and would not be comparable with the other rows.

The judged files store the score on 0-1; every consumer of scores.csv and the
table itself are in percent, so it is scaled here and nowhere else.
"""
import json, pathlib, sys
import pandas as pd

D = pathlib.Path(__file__).resolve().parent.parent
SEATS = ['gpt-oss-120b', 'Qwen3.6-35B', 'DeepSeek-V4-Flash-0731']

frames = {}
for seat in SEATS:
    f = D / 'judged' / f'{seat}.jsonl'
    recs = [json.loads(l) for l in open(f, encoding='utf-8')]
    s = pd.Series({r['index']: r['score'] for r in recs}, name=seat) * 100
    frames[seat] = s
    bad = [r['index'] for r in recs if r.get('score') is None]
    if bad:
        print(f'{seat}: unparsed score at {bad}', file=sys.stderr)

df = pd.concat([frames[s] for s in sorted(SEATS)], axis=1).sort_index().round(6)
missing = df[df.isna().any(axis=1)]
if len(missing):
    raise SystemExit(f'incomplete panel for cases {list(missing.index)}:\n{missing}')

df['median'] = df[sorted(SEATS)].median(axis=1)
df.index.name = 'index'
df.to_csv(D / 'scores.csv')

n = len(df)
print(f'wrote {D / "scores.csv"}  n={n}')
print(f'ensemble median  {df["median"].mean():.2f}  '
      f'(sem {df["median"].std(ddof=1) / n ** .5:.2f}, n={n})')
print('per seat         ' + ', '.join(f'{s} {df[s].mean():.2f}' for s in SEATS))

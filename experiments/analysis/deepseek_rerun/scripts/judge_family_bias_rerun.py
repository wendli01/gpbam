"""Judge self-preference and vendor affinity, re-measured on the re-judged panel.

A configuration override on ``experiments/analysis/judge_family_bias.py``, not a copy
of it. Method, estimator, inference and table style all come from that module; this
file changes the panel underneath them and nothing else. Renamed so that importing the
original from a sibling directory cannot resolve back to this file.

What changed and what did not
-----------------------------
The published panel was ``gpt-5-nano`` / ``qwen3.6-35b-a3b`` / ``DeepSeek-V4-Flash``,
scored against a DeepSeek checkpoint NHR@FAU replaced on 2026-08-01. The new panel is
``gpt-oss-120b`` / ``Qwen3.6-35B-A3B`` / ``DeepSeek-V4-Flash-0731``, all three seats
re-judged in one window on 2026-08-19/20. The billed nano seat is gone, so the OpenAI
judge is now a model that is also in the generator zoo at a *different* version series,
and DeepSeek judges both its own weights and its own predecessor.

One condition, not two. The law-RAG arms have not been re-judged on these seats, so
pooling them would compare judge versions rather than retrieval conditions.

Results, against the published panel: self-preference survives and hardens (+9.94
consensus-percentile points, permutation p = 0.0010, against +9.6 at p = 0.004). The
vendor effect does not (+1.21, p = 0.51, n.s., against +5.5 at p = 0.016) --
``gpt-oss-120b`` shows -0.64 for its own vendor where ``gpt-5-nano`` showed +11.73.

``family*`` is a cleaner pair here than in the published run. It was ``gpt-5-mini``
judged by ``gpt-5-nano``; it is now the retired 0423 DeepSeek checkpoint judged by 0731
-- same vendor, same version series, genuinely different weights, which is what that
tier is meant to mean. 0423 and 0731 are therefore two model identities in ``TAX``, so
the 0731 judge scoring the published DeepSeek essays marks F, not M.

Run from anywhere::

    python experiments/analysis/deepseek_rerun/scripts/judge_family_bias_rerun.py
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'experiments/analysis'))
import judge_family_bias as fb  # noqa: E402  -- the published-panel module

RESULTS = ROOT / 'experiments/analysis/deepseek_rerun/results'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from seats import load_seat  # noqa: E402
GEN0731 = ROOT / 'experiments/zubaers_result/essay_writing/without_rag_0731'
DS0731 = 'deepseek-ai/DeepSeek-V4-Flash-0731'

# judge label -> (seat directory under RESULTS, its judgements of the 0731 re-generation)
SEAT_DIR = {
    'gpt-oss-120b':           ('gpt-oss-120b',           'gpt-oss-120b.jsonl'),
    'qwen3.6-35b-a3b':        ('Qwen3.6-35B',            'Qwen3.6-35B.jsonl'),
    'DeepSeek-V4-Flash-0731': ('DeepSeek-V4-Flash-0731', 'DeepSeek-V4-Flash-0731.jsonl'),
}

fb.CONDITIONS = [('no_rag', 'No RAG', None)]
fb.JUDGES = {
    'gpt-oss-120b':           ('OpenAI', 'gpt-oss', 'gpt-oss-120b'),
    'qwen3.6-35b-a3b':        ('Qwen', 'qwen3.6', 'qwen3.6-35b-a3b'),
    'DeepSeek-V4-Flash-0731': ('DeepSeek', 'deepseek-v4', 'deepseek-v4-flash-0731'),
}
fb.JUDGE_SHORT = {
    'gpt-oss-120b': 'gpt-oss-120b',
    'qwen3.6-35b-a3b': 'qwen3.6-35b',
    'DeepSeek-V4-Flash-0731': 'DeepSeek-V4-0731',
}
# 0423 and 0731 are the same vendor and the same version series but NOT the same
# weights -- that is the premise of this whole re-run -- so they are two model
# identities, and the 0731 judge scoring the 0423 essays is family*, not self.
fb.TAX = dict(fb.TAX)
fb.TAX['deepseek-ai/DeepSeek-V4-Flash'] = ('DeepSeek', 'deepseek-v4', 'deepseek-v4-flash-0423')
fb.TAX[DS0731] = ('DeepSeek', 'deepseek-v4', 'deepseek-v4-flash-0731')

_QUANT = re.compile(r'-FP8(-block|-Dynamic)?$', re.I)
_texttt = fb.texttt
fb.texttt = lambda s: _texttt(_QUANT.sub('', s))   # same compaction as the main table


def load_long():
    """Long-format panel, read from the three re-judged seats.

    Replaces the published-panel loader, which reads the ``score_Judge (...)`` columns
    of the two wide result CSVs. Each seat wrote one CSV per generator; the 0731
    re-generation was judged separately and lands as one more generator.
    """
    frames = []
    for label, (sub, jsonl) in SEAT_DIR.items():
        d = load_seat(sub, ['index', 'model', 'score'], results=RESULTS)
        frames.append(pd.DataFrame({'cond': 'no_rag', 'case': d['index'], 'gen': d['model'],
                                    'judge': label, 'score': d['score'] * 100}))
        j = pd.DataFrame([json.loads(l) for l in open(GEN0731 / 'judged' / jsonl)]
                         ).drop_duplicates('index')
        frames.append(pd.DataFrame({'cond': 'no_rag', 'case': j['index'], 'gen': DS0731,
                                    'judge': label, 'score': j['score'] * 100}))

    long = pd.concat(frames, ignore_index=True).dropna(subset=['score'])
    unmapped = set(long['gen']) - set(fb.TAX)
    assert not unmapped, f'generators missing from TAX: {sorted(unmapped)}'
    assert set(long['judge']) == set(fb.JUDGES), \
        f'judge labels {sorted(set(long["judge"]))} != {sorted(fb.JUDGES)}'
    n = long.groupby(['gen', 'judge']).size().unstack()
    assert n.notna().all().all(), f'ragged panel:\n{n[n.isna().any(axis=1)]}'

    long[['vendor', 'family', 'model']] = long['gen'].map(fb.TAX).apply(pd.Series)
    long[['j_vendor', 'j_family', 'j_model']] = long['judge'].map(fb.JUDGES).apply(pd.Series)
    long['tier'] = pd.Categorical([fb.tier_of(r) for r in long.itertuples()],
                                  categories=fb.TIERS)
    long['essay'] = long['cond'] + '|' + long['case'].astype(str) + '|' + long['gen']
    return long


fb.load_long = load_long

# The emitted table names the script that wrote it, and that stamp has to say *this*
# one -- the numbers in it are not the published-panel numbers the base module makes.
_HERE = 'experiments/analysis/deepseek_rerun/scripts/judge_family_bias_rerun.py'

# The caption has to name the panel. Without it a reader cannot tell this table from
# the published-panel one, which reports different numbers from the same estimator --
# and the family* mark means something different here, so it is spelled out too.
_PANEL = (
    r' The panel is the three seats re-judged in one window on 2026-08-19/20: '
    r'\texttt{gpt-oss-120b}, \texttt{Qwen3.6-35B-A3B} and '
    r'\texttt{DeepSeek-V4-Flash-0731}. The published panel\textquotesingle s '
    r'\texttt{gpt-5-nano} seat and its \texttt{DeepSeek-V4-Flash} checkpoint, which '
    r'NHR@FAU replaced on 2026-08-01, are not used here.')
_FAMILY = (
    r' The one $\mathrm{F}$ row is the retired 0423 DeepSeek checkpoint judged by 0731: '
    r'same vendor, same version series, different weights.')

_build_matrix = fb.build_matrix


def _matrix(*a, **k):
    out = []
    for l in _build_matrix(*a, **k):
        l = l.replace('experiments/analysis/judge_family_bias.py', _HERE)
        if l.endswith('for the No RAG condition.'):
            l += _PANEL
        elif l.endswith(r'blank means related to none of them.'):
            l += _FAMILY
        out.append(l)
    return out


fb.build_matrix = _matrix

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-matrix', default=str(RESULTS / 'judge_score_matrix.tex'))
    ap.add_argument('--csv', default=str(RESULTS / 'judge_family_bias.csv'))
    ap.add_argument('--csv-models', default=str(RESULTS / 'judge_family_bias_models.csv'))
    ap.add_argument('--n-perm', type=int, default=fb.N_PERM)
    ap.add_argument('--cmap', default=fb.CMAP, choices=sorted(fb.CMAPS))
    a = ap.parse_args()
    fb.CMAP = a.cmap
    fb.main(a.out_matrix, a.csv, a.csv_models, a.n_perm)

"""Recompute `legal_ref_sim` for every no-RAG essay with the currently installed refex.

Why this exists
---------------
The published column cannot be reproduced. Re-running `src.scoring.legal_ref_similarity`
over the published answers gives max |delta| 0.38 per essay on the DeepSeek run, and at
model level r = 0.955, tau-b = 0.848, with `gpt-oss-120b` moving 10.65 -> 5.58. The
citation extractor has changed since the sweep was scored.

That left the 0731 re-generation with no value at all: computing one row with today's
extractor while thirty came from an older one would confound extractor version with
model, which is the exact error this branch exists to correct. Recomputing all of them
removes the confound in the other direction -- one extractor, every row -- at the price
of moving published numbers.

Two corrections to the published metric, both of which affect every model equally
--------------------------------------------------------------------------------
1. A case whose *reference solution* has no extractable references carries no signal
   about citation agreement, so it is NaN here rather than scored. One of the 81 does
   (case 7) under the current extractor.

   Case 7 joined that list on 2026-08-21, once the empty reference below stopped
   counting as a reference. Cases 34, 73 and 74 were on it too until correction 5
   on 2026-08-22: their solutions were not reference-free, they were being discarded
   whole by the overlap check.

3. Unicode spacing. The extractor matches ``\u00a7 31 BauGB`` with an ASCII space and
   not with the typographic spaces (U+202F narrow no-break, U+00A0, thin spaces) that
   some models use inside citations. `gpt-oss-120b` averages 307 such characters per
   essay against at most 12.5 for any other model, and essentially none of its
   citations parsed: 601 raw references across its 81 essays, none of them usable.
   Its published 10.65 and the 4.51 recomputed without this correction are both
   artefacts, in opposite directions. Spacing is normalised to ASCII before extraction,
   which recovers 15 real references from its first essay alone.

4. The extractor emits a degenerate ``('', '')`` reference -- empty book, empty
   section. `src.scoring` keeps it, because its filter is ``!= None`` and ``''`` is
   not None, so two texts that both yield only that token score a *perfect* 1.0 for
   agreeing about nothing. Case 7's solution extracts to exactly that and nothing
   else, which is why seven models scored 1.0 on it. References are required here to
   have a non-empty book and section, which drops the token and, with it, case 7 into
   the undefined set.

2. That matters because `jaccard_similarity` returns 1.0 when both sets are empty --
   mathematically correct for Jaccard, wrong as a citation metric. It means a model
   that answers nothing scores a *perfect* citation match on exactly those three cases.
   `mistral-small-3.1`, whose 81 answers are 25-character stubs, scores 3.70% instead
   of 0 for that reason alone. Dropping the undefined cases removes it.

5. `RefExtractor.extract` collects the markers and then calls `replace_content`,
   which rewrites the text with ``[ref=...]`` spans and raises ``RefExError`` when two
   markers overlap. Neither this script nor `src.scoring` uses that rewritten text --
   only the markers -- so the check was discarding every reference in a document over
   one bad marker. Essay 61 of the Opus row scored 0.00 with 111 ``\u00a7`` and 247
   ``Art.`` citations in it, killed by one marker overlapping a ``____________``
   blank. 12 of the 2,754 essays were hit, and so were the reference solutions of
   cases 34, 73 and 74, which made those cases undefined for every model at once.
   Extraction now stops before `replace_content`. The markers are the same objects
   `extract` would have returned; only the text-rewriting step is skipped.

Corrections 3, 4 and 5 were folded into `src.scoring.legal_ref_similarity`, the first
two on 2026-08-21 and the third on 2026-08-22, so every caller gets them; the extraction is still repeated here because correction 1 --
NaN rather than a Jaccard 1.0 when both sides are empty -- is a property of this metric
rather than of Jaccard, and `jaccard_similarity` is shared. The two agree exactly on
every pair where either side has a reference.

    python recompute_legal_ref.py
"""
import contextlib
import io
import unicodedata
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.scoring import RefExError, RefExtractor  # noqa: E402

PUB = ROOT / 'experiments/zubaers_result/essay_writing/without_rag/ji2/no_rag_ji2_result.csv'
GEN0731 = ROOT / 'experiments/zubaers_result/essay_writing/without_rag_0731/ds_v4_flash_0731_high.jsonl'
FRONTIER = ROOT / 'experiments/zubaers_result/essay_writing/frontier'
OPUS = ROOT / 'experiments/analysis/additional_models/opus5_agent/opus5_agent.jsonl'
OUT = ROOT / 'experiments/analysis/deepseek_rerun/results/legal_ref_recomputed.csv'
DS0731 = 'deepseek-ai/DeepSeek-V4-Flash-0731'

# Generations that live outside the published CSV, each as (jsonl, model id). They
# go through the identical extractor call as every published row -- the whole point
# of this script is that no row gets a different one.
EXTRA = [
    (GEN0731, DS0731),
    (FRONTIER / 'gemini-3.7-flash_essays.jsonl', 'google/gemini-3.7-flash'),
    # `answer` here is already the text after `</think>` -- run_model.py splits
    # the trace off, so the extractor never sees the model's own monologue.
    (FRONTIER / 'soofi-s-isar-preview_essays.jsonl', 'soofi-s-isar-preview'),
    # All 81, but written through the agent harness rather than the API -- see
    # additional_models/opus5_agent/README.md. The extractor does not care how the
    # text was produced, and the table marks the row for the harness.
    (OPUS, 'anthropic/claude-opus-5'),
]

sol = {int(k): v for k, v in json.load(open(ROOT / 'experiments/data/gpbam.json'))['solutions'].items()}

# Every Unicode space that is not the ASCII one, mapped to it; and the zero-width
# characters dropped. See correction 3.
_SPACES = {c: ' ' for c in range(0x110000)
           if chr(c) != ' ' and unicodedata.category(chr(c)) == 'Zs'}
_SPACES.update({0x200b: None, 0x200c: None, 0x200d: None, 0xfeff: None})


def _normalise(text):
    return str(text).translate(_SPACES)


def _refs(text):
    """Statutory references in `text`, as (book, section).

    A fresh extractor per call, spacing normalised first, and both parts required
    to be non-empty: see corrections 3 and 4. This mirrors what
    `src.scoring.legal_ref_similarity` now does; it is repeated rather than reused
    so the undefined cases can be detected from the solutions alone.

    Not `extract()`: that runs the markers through `replace_content`, which
    rewrites the text and raises on overlapping markers -- discarding every
    reference in the document over one bad one. The rewritten text is not used
    here or in `src.scoring`. See correction 5.
    """
    rx = RefExtractor()
    try:
        content = rx.remove_markers(_normalise(text))
        markers = list(rx.extract_law_ref_markers(content, False))
        markers += list(rx.extract_case_ref_markers(content))
    except RefExError:
        return set()
    return {(r.book, r.section) for m in markers for r in m.get_references()
            if r.book and r.section}


def _jaccard(a, b):
    """Jaccard, undefined rather than 1.0 when both sides are empty."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# Which cases are undefined: the solution itself yields no extractable references.
with contextlib.redirect_stdout(io.StringIO()):
    sol_refs = {i: _refs(v) for i, v in sol.items()}
undefined = {i for i, r in sol_refs.items() if not r}
print(f'cases with no extractable references in the reference solution: {sorted(undefined)}')

pub = pd.read_csv(PUB, usecols=['index', 'model', 'answer', 'legal_ref_sim'])
frames = [pub]
for path, mid in EXTRA:
    if not path.exists():
        print(f'skipping {mid}: no {path}')
        continue
    g = pd.DataFrame([json.loads(l) for l in open(path)]).drop_duplicates('index')
    g = g[g.answer.notna()]
    frames.append(g.assign(model=mid, legal_ref_sim=np.nan)
                   [['index', 'model', 'answer', 'legal_ref_sim']])
    print(f'{mid}: {len(g)} essays from {path.name}')
d = pd.concat(frames, ignore_index=True)

t0 = time.time()
with contextlib.redirect_stdout(io.StringIO()):              # refex is chatty on odd markers
    d['refex_now'] = [np.nan if i in undefined else _jaccard(_refs(a), sol_refs[i])
                      for a, i in zip(d.answer, d['index'])]
print(f'{len(d)} essays in {time.time() - t0:.0f}s; '
      f'{int(d.refex_now.isna().sum())} undefined ({len(undefined)} cases x {d.model.nunique()} models)')

d[['index', 'model', 'legal_ref_sim', 'refex_now']].to_csv(OUT, index=False)
m = d.groupby('model').agg(published=('legal_ref_sim', 'mean'),
                           recomputed=('refex_now', 'mean')) * 100
m['delta'] = m.recomputed - m.published
print('\n' + m.round(2).sort_values('recomputed', ascending=False).to_string())

from scipy import stats                                              # noqa: E402
ok = m.dropna()
print(f'\nmodel level: mean delta {ok.delta.mean():+.2f}, max |delta| {ok.delta.abs().max():.2f}, '
      f'r = {stats.pearsonr(ok.published, ok.recomputed)[0]:.4f}, '
      f'tau-b = {stats.kendalltau(ok.published, ok.recomputed).statistic:.3f}, '
      f'rank moves {(ok.published.rank(ascending=False) != ok.recomputed.rank(ascending=False)).sum()}'
      f'/{len(ok)}')
print(f'wrote {OUT}')

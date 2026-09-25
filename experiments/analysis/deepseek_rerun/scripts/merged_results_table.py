"""One table where there were two: the main results table and the judge-bias table.

The paper carried the same models twice -- once in ``main_essay_table.tex``
(judge median, legal reference similarity, both recitation scores, AA index) and
once in ``judge_score_matrix.tex`` (per-judge scores, bias tier, debiased median,
delta). A reader comparing a model's debiased score against its recitation score
had to hold a row identity in their head across two pages. This merges them.

Neither source is recomputed here. Both are read as artefacts:

  ``main_essay_table.csv``          written by ``rebuild_tables.py``
  ``judge_family_bias_models.csv``  written by ``judge_family_bias_rerun.py``

so the merged table cannot disagree with either of its parents by construction,
and running this script costs nothing and calls no model.

Column structure
----------------
Three groups, separated by a wide gutter, with a narrower one inside the essay
group::

    Model || Med.  Deb.  D | gpt-oss  qwen3.6  DS-V4 | Ref. || GP  Cited || AA
              <-- ensemble -->  <---- per judge ---->

*Med.* is the reported metric: the per-case median over the three judges,
averaged over the cases. *Deb.* is the same median after each related judge's
tier effect is subtracted; *D* is the gap, and carries the bias tier as a
superscript -- M same model, F same family, V same vendor -- so relatedness costs
a superscript rather than a whole column. *D* is blank where no judge was related
at all, which is a different statement from the ``0.00`` of a row that was
corrected and did not move.

*GP* and *Cited* are the two recitation prompt sets: the laws the GPBam cases
cite, and the most-cited German laws generally. *Ref.* is legal reference
similarity. *AA* is the Artificial Analysis index, the one column not produced by
this study, which is why it gets a group of its own.

Rows that ``rebuild_tables.py`` reports but the bias analysis never saw -- the
models added after that analysis ran -- get ``Deb. = Med.`` and a blank *D*. That
is not a gap in the data: all of them are unrelated to all three judges, and the
debiasing step is exactly the identity on unrelated rows (the base module asserts
this). Re-fitting the spline with them included would perturb the tier
coefficients of the *other* rows, which is a different analysis than the one the
paper reports, so it is deliberately not done here.

Names
-----
Compacted so eleven columns fit the text width: vendor prefix and quantisation
suffix dropped, trailing weight dates dropped (``-0731``, ``-2506``,
``-2024-07-18``), and the trailing variant word abbreviated to one letter --
``-instruct``/``-it`` to ``-I``, ``-reasoning`` to ``-R``, ``-thinking`` to ``-T``.
Dropping the date makes ``DeepSeek-V4-Flash-0731`` print as ``DeepSeek-V4-Flash``,
which is only unambiguous because the retired 0423 checkpoint is not in this table;
the header comments say which checkpoint it is. Uniqueness is asserted, so a future
row that collides fails loudly instead of printing twice.

Colour
------
On by default, ``--no-colour`` to turn it off. Only columns whose *value is a
difference* are shaded, because shading a level column as well would put every
cell in the table on a fill and leave the eye nothing to land on:

  ``Deb.`` and ``D``   diverging on ``Deb. - Med.``
  per-judge columns    diverging on ``judge - median of the row's three judges``

Both use RdBu on a symmetric scale with white at zero, so the sign is the hue and
the magnitude is the saturation, and a cell that did not move stays paper-white.

The per-judge anchor is the row's own middle judge, not *Med.* Those are not the
same number: *Med.* averages a per-case median, while each judge column averages
that judge's own scores, and a mean of medians is not the median of means. Taking
*Med.* as the anchor left all three cells shaded and none of them at zero, which
read as though every judge disagreed with the panel. Against the row median
exactly one cell per row is white by construction -- the middle judge -- and the
other two carry the spread, which is what the column is for.

Scale anchor
------------
Not the observed maximum. Anchoring the ramp at ``max |value|`` means one outlier
row defines the scale for every other row, the endpoint colour is guaranteed to
be hit exactly once whatever the data looks like, and the ramp silently
recalibrates on every rebuild. ``SD_ANCHOR`` standard deviations is used instead.

Both quantities are deviations whose null is zero, not sample means, so the
dispersion is taken about zero -- a root-mean-square -- rather than about the
observed mean. Cells beyond the anchor saturate, which is the intended reading:
"off the scale", not "the largest one here".

Which cells enter the RMS differs between the two blocks, and the difference is
structural rather than a choice. For the per-judge block the middle judge's own
cell is zero *by definition*, so including 31 forced zeros would deflate the
scale by roughly sqrt(2/3); only the two deviating cells per row are counted. For
the delta block the ``0.00`` entries are measurements -- a related model whose
median happened not to move -- so they are counted.

Every shaded cell also prints its number, so colour is redundant encoding, never
the only channel, and the body text flips to white where black would fall below
4.5:1 on the fill.

Run from anywhere::

    python experiments/analysis/deepseek_rerun/scripts/merged_results_table.py
    python experiments/analysis/deepseek_rerun/scripts/merged_results_table.py --no-colour
    python experiments/analysis/deepseek_rerun/scripts/merged_results_table.py --sd 2.5
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / 'experiments/analysis/deepseek_rerun/results'

OPUS = 'anthropic/claude-opus-5'
OPUS_N = 81

# How many standard deviations from zero the ends of the colour ramp sit at.
SD_ANCHOR = 3.0

# (column in main_essay_table.csv, one-line label, the same label over two lines)
#
# These three columns are bound by their HEADERS, not by their numbers: at
# \scriptsize "GPT-oss" is 30.3pt and "Qwen3.6" 29.8pt against 18.2pt for the
# widest figure under them, "91.88". That overhang is 28.5pt of table width doing
# nothing, which is why --judge-header stacked exists: breaking each label over
# two lines leaves the columns bound by the data and gives 27.1pt back.
# \shortstack is plain LaTeX, so the stacked variant needs no extra package.
JUDGES = [('gpt_oss', 'GPT-oss', r'GPT\\oss'),
          ('qwen36', 'Qwen3.6', r'Qwen\\3.6'),
          ('dsv4', 'DS-V4', r'DS\\V4')]

# Columns that get a best-in-class mark, and the precision they print at. Ranking
# is on the ROUNDED value, so two rows that print the same number are both marked
# rather than one of them winning on a digit the reader cannot see.
# Deliberately absent: the per-judge columns, which are properties of the judge
# rather than of the model, and Deb., which prints only on the rows that carried
# a bias -- the highest value shown there is not the best model, it is the best
# of the corrected ones, and bolding it would say the wrong thing.
MARKED = [('essay', 2), ('legal_ref', 2),
          ('rec_gpbam', 2), ('rec_mostcited', 2), ('aa_index', 0)]

# Weight availability, per row, for the best-open-weights mark. Explicit rather
# than pattern-matched on the vendor prefix, and asserted to cover every row, so
# a model added later fails loudly instead of being silently called open.
#
# The self-hosted rows are settled by the fact that we ran them on our own GPUs
# (see OR_PRICE in rebuild_tables.py): whatever the licence says, the weights
# were obtainable. That resolves Mistral-Medium-3.5-128B and
# Ministral-3-14B-Reasoning, which the vendor otherwise positions as API tiers.
#
# mistralai/mistral-large-2512 is filed OPEN on Lorenz's call. It carries both
# open-weight marks as a result: it is second on GP and on Cited, so the italic
# there lands on it rather than on DeepSeek-V4-Flash.
WEIGHTS = {
    'anthropic/claude-opus-5': 'closed',
    'anthropic/claude-haiku-4.5': 'closed',
    'google/gemini-3.7-flash': 'closed',
    'google/gemini-2.5-flash-lite': 'closed',
    'openai/gpt-5-mini': 'closed',
    'openai/gpt-5-nano': 'closed',
    'openai/gpt-4o-mini-2024-07-18': 'closed',
    'mistralai/mistral-large-2512': 'open',
    'deepseek-ai/DeepSeek-V4-Flash': 'open',
    'deepseek-ai/DeepSeek-V4-Flash-0731': 'open',
    'deepseek/deepseek-v3.2': 'open',
    'deepseek/deepseek-r1-0528': 'open',
    'deepseek/deepseek-chat-v3-0324': 'open',
    'qwen/qwen3.5-397b-a17b': 'open',
    'qwen/qwen3.5-122b-a10b': 'open',
    'qwen/qwen3-235b-a22b-thinking-2507': 'open',
    'qwen/qwen3-32b': 'open',
    'qwen3-next-80b-a3b-instruct': 'open',
    'Qwen/Qwen3.6-35B-A3B-FP8': 'open',
    'openai/gpt-oss-120b': 'open',
    'RedHatAI/gemma-4-31B-it-FP8-block': 'open',
    'meta-llama/llama-4-maverick': 'open',
    'meta-llama/llama-3.3-70b-instruct': 'open',
    'meta-llama/llama-3.1-8b-instruct': 'open',
    'mistralai/Mistral-Medium-3.5-128B': 'open',       # self-hosted
    'mistralai/Ministral-3-14B-Reasoning-2512': 'open',  # self-hosted
    'RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8': 'open',
    'GaleneAI/Magistral-Small-2509-FP8-Dynamic': 'open',
    'mistralai/mistral-small-3.1-24b-instruct': 'open',
    'utter-project/EuroLLM-22B-Instruct-2512': 'open',
    'ibm-granite/granite-4.1-3b': 'open',
    'Microsoft/Phi-4-mini-instruct': 'open',
    'windprak/open_steuerllm': 'open',
    'soofi-s-isar-preview': 'open',                    # self-hosted on InnKube
}

# tier -> the letter it prints as, most severe first.
TIER_SYM = [('self', r'\mathrm{M}'), ('family*', r'\mathrm{F}'), ('vendor', r'\mathrm{V}')]

# Names the rules below cannot reach: nothing about "-preview" says "drop me", and
# the vendor's own capitalisation is not recoverable from the endpoint id.
SHORT_OVERRIDE = {'soofi-s-isar-preview': 'Soofi-S-Isar'}

# Size words cost more width than any other part of a name and carry one bit
# each. Mistral spends two of them on its longest two rows.
SIZE_WORD = {'small': 'S', 'medium': 'M', 'large': 'L'}

# Capitalising the first letter is right for every row except the ones whose
# leading token is an acronym or a camel-cased brand, where it produces "Gpt" and
# "Deepseek". Fixed by exact match on the leading token, not by a general rule --
# there is no rule, only a list of names vendors write a particular way.
CASE_FIX = {'Gpt': 'GPT', 'Deepseek': 'DeepSeek'}

# ColorBrewer RdBu, 11 classes, low -> high. Red is a decrease, blue an increase,
# and the middle class is near-white so an unmoved cell reads as unshaded.
RDBU = ['67001F', 'B2182B', 'D6604D', 'F4A582', 'FDDBC7', 'F7F7F7',
        'D1E5F0', '92C5DE', '4393C3', '2166AC', '053061']


def short(model_id):
    """Compact display name. See the module docstring for the rules."""
    n = model_id.split('/')[-1]
    if n in SHORT_OVERRIDE:
        return SHORT_OVERRIDE[n]
    n = re.sub(r'-FP8(-block|-Dynamic)?$', '', n, flags=re.I)
    n = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', n)          # -2024-07-18
    n = re.sub(r'-\d{4}$', '', n)                      # -0731, -2506, -2512
    n = re.sub(r'-instruct$', '-I', n, flags=re.I)
    n = re.sub(r'-it$', '-I', n)
    n = re.sub(r'-reasoning$', '-R', n, flags=re.I)
    n = re.sub(r'-thinking$', '-T', n, flags=re.I)
    n = re.sub(r'(?<=-)(small|medium|large)(?=-|$)',
               lambda m: SIZE_WORD[m.group(1).lower()], n, flags=re.I)
    # The active-parameter count of an MoE, "-a17b" after the "-397b" total.
    # Case-sensitive on purpose: Qwen3.6-35B-A3B writes it in caps and keeps it,
    # which is how Lorenz tuned these by hand -- it is also the one name here
    # that doubles as a judge seat label, where the A3B is worth carrying.
    n = re.sub(r'-a\d+b(?=-|$)', '', n)
    n = n[:1].upper() + n[1:]
    return re.sub(r'^[A-Za-z]+', lambda m: CASE_FIX.get(m.group(0), m.group(0)), n)


def esc(s):
    return s.replace('_', r'\_')


def rms_anchor(values, sd=SD_ANCHOR):
    """`sd` standard deviations about ZERO, for a quantity whose null is zero.

    `np.std` would centre on the sample mean, which for a deviation column is an
    estimate of a quantity we already know to be zero and would shrink the scale
    by whatever asymmetry the sample happens to carry.
    """
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    return float(sd * np.sqrt((v ** 2).mean())) if len(v) else 0.0


def _luminance(rgb):
    c = [x / 255 for x in rgb]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def diverge(value, vmax, body, on=True):
    """`\\cellcolor` for `value` on a symmetric -vmax..+vmax RdBu ramp.

    Values past +-vmax clamp to the end classes rather than extending the scale:
    the anchor is a fixed number of SDs, so saturation means "off the scale",
    which is information rather than an artefact.
    """
    if not on or value is None or not np.isfinite(value) or vmax <= 0:
        return body
    pos = (np.clip(value / vmax, -1.0, 1.0) + 1) / 2 * (len(RDBU) - 1)
    lo, frac = int(pos), pos - int(pos)
    hi = min(lo + 1, len(RDBU) - 1)
    rgb = [round(int(RDBU[lo][i:i + 2], 16) * (1 - frac)
                 + int(RDBU[hi][i:i + 2], 16) * frac) for i in (0, 2, 4)]
    fill = r'\cellcolor[HTML]{' + ''.join(f'{c:02X}' for c in rgb) + '}'
    if (_luminance(rgb) + 0.05) / 0.05 < 4.5:
        body = r'\textcolor{white}{' + body + '}'
    return fill + body


def num(v, d=2):
    return '--' if pd.isna(v) else f'{v:.{d}f}'


def column_marks(col, dec, is_open, runner_up):
    """Which rows of one column get which mark.

    Ranking is on the rounded value: two rows printing the same number are both
    marked, because bolding one of two identical printed figures is arbitrary and
    the reader has no way to see why.
    """
    v = col.round(dec).dropna()
    marks = {m: set() for m in v.index}
    if v.empty:
        return marks
    levels = sorted(v.unique(), reverse=True)
    for m, x in v.items():
        if x == levels[0]:
            marks[m].add('best')
        elif runner_up in ('second', 'both') and len(levels) > 1 and x == levels[1]:
            marks[m].add('second')
    if runner_up in ('open', 'both'):
        ov = v[is_open.reindex(v.index).fillna(False).to_numpy()]
        if not ov.empty:
            top = ov.max()
            for m, x in ov.items():
                # If the best row overall is already open there is nothing for
                # this mark to say, and stacking it on the bold only adds ink.
                if x == top and 'best' not in marks[m]:
                    marks[m].add('open')
    return marks


def apply_marks(body, marks):
    """bold = best, underline = runner-up, italic = best open weights.

    Three separate font channels rather than three colours: the shaded columns
    already spend colour on something else, and these have to survive greyscale
    print. `second` and `open` can co-occur on one cell -- the runner-up overall
    may also be the best open-weight row -- and nest cleanly.
    """
    if 'best' in marks:
        body = r'\textbf{' + body + '}'
    if 'open' in marks:
        body = r'\textit{' + body + '}'
    if 'second' in marks:
        body = r'\underline{' + body + '}'
    return body


def load(results):
    """The two parent artefacts, joined on the endpoint id."""
    tab = pd.read_csv(results / 'main_essay_table.csv', index_col='model')
    bias = pd.read_csv(results / 'judge_family_bias_models.csv')
    bias = bias[bias.cond == 'no_rag'].set_index('gen')

    tab['relation'] = bias['relation'].reindex(tab.index).fillna('')
    # Carry the *shift*, not the level. The two artefacts agree on the reported
    # score to well under a display digit but not to the last bit, so lifting
    # `score_debiased` across as an absolute value can make an unrelated row --
    # where the correction is exactly the identity -- print Deb. one hundredth
    # away from its own Med. Adding the shift instead keeps that difference at a
    # hard zero, which is what the column claims.
    shift = (bias['score_debiased'] - bias['score']).reindex(tab.index).fillna(0.0)
    tab['debiased'] = tab['essay'] + shift
    tab['delta'] = shift

    scored = bias['score'].reindex(tab.index).dropna()
    drift = (scored - tab['essay'].reindex(scored.index)).abs().max()
    assert drift < 0.05, (
        f'main_essay_table.csv and judge_family_bias_models.csv disagree about the '
        f'reported score by up to {drift:.3f} points -- one of them is stale, '
        f'rebuild both before merging')

    # Anchor the per-judge shading on the row's own middle judge, not on Med.
    # Med. is a mean of per-case medians and these are means of scores, so no
    # judge sits at zero against Med. and all three cells end up shaded.
    cols = [k for k, _, _ in JUDGES]
    tab['judge_mid'] = tab[cols].median(axis=1)
    for k in cols:
        tab[f'dev_{k}'] = tab[k] - tab['judge_mid']

    unknown = [m for m in tab.index if m not in WEIGHTS]
    assert not unknown, (
        f'no weight-availability entry for {unknown} -- add them to WEIGHTS rather '
        f'than letting a new row default to open')
    tab['open_weights'] = [WEIGHTS[m] == 'open' for m in tab.index]

    names = tab.index.map(short)
    dup = names[names.duplicated()].tolist()
    assert not dup, f'compacted names collide: {dup} -- extend SHORT_OVERRIDE'
    tab['short'] = names
    return tab


def relation_letters(rel):
    """M/F/V for the model's tiers, most severe first. Blank if unrelated."""
    ts = set(rel.split('+')) if rel else set()
    return '/'.join(sym for t, sym in TIER_SYM if t in ts)


def build(tab, colour=True, sd=SD_ANCHOR, runner_up='both', gutter=(1.0, 0.5),
          judge_header='flat'):
    # Structural zeros are excluded here and included below -- see the docstring.
    dev_cells = [tab.loc[m, f'dev_{k}'] for m in tab.index for k, _, _ in JUDGES
                 if abs(tab.loc[m, f'dev_{k}']) > 1e-9]
    vmax_judge = rms_anchor(dev_cells, sd)
    vmax_delta = rms_anchor(tab.loc[tab['relation'] != '', 'delta'], sd)
    marks = {c: column_marks(tab[c], dec, tab['open_weights'], runner_up)
             for c, dec in MARKED}

    lines = [
        r'% Generated by experiments/analysis/deepseek_rerun/scripts/merged_results_table.py',
        r'% -- do not hand-edit. Rebuild after rebuild_tables.py.',
    ]
    if colour:
        lines.append(r'% Needs \usepackage[table]{xcolor} (or colortbl) for \cellcolor.')
    lines += [
        '%',
        '% Merges the former main_essay_table.tex and judge_score_matrix.tex. Both',
        '% parents are still emitted; this is the joined view, not a replacement.',
        '%',
        '% no-RAG throughout. The panel is the three judges re-run in one window on',
        '% 2026-08-19/20: gpt-oss-120b, Qwen3.6-35B-A3B, DeepSeek-V4-Flash-0731.',
        '% Med. is the per-case median over the three, averaged over the 81 cases',
        (f'% ({OPUS_N} for the ddagger row), bootstrap SE in the subscript. The per-judge'
         if OPUS_N < 81 else
         '% bootstrap SE in the subscript. The per-judge'),
        '% columns are each judge\'s own mean, so they need not bracket Med.',
        '%',
        '% Deb. subtracts the fitted tier effect from every judgement a related judge',
        '% made, then re-medians. Tier effects, in consensus-percentile points, from',
        '% judge_family_bias_rerun.py: self +9.94, family* +4.51, vendor +1.21. The',
        '% superscript is the closest relation the model has to any judge on the',
        '% panel: M same model, F same family, V same vendor.',
        '%',
        '% Deb. is BLANK wherever no judge was related to the model. On those rows the',
        '% correction is exactly the identity, so the cell would repeat Median digit',
        '% for digit -- a column that is mostly a copy of its neighbour costs width',
        '% and reads as though a correction had happened. A printed Deb. therefore',
        '% means "this row carried a bias"; the fourteen that print are the fourteen',
        '% related rows, including the ones whose median did not move.',
        '%',
        '% There is no separate Delta column. The size and sign of the correction are',
        '% recoverable exactly by reading Deb. against Median on the same row, which',
        '% is two columns apart, and the cell fill encodes the same difference -- red',
        '% for a score that fell under correction, blue for one that rose, saturation',
        '% for how far.',
        '%',
        '% The family* tier was identified from the retired DeepSeek-V4-Flash 0423',
        '% checkpoint judged by 0731. That row is no longer reported (see',
        '% rebuild_tables.py RETIRED), so no F appears in this table even though the',
        '% coefficient it contributes is still in the correction.',
        '%',
        '% Best in class is bold; second best is underlined; the best open-weights row',
        '% is italic where it is not already the best overall. Not marked: the',
        '% per-judge columns, which are properties of the judge rather than of the',
        '% model, and Deb., where the highest value shown is the best of the corrected',
        '% rows rather than the best model. Ranking is on the ROUNDED value, so two',
        '% rows printing the same number are both marked. claude-opus-5 takes the bold',
        '% in several columns: the ddagger qualifies the row, not the number.',
        '%',
        '% GP and Cited are the two recitation prompt sets: the laws the GPBam cases',
        '% cite, and the most-cited German laws generally. Ref. is legal reference',
        '% similarity, recomputed for every row by recompute_legal_ref.py. AA is the',
        '% Artificial Analysis index -- the one column this study does not produce,',
        '% which is why it sits in a group of its own.',
        '%',
        '% DeepSeek-V4-Flash prints without its date suffix and is the 0731',
        '% re-generation. The 0423 checkpoint NHR@FAU retired on 2026-08-01 is not in',
        '% this table, so the name is unambiguous within it.',
        '%',
        '% ddagger claude-opus-5 is measured, but not on equal terms with the rest of',
        '% the table, and the row should not be read as one more benchmarked model.',
        *([
            '% Two differences, both of which favour it:',
            f'%   1. Coverage, essay columns only. {OPUS_N} of the 81 cases -- a contiguous',
            f'%      0-{OPUS_N - 1}, not a sample. Med. is the mean over those, so the SE is over',
            f'%      {OPUS_N}, not 81. GP and Cited are unaffected: all 200 recitation prompts',
            '%      were answered, so those two cells are full coverage.',
            '%   2. Harness. Written through the Claude Code agent loop on the benchmark',
            '%      prompt, not through the API path every other row used. The prompt is',
            '%      byte-identical; the surrounding system prompt and tool loop are not.',
            '%      See experiments/analysis/additional_models/opus5_agent/README.md.',
            '% Its recitation columns are measured on all 200 prompts through the same agent',
            '% harness, scored by the same ROUGE-L call as every other row. All 200 transcripts',
            '% were audited: one tool call each, the Write of the answer, no read of the',
            '% reference corpora in this repo. Caveat 2 still applies to those cells.',
        ] if OPUS_N < 81 else [
            '% Coverage is no longer one of the reasons: all 81 essay cases are answered,',
            '% so Med. averages the same 81 cases as every other row. One difference',
            '% remains, and it favours the row:',
            '%   Harness. Written through the Claude Code agent loop on the benchmark',
            '%   prompt, not through the API path every other row used. The prompt is',
            '%   byte-identical; the surrounding system prompt and tool loop are not.',
            '%   See experiments/analysis/additional_models/opus5_agent/README.md.',
            '% Its recitation columns are measured on all 200 prompts through the same agent',
            '% harness, scored by the same ROUGE-L call as every other row. All 200 transcripts',
            '% were audited: one tool call each, the Write of the answer, no read of the',
            '% reference corpora in this repo. The harness caveat applies to those cells too.',
        ]),
        '%',
    ]
    if colour:
        lines += [
            '% Colour: RdBu, white at zero, red below and blue above. Only the columns',
            '% whose value IS a difference are shaded. Two independent symmetric',
            f'% scales, each anchored at {sd:g} SD about zero rather than at the observed',
            '% maximum, so one outlier row cannot define the ramp for the rest and the',
            '% scale does not recalibrate on every rebuild. Cells past the anchor',
            '% saturate, which reads as "off the scale", not "the largest one here".',
            f'%   Deb.            +-{vmax_delta:.2f} points, on Deb. - Median',
            f'%   per-judge       +-{vmax_judge:.2f} points, on judge - median of the',
            '%                   row\'s three judges. Exactly one cell per row is white',
            '%                   by construction: the middle judge, which IS that',
            '%                   median. The anchor is the row median rather than Med.',
            '%                   because Med. averages per-case medians while these',
            '%                   average scores, so no judge sits at zero against it.',
            '%',
        ]

    lines += [
        '% Emitted as a bare tabular, like main_essay_table.tex, so it drops into',
        '% the draft the same way. It is ten columns wide -- wrap it in',
        '% \\resizebox{\\textwidth}{!}{...} inside the table environment.',
        '%',
        (r'\begin{tabular}{l @{\hspace{%(g)gem}} r r @{\hspace{%(s)gem}} r r r'
         r' @{\hspace{%(s)gem}} r @{\hspace{%(g)gem}} r r @{\hspace{%(g)gem}} r}'
         % {'g': gutter[0], 's': gutter[1]}),
        r'\toprule',
        (r' & \multicolumn{6}{c}{Essay writing} & \multicolumn{2}{c}{Art. recit.}'
         r' & \multicolumn{1}{c}{Gen.} \\'),
        r'\cmidrule(lr){2-7} \cmidrule(lr){8-9} \cmidrule(lr){10-10}',
        r' & \multicolumn{2}{c}{Judge panel} & \multicolumn{3}{c}{Per judge} & & & & \\',
        r'\cmidrule(lr){2-3} \cmidrule(lr){4-6}',
        ' & '.join(['Model', 'Median', 'Deb.']
                   + [lab if judge_header == 'flat'
                      else r'\shortstack{' + two + '}'
                      for _, lab, two in JUDGES]
                   + ['Ref.', 'GP', 'Cited', 'AA']) + r' \\',
        r'\midrule',
    ]

    for m, r in tab.iterrows():
        name = esc(r['short'])
        if m == OPUS:
            name += r'$^{\ddagger}$'
        def mk(col, dec=2):
            return apply_marks(num(r[col], dec), marks[col].get(m, set()))

        # The SE subscript stays outside the bold: it is a property of the
        # estimate, not part of the figure being ranked, and bolding it too makes
        # the winning row's subscript louder than other rows' main numbers.
        med = mk('essay') + f'$_{{\\pm{r.essay_sem:.2f}}}$'
        # Deb. prints only where something was corrected. On every other row it
        # repeats Median exactly, and a column that is mostly a copy of its
        # neighbour costs width and reads as though a correction happened.
        if r['relation']:
            deb = diverge(r['delta'], vmax_delta,
                          f'{r.debiased:.2f}$^{{{relation_letters(r["relation"])}}}$',
                          colour)
        else:
            deb = ''
        cells = [name, med, deb]
        cells += [diverge(r[f'dev_{k}'], vmax_judge, num(r[k]), colour) for k, _, _ in JUDGES]
        cells += [mk('legal_ref'), mk('rec_gpbam'), mk('rec_mostcited'),
                  mk('aa_index', 0)]
        lines.append(' & '.join(cells) + r' \\')

    lines += [r'\bottomrule', r'\end{tabular}']
    return lines, vmax_delta, vmax_judge


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out_dir', nargs='?', default=str(RESULTS),
                    help='where to write main_results_table.{tex,csv} (default: results/)')
    ap.add_argument('--results', default=str(RESULTS),
                    help='where to read the two parent artefacts from')
    # Width, measured on this table at \scriptsize: the default 1.5/0.9em gutters
    # cost 23pt against 0.8/0.5em, which is the cheapest width available -- it
    # spends only the visual separation between groups, and the rules above the
    # header already carry that. For comparison, tabcolsep 4pt -> 2pt is 26pt and
    # dropping the SE subscript is 13pt, both of which are the document's call
    # rather than this script's.
    ap.add_argument('--gutter', type=float, nargs=2, default=(1.0, 0.5),
                    metavar=('GROUP', 'SUB'),
                    help='column gutters in em: between groups, and inside the '
                         'essay group (default 1.0 0.5; 0 0 saves a further 18pt '
                         'but flattens the group structure in the body)')
    ap.add_argument('--judge-header', choices=['flat', 'stacked'], default='flat',
                    help='per-judge column labels on one line, or broken over two '
                         '(saves 27pt: those columns are bound by their headers)')
    ap.add_argument('--runner-up', choices=['second', 'open', 'both', 'none'],
                    default='both',
                    help='what to mark besides the best: second best (underline), '
                         'best open weights (italic), both, or neither')
    ap.add_argument('--sd', type=float, default=SD_ANCHOR,
                    help=f'colour ramp ends at this many SD about zero (default {SD_ANCHOR:g})')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--colour', dest='colour', action='store_true', default=True,
                   help='shade the difference columns (default)')
    g.add_argument('--no-colour', dest='colour', action='store_false',
                   help='emit a table with no \\cellcolor at all')
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tab = load(Path(a.results))
    lines, vmax_delta, vmax_judge = build(tab, a.colour, a.sd, a.runner_up,
                                          tuple(a.gutter), a.judge_header)

    (out / 'main_results_table.tex').write_text('\n'.join(lines) + '\n')
    cols = (['short', 'relation', 'essay', 'essay_sem', 'debiased', 'delta']
            + [k for k, _, _ in JUDGES] + ['judge_mid']
            + [f'dev_{k}' for k, _, _ in JUDGES]
            + ['legal_ref', 'rec_gpbam', 'rec_mostcited', 'aa_index', 'open_weights'])
    tab[cols].to_csv(out / 'main_results_table.csv')

    dev = np.concatenate([(tab[k] - tab['judge_mid']).to_numpy() for k, _, _ in JUDGES])
    dev = dev[np.abs(dev) > 1e-9]
    shown = tab.loc[tab['relation'] != '', 'delta']
    print(f'{len(tab)} rows, {len(shown)} with a judge relation, '
          f'colour {"on" if a.colour else "off"}')
    print(f'  delta      n={len(shown):3d}  max |x| {shown.abs().max():5.2f}  '
          f'anchor {a.sd:g} SD = {vmax_delta:5.2f}  '
          f'saturated {int((shown.abs() > vmax_delta).sum())}')
    print(f'  per-judge  n={len(dev):3d}  max |x| {np.abs(dev).max():5.2f}  '
          f'anchor {a.sd:g} SD = {vmax_judge:5.2f}  '
          f'saturated {int((np.abs(dev) > vmax_judge).sum())}')
    if a.runner_up != 'none':
        print(f'  marks: bold best'
              + (', underline second' if a.runner_up in ('second', 'both') else '')
              + (', italic best open weights' if a.runner_up in ('open', 'both') else ''))
        for c, dec in MARKED:
            mk = column_marks(tab[c], dec, tab['open_weights'], a.runner_up)
            lab = {k: sorted(tab.loc[[m for m, v in mk.items() if k in v], 'short'])
                   for k in ('best', 'second', 'open')}
            print(f'    {c:<16s} best {"/".join(lab["best"]) or "--":<24s}'
                  f' second {"/".join(lab["second"]) or "--":<24s}'
                  f' open {"/".join(lab["open"]) or "--"}')
    print(f'  gutters: {a.gutter[0]:g}em between groups, {a.gutter[1]:g}em inside; '
          f'judge headers {a.judge_header}')
    print(f'wrote main_results_table.{{tex,csv}} to {out}')


if __name__ == '__main__':
    main()

"""What the oracle's advantage is actually made of.

The oracle hands the generator the statutes the reference solution cites. That
is an upper bound on retrieval, and it is the only condition in this paper that
moves the essay score at all -- so what it is made of decides how to read it.
Two channels are confounded in it, and only one of them is retrieval:

  * the **wording**: the text of the provisions, which a working retriever
    could in principle also supply;
  * the **selection**: *which* provisions this case turns on. That list is the
    reference solution's issue list, i.e. the skeleton of the argument the
    essay is being asked to produce. No retriever has it, and if the oracle's
    gain came from here it would be leakage rather than a ceiling.

A third, smaller channel rides along with the selection: ``gold_sets`` orders
the norms by how heavily the solution cites each one, so the oracle also hands
over a prominence ranking derived from the solution.

Each condition holds the others and moves one:

    oracle              wording + selection + order      the ceiling itself
    + uncovered named   selection widened to the norms the corpus lacks
    order shuffled      selection and wording held, prominence destroyed
    + junk norms        wording and selection held, purity destroyed
    wording withheld    selection and order held, no statute text at all
    top-10 only         selection cut to the ten most-cited norms

``wording withheld`` is the leakage test: it names exactly the norms the text
arm supplies -- the same list, lifted verbatim out of that arm's own prompt --
and supplies none of their text. Whatever the selection is worth on its own is
the whole of what that row scores.

Three generators over the same 81 cases.

The oracle column is not recomputed here: it is read off the ladder's banked
cell in ``analysis/out/ladder_values.csv``, so the two tables cannot print
different gains for the same model -- by construction rather than by both
happening to read the same file. They used to do the latter, and it broke:
the 09-12 re-generation reused
``oracle_deepseek-ai_DeepSeek-V4-Flash_gold_combined.csv``, the very file
``ladder_table`` reads, so the arm behind the ladder's 47.90 is no longer on
disk. Reading the banked cell also means this column needs no judging.

**DeepSeek-V4-Flash: which arms.** DeepSeek's reasoning effort is a prompt
prefix, and NHR@FAU served the August ladder's ``high`` requests without it --
generation and the DeepSeek judge seat both ran at DeepSeek's ``low`` -- while
DeepInfra, and FAU after its restart, honour ``high``. The printed row is
``oracle_rag/DS4-ablation-low`` (2026-09-13, DeepInfra, generation and DeepSeek
judge at ``low``). Its no-RAG arm reproduces August (+0.19 +- 4.06, gpt-oss
seat) and its oracle arm the ladder's cell (47.90, +4.14 with the DeepSeek judge
at ``high``; +5.49 at ``low``), so its oracle cell is read off the ladder like
the other rows (``ORACLE_FROM_LADDER``).

The two ``high``-effort generations stay in the file as commented-out rows
(``COMMENTED``, ``HIGH_ROWS``), judged with the DeepSeek seat at ``low`` and
differenced against the ``high``-effort no-RAG arm, each with the oracle from its
own arm: the 2026-09-12 NHR@FAU
arms in ``D`` and the 2026-09-13 DeepInfra rerun (``DS4-ablation-rerun``, no
top-10). They sit 6-11 points over no retrieval where the ladder's regime sits
at 0-5; generation ``low`` minus ``high`` within DeepInfra is -6.49 +- 1.96
averaged over five arms. The August federal-corpus DeepSeek conditions
(08-17/18) are in neither. Full account: ``DS4-ablation-rerun.md``.

This still costs something. ``refs.canonicalise()`` has drifted since the
stored arms ran, so the oracle column's selection is 47.4-47.7 norms per case
where the five conditions' is 47.9 -- half a norm, under a percent of the list,
and the conditions are differenced against no retrieval rather than against the
oracle, so it does not enter any printed cell.


Run from ``experiments/``::

    PYTHONPATH=analysis python analysis/oracle_ablation_table.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paper_table as pt
from ladder_table import shade

D = f'{pt.B}/oracle_rag/ji2'
OUT = 'analysis/out/oracle_ablation.tex'
SCALE = pt.SCALES['free3']

#: Colour half-width, in standard deviations of the deltas shown. Six, as in
#: ladder_table, so a cell means the same thing in both tables.
SIGMA = 6.0

#: Frozen at the value the published table was coloured on, rather than
#: recomputed from whatever cells happen to be present. ``6 * std`` over the
#: twelve deltas of the two-generator table is 13.0038, and that reproduces all
#: twelve published colours exactly (0/12 mismatches, any vmax in
#: [12.9695, 13.0255] does). Recomputing it would recolour published cells every
#: time a row or condition lands -- adding the DeepSeek row alone moves it -- so
#: new cells go onto the existing scale, as in ladder_table. Set to ``None`` to
#: recompute, which is only right if the whole table is being rebuilt at once.
VMAX = 13.0038

#: Cells whose arm files the 2026-09-11 ``git clean -fd`` deleted, banked off
#: the published table so a regeneration cannot hollow it. Live arm files win;
#: this is the fallback. Same role ``ladder_values.csv`` plays for the ladder.
BANKED = 'analysis/out/oracle_ablation_values.csv'

#: (key, file suffix, two-line column head)
#:
#: ``_gold_combined`` for the oracle, the same file ``ladder_table`` reads, so
#: the two tables cannot print different gains for the same model. See the
#: module docstring for what that trades away against ``_gold_now``.
#:
#: Every column is differenced against *no retrieval*, not against the arm it
#: is mechanically comparable with. That is the presentation the paper wants:
#: one baseline for the whole table, so a reader can compare any two cells
#: directly and see at a glance that every manipulation still sits at or above
#: the no-retrieval level. It does cost something -- ``+ junk norms`` is
#: padding *on top of* the shuffled arm, so against no retrieval it carries the
#: shuffle as well as the padding, and the isolated effect of each channel is
#: the difference between two adjacent columns rather than a column itself.

#: The condition whose cell comes from the ladder, not from an arm here.
ORACLE = '_gold_combined'

CONDS = [(ORACLE, ('', 'oracle')),
         ('_gold_unc', ('+ uncov.', 'named')),
         ('_gold_shuf', ('order', 'shuffled')),
         ('_gold_pad', ('+ junk', 'norms')),
         ('_gold_cites', ('wording', 'withheld')),
         ('_gold_top10', ('top-10', 'only'))]

#: The 2026-09-13 OpenRouter/DeepInfra regeneration of DeepSeek's arms.
RERUN_D = f'{pt.B}/oracle_rag/DS4-ablation-rerun'
#: The same, at DeepSeek's `low` effort for generation *and* the DeepSeek judge
#: seat -- the regime NHR@FAU served the August ladder at (DS4-ablation-rerun.md).
LOW_D = f'{pt.B}/oracle_rag/DS4-ablation-low'
#: Arm dirs whose oracle cell is still read off the ladder: their generation and
#: judging regime is the ladder's, so the two tables keep one oracle gain.
ORACLE_FROM_LADDER = {LOW_D}

#: (row head, file slug, the identifier the no-retrieval run stores, arm dir)
#:
#: ``arm dir`` None means ``D`` and the ladder's oracle cell; a row with its
#: own directory computes its oracle cell from its own ``_gold_combined``.
#:
#: Qwen3-Next and Qwen3.6 have an oracle arm and none of the five conditions,
#: so they would print one number and five dashes.
GEN = [('Gemma-4-31B', 'RedHatAI_gemma-4-31B-it-FP8-block',
        'RedHatAI/gemma-4-31B-it-FP8-block', None),
       ('gpt-oss-120B', 'openai_gpt-oss-120b', 'openai/gpt-oss-120b', None)]

#: DeepSeek first, as in the ladder. The printed row is the ``low`` regeneration
#: with the ladder's oracle cell; the ``high``-effort rows follow as comments.
#: See the module docstring.
DS_HIGH_FAU = r'DeepSeek-V4-Flash {\tiny high}'
DS_HIGH_DI = r'DeepSeek-V4-Flash {\tiny high, DeepInfra}'
DEEPSEEK_ROWS = [('DeepSeek-V4-Flash', 'deepseek-ai_DeepSeek-V4-Flash',
                  'deepseek-ai/DeepSeek-V4-Flash', LOW_D),
                 (DS_HIGH_FAU, 'deepseek-ai_DeepSeek-V4-Flash',
                  'deepseek-ai/DeepSeek-V4-Flash', D),
                 (DS_HIGH_DI, 'deepseek-ai_DeepSeek-V4-Flash',
                  'deepseek-ai/DeepSeek-V4-Flash', RERUN_D)]
ROWS = DEEPSEEK_ROWS + GEN

#: Rows written into the file as LaTeX comments: kept, not printed.
COMMENTED = {DS_HIGH_FAU, DS_HIGH_DI}

#: The high-effort rows are judged the way the ladder is -- DeepSeek seat at
#: ``low`` (GWDG, validated against DeepInfra ``low``), read from
#: ``judge_effort_low_deepseek<cond>.csv`` beside each arm -- and differenced
#: against the high-effort no-RAG arm, judged the same way. A cell whose side
#: file is missing prints ``--``, never the high-effort judge.
HIGH_ROWS = {DS_HIGH_FAU, DS_HIGH_DI}
HIGH_NORAG = f'{pt.B}/oracle_rag/ji2/norag_regime_deepseek-ai_DeepSeek-V4-Flash.csv'
DS_SEAT_COLS = (pt.DS, pt.DS_ALT, pt.DS_OR)


def with_low_judge(path, side):
    """The arm with every DeepSeek seat column replaced by the low-effort side file."""
    if not (os.path.exists(path) and os.path.exists(side)):
        return None
    a = pd.read_csv(path, low_memory=False).sort_values('index').reset_index(drop=True)
    s = pd.read_csv(side, low_memory=False).sort_values('index').reset_index(drop=True)
    if len(s) != len(a) or not (s['index'].values == a['index'].values).all():
        return None
    a = a.drop(columns=[c for c in DS_SEAT_COLS if c in a.columns])
    a[pt.DS_OR] = s[pt.DS_OR].values
    return a

#: LaTeX comment written above a row, for rows the draft's caption does not
#: explain (the caption is kept verbatim: the paper is over its page limit).
ROW_NOTE = {DS_HIGH_FAU: '% DeepSeek-V4-Flash, generation at high effort (NHR@FAU 2026-09-12 arms), DeepSeek '
                         'judge at low; against the high-effort no-RAG arm; oracle from its own arm '
                         '(+ junk n=55, wording withheld n=60)',
            DS_HIGH_DI: '% DeepSeek-V4-Flash, generation at high effort (OpenRouter/DeepInfra 2026-09-13, '
                        'DS4-ablation-rerun), DeepSeek judge at low; against the high-effort no-RAG arm; '
                        'oracle from its own arm, top-10 not run'}

#: The ladder's banked cells, and this table's row head -> the ladder's.
LADDER_VALUES = 'analysis/out/ladder_values.csv'
LADDER_BLOCK = 'Essay score (0--100)'
LADDER_ROW = {'DeepSeek-V4-Flash': 'DeepSeek-V4-Flash',
              'Gemma-4-31B': 'Gemma-4-31B-it',
              'gpt-oss-120B': 'gpt-oss-120B'}


def ladder_oracle():
    """``{row head: (gain over no retrieval, bold)}`` from the ladder table.

    The ladder prints levels, so the gain is its oracle cell minus its
    no-retrieval cell, and its bold flag already means the same thing this
    table's does -- a paired difference beyond twice its standard error -- so
    it carries over rather than being recomputed from per-case vectors that,
    for DeepSeek, no longer exist. Costs a hundredth of a point to rounding:
    the ladder banks two decimals, so gpt-oss reads +5.87 where the per-case
    computation gave +5.86.
    """
    v = pd.read_csv(LADDER_VALUES)
    v = v[(v.table == 1) & (v.block == LADDER_BLOCK)]
    out = {}
    for head, row in LADDER_ROW.items():
        cell = v[v.row == row].set_index('col')
        if not {'oracle', 'no_rag'} <= set(cell.index):
            continue
        o, b = cell.at['oracle', 'value'], cell.at['no_rag', 'value']
        if pd.isna(o) or pd.isna(b):
            continue
        out[head] = (float(o) - float(b), bool(cell.at['oracle', 'bold']))
    return out


def banked():
    """``{(row, cond): (gain, bold)}`` from the published table."""
    if not os.path.exists(BANKED):
        return {}
    b = pd.read_csv(BANKED)
    return {(r.row, r.cond): (float(r.value), bool(r.bold))
            for r in b.itertuples()}


def collect(gen=ROWS):
    """No-retrieval level per generator, and every condition's gain over it.

    Cells are ``(gain, bold)``. The five conditions get theirs from their own
    per-case vectors; ``ORACLE`` gets it from the ladder, see
    :func:`ladder_oracle`.
    """
    norag = pd.read_csv(f'{pt.B}/without_rag/ji2/no_rag_ji2_result.csv')
    oracle, bank = ladder_oracle(), banked()
    base, delta = {}, {}
    for name, slug, mid, arm_dir in gen:
        high = name in HIGH_ROWS
        b = (pt.med(pd.read_csv(HIGH_NORAG, low_memory=False).sort_values('index'), SCALE)
             if high else pt.baseline(mid, SCALE, norag))
        if b is not None:
            base[name] = float(np.mean(b))
        cells = {}
        for suffix, _ in CONDS:
            if suffix == ORACLE and (arm_dir is None or arm_dir in ORACLE_FROM_LADDER):
                cells[suffix] = oracle.get(name)
                continue
            path = f'{arm_dir or D}/oracle_{slug}{suffix}.csv'
            if high:
                frame = with_low_judge(path, f'{arm_dir}/judge_effort_low_deepseek{suffix}.csv')
                arm = pt.med(frame, SCALE) if frame is not None else None
                if arm is None or b is None:
                    cells[suffix] = None
                    continue
            else:
                arm = pt.med(pd.read_csv(path).sort_values('index'), SCALE) \
                    if os.path.exists(path) else None
            if arm is None or b is None:
                cells[suffix] = bank.get((name, suffix))
                continue
            k = min(len(arm), len(b))
            d = pd.Series(arm[:k] - b[:k]).dropna()
            s = d.std(ddof=1) / np.sqrt(len(d))
            cells[suffix] = (d.mean(), bool(abs(d.mean()) > 2 * s))
        delta[name] = cells
    return dict(base=base, delta=delta)


def tex(v, gen=ROWS):
    """Conditions across, generators down, every cell against no retrieval.

    Transposed from the generators-in-columns form it had: with six conditions
    and three generators the upright table spent nine rows on eighteen numbers,
    and the row labels were long enough to need their own column width. This
    way the condition names carry the two-line treatment the ladder's heads
    already use and the whole thing is four rows deep.
    """
    if VMAX:
        vmax = VMAX
    else:
        d = np.array([c[0] for cells in v['delta'].values()
                      for c in cells.values() if c is not None], dtype=float)
        sd = np.std(d, ddof=1) if len(d) > 1 else np.nan
        vmax = SIGMA * sd if sd and not np.isnan(sd) and sd > 0 else 1.0

    heads = [('no', 'retrieval')] + [h for _, h in CONDS]
    out = [r'\begin{table}[t]', r'  \centering', r'  \scriptsize',
           r'  \setlength{\tabcolsep}{3.5pt}',
           r'  \renewcommand{\arraystretch}{.95}',
           r'  \begin{tabular}{l' + 'r' * len(heads) + '}', r'    \toprule',
           '     & ' + ' & '.join(t for t, _ in heads) + r' \\',
           '     & ' + ' & '.join(b for _, b in heads) + r' \\',
           r'    \midrule']
    for name, _, _, _ in gen:
        if name in ROW_NOTE:
            out.append(f'    {ROW_NOTE[name]}')
        cells = [f'{v["base"][name]:.2f}' if name in v['base'] else '--']
        for suffix, _ in CONDS:
            c = v['delta'].get(name, {}).get(suffix)
            if c is None:
                cells.append('--')
                continue
            m, bold = c
            txt = f'{m:+.2f}'
            if bold:
                txt = f'\\mathbf{{{txt}}}'
            rgb, dark = shade(m, vmax)
            txt = f'${txt}$'
            cells.append(f'\\cellcolor[HTML]{{{rgb}}}'
                         + (f'\\textcolor{{white}}{{{txt}}}' if dark else txt))
        row = f'{name} & ' + ' & '.join(cells) + r' \\'
        out.append(f'    % {row}' if name in COMMENTED else f'    {row}')
    out += [r'    \bottomrule', r'  \end{tabular}',
            '  \\caption{' + CAPTION + '}',
            r'  \label{tab:oracle-ablation}', r'\end{table}']
    return '\n'.join(out)


#: Close to the draft's wording -- the table is generated into a paper that
#: already reads around it -- but the two sentences that trailed off are
#: finished, and the claims are the ones the printed cells actually support.
#: The draft's wording verbatim, not a fuller one generated alongside it. The
#: paper is over its page limit and this caption was trimmed there to 400
#: characters; regenerating a 1100-character version put the two out of sync
#: and silently handed the trim back. What the conditions mean and what the
#: oracle column is read off is documented in the module docstring instead,
#: which is where it belongs -- it is provenance, not something a reader of the
#: table needs.
CAPTION = (
    r'Oracle advantage decomposition, relative to \norag{}.' '\n'
    r'  Bold exceeds twice the standard error.' '\n'
    r'  \emph{Wording withheld} supplies norms without their text; padding '
    r'with junk norms and keeping 10 most-cited references test dilution; '
    r'\emph{+ uncov. named} names the cited norms not in the corpus.' '\n'
    r'    \emph{Order shuffled} drops the ranking by citation frequency, not '
    r'an argument order.' '\n  ')


def main():
    gen, out_path = ROWS, OUT
    v = collect(gen)
    with open(out_path, 'w') as fh:
        fh.write('% generated by analysis/oracle_ablation_table.py -- '
                 'do not edit by hand.\n'
                 '% Requires \\usepackage{booktabs}, \\usepackage{amsmath} '
                 'and \\usepackage[table]{xcolor}.\n\n' + tex(v, gen) + '\n')
    print(f'-> {out_path}\n')
    print(pd.DataFrame({g: {k: ('--' if c is None else
                                f'{c[0]:+.2f}{"*" if c[1] else " "}')
                            for k, c in cells.items()}
                        for g, cells in v['delta'].items()}).fillna('--').to_string())
    print('\nno-retrieval level: '
          + ', '.join(f'{g} {s:.2f}' for g, s in v['base'].items()))


if __name__ == '__main__':
    main()

"""Rewrite the commented DeepSeek {\\tiny high} rows of analysis/out/ladder_table.tex.

The ladder is hand-maintained (it cannot be regenerated), so only its commented
high-effort rows are rewritten; every printed cell stays byte-identical. Cells are
judged the way the ladder is -- DeepSeek seat at `low`, from the
``judge_effort_low_deepseek*.csv`` side files -- and differenced against the
high-effort no-RAG arm, judged the same way. Shades sit on the blocks' existing
scales (8.46 essay, 9.4709 citations); bold is a paired difference beyond 2 SEM.
A cell whose arm or side file is missing prints ``--``.

    python analysis/ds4_low/ladder_high_rows.py
"""
import os, sys
import numpy as np, pandas as pd
R = os.environ.get('GPBAM_ROOT') or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root: set GPBAM_ROOT to override
os.chdir(f'{R}/experiments'); sys.path[:0] = [f'{R}/experiments/analysis', R]
import ladder_table as lt, paper_table as pt
from judge_recall_slope import gold_by_case

E = pt.B
TEX = 'analysis/out/ladder_table.tex'
HIGH_NORAG = f'{E}/oracle_rag/ji2/norag_regime_deepseek-ai_DeepSeek-V4-Flash.csv'
VMAX = {'essay': 8.46, 'cited': 9.4709}
DS_COLS = (pt.DS, pt.DS_ALT, pt.DS_OR)
ORC = 'oracle_deepseek-ai_DeepSeek-V4-Flash_gold_combined.csv'
ARM = 'rag_deepseek-ai_DeepSeek-V4-Flash.csv'
LOW = 'judge_effort_low_deepseek.csv'
#: ladder column -> arm directory of the GWDG high-effort generation
GWDG_COLS = {'rrf': 'rag_titled_combined_rrf', 'rerank': 'rag_titled_combined_rerank',
             'cite_only': 'rag_titled_combined_cite_only', 'cite_rrf': 'rag_titled_combined_cite_rrf',
             'rrf50': 'rag_titled_combined_rrf_k50', 'cite_rrf50': 'rag_titled_combined_cite_rrf_k50',
             'ls_text': 'rag_titled_combined_ls_text', 'cite_ls_rrf50': 'rag_titled_combined_cite_ls_rrf_k50'}
GWDG_NORAG = (f'{E}/without_rag_high_gwdg/norag_deepseek-ai_DeepSeek-V4-Flash.csv', f'{E}/without_rag_high_gwdg/{LOW}')


def gwdg(base):
    d = f'{E}/{base}/DS4-high-gwdg'
    return f'{d}/{ARM}', f'{d}/{LOW}'


#: (row head, no-RAG (arm, low-judge side or None), {ladder column: (arm csv, low-judge side file)})
ROWS = [
    (r'DeepSeek-V4-Flash {\tiny high}', GWDG_NORAG,
     {**{c: gwdg(b) for c, b in GWDG_COLS.items()},
      'oracle': (f'{E}/oracle_rag/ji2/{ORC}', f'{E}/oracle_rag/ji2/judge_effort_low_deepseek_gold_combined.csv')}),
    (r'DeepSeek-V4-Flash {\tiny high, DeepInfra}', (HIGH_NORAG, None),
     {'ls_text': (f'{E}/rag_titled_combined_ls_text/ji2/{ARM}', f'{E}/rag_titled_combined_ls_text/ji2/{LOW}'),
      'oracle': (f'{E}/oracle_rag/DS4-ablation-rerun/{ORC}',
                 f'{E}/oracle_rag/DS4-ablation-rerun/judge_effort_low_deepseek_gold_combined.csv')}),
]
COLS = ['rrf', 'rerank', 'cite_only', 'cite_rrf', 'rrf50', 'cite_rrf50', 'ls_text', 'cite_ls_rrf50', 'oracle']
NOTE = ("    % DeepSeek-V4-Flash {\\tiny high}: generation at DeepSeek's real high effort, DeepSeek judge at low; "
        "each row against its own high-effort no-RAG arm (first cell), on this block's scale. "
        "{\\tiny high} = GWDG generation (DeepSeek's high prefix) for no-RAG and every RAG cell, oracle from "
        "the NHR@FAU 2026-09-12 arm; {\\tiny high, DeepInfra} = OpenRouter/DeepInfra 2026-09-13.")
OLD_PREFIXES = ('    % DeepSeek-V4-Flash (high, ', '    % DeepSeek-V4-Flash at high effort', '    % DeepSeek-V4-Flash {\\tiny high')


def low_judged(arm, side):
    if not (os.path.exists(arm) and os.path.exists(side)):
        return None
    a = pd.read_csv(arm, low_memory=False).sort_values('index').reset_index(drop=True)
    s = pd.read_csv(side, low_memory=False).sort_values('index').reset_index(drop=True)
    if len(a) != len(s) or not (a['index'].values == s['index'].values).all() or pt.DS_OR not in s:
        return None
    a = a.drop(columns=[c for c in DS_COLS if c in a.columns])
    a[pt.DS_OR] = s[pt.DS_OR].values
    return a


def cell(v, b, vmax):
    k = min(len(v), len(b)); d = pd.Series(v[:k] - b[:k]).dropna()
    level, delta, s2 = np.nanmean(v), d.mean(), 2 * d.std(ddof=1) / np.sqrt(len(d))
    rgb, dark = lt.shade(delta, vmax)
    txt = f'{level:.2f}'
    if abs(delta) > s2:
        txt = f'\\textbf{{{txt}}}'
    if dark:
        txt = f'\\textcolor{{white}}{{{txt}}}'
    return f'\\cellcolor[HTML]{{{rgb}}}{txt}', (level, delta, s2)


def main(write=True, interim=False):
    """``interim``: while the GWDG no-RAG rerun is not judged, difference the GWDG row
    against the DeepInfra high no-RAG arm and say so in the table comment."""
    gold = gold_by_case()
    stats, lines = {}, {'essay': [], 'cited': []}
    interim_used = False
    for head, (norag, norag_side), arms in ROWS:
        if norag_side is None:
            base_df = pd.read_csv(norag, low_memory=False).sort_values('index') if os.path.exists(norag) else None
        else:
            base_df = low_judged(norag, norag_side)
            if (base_df is None or pt.med(base_df, lt.SCALE) is None) and interim:
                base_df = pd.read_csv(HIGH_NORAG, low_memory=False).sort_values('index')
                interim_used = True
        base = {'essay': None if base_df is None else pt.med(base_df, lt.SCALE),
                'cited': None if base_df is None else lt.cite_recall(base_df[['index', 'answer']], gold)}
        for block in ('essay', 'cited'):
            b = None if base[block] is None else np.asarray(base[block], float)
            cells = ['--' if b is None else f'{np.nanmean(b):.2f}']
            stats[(head, block, 'no_rag')] = None if b is None else (float(np.nanmean(b)), 0.0, 0.0)
            for col in COLS:
                if col not in arms or b is None:
                    cells.append('--'); continue
                arm, side = arms[col]
                if block == 'essay':
                    f = low_judged(arm, side)
                    v = pt.med(f, lt.SCALE) if f is not None else None
                else:
                    v = lt.cite_recall(arm, gold) if os.path.exists(arm) and os.path.exists(side) else None
                if v is None:
                    cells.append('--'); continue
                c, st = cell(np.asarray(v, float), b, VMAX[block])
                cells.append(c)
                stats[(head, block, col)] = tuple(float(x) for x in st)
                print(f'{block:<6} {head:<45} {col:<14} level {st[0]:.2f} delta {st[1]:+.2f} 2SEM {st[2]:.2f}')
            lines[block].append(f'    % {head} & ' + ' & '.join(cells) + r' \\')
    if write:
        text = open(TEX).read()
        cut = text.index('\\label{tab:ladder}')
        t = [l for l in text[:cut].split('\n') if not l.startswith(OLD_PREFIXES)]
        for block, lead in (('essay', '    DeepSeek-V4-Flash & 43.77 &'), ('cited', '    DeepSeek-V4-Flash & 25.86 &')):
            k = next(n for n, l in enumerate(t) if l.startswith(lead))
            note = NOTE + (' INTERIM: the {\\tiny high} row is against the DeepInfra high no-RAG arm (47.84) '
                           'until the GWDG no-RAG rerun is judged.' if interim_used else '')
            t[k + 1:k + 1] = [note] + lines[block]
        open(TEX, 'w').write('\n'.join(t) + text[cut:])
    return stats


if __name__ == '__main__':
    main(interim='--interim' in sys.argv)

"""Cells for the DeepSeek-at-low regeneration: free3 gain over the August no-RAG arm, with the DeepSeek judge at high and at low."""
import os, sys, json
import numpy as np, pandas as pd
os.chdir(os.path.join(os.environ.get('GPBAM_ROOT') or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'experiments')); sys.path.insert(0, 'analysis')
import paper_table as pt
S = pt.SCALES['free3']; base = np.asarray(pt.baseline('deepseek-ai/DeepSeek-V4-Flash', S, None), float)
SEAT = 'Judge (deepseek-v4-flash-0731)'; HI = 'Judge (deepseek-v4-flash-0731, effort high)'
O = 'zubaers_result/essay_writing/oracle_rag/DS4-ablation-low'
L = 'zubaers_result/essay_writing/rag_titled_combined_ls_text/DS4-ablation-low'
ARMS = [('oracle', f'{O}/oracle_deepseek-ai_DeepSeek-V4-Flash_gold_combined.csv', f'{O}/judge_effort_low_deepseek_gold_combined.csv'),
        ('top-10', f'{O}/oracle_deepseek-ai_DeepSeek-V4-Flash_gold_top10.csv', f'{O}/judge_effort_low_deepseek_gold_top10.csv'),
        ('+ uncovered', f'{O}/oracle_deepseek-ai_DeepSeek-V4-Flash_gold_unc.csv', f'{O}/judge_effort_low_deepseek_gold_unc.csv'),
        ('shuffled', f'{O}/oracle_deepseek-ai_DeepSeek-V4-Flash_gold_shuf.csv', f'{O}/judge_effort_low_deepseek_gold_shuf.csv'),
        ('+ junk', f'{O}/oracle_deepseek-ai_DeepSeek-V4-Flash_gold_pad.csv', f'{O}/judge_effort_low_deepseek_gold_pad.csv'),
        ('wording withheld', f'{O}/oracle_deepseek-ai_DeepSeek-V4-Flash_gold_cites.csv', f'{O}/judge_effort_low_deepseek_gold_cites.csv'),
        ('ls_text', f'{L}/rag_deepseek-ai_DeepSeek-V4-Flash.csv', f'{L}/judge_effort_low_deepseek.csv')]
def cell(frame):
    if 'score_Judge (gpt-oss-120b)' not in frame or not frame['score_Judge (gpt-oss-120b)'].notna().any():
        return None
    v = pt.med(frame, S)
    if v is None: return None
    v = np.asarray(v, float); d = pd.Series(v - base).dropna()
    return dict(level=float(np.nanmean(v)), gain=float(d.mean()), sem2=float(2 * d.std(ddof=1) / np.sqrt(len(d))), n=len(d))
out = {}
for name, arm_p, side_p in ARMS:
    a = pd.read_csv(arm_p, low_memory=False).sort_values('index').reset_index(drop=True)
    lo = pd.read_csv(side_p, low_memory=False).sort_values('index').reset_index(drop=True)
    c = f'score_{SEAT}'
    hi_frame = a.assign(**{c: a[f'score_{HI}'].values}) if f'score_{HI}' in a else (a if c in a else None)
    lo_frame = a.assign(**{c: lo[c].values})
    out[name] = dict(tokens=float(a.completion_tokens.median()),
                     high=cell(hi_frame) if hi_frame is not None else None, low=cell(lo_frame),
                     judge_low_minus_high=None)
    if hi_frame is not None and c in hi_frame:
        dd = 100 * (lo[c].values - hi_frame[c].values); dd = dd[~np.isnan(dd)]
        out[name]['judge_low_minus_high'] = (float(dd.mean()), float(2 * dd.std(ddof=1) / np.sqrt(len(dd))))
fmt = lambda x: 'pending' if x is None else f"{x['gain']:+.2f} ± {x['sem2']:.2f}"
print(f'{"arm":<18}{"tok":>6}{"DS judge high":>17}{"DS judge low":>17}{"judge low-high":>17}')
for k, r in out.items():
    j = r['judge_low_minus_high']
    print(f'{k:<18}{r["tokens"]/1000:>5.1f}k{fmt(r["high"]):>17}{fmt(r["low"]):>17}{(f"{j[0]:+.2f} ± {j[1]:.2f}" if j else "-"):>17}'
          + (f'   level {r["low"]["level"]:.2f}' if r['low'] else ''))
json.dump(out, open('analysis/out/ds4_low_cells.json', 'w'), indent=1)

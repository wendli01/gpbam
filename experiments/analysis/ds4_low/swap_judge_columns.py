"""Make the low-effort DeepSeek judge seat canonical in DS4-ablation-low arm files; keep the high-effort seat as '..., effort high' columns."""
import os, sys, fcntl, tempfile
import pandas as pd
os.chdir(os.path.join(os.environ.get('GPBAM_ROOT') or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'experiments'))
SEAT = 'Judge (deepseek-v4-flash-0731)'
for arm_p, side_p in zip(sys.argv[1::2], sys.argv[2::2]):
    a = pd.read_csv(arm_p, low_memory=False).sort_values('index').reset_index(drop=True)
    lo = pd.read_csv(side_p, low_memory=False).sort_values('index').reset_index(drop=True)
    assert (a['index'].values == lo['index'].values).all()
    cols = [c for c in a.columns if c.endswith(SEAT)]
    if any(c.endswith('effort high)') for c in a.columns):
        print(f'{os.path.basename(arm_p)}: already swapped'); continue
    for c in cols:
        a[c.replace(SEAT, 'Judge (deepseek-v4-flash-0731, effort high)')] = a[c]
        if c in lo:
            a[c] = lo[c].values
    a['deepseek_judge_effort'] = 'low'
    a.to_csv(arm_p + '.tmp', index=False); os.replace(arm_p + '.tmp', arm_p)
    s = f'score_{SEAT}'
    print(f'{os.path.basename(arm_p)}: DeepSeek seat now low ({100*a[s].mean():.2f}, n={int(a[s].notna().sum())}); high kept ({100*a[s.replace(SEAT, "Judge (deepseek-v4-flash-0731, effort high)")].mean():.2f})')

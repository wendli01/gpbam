"""No-RAG DeepSeek at DeepSeek's real high on GWDG, judged like the ladder.

1. ``run_model.py`` essays on GWDG (effort low + DeepSeek's high prefix), written
   to ``essay_writing/without_rag_high_gwdg/``;
2. converted to an arm CSV with ``index``/``answer`` like the other arms;
3. gpt-oss-120b and Qwen3.6-35B-A3B-FP8 (NHR@FAU) into the arm file, one seat at a
   time; the DeepSeek judge at low comes from ``judge_low_arms.py --endpoint deepinfra``.

    python analysis/ds4_low/norag_high_gwdg.py generate   # steps 1-2
    python analysis/ds4_low/norag_high_gwdg.py judge      # step 3, FAU seats
"""
import json, os, subprocess, sys, time
import numpy as np, pandas as pd
R = os.environ.get('GPBAM_ROOT') or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root: set GPBAM_ROOT to override
os.chdir(f'{R}/experiments'); sys.path[:0] = [f'{R}/experiments/analysis', R]
D = 'zubaers_result/essay_writing/without_rag_high_gwdg'
SLUG = 'ds4_gwdg_high'
JSONL, ARM = f'{D}/{SLUG}_essays.jsonl', f'{D}/norag_deepseek-ai_DeepSeek-V4-Flash.csv'


def generate():
    env = dict(os.environ, RUN_ENDPOINT='gwdg', RUN_MODEL='deepseek-v4-flash-0731', RUN_EFFORT='low',
               RUN_PREFIX='high', RUN_TASKS='essay', RUN_SLUG=SLUG, RUN_WORKERS='20', RUN_TIMEOUT='3600',
               RUN_ESSAY_DIR=os.path.abspath(D))
    os.makedirs(D, exist_ok=True)
    subprocess.run([sys.executable, '-u', f'{R}/experiments/analysis/deepseek_rerun/scripts/run_model.py'],
                   cwd=R, env=env, check=True)
    d = pd.DataFrame([json.loads(l) for l in open(JSONL)]).drop_duplicates('index', keep='last')
    d = d[d.error.isna()].sort_values('index')
    d['model'] = 'deepseek-ai/DeepSeek-V4-Flash'
    d.to_csv(ARM, index=False)
    print(f'{len(d)} essays -> {ARM}; tokens median {d.completion_tokens.median():.0f}', flush=True)


def judge():
    import oracle_experiment as oe
    oe.api_keys()
    facts, sols = oe.cases()
    from src import scoring, prompts, evaluate
    instr = prompts.build_judge_user(prompts.JUDGE_INSTRUCTIONS_BY_NAME['ji2'])
    for seat in ('openai/gpt-oss-120b', 'Qwen/Qwen3.6-35B-A3B-FP8'):
        name = seat.split('/')[-1]
        d = pd.read_csv(ARM, low_memory=False).sort_values('index').reset_index(drop=True)
        col = f'score_Judge ({name})'
        if col in d and d[col].notna().any():
            print(f'{name}: present'); continue
        live = [i for i in range(len(d)) if isinstance(d.answer[i], str) and d.answer[i].strip()]
        idx = [int(d['index'][i]) for i in live]
        t0 = time.time()
        out = evaluate.rejudge_model([d.answer[i] for i in live], [facts[i] for i in idx], [sols[i] for i in idx],
                                     judge=scoring.JudgeEnsemble([scoring.Judge(model=seat, prompt=instr, max_concurrency=12)], verbose=True),
                                     model_name='deepseek-ai/DeepSeek-V4-Flash', verbose=True)
        d = pd.read_csv(ARM, low_memory=False).sort_values('index').reset_index(drop=True)
        for c in [c for c in out.columns if c.endswith(f'Judge ({name})')]:
            d[c] = np.nan
            d[c] = d[c].astype(object)
            for k, i in enumerate(live):
                d.at[i, c] = out[c].iloc[k]
        d.to_csv(ARM + '.tmp', index=False); os.replace(ARM + '.tmp', ARM)
        print(f'{name}: n={int(pd.to_numeric(d[col], errors="coerce").notna().sum())} mean {100*pd.to_numeric(d[col], errors="coerce").mean():.2f} ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    {'generate': generate, 'judge': judge}[sys.argv[1]]()

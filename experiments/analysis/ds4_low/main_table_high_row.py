"""A commented DeepSeek-V4-Flash {\\tiny high} row for the main results table.

Inserted below the DeepSeek row of ``deepseek_rerun/results/main_results_table.tex``
and ``..._stacked.tex``, as a LaTeX comment, without regenerating the table (so no
published row moves). Every cell is computed the way ``merged_results_table.py``
computes the August row:

* Median and SE: per-case median over gpt-oss-120b, Qwen3.6-35B-A3B (NHR@FAU) and
  the DeepSeek judge at ``low``, mean over the 81 cases, SE = sd / sqrt(n).
* Deb.: ``judge_family_bias.debias`` fitted on the published panel; its ``self``
  effect is applied to this row's DeepSeek judgements through that judge's own
  empirical quantile function, differentially -- no refit, so the published
  coefficients and every other row stay as they are.
* Ref.: the extractor calls of ``recompute_legal_ref.py``.
* GP / Cited: ROUGE-L of the recitation run at DeepSeek's real high (GWDG,
  official prefix), if complete; ``--`` otherwise.
* Colours: the table's own scales (``rms_anchor`` over the published cells).

``--check`` reproduces the August row's Deb., Ref. and cell strings and writes nothing.

    python analysis/ds4_low/main_table_high_row.py [--check]
"""
import ast, contextlib, io, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import percentileofscore

ROOT = Path(__file__).resolve().parents[3]
SCR = ROOT / 'experiments/analysis/deepseek_rerun/scripts'
sys.path[:0] = [str(SCR), str(ROOT / 'experiments/analysis'), str(ROOT)]
os.chdir(ROOT)
import judge_family_bias_rerun as jfr                     # noqa: E402  patches the panel loader
import judge_family_bias as fb                            # noqa: E402
import merged_results_table as mrt                        # noqa: E402

E = ROOT / 'experiments/zubaers_result/essay_writing'
ARM = E / 'oracle_rag/ji2/norag_regime_deepseek-ai_DeepSeek-V4-Flash.csv'
DS_SIDE = [E / 'oracle_rag/ji2/norag_regime_deepseek-ai_DeepSeek-V4-Flash_judge_deepseek_low_deepinfra.csv',
           E / 'oracle_rag/ji2/norag_regime_deepseek-ai_DeepSeek-V4-Flash_judge_deepseek_low_gwdg.csv']
REC = ROOT / 'experiments/zubaers_result/article_recitation/DS4-effort/recitation_ds4_gwdg_high.csv'
TABLES = [ROOT / 'experiments/analysis/deepseek_rerun/results/main_results_table.tex',
          ROOT / 'experiments/analysis/deepseek_rerun/results/main_results_table_stacked.tex']
DS0731 = 'deepseek-ai/DeepSeek-V4-Flash-0731'
LABEL = r'DeepSeek-V4-Flash {\tiny high}'
GENERATED_ON = 'OpenRouter/DeepInfra'
GO, QW, DS = 'score_Judge (gpt-oss-120b)', 'score_Judge (Qwen3.6-35B-A3B-FP8)', 'score_Judge (deepseek-v4-flash-0731)'


def legal_ref_functions():
    """``_refs``, ``_jaccard`` and the solution sets from recompute_legal_ref.py, without its I/O."""
    src = (SCR / 'recompute_legal_ref.py').read_text()
    head = src[:src.index('pub = pd.read_csv(PUB')]
    ns = {'__file__': str(SCR / 'recompute_legal_ref.py'), '__name__': 'recompute_legal_ref_defs'}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(head, 'recompute_legal_ref.py', 'exec'), ns)
    return ns


def ref_mean(ns, answers, index):
    with contextlib.redirect_stdout(io.StringIO()):
        v = [np.nan if i in ns['undefined'] else ns['_jaccard'](ns['_refs'](a), ns['sol_refs'][i])
             for a, i in zip(answers, index)]
    return 100 * np.nanmean(v)


def debias_fn():
    long = fb.add_normalised(fb.load_long())
    out, gamma = fb.debias(long)
    emp = {j: np.sort(sub['score'].to_numpy()) for j, sub in out.groupby('judge')}

    def adj(judge, scores, tier):
        s = np.asarray(scores, float)
        q = np.array([percentileofscore(emp[judge], x, kind='mean') for x in s])
        qa = np.clip(q - gamma[tier], 0.01, 100)
        Q = lambda x: np.quantile(emp[judge], np.asarray(x) / 100, method='linear')  # noqa: E731
        return s + Q(qa) - Q(q)
    return long, gamma, adj


def judge_label(key):
    return next(j for j in fb.JUDGES if key in j.lower())


def row_cells(tab, essay, sem, deb, judge_means, ref, gp, cited, aa):
    dev_cells = [tab.loc[m, f'dev_{k}'] for m in tab.index for k, _, _ in mrt.JUDGES
                 if abs(tab.loc[m, f'dev_{k}']) > 1e-9]
    vmax_judge = mrt.rms_anchor(dev_cells, mrt.SD_ANCHOR)
    vmax_delta = mrt.rms_anchor(tab.loc[tab['relation'] != '', 'delta'], mrt.SD_ANCHOR)
    mid = np.median(list(judge_means.values()))
    med = mrt.num(essay) + f'$_{{\\pm{sem:.2f}}}$'
    debc = mrt.diverge(deb - essay, vmax_delta, f'{deb:.2f}$^{{{mrt.relation_letters("self")}}}$')
    judges = [mrt.diverge(judge_means[k] - mid, vmax_judge, mrt.num(judge_means[k])) for k, _, _ in mrt.JUDGES]
    fmt = lambda v, d=2: '--' if v is None or not np.isfinite(v) else mrt.num(v, d)  # noqa: E731
    return [med, debc, *judges, fmt(ref), fmt(gp), fmt(cited), fmt(aa, 0)]


def main(check):
    tab = mrt.load(mrt.RESULTS)
    ref_ns = legal_ref_functions()
    long, gamma, adj = debias_fn()
    lab = {'gpt_oss': judge_label('gpt-oss'), 'qwen36': judge_label('qwen'), 'dsv4': judge_label('deepseek')}

    # -- check: the August row, rebuilt from the panel and the stored essays --
    p = long[long.gen == DS0731].pivot_table(index='case', columns='judge', values='score')
    ds_adj = adj(lab['dsv4'], p[lab['dsv4']].values, 'self')
    deb_aug = np.median(np.column_stack([p[lab['gpt_oss']], p[lab['qwen36']], ds_adj]), axis=1).mean()
    bias = pd.read_csv(mrt.RESULTS / 'judge_family_bias_models.csv').set_index('gen')
    aug_essays = pd.read_json(ROOT / 'experiments/zubaers_result/essay_writing/without_rag_0731/ds_v4_flash_0731_high.jsonl', lines=True)
    ref_aug = ref_mean(ref_ns, aug_essays.answer, aug_essays['index'])
    r = tab.loc[DS0731]
    cells = row_cells(tab, r.essay, r.essay_sem, r.debiased, {k: r[k] for k, _, _ in mrt.JUDGES}, r.legal_ref, r.rec_gpbam, r.rec_mostcited, r.aa_index)
    line = next(l for l in TABLES[0].read_text().splitlines() if l.startswith('DeepSeek-V4-Flash & '))
    print(f'check Deb.: panel reproduction {deb_aug:.3f} vs published score_debiased {bias.loc[DS0731, "score_debiased"]:.3f} (gamma self {gamma["self"]:+.2f} pct points)')
    print(f'check Ref.: recomputed {ref_aug:.2f} vs table {r.legal_ref:.2f}')
    print(f'check cells: Deb. and per-judge strings found in the published row: {all(c in line for c in cells[1:5])}')
    assert abs(deb_aug - bias.loc[DS0731, 'score_debiased']) < 0.05 and abs(ref_aug - r.legal_ref) < 0.05 and all(c in line for c in cells[1:5])
    if check:
        return

    # -- the high row --
    arm = pd.read_csv(ARM, low_memory=False).sort_values('index').reset_index(drop=True)
    side = next(s for s in DS_SIDE if Path(s).exists())
    sd = pd.read_csv(side, low_memory=False).sort_values('index').reset_index(drop=True)
    assert (sd['index'].values == arm['index'].values).all()
    go, qw, ds = (100 * arm[GO].to_numpy(float), 100 * arm[QW].to_numpy(float), 100 * sd[DS].to_numpy(float))
    m = np.column_stack([go, qw, ds]); ok = ~np.isnan(m).any(axis=1)
    med = np.median(m[ok], axis=1)
    essay, sem = med.mean(), pd.Series(med).std() / np.sqrt(ok.sum())
    deb = np.median(np.column_stack([go[ok], qw[ok], adj(lab['dsv4'], ds[ok], 'self')]), axis=1).mean()
    ref = ref_mean(ref_ns, arm.answer, arm['index'])
    gp = cited = None
    if REC.exists():
        rec = pd.read_csv(REC)
        if len(rec) == 200:
            by = rec.groupby('dataset').score.mean() * 100
            gp, cited = by.get('GPBam Laws'), by.get('Most cited Laws')
    judge_means = {'gpt_oss': np.nanmean(go), 'qwen36': np.nanmean(qw), 'dsv4': np.nanmean(ds)}
    cells = row_cells(tab, essay, sem, deb, judge_means, ref, gp, cited, tab.loc[DS0731, 'aa_index'])
    note = (f"% {{\\tiny high}}: generation at DeepSeek's real high effort ({GENERATED_ON}), judges "
            f"gpt-oss-120b and Qwen3.6-35B-A3B (NHR@FAU) and DeepSeek at low ({'GWDG' if 'gwdg' in Path(side).name else 'DeepInfra'}); "
            "Deb. applies the fitted self effect without refitting; GP/Cited: recitation at DeepSeek's high prefix on GWDG"
            + ('' if gp is not None else ' (pending)') + '. See DS4-ablation-rerun.md.')
    row = f'% {LABEL} & ' + ' & '.join(cells) + r' \\'
    for t in TABLES:
        lines = [l for l in t.read_text().splitlines() if not (l.startswith('% {\\tiny high}:') or l.startswith(f'% {LABEL} & '))]
        i = next(k for k, l in enumerate(lines) if l.startswith('DeepSeek-V4-Flash & '))
        lines[i + 1:i + 1] = [note, row]
        t.write_text('\n'.join(lines) + '\n')
    print(f'Median {essay:.2f} ± {sem:.2f} | Deb. {deb:.2f} | gpt-oss {judge_means["gpt_oss"]:.2f} Qwen {judge_means["qwen36"]:.2f} '
          f'DeepSeek(low) {judge_means["dsv4"]:.2f} | Ref. {ref:.2f} | GP {gp} Cited {cited} | DS judge from {side.name}')
    print(row)


if __name__ == '__main__':
    argv = sys.argv[1:]
    # relative paths are relative to experiments/, like every other script here (this one chdirs to the repo root)
    here = lambda x: Path(x) if Path(x).is_absolute() else ROOT / 'experiments' / x   # noqa: E731
    if '--arm' in argv:          # e.g. the GWDG no-RAG rerun instead of the DeepInfra arm
        ARM = here(argv[argv.index('--arm') + 1])
    if '--ds-side' in argv:
        DS_SIDE = [here(argv[argv.index('--ds-side') + 1])]
    if '--generated-on' in argv:
        GENERATED_ON = argv[argv.index('--generated-on') + 1]
    main('--check' in argv)

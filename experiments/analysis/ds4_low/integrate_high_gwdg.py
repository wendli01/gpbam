"""Put the DeepSeek {\\tiny high} GWDG arms into the tables, then commit and push.

Run by ``supervise_high.py`` once every arm is generated and judged. Idempotent:
every step rewrites only commented {\\tiny high} lines, so a re-run after a failed
push repeats the same edits and retries the push.

1. ladder essay + cited blocks: ``ladder_high_rows.py`` ({\\tiny high} = GWDG
   no-RAG and RAG cells, oracle from NHR@FAU 09-12; {\\tiny high, DeepInfra} kept);
2. tab:corpus essay block: a commented {\\tiny high} row, federal vs combined
   round-robin, shaded on that block's existing scale (recovered from its printed
   cells and asserted to reproduce every one of them);
3. main table: the {\\tiny high} row recomputed on the GWDG no-RAG arm
   (recitation unchanged);
4. ``analysis/out/ds4_high_gwdg_cells.csv`` and a section in DS4-ablation-rerun.md;
5. git add, commit, pull --rebase, push.

    python analysis/ds4_low/integrate_high_gwdg.py [--dry]
"""
import os, re, subprocess, sys, time
import numpy as np, pandas as pd
R = os.environ.get('GPBAM_ROOT') or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root: set GPBAM_ROOT to override
os.chdir(f'{R}/experiments'); sys.path[:0] = [f'{R}/experiments/analysis/ds4_low', f'{R}/experiments/analysis', R]
import ladder_high_rows as lhr, ladder_table as lt, paper_table as pt   # noqa: E402

DRY = '--dry' in sys.argv
E = pt.B
TEX = 'analysis/out/ladder_table.tex'
CELLS = 'analysis/out/ds4_high_gwdg_cells.csv'
HEAD = r'DeepSeek-V4-Flash {\tiny high}'


def printed_corpus_cells(text):
    """(federal level, combined level, fill colour) for every printed row of the tab:corpus essay block."""
    block = text[text.index('\\emph{Essay score (0--100)}, against federal only'):]
    block = block[:block.index('\\bottomrule')]
    out = []
    for line in block.split('\n')[1:]:
        if line.strip().startswith('%') or '&' not in line:
            continue
        cells = [x.strip() for x in line.rstrip(' \\').split('&')]
        rgb = re.search(r'\\cellcolor\[HTML\]\{([0-9A-F]{6})\}', cells[2]).group(1)
        out.append((float(re.findall(r'-?\d+\.\d+', cells[1])[-1]), float(re.findall(r'-?\d+\.\d+', cells[2])[-1]), rgb))
    return out


def corpus_vmax(text):
    """The block's colour half-width: every vmax that reproduces all printed fills, and their median."""
    printed = printed_corpus_cells(text)
    grid = np.round(np.arange(0.20, 30.0, 0.001), 3)
    ok = [v for v in grid if all(lt.shade(cl - fl, v)[0] == rgb for fl, cl, rgb in printed)]
    assert ok, f'no single vmax reproduces the printed tab:corpus essay colours {printed}'
    return float(np.median(ok)), ok


def corpus_row():
    fed = lhr.low_judged(*lhr.gwdg('rag_titled_federal'))
    comb = lhr.low_judged(*lhr.gwdg('rag_titled_combined'))
    if fed is None or comb is None:
        return None, None
    f, c = (np.asarray(pt.med(x, lt.SCALE), float) for x in (fed, comb))
    d = pd.Series(c - f).dropna()
    delta, s2 = d.mean(), 2 * d.std(ddof=1) / np.sqrt(len(d))
    vmax, ok = corpus_vmax(open(TEX).read())
    rgb, dark = lt.shade(delta, vmax)
    txt = f'{np.nanmean(c):.2f}'
    if abs(delta) > s2:
        txt = f'\\textbf{{{txt}}}'
    if dark:
        txt = f'\\textcolor{{white}}{{{txt}}}'
    row = f'    % {HEAD} & {np.nanmean(f):.2f} & \\cellcolor[HTML]{{{rgb}}}{txt} \\\\'
    return row, dict(federal=float(np.nanmean(f)), combined=float(np.nanmean(c)), delta=float(delta), sem2=float(s2),
                     vmax=vmax, vmax_range=(min(ok), max(ok)))


def write_corpus_row(row):
    text = open(TEX).read()
    cut = text.index('\\label{tab:ladder}')
    head, tail = text[:cut], text[cut:]
    t = [l for l in tail.split('\n') if not l.startswith(f'    % {HEAD} &')]
    k = next(n for n, l in enumerate(t) if l.startswith('    DeepSeek-V4-Flash & 43.33 &'))
    t[k + 1:k + 1] = [row]
    open(TEX, 'w').write(head + '\n'.join(t))


def sh(cmd):
    env = dict(os.environ, PATH='/usr/local/bin:/usr/bin:/bin:' + os.environ.get('PATH', ''))
    r = subprocess.run(cmd, cwd=R, capture_output=True, text=True, shell=True, env=env)
    print(f'$ {cmd}\n{r.stdout[-2000:]}{r.stderr[-2000:]}', flush=True)
    return r.returncode


def main():
    print(f'### {time.strftime("%F %T")} integrate (dry={DRY})', flush=True)
    stats = lhr.main(write=not DRY)
    row, corpus = corpus_row()
    print('tab:corpus row:', row, corpus)
    if DRY:
        return 0
    if row is None:
        print('tab:corpus row skipped: the federal or combined round-robin arm is not generated and judged', flush=True)
    else:
        write_corpus_row(row)
    r = subprocess.run([sys.executable, 'analysis/ds4_low/main_table_high_row.py',
                        '--arm', f'{E}/without_rag_high_gwdg/norag_deepseek-ai_DeepSeek-V4-Flash.csv',
                        '--ds-side', f'{E}/without_rag_high_gwdg/judge_effort_low_deepseek.csv',
                        '--generated-on', 'GWDG, DeepSeek high prefix'], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH='analysis'))
    print(r.stdout[-1500:], r.stderr[-1500:])
    assert r.returncode == 0, 'main table row failed'
    rows = [dict(row=h, block=b, col=c, level=v[0], delta=v[1], sem2=v[2]) for (h, b, c), v in stats.items() if v]
    if corpus:
        rows += [dict(row=HEAD, block='corpus', col=k, level=corpus[k], delta=corpus['delta'] if k == 'combined' else 0.0,
                      sem2=corpus['sem2'] if k == 'combined' else 0.0) for k in ('federal', 'combined')]
    cells = pd.DataFrame(rows)
    cells.to_csv(CELLS, index=False)
    ess = cells[(cells.row == HEAD) & (cells.block == 'essay')].set_index('col')
    md = open(f'{R}/DS4-ablation-rerun.md').read()
    marker = '**The {\\tiny high} GWDG row (integrated '
    if marker not in md:
        table = '\n'.join(f'| {c} | {ess.level[c]:.2f} | {ess.delta[c]:+.2f} ± {ess.sem2[c]:.2f} |' for c in ess.index)
        section = (f"{marker}{time.strftime('%F %H:%M')} by ds4_low/integrate_high_gwdg.py).** Every "
                   "RAG cell and the no-RAG arm regenerated on GWDG at DeepSeek's real high (effort prefix), judged "
                   "by gpt-oss-120b and Qwen3.6-35B-A3B (NHR@FAU) and DeepSeek at low (DeepInfra); oracle from the "
                   "NHR@FAU 09-12 arm. Ladder essay block, against the GWDG no-RAG arm:\n\n| column | level | gain |\n"
                   f"|---|---|---|\n{table}\n\n" + (f"tab:corpus {{\\tiny high}}: federal {corpus['federal']:.2f}, combined "
                   f"{corpus['combined']:.2f} ({corpus['delta']:+.2f} ± {corpus['sem2']:.2f}). " if corpus else
                   "tab:corpus {\\tiny high}: not written -- the federal round-robin arm could not be generated. ") +
                   f"All cells: `{CELLS}`. "
                   "The main table's {\\tiny high} row now uses the GWDG no-RAG arm (recitation unchanged).\n\n")
        anchor = '**The {\\tiny high} rows, judged like the ladder'
        md = md.replace(anchor, section + anchor, 1) if anchor in md else md + '\n' + section
        open(f'{R}/DS4-ablation-rerun.md', 'w').write(md)
    rc = sh('git add DS4-ablation-rerun.md experiments/analysis/out/ladder_table.tex '
            'experiments/analysis/deepseek_rerun/results/main_results_table.tex '
            'experiments/analysis/deepseek_rerun/results/main_results_table_stacked.tex '
            'experiments/zubaers_result/essay_writing/*/DS4-high-gwdg experiments/zubaers_result/essay_writing/without_rag_high_gwdg '
            f'&& git add -f experiments/{CELLS}')
    assert rc == 0, 'git add failed'
    sh('git diff --cached --quiet || git commit -q -m "DeepSeek {\\tiny high} GWDG arms: generated, judged, in the tables" '
       '-m "Integrated by experiments/analysis/ds4_low/integrate_high_gwdg.py (cron supervisor)."')
    if sh('git pull --rebase --autostash -q') != 0:
        return 1
    return sh('git push -q')


if __name__ == '__main__':
    sys.exit(main())

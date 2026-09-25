"""Regenerate every figure with essay quality on an axis, on the re-judged panel.

Seven figures across two code paths, both driven from one place so they cannot
drift apart:

  experiments/analysis/plot_*.py     cost_performance_norag, aa_index_vs_essay_norag,
                                     recitation_vs_essay_norag
  src/notebook_eval/essay_plots.py   quality_vs_answer_length, quality_vs_generation_cost,
                                     quality_vs_legal_reference_similarity, and the
                                     combined essay_quality_relationships_combined

Both paths read the published no-RAG result CSV. Rather than fork them, they are
pointed at the re-judged drop-in that make_rejudged_result.py writes -- same columns,
three new judge seats in place of the retired panel's four, plus the 0731
re-generation as one more model. The plot_* scripts take it through GPBAM_RESULTS;
essay_plots takes it as the DataFrame it was always given.

    python rebuild_figures.py

Figures land where the published set already lives: experiments/analysis/out/ for the
first three, and beside the no-RAG result CSV for the other four. Both sets are
overwritten in place, so there is never a second copy to disambiguate.
"""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
REJUDGED = ROOT / 'experiments/zubaers_result/essay_writing/without_rag_0731/no_rag_ji2_rejudged.csv'
assert REJUDGED.exists(), f'run make_rejudged_result.py first -- no {REJUDGED}'

from src.notebook_eval import essay_evaluation, essay_plots  # noqa: E402

print('=== analysis/plot_*.py')
env = dict(os.environ, GPBAM_RESULTS=str(REJUDGED.relative_to(ROOT / 'experiments')))
for script in ('plot_cost_performance', 'plot_aa_index_vs_essay', 'plot_recitation_vs_essay'):
    r = subprocess.run([sys.executable, '-u', f'analysis/{script}.py'],
                       cwd=ROOT / 'experiments', env=dict(env, PYTHONPATH='analysis'),
                       capture_output=True, text=True)
    tail = [l for l in r.stdout.strip().split('\n') if l.strip()][-1:] or ['(no output)']
    print(f'  {script:26s} exit={r.returncode}  {tail[0][:90]}')
    if r.returncode:
        print(r.stderr.strip()[-600:])

print('\n=== src/notebook_eval/essay_plots.py')
summary = essay_evaluation.make_summary(
    pd.read_csv(REJUDGED),
    show_answer_tokens=True, show_gen_cost=True, show_legal_ref_sim=True,
    print_latex=False)
print(f'  summary: {len(summary)} models, columns {list(summary.columns)}')
# Beside the result CSV, which is where the published set lives -- not the module's
# default experiments/figures/essay_quality/, which would leave two sets of the same
# four figures on disk with no way to tell which the paper used.
FIGS = ROOT / 'experiments/zubaers_result/essay_writing/without_rag/ji2'
essay_plots.make_plots(summary, save_figures=True, show_plots=False, verbose=True,
                       output_directory=FIGS)
print('\ndone')

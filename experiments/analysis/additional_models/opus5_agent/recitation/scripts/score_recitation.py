"""Score the agent-written recitations against the reference CSVs.

Same metric as `experiments/legal_knowledge.ipynb`: row-wise ROUGE-L F1 from
`rouge_score.RougeScorer(['rougeL'], use_stemmer=False)`, target first,
prediction second, reported x100. Emits one CSV in the same shape as the newer
per-model files in `experiments/zubaers_result/article_recitation/`.

`completion_tokens` and `seconds` stay empty: the agent harness reports no
usage record, exactly as for the essay probe.
"""
import csv, pathlib, sys
import numpy as np
from rouge_score import rouge_scorer

HERE = pathlib.Path(__file__).resolve().parents[1]
ROOT = HERE.parents[4]
MODEL = 'anthropic/claude-opus-5'
N_BOOT, SEED = 10_000, 0
DATASETS = {
    'gpbam': ('GPBam Laws', 'gp_laws.csv'),
    'mostcited': ('Most cited Laws', 'most_cited_laws.csv'),
}

csv.field_size_limit(10 ** 9)
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)


def boot_sem(a, seed=SEED):
    rng = np.random.default_rng(seed)
    return rng.choice(a, size=(N_BOOT, a.size), replace=True).mean(axis=1).std(ddof=0)


rows, missing = [], []
if not (HERE / 'answers').is_dir():
    raise SystemExit(
        f"no {HERE / 'answers'}.\n"
        "answers/ is a generation output and is not tracked -- "
        "recitation_opus5_agent.csv,\n"
        "which this script writes, carries every answer verbatim and is the\n"
        "repository's copy. Re-run this only after re-generating.")

for key, (dataset, csv_name) in DATASETS.items():
    laws = list(csv.DictReader(open(ROOT / 'experiments/data' / csv_name, encoding='utf-8')))
    for i, law in enumerate(laws):
        path = HERE / 'answers' / key / f'ar_{i:03d}.txt'
        if not path.exists():
            missing.append(f'{key}/{i}')
            continue
        answer, target = path.read_text(encoding='utf-8').strip(), law['content']
        rows.append({'i': i, 'model': MODEL,
                     'query': f"{law['law_book']} {law['article']}",
                     'target': target, 'dataset': dataset, 'answer': answer,
                     'score': scorer.score(target, answer)['rougeL'].fmeasure,
                     'completion_tokens': '', 'seconds': '', 'error': ''})

if missing:
    sys.exit(f'missing {len(missing)} answers: {missing}')

# Two destinations, same bytes. The probe's own copy sits with the prompts,
# answers and audit that justify it; the copy in zubaers_result is what
# rebuild_tables.py and plot_recitation_vs_essay.py glob as `recitation_*.csv`,
# the same drop-point the 0731 / gemma / phi rows use. Written together so the
# table can never be reading a stale score.
DESTS = [HERE / 'recitation_opus5_agent.csv',
         ROOT / 'experiments/zubaers_result/article_recitation/recitation_opus5_agent.csv']
for out in DESTS:
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

for dataset in (d for d, _ in DATASETS.values()):
    s = np.array([r['score'] for r in rows if r['dataset'] == dataset]) * 100
    lens = [len(r['answer']) for r in rows if r['dataset'] == dataset]
    print(f'{dataset}: n={len(s)}')
    print(f'  ROUGE-L F1  {s.mean():.2f} (sem {boot_sem(s):.2f}, std {s.std(ddof=1):.2f})')
    print(f'  min {s.min():.2f}  median {np.median(s):.2f}  max {s.max():.2f}'
          f'   >=90: {(s >= 90).sum()}')
    print(f'  answers {min(lens)}-{max(lens)} chars, mean {np.mean(lens):.0f}')
print(f'wrote {len(rows)} rows to:\n  ' + '\n  '.join(str(d) for d in DESTS))

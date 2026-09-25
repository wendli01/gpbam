"""Regenerate the exact Article-Recitation prompt each benchmarked model receives.

`experiments/legal_knowledge.ipynb` builds the run as

    qa.AnswerGenerator(prompt=prompts.AR_USER, system_prompt=prompts.AR_SYSTEM,
                       model=model, max_tokens=None)
    ar.predict(dataset.law_book + ' ' + dataset.article)

With no retriever, `AnswerGenerator.predict` passes an empty context, so
`build_messages` reduces the whole wire payload to

    [{'role': 'system', 'content': AR_SYSTEM},
     {'role': 'user',   'content': AR_USER.format('', query)}]

The two datasets spell the query differently -- "vwgo 80" for GPBam Laws,
"VwGO § 154" for Most cited Laws -- because each is just its own CSV's
``law_book + ' ' + article``. Both are reproduced verbatim rather than
normalised, and both are asserted against the `prompt` column of the published
`article_recitation.csv`.
"""
import ast, csv, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parents[1]
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
from src.prompts import AR_SYSTEM, AR_USER

DATASETS = {
    'gpbam': ('GPBam Laws', 'gp_laws.csv'),
    'mostcited': ('Most cited Laws', 'most_cited_laws.csv'),
}
csv.field_size_limit(10 ** 9)


def payload(query):
    return [{'role': 'system', 'content': AR_SYSTEM},
            {'role': 'user', 'content': AR_USER.format('', query)}]


def queries(csv_name):
    rows = list(csv.DictReader(open(ROOT / 'experiments/data' / csv_name, encoding='utf-8')))
    return [f"{r['law_book']} {r['article']}" for r in rows], rows


published = {}
for row in csv.DictReader(open(
        ROOT / 'experiments/zubaers_result/article_recitation/article_recitation.csv',
        encoding='utf-8')):
    if row['prompt']:
        published.setdefault((row['dataset'], row['query']), set()).add(row['prompt'])

for key, (dataset, csv_name) in DATASETS.items():
    qs, _ = queries(csv_name)
    checked = 0
    for q in qs:
        for stored in published.get((dataset, q), ()):
            assert ast.literal_eval(stored) == payload(q), (dataset, q)
            checked += 1
    assert checked, f'no published prompt matched for {dataset}'
    out = HERE / 'prompts' / key
    out.mkdir(parents=True, exist_ok=True)
    for i, q in enumerate(qs):
        (out / f'ar_{i:03d}.txt').write_text(
            f'[SYSTEM]\n{AR_SYSTEM}\n\n[USER]\n{AR_USER.format("", q)}\n', encoding='utf-8')
    print(f'{dataset}: {checked} published payloads verified, {len(qs)} prompts -> {out}')

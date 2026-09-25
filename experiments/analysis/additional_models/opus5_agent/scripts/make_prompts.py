"""Regenerate the exact essay-writing prompt each benchmarked model receives.

No-RAG generation goes through src.qa.AnswerGenerator with system_prompt=''
(the class default, which essay_writing_norag.py does not override) and an
empty retrieval context, so the whole wire payload is

    [{'role': 'system', 'content': ''},
     {'role': 'user',   'content': QA_USER.format('', facts)}]

Verified against the qa_prompt column stored in the published no-RAG result
CSV: identical for case 0 (7,649 characters).
"""
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from src.prompts import QA_USER

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else '.')
# argv[2] is either a count ("10" -> cases 0-9) or a half-open range ("62:72").
spec = sys.argv[2] if len(sys.argv) > 2 else '10'
CASES = range(*map(int, spec.split(':'))) if ':' in spec else range(int(spec))

facts = json.load(open(ROOT / 'experiments/data/gpbam.json'))['facts']
OUT.mkdir(parents=True, exist_ok=True)
for i in CASES:
    f = facts[str(i)] if isinstance(facts, dict) else facts[i]
    (OUT / f'prompt_{i}.txt').write_text(QA_USER.format('', f), encoding='utf-8')
print(f'wrote {len(list(CASES))} prompts to {OUT}')

"""Turn the agent-written essays into the jsonl shape judge_0731.py expects.

Only essays whose agent reported completion are listed here; inferring
completion from file existence once caught essay 5 mid-write.

The `model` field is what the downstream analysis joins on, and it is the real
OpenRouter slug rather than an invented one: the essays answer the benchmark
prompt with the benchmark's model, and every consumer of this file documents
separately that the *harness* was the agent loop, not the API.
"""
import json, pathlib, sys
D = pathlib.Path(__file__).resolve().parent.parent
if not (D / 'essays').is_dir():
    raise SystemExit(
        f"no {D / 'essays'}.\n"
        "essays/ is a generation output and is not tracked -- opus5_agent.jsonl,\n"
        "which this script writes, is the repository's copy of those essays and\n"
        "carries their text verbatim. Re-run this only after re-generating.")
COMPLETE = list(range(81))
MODEL = 'anthropic/claude-opus-5'
rows = []
for i in COMPLETE:
    f = D / 'essays' / f'essay_{i:02d}.md'
    if not f.exists():
        print(f'MISSING essay_{i:02d}.md', file=sys.stderr); continue
    txt = f.read_text(encoding='utf-8').strip()
    rows.append({'index': i, 'model': MODEL, 'answer': txt, 'chars': len(txt)})
with (D / 'opus5_agent.jsonl').open('w', encoding='utf-8') as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'wrote {len(rows)} essays to {D / "opus5_agent.jsonl"}')

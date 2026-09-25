"""Audit the subagent transcripts behind the agent-harness essays.

Each agent was told to make exactly two tool calls -- a Read of its own prompt
file, which holds the benchmark payload and nothing else, and a Write of its
answer. This parses every transcript that wrote into `essays/` and reports the
tool_use sequence it actually made, plus a scan for any reference to the
examiner solutions or the reference corpora that sit in this repo. Anything
other than [Read, Write], or any hit on the corpus strings, is a finding.

The legitimate Read targets a prompt file outside the repo, so none of the
CORPUS patterns can match it; they appear nowhere in a clean transcript.

Run with the transcript directory as argv[1] (the session's tasks/ directory).
"""
import collections, glob, json, pathlib, sys

TASKS = pathlib.Path(sys.argv[1])
MARKER = 'essays/essay_'
CORPUS = ('02_Loesungen', 'gpbam.json', 'knowledge_base', 'zubaers_result',
          'gp_laws', 'most_cited', 'no_rag_ji2')
EXPECT = ['Read', 'Write']

tools, suspect, corpus_hits, seen, n = collections.Counter(), [], [], [], 0
for f in sorted(glob.glob(str(TASKS / 'a*.output'))):
    txt = open(f, encoding='utf-8', errors='replace').read()
    if MARKER not in txt:
        continue
    n += 1
    names, targets = [], []
    for line in txt.splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        content = (rec.get('message') or {}).get('content')
        for blk in content if isinstance(content, list) else []:
            if isinstance(blk, dict) and blk.get('type') == 'tool_use':
                names.append(blk.get('name'))
                inp = blk.get('input') or {}
                targets.append(inp.get('file_path') or '')
    tools.update(names)
    seen.append((pathlib.Path(f).name, names, targets))
    if names != EXPECT:
        suspect.append((pathlib.Path(f).name, names))
    for pat in CORPUS:
        if pat in txt:
            corpus_hits.append((pathlib.Path(f).name, pat))

print(f'essay transcripts: {n}')
print(f'tool calls: {dict(tools)}')
for name, names, targets in seen:
    print(f'  {name}: {names} -> {[pathlib.Path(t).name for t in targets if t]}')
print(f'not exactly {EXPECT}: {suspect if suspect else "none"}')
print(f'corpus references:  {corpus_hits if corpus_hits else "none"}')
ok = (n and not suspect and not corpus_hits
      and tools['Read'] == n and tools['Write'] == n and set(tools) == {'Read', 'Write'})
print('AUDIT PASS' if ok else 'AUDIT FAIL')
sys.exit(0 if ok else 1)

"""Audit the subagent transcripts behind the recitation answers.

Each agent was told to make exactly one tool call -- the Write of its own
answer -- and nothing else. This parses every transcript that wrote into
`recitation/answers/` and reports the tool_use sequence it actually made, plus
a scan for any reference to the reference corpora. Anything other than a single
Write, or any hit on the corpus strings, is a finding.

Run with the transcript directory as argv[1] (the session's tasks/ directory).
"""
import collections, glob, json, pathlib, sys

TASKS = pathlib.Path(sys.argv[1])
MARKER = 'recitation/answers/'
CORPUS = ('gp_laws', 'most_cited', 'experiments/data', 'article_recitation',
          'knowledge_base', 'gpbam.json')

tools, suspect, corpus_hits, n = collections.Counter(), [], [], 0
for f in sorted(glob.glob(str(TASKS / 'a*.output'))):
    txt = open(f, encoding='utf-8', errors='replace').read()
    if MARKER not in txt:
        continue
    n += 1
    names = []
    for line in txt.splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        content = (rec.get('message') or {}).get('content')
        for blk in content if isinstance(content, list) else []:
            if isinstance(blk, dict) and blk.get('type') == 'tool_use':
                names.append(blk.get('name'))
    tools.update(names)
    if names != ['Write']:
        suspect.append((pathlib.Path(f).name, names))
    # The tool *definitions* in every system prompt mention many tool names, so
    # only the corpus paths are meaningful here -- they appear nowhere by default.
    for pat in CORPUS:
        if pat in txt:
            corpus_hits.append((pathlib.Path(f).name, pat))

print(f'recitation transcripts: {n}')
print(f'tool calls: {dict(tools)}')
print(f'not exactly [Write]: {suspect if suspect else "none"}')
print(f'corpus references:   {corpus_hits if corpus_hits else "none"}')
ok = n and not suspect and not corpus_hits and set(tools) == {'Write'} and tools['Write'] == n
print('AUDIT PASS' if ok else 'AUDIT FAIL')
sys.exit(0 if ok else 1)

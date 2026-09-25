# Claude-Opus-5 article-recitation probe (agent-harness generation)

All 200 GPBam article-recitation prompts -- 100 GPBam Laws and 100 Most cited
Laws -- recited by Claude Opus 5 on 2026-08-21. The recitation counterpart to the
essay probe one directory up; read that README first, because the harness caveats
there apply here too, with one important exception noted below.

    dataset            ROUGE-L F1              next best
    GPBam Laws         80.40  (sem 2.43)       48.46  mistral-large-2512
    Most cited Laws    89.97  (sem 1.44)       50.54  mistral-large-2512

Full coverage, so both numbers are directly comparable to the published columns.
Opus is roughly 1.7x-1.8x the runner-up on each and about 10 SEMs clear.

## What the prompt was

Byte-identical to what every benchmarked model receives. `legal_knowledge.ipynb`
builds the run as `AnswerGenerator(prompt=AR_USER, system_prompt=AR_SYSTEM,
max_tokens=None)` over `law_book + ' ' + article`, and with no retriever attached
`predict` passes an empty context, so the wire payload is just

    [{'role': 'system', 'content': AR_SYSTEM},
     {'role': 'user',   'content': 'Wie lautet der genaue Wortlaut von vwgo 80?'}]

The two datasets spell the query differently -- `vwgo 80` for GPBam Laws,
`VwGO § 154` for Most cited Laws -- because each is its own CSV's `law_book`
plus `article`. Both surface forms are reproduced verbatim rather than
normalised. `scripts/make_ar_prompts.py` regenerates them and asserts them
against the `prompt` column of the published `article_recitation.csv`: all 5,400
published rows collapse to 200 distinct payloads and every one matches.

## How this differs from a benchmarked row

Same caveats as the essay probe, except that **its caveat 2 does not apply
here**: the benchmarked recitation runs pass `max_tokens=None` themselves, so
there is no generation ceiling to be unmatched on. What remains:

1. **System prompt.** A subagent cannot run with a custom system prompt.
   `AR_SYSTEM` was delivered as labelled text inside the user turn; the real
   system message was the agent harness, its tool definitions, CLAUDE.md and
   memory. Every benchmarked row had `AR_SYSTEM` as an actual system message.
2. **Sampling settings.** Harness defaults, not the benchmarked configuration.
3. **Tool access.** The agents had file tools and the reference statute texts sit
   in `experiments/data/` in this repo. Each agent was forbidden any tool call but
   the single Write of its own answer. Unlike the essay probe this audit is
   exhaustive, not a spot check: `scripts/audit_transcripts.py` parses all 200
   transcripts and every one contains **exactly one tool_use block, and it is a
   Write** -- 200 calls across 200 agents, no Read, Grep, Glob, Bash, WebSearch
   or WebFetch, and no occurrence of `gp_laws`, `most_cited`, `experiments/data`,
   `article_recitation`, `knowledge_base` or `gpbam.json` in any transcript.

## Corpus ordering: the first half is the harder half

Not a caveat on this row any more -- it is fully covered -- but a real property
of the corpus, recorded so nobody re-derives it from a partial run.

`gp_laws.csv` is sorted by citation count descending, and the most-cited norms
are the longer ones (median target 772 characters in the first half against 512
in the second). Longer targets score lower on ROUGE-L F1, so rows 0-49 are the
harder half: across all 31 models the first-50 mean sits **2.24 points below**
the full-100 mean, 27 of 31 negative.

Opus is the exception, and informatively so: **+0.20** on GPBam and **+0.01** on
Most cited. It sits near ceiling on both halves, so the length gradient that
costs weaker models two points does not bite. Any partial recitation run on a
weak model must not be read against a full-100 column; on a strong one it
happens not to matter.

## What the failures look like

Both distributions are heavily top-loaded -- 59 of 100 GPBam articles and 72 of
100 Most cited above 90, i.e. essentially verbatim -- with a thin tail.

The tail is **verbosity on short norms, not ignorance**. Score correlates
*positively* with target length (Spearman +0.31 GPBam, +0.20 Most cited), the
opposite of the corpus-wide pattern, because the model appends explanatory notes
that wreck precision when the norm itself is one or two sentences. The worst
cases are all of that shape: `vwgo 45` is a 125-character norm answered in 1,162
characters, `SGG § 86` 220 characters answered in 2,222, `urhdag 19` 352
answered in 1,524. Recall is high in these cases and F1 is not.

Stripping the commentary would raise both numbers, and no benchmarked model was
given that courtesy, so nothing is stripped. The answers are scored exactly as
written, through the same `rouge_scorer(['rougeL'], use_stemmer=False)` call
with the target first and the prediction second.

## Files

| path | what |
|---|---|
| `prompts/{gpbam,mostcited}/ar_NNN.txt` | the regenerated wire payload per article |
| `answers/{gpbam,mostcited}/ar_NNN.txt` | one agent's answer, exactly as written |
| `recitation_opus5_agent.csv` | 200 per-article scores, schema of the newer per-model files |

`answers/` and `prompts/` are **not tracked**: the CSV above carries every
answer verbatim in its `answer` column, and `make_ar_prompts.py` regenerates the
prompts deterministically from the law CSVs. That is 400 files of exact
duplication kept out of every diff. Both directories are still what a re-run
writes and reads; they are simply not the repository's copy.
| `model_comparison.csv` | Opus against all 30 benchmarked models on both datasets |
| `scripts/make_ar_prompts.py` | regenerate + verify the payloads |
| `scripts/score_recitation.py` | ROUGE-L scoring |
| `scripts/compare_models.py` | the ranking, plus the corpus-ordering effect |
| `scripts/audit_transcripts.py` | the tool-call audit; exits non-zero on any finding |

`completion_tokens` and `seconds` are empty in the CSV: the agent harness
reports no usage record, exactly as for the essay probe.

## Rebuild

    conda activate plexam
    cd experiments/analysis/additional_models/opus5_agent/recitation
    python scripts/make_ar_prompts.py     # regenerate + verify prompts
    python scripts/score_recitation.py    # -> recitation_opus5_agent.csv
    python scripts/compare_models.py      # -> model_comparison.csv
    python scripts/audit_transcripts.py <session tasks dir>

Generation itself is not scripted: it ran as 200 Claude Code subagents, one per
article, each given only its payload.

## Where this row should appear

Not yet placed in the paper tables. It is a full-coverage measurement on both
recitation datasets, so unlike the essay row it has no sampling caveat -- but it
shares the essay row's harness caveats and should carry the same `$^\ddagger$`
marking if it is used. The `recitation_vs_essay_norag` figure currently excludes
Opus for want of a recitation run; that reason no longer holds, though the essay
axis for that point is still the 48-case agent-harness score.

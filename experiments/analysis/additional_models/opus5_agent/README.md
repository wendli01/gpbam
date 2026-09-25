# Claude-Opus-5 essay probe (agent-harness generation)

All 81 GPBam essays written by Claude Opus 5 between 2026-08-20 and 2026-08-22,
replacing the pending `anthropic/claude-opus-5` placeholder row with something
measured. Coverage is now complete, so the row's only remaining caveat is the
harness one below -- which is the important one and does not shrink with n.

## What the prompt was

Byte-identical to what every benchmarked model receives for no-RAG essay writing.
`src.qa.AnswerGenerator` defaults `system_prompt=''` and `essay_writing_norag.py`
does not override it, so the system message is literally empty -- `QA_SYSTEM` in
`src/prompts.py` is dead code on this path. No-RAG also passes an empty retrieval
context, so the payload reduces to `QA_USER.format('', facts)`.

`scripts/make_prompts.py` regenerates it. Checked against the `qa_prompt` column
stored in the published no-RAG result CSV: case 0 matches at 7,649 characters.

## How these differ from a benchmarked row -- READ BEFORE CITING

These essays were produced by Claude Code subagents, not by an API call through
`AnswerGenerator`. That is not equivalent, in four ways:

1. **System prompt.** A subagent cannot run with an empty system prompt. It carries
   the agent harness prompt, tool definitions, CLAUDE.md and memory. Every other
   row in the table was generated with an empty system message.
2. **No `max_tokens` ceiling.** Benchmarked runs cap generation; these did not.
   The essays average 56.1k characters against a 11.4k median for the 30 benchmarked
   models on case 0 -- roughly 5x. Since `Judge Tokens` correlates with essay
   quality at rho=0.95 and `Answer Tokens` at 0.63, length is partly *why* any
   score here will look good. Do not read the score as effort-matched.
3. **Sampling settings.** Benchmarked rows run at `reasoning_effort='high'` with
   provider-default temperature; the agents used harness defaults.
4. **Tool access.** The agents had file tools, and the official examiner solutions
   sit in `experiments/data/02_Loesungen` in this repo. Each agent was explicitly
   forbidden to read any file but its own prompt. That is an instruction, not a
   sandbox. Every transcript was audited afterwards for references to the solution
   directory and to `gpbam.json`; all came back zero, and tool-call counts were
   consistent with one Read plus the writes. Treat as strong evidence, not proof.

The honest path to a citable row is `experiments/analysis/deepseek_rerun/scripts/run_model.py`
with `RUN_MODEL=anthropic/claude-opus-5`, which goes through the identical code
path as every other model.

## Coverage

All 81 cases, 0-80. There is no subset to argue about any more: the essay
columns average the same cases as every other row in the table, and the SE is
over the same 81.

The row got here in five batches (0-47, 48-51, 52-61, 62-71, 72-80), each run
the same way. `scripts/subset_deviation.py` is what quantified the gap while it
was open -- for each of the models with all 81 cases, a random 62-case subset
lands a mean absolute 0.44 points from that model's own full-81 mean (p95 1.26),
and 0.29 at 72 (p95 0.79). Those figures moved slightly when this row reached 81
itself: the reference set is "models with full coverage", which this row now
joins, so it is 34 models rather than the 33 quoted before. It is kept because it
is the tool that says what a
partial row is worth, not because this row still needs it.

Essay lengths run 34,439 to 82,771 characters, mean 56,104.

## Judging

`essays/` holds one file per case as each agent wrote it, and is **not tracked**:
`opus5_agent.jsonl` carries the same text verbatim and is the repository's copy
of the corpus. Tracking both put 81 files in every diff for a second copy of the
same bytes. The directory is still what a re-generation writes into, and
`build_jsonl.py` still reads it -- it just is not the record.

`scripts/build_jsonl.py` converts the essays into the `{index, answer}` shape that
`experiments/analysis/deepseek_rerun/scripts/judge_0731.py` consumes. That script is
parameterised via `JUDGE_GEN`, `JUDGE_OUTD` and `JUDGE_WAIT=0`, and scores with the
same three NHR@FAU seats as the re-judged panel -- gpt-oss-120b, Qwen3.6-35B-A3B-FP8
and DeepSeek-V4-Flash at current (0731) weights -- under `JUDGE_INSTRUCTION_V2`.

`judged/` holds the raw per-seat output; `scores.csv` the per-case matrix,
built by `scripts/build_scores.py`.

    ensemble median  76.42  (sem 2.35, n=81)
    per seat         qwen 90.49, ds-0731 74.32, gpt-oss 70.12

`scripts/build_scores.py` rebuilds that table from `judged/`; it reproduces the
n=48 CSV byte-for-byte, so the number above is not a hand count.

The five batches and what each did to the mean, since the stability of the
number across them is the reason to trust it:

| cases | batch mean | row mean after |
|---|---|---|
| 0-47   | 77.71 | 77.71 |
| 48-51  | 80.00 | 77.88 |
| 52-61  | 77.00 | 77.74 |
| 62-71  | 65.00 | 75.97 |
| 72-80  | 80.00 | 76.42 |

Four of the five batches land within three points of the row mean. The
exception is 62-71 at 65.00, which pulled the row down 1.77 -- the largest
single move, and still inside the row's own SE. Nothing about that batch was
run differently; it is the same spread the bimodality below describes,
showing up in a block of ten.

Two things to read before quoting that number.

The seats disagree by 20 points -- qwen 90.49 against gpt-oss 70.12 -- which is
much wider than their spread on the benchmarked rows, and the median is doing a
lot of work.

The distribution is bimodal, which is why `scores.csv` is worth reading over the
mean: 42 of 81 cases land at 90 or above, but eleven land at 30 or 40, with a
thin waist between. Adding cases never smoothed this out -- it was there at
n=33 and survived every batch to full coverage -- so it is a property of how the
model meets particular cases, not sampling noise. The mean is a poor summary of
that shape and the sem understates it.

`deepseek_rerun/scripts/score_shape.py` checks whether this is peculiar to the
row or just what a strong model looks like. It is peculiar: Sarle's bimodality
coefficient is 0.64 here against the 5/9 reference, with 51.9% of cases in the
top bin and 13.6% in the bottom, while gemini-3.7-flash -- the next-strongest
row -- sits at 0.50 and is single-humped.

For scale, gemini-3.7-flash -- measured properly, end to end, through
`run_model.py` and the same seats -- scores 67.10. These numbers are not
comparable, for the four reasons at the top of this file.

## Where this row appears

It is a marked row, never a plain one. As of 2026-08-22 it is carried in:

| artefact | how it is marked |
|---|---|
| `deepseek_rerun/results/main_essay_table.{tex,csv}` | `$^\ddagger$` on the name, with the harness caveat spelled out in the header comment |
| `deepseek_rerun/results/main_results_table.tex` | same marker, same caveat, in the merged view |
| `analysis/out/cost_performance_norag.{pdf,png,csv}` | `*` for the imputed price. **No longer hollow**: the hollow marker meant "x and bubble size extrapolated from a short row", and at 81 of 81 there is nothing left to extrapolate |
| `analysis/out/aa_index_vs_essay_norag.{pdf,png,csv}` | plotted at AA 63, which is an upper bound -- see below |
| `analysis/out/recitation_vs_essay_norag.*` | present, drawn hollow for the harness. It was excluded while the essay axis was a partial score; that reason is gone |
| `analysis/out/case_difficulty_*_norag.*` | drawn dashed/hollow for the harness, now over all 81 cases |
| `deepseek_rerun/results/associations_essay_quality.tex` | inside every association; the header quantifies what it is worth to the Cost row |
| `without_rag/ji2/quality_vs_generation_cost.pdf` | **absent**: that figure plots reported costs only, and this row has none -- the harness reports no usage record, so `total_cost` is empty for all 81 |

The AA index is 63, and that is the only configuration Artificial Analysis
publishes for Opus 5 -- (Adaptive Reasoning, Max Effort), re-checked 2026-08-21.
Our essays are at high effort. Max effort never scores below high, so the point
can only move left, never right.

## Cost, which is estimated twice over

`scripts/estimate_tokens.py` writes `token_estimate.csv`. The agent harness
reports no usage record at all, so there is nothing to read off; the price and
performance figure would otherwise have to drop the most expensive model in the
study. Three steps, in decreasing order of how much they can be trusted:

1. **Prompt tokens are a lookup, not an estimate.** `claude-haiku-4.5` answered
   the same 81 prompts through the API and Anthropic tokenises with one tokeniser
   across the family, so its per-case `prompt_tokens` *is* Opus's.
2. **Visible output is measured** from the essays themselves, at the Claude
   chars-per-token ratio read off that same run: 8,958 prompt characters against
   3,220 prompt tokens, so 2.782 chars/token on German legal prose.
3. **Thinking tokens are carried over** from haiku on the same case index --
   they are billed as output and are invisible here. This is the weak step. It
   assumes the thinking budget tracks the difficulty of the case rather than the
   length of the answer. The opposite assumption, that Opus thinks 2.3x as much
   because it writes 2.3x as much, is in the CSV as
   `completion_tokens_proportional` and prices about 45% higher.

At OpenRouter list ($5.00 / $25.00 per 1M, fetched 2026-08-21) that is **$0.78
per essay and $63.36 over all 81** -- against $88.66 under the
proportional-thinking assumption. Eight times what `claude-haiku-4.5` actually
cost, and six times the next-largest column entry.

Nothing is extrapolated any more: the figure's total is the sum over the 81
essays that exist. What remains estimated is the token count itself, by the
three steps above, and that does not go away with coverage.

## Legal reference similarity

34.25, recomputed by `deepseek_rerun/scripts/recompute_legal_ref.py` through the
same extractor as every other row in the table -- the highest of the 32, above
gemini-3.7-flash at 33.56.

The mean is over 80 of the 81 cases. Case 7 has no defined similarity for any
model, because its examiner solution cites nothing the extractor recognises, so
there is no reference set to agree with. Same denominator for every row.

**This section used to describe a defect that has now been fixed** (2026-08-22),
and the number above is the post-fix one. `RefExtractor.extract` collected the
markers and then ran them through `replace_content`, which rewrites the text
with `[ref=...]` spans and raises `RefExError` when two markers overlap.
`src.scoring.legal_ref_similarity` never used that rewritten text -- only the
markers -- so the check was discarding every reference in a document over one
bad marker. Case 61 scored 0.00 despite 111 `§` and 247 `Art.` citations, killed
by one marker overlapping a `____________` blank; it now scores on its 47
references. Twelve of the 2,754 corpus essays were hit, and so were the
solutions of cases 34, 73 and 74 -- which is why those three were undefined for
every model at once and are now scored. Extraction stops before
`replace_content`; the markers are the same objects `extract` would have
returned. On 396 sampled essays that never raised, the reference set is
identical, so the change is a no-op except where it used to throw everything
away. It moved this row +1.06 and every other row by less.

Cases 57 and 69 still score 0.00 and are *not* that defect -- both extract
cleanly and genuinely share no `(book, section)` pair with their solution.

Read the number with caveat 2 above: these essays are about five times longer
than the benchmarked median, and a longer essay has more room to cite.

## Article recitation

`recitation/` holds the recitation counterpart, added 2026-08-21: all 200
prompts, both datasets, same agent harness, one Write per agent and nothing else.

    GPBam Laws       80.40 (sem 2.43)    next best 48.46
    Most cited Laws  89.97 (sem 1.44)    next best 50.54

These carry the harness caveats above, minus the `max_tokens` one, which does not
apply because the benchmarked recitation runs are uncapped too. The audit is
stronger than the essay probe's: all 200 transcripts were parsed and every one
holds exactly one tool_use block, a Write.

## Notes on individual cases

Essay 5 was generated before the agents were told to write in a single operation.
Its file was written in nine appends and an early judging pass snapshotted it
mid-write at 60,621 of its final 82,772 characters; those rows were discarded and
it was re-judged on the complete text, scoring 60 rather than the 70 the truncated
version drew. Every later agent was instructed to write in one operation, and no
further mid-write snapshot occurred. The essay here is the complete one.

Cases 41 and 45 were killed by a session limit mid-composition on the first
attempt, wrote no file, and were regenerated from scratch afterwards.

Every batch was audited by `scripts/audit_essay_transcripts.py` before its
essays were judged: one Read of the agent's own prompt file, which lives outside
the repository, one Write of the essay, and no other tool call or reference to
the solution directory in any transcript.

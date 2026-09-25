# Results: the 0731 judge seat, and gpt-oss-120b

Two seats re-judged all **2,430** no-RAG essays (30 models × 81 cases) from the
byte-identical `ji2` `judge_prompt` in the published CSV, at `effort=high`.
81/81 parsed, **0 errors**, every model, both seats.

| file | seat | note |
|---|---|---|
| `DeepSeek-V4-Flash-0731.jsonl` | `deepseek-ai/DeepSeek-V4-Flash` | current weights; the published column is the retired checkpoint |
| `gpt-oss-120b.jsonl` | `gpt-oss-120b` | new GPT seat, replacing the billed and AA-deprecated `gpt-5-nano` |

**One file per seat**, one JSON object per (model, case):
`index, model, score, n_matches, judgement, prompt_tokens, completion_tokens,
seconds, finish_reason, error`. Full judgement text is kept, so every score is
auditable. `summary_by_model.csv` has the per-model means, and
`<seat>.run.log` each seat's run log.

Until 2026-08-23 a seat was a *directory* of one CSV per model -- 30 files a
seat, 90 in the tree, which is most of why a merge request showed 700 changed
files. A seat is one experiment, so it is now one file. `scripts/seats.py` is
the only thing that knows the layout; `scripts/merge_seat_files.py` did the
conversion and verified it cell by cell, and is kept for provenance. Nothing was
dropped: every field of every row survived, and every table and figure in this
directory rebuilds byte-identically across the change.

## The checkpoint is worth +4 points, uniformly — except for DeepSeek itself

DeepSeek-V4-Flash as a judge, published (retired checkpoint) against 0731, same
essays, same prompts, same effort:

```
mean shift  +4.04   sd 1.32   range 0.00 .. +7.27
higher on   29/30 models     paired t = 16.69, p = 2.1e-16
model-level r = 0.989        Kendall tau-b = 0.941
```

Two things follow, and they point in opposite directions.

**The leaderboard ordering survives.** τ-b = 0.941 and r = 0.989: the swap is close
to a uniform level shift, so relative model ranking is largely intact. The published
*ordering* is not wrong.

**But the leader gains disproportionately.** `DeepSeek-V4-Flash`'s own row moves
**+7.27** against a mean of +4.04 (sd 1.32) — 2.4 sd above every other model, and the
largest shift in the table. The new checkpoint is markedly more generous to its own
essays than to anyone else's, which is a self-preference effect landing precisely on
the model at the top of the leaderboard. Net of the uniform shift it is worth about
**+3.3 points** to the leader alone.

(`mistral-small-3.1-24b-instruct` shows +0.00 because both seats score it 0.0
throughout — it emitted a 25-character stub on all 81 cases.)

## Why this matters beyond the AA figure

The published no-RAG DeepSeek judge column is **not internally consistent**.
Fingerprinting every model across the swap boundary (`0a3baf52`, 2026-07-24, against
`HEAD`):

- **25/30** models unchanged since before the swap → judged by the retired checkpoint
- **4** re-judged after 07-24 — `Mistral-Medium-3.5-128B`, `gemma-4-31B`,
  `Mistral-Small-3.2-24B`, `granite-4.1-3b`
- **1** added after 07-24 — `qwen3-next-80b-a3b-instruct`

Those five sit in a window (07-24 → 08-17, committed 08-03) that straddles 2026-08-01,
so they may carry the +4-point-more-lenient judge while the other twenty-five do not.
That is larger than most gaps between adjacent models in the table.

On the RAG side it is sharper still: the DeepSeek seat's measured RAG benefit is
**+4.74** (`rrf` − `no_rag`), while the checkpoint effect is **+4.04** on average and
`no_rag` was judged pre-swap for most models where `rrf` was judged 2026-08-17. Most
of that seat's apparent retrieval benefit could be the judge changing underneath the
comparison rather than retrieval doing anything.

## The control that makes this interpretable

`Qwen3.6-35B-A3B-FP8` was **not** swapped, so re-running it isolates re-run drift from
the checkpoint effect. On `DeepSeek-V4-Flash`'s 81 essays:

| seat | published | today | delta |
|---|---|---|---|
| Qwen3.6-35B (model unchanged) | 51.98 | 53.02 | **+1.05 ± 1.24 SEM** — n.s. |
| DeepSeek-V4-Flash (0731) | 30.88 | 38.15 | **+7.38 ± 1.64 SEM** |

Re-run drift is ~1 point and indistinguishable from zero; the checkpoint effect net of
it is **+6.33** on that model. Without this control the 7.4-point shift would have been
unattributable. The Qwen seat is still running to extend the control to all 30 models.

---

# Final results (all seats complete)

Three seats × 2,430 essays, plus the 0731 re-generation and its judging. 81/81 parsed,
0 errors, everywhere.

| directory | what |
|---|---|
| `gpt-oss-120b.jsonl`, `DeepSeek-V4-Flash-0731.jsonl`, `Qwen3.6-35B.jsonl` | 30 models × 81 essays per seat, one file each |
| `generator_0731/` | the 81 re-generated essays + all three seats' judgements of them |

## The checkpoint moves the judge, not the generator

**As a judge (n = 30 models):**

| | shift | |
|---|---|---|
| Qwen3.6-35B — *unchanged model, the control* | **+0.18 ± 0.15** | t = 1.21, **p = 0.24, n.s.** |
| DeepSeek-V4-Flash — 0731 | **+4.04 ± 0.24** | |
| **checkpoint, net of drift** | **+3.85 ± 0.26** | |

The control is decisive: re-running an *unchanged* model moves scores by 0.18 points
and is not distinguishable from zero, so essentially the whole +4.04 is the checkpoint.

**As a generator (n = 81 essays, both sets judged by the same three seats today, so the
generator is the only difference):**

| seat | published | 0731 | delta | p |
|---|---|---|---|---|
| gpt-oss-120b | 44.14 | 41.54 | −2.59 | 0.238 |
| DeepSeek-V4-Flash-0731 | 38.15 | 37.78 | −0.37 | 0.856 |
| Qwen3.6-35B | 53.02 | 53.83 | +0.80 | 0.716 |
| **panel median** | **44.44** | **43.77** | **−0.68** | **0.732** |

A clean null. And it is a null despite the generation changing substantially:

```
essay length     20,066 -> 17,709 chars   (-11.7%)
completion tok    6,064 -> 10,618         (+75.1%)
```

The 0731 checkpoint thinks **75% harder**, writes **12% less**, and scores the same.

**So the swap matters for judging and not for generating.** The published
`DeepSeek-V4-Flash` generation row is defensible as it stands; the judge column is not.

## The new leaderboard

All three seats today (`gpt-oss-120b` / `Qwen3.6-35B` / `DeepSeek-V4-Flash-0731`):

| model | published | new | Δ |
|---|---|---|---|
| DeepSeek-V4-Flash | 46.36 | 44.44 | −1.91 |
| qwen3.5-397b-a17b | 43.33 | 41.48 | −1.85 |
| mistral-large-2512 | 41.73 | 37.84 | −3.89 |
| Mistral-Medium-3.5-128B | 41.73 | 37.41 | −4.32 |
| deepseek-v3.2 | 38.15 | 36.79 | −1.36 |
| gemini-2.5-flash-lite | 36.67 | 35.19 | −1.48 |
| gpt-5-mini | 38.33 | 34.81 | −3.52 |
| claude-haiku-4.5 | 36.30 | 34.81 | −1.48 |

Mean Δ **−1.10**; τ-b **0.964**, r **0.996** against the published order. Scores come
down slightly — the stricter `gpt-oss-120b` seat outweighs the more lenient DeepSeek —
and the ordering is essentially preserved. `gpt-5-mini` is the largest mover, 5th → 7th.

## For the two-row table

`generator_0731/` supports carrying both `DeepSeek-V4-Flash` and
`DeepSeek-V4-Flash-0731` as separate rows, each judged by the same three seats. On the
evidence above the two rows will be statistically indistinguishable (p = 0.73), which is
itself the result worth reporting: the retired checkpoint's row is not an artefact.

---

# Judge reproducibility (`judge_reproducibility.py`)

`Qwen3.6-35B` was not touched by the swap, so re-running it a month later measures how
much of a GPBam score is reproducible and how much is judge non-determinism. Same model,
same prompts, 2,429 essays, 30 models.

| level | result |
|---|---|
| **essay** | mean Δ +0.19 (sd 8.40) · mean \|Δ\| **5.69** · **49.4%** identical · **94.2%** within one 10-point step · r = **0.886** |
| **model** (81 essays) | mean Δ +0.18 ± 0.15 (t = 1.21, p = 0.24) · mean \|Δ\| **0.66** · max **1.85** · r = **0.9986** · τ-b = 0.985 |
| **rank** | 6/30 models move at all · **0/30 move ≥ 2 places** · top-5 set identical |

A single judgement is barely better than a coin flip on the 10-point grid — only half
reproduce exactly — yet averaging 81 cases turns r = 0.886 into r = 0.9986 and leaves the
leaderboard immovable. This is the empirical warrant for reporting model means from a
stochastic judge, and it is the control that makes the DeepSeek checkpoint effect
(+3.85 net) attributable rather than merely observed.

## How many cases does a stable score need?

Bootstrap over cases (400 draws each, no API calls): mean absolute test-retest gap in a
model's score when only *k* cases are used.

| cases | mean \|Δ\| | p95 |
|---|---|---|
| 5 | 2.82 | 3.53 |
| 10 | 2.02 | 2.57 |
| 20 | 1.42 | 1.78 |
| 40 | 0.99 | 1.23 |
| 60 | 0.78 | 0.95 |
| **81** | **0.66** | 0.66 |

Close to the 1/√k you would hope for. GPBam's 81 cases buy sub-point reproducibility;
halving to 40 costs about 50% more error. Useful for anyone reusing the benchmark at
reduced cost.

**Caveat.** The re-run confounds sampling non-determinism with any serving-side change in
that month (vLLM version, batching, quantisation). That is arguably the right quantity —
it is what a re-user actually experiences — but it is *practical* reproducibility, not
pure sampling noise, and should be described as such.

# Inter-judge agreement — moved

Lives in `experiments/analysis/judge_agreement_zb/` (Zubaer), which reads
`main_essay_table.csv` and the seat files from here rather than copying them.
It replaces the draft's `Table 15` and `Table 16`, both of which described the
pre-swap panel — `gpt-5-nano`, 26 models, ~2,100 essays.

# The DeepSeek-V4-Flash weight swap, and the re-run it forces

NHR@FAU's changelog:

> **2026-08-01**: `deepseek-ai/DeepSeek-V4-Flash` has been updated to the new
> `deepseek-ai/DeepSeek-V4-Flash-0731` weights with speculative DSpark decoding enabled.

The published no-RAG run predates that. Everything here follows from it.

## 1. The published DeepSeek essays are from the retired checkpoint

Three independent lines of evidence, all from git:

- `git log -S'deepseek-ai/DeepSeek-V4-Flash'` on the result CSV returns exactly one
  commit, **`db759642`, 2026-07-12** — the rows entered the file then and their count
  never changed again.
- The blob only ever grew (618 MB on 07-12 → 910 MB on 08-03, unchanged since), which
  is the signature of models being *appended*, not regenerated.
- The DeepSeek rows are **byte-identical** between `0a3baf52` (2026-07-24, pre-swap)
  and `HEAD`: 81 matching lines, fingerprint `431c7e85a339177e` in both.

So those 81 essays were generated on the pre-0731 weights and never touched again.
**They are not reproducible** — the weights are gone from the serving.

Corroborating measurement: re-running case 0 today at the *identical* setting
(`effort=high`) gives a materially different generation.

| | published (07-12) | today (0731 + DSpark) | ratio |
|---|---|---|---|
| time | 103.0 s | 409.4 s | 4.0× |
| completion tokens | 5,892 | 9,978 | 1.69× |
| answer chars | 19,261 | 16,635 | 0.86× |

The 4× is gateway contention; the token and length changes are the model.

## 2. Consequence for the Artificial Analysis figure

`plot_aa_index_vs_essay.py` maps this row to *"DeepSeek V4 Flash 0731 (Reasoning,
Max Effort)" = 52*, justified in its docstring by "FAU serves the 0731 checkpoint".
It did not in July. The docstring already names the older checkpoint's high-effort
entry as **39**, so that point's x is overstated by ~13 index points. Its own caveat —
"this one point's x plausibly too high for how we ran it" — is confirmed, but the
cause is the **checkpoint**, not the effort knob.

## 3. `max` effort cannot be run on this gateway at all

AA scores the 0731 checkpoint only at max effort, so the obvious fix is to generate at
`max`. It is impossible here:

```
case 0  effort=high   409.4 s  compl=9978  answer=16,635 chars  finish=stop
case 0  effort=max    FAILED after 1801 s: 502 Proxy Error
```

The client timeout was 3600 s; the failure came at 1801 s from the **proxy**, i.e. a
hard ~30-minute server-side cap. An earlier 81-essay attempt at `max` sat for an hour
without completing a single item — it was not slow, it was hitting that ceiling and
retrying. Full-length Gutachten at `max` cannot complete through NHR@FAU.

**Therefore the defensible AA match is the 0731 _high_-effort entry**, and the
re-generation here uses `high` — which also keeps the effort knob identical to every
other model in the table.

## 4. `reasoning_effort` is honoured, and the ladder differs per model

Sending a deliberately invalid value returns each server's permitted literals
(`scripts/effort_literals.py`; rejected requests generate no tokens):

| model (NHR@FAU) | permitted `reasoning_effort` |
|---|---|
| `deepseek-ai/DeepSeek-V4-Flash` | none, minimal, low, medium, high, **xhigh, max** |
| `RedHatAI/gemma-4-31B-it-FP8-block` | none, minimal, low, medium, high, **xhigh, max** |
| `google/gemma-4-E4B-it` | none, minimal, low, medium, high, **xhigh, max** |
| `MiniMaxAI/MiniMax-M3-MXFP8` | none, minimal, low, medium, high, **xhigh, max** |
| `Qwen/Qwen3.6-35B-A3B-FP8` | none, low, medium, high |
| `gpt-oss-120b` | none, low, medium, high |
| `mistralai/Mistral-Medium-3.5-128B` | none, low, medium, high |
| `moonshotai/Kimi-K2.6` | none, low, medium, high |
| `GaleneAI/Magistral-Small-2509` | none, low, medium, high |
| `ibm-granite/granite-4.1-3b` | none, low, medium, high |
| `Microsoft/Phi-4-mini-instruct` | low, medium, high |
| `RedHatAI/Mistral-Small-3.2-24B` | low, medium, high |

It is not a no-op. Reasoning volume by rung (`scripts/ds_effort_probe.py`,
`scripts/effort_test.py`):

| effort | DeepSeek-V4-Flash reasoning chars | gpt-oss-120b reasoning chars |
|---|---|---|
| low | 5,703 | 230 |
| medium | 6,719 | 1,354 |
| high | 8,482 | 20,265 |
| xhigh | 30,715 | *(not permitted)* |
| max | 21,217 | *(not permitted)* |
| *omitted* | **0** | 2,543 |

Two things worth knowing. **Omitting the parameter is not "high"** — for DeepSeek it
disables reasoning entirely, so `src/llm.py` sending it unconditionally is
load-bearing. And on OpenRouter this probe is uninformative: every model returns the
same `max|xhigh|high|medium|low|minimal|none`, because that is OpenRouter's unified
schema, not the provider's.

Caveat: `xhigh` produced *more* reasoning than `max`. n = 1 per rung and reasoning
length is stochastic, so the ordering at the top is not to be read into.

Reasoning is billed inside `completion_tokens` — `completion_tokens_details.reasoning_tokens`
comes back `None`, but the arithmetic checks out (gpt-oss high: 20,265 reasoning chars
≈ 5.0k tok + 1,990 answer chars ≈ 0.5k = 5.5k, reported 5,494).

## 5. The judge seats straddle the swap — this is the worst of it

`DeepSeek-V4-Flash` is not only a model under test, it is a **judge seat**. And the
arms were judged on opposite sides of 2026-08-01:

| arm | judged | DeepSeek judge weights |
|---|---|---|
| `without_rag` | 2026-07-12 | pre-0731 |
| `with_rag`, `oracle_rag/*` | 2026-08-12 / 08-14 | 0731 |

So `delta = rag − no_rag` for that seat differs by judge *version* as well as by arm.
DeepSeek is the seat carrying the largest positive flat credit (+1.64 at zero gold
recall), which is the term the whole panel/free disagreement rests on — so this is not
a rounding concern.

## 6. What is being re-run

**Judging — all three seats, one window, all free at NHR@FAU.** Each seat re-judges
every no-RAG essay (30 models × 81 cases = 2,430) from the byte-identical `ji2`
`judge_prompt` stored in the published CSV, at `effort=high` to match `src/llm.py`:

| seat | family | role |
|---|---|---|
| `gpt-oss-120b` | GPT | replaces `gpt-5-nano` — free, larger, and nano is deprecated on AA |
| `Qwen/Qwen3.6-35B-A3B-FP8` | Qwen | incumbent, re-run for a single time window |
| `deepseek-ai/DeepSeek-V4-Flash` | DeepSeek | incumbent, re-run on 0731 |

That is also the Qwen/DeepSeek/GPT three-family panel LEXam uses, with the billed seat
removed. Models are processed best-first on the existing panel median
(`model_rank.csv`); one CSV per model, written atomically, so the run is resumable.

**Generation — `deepseek-ai/DeepSeek-V4-Flash` only.** Per NHR@FAU, only this model's
weights changed, so every other FAU-served generation arm remains valid. Re-generated
at `effort=high` via the same `qa.AnswerGenerator` path (no retriever → `QA_USER`,
empty context), output to `zubaers_result/essay_writing/without_rag_0731/`. Nothing
published is overwritten.

## Not affected

The **expert-annotation packet** stays frozen. It validates the judge against a human
on 21 fixed essays; their provenance matters for the leaderboard, not for judge
validation. `item_10` is a pre-swap DeepSeek essay and is still a perfectly good essay
to grade.

## Running

```bash
conda activate plexam
cd experiments/analysis/deepseek_rerun/scripts
python rejudge.py extract                      # one pass over the result CSV -> per-model prompts
SEAT='gpt-oss-120b' SEAT_OUT=out python -u rejudge_seat.py judge
python ds_high_rerun.py                        # the 0731 re-generation
python effort_literals.py                      # permitted efforts per served model
```

`.env` uses `API_KEY_FAU` / `API_KEY_OR` where `endpoints.yaml` expects
`nhr_fau_api` / `openrouter_api`; the scripts alias them, as
`oracle_experiment.api_keys()` does. `load_dotenv()` must be given the repo `.env`
path explicitly — with no argument it resolves relative to the *calling file*.

## 7. One judgement in the no-RAG arm carries no score, and nothing says so

Found 2026-09-13, by an arithmetic check a reviewer could repeat. Recorded, not
fixed: the effect is 0.06 points against a printed SE of 1.07, and correcting it
would ripple through three hand-patched tables to buy nothing.

Judges score 0.0--1.0 in 0.1 steps, so a mean of 81 per-case medians over three
judges must be an integer multiple of `100 * 0.1 / 81 = 0.1234568`. Six of the 32
rows in `main_results_table.tex` are an *odd* multiple of **half** that step:
Gemini-3.7-flash 67.10, DeepSeek-V4-Flash 43.77, Mistral-L 37.84, Gemma-4-31B-I
32.90, DeepSeek-chat-v3 26.36, Soofi-S-Isar 26.11.

**Five of the six are fine.** The score regex
`\[\[(\d+(?:[.,]\d+)?)\]\]` (`scripts/rejudge_seat.py:45`, same in `rejudge.py:36`
and `judge_0731.py:34`) accepts a decimal comma, so a judge writing `[[0,45]]`
yields a legitimate 0.45. Each of those five rows has exactly one case where such
a half-step landed in the *middle* of the three scores and so survived into the
median. Real data, not an error.

**One is a genuine gap.** Gemma-4-31B-I, case index 13, seat `gpt-oss-120b`
(`results/gpt-oss-120b.jsonl:338`): `score=None`, but `error=None`,
`finish_reason='stop'`, 10,980 characters of judgement. The judge wrote its
verdict in prose as `**Gesamtbewertung:** 0,4 / 1,0` instead of the `[[0.4]]`
envelope, the regex found nothing, and the null flowed on. That case's panel
median was taken over two judges (0.35 instead of 0.40). Consequences:

- Gemma's **GPT-oss per-judge cell, 32.62, is a mean over 80 essays, not 81**;
  with the recoverable 0.4 it is 32.72.
- Gemma's **Median 32.90 would become 32.96**, tying Qwen3.5-122b.

One null in 7,776 judgements; every other reported row is 81/81 at 3/3.

### The two latent gaps this exposes

1. `scripts/rebuild_tables.py:133` and `:164` call `median(axis=1)` and `.mean()`
   with pandas' default `skipna=True` and **no coverage assertion**. The only
   assert in that path (`:163`) counts *cases*, not *judgements*, and is armed
   for Opus alone. A parse failure therefore degrades a row silently.
2. `judge_agreement_zb/judge_agreement.py:57-60` already knows this class of
   score -- "15 of 7775 judgements land off that grid (0.35, 0.37, 0.45, 0.55,
   0.75, 0.85, all parser strays)" -- and **snaps** them before computing kappa.
   `rebuild_tables.py` does not. So the agreement analysis and the main table run
   on slightly different score sets. That mismatch, not the 0.06, is the thing
   worth disclosing if it is ever raised.

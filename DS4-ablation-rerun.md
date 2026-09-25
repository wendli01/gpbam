# DS4-ablation-rerun

Run identifier **`DS4-ablation-rerun`** — used for the output directory, the log
names, the `run_id` column of every generated row, and commit messages. Started
2026-09-13; exact times are in `run.json` inside the output directory.

## TL;DR

> **Resolved (2026-09-13 17:10).** Regenerated on DeepInfra at `low` for
> generation *and* the DeepSeek judge (August FAU's effective regime), the DS4
> oracle gain is +5.49 ± 4.32 (ladder +4.13), no-RAG reproduces August, and the
> ablation row takes the other generators' shape (wording withheld −0.19). See
> "DeepSeek at `low`".

> **Mechanism found (2026-09-13 12:23) — read this first.** DeepSeek-V4-Flash's
> reasoning budget depends on the deployment, not on the repo's
> `reasoning_effort='high'`: ~10k completion tokens per essay on NHR@FAU in
> August, ~40k on DeepInfra and on FAU since its restart. The ~40k regime is
> DeepSeek's genuine `high` (its official prompt prefix alone produces it);
> August FAU served `high` without that prefix, effectively as `low`.
> Regenerated on the GWDG academic cloud at the August budget, the
> oracle gain is **+0.37 ± 4.11** (ladder +4.13; DeepInfra rerun +7.72; FAU
> window +9.81), 7.35 ± 3.77 below the DeepInfra rerun case by case. See
> "Reasoning budget".

> **Correction (2026-09-13 ~08:00) — read this first.** The run logs record the
> corpus of every old DS4 arm: `_gold_now` (the old oracle), `_gold_unc`,
> `_gold_cites`, `_gold_shuf`, `_gold_pad` were generated 08-17/18 at **35.0 of
> 51.9 norms per case — the federal corpus**; `_gold_combined` (the ladder's
> 47.90) on 08-19 at 47.4 (combined); all FAU-window and rerun arms at 47.9.
> Commit `dcbae559` says so explicitly. So the draft's DS4 condition cells are
> federal-corpus and **not comparable** to either rerun; the pooled "draft
> +2.12" comparison below compares corpora. The draft's top-10 +6.98 is the
> 09-12 window arm (no old DS4 top-10 was ever produced). The only like-for-like
> old cell is the oracle: 08-19 +4.13, rerun +7.72 ± 4.30 (inside noise), window
> +9.81. "Wording withheld costs marks" holds in 4 of 5 runs (old DS4 −3.21,
> window −4.33, Gemma −5.37, gpt-oss −4.38; rerun −0.80).

> **Final result (2026-09-13 05:16) supersedes the framing below.** The
> OpenRouter rerun reproduces the FAU window on all five arms, not the draft:
> pooled over the five arms it is +8.01 ± 3.26 over no-RAG, the window +9.45 ±
> 3.13, the draft +2.12. The *published* DS4 ablation cells are the outlier, not
> the window. See "Final results".

The DeepSeek-V4-Flash oracle-ablation arms generated on NHR@FAU in the run that
ended with the 2026-09-12 outage ("the FAU window") score **6–9 points above**
every earlier DS4 generation of the same conditions. Everything downstream of
generation has been ruled out, and a Gemma-4 control regenerated today
reproduces its published cell, so the anomaly is DS4 generation in that window.
The FAU DS4 deployment has been down since, so the window cannot be probed.

**Plan:** regenerate the five window arms on OpenRouter pinned to DeepInfra, with
inputs byte-identical to the window arms, judge them on free3, and swap them into
the ablation table **only if** the OpenRouter generator reproduces the original
FAU generator on the no-RAG arm and on the oracle arm. Owner's criterion: *if the
OR DS4 oracle and no-RAG cells match the ones from the original FAU DS4, they are
interchangeable.*

## Symptom

DeepSeek row of the oracle ablation, free3, paired per case against the no-RAG
baseline (43.77):

| cell | draft (original FAU) | FAU window re-run | difference | essays regenerated? |
|---|---|---|---|---|
| oracle (`_gold_combined`) | +4.13 | +9.81 | +5.68 | yes |
| + uncov. named (`_gold_unc`) | +3.64 | +11.30 | +7.66 | yes |
| order shuffled (`_gold_shuf`) | +2.47 | +10.00 | +7.53 | yes |
| + junk norms (`_gold_pad`) | +0.19 | +8.91 (n=55) | +8.72 | yes |
| wording withheld (`_gold_cites`) | +0.19 | +5.83 (n=60) | +5.64 | yes |
| top-10 only (`_gold_top10`) | +6.98 | +6.98 | **0.00** | **no** (re-judged only) |

Every regenerated arm moved by a similar amount. (`_gold_top10` "not moving" is
**not** evidence of anything: the draft's +6.98 was computed from the same stored
scores, so it matches by construction — see Corrections.)

## What was ruled out, and how

| candidate | test | result |
|---|---|---|
| blank essays scored 0 | fixed in `fec264a2`; cites/pad paired on live rows only | explains the earlier ±20 swing, not this |
| federal-only corpus | `ladder_table.py` comment written 2026-09-11 13:27 measured the original `_gold_combined` at 47.4 of 51.9 norms/case; the window arm reads 47.9 | both combined — **ruled out** for the oracle arm |
| DeepSeek judge seat on DeepInfra | per-seat deltas on the window `_gold_combined` vs banked no-RAG: gpt-oss (FAU both sides) **+11.60 ± 3.96**, Qwen +12.59 ± 3.86, DeepSeek/DeepInfra +6.67 ± 3.46 | the gap is full size on an untouched FAU seat; the fallback seat is the conservative one — **ruled out** |
| reference solution in the judge prompt | parsed every stored judge prompt — `_gold_top10` (original era), the 09-12 window arm, today's no-RAG and Gemma passes: the `<referenzantwort>` block is `cases()` verbatim in all of them | the judge saw the same reference in every pass — **ruled out** |
| shared pipeline / re-judging pass | Gemma-4-31B oracle regenerated and judged 2026-09-13 on FAU; Qwen-seat delta **+6.83 ± 3.10** (n=60) vs published free3 +5.37. Same seat pairing gives DS4 +12.59 | Gemma reproduces — **ruled out** |
| anything downstream of generation | ~~`_gold_top10` moved 0.00~~ | **withdrawn**: tautological, same stored scores as the draft |

## What is left

> **Superseded — see "Final results".** Two independent generations on two
> endpoints agree with each other and not with the draft, so the anomaly is the
> original DS4 run, not the window. Everything in this section describing the
> window as anomalous is kept for the record only.

DS4 generations on FAU in the window differ from earlier DS4 generations on FAU:
same weights name, same host, same conditions, same corpus. Candidates are a
redeployment with different sampler defaults after a restart, or a change in
what the deployment served. The CSVs record no sampler settings (FAU returns no
model metadata, so `llm.py` has nothing to store), and FAU DS4 has been returning
`no healthy deployments` since ~13:10 on 2026-09-12.

Consequence: every DS4 number generated in the window is on a different footing
from the DS4 numbers generated before it. The published ones — the no-RAG
baseline (08-17), `_gold_top10`, the ladder's oracle 47.90, the draft's ablation
cells — are mutually consistent.

## Plan

1. **Regenerate** the five window arms (`_gold_combined`, `_gold_unc`,
   `_gold_shuf`, `_gold_pad`, `_gold_cites`) on OpenRouter/DeepInfra into
   `experiments/zubaers_result/essay_writing/oracle_rag/DS4-ablation-rerun/`,
   canonical filenames. Inputs are checked against the window arms first:
   `verify` rebuilds every stored prompt and refuses to generate on any mismatch.
2. **Judge** each arm on free3 — gpt-oss-120b (FAU, free), Qwen3.6-35B-A3B-FP8
   (FAU, free), DeepSeek via DeepInfra (paid). One judge process per arm, seats in
   sequence, so no two writers ever touch the same file.
3. **Accept or reject** interchangeability:
   - **(a) no-RAG** — fresh OR no-RAG vs banked FAU no-RAG on the gpt-oss seat
     (`analysis/norag_regime_check.py`, already running). Pass: shift inside
     2 SEM at n=81.
   - **(b) oracle** — re-run `_gold_combined` free3 vs the ladder's 47.90, i.e.
     paired delta vs +4.13. Pass: difference inside 2 SEM of the paired delta.
   - *(c) optional, not started* — regenerate `_gold_top10` on OR and pair it per
     case against the original-FAU `_gold_top10` on the same seats. The only
     direct per-case test, since it is the one original-FAU oracle arm whose
     essays still exist.
4. **If (a) and (b) pass:** copy the five judged arms into `oracle_rag/ji2/` and
   regenerate `experiments/analysis/out/oracle_ablation.tex`. The oracle cell
   stays the ladder's 47.90 (+4.13), now consistent with the rest of the row.
   **If either fails:** revert the DS4 row to
   `43.77 & +4.13 & -- & -- & -- & -- & +6.98`.

The ablation table pushed in `0fa675f6` currently carries the **window** cells.

## Inputs

Byte-identical to the FAU window arms (`verify` records the per-arm counts in
`run.json`). Flags as `run_ds_ablation_fau.sh` gave them, `--max-passages 120`,
prompt `qa1`, knowledge base `my_knowledge_base_bayern_titled`:

| arm | flags |
|---|---|
| `_gold_combined` | — |
| `_gold_unc` | `--include-uncovered` |
| `_gold_shuf` | `--shuffle-refs` |
| `_gold_pad` | `--shuffle-refs --pad-junk 20 --pad-from rag_titled_combined_rrf_k50/ji2/rag_deepseek-ai_DeepSeek-V4-Flash.csv` |
| `_gold_cites` | `--cites-only --cites-from oracle_…_gold_combined.csv` (stand-in for the deleted `_gold_now.csv`; 60/60 identical citation lists) |

The *original* run's oracle context differs slightly from both the window run and
this re-run: 47.4 → 47.9 supplied norms per case, under one norm per case. The
judge's reference solution is identical across all runs.

## Interim results (2026-09-13 03:47)

- **Gemma control, n=81, free3:** published 38.27 → regenerated 39.14 (+0.87).
  The shared pipeline is stable.
- **(a) no-RAG, gpt-oss judge only, n=60:** fresh OR vs banked FAU **+3.00 ± 4.95**.
  One-judge test; the table baseline 43.77 is the three-judge median.
- **`ls_text` DS4 on DeepInfra:** +2.65 ± 3.77 over no-RAG, inside its six
  siblings' range (−1.36 to +2.04).
- **Rerun `_gold_combined`, two of three judges, paired per case:** rerun − FAU
  window **−3.58 ± 4.14** (gpt-oss), **−1.11 ± 3.68** (Qwen); rerun − baseline
  +8.02 / +11.48. **The OpenRouter oracle agrees with the FAU window, not with the
  draft.** Two independent generations on two endpoints now agree, so if (b)
  fails on the full median, the likelier outlier is the *original* DS4 oracle cell
  (47.90), and "What is left" above has to be revisited rather than the window
  being discarded.
- **(b) oracle, full free3, 2026-09-13 03:54:** rerun 51.48, **+7.72 ± 4.30** over
  no-RAG; FAU window 53.58 (+9.81 ± 3.52); original 47.90 (+4.13). Rule: |7.72 −
  4.13| = 3.59 < 4.30 → **PASS, but non-discriminating**: rerun − window is
  −2.10 ± 3.80, so the rerun is inside noise of *both* regimes, which are 5.68
  apart (about one 2SEM). Per judge the rerun sits just below the window (gpt-oss
  −3.58, Qwen −1.11, DeepSeek/DeepInfra 44.32 vs 44.44). Best reading: a third
  draw from roughly the window's distribution, with 47.90 at its low edge. The
  acceptance rule as written cannot separate "OR = original" from "OR = window";
  only a paired per-case test on the same essays' regime could.
- **(a) no-RAG, gpt-oss judge only, n=81, 2026-09-13 ~04:00:** fresh OR 44.57 vs
  banked FAU 41.54, **+3.02 ± 4.22** → PASS by the rule; one judge only.
- **Within-OR differencing (gpt-oss judge only)** — rerun arms against the fresh
  OR no-RAG instead of the banked one, so baseline vintage drops out:

  | arm | rerun − OR no-RAG | rerun − banked | window − banked | draft (free3) |
  |---|---|---|---|---|
  | oracle | +5.00 ± 4.13 | +8.02 ± 4.71 | +11.60 ± 3.96 | +4.13 |
  | + uncov. named | +8.15 ± 4.56 | +11.17 ± 4.49 | +12.59 ± 3.82 | +3.64 |

  The fresh baseline removes ~3 points from every delta. The oracle then sits
  next to the draft, `unc` still ~4.5 above it: vintage explains part of the gap,
  not all, and single cells at ±4.5 cannot say whether the remainder is real.
  One-judge deltas also run ~2 points above the free3 median (window: gpt-oss
  +11.60 vs free3 +9.81). A free3 within-OR comparison needs the Qwen and
  DeepSeek judges on the fresh no-RAG arm (Qwen free, DeepSeek ~$0.23).

## Final results (2026-09-13 05:16)

All 405 rerun essays generated (0 failures, 0 refusals), all five arms judged on
free3 (gpt-oss and Qwen on NHR@FAU, DeepSeek via DeepInfra). Generation $3.37,
DeepSeek judge seat ~$1.15.

**Against the no-RAG baseline (43.77), free3, paired per case:**

| arm | rerun | 2SEM | FAU window | draft | rerun − window | rerun − draft |
|---|---|---|---|---|---|---|
| oracle | +7.72 | 4.30 | +9.81 | +4.13 | −2.10 ± 3.80 | +3.59 |
| + uncov. named | +9.94 | 3.80 | +11.30 | +3.64 | −1.36 ± 3.40 | +6.30 |
| order shuffled | +8.02 | 4.23 | +10.00 | +2.47 | −1.98 ± 3.60 | +5.55 |
| + junk norms | +7.47 | 3.58 | +8.91 (n=55) | +0.19 | −1.82 ± 4.12 | +7.28 |
| wording withheld | +6.91 | 4.07 | +5.83 (n=60) | +0.19 | +1.75 ± 3.67 | +6.72 |
| **five-arm mean** | **+8.01** | **3.26** | **+9.45 ± 3.13** | **+2.12** | **−1.44 ± 1.52** | **+5.89** |

*The draft column's condition cells are federal-corpus (35.0 norms/case) and its top-10 is the window arm; `rerun − draft` is therefore not a like-for-like comparison except in the oracle row.*

**Within each run, condition − that run's own oracle, free3:**

| condition | rerun | FAU window |
|---|---|---|
| + uncov. named | +2.22 ± 3.90 | +1.48 ± 3.24 |
| order shuffled | +0.31 ± 3.80 | +0.19 ± 3.06 |
| + junk norms | −0.25 ± 4.02 | −1.45 ± 3.46 (n=55) |
| wording withheld | −0.80 ± 3.58 | **−4.33 ± 4.02 (n=60)** |

**Conclusions**

1. **Acceptance.** (a) no-RAG reproduces the original FAU arm (+3.02 ± 4.22,
   gpt-oss judge). (b) passes by the rule arm-by-arm but is non-discriminating
   alone; pooled over five arms it is decisive: the rerun matches the window and
   excludes the draft.
2. ~~The draft's DS4 ablation cells do not reproduce~~ — **withdrawn**: they are
   federal-corpus arms (35.0 norms/case, 08-17/18), so reruns on the combined
   corpus were never expected to match them. The draft's oracle (+4.13, 08-19,
   combined) is inside noise of the rerun (+7.72 ± 4.30).
3. ~~"Withholding the statute wording costs marks" does not replicate~~ —
   **withdrawn**: −3.21 (old DS4, federal), −4.33 (window), −5.37 (Gemma), −4.38
   (gpt-oss) against −0.80 in the rerun; it holds in 4 of 5 runs.
4. The combined-corpus DS4 oracle has now been measured three times: +4.13
   (08-19), +9.81 (09-12 FAU), +7.72 (09-13 OR). That spread is the main open
   question, and it is within about one 2SEM of the rerun either way.

**Open decisions (owner):** which DS4 row to publish (window arms, already in
`0fa675f6`; rerun arms; or their mean), what to do about the ladder's DS4 oracle
cell, and whether the paper's claims about junk and withheld wording can rest on
Gemma and gpt-oss alone, whose condition arms are banked values with no files to
re-check.

## Decision (2026-09-13)

> **Reversed the same evening (17:30).** With the cause found (see "DeepSeek at
> `low`"), DeepSeek is back in the oracle ablation table: the ablation table prints the `low` row with the ladder's oracle cell
> (+4.13, +5.06, +2.22, +4.20, −0.19, top-10 +2.84) and keeps the two
> `high`-effort rows (NHR@FAU 09-12, DeepInfra 09-13) as LaTeX comments; the
> ladder's DeepSeek `ls_text` cell is 42.96 (cited norms 26.82), with the
> `high`-effort cells (`ls_text` 46.42 / 26.38, oracle 51.48 / 43.26 and 53.58 /
> 42.71) as commented rows under each DeepSeek row.

**DeepSeek is excluded from the paper's oracle ablation table**, which carries
Gemma-4-31B and gpt-oss-120B only (both complete, including top-10, and on the
combined corpus throughout).

Rationale, as decided by the author: the two combined-corpus regenerations (09-12
NHR@FAU, 09-13 OpenRouter) agree with each other, and `ls_text` regenerated on
OpenRouter sits where its siblings do, but the oracle gain does not match the
ladder's DeepSeek row (+4.13, 08-19), and none of the checks above identified
why. The author's judgement is that something systematic moved between the
August runs and September. Putting either regeneration into the ablation table
would contradict the ladder; reconciling them would mean regenerating the whole
DeepSeek ladder row -- and, if the shift is systematic, all ladder rows -- which
is not feasible before submission.

This is a scope decision, not a suppressed result. Every DeepSeek arm is in LFS;
`analysis/oracle_ablation_table.py --with-deepseek` prints both DeepSeek rows
(09-12 and `DeepSeek-V4-Flash*`) to an untracked file. Note for the text: on the
combined corpus DeepSeek does not show the collapse under wording withheld
(+5.83, +6.91, both beyond 2 SEM), so a claim about that collapse should be
scoped to the generators in the table.

## Reasoning budget (2026-09-13 12:23)

**Finding.** DeepSeek-V4-Flash reasons in two regimes, and the deployment chose
the regime; every arm records `reasoning_effort='high'`.

| deployment | arm | median completion tokens |
|---|---|---|
| NHR@FAU, August | no-RAG (`ds_v4_flash_0731_high.jsonl`, the shared baseline) | 10.7k |
| DeepInfra | no-RAG, same prompts (`norag_regime_*.csv`) | 44.6k, higher on 81/81 cases |
| NHR@FAU window, 09-12 | oracle arms | 41–48k |
| DeepInfra | DS4-ablation-rerun arms; ladder `ls_text` | 39–46k; 44.7k |

The August oracle essays are gone, so their budget is inferred from the no-RAG
arm of the same deployment weeks. All 81 August no-RAG essays stopped normally,
so it is not truncation.

**Which regime is `high`.** DeepSeek's own encoding
(`encoding/encoding_dsv4.py` in the Hugging Face repo) has three effort levels,
realised purely as a text prefix at the very start of the prompt: `low` (the
default, no prefix), `high` ("Reasoning Effort: Absolute maximum with no
shortcuts permitted. ...") and `max` ("Reasoning Effort: Beyond maximum ...").
Each server translates the OpenAI-style `reasoning_effort` value into one of
these. Probes on no-RAG case 46 (FAU-August 10,749 tokens, DeepInfra `high`
44,109; GWDG is `chat-ai.academiccloud.de`, same checkpoint):

| endpoint | request | completion tokens |
|---|---|---|
| GWDG | no reasoning parameter | no thinking |
| GWDG | `chat_template_kwargs: {thinking: true}` | 6,895 |
| GWDG | `low` / `medium` / `high` | 9,593 / 10,495 / 13,051 |
| GWDG | `high`, empty system message dropped | 11,104 |
| GWDG | `max` / `xhigh` / thinking + `max` | 36,232 / 45,401 / 44,467 |
| GWDG | `low` + DeepSeek's **`high` prefix** as the system message | **42,870** |
| GWDG | `low` + DeepSeek's `max` prefix as the system message | 49,967 |
| DeepInfra | OpenRouter `reasoning.effort` `low` / `medium` | 14,109 / 11,344 |
| DeepInfra | OpenRouter `reasoning.enabled`, no effort | 43,556 |
| DeepInfra | top-level `reasoning_effort: high` only | 35,355 |

Case 47 on GWDG: `low` 10,708, `medium` 9,729, `high` 14,082.

So the ~40k regime **is DeepSeek's `high`**: the official prefix alone produces
it. DeepInfra and FAU since its restart apply it for `high` (DeepInfra also
defaults to it). GWDG today, and NHR@FAU in August, send `high` without the
prefix, effectively as DeepSeek's `low`; only `xhigh`/`max` get a long prefix.
The FAU effort ladder in `analysis/deepseek_rerun/README.md` (08-19: `low`
5.7k, `medium` 6.7k, `high` 8.5k, `xhigh` 30.7k, `max` 21.2k reasoning chars)
has the same shape as GWDG. This is the serving stack's effort mapping, not our
prompts: every arm's prompts are byte-identical, and dropping the empty system
message changes nothing. What FAU changed at its restart cannot be seen from
outside while its DS4 is down.

**Test.** `ds4_ablation_rerun.py --backend gwdg` regenerated `_gold_combined` on
GWDG at `medium`, inputs verified byte-identical (81/81), into
`oracle_rag/DS4-ablation-gwdg/`, and judged it on the same free3 seats
(DeepSeek seat on DeepInfra). 81/81 essays, median 9.8k tokens (fewer than the
rerun on 81/81 cases), answers 17.0k chars median.

| DS4 `_gold_combined` | median tokens | free3 | vs no-RAG (43.77) |
|---|---|---|---|
| GWDG `medium` | 9.8k | 44.14 | **+0.37 ± 4.11** |
| ladder, NHR@FAU 08-19 | ~10k (inferred) | 47.90 | +4.13 |
| DS4-ablation-rerun, DeepInfra | 40.9k | 51.49 | +7.72 ± 4.30 |
| FAU window, 09-12 | 41.3k | 53.58 | +9.81 |

Paired on free3: GWDG − rerun **−7.35 ± 3.77**; GWDG − window −9.44 ± 3.20;
rerun − window −2.10 ± 3.80. Seats for the GWDG arm: gpt-oss 43.89, Qwen 56.79,
DeepSeek 34.69; on the gpt-oss seat alone GWDG − rerun is −5.68 ± 3.99.

**Reading.** At the August budget the oracle gain is within noise of the
ladder's cell and clearly below the same prompts at DeepSeek's `high` budget. The budget
accounts for the September regenerations' excess; nothing else has to have
moved, and it is DS4-specific, which fits Gemma-4 and gpt-oss reproducing. The
no-RAG arm moves less with the budget (+3.02 ± 4.22, gpt-oss seat) than the
oracle arm does, which suggests the long regime uses the oracle context better:
one arm each, not established.

**Consequences.**
- The ladder's DS4 row is August NHR@FAU, the ~10k regime, **except `ls_text`**
  (46.42, added 2026-09-13 from DeepInfra at 44.7k tokens), which is the
  genuine-`high` regime and not comparable to its row. `rrf50` (44.81) is August: the 09-12
  `rag_titled_combined_rrf_k50` file carries no judge scores, so the cell cannot
  come from it.
- The draft's DS4 ablation row mixes regimes too: August federal conditions
  (~10k) and the window's top-10 (~40k).
- A regime-consistent DS4 ablation row is possible at no generation cost:
  regenerate the conditions on GWDG at `medium` (~15 min generation and ~50 min
  judging per arm; $0.24 per arm for the DeepSeek seat; 81 GWDG requests per
  arm against 200/h and 1000/day). Not done; the decision above stands until the
  author revisits it.

Cost of the test: generation free (GWDG: 81 essays plus 11 probe requests);
DeepSeek seat $0.24. Log: `experiments/logs/DS4-ablation-gwdg.log`.

## DeepSeek at `low`: matching the ladder (2026-09-13 17:10)

Owner's decision after "Reasoning budget": regenerate the DeepInfra DeepSeek arms
at `low`, the regime NHR@FAU served the August ladder at.

**The judge seat has the same two regimes.** NHR@FAU's August DeepSeek *judge*
seat also received `high` without the prefix; DeepInfra honours it. Re-judging
the same essays at `low` scores higher on every arm: combined +5.56 ± 2.85,
top-10 +6.05 ± 2.07, uncovered +4.69 ± 2.28, shuffled +5.19 ± 2.31, junk +6.91 ±
2.24, wording withheld +5.31 ± 2.14. The August regime is therefore `low`
generation **and** a `low` DeepSeek judge. The DeepInfra fallback validation in
`endpoints.yaml` compared against `_gold_top10`'s FAU judgements, which are
post-restart (`high`), so it validated against the wrong regime for August cells.

**What ran** (inputs verified byte-identical to the earlier arms):
- oracle conditions plus `_gold_top10`: `ds4_ablation_rerun.py --backend openrouter-low`
  -> `oracle_rag/DS4-ablation-low/`;
- `ls_text`: `corpus_rag_run.py --effort low` -> `rag_titled_combined_ls_text/DS4-ablation-low/`
  (prompts identical to the earlier DeepInfra arm, 81/81);
- no-RAG check: `norag_regime_check.py --effort=low` ->
  `oracle_rag/ji2/norag_regime_deepseek-ai_DeepSeek-V4-Flash_low.csv`;
- DeepSeek judge at `low`: `analysis/ds4_low/` -> `judge_effort_low_deepseek*.csv`.
  In the arm files the `low` seat is the canonical `score_Judge (deepseek-v4-flash-0731)`
  column, the `high` seat is kept as `..., effort high)`, and
  `deepseek_judge_effort` says `low`.

**Acceptance.** No-RAG at `low`: 10.9k tokens (August 10.7k); gpt-oss seat 41.73
vs the banked 41.54, +0.19 ± 4.06. Oracle at `low` judged `high`: 47.90, +4.14
(ladder 47.90, +4.13); judged `low`: 49.26, +5.49 ± 4.32.

**Cells**, generation and DeepSeek judge at `low`, free3 gain over the August
no-RAG arm (43.77); `*` beyond 2 SEM:

| arm | median tokens | gain | level | judged `high` instead |
|---|---|---|---|---|
| oracle | 10.0k | +5.49 ± 4.32 * | 49.26 | +4.14 |
| top-10 | 11.0k | +2.84 ± 3.80 | 46.60 | +0.86 |
| + uncovered | 10.1k | +5.06 ± 4.32 * | 48.83 | +2.72 |
| shuffled | 10.1k | +2.22 ± 3.17 | 45.99 | +0.80 |
| + junk norms | 9.3k | +4.20 ± 3.72 * | 47.96 | +2.35 |
| wording withheld | 11.4k | −0.19 ± 3.42 | 43.58 | −2.41 |
| `ls_text` (ladder) | 10.9k | −0.80 ± 3.74 | 42.96 | not judged (402) |

`ls_text` cites 26.82% of the gold norms (+0.96 ± 1.71; ladder now 26.38).

**Effort within DeepInfra**, same prompts, DeepSeek judge `high` on both sides:
generation `low` − `high` is −3.58 (combined), −7.22 (uncovered), −7.22
(shuffled), −5.12 (junk), −9.32 (wording withheld); pooled −6.49 ± 1.96.

**Reading.** At the ladder's regime DeepSeek's ablation row has the shape of the
other two generators: wording withheld collapses (−0.19; Gemma-4 +0.56, gpt-oss
+0.00), naming uncovered norms and adding junk norms do not hurt, and shuffling
and top-10 cost a little. The September rows (+6 to +11) were the `high` regime.

**Decisions taken (owner, 17:30).**
- Ladder `ls_text` cell replaced: 42.96 (−0.80 ± 3.74, not bold, `FAE9DF`) and
  cited norms 26.82 (+0.96, not bold, `E4EEF4`), on the ladder's existing scales
  (8.46 and 9.4709; the formatter reproduces the old 46.42 / 26.38 cells exactly).
- DeepSeek back in the oracle ablation table, oracle cell read off the ladder.
- The `high`-effort results stay in both tables as commented-out rows: in the
  ablation table the NHR@FAU 09-12 and DeepInfra 09-13 rows; in the ladder
  `ls_text` 46.42 / 26.38 and the oracle at 51.48 / 43.26 (DeepInfra) and
  53.58 / 42.71 (NHR@FAU).

**No-RAG at `high`, judged like the ladder (2026-09-13 18:30).** The DeepInfra
`high` no-RAG arm (`norag_regime_deepseek-ai_DeepSeek-V4-Flash.csv`, 44.6k tokens)
now has all three seats: gpt-oss and Qwen3.6-FP8 on NHR@FAU, the DeepSeek judge at
`low`. With the DeepSeek seat on DeepInfra (canonical, like every other `low`-judged
arm): free3 **47.84 ± 1.62**, +4.07 ± 3.64 over the August no-RAG arm (43.77); per
seat gpt-oss +3.02 ± 4.22, Qwen +5.19 ± 4.02, DeepSeek (low) 43.33, +5.56 ± 3.75.
The same seat on GWDG (side file) gives 42.72 and free3 46.60: one judge run moves
the three-judge median by about a point here, inside the arm's noise.

**Main table.** `analysis/ds4_low/main_table_high_row.py` writes a commented
`DeepSeek-V4-Flash {\tiny high}` row under the DeepSeek row of
`deepseek_rerun/results/main_results_table{,_stacked}.tex`: Median 47.84 ± 1.62,
Deb. 43.77 (the published panel's `self` effect, +9.94 consensus-percentile points,
applied through the DeepSeek judge's quantile function without refitting; the
script reproduces the August row's 41.05 and Ref. 22.98 before writing), per judge
44.57 / 59.01 / 43.33, Ref. 22.89, AA 52; GP and Cited follow when the GWDG
recitation at DeepSeek's real `high` completes.

**The {\tiny high} GWDG row (integrated 2026-09-15 00:00 by ds4_low/integrate_high_gwdg.py).** Every RAG cell and the no-RAG arm regenerated on GWDG at DeepSeek's real high (effort prefix), judged by gpt-oss-120b and Qwen3.6-35B-A3B (NHR@FAU) and DeepSeek at low (DeepInfra); oracle from the NHR@FAU 09-12 arm. Ladder essay block, against the GWDG no-RAG arm:

| column | level | gain |
|---|---|---|
| no_rag | 48.02 | +0.00 ± 0.00 |
| rrf | 50.74 | +2.72 ± 3.39 |
| rerank | 50.74 | +2.72 ± 3.06 |
| cite_only | 50.00 | +1.98 ± 2.78 |
| cite_rrf | 49.26 | +1.23 ± 3.28 |
| rrf50 | 51.05 | +3.02 ± 3.54 |
| cite_rrf50 | 50.74 | +2.72 ± 3.24 |
| ls_text | 50.25 | +2.22 ± 3.67 |
| cite_ls_rrf50 | 50.99 | +2.96 ± 3.56 |
| oracle | 56.36 | +8.33 ± 3.72 |

tab:corpus {\tiny high}: federal 50.00, combined 50.00 (+0.00 ± 3.30). All cells: `analysis/out/ds4_high_gwdg_cells.csv`. The main table's {\tiny high} row now uses the GWDG no-RAG arm (recitation unchanged).

**The {\tiny high} rows, judged like the ladder (2026-09-13 22:00).** Every
high-effort arm now also has the DeepSeek judge at `low` (DeepInfra, side files
`judge_effort_low_deepseek*.csv`; `analysis/ds4_low/judge_low_arms.py`), and the
commented high rows are differenced against the high-effort no-RAG arm (47.84;
cited norms 25.15). Printed rows are unchanged.

Oracle ablation (`oracle_ablation_table.py`, frozen vmax), gain over no-RAG high:

| row | oracle | + uncov. | shuffled | + junk | wording withheld | top-10 |
|---|---|---|---|---|---|---|
| {\tiny high}: NHR@FAU 09-12 | +8.52 * | +9.69 * | +7.78 * | +8.55 * | +2.83 (n=60) | +5.62 * |
| {\tiny high, DeepInfra}: 09-13 rerun | +5.74 * | +7.35 * | +6.48 * | +5.62 * | +4.32 * | -- |

Ladder (`analysis/ds4_low/ladder_high_rows.py`, existing scales): {\tiny high}
oracle 56.36 (+8.52); {\tiny high, DeepInfra} `ls_text` 48.89 (+1.05 ± 2.71) and
oracle 53.58 (+5.74); cited norms `ls_text` 26.38, oracle 42.71 / 43.26.

**Article Recitation at `high` (GWDG, 2026-09-13 22:20).** `deepseek_rerun/scripts/run_model.py`
(`RUN_ENDPOINT=gwdg`, output `article_recitation/DS4-effort/`), same 200 items and
prompts as `recitation_0731.csv` (NHR@FAU, August), scored with the same ROUGE-L:

| run | items | n | new | August | paired | tokens med (new / Aug) |
|---|---|---|---|---|---|---|
| GWDG, DeepSeek real high (prefix) | all | 200 | 43.53 | 40.75 | +2.78 ± 2.42 | 2562 / 2034 |
| GWDG, DeepSeek real high (prefix) | GPBam Laws | 100 | 45.58 | 42.45 | +3.14 ± 3.72 | 2987 / 1938 |
| GWDG, DeepSeek real high (prefix) | Most cited Laws | 100 | 41.48 | 39.06 | +2.42 ± 3.11 | 2492 / 2054 |
| GWDG, August request (high, no prefix) | all | 200 | 41.83 | 40.75 | +1.08 ± 2.23 | 1916 / 2034 |
| GWDG, August request (high, no prefix) | GPBam Laws | 100 | 43.41 | 42.45 | +0.96 ± 3.79 | 1968 / 1938 |
| GWDG, August request (high, no prefix) | Most cited Laws | 100 | 40.26 | 39.06 | +1.19 ± 2.38 | 1778 / 2054 |

The August-request replica (`reasoning_effort=high`, which GWDG, like August FAU,
sends without DeepSeek's prefix) reproduces August within noise (+1.08 ± 2.23), so
GWDG generation stands in for FAU here. DeepSeek's real `high` (`RUN_EFFORT=low
RUN_PREFIX=high`) recites +2.78 ± 2.42 better, on only ~25% more tokens: a short
task leaves the effort prefix little room. The main table's {\tiny high} row carries
GP 45.58 and Cited 41.48.

**GWDG as the DeepSeek judge (validated 2026-09-13 17:50).** With OpenRouter out of
credit, `analysis/ds4_low/judge_low_gwdg.py` re-judged `_gold_combined` (low arm)
with GWDG's `deepseek-v4-flash-0731` at `low`: mean 42.78 vs DeepInfra `low`
42.84, paired −0.06 ± 2.97, correlation 0.64 (DeepInfra `low` vs `high`: 0.64),
free3 48.83 vs 49.26. GWDG can take the DeepSeek judge seat at `low`
(`judge_gwdg_low_deepseek_gold_combined.csv`). GWDG *generation* is not validated
at `low`: its `medium` oracle arm sat 3.77 ± 3.50 below DeepInfra `low`.

**Other `high`-regime DeepSeek judge scores:** every DeepInfra-judged arm,
including the Gemma-4 oracle control of 2026-09-13. FAU-judged arms after FAU's
restart are `high` too; the restart date is not known.

**Cost:** generation $1.27 (oracle arms), $0.18 (`ls_text`), $0.17 (no-RAG);
DeepSeek judge `high` $1.40, `low` ~$1.00. OpenRouter credit ran out at ~16:46
(402 on the `ls_text` `high` judge pass); further DeepInfra runs need a top-up.

## Cost and throughput

Measured on DeepInfra: **$0.00812 per no-RAG essay** (44.3k completion tokens at
high reasoning effort); oracle prompts add roughly $0.001 of input, so ~$0.009.
DeepSeek judge seat **$0.00283 per call**.

| item | count | cost |
|---|---|---|
| generation | 405 | ~$3.70 |
| DeepSeek judge seat | ≤405 | ~$1.15 |
| gpt-oss + Qwen seats | ≤810 | free |
| | | **~$4.85** |

Wall clock at concurrency 40: roughly 4–6 h; the longest oracle prompts take up
to ~40 min per essay. Generation checkpoints every essay, so a restart resumes.

## Commands

```bash
cd experiments
PYTHONPATH=analysis python -u analysis/ds4_ablation_rerun.py verify
PYTHONPATH=analysis python -u analysis/ds4_ablation_rerun.py generate --concurrency 40
PYTHONPATH=analysis python -u analysis/ds4_ablation_rerun.py judge _gold_combined --wait
PYTHONPATH=analysis python -u analysis/ds4_ablation_rerun.py status
```

Logs: `experiments/logs/DS4-ablation-rerun_generate.log` and
`experiments/logs/DS4-ablation-rerun_judge_<arm>.log`.

## Running alongside

| job | purpose | log |
|---|---|---|
| `norag_regime_check.py` | acceptance criterion (a) | `logs/norag_regime_check.log` |
| Gemma oracle control + judge chain | pipeline control (already reproduces at n=60) | `logs/gemma_oracle_control.log`, `logs/gemma_oracle_judge.log` |
| `ls_text` DS4 on DeepInfra + judge chain | the ladder's missing cell; "is OR DS4 unreasonably good" | `logs/ls_text_deepseek.log`, `logs/ls_text_judge_chain.log` |

## Open

- ~~The ladder's DS4 `rrf50` arm falls inside the window.~~ Resolved: the 09-12
  file has no judge scores, so the ladder cell (44.81) is the August arm.
- ~~"FAU changed the deployment" or "DS4 at high effort is this volatile".~~
  Resolved as a change in reasoning budget; see "Reasoning budget". What changed
  inside FAU's serving stack stays open while its DS4 is down.

## Corrections to earlier claims

- README section "The DeepSeek ablation row on the combined corpus" and commits
  `54ee3747` / `0fa675f6` attributed the gap to a federal→combined corpus change.
  That is wrong for the oracle arm (47.4 vs 47.9 norms/case) and unnecessary for
  the conditions, which all moved by a similar amount. Corrected in the README.
- The judge-seat and reference-text explanations offered during the
  investigation are ruled out as well (table above).
- "De-hyphenation landed in the judged arms and was reverted in the cached
  file" was a **parsing error**: the extraction regex matched the
  `<referenzantwort>` mention inside the judge instruction, and the CSV repr was
  read without unescaping, producing a 60k-char "reference" with no visible wrap
  hyphens. Parsed properly, every stored judge prompt contains `cases()` verbatim.
  The 12-case A/B built on that premise tested nothing and is void.
- `_gold_top10` "moving 0.00" is not a reproduction: the draft's +6.98 was
  computed from the same stored scores.

# GPBam: German Public Law Bar Exam

GPBam evaluates LLMs on 81 German public-law exam cases. Models write full legal essays, optionally with retrieved statutes, and a judge ensemble compares each essay with the reference solution. Every run also saves the individual judgments, prompts, token usage, latency, and cost.

**Read [SUBMISSION.md](SUBMISSION.md) first.** It lists what this branch withholds (the cases, the reference solutions, and the generated essays) and what that costs in reproducibility. This file documents the code as it was run; some entry points below need the withheld inputs and cannot execute here.

## The score we report

> **The `score` column in the result CSV is never reported.** Nothing in the reporting pipeline reads it. The reported Score is recomputed from the per-judge `score_Judge (<name>)` columns by [src/notebook_eval/essay_evaluation.py](src/notebook_eval/essay_evaluation.py):
>
> **per essay:** median across the judges (NaN-skipping) × 100 → **per model:** mean over its 81 essays → **± bootstrap SE**.
>
> The CSV `score` column is the ensemble's conservative **minimum** (`scoring.JudgeEnsemble`, `aggregation=np.min`, 0–1 scale, failed judges imputed as 1.0). It is a legacy run-time artifact, kept for diagnostics only, and it is **much lower** than the reported median — e.g. mean 0.161 vs 0.259 on the no-RAG set.

The two paths also treat a failed judge oppositely: the CSV `score` column replaces the NaN with **1.0** before aggregating (`np.nan_to_num(..., nan=1.0)` in `_aggregate`), so a failed judge counts as a perfect score, while `make_summary` skips it (`.median(axis=1)`, skipna).

That one summary DataFrame feeds both [src/notebook_eval/essay_correlation.py](src/notebook_eval/essay_correlation.py) (coefficients) and [src/notebook_eval/essay_plots.py](src/notebook_eval/essay_plots.py) (figures), so tables and figures always report the same numbers.

## What runs here, and what does not

Experiments are driven from notebooks in `experiments/`, each defining one entry-point function. Run cells from that directory; paths are relative to it. Pass `require_confirmation=False` to skip the interactive `yes` gate while iterating.

```python
df = ew_norag(judge_instruction_name="ji2", require_confirmation=False)
# -> ./zubaers_result/essay_writing/without_rag/ji2/no_rag_ji2_result.csv

df = ew_withrag(judge_instruction_name="ji2", require_confirmation=False)
# -> ./zubaers_result/essay_writing/with_rag/ji2/with_rag_ji2_result.csv
```

Results are written after each model, so an interrupted run resumes from the CSV; reruns skip completed models unless `allow_overwrite_existing_models=True`. `ew_withrag` builds the knowledge base when none exists and otherwise reuses it. `ew_rejudge` applies a new judge prompt or panel to already-generated answers without regenerating them — this is how `ji1`/`ji2`/`ji3` were applied to the same essays, and how a judge missing for some model was filled in. It writes a **separate** CSV; to patch only specific judge columns into an existing file, copy those `*_Judge (<name>)` columns back aligned on `index`. Each model's answers must cover every case exactly once, aligned by `index`, and the function enforces that before judging.

The notebooks are published here **with all outputs cleared**, because their outputs render case text, reference solutions and generated essays. They are the interface as it was run, but they cannot execute on this branch: every one of them reads the withheld `data/gpbam.json`. `experiments/dataset.ipynb` is additionally **redacted** — its cell 21 held one complete case, its reference solution and its rubric as a one-shot example for rubric generation; the three literals are replaced by placeholders and the prompt scaffold around them is kept verbatim. See [SUBMISSION.md](SUBMISSION.md).

What the notebooks call is published in full: `evaluate.evaluate_model` and `evaluate.rejudge_model` do the generation-plus-judging, `qa.AnswerGenerator` / `ReWriter` / `HyDE` the generation and query rewriting, `rag.py` the retrieval, `scoring.py` the judges and citation overlap. Every analysis script under `experiments/analysis/` runs as-is against the published result CSVs.

The prompts are published in full in [src/prompts.py](src/prompts.py), with no case data:

- `QA_SYSTEM`, `QA_USER` — essay generation without retrieval.
- `QA_USER_RAG` — the same task with retrieved statutes. The two differ by more than the `<context>` block: `QA_USER_RAG` drops three sentences that `QA_USER` has (among them *"Zitiere ausschließlich aus den von dir identifizierten Erwägungen"*) and adds a warning that the retrieved selection is incomplete and may be irrelevant. The conditions therefore differ in framing as well as in statutory context, which the paper flags as a confound for their comparison; the exact texts are the record of it.
- `JUDGE_SYSTEM`, `JUDGE_INSTRUCTION_V1`/`V2`/`V3` (keyed `ji1`/`ji2`/`ji3` in `JUDGE_INSTRUCTIONS_BY_NAME`), and `build_judge_user`, which wraps the selected instruction with the case, reference answer, and model answer. The paper uses `ji2`.
- `AR_SYSTEM`, `AR_USER` — the Article Recitation task.

## Setup

Use Python 3.11. The full GPU setup is documented in [environment_setup_guideline.txt](environment_setup_guideline.txt); the short version is:

```bash
conda env create -f gpbam.yml -n gpabam   # `gpbam.yml` declares `name: plexam`; -n overrides it
conda activate gpabam
```

Create `.env` in the repository root and add a key for every endpoint you use. OpenRouter is the default:

```dotenv
openrouter_api=...
nhr_fau_api=...
innkube_api=...
OPENAI_API_KEY=...
```

The authoritative endpoint names, URLs, and environment-variable names are in [endpoints.yaml](endpoints.yaml). See [RUNNING_EXPERIMENTS.md](RUNNING_EXPERIMENTS.md) for more endpoint examples and local-vLLM setup; if documentation and configuration differ, `endpoints.yaml` is the source of truth.

Models live in [src/config.py](src/config.py): `MODELS_DEV` generate essays, `JUDGE_MODELS_DEV` is the three-judge panel (`openai/gpt-5-nano`, `Qwen/Qwen3.6-35B-A3B-FP8`, `deepseek-ai/DeepSeek-V4-Flash`). These two tuples are the only names the module exports; the other lists sit in a `'''...'''` archive block and are a string, not importable. Judge names must be unique — the last path segment becomes the CSV column names. Models absent from `endpoints.yaml` use the `default_endpoint` (OpenRouter); to route one elsewhere, add an exact match:

```yaml
models:
  "project/model-name":
    endpoint: nhr_fau
    model_id: "provider/model-name"
```

> **`deepseek-ai/DeepSeek-V4-Flash` is a dead name.** The live NHR@FAU deployment
> is **`deepseek-ai/DeepSeek-V4-Flash-0731`**; the unsuffixed key survives in
> `endpoints.yaml` only as an alias, so stored CSVs carrying the old name keep
> resolving. FAU swapped the April weights for the 0731 checkpoint on 2026-08-01
> *without renaming the model*, so **the model name in a CSV does not identify the
> checkpoint, the date does. Every DeepSeek result in the paper is 0731 weights**,
> and data from before 2026-08-01 is on the April weights and must not be
> differenced against anything after it.

## Which knowledge base is which

The names mislead: `bayern` does **not** mean "Bavarian only", it means
"federal **plus** Bavarian" — i.e. the combined corpus. `titled` means the
rebuild that puts the citation line into the embedded text
(`rebuild_kb_with_titles.py`), which is what every current arm uses.

| directory | rows | scope | embedded text | use |
|---|---|---|---|---|
| `my_knowledge_base` | 106,354 | federal only | body only, bge-large | legacy; source for the titled federal rebuild |
| `my_knowledge_base_titled` | 106,354 | federal only | citation line + body, jina-v5 | **canonical federal** (`--corpora federal`) |
| `my_knowledge_base_bayern` | 113,362 | **combined** | body only, bge-large | legacy; source for the titled combined rebuild |
| `my_knowledge_base_bayern_titled` | 113,362 | **combined** | citation line + body, jina-v5 | **canonical combined** — the default everywhere |

`my_knowledge_base_bayern_titled` is what `oracle_experiment.KB` points at and
what `corpus_rag_run.KBS['combined']` resolves to; `KBS['federal']` is
`my_knowledge_base_titled`. The paper's 113,362 provisions are the combined
store: 106,354 federal from `gesetze-im-internet` at revision `67e0567`, plus
7,008 from 720 Bavarian statutes crawled from BAYERN.RECHT on 2026-08-31.

The vector stores themselves are too large to publish. Their text content is committed as `experiments/data/kb_corpus.parquet` (52 MB, no vectors, no git-LFS) and the Bavarian crawl separately as `analysis/out/landesrecht_corpus.json`, so a store can be rebuilt without re-crawling a site that changes underneath the paper:

```bash
cd experiments
python analysis/rebuild_kb_with_titles.py --only federal   # or --only bayern
#   --fresh        discard the target and start over; required if it is damaged,
#                  because resume reads the target to find its restart point
#   --char-budget  lower it to share the GPU with another job
```

## Summarise results

[src/notebook_eval/essay_evaluation.py](src/notebook_eval/essay_evaluation.py) builds the per-model LaTeX tables from a result CSV loaded with pandas:

```python
from src.notebook_eval import essay_evaluation

# MAIN table: Score + Legal Ref. Sim.
main = essay_evaluation.make_summary(model_study_df)

# EXTENDED table: + per-judge scores, tokens, cost
extended = essay_evaluation.make_summary(
    model_study_df,
    show_judge_scores=True,
    show_answer_tokens=True, show_judge_tokens=True,
    show_gen_cost=True, show_judge_cost=True,
    token_divisor=1000, token_digits=1,
)
```

- **Score** is the row-wise **median** of the judges' `score_Judge (…)` columns (×100), summarised as **mean ± bootstrap SE**. `legal_ref_sim` gets the same treatment.
- **Tokens** are descriptive — **mean ± std**. **Cost** is total spend per model.
- The two **Qwen judge variants** (`Qwen3.6-35B-A3B-FP8` and `qwen3.6-35b-a3b`, run as API fallbacks for each other and mutually exclusive per row) are **coalesced into one logical judge**, so every model uses a consistent 3-judge panel.
- Columns are toggled with the `show_*` knobs; metrics with no data render blank rather than `NaN`.

[src/notebook_eval/article_recitation_evaluation.py](src/notebook_eval/article_recitation_evaluation.py) does the same for **Article Recitation**, where every model is evaluated on two datasets (`GPBam Laws`, `Most cited Laws`). `make_article_summary` returns the per-`(model, dataset)` summary and, with `print_latex=True`, prints a `\multirow` table that matches it cell for cell. Each cell aggregates that model's ~100 per-article ROUGE-L scores: **Score** is the mean ×100 with a bootstrap SE, and **Score Std** the article-to-article sample std, reported alongside because the per-article spread is roughly the size of the mean.

### Correlations

[essay_correlation.py](src/notebook_eval/essay_correlation.py) takes the `make_summary` DataFrame and reports **Spearman rank correlations across models** — one point per model — between `Score` and every other numeric column, plus Article Recitation when `recitation_csv=` is given:

```python
summary = essay_evaluation.make_summary(model_study_df, print_latex=False)
res = essay_correlation.correlate(
    summary,
    recitation_csv="zubaers_result/article_recitation/article_recitation.csv",
    exclude={"mistral-small-3.1-24b-instruct", "EuroLLM-22B-Instruct-2512",
             "Phi-4-mini-instruct", "gemma-4-31B-it-FP8-block"},
)
```

Only the point estimate is used; the `$_{\pm SE}$` inside a cell is the uncertainty *of* that mean, and parsing the rounded 1 dp display value matches the full-precision correlation to ≤0.001. `Qwen3.6-35B-A3B-FP8` (no-RAG) and `qwen3.6-35b-a3b` (RAG / recitation) are aliased to one model, or the recitation join silently drops it. With the four paper exclusions above the no-RAG summary has `n=26`, falling to `n=18` for Gen. Cost where providers reported none. Each coefficient carries a 95% bootstrap CI over resampled model pairs.

[judge_correlation.py](src/notebook_eval/judge_correlation.py) compares the individual judges at two levels: `level="model"` averages each judge's case scores per model and measures agreement in ranking; `level="essay"` correlates over individual essays with generation-model cluster bootstrap CIs. Pass any subset of `["spearman", "pearson", "qwk"]`. QWK uses a fixed 0–100 ordinal grid, so it penalises systematic scoring-level differences that correlation alone does not. Missing-score counts differ between runs — no-RAG loses 3–7 essays per judge pair, law-RAG loses 58 (the gpt-5-nano judge) — so quote the pairwise `n` from the table rather than assuming 2106. See [documentation/judge_analysis.md](documentation/judge_analysis.md) for the verified counts, the model- versus essay-level estimands, and the bootstrap and p-value procedures.

### Figures

[essay_plots.py](src/notebook_eval/essay_plots.py) draws the same relationships as scatter plots from the **same summary DataFrame** the tables use, so a figure cannot drift from its table. Four standalone scripts build the paper's figures; run each from `experiments/`:

```bash
python analysis/plot_cost_performance.py       # -> analysis/out/cost_performance_norag.{pdf,png,csv}
python analysis/plot_recitation_vs_essay.py    # -> analysis/out/recitation_vs_essay_norag.{pdf,png,csv}
python analysis/plot_aa_index_vs_essay.py      # cached catalogue; --refresh re-fetches from the API
```

The `AA_INDEX` table inside `plot_aa_index_vs_essay.py` was read off the public Artificial Analysis model pages on 2026-08-17 (index v4.1.1) and is the default source, so that figure rebuilds with no account and no network. `--refresh` re-reads the same models through the free data API and needs a key in `.env` as `artificialanalysis_api=...`.

> **Run them through `deepseek_rerun/scripts/rebuild_figures.py`, not bare.** Bare,
> `plot_recitation_vs_essay.py` reads the PUBLISHED no-RAG CSV (25 models, no Opus)
> instead of the re-judged drop-in that `GPBAM_RESULTS` points at, and silently
> writes a smaller figure.

Shared behaviour, in [figure_common.py](experiments/analysis/figure_common.py): one point per model over the same paired models the tables report, colour by vendor family, and every point labelled by a solver that treats markers and legends as obstacles and draws a leader line when a name cannot sit beside its point. A model with fewer than 81 essays is placed by its per-essay mean × 81 and drawn **hollow**, with the shortfall named in the figure, so a short row can never be mixed in silently.

**Cost imputation** (price/performance figure). `total_cost` is provider-reported and exists only for the OpenRouter-served models; the self-hosted endpoints (NHR@FAU, InnKube, local vLLM) report nothing or a literal $0. Those are priced at `prompt_tokens × price_in + completion_tokens × price_out` using OpenRouter list prices (`/api/v1/models`, retrieved 2026-08-17) hard-coded in the script's `PRICES` table, and carry a `*` after the model name. Billing status is decided per model, not per row: a model counts as billed if any of its 81 rows has a positive cost. `Magistral-Small`, `granite-4.1-3b` and `steuerllm` have no catalogue entry and are priced by the nearest listed sibling, named in the CSV's `price_basis` column. `DeepSeek-V4-Flash` is priced from `deepseek/deepseek-v4-flash-0731` ($0.14/$0.28 per 1M) rather than the undated slug ($0.083/$0.165), because NHR@FAU serves the 0731 checkpoint.

The recitation figure is one **dumbbell** per model: essay Score on the vertical axis, and the model's two Article Recitation scores as the ends of a horizontal segment, so the segment length is the gap between the two knowledge sets. The x axis is logarithmic because two thirds of the models recite below 20% while `mistral-large` sits near 50%; Spearman ρ is rank-based and unaffected. Models are matched on the last path segment, lower-cased, with quantisation suffixes stripped; all 26 paired models match, and an unmatched model raises rather than being dropped. ρ with a 10,000-resample percentile bootstrap CI (seed 0) is **0.83 [0.61, 0.94]** on GPBam statutes — the value in `tab:essay_corr_combined` — and **0.77 [0.49, 0.90]** on the most-cited set; the two recitation sets rank models nearly identically (ρ = 0.97), which is why one panel suffices.

## Repository overview

| Path | Purpose |
|---|---|
| `src/config.py` | The two active model lists (`MODELS_DEV`, `JUDGE_MODELS_DEV`) plus `get_model_endpoint`, which resolves a model to its endpoint via `endpoints.yaml`. |
| `src/prompts.py` | Generation and judge prompts (`ji1`/`ji2`/`ji3`) and the `build_judge_user` prompt builder. |
| `src/llm.py` | Concurrent OpenAI-compatible API client, retries, usage, and cost metadata. |
| `src/qa.py` | Essay generation, response cleanup, retrieval context, query rewriting, and HyDE. |
| `src/rag.py` | LanceDB statute ingestion and vector/hybrid retrieval. |
| `src/scoring.py` | Individual judges, score parsing, conservative (min) ensemble aggregation, and citation overlap. |
| `src/evaluate.py` | Generation-plus-judging and rejudging result assembly. |
| `src/data.py` | Exam PDF loading, law-corpus download/parsing, and legacy graph helpers. |
| `src/notebook_eval/essay_evaluation.py` | Per-model summary + main/extended LaTeX tables (median-of-judges, bootstrap SE, Qwen coalescing). |
| `src/notebook_eval/article_recitation_evaluation.py` | Per-`(model, dataset)` summary + `\multirow` LaTeX table for Article Recitation. |
| `src/notebook_eval/essay_correlation.py` | Spearman/Pearson correlations across models; joins Article Recitation from its CSV. |
| `src/notebook_eval/judge_correlation.py` | Pairwise judge agreement at model and essay level (Spearman, Pearson, QWK). |
| `src/notebook_eval/essay_plots.py` | Three standalone and one combined score-vs-predictor figure from the `make_summary` DataFrame. |
| `src/notebook_eval/article_recitation_plots.py` | Essay Score vs. both Article Recitation datasets, with normalized model matching. |
| `experiments/essay_writing_norag.ipynb` | Generate + judge essays without retrieval (`ew_norag`). Outputs cleared; needs the withheld cases to run. |
| `experiments/essay_writing_lawrag.ipynb` | Generate + judge with statute retrieval (`ew_withrag`); auto-builds the knowledge base. Outputs cleared. |
| `experiments/eassy_writing_rejudge.ipynb` | Re-judge existing answers under a new prompt/ensemble (`ew_rejudge`). Outputs cleared. |
| `experiments/dataset.ipynb` | Builds `data/gpbam.json` from the exam PDFs. Outputs cleared and cell 21 redacted; see above. |
| `experiments/analysis/` | Every table, figure and retrieval experiment the paper reports; `figure_common.py` holds the shared label placer, palette and bootstrap. |
| `experiments/analysis/out/` | Derived tables (`*.tex`), the cell banks behind them (`*_values.csv`), the retrieval battery, the gold set, and the cached query rewrites. |
| `experiments/analysis/ds4_low/` | The DeepSeek-V4-Flash low- and high-effort arms: generation, judging, and table integration. |
| `experiments/zubaers_result/` | Per-essay judge scores, citation overlap, token counts and costs; Article Recitation with the model answers. |

## Judge family bias

All three judges also compete in the generator zoo, and ten further models share a vendor with one of them,
so self-preference propagates into the median Score. `experiments/analysis/judge_family_bias.py` measures it
from the stored per-judge score columns — no new LLM calls — writing
`out/judge_family_bias{,_models}.csv` and the matrix table to
`../literature/paper/judge_score_matrix.tex`, a directory it creates (the paper source
is not on this branch):

```bash
PYTHONPATH=analysis python analysis/judge_family_bias.py   # from experiments/
```

Holding essay quality fixed, a judge gives its own model **+9.6** and its own vendor **+5.5** percentile
points of the consensus scale (permutation p = 0.004 and 0.016); *same family* (same vendor **and** version
series) has one instance in the zoo, `gpt-5-mini` judged by `gpt-5-nano` at +9.5 points, estimated with its
own indicator and kept out of the same-vendor estimate. The tier assignment and the coefficients behind the
debiased column are `.tex` comments, so the table is auditable from the paper source. It does not overturn
the reported leaderboard: the Score is a *median* over three judges, so deltas run from −0.1 to −3.2 points,
though the upper half shifts by up to three places (largest: `gpt-5-mini` 5→8 and `qwen3.5-397b` 2→4 without
retrieval). A blank `\Delta` means no judge was related to that model, which is not the `0.0` of a corrected
model whose median did not move.

## Two rows added after the sweep

Two models were run after the published no-RAG sweep and appear in the main table and the figures.

### `google/gemini-3.7-flash` — a measured row

Generated end to end by `experiments/analysis/deepseek_rerun/scripts/run_model.py`
(`RUN_MODEL=google/gemini-3.7-flash`), the identical code path as every benchmarked model, judged on the
same three NHR@FAU seats, and sent through the 200 article-recitation prompts as well. Nothing is imputed —
OpenRouter billed the run, so its cost is a charge rather than a list price.

    Judge median 67.10 ±1.92   Legal ref. sim. 32.26
    Recitation   72.39 GPBam / 69.83 most-cited      AA 56

### `anthropic/claude-opus-5` — a marked row

Written through the Claude Code agent harness on the byte-identical benchmark prompt, not through the API.
As of 2026-08-22 it covers **all 81 cases**, replacing the `$^\dagger$` placeholder that used to hold its
position, but it is still **not** a benchmarked row:

    Judge median 76.42 ±2.35 (n=81)   Legal ref. sim. 34.25
    Recitation   80.40 GPBam / 89.97 Most cited      AA 63 (upper bound)

Coverage is no longer a reason for the marker — every cell averages the same 81 cases as every other row,
and the recitation cells, `--` until 2026-08-21, are measured on **all 200 prompts of both datasets** — so
the `$^\ddagger$` carries the harness difference alone, the one that does not shrink with n. Read
`experiments/analysis/additional_models/opus5_agent/README.md` before citing the row: the harness cannot run
with an empty system prompt and imposed no `max_tokens` ceiling (these essays are about five times the
benchmarked median length, and length correlates with Score at ρ = 0.71), and the three judge seats disagree
about them by 20 points. Its cost is estimated, the harness recording no usage:
`opus5_agent/scripts/estimate_tokens.py` takes the prompt tokens from the `claude-haiku-4.5` run on the same
prompts, the visible output at the Claude tokeniser's 2.782 chars/token, and haiku's thinking tokens on the
same case — $63.36 for all 81 essays at OpenRouter list. The price stays marked `*` because it is imputed;
the honest path to an unmarked row is `run_model.py` with `RUN_MODEL=anthropic/claude-opus-5`.

### Two rows removed at the same time

`RETIRED` in `experiments/analysis/deepseek_rerun/scripts/rebuild_tables.py` drops two rows at the
presentation layer without deleting data: `mistralai/mistral-small-3.1-24b-instruct`, whose 81 calls all
errored into 25-character stubs that every judge scored 0.0, and `deepseek-ai/DeepSeek-V4-Flash`, the
checkpoint NHR@FAU retired on 2026-08-01 and superseded by the `-0731` re-generation reported in its place.
That DeepSeek row was high-scoring with an imputed near-zero cost, so dropping it moves Gen. Cost
0.662 → 0.773 and the AA index 0.861 → 0.846.

### Rebuilding everything after adding a row

Order matters; each step reads the previous one's output.

```bash
cd experiments/analysis/deepseek_rerun/scripts
python recompute_legal_ref.py     # one extractor over every row, incl. the new ones
python make_rejudged_result.py    # the drop-in CSV every figure reads
python rebuild_figures.py         # 7 figures, both code paths
python rebuild_tables.py ../results
python merged_results_table.py ../results   # joins the two tables above into one
```

`merged_results_table.py` re-reads the two finished tables as artefacts (`main_essay_table.csv`,
`judge_family_bias_models.csv`) and emits `main_results_table.{tex,csv}`, which carries the essay metrics
and the per-judge/debiased columns together; in it `Deb.` is blank wherever no judge was related to the
model, because there the correction is exactly the identity. Adding a further model means an entry in
`EXTRA` (`recompute_legal_ref.py` and `make_rejudged_result.py`), in `MODELS` and, if self-hosted, `PRICES`
(`plot_cost_performance.py`), and in `AA_INDEX`
(`plot_aa_index_vs_essay.py`); `plot_cost_performance.build_table` raises on a model it has never heard of.
The two added rows are leverage points, and including them strengthens every association — Legal Ref. Sim.
goes 0.899 (n=30) → **0.917** (n=32) and the Artificial Analysis index 0.817 (n=27) → **0.852** (n=29).
Opus's Gen. Cost and Answer Tokens are the estimated values, and dropping it from the Cost row alone moves ρ
from 0.769 to 0.741.

### A fix to Legal Ref. Sim. (2026-08-22)

Every number in the `Legal Ref. Sim.` column moved on 2026-08-22, by a mean of +0.12 and at most +1.06.
`RefExtractor.extract()` raised `RefExError` when two reference markers overlap, while
`src.scoring.legal_ref_similarity` keeps only the markers — so the check discarded **every** reference in a
document over one bad marker, including in 3 of the 81 reference solutions (cases 34, 73, 74), which made
those cases undefined for *every* model. The fix stops before `replace_content` — `remove_markers`, then the
two marker extractors, then the same `(book, section)` filter as before — and takes the column from a
77-case statistic to an 80-case one (only case 7 is genuinely undefined). It reshuffles near-ties (10 of the
34 rows change rank) and moves the association ρ 0.928 → 0.917; no judge score is affected, so nothing was
re-judged. Rebuild in the order given above.

### The oracle ablation table, and generation checkpoints (2026-09-11)

`tab:oracle-ablation` decomposes the oracle advantage into the statute **wording**, which a working
retriever could in principle supply, and the **selection** — *which* provisions the case turns on, the
reference solution's issue list, which no retriever has. Generated by:

```bash
cd experiments
PYTHONPATH=analysis python analysis/oracle_ablation_table.py
# -> analysis/out/oracle_ablation.tex
```

Three rules keep it honest. **The oracle column is read off `tab:ladder`, not recomputed** — from
`analysis/out/ladder_values.csv` via `oracle_ablation_table.ladder_oracle()`, so the two tables cannot
disagree by construction, at a cost of a hundredth of a point to the ladder's two-decimal rounding.
**Published cells are banked**, in `analysis/out/oracle_ablation_values.csv` (`row,cond,value,bold`), used
only as a fallback behind a live arm file, and `VMAX` is frozen at 13.0038 so that a new row cannot recolour
them; that value reproduces all twelve published colours exactly (any vmax in [12.9695, 13.0255] does), so
this table, unlike `ladder_table.tex`, can be regenerated faithfully with the bank and the frozen scale in
place — set `VMAX = None` only when rebuilding the whole table at once. **Rows must not mix corpora**:
`oracle_experiment.KB` pointed at the federal-only knowledge base until 2026-08-21, and arms generated
before that carry it, so check an arm against its own stored prompt rather than its filename, with
`judge_recall_slope.context_norms(path)`. The caption is the draft's, verbatim, trimmed there to 400
characters; what the conditions mean is in the module docstring.

#### Regenerating a generator's ablation

```bash
cd experiments
MODEL=RedHatAI/gemma-4-31B-it-FP8-block bash analysis/run_oracle_ablation_model.sh
# on a degraded endpoint, so that a cut-off attempt still advances the arm:
MODEL=... CHUNK=5 LIMIT=45m ATTEMPTS=20 bash analysis/run_oracle_ablation_model.sh
```

`CHUNK` is the point: `oracle_experiment.generate_chunked` writes its `.part` only *between* chunks, so an
attempt shorter than `CHUNK × per-case` banks nothing and the next attempt restarts from zero. `GEN_CHUNK`
is now 5 and `oracle_experiment.py run --chunk N` overrides it.

### A blank answer is no longer scored zero (2026-09-12)

A judge shown an empty answer replies with a well-formed `[[0]]`, indistinguishable downstream from a real
zero — `paper_table.med` uses `skipna=True`, which cannot help — so a generation cut off partway read as a
score collapse rather than as missing data. `Judge.predict` now skips a row whose answer is float `nan`, the
literal string `'nan'` or blank, scoring it `None` → `NaN`; `JudgeEnsemble._aggregate` maps an absent seat to
1.0, and all-NaN rows stay NaN. Before trusting any arm mean, check `d['answer'].isna().sum()`: a cut-off run
leaves a contiguous tail, so the tell is whether the missing `index` values run to 80.

### DeepSeek's reasoning regimes and the GWDG backend (2026-09-13)

The DeepSeek judge seat can also be served through OpenRouter with the provider pinned to DeepInfra
(`endpoints.yaml` has the measured comparison); use `deepseek/deepseek-v4-flash-0731` and never
`deepseek/deepseek-v4-flash`, which OpenRouter names "DeepSeek V4 Flash 0423", the April weights, not to be
differenced against anything after 2026-08-01. That slug is lowercase, so the seat writes
`score_Judge (deepseek-v4-flash-0731)` and `paper_table.DS_OR` is a third spelling in `SEAT_ALIASES[DS]`.

At the same `reasoning_effort='high'`, DeepSeek-V4-Flash spent ~10k completion tokens per essay on NHR@FAU
in August and spends ~40k on DeepInfra and on FAU since its restart. The ~40k regime is DeepSeek's own
`high` (its official prompt prefix produces it on GWDG too); August FAU, like GWDG today, served `high`
without the prefix, effectively as DeepSeek's `low`. **Check `completion_tokens` medians before differencing
two DS4 arms.** At the August budget the DS4 oracle gain is +0.37 ± 4.11 (ladder +4.13, DeepInfra rerun
+7.72); that regime is `low` for generation *and* for the DeepSeek judge seat (DeepSeek judges ~5.6 points
harsher at `high`), so the DeepInfra DeepSeek arms were regenerated and re-judged that way. GWDG counts every
request, even rejected ones, against 30/min, 200/h, 1000/day and 3000/month.

```bash
cd experiments
# --backend gwdg: the GWDG academic cloud (ENDPOINT_AC / API_KEY_AC in .env, effort medium, streamed),
# into oracle_rag/DS4-ablation-gwdg/, with the same prompts and judges
PYTHONPATH=analysis python -u analysis/ds4_ablation_rerun.py --backend gwdg generate --tags _gold_combined --concurrency 10
PYTHONPATH=analysis python -u analysis/ds4_ablation_rerun.py --backend gwdg judge _gold_combined
# the published `low` cells, on DeepInfra
PYTHONPATH=analysis python -u analysis/ds4_ablation_rerun.py --backend openrouter-low generate --concurrency 40
PYTHONPATH=analysis python -u analysis/ds4_ablation_rerun.py --backend openrouter-low judge <arm> --wait [--seats <seat>]
PYTHONPATH=analysis python -u analysis/corpus_rag_run.py generate --pipeline ls_text --top-k 10 --corpora combined \
    --out-dir zubaers_result/essay_writing/rag_titled_combined_ls_text/DS4-ablation-low \
    --models deepseek/deepseek-v4-flash-0731 --store-as deepseek-ai/DeepSeek-V4-Flash --pin-deepinfra --effort low
```

`--backend openrouter-low` adds `_gold_top10` as a sixth condition. Results: oracle +5.49 ± 4.32, wording
withheld −0.19, `ls_text` 42.96; the no-RAG arm reproduces August. These are the DeepSeek cells in both
tables: the ablation table prints the `low` row with the ladder's oracle cell (+4.13, +5.06, +2.22, +4.20,
−0.19, top-10 +2.84), the ladder's DeepSeek `ls_text` cell is 42.96 (cited norms 26.82), and the two
`high`-effort rows (NHR@FAU 09-12, DeepInfra 09-13) are kept as LaTeX comments.
`oracle_ablation_table.py --with-deepseek` prints both DeepSeek rows to an untracked
`analysis/out/oracle_ablation_with_deepseek.tex`. Details: `DS4-ablation-rerun.md`, "DeepSeek at `low`".

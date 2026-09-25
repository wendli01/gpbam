# GPBam — submission artifact

Code and intermediate results for *GPBam: A Retrieval-Aware German Public Law
Exam Benchmark*.  This branch is an **orphan**: it has no parent commit, so the
examination material is not reachable from it at any point in history.

## What is not here, and why

The 81 *Sachverhalte* and their reference solutions are copyright-protected and
in active use in the State Examination Preparation Course in Public Law at the
University of Passau.  They are withheld, and so is everything that embeds them:

| Withheld | Reason |
| --- | --- |
| `experiments/data/01_Sachverhalte/`, `02_Lösungen/` | the cases and solutions themselves |
| `experiments/data/gpbam.json`, `gpbam_w_rubric.json` | the same material as JSON |
| `qa_prompt`, `message` columns | contain the *Sachverhalt* |
| `judge_prompt` columns | contain the reference solution |
| `judgement*` columns and keys | judge prose that restates the solution's *Gliederung*, its governing norms and its outcomes |
| `experiments/expert_annotation/` | packet ships the case PDFs; the human study is not reported in the paper |
| `answer` columns and keys, essay-writing arms | the generated *Gutachten*. A strong model's essay is close enough to a reference solution to substitute for one, and pooling essays for one case recovers part of its *Sachverhalt* |
| `thinking` / reasoning traces | restate the facts and the analysis; §4 strips them before scoring, so nothing depends on them |
| notebook outputs, and `dataset.ipynb` cell 21 | the outputs render case text, solutions and essays; that one cell held a complete case, its solution and its rubric as a one-shot example, in the cell *source* |

Judge **scores** are kept in full.  Every table in the paper is built from the
scores, not from the judgement text, so removing the prose costs no
reproducibility.

Systems can be evaluated against the held-out set on request.

## What is here

- **Code** — `src/`, `experiments/analysis/**/*.py`, the run scripts, and the
  full prompt templates in `src/prompts.py` (generator and judge, no case data).
  `QA_USER` and `QA_USER_RAG` sit side by side there, which is the record of the
  framing difference between the two conditions that §3.2 flags as a confound:
  the retrieval prompt drops three sentences the closed-book one has and adds a
  warning that the retrieved selection is incomplete and may be irrelevant.
- **Notebooks** — the four in `experiments/`, which define the entry points
  (`ew_norag`, `ew_withrag`, `ew_rejudge`) and the dataset build.  Published with
  every output cleared, and `dataset.ipynb` redacted as the table above records.
  They are the interface as it was run; none of them can execute here, because all
  four read the withheld `data/gpbam.json`.
- **Corpora** — `experiments/data/kb_corpus.parquet` (the full combined corpus, 106,354 federal plus 7,008 Bavarian provisions, committed directly — this branch needs no git-LFS),
  `experiments/analysis/out/landesrecht_corpus.json` (the Bavarian provisions again, with their BAYERN.RECHT source URLs), and the cached
  query rewrites `out/rewrites_*.json`.  The rewriter samples, so these *are*
  the experiment: every arm has to see the same rewrites.
- **Retrieval gold set** — `experiments/analysis/out/gold_norms_by_case.json`,
  the norms each reference solution cites (81 cases, 4,202 norms, 51.9 per
  case) as law-book key and section only. Every recall number in the paper is
  computed against it; it carries no solution text.
- **Task definitions** — the Article Recitation sets.  *Most-cited laws* is
  `experiments/data/most_cited_laws.csv` (100 provisions).  *GPBam laws* exists
  in two versions, and **the paper reports the second**:
  `gp_laws.csv` is derived with `refex`, which resolves `§` citations only and
  so contains no GG article at all, although Art. 2 GG is the third-most-cited
  norm in the corpus; `gp_laws_dual.csv` re-derives the same 100 slots with
  `analysis/refs.py`, which reads both citation forms.  73 provisions are shared,
  27 are replaced, and 24 of the entrants are GG articles.  Table 1's
  *Art. recit. — GP* column and the $\rho=0.92$ correlation are computed on the
  **dual** set; `build_gp_laws_dual.py` rebuilds it from the pinned
  `gesetze-im-internet` snapshot.
- **Results** — per-essay judge scores, legal reference similarity, token counts,
  costs, and recitation scores *with* the model answers.  Article Recitation
  answers are kept because a model reciting a public statute discloses nothing
  about a case, and they are what makes the ROUGE-L scores checkable.
  Essay-writing answers are **not** kept; see the table above.
  `article_recitation/article_recitation.csv` is the `refex` run over
  `gp_laws.csv`; `article_recitation/dual/recitation_dual_gpbam_laws.csv` is the
  dual battery the paper reports, and its per-model means reproduce all 32
  *Art. recit. — GP* cells of Table 1 exactly.  `dual/delta_*.csv` are the 27
  entrants generated per model, `dual/recitation_dual_by_model.csv` the
  published-vs-dual comparison with the rank moves.  The recitation CSV keeps its
  `prompt` column: a recitation prompt asks for a statute's wording and discloses
  no case, and `analysis/additional_models/opus5_agent/recitation/scripts/make_ar_prompts.py`
  asserts against it to show the agent-harness row received the benchmark prompt
  verbatim.  The `message` column is dropped there as everywhere.
- **Derived tables and figures** — `experiments/analysis/out/*.tex`, `*.csv`,
  `deepseek_rerun/results/*`, as printed in the submitted paper.  Rows commented
  out in a `.tex` were not printed.  `out/ladder_values.csv`,
  `out/ladder_citations.csv` and `out/oracle_ablation_values.csv` bank the cells
  behind Tables 2 and 3 as plain numbers, so the tables can be checked without
  the arms that produced them.
- **Retrieval battery** — `out/recall_battery.csv`, every retrieval arm's
  recall@k, hit@k and state-law recall over the 81 cases, on both gold sets.
  It carries the query ablation behind §4's claim that the rewriter beats the
  raw task: the `baseline` rows use the deployed rewrite, `query_facts` searches
  with the unrewritten *Sachverhalt*, and `rw_statute` / `rw_citations` are the
  two alternative rewrite prompts (all four prompts are at
  `recall_battery.py:75-111`, their outputs cached in `out/rewrites_*.json`).
  `out/recall_battery_v2.csv` replicates the *Leitsatz* arms on a second index;
  it corroborates the same ordering but produces **no** number printed in the
  paper, which uses `my_leitsatz_index` throughout (`recall_battery.py:64`,
  `corpus_rag_run.py:180`). Neither file contains case text, prompts or answers.

## Known gaps

- The raw per-arm outputs of the retrieval ladder (`rag_titled_combined_*`,
  `oracle_*`) are **not** present; only `rag_titled_combined_ls_text` survives.
  The ladder's numbers are in `out/ladder_table.tex` and `out/oracle_ablation.tex`
  but cannot be recomputed from this branch.
- The *Leitsatz* index is built from the German Legal Decision Corpus
  (Urchs et al.), which is published separately and is not vendored here.
- The cached query rewrites (`out/rewrites_*.json`) are three to five doctrinal
  concepts per case.  They carry no facts, names or outcomes, but they do
  disclose each case's issue list.  They are kept because the rewriter samples:
  without them the `cite*` arms cannot be reproduced at all.
- Because the essays are withheld, the judge scores cannot be re-derived from
  this branch; they can be checked against each other and against the tables,
  not recomputed from the text.
- Some entry points read the withheld case file and therefore cannot run here,
  even though the number they produce is checkable by another route.
  `analysis/build_case_corpus.py coverage` is the one that matters: it derives
  the gold norms from `gpbam_w_rubric.json` before measuring what share of them
  the corpus holds.  The gold set it would derive is published as
  `out/gold_norms_by_case.json` (81 cases, 4,202 summed norms, 51.9 per case), so
  the §5 coverage figure can be recomputed from that file and
  `data/kb_corpus.parquet` with `analysis/refs.py`'s `corpus_key`, which is how
  the norm keys in both are spelled.  A crash from such a script is a withheld
  input, not a missing file.
- The oracle-ablation table's `DeepSeek-V4-Flash` row is not on the same footing
  as the other two.  Its conditions were regenerated in September at DeepSeek's
  low reasoning effort — the regime the rest of the ladder runs at — after the
  August arms turned out to have used a different reasoning budget; the scripts
  are `analysis/ds4_ablation_rerun.py` and `analysis/ds4_low/`, and
  `analysis/ds4_low/README.md` and `DS4-ablation-rerun.md` record which arm each
  cell comes from and what was ruled out on the way there.  The
  `{\tiny high}` rows in `out/ladder_table.tex` and the stacked main table are
  that model at its real high effort; they are commented out and appear nowhere
  in the paper, with their cells banked in `out/ds4_high_gwdg_cells.csv`.
- Three cells in Table 1's *Art. recit. — GP* column carry an asterisk
  (`Open_steuerllm`, `EuroLLM-22B-I`, `Ministral-3-14B-R`).  Those models were
  served locally, no host still serves them, and they have no answers for the 27
  provisions the dual battery adds: their cell is a mean over the 73 shared
  provisions.  `analysis/recitation_repro_probe.py` is the check a new serving
  would have to pass before the cells could be completed.
- Scripts under `analysis/` and `analysis/ds4_low/` derive the repository root
  from their own location (override with `GPBAM_ROOT` / `GPBAM_EXPERIMENTS`).
  They were run with that path hardcoded; nothing else about them changed.
  Edits for publication, beyond that: `experiments/dataset.ipynb` cell 21 is
  redacted as the table above records; the tooling trailers are gone from the
  commit-message template in `analysis/ds4_low/integrate_high_gwdg.py`;
  `analysis/build_case_corpus.py` defaults its scratch directory under the system
  temp directory instead of one machine's path (`GPBAM_CORPUS_SCRATCH` overrides);
  and two `deepseek_rerun/results/*.run.log` have the absolute scratch path in
  their first line replaced by `<scratch>`.  `integrate_high_gwdg.py` is a cron integration step that ends
  by committing and pushing to the private repository this artifact was cut from;
  it is here as the record of how the `{\tiny high}` cells were filled, not as
  something to run.

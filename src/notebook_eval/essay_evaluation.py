"""
Per-model summary + LaTeX table for the essay-writing evaluation.

Import `make_summary` into a notebook and point it at the results DataFrame
(e.g. `no_rag_ji2_result.csv` loaded with pandas):

    from src.notebook_eval.essay_evaluation import make_summary

    # MAIN table: score + legal_ref_sim only
    main = make_summary(model_study_df)

    # EXTENDED table: + tokens + cost
    extended = make_summary(
        model_study_df,
        show_answer_tokens=True,
        show_judge_tokens=True,
        show_gen_cost=True,
        show_judge_cost=True,
    )

Metrics and how each is summarised across a model's ~81 answers
---------------------------------------------------------------
  score          row-wise MEDIAN of the judges (^score_Judge, NaN-skipping), x100
                 -> mean +/- bootstrap SE            (inferential precision)
  judge scores   optional (show_judge_scores): one column per INDIVIDUAL judge
                 (the two Qwen variants merged into one), each x100
                 -> mean +/- bootstrap SE            (inferential precision)
  legal_ref_sim  x100
                 -> mean +/- bootstrap SE            (inferential precision)
  answer_tokens  answering model's completion_tokens
                 -> mean +/- std                     (descriptive spread)
  judge_tokens   SUM of completion_tokens over the judge panel, per answer
                 -> mean +/- std                     (descriptive spread)
  gen_cost       reported generation cost (`total_cost`); blank where missing
  judge_cost     SUM of the judges' total_cost, per answer

Data-specific handling
----------------------
* The two Qwen judge columns ("...FP8" and "qwen3.6-35b-a3b") are ONE logical
  judge: the plain variant was only run as an API fallback for a single model,
  so exactly one is present per row. They are coalesced (combine_first) so every
  model uses a consistent 3-judge panel for both the score median and the panel
  sums (tokens / cost).
* Judge cost/tokens are reconstructed from the per-judge dicts, NOT from the
  top-level `judging_cost` column (that column is empty for ~half the models).
* Generation cost (`total_cost`) is used as reported; it is missing entirely for
  a few models (rendered blank) and $0 for free / self-hosted ones.
"""

import ast

import numpy as np
import pandas as pd

__all__ = ["make_summary"]


# --------------------------------------------------------------------------- #
# Judge-dict parsing
# --------------------------------------------------------------------------- #
def _judge_field(cell, key):
    """One numeric field from a judgement cell; NaN if missing/empty/failed.

    Cells are stringified Python dicts (occasionally already parsed, empty, or
    None for a failed/absent judge).
    """
    if isinstance(cell, dict):
        return cell.get(key, np.nan)
    if not isinstance(cell, str) or cell.strip() == "":
        return np.nan
    try:
        d = ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return np.nan
    return d.get(key, np.nan) if isinstance(d, dict) else np.nan


def _judge_panel_sum(df, key):
    """Per-row SUM over the judge panel of `key` (e.g. 'completion_tokens' or
    'total_cost'). The two Qwen variants are merged into one logical judge.

    `min_count=1` -> a row where every judge failed becomes NaN, not a spurious 0.
    """
    jcols = [c for c in df.columns if c.startswith("judgement_Judge")]
    vals = {c: df[c].map(lambda x: _judge_field(x, key)) for c in jcols}

    qwen = [c for c in jcols if "qwen" in c.lower()]
    logical = [vals[c] for c in jcols if c not in qwen]
    if qwen:  # FP8 + fallback -> single judge
        merged = vals[qwen[0]]
        for c in qwen[1:]:
            merged = merged.combine_first(vals[c])
        logical.append(merged)

    return pd.concat(logical, axis=1).sum(axis=1, skipna=True, min_count=1)


def _logical_judge_scores(df):
    """Per-row score of each LOGICAL judge, coalescing the two Qwen variants
    (FP8 + fallback) into a single judge -- they are the same judge model run
    interchangeably and are mutually exclusive per row.

    Returns an ordered dict {label -> Series}, where `label` is the name inside
    `score_Judge (...)`; the merged Qwen judge is labelled 'qwen3.6-35b-a3b'.
    """
    scols = [c for c in df.columns if c.startswith("score_Judge")]
    name = lambda c: c[c.find("(") + 1:c.rfind(")")] if "(" in c else c
    series = {c: pd.to_numeric(df[c], errors="coerce") for c in scols}

    qwen = [c for c in scols if "qwen" in c.lower()]
    out = {name(c): series[c] for c in scols if c not in qwen}
    if qwen:
        merged = series[qwen[0]]
        for c in qwen[1:]:
            merged = merged.combine_first(series[c])
        out["qwen3.6-35b-a3b"] = merged
    return out


# --------------------------------------------------------------------------- #
# Cell formatters (return LaTeX-ready strings)
# --------------------------------------------------------------------------- #
def _bootstrap_se(a, n_boot=10_000, seed=0):
    """Standard error of the mean via nonparametric bootstrap: resample values
    with replacement `n_boot` times and return the std of the resampled means."""
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return np.nan
    rng = np.random.default_rng(seed)
    return rng.choice(a, size=(n_boot, a.size), replace=True).mean(axis=1).std(ddof=0)


def _fmt_boot(a, digits=1):
    """mean$_{+/- bootstrap SE}$ -- for score / legal_ref_sim."""
    if a.isna().all():
        return None
    return f"{round(np.nanmean(a), digits)}$_{{\\pm{round(_bootstrap_se(a), digits)}}}$"


def _fmt_std(a, digits=0):
    """mean$_{+/- std}$ -- for tokens (or per-answer cost)."""
    if a.isna().all():
        return None
    m, s = np.nanmean(a), np.nanstd(a, ddof=1)
    r = (lambda x: int(round(x))) if digits == 0 else (lambda x: round(x, digits))
    return f"{r(m)}$_{{\\pm{r(s)}}}$"


def _fmt_sum(a, digits=2):
    """total over the model's answers -- for cost as a per-model spend figure."""
    if a.isna().all():
        return None
    return f"{round(np.nansum(a), digits)}"


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def make_summary(
    df,
    *,
    show_score=True,
    show_judge_scores=False,  # add one Score column per individual judge (Qwen merged)
    show_legal_ref_sim=True,
    show_answer_tokens=False,
    show_judge_tokens=False,
    show_gen_cost=False,
    show_judge_cost=False,
    score_digits=1,
    token_digits=0,
    token_divisor=1000,       # set 1000 to report tokens in thousands ("(k)")
    cost_agg="sum",           # "sum" = total spend per model, or "mean" = per-answer mean+/-std
    cost_digits=2,
    caption="Model Performance",
    label="tab:model_comparison",
    print_latex=True,
):
    """Build the per-model summary DataFrame (and optionally print its LaTeX).

    Toggle columns with the ``show_*`` knobs. Rows are always ordered by mean
    median-score, ascending. See the module docstring for how each metric is
    computed and summarised.

    Returns the summary DataFrame (index = short model name, cells = LaTeX-ready
    "mean +/- error" strings, or totals for cost sums).
    """
    w = df.copy()
    w["model_"] = w["model"].str.split("/").str[-1]

    # --- per-row metrics ---------------------------------------------------
    w["score"] = w.filter(regex=r"^score_Judge").median(axis=1) * 100
    w["legal_ref_sim"] = w["legal_ref_sim"] * 100
    w["answer_tokens"] = w["completion_tokens"] / token_divisor
    w["judge_tokens"] = _judge_panel_sum(w, key="completion_tokens") / token_divisor
    w["judge_cost"] = _judge_panel_sum(w, key="total_cost")
    w["gen_cost"] = pd.to_numeric(w["total_cost"], errors="coerce")  # reported; NaN where missing

    order = w.groupby("model_")["score"].mean().sort_values().index

    # --- assemble columns (col, formatter, digits, header) -----------------
    tok_suffix = " (k)" if token_divisor == 1000 else ""
    cost_fmt = _fmt_sum if cost_agg == "sum" else _fmt_std

    specs = []
    if show_score:
        specs.append(("score", _fmt_boot, score_digits, "Score"))
    if show_judge_scores:  # one column per logical judge, right after the ensemble
        for lbl, s in _logical_judge_scores(w).items():
            col = f"judge_score::{lbl}"
            w[col] = s * 100
            specs.append((col, _fmt_boot, score_digits, lbl))
    if show_legal_ref_sim:
        specs.append(("legal_ref_sim", _fmt_boot, score_digits, "Legal Ref. Sim."))
    if show_answer_tokens:
        specs.append(("answer_tokens", _fmt_std, token_digits, f"Answer Tokens{tok_suffix}"))
    if show_judge_tokens:
        specs.append(("judge_tokens", _fmt_std, token_digits, f"Judge Tokens{tok_suffix}"))
    if show_gen_cost:
        specs.append(("gen_cost", cost_fmt, cost_digits, r"Gen. Cost (\$)"))
    if show_judge_cost:
        specs.append(("judge_cost", cost_fmt, cost_digits, r"Judge Cost (\$)"))
    if not specs:
        raise ValueError("Nothing to show: enable at least one show_* column.")

    agg = {col: (lambda a, f=fn, d=dig: f(a, d)) for col, fn, dig, _ in specs}
    summary = w.groupby("model_").agg(agg).loc[order, [s[0] for s in specs]]
    summary.columns = [s[3] for s in specs]
    summary.index.name = "Model"

    summary = summary.fillna("")  # render unavailable metrics as blank, not "NaN"

    if print_latex:
        print(summary.to_latex(caption=caption, label=label, escape=False))

    return summary

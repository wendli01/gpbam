"""Rank correlations between the essay-writing summary metrics and, optionally,
the article-recitation task.

Companion to `essay_evaluation.make_summary`: it consumes the *summary
DataFrame* that function returns (index = short model name, cells = LaTeX-ready
"mean$_{\\pm SE}$" strings) and correlates the per-model point estimates across
models. Article-recitation scores live in a separate CSV, so they are read and
aggregated here and joined on the model name.

Typical use inside a notebook
-----------------------------
    from src.notebook_eval import essay_evaluation, essay_correlation

    summary = essay_evaluation.make_summary(model_study_df, print_latex=False)

    res = essay_correlation.correlate(
        summary,
        recitation_csv="zubaers_result/article_recitation/article_recitation.csv",
        exclude=["mistral-small-3.1-24b-instruct"],   # e.g. drop a failed run
    )
    res            # tidy DataFrame: one row per predictor

What is correlated
------------------
* Spearman rho (rank correlation) ACROSS MODELS -- each model is one point.
* `Score` (the essay judge score) vs. every other numeric column in the summary
  (`Legal Ref. Sim.`, and in the extended table the per-judge scores, tokens,
  costs), plus `Article Recitation` when `recitation_csv` is given.
* Only the point estimate (the mean) is used. The bootstrap SE printed inside
  each cell as `$_{\\pm...}$` is the uncertainty OF that mean, not a quantity to
  correlate, so it is stripped and ignored -- correct behaviour.

Caveats
-------
* n is small (~21 models); the bootstrap CI on each rho is correrespondingly wide.
* Cells are rounded to the summary's display precision (1 decimal by default),
  which can introduce rank ties. To correlate full-precision means instead,
  pass a numeric per-model table straight to `correlate` (any DataFrame whose
  cells are numbers is accepted unchanged).
"""

import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

__all__ = ["correlate", "recitation_scores", "summary_to_numeric", "to_latex"]

# Same generation model served under two names (self-hosted FP8 vs. OpenRouter).
# Normalise so it joins across the essay summary and the recitation CSV.
MODEL_ALIASES = {"Qwen3.6-35B-A3B-FP8": "qwen3.6-35b-a3b"}

_LEADING_NUMBER = re.compile(r"\s*(-?\d+(?:\.\d+)?)")


# --------------------------------------------------------------------------- #
# Parsing / loading
# --------------------------------------------------------------------------- #
def _short(model: str) -> str:
    """'deepseek/deepseek-v3.2' -> 'deepseek-v3.2', with endpoint aliases merged."""
    name = str(model).split("/")[-1]
    return MODEL_ALIASES.get(name, name)


def _parse_cell(cell):
    """Leading numeric value of a summary cell; NaN if blank/unparsable.

    Handles 'mean$_{\\pm SE}$', 'mean$_{\\pm std}$', a bare number, '' and NaN.
    Already-numeric cells pass through unchanged.
    """
    if isinstance(cell, (int, float)):
        return float(cell)
    m = _LEADING_NUMBER.match(str(cell))
    return float(m.group(1)) if m else np.nan


def summary_to_numeric(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Parse a `make_summary` DataFrame into per-model floats (index normalised)."""
    num = summary_df.apply(lambda col: col.map(_parse_cell))
    num.index = [_short(m) for m in summary_df.index]
    num.index.name = summary_df.index.name or "Model"
    return num


def recitation_scores(
    csv_path, dataset: str = "GPBam Laws", scale: float = 100.0
) -> pd.Series:
    """Per-model mean recitation score (ROUGE-L F1) for one dataset, as a Series
    indexed by short model name. `scale=100` reports percentages (rank-neutral)."""
    df = pd.read_csv(csv_path)
    df = df[df["dataset"] == dataset].copy()
    if df.empty:
        raise ValueError(
            f"No rows for dataset={dataset!r}. "
            f"Available: {sorted(pd.read_csv(csv_path)['dataset'].unique())}"
        )
    df["model_"] = df["model"].map(_short)
    df["score"] = pd.to_numeric(df["score"], errors="coerce") * scale
    return df.groupby("model_")["score"].mean().rename("Article Recitation")


# --------------------------------------------------------------------------- #
# Correlation
# --------------------------------------------------------------------------- #
def _boot_ci(x, y, corr, n_boot, seed, alpha=0.05):
    """Percentile bootstrap CI for a correlation, resampling model PAIRS jointly."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    vals = [
        corr(x[i], y[i]).statistic
        for i in idx
        if np.unique(x[i]).size > 2 and np.unique(y[i]).size > 2
    ]
    if not vals:
        return (np.nan, np.nan)
    return tuple(np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def correlate(
    summary_df: pd.DataFrame,
    *,
    recitation_csv=None,
    recitation_dataset: str = "GPBam Laws",
    score_col: str = "Score",
    against=None,
    method: str = "spearman",
    exclude=None,
    n_boot: int = 10_000,
    seed: int = 0,
    verbose: bool = True,
):
    """Correlate the essay `score_col` against other per-model metrics across models.

    Args:
        summary_df: DataFrame from `essay_evaluation.make_summary` (LaTeX-string
            cells) or any per-model table of numbers. Index = model name.
        recitation_csv: If given, its per-model mean recitation score is joined in
            as an extra predictor column `Article Recitation`.
        recitation_dataset: Which recitation dataset to use ("GPBam Laws" or
            "Most cited Laws").
        score_col: The column treated as the outcome (default "Score").
        against: Predictor columns to correlate against `score_col`. Default:
            every other column present (summary columns + recitation).
        method: "spearman" (default) or "pearson".
        exclude: Model names to drop before correlating (matched on the short,
            alias-normalised name). Unmatched names raise, to catch typos.
        n_boot: Bootstrap resamples for the CI on each coefficient; 0 to skip.

    Returns:
        Tidy DataFrame, one row per predictor: n, the coefficient, and (if
        n_boot) its 95% bootstrap CI. Rows ordered by |coefficient| descending.
    """
    corr = {"spearman": spearmanr, "pearson": pearsonr}[method]

    data = summary_to_numeric(summary_df)
    if recitation_csv is not None:
        rec = recitation_scores(recitation_csv, dataset=recitation_dataset)
        data = data.join(rec, how="left")

    # Drop excluded models (validate names to catch typos early).
    if exclude:
        exclude = [_short(m) for m in exclude]
        missing = sorted(set(exclude) - set(data.index))
        if missing:
            raise KeyError(f"exclude names not in data: {missing}")
        data = data.drop(index=exclude)

    if score_col not in data.columns:
        raise KeyError(f"score_col={score_col!r} not in columns {list(data.columns)}")

    predictors = [c for c in (against or data.columns) if c != score_col]

    rows = []
    for col in predictors:
        pair = data[[score_col, col]].dropna()
        n = len(pair)
        if n < 3 or pair[col].nunique() < 2:
            rows.append({"predictor": col, "n": n, method: np.nan,
                         "ci_low": np.nan, "ci_high": np.nan, "dropped": ""})
            continue
        x, y = pair[score_col].to_numpy(), pair[col].to_numpy()
        rho = corr(x, y).statistic
        lo, hi = _boot_ci(x, y, corr, n_boot, seed) if n_boot else (np.nan, np.nan)
        dropped = sorted(set(data.index) - set(pair.index))
        rows.append({"predictor": col, "n": n, method: round(float(rho), 3),
                     "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                     "dropped": ", ".join(dropped)})

    res = (
        pd.DataFrame(rows)
        .set_index("predictor")
        .reindex(columns=["n", method, "ci_low", "ci_high", "dropped"])
    )
    res = res.reindex(res[method].abs().sort_values(ascending=False).index)

    if verbose:
        kept = ", ".join(data.index)
        print(f"{method.capitalize()} correlation of '{score_col}' across "
              f"{len(data)} models (rows=points).")
        print(f"models: {kept}\n")
        print(res.to_string())

    return res


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #
def to_latex(
    res: pd.DataFrame,
    *,
    caption: str = "",
    label: str = "tab:essay_correlation",
    digits: int = 2,
    outcome: str = "Score",
    show_ci: bool = True,
    print_latex: bool = True,
) -> str:
    """Render a `correlate` result DataFrame as a paper-ready LaTeX table.

    One row per predictor: n, the coefficient (rho for Spearman, r for Pearson),
    and its 95% bootstrap CI. The `dropped` column is intentionally omitted from
    the table (report any non-uniform n in the caption instead).

    Returns the LaTeX string (and prints it when `print_latex`).
    """
    method = "spearman" if "spearman" in res.columns else "pearson"
    symbol = {"spearman": r"$\rho$", "pearson": r"$r$"}[method]
    has_ci = show_ci and "ci_low" in res.columns and res["ci_low"].notna().any()

    colspec = "lrr" + ("r" if has_ci else "")
    header = f"Predictor & $n$ & {symbol}" + (r" & 95\% CI" if has_ci else "")

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}" if caption else None,
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    for pred, row in res.iterrows():
        cells = [str(pred), f"{int(row['n'])}"]
        cells.append("--" if pd.isna(row[method]) else f"{row[method]:.{digits}f}")
        if has_ci:
            cells.append(
                "--" if pd.isna(row["ci_low"])
                else f"$[{row['ci_low']:.{digits}f},\\,{row['ci_high']:.{digits}f}]$"
            )
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    latex = "\n".join(l for l in lines if l is not None)
    if print_latex:
        print(latex)
    return latex

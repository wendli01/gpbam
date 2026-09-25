"""
Per-(model, dataset) summary + LaTeX table for the Article Recitation task.

Import `make_article_summary` into a notebook and point it at the results
DataFrame (e.g. `article_recitation.csv` loaded with pandas), the same way
`essay_evaluation.make_summary` is used elsewhere:

    from src.notebook_eval import article_recitation_evaluation

    # MAIN table: Score (mean +/- bootstrap SE) + Score Std
    main = article_recitation_evaluation.make_article_summary(
        model_study_df,
        print_latex=False,
    )

    # EXTENDED table: + Total Cost + Completion Tokens (k), with LaTeX
    extended = article_recitation_evaluation.make_article_summary(
        model_study_df,
        show_total_cost=True,
        show_completion_tokens=True,
    )

Metrics and how each is summarised (per model x dataset, ~100 articles each)
---------------------------------------------------------------------------
  score              -> row-wise ROUGE-L F1 (from the CSV `score` column), x100
                        Score column : mean $_{+/- bootstrap SE}$   (estimate precision)
                        Score Std    : sample std across articles    (spread)
  total_cost         -> mean reported generation cost; blank where missing
  completion_tokens  -> mean answering-model completion tokens, in thousands
                        ("(k)") when token_divisor == 1000

Only `score` is scaled to a percentage; cost and tokens are reported in their
true units (cost in $, tokens divided by `token_divisor`).

Why both SE and std
-------------------
Each cell averages ~100 per-article ROUGE-L scores. The per-article spread is
large (std ~ the mean itself), so:
  * the bootstrap SE (= std / sqrt(n) for a mean) quantifies how precisely the
    mean is known and lets you tell whether two models really differ -- this is
    the same summary `essay_evaluation.make_summary` prints for its scores;
  * the std reports the article-to-article consistency of a model.

Layout
------
Rows are grouped by model (a \\multirow spanning that model's datasets) and the
models are ordered by their `order_dataset` mean score, ascending. Which columns
appear is controlled by the `show_*` knobs, and the tabular column spec / header
row adapt automatically.
"""

import numpy as np
import pandas as pd

__all__ = ["make_article_summary"]


def _bootstrap_se(a, n_boot=10_000, seed=0):
    """Standard error of the mean via nonparametric bootstrap: resample the
    values with replacement `n_boot` times and return the std of the resampled
    means. Same method as `essay_evaluation._bootstrap_se`, for consistency."""
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return np.nan
    rng = np.random.default_rng(seed)
    return rng.choice(a, size=(n_boot, a.size), replace=True).mean(axis=1).std(ddof=0)


def _score_cell(mean, err, digits):
    """Score cell string: blank if missing, ``mean`` alone when no error is
    requested, else ``mean$_{\\pm err}$`` (LaTeX-ready, as in essay_evaluation)."""
    if pd.isna(mean):
        return ""
    if err is None or pd.isna(err):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}$_{{\\pm{err:.{digits}f}}}$"


def _fmt(value, digits):
    """LaTeX cell string matching how the DataFrame shows a numeric value: blank
    if missing, a fixed-decimal string when ``digits`` is an int, and -- when
    ``digits`` is None (raw/unrounded column) -- fixed-point at pandas' own
    display precision, so the stored value stays unrounded yet the printed cell
    matches the DataFrame's on-screen figure instead of a scientific repr."""
    if pd.isna(value):
        return ""
    if digits is None:
        digits = pd.get_option("display.precision")   # match the DataFrame display
    return f"{value:.{digits}f}"


def make_article_summary(
    df,
    *,
    show_score=True,
    show_score_std=True,            # separate std column (article-to-article spread)
    show_total_cost=False,          # extended column
    show_completion_tokens=False,   # extended column
    score_error="se",              # "se" (bootstrap) | "std" | None -> the +/- in the Score cell
    order_dataset="GPBam Laws",     # order models by this dataset's mean score (ascending)
    strip_model_path=True,          # "openai/gpt-5-nano" -> "gpt-5-nano"
    score_digits=2,
    cost_digits=None,               # None -> total cost kept raw/unrounded (values are tiny)
    token_digits=1,
    token_divisor=1000,             # 1000 -> report completion tokens in thousands ("(k)")
    n_boot=10_000,                  # bootstrap resamples for the SE
    seed=0,
    dataset_short=None,             # {long_name: short_name} for the LaTeX Dataset column
    caption=r"Model performance on \textbf{Article Recitation} task.",
    label="tab:article_recitation",
    print_latex=True,
):
    """Per-(model, dataset) summary for the Article Recitation task, MAIN or EXTENDED.

    MAIN (default)  : Score (mean +/- SE) and Score Std.
    EXTENDED        : add Total Cost / Completion Tokens via the show_* knobs::

        make_article_summary(
            model_study_df,
            show_total_cost=True,
            show_completion_tokens=True,
        )

    Each ``(model, dataset)`` cell aggregates that model's ~100 per-article
    ROUGE-L scores. **Score** is the mean scaled to a percentage (x100) with the
    ``score_error`` annotation appended (``"se"`` = bootstrap standard error,
    ``"std"`` = sample std, ``None`` = mean only). **Score Std** is the sample
    std across articles (percentage), shown when ``show_score_std``. Cost is the
    mean generation cost (in $); completion tokens the mean divided by
    ``token_divisor`` (1000 -> thousands, header suffixed " (k)").

    Models are ordered by their ``order_dataset`` mean score (ascending), keeping
    each model's dataset rows together. Numeric columns are rounded to their
    display precision so the returned DataFrame matches the printed LaTeX cell
    for cell; the Score cell is one LaTeX-ready string used in both.
    ``cost_digits=None`` (the default) leaves ``total_cost`` unrounded.

    Returns the summary DataFrame (MultiIndex model/dataset); when ``print_latex``
    also prints the \\multirow LaTeX table.
    """
    if score_error not in ("se", "std", None):
        raise ValueError("score_error must be 'se', 'std', or None.")

    w = df.copy()
    if strip_model_path:
        w["model"] = w["model"].str.split("/").str[-1]

    # --- per-(model, dataset) statistics -----------------------------------
    score100 = w["score"] * 100
    g = score100.groupby([w["model"], w["dataset"]])
    other = w.groupby(["model", "dataset"])[["total_cost", "completion_tokens"]].mean()

    stats = pd.DataFrame({
        "score_mean": g.mean(),
        "score_std": g.std(),                                   # sample std (ddof=1)
        "score_se": g.apply(lambda a: _bootstrap_se(a.to_numpy(), n_boot=n_boot, seed=seed)),
        "total_cost": other["total_cost"],                      # raw; rounded below if requested
        "completion_tokens": (other["completion_tokens"] / token_divisor).round(token_digits),
    })

    # order models by their `order_dataset` mean score (ascending); rows stay grouped
    model_order = (
        stats.xs(order_dataset, level="dataset")["score_mean"]
        .sort_values(ascending=True)
        .index
    )
    stats = stats.reindex(model_order, level="model")

    # --- assemble display columns: (header, kind) ---------------------------
    # kind == "text" -> cell already a LaTeX string; otherwise the decimals to format
    tok_suffix = " (k)" if token_divisor == 1000 else ""
    err = {"se": stats["score_se"], "std": stats["score_std"], None: None}[score_error]

    data, cols = {}, []
    if show_score:
        errs = err if err is not None else pd.Series(np.nan, index=stats.index)
        data["Score"] = [
            _score_cell(m, e, score_digits) for m, e in zip(stats["score_mean"], errs)
        ]
        cols.append(("Score", "text"))
    if show_score_std:
        data["Score Std"] = stats["score_std"].round(score_digits).to_numpy()
        cols.append(("Score Std", score_digits))
    if show_total_cost:
        tc = stats["total_cost"] if cost_digits is None else stats["total_cost"].round(cost_digits)
        data[r"Total Cost (\$)"] = tc.to_numpy()
        cols.append((r"Total Cost (\$)", cost_digits))
    if show_completion_tokens:
        header = f"Completion Tokens{tok_suffix}"
        data[header] = stats["completion_tokens"].to_numpy()
        cols.append((header, token_digits))
    if not cols:
        raise ValueError("Nothing to show: enable at least one show_* column.")

    summary = pd.DataFrame(data, index=stats.index)[[h for h, _ in cols]]

    if print_latex:
        short = dataset_short or {"GPBam Laws": "GPBam", "Most cited Laws": "Most cited"}
        colspec = "l l " + " ".join("r" for _ in cols)
        headers = " & ".join(rf"\textbf{{{h}}}" for h, _ in cols)

        print(r"\begin{table}[ht]")
        print(r"\centering")
        print(rf"\begin{{tabular}}{{{colspec}}}")
        print(r"\hline")
        print(rf"\textbf{{Model}} & \textbf{{Dataset}} & {headers} \\")
        print(r"\hline")

        for model, group in summary.reset_index().groupby("model", sort=False):
            group = group.reset_index(drop=True)
            for i, row in group.iterrows():
                dataset = short.get(row["dataset"], row["dataset"])
                cells = [
                    (row[h] if kind == "text" else _fmt(row[h], kind))
                    for h, kind in cols
                ]
                if i == 0:
                    print(rf"\multirow{{{len(group)}}}{{*}}{{{model}}}")
                print(rf"  & {dataset} & {' & '.join(cells)} \\")
            print(r"\hline")

        print(r"\end{tabular}")
        print(rf"\caption{{{caption}}}")
        if label:
            print(rf"\label{{{label}}}")
        print(r"\end{table}")

    return summary

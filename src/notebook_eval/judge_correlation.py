"""Judge-to-judge correlation analysis for the essay-writing notebooks.

Two analysis levels are available. At model level, each judge score is averaged
over a model's exam cases and each model is one observation. At essay level,
every individual essay is one observation. Spearman measures rank agreement,
Pearson measures linear association, and quadratic-weighted kappa (QWK) also
penalises differences in the judges' absolute score levels.

Typical notebook use:

    from src.notebook_eval import judge_correlation_codex

    analysis = judge_correlation_codex.analyze(
        model_study_df,
        level="model",  # or "essay"
        methods=["spearman", "pearson", "qwk"],
    )
    analysis["pairwise"]
    analysis["scale_summary"]

The mutually exclusive Qwen FP8/API fallback columns are coalesced into one
logical qwen3.6-35b-a3b judge before aggregation.
"""

from itertools import combinations
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

__all__ = [
    "DEFAULT_FIGURE_CAPTION",
    "DEFAULT_TABLE_CAPTION",
    "agreement_figure_to_latex",
    "agreement_table_to_latex",
    "analyze",
    "analyze_levels",
    "essay_level_scores",
    "logical_judge_scores",
    "model_level_scores",
    "pairwise_correlations",
    "plot_correlation_heatmap",
    "plot_level_agreement",
    "plot_score_distributions",
]

_JUDGE_NAME = re.compile(r"^score_Judge \((.*)\)$")
_SUPPORTED_METHODS = ("spearman", "pearson", "qwk")

DEFAULT_TABLE_CAPTION = (
    r"\textbf{Inter-judge agreement in the GPBam essay-writing experiment "
    r"(no retrieval).} Agreement is evaluated pairwise for three LLM judges. "
    r"At the model level, one observation is a generation model and each "
    r"judge's value is its mean score over that model's available essays; at "
    r"the essay level, one observation is an individual essay for which both "
    r"judges supplied a score. Spearman's $\rho$ measures rank agreement, "
    r"Pearson's $r$ linear association, and quadratic-weighted kappa "
    r"(QWK, $\kappa_w$) agreement in absolute score levels. Scores lie on "
    r"0--1 and are represented on an equivalent fixed 0--100 ordinal grid for "
    r"QWK. Entries are coefficients with percentile 95\% bootstrap confidence "
    r"intervals. Model-level intervals resample models; essay-level intervals "
    r"resample complete generation-model clusters. Model-level $p$-values are "
    r"analytic for Spearman/Pearson and permutation-based for QWK; essay-level "
    r"$p$-values are omitted because essays from the same model are dependent. "
    r"$n$ is the pairwise-complete number of models or essays, so it can vary "
    r"when a judge score is missing. The FP8 and API Qwen judge columns are "
    r"coalesced into one logical judge. Higher coefficients indicate stronger "
    r"agreement; QWK equals 1 under perfect agreement."
)

DEFAULT_FIGURE_CAPTION = (
    r"\textbf{Inter-judge agreement at model and essay levels in the GPBam "
    r"essay-writing experiment (no retrieval).} Points show Spearman's "
    r"$\rho$, Pearson's $r$, and quadratic-weighted kappa "
    r"(QWK, $\kappa_w$) for each judge pair; horizontal bars are percentile "
    r"95\% bootstrap confidence intervals. In the left panel, each observation "
    r"is a generation model and each judge's value is its mean over that "
    r"model's available essays, with models resampled for the intervals. In "
    r"the right panel, each observation is an individual essay scored by both "
    r"judges, with complete generation-model clusters resampled to preserve "
    r"within-model dependence. Spearman measures rank agreement and Pearson "
    r"linear association, whereas QWK additionally penalises systematic "
    r"differences in absolute score levels after representing the original "
    r"0--1 scores on an equivalent fixed 0--100 ordinal grid. The displayed "
    r"essay counts vary slightly because correlations use pairwise-complete "
    r"judge scores. The FP8 and API Qwen columns are coalesced into one logical "
    r"judge. Higher values indicate stronger agreement."
)


def _short_model_name(model) -> str:
    """Return the final component of a provider/model ID."""
    return str(model).split("/")[-1]


def _judge_name(column: str) -> str:
    match = _JUDGE_NAME.match(column)
    return match.group(1) if match else column


def logical_judge_scores(
    df: pd.DataFrame,
    *,
    model_col: str = "model",
    scale: float = 100.0,
) -> pd.DataFrame:
    """Return per-essay scores with one column per logical judge.

    The output contains a Model column followed by numeric judge columns.
    Scores are multiplied by scale; the default reports percentages.
    """
    if model_col not in df.columns:
        raise KeyError(f"model_col={model_col!r} not found in the DataFrame")

    score_cols = [c for c in df.columns if c.startswith("score_Judge")]
    if not score_cols:
        raise ValueError("No columns beginning with 'score_Judge' were found")

    numeric = df[score_cols].apply(pd.to_numeric, errors="coerce")
    qwen_cols = [c for c in score_cols if "qwen" in c.lower()]

    out = pd.DataFrame(index=df.index)
    out["Model"] = df[model_col].map(_short_model_name)

    for column in score_cols:
        if column not in qwen_cols:
            out[_judge_name(column)] = numeric[column] * scale

    if qwen_cols:
        out["qwen3.6-35b-a3b"] = (
            numeric[qwen_cols].bfill(axis=1).iloc[:, 0] * scale
        )

    judge_cols = [c for c in out.columns if c != "Model"]
    if len(judge_cols) < 2:
        raise ValueError(
            "Judge correlation requires scores from at least two logical judges"
        )
    return out


def model_level_scores(
    df: pd.DataFrame,
    *,
    model_col: str = "model",
    scale: float = 100.0,
    exclude=None,
) -> pd.DataFrame:
    """Average every judge's case scores per model."""
    essay_scores = logical_judge_scores(df, model_col=model_col, scale=scale)
    scores = essay_scores.groupby("Model", sort=False).mean(numeric_only=True)

    if exclude:
        excluded = [_short_model_name(model) for model in exclude]
        missing = sorted(set(excluded) - set(scores.index))
        if missing:
            raise KeyError(f"exclude names not in data: {missing}")
        scores = scores.drop(index=excluded)

    return scores


def essay_level_scores(
    df: pd.DataFrame,
    *,
    model_col: str = "model",
    scale: float = 100.0,
    exclude=None,
) -> pd.DataFrame:
    """Return one row per essay with its model and logical-judge scores."""
    scores = logical_judge_scores(df, model_col=model_col, scale=scale)

    if exclude:
        excluded = [_short_model_name(model) for model in exclude]
        missing = sorted(set(excluded) - set(scores["Model"]))
        if missing:
            raise KeyError(f"exclude names not in data: {missing}")
        scores = scores.loc[~scores["Model"].isin(excluded)]

    # A unique index keeps cluster labels aligned after pairwise NaN removal.
    return scores.reset_index(drop=True)


def _normalise_methods(methods=None, method=None):
    """Validate methods while retaining the old singular method argument."""
    if methods is not None and method is not None:
        raise ValueError("Pass either methods or method, not both")
    selected = methods if methods is not None else method
    if selected is None:
        selected = ["spearman"]
    elif isinstance(selected, str):
        selected = [selected]

    normalised = []
    for name in selected:
        name = str(name).lower()
        if name not in _SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method {name!r}; choose from {_SUPPORTED_METHODS}"
            )
        if name not in normalised:
            normalised.append(name)
    if not normalised:
        raise ValueError("methods must contain at least one method")
    return normalised


def _qwk_score(x, y, *, score_scale: float, qwk_categories: int) -> float:
    """Quadratic-weighted kappa after fixed-grid ordinal quantisation.

    The direct moment formula is mathematically equivalent to QWK's quadratic
    disagreement matrix, but avoids repeatedly constructing a large matrix
    during bootstrap and permutation resampling.
    """
    if qwk_categories < 2:
        raise ValueError("qwk_categories must be at least 2")
    if score_scale <= 0:
        raise ValueError("score_scale must be positive")

    maximum_label = qwk_categories - 1
    x_labels = np.rint(
        np.clip(np.asarray(x, float), 0, score_scale)
        / score_scale
        * maximum_label
    )
    y_labels = np.rint(
        np.clip(np.asarray(y, float), 0, score_scale)
        / score_scale
        * maximum_label
    )

    observed = np.mean((x_labels - y_labels) ** 2)
    expected = (
        np.var(x_labels)
        + np.var(y_labels)
        + (np.mean(x_labels) - np.mean(y_labels)) ** 2
    )
    return np.nan if expected == 0 else float(1 - observed / expected)


def _coefficient(x, y, method, *, score_scale, qwk_categories):
    if method == "spearman":
        return float(spearmanr(x, y).statistic)
    if method == "pearson":
        return float(pearsonr(x, y).statistic)
    return _qwk_score(
        x,
        y,
        score_scale=score_scale,
        qwk_categories=qwk_categories,
    )


def _analytic_p_value(x, y, method):
    if method == "spearman":
        return float(spearmanr(x, y).pvalue)
    if method == "pearson":
        return float(pearsonr(x, y).pvalue)
    return np.nan


def _permutation_p_value(
    x,
    y,
    statistic,
    *,
    observed,
    n_permutations,
    seed,
):
    """Two-sided permutation p-value for a statistic whose null value is zero."""
    if not n_permutations:
        return np.nan
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_permutations):
        estimate = statistic(x, rng.permutation(y))
        if np.isfinite(estimate) and abs(estimate) >= abs(observed):
            extreme += 1
    return (extreme + 1) / (n_permutations + 1)


def _bootstrap_ci(
    x,
    y,
    statistic,
    *,
    n_boot: int,
    seed: int,
    clusters=None,
    alpha: float = 0.05,
):
    """Percentile CI, resampling observations or whole clusters jointly."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    estimates = []

    for _ in range(n_boot):
        if clusters is None:
            indices = rng.integers(0, x.size, size=x.size)
        else:
            cluster_values = np.asarray(clusters)
            unique_clusters = pd.unique(cluster_values)
            sampled_clusters = rng.choice(
                unique_clusters,
                size=len(unique_clusters),
                replace=True,
            )
            indices = np.concatenate(
                [
                    np.flatnonzero(cluster_values == cluster)
                    for cluster in sampled_clusters
                ]
            )
        x_sample = x[indices]
        y_sample = y[indices]
        if np.unique(x_sample).size < 2 or np.unique(y_sample).size < 2:
            continue
        estimate = statistic(x_sample, y_sample)
        if np.isfinite(estimate):
            estimates.append(float(estimate))

    if not estimates:
        return np.nan, np.nan
    return tuple(
        np.quantile(estimates, [alpha / 2, 1 - alpha / 2]).astype(float)
    )


def pairwise_correlations(
    scores: pd.DataFrame,
    *,
    methods=None,
    method=None,
    n_boot: int = 10_000,
    n_permutations: int = 10_000,
    seed: int = 0,
    clusters: pd.Series | None = None,
    score_scale: float = 100.0,
    qwk_categories: int = 101,
) -> pd.DataFrame:
    """Return a tidy table containing every judge pair and requested method.

    When clusters are supplied, the coefficient still uses all observations,
    but the confidence interval resamples whole clusters. The ordinary
    correlation p-value assumes independent observations, so it is not reported
    for clustered analysis. At unclustered model level, Spearman/Pearson use
    analytic p-values and QWK uses a two-sided permutation p-value.
    """
    selected_methods = _normalise_methods(methods, method)
    rows = []

    if clusters is not None:
        # A cluster Series is aligned to scores by index label, so a mismatched
        # index silently yields NaN labels that match no row and quietly shrink
        # every bootstrap resample. Fail loudly instead.
        clusters = pd.Series(clusters, index=scores.index)
        n_unlabelled = int(clusters.isna().sum())
        if n_unlabelled:
            raise ValueError(
                f"clusters has no label for {n_unlabelled} of {len(scores)} "
                "rows in scores; its index must match scores.index. Pass "
                "clusters.to_numpy() to align by position instead."
            )

    for pair_number, (judge_a, judge_b) in enumerate(
        combinations(scores.columns, 2)
    ):
        paired = scores[[judge_a, judge_b]].dropna()
        n = len(paired)
        n_total = len(scores)
        pair_clusters = (
            None if clusters is None else clusters.loc[paired.index].to_numpy()
        )
        n_clusters = n if pair_clusters is None else pd.unique(pair_clusters).size

        valid = (
            n >= 3
            and paired[judge_a].nunique() >= 2
            and paired[judge_b].nunique() >= 2
        )
        x = paired[judge_a].to_numpy()
        y = paired[judge_b].to_numpy()

        for method_name in selected_methods:
            statistic = lambda a, b, name=method_name: _coefficient(
                a,
                b,
                name,
                score_scale=score_scale,
                qwk_categories=qwk_categories,
            )

            if not valid:
                coefficient = p_value = ci_low = ci_high = np.nan
                p_value_type = "not available"
            else:
                coefficient = statistic(x, y)
                if pair_clusters is not None:
                    p_value = np.nan
                    p_value_type = "not reported (clustered essays)"
                elif method_name == "qwk":
                    p_value = _permutation_p_value(
                        x,
                        y,
                        statistic,
                        observed=coefficient,
                        n_permutations=n_permutations,
                        seed=seed + pair_number,
                    )
                    p_value_type = (
                        f"permutation ({n_permutations})"
                        if n_permutations
                        else "not requested"
                    )
                else:
                    p_value = _analytic_p_value(x, y, method_name)
                    p_value_type = "analytic"

                if n_boot:
                    ci_low, ci_high = _bootstrap_ci(
                        x,
                        y,
                        statistic,
                        n_boot=n_boot,
                        seed=seed + pair_number,
                        clusters=pair_clusters,
                    )
                else:
                    ci_low = ci_high = np.nan

            rows.append(
                {
                    "judge_a": judge_a,
                    "judge_b": judge_b,
                    "n": n,
                    "n_total": n_total,
                    "n_missing": n_total - n,
                    "n_clusters": n_clusters,
                    "method": method_name,
                    "coefficient": coefficient,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": p_value,
                    "p_value_type": p_value_type,
                }
            )

    return pd.DataFrame(rows)


def _correlation_matrices(pairwise, judge_names, methods):
    matrices = {}
    for method in methods:
        matrix = pd.DataFrame(
            np.eye(len(judge_names)),
            index=judge_names,
            columns=judge_names,
        )
        subset = pairwise.loc[pairwise["method"] == method]
        for row in subset.itertuples(index=False):
            matrix.loc[row.judge_a, row.judge_b] = row.coefficient
            matrix.loc[row.judge_b, row.judge_a] = row.coefficient
        matrices[method] = matrix
    return matrices


def analyze(
    df: pd.DataFrame,
    *,
    model_col: str = "model",
    level: str = "model",
    methods=None,
    method=None,
    scale: float = 100.0,
    exclude=None,
    n_boot: int = 10_000,
    n_permutations: int = 10_000,
    seed: int = 0,
    qwk_categories: int = 101,
):
    """Run judge agreement and score-scale analyses.

    Args:
        level: "model" averages case scores first, so each generation model is
            one observation. "essay" correlates all individual essays and uses
            generation-model cluster bootstrap confidence intervals.
        methods: Any subset of ["spearman", "pearson", "qwk"]. A single string
            is also accepted. The legacy singular method argument still works.
        qwk_categories: Fixed ordinal grid for QWK. The default 101 rounds the
            percentage scores to the nearest percentage point.
    """
    selected_methods = _normalise_methods(methods, method)
    if level == "model":
        scores = model_level_scores(
            df,
            model_col=model_col,
            scale=scale,
            exclude=exclude,
        )
        clusters = None
    elif level == "essay":
        essays = essay_level_scores(
            df,
            model_col=model_col,
            scale=scale,
            exclude=exclude,
        )
        clusters = essays.pop("Model")
        scores = essays
    else:
        raise ValueError("level must be 'model' or 'essay'")

    pairwise = pairwise_correlations(
        scores,
        methods=selected_methods,
        n_boot=n_boot,
        n_permutations=n_permutations,
        seed=seed,
        clusters=clusters,
        score_scale=scale,
        qwk_categories=qwk_categories,
    )
    matrices = _correlation_matrices(
        pairwise,
        list(scores.columns),
        selected_methods,
    )
    scale_summary = scores.agg(["mean", "median", "std", "min", "max"]).T
    scale_summary.index.name = "Judge"
    availability = pd.DataFrame(
        {
            "n_total": len(scores),
            "n_available": scores.notna().sum(),
            "n_missing": scores.isna().sum(),
        }
    )
    availability.index.name = "Judge"

    return {
        "level": level,
        "methods": selected_methods,
        "scores": scores,
        # Backward-compatible aliases; use the generic "scores" key in new code.
        "model_scores": scores if level == "model" else None,
        "essay_scores": scores if level == "essay" else None,
        "correlation_matrices": matrices,
        # Backward-compatible alias for single-method notebook code.
        "correlation_matrix": matrices[selected_methods[0]],
        "pairwise": pairwise,
        "availability": availability,
        "scale_summary": scale_summary,
        "inference_note": (
            "Rows are independent generation models; p-values and bootstrap "
            "confidence intervals use models as observations."
            if level == "model"
            else
            "The coefficient pools individual essays. Confidence intervals "
            "resample generation-model clusters; ordinary p-values are omitted "
            "because essays from the same model are not independent. The CI "
            "does not additionally model dependence from the same case being "
            "answered by multiple models."
        ),
    }



def analyze_levels(
    df: pd.DataFrame,
    *,
    model_col: str = "model",
    methods=None,
    method=None,
    scale: float = 100.0,
    exclude=None,
    n_boot: int = 10_000,
    n_permutations: int = 10_000,
    seed: int = 0,
    qwk_categories: int = 101,
):
    """Analyze both levels and return their results plus one combined table.

    methods/method: same option, two names. `methods` (list or str) is the
    current API; `method` (singular) is the legacy alias kept for old
    notebook code. Pass only one; if neither is given it defaults to
    ["spearman"].
    """
    common = {
        "model_col": model_col,
        "methods": methods,
        "method": method,
        "scale": scale,
        "exclude": exclude,
        "n_boot": n_boot,
        "n_permutations": n_permutations,
        "seed": seed,
        "qwk_categories": qwk_categories,
    }
    model = analyze(df, level="model", **common)
    essay = analyze(df, level="essay", **common)
    table = pd.concat(
        [
            model["pairwise"].assign(level="model"),
            essay["pairwise"].assign(level="essay"),
        ],
        ignore_index=True,
    )
    table.insert(0, "level", table.pop("level"))
    return {"model": model, "essay": essay, "table": table}


def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    *,
    title: str | None = None,
    method: str = "spearman",
    level: str = "model",
    figsize=(7, 5),
):
    """Plot an annotated correlation heatmap and return (figure, axes)."""
    if title is None:
        title = (
            f"{method.capitalize()} correlations between LLM judges "
            f"({level} level)"
        )
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        vmin=-1,
        vmax=1,
        center=0,
        cmap="vlag",
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_score_distributions(
    scores: pd.DataFrame,
    *,
    title: str | None = None,
    level: str = "model",
    figsize=(9, 5),
):
    """Plot judge score distributions and return (figure, axes)."""
    if level not in {"model", "essay"}:
        raise ValueError("level must be 'model' or 'essay'")
    if title is None:
        title = f"Score distributions of the LLM judges ({level} level)"
    y_label = (
        "Mean score across cases (%)" if level == "model" else "Essay score (%)"
    )
    long = (
        scores.rename_axis("Observation")
        .reset_index()
        .melt(id_vars="Observation", var_name="Judge", value_name="Score")
    )

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=long, x="Judge", y="Score", color="lightgray", ax=ax)
    sns.stripplot(
        data=long,
        x="Judge",
        y="Score",
        color="black",
        alpha=0.65,
        size=4,
        ax=ax,
    )
    ax.set(xlabel="", ylabel=y_label, title=title)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig, ax


def plot_level_agreement(
    model_analysis,
    essay_analysis=None,
    *,
    methods=None,
    x_limits=(-0.05, 1.02),
    figsize=(13, 5.5),
    title="Inter-judge agreement at model and essay levels",
    save_path=None,
    dpi=300,
):
    """Draw a two-panel coefficient forest plot with 95% confidence intervals.

    The left panel shows agreement across model-level means; the right panel
    shows agreement across individual essays. Methods are offset vertically
    within each judge pair so their point estimates and intervals remain legible.
    Pass either the two analyses separately or the bundle from analyze_levels.
    """
    if essay_analysis is None:
        if not {"model", "essay"}.issubset(model_analysis):
            raise ValueError(
                "A single argument must be the result of analyze_levels"
            )
        model_analysis, essay_analysis = (
            model_analysis["model"],
            model_analysis["essay"],
        )
    if model_analysis.get("level") != "model":
        raise ValueError("model_analysis must come from analyze(level='model')")
    if essay_analysis.get("level") != "essay":
        raise ValueError("essay_analysis must come from analyze(level='essay')")

    available = [
        name
        for name in model_analysis["methods"]
        if name in essay_analysis["methods"]
    ]
    selected = _normalise_methods(methods) if methods is not None else available
    missing = sorted(set(selected) - set(available))
    if missing:
        raise KeyError(f"methods missing from one or both analyses: {missing}")

    pair_order = list(
        dict.fromkeys(
            zip(
                model_analysis["pairwise"]["judge_a"],
                model_analysis["pairwise"]["judge_b"],
            )
        )
    )
    pair_labels = [f"{a}\nvs.\n{b}" for a, b in pair_order]
    base_y = np.arange(len(pair_order), dtype=float)
    offsets = (
        np.array([0.0])
        if len(selected) == 1
        else np.linspace(-0.22, 0.22, len(selected))
    )
    display_names = {
        "spearman": r"Spearman $\rho$",
        "pearson": r"Pearson $r$",
        "qwk": r"QWK $\kappa_w$",
    }
    colors = dict(
        zip(selected, sns.color_palette("colorblind", n_colors=len(selected)))
    )
    markers = {"spearman": "o", "pearson": "s", "qwk": "D"}

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )

    for ax, analysis in zip(axes, [model_analysis, essay_analysis]):
        table = analysis["pairwise"].copy()
        table["pair"] = list(zip(table["judge_a"], table["judge_b"]))

        for offset, method_name in zip(offsets, selected):
            subset = (
                table.loc[table["method"] == method_name]
                .set_index("pair")
                .reindex(pair_order)
            )
            estimate = subset["coefficient"].to_numpy(float)
            low = subset["ci_low"].to_numpy(float)
            high = subset["ci_high"].to_numpy(float)
            x_error = np.vstack(
                [
                    np.maximum(estimate - low, 0),
                    np.maximum(high - estimate, 0),
                ]
            )
            ax.errorbar(
                estimate,
                base_y + offset,
                xerr=x_error,
                fmt=markers[method_name],
                color=colors[method_name],
                ecolor=colors[method_name],
                elinewidth=1.5,
                capsize=3,
                markersize=6,
                label=display_names[method_name],
            )

        n_min = int(table["n"].min())
        n_max = int(table["n"].max())
        n_text = str(n_min) if n_min == n_max else f"{n_min}--{n_max}"
        if analysis["level"] == "model":
            panel_title = f"Model level ({n_text} models)"
        else:
            cluster_count = int(table["n_clusters"].max())
            panel_title = (
                f"Essay level ({n_text} essays; "
                f"{cluster_count} model clusters)"
            )

        ax.set_title(panel_title)
        ax.set_yticks(base_y, labels=pair_labels)
        ax.axvline(0, color="0.45", linewidth=1, linestyle="--", zorder=0)
        ax.grid(axis="x", color="0.9", linewidth=0.8)
        ax.set_axisbelow(True)
        if x_limits is not None:
            ax.set_xlim(*x_limits)

    axes[0].invert_yaxis()
    fig.subplots_adjust(
        left=0.17,
        right=0.985,
        top=0.84,
        bottom=0.22,
        wspace=0.035,
    )
    fig.supxlabel("Agreement coefficient (95% bootstrap CI)", y=0.105)
    fig.suptitle(title, fontsize=13, y=0.97)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=len(selected),
        frameon=False,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig, axes


def _latex_escape(value) -> str:
    """Escape text inserted into LaTeX table cells."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _format_p_value(value, digits: int) -> str:
    if pd.isna(value):
        return "--"
    threshold = 10 ** (-digits)
    if value < threshold:
        return f"$<{threshold:.{digits}f}$"
    return f"{value:.{digits}f}"


def agreement_table_to_latex(
    agreement_table: pd.DataFrame,
    *,
    caption: str = DEFAULT_TABLE_CAPTION,
    label: str = "tab:judge_agreement",
    digits: int = 3,
    p_digits: int = 3,
    print_latex: bool = True,
) -> str:
    """Render the combined agreement table as a standalone LaTeX table.

    The input is normally agreement["table"] from analyze_levels. Required
    columns are validated so a stale or unrelated DataFrame fails explicitly.
    """
    required = {
        "level",
        "judge_a",
        "judge_b",
        "n",
        "n_clusters",
        "method",
        "coefficient",
        "ci_low",
        "ci_high",
        "p_value",
    }
    missing = sorted(required - set(agreement_table.columns))
    if missing:
        raise KeyError(f"agreement_table is missing required columns: {missing}")

    method_labels = {
        "spearman": r"Spearman $\rho$",
        "pearson": r"Pearson $r$",
        "qwk": r"QWK $\kappa_w$",
    }
    level_labels = {"model": "Model", "essay": "Essay"}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{llrrlrrr}",
        r"\toprule",
        (
            r"Level & Judge pair & $n$ & Model clusters & Measure & "
            r"Coefficient & 95\% CI & $p$ \\"
        ),
        r"\midrule",
    ]

    previous_group = None
    for row in agreement_table.itertuples(index=False):
        group = (row.level, row.judge_a, row.judge_b)
        if previous_group is not None and group != previous_group:
            lines.append(r"\addlinespace")
        previous_group = group

        level = level_labels.get(row.level, _latex_escape(row.level))
        pair = f"{_latex_escape(row.judge_a)}--{_latex_escape(row.judge_b)}"
        clusters = "--" if row.level == "model" else str(int(row.n_clusters))
        method = method_labels.get(row.method, _latex_escape(row.method))
        coefficient = (
            "--"
            if pd.isna(row.coefficient)
            else f"{row.coefficient:.{digits}f}"
        )
        confidence_interval = (
            "--"
            if pd.isna(row.ci_low) or pd.isna(row.ci_high)
            else (
                f"$[{row.ci_low:.{digits}f},"
                rf"\,{row.ci_high:.{digits}f}]$"
            )
        )
        p_value = _format_p_value(row.p_value, p_digits)
        lines.append(
            " & ".join(
                [
                    level,
                    pair,
                    str(int(row.n)),
                    clusters,
                    method,
                    coefficient,
                    confidence_interval,
                    p_value,
                ]
            )
            + r" \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    latex = "\n".join(lines)
    if print_latex:
        print(latex)
    return latex


def agreement_figure_to_latex(
    figure_path="judge_agreement.pdf",
    *,
    caption: str = DEFAULT_FIGURE_CAPTION,
    label: str = "fig:judge_agreement",
    width: str = r"\textwidth",
    wide: bool = True,
    print_latex: bool = True,
) -> str:
    """Return a LaTeX figure wrapper with a standalone default caption."""
    environment = "figure*" if wide else "figure"
    safe_path = str(figure_path).replace("\\", "/")
    latex = "\n".join(
        [
            rf"\begin{{{environment}}}[t]",
            r"\centering",
            rf"\includegraphics[width={width}]{{\detokenize{{{safe_path}}}}}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            rf"\end{{{environment}}}",
        ]
    )
    if print_latex:
        print(latex)
    return latex

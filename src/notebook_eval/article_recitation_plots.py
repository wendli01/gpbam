"""Plot essay quality against Article Recitation scores across models.

Call :func:`make_plot` once from the no-RAG notebook and once from the law-RAG
notebook. Each call consumes that notebook's essay ``extended`` summary and the
shared Article Recitation CSV, then draws both legal-knowledge datasets on one
axis with dataset-specific colors and markers.

Model matching deliberately reuses :mod:`essay_correlation`: short endpoint
names and the Qwen FP8/API alias are normalized identically to
``essay_correlation.correlate`` before joining the recitation and essay scores.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from .essay_correlation import recitation_scores, summary_to_numeric
from .essay_plots import (
    DEFAULT_EXCLUDE_MODELS,
    DEFAULT_OUTPUT_DIRECTORY,
    STATS_BOX_LOCATIONS,
    _casefold_set,
    _validate_limits,
    _warn_unknown_models,
)

try:
    from adjustText import adjust_text

    ADJUST_TEXT_AVAILABLE = True
except ImportError:  # optional; fixed-offset labels remain available
    adjust_text = None
    ADJUST_TEXT_AVAILABLE = False


DEFAULT_RECITATION_DATASETS = (
    "GPBam Laws",
    "Most cited Laws",
)

DEFAULT_DATASET_MARKERS = (
    "o",
    "s",
    "^",
    "D",
    "X",
    "P",
)

__all__ = [
    "ADJUST_TEXT_AVAILABLE",
    "DEFAULT_DATASET_MARKERS",
    "DEFAULT_RECITATION_DATASETS",
    "make_plot",
]


def _prepare_essay_scores(summary_df, *, score_col, exclude_models, verbose):
    """Parse essay scores and apply case-insensitive model exclusions."""
    summary_numeric = summary_to_numeric(summary_df)
    if score_col not in summary_numeric.columns:
        raise KeyError(
            f"score_col={score_col!r} not in columns "
            f"{list(summary_numeric.columns)}"
        )

    essay_scores = pd.to_numeric(
        summary_numeric[score_col],
        errors="coerce",
    ).rename("Essay Score")
    essay_scores.index = essay_scores.index.astype(str)
    essay_scores.index.name = "Model"

    available = {model.casefold() for model in essay_scores.index}
    if verbose:
        _warn_unknown_models(exclude_models, available, "exclude_models")

    excluded = _casefold_set(exclude_models)
    essay_scores = essay_scores.loc[
        ~essay_scores.index.str.casefold().isin(excluded)
    ].dropna()

    if essay_scores.empty:
        raise ValueError(
            "No numeric essay scores remain after applying exclude_models."
        )

    return essay_scores


def _match_recitation_scores(
    essay_scores,
    *,
    recitation_csv,
    recitation_datasets,
):
    """Join each requested recitation dataset to the normalized essay models."""
    matched = {}
    for dataset in recitation_datasets:
        recitation = recitation_scores(
            recitation_csv,
            dataset=dataset,
            scale=100.0,
        ).rename("Article Recitation Score")

        dataset_df = (
            essay_scores.to_frame()
            .join(recitation, how="inner")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        dataset_df["Dataset"] = dataset
        matched[dataset] = dataset_df

    return matched


def _dataset_stats(essay_scores, dataset_df, dataset):
    rho = np.nan
    p_value = np.nan
    if (
        len(dataset_df) >= 3
        and dataset_df["Article Recitation Score"].nunique() >= 2
        and dataset_df["Essay Score"].nunique() >= 2
    ):
        correlation = spearmanr(
            dataset_df["Article Recitation Score"],
            dataset_df["Essay Score"],
            nan_policy="omit",
        )
        rho = float(correlation.statistic)
        p_value = float(correlation.pvalue)

    dropped = sorted(set(essay_scores.index).difference(dataset_df.index))
    return {
        "dataset": dataset,
        "n": len(dataset_df),
        "spearman": rho,
        "p_value": p_value,
        "dropped": ", ".join(dropped),
    }


def _auto_limits(values, *, padding_fraction, minimum_padding):
    values = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        raise ValueError("Cannot calculate axis limits from empty data.")
    value_range = values.max() - values.min()
    padding = max(minimum_padding, value_range * padding_fraction)
    return values.min() - padding, values.max() + padding


def _label_rows(plot_data, *, label_position, recitation_datasets):
    """Choose one point per model to carry its direct text label."""
    rows = []
    for _, group in plot_data.groupby("Model", sort=False):
        if label_position == "rightmost":
            rows.append(group.loc[group["Article Recitation Score"].idxmax()])
        elif label_position == "leftmost":
            rows.append(group.loc[group["Article Recitation Score"].idxmin()])
        elif label_position in recitation_datasets:
            selected = group[group["Dataset"] == label_position]
            if not selected.empty:
                rows.append(selected.iloc[0])
        else:
            raise ValueError(
                "label_position must be 'rightmost', 'leftmost', or one of "
                f"the recitation datasets: {list(recitation_datasets)}"
            )
    return rows


def _save_figure(
    fig,
    *,
    output_directory,
    filename,
    output_format,
    save_dpi,
    verbose,
):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{filename}.{output_format}"
    fig.savefig(
        output_path,
        format=output_format,
        dpi=save_dpi,
        bbox_inches="tight",
    )
    if verbose:
        print(f"Saved: {output_path}")
    return output_path


def make_plot(
    summary_df,
    *,
    recitation_csv,
    recitation_datasets=DEFAULT_RECITATION_DATASETS,
    score_col="Score",
    exclude_models=DEFAULT_EXCLUDE_MODELS,
    dataset_labels=None,
    dataset_colors=None,
    dataset_markers=None,
    connect_dataset_points=True,
    label_models=True,
    label_position="rightmost",
    avoid_label_overlap=True,
    show_legend=True,
    legend_location="lower right",
    show_stats_box=True,
    stats_box_location="lower left",
    save_figure=False,
    output_directory=DEFAULT_OUTPUT_DIRECTORY,
    output_format="pdf",
    filename="essay_score_vs_article_recitation",
    figsize=(9, 7),
    figure_dpi=200,
    save_dpi=300,
    show_plot=True,
    title="Essay quality vs. article-recitation performance",
    xlabel="Article Recitation score (%)",
    ylabel="Essay score (%)",
    style="whitegrid",
    context="notebook",
    palette="colorblind",
    point_size=85,
    point_alpha=0.9,
    label_fontsize=6,
    stats_fontsize=8,
    connection_color="0.65",
    connection_alpha=0.35,
    connection_linewidth=0.6,
    x_limits=None,
    y_limits=None,
    axis_padding_fraction=0.08,
    minimum_x_padding=1.0,
    minimum_y_padding=2.0,
    adjust_text_kwargs=None,
    verbose=True,
):
    """Plot essay scores against both Article Recitation datasets.

    Parameters
    ----------
    summary_df:
        Essay ``extended`` DataFrame returned by
        ``essay_evaluation.make_summary``. Its ``Score`` column supplies y.
    recitation_csv:
        Article Recitation result CSV also accepted by
        ``essay_correlation.correlate``. Per-model means supply x.
    recitation_datasets:
        Dataset names to plot. The default contains ``GPBam Laws`` and
        ``Most cited Laws``.
    exclude_models:
        Models removed before matching. Names are normalized exactly as in
        ``essay_correlation.correlate``, so either the spelling shown in the
        summary table or its alias-merged form matches. Names matching no
        model are reported when ``verbose``.
    dataset_labels, dataset_colors, dataset_markers:
        Optional mappings keyed by dataset name. Defaults use the original
        dataset names, a colorblind palette, and distinct marker shapes.
    connect_dataset_points:
        Draw a faint connector between a model's two recitation measurements.
        Because its essay score is shared, this connector is horizontal.
    label_models, label_position, avoid_label_overlap:
        Control direct model labels. One label per model is placed at its
        rightmost point by default; use ``leftmost`` or a dataset name instead.
    show_legend, legend_location:
        Toggle and position the dataset legend.
    show_stats_box, stats_box_location:
        Toggle and position the box containing one Spearman statistic per
        recitation dataset. Statistics are always returned.
    x_limits, y_limits:
        Fixed ``(low, high)`` axis ranges. Left as ``None`` each axis is scaled
        to this notebook's own data, so the same axis spans a different range
        in the no-RAG and law-RAG figures; pass the same values in both to make
        them directly comparable.
    save_figure, output_directory, output_format, filename, save_dpi:
        Optional saving controls.

    Returns
    -------
    dict
        ``figure``, ``axis``, per-dataset ``stats``, long-form ``plot_data``,
        matched ``dataset_data``, normalized ``essay_scores``, and
        ``saved_path``.
    """
    recitation_datasets = tuple(recitation_datasets)
    if not recitation_datasets:
        raise ValueError("recitation_datasets must not be empty.")
    if len(recitation_datasets) != len(set(recitation_datasets)):
        raise ValueError("recitation_datasets must contain unique names.")
    if stats_box_location not in STATS_BOX_LOCATIONS:
        raise ValueError(
            "stats_box_location must be one of: "
            + ", ".join(STATS_BOX_LOCATIONS)
        )
    if figure_dpi <= 0 or save_dpi <= 0:
        raise ValueError("figure_dpi and save_dpi must be positive.")
    if axis_padding_fraction < 0:
        raise ValueError("axis_padding_fraction must be non-negative.")

    essay_scores = _prepare_essay_scores(
        summary_df,
        score_col=score_col,
        exclude_models=exclude_models,
        verbose=verbose,
    )
    dataset_data = _match_recitation_scores(
        essay_scores,
        recitation_csv=recitation_csv,
        recitation_datasets=recitation_datasets,
    )

    empty_datasets = [
        dataset
        for dataset, data in dataset_data.items()
        if data.empty
    ]
    if empty_datasets:
        raise ValueError(
            "No matched essay/recitation models for datasets: "
            + ", ".join(empty_datasets)
        )

    stats_rows = [
        _dataset_stats(essay_scores, dataset_data[dataset], dataset)
        for dataset in recitation_datasets
    ]
    stats_df = pd.DataFrame(stats_rows).set_index("dataset")

    long_parts = []
    for dataset in recitation_datasets:
        part = dataset_data[dataset].reset_index()
        long_parts.append(part)
    plot_data = pd.concat(long_parts, ignore_index=True)

    dataset_labels = {
        dataset: (dataset_labels or {}).get(dataset, dataset)
        for dataset in recitation_datasets
    }

    if dataset_colors is None:
        colors = sns.color_palette(palette, n_colors=len(recitation_datasets))
        dataset_colors = dict(zip(recitation_datasets, colors))
    else:
        missing = set(recitation_datasets).difference(dataset_colors)
        if missing:
            raise KeyError(f"dataset_colors is missing: {sorted(missing)}")

    if dataset_markers is None:
        if len(recitation_datasets) > len(DEFAULT_DATASET_MARKERS):
            raise ValueError(
                "Provide dataset_markers when plotting more than "
                f"{len(DEFAULT_DATASET_MARKERS)} datasets."
            )
        dataset_markers = dict(
            zip(recitation_datasets, DEFAULT_DATASET_MARKERS)
        )
    else:
        missing = set(recitation_datasets).difference(dataset_markers)
        if missing:
            raise KeyError(f"dataset_markers is missing: {sorted(missing)}")

    if label_models and avoid_label_overlap and not ADJUST_TEXT_AVAILABLE:
        print(
            "Note: adjustText is not installed. Labels will use fixed offsets. "
            "Install `adjustText` for automatic overlap reduction."
        )

    sns.set_theme(style=style, context=context)
    fig, ax = plt.subplots(figsize=figsize, dpi=figure_dpi)

    if connect_dataset_points:
        for _, group in plot_data.groupby("Model", sort=False):
            if len(group) >= 2:
                ordered = group.sort_values("Article Recitation Score")
                ax.plot(
                    ordered["Article Recitation Score"],
                    ordered["Essay Score"],
                    color=connection_color,
                    alpha=connection_alpha,
                    linewidth=connection_linewidth,
                    zorder=1,
                )

    for dataset in recitation_datasets:
        data = dataset_data[dataset]
        ax.scatter(
            data["Article Recitation Score"],
            data["Essay Score"],
            s=point_size,
            color=dataset_colors[dataset],
            marker=dataset_markers[dataset],
            edgecolor="white",
            linewidth=0.7,
            alpha=point_alpha,
            label=dataset_labels[dataset],
            zorder=3,
        )

    labels = []
    if label_models:
        label_rows = _label_rows(
            plot_data,
            label_position=label_position,
            recitation_datasets=recitation_datasets,
        )
        for row in label_rows:
            if avoid_label_overlap and ADJUST_TEXT_AVAILABLE:
                labels.append(
                    ax.text(
                        row["Article Recitation Score"],
                        row["Essay Score"],
                        row["Model"],
                        fontsize=label_fontsize,
                        alpha=0.9,
                        zorder=4,
                    )
                )
            else:
                ax.annotate(
                    row["Model"],
                    xy=(
                        row["Article Recitation Score"],
                        row["Essay Score"],
                    ),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=label_fontsize,
                    alpha=0.85,
                    zorder=4,
                )

    if (
        label_models
        and avoid_label_overlap
        and ADJUST_TEXT_AVAILABLE
        and labels
    ):
        default_adjust_text_kwargs = {
            "expand": (1.08, 1.18),
            "force_text": (0.25, 0.4),
            "force_static": (0.15, 0.25),
            "arrowprops": {
                "arrowstyle": "-",
                "color": "0.55",
                "linewidth": 0.4,
                "alpha": 0.7,
                "shrinkA": 10,
                "shrinkB": 3,
            },
        }
        default_adjust_text_kwargs.update(adjust_text_kwargs or {})
        adjust_text(labels, ax=ax, **default_adjust_text_kwargs)

    if show_stats_box:
        stats_x, stats_y, stats_ha, stats_va = STATS_BOX_LOCATIONS[
            stats_box_location
        ]
        stats_lines = []
        for dataset in recitation_datasets:
            row = stats_df.loc[dataset]
            stats_lines.append(
                f"{dataset_labels[dataset]}: "
                rf"$\rho_s$ = {row['spearman']:.2f}, "
                f"n = {int(row['n'])}, p = {row['p_value']:.3g}"
            )
        ax.text(
            stats_x,
            stats_y,
            "\n".join(stats_lines),
            transform=ax.transAxes,
            ha=stats_ha,
            va=stats_va,
            fontsize=stats_fontsize,
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "edgecolor": "0.8",
                "alpha": 0.9,
            },
            zorder=5,
        )

    if show_legend:
        ax.legend(loc=legend_location, title="Recitation dataset")

    # Explicit limits win; otherwise each axis is scaled to this notebook's own
    # data, which is why two experiments otherwise end up with different ranges.
    x_limits = (
        _validate_limits(x_limits, "x_limits")
        if x_limits is not None
        else _auto_limits(
            plot_data["Article Recitation Score"],
            padding_fraction=axis_padding_fraction,
            minimum_padding=minimum_x_padding,
        )
    )
    y_limits = (
        _validate_limits(y_limits, "y_limits")
        if y_limits is not None
        else _auto_limits(
            plot_data["Essay Score"],
            padding_fraction=axis_padding_fraction,
            minimum_padding=minimum_y_padding,
        )
    )

    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    saved_path = None
    if save_figure:
        saved_path = _save_figure(
            fig,
            output_directory=output_directory,
            filename=filename,
            output_format=output_format,
            save_dpi=save_dpi,
            verbose=verbose,
        )

    if verbose:
        print(stats_df.to_string())
    if show_plot:
        plt.show()

    return {
        "figure": fig,
        "axis": ax,
        "stats": stats_df,
        "plot_data": plot_data,
        "dataset_data": dataset_data,
        "essay_scores": essay_scores,
        "saved_path": saved_path,
    }

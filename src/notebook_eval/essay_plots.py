"""Reusable quality-vs-metric plots for essay-evaluation notebooks.

The public ``make_plots`` function consumes the presentation-ready DataFrame
returned by :func:`essay_evaluation.make_summary` and creates three standalone
scatter plots plus one combined panel. All configuration that previously lived
as notebook-level constants is exposed as a keyword-only argument.

Typical use::

    from src.notebook_eval import essay_plots

    result = essay_plots.make_plots(
        extended,
        combined_title="Essay quality relationships — no retrieval",
        save_figures=False,
    )

    result["stats"]
    result["figures"]["combined"]
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from .essay_correlation import _short, summary_to_numeric

try:
    from adjustText import adjust_text

    ADJUST_TEXT_AVAILABLE = True
except ImportError:  # optional dependency; fixed-offset labels remain available
    adjust_text = None
    ADJUST_TEXT_AVAILABLE = False


COL_SCORE = "Score"
COL_ANSWER_TOKENS = "Answer Tokens (k)"
COL_GEN_COST = r"Gen. Cost (\$)"
COL_LEGAL_REF_SIM = "Legal Ref. Sim."

DEFAULT_EXCLUDE_MODELS = (
    "mistral-small-3.1-24b-instruct",
    # The checkpoint NHR@FAU retired on 2026-08-01; the -0731 re-generation is
    # reported in its place. Matching is exact after casefolding, so this never
    # catches "DeepSeek-V4-Flash-0731".
    "DeepSeek-V4-Flash",
    "EuroLLM-22B-Instruct-2512",
    "Phi-4-mini-instruct",
    "gemma-4-31B-it-FP8-block",
)

# The FP8 endpoint was free/self-hosted in one experiment while the API alias
# was paid in the other, so this model is dropped from generation-cost plots to
# keep them comparable. Only the normalised name is listed: by the time the
# exclusion runs, ``summary_to_numeric`` has already merged
# "Qwen3.6-35B-A3B-FP8" into "qwen3.6-35b-a3b", so the FP8 spelling can never
# match a row.
DEFAULT_EXCLUDE_FROM_COST_PLOT = ("qwen3.6-35b-a3b",)

DEFAULT_OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "figures"
    / "essay_quality"
)

DEFAULT_PANELS = (
    {
        "x": COL_ANSWER_TOKENS,
        "xlabel": "Mean answer length (thousand tokens)",
        "title": "Quality vs. answer length",
        "filename": "quality_vs_answer_length",
    },
    {
        "x": COL_GEN_COST,
        "xlabel": "Total generation cost (USD)",
        "title": "Quality vs. generation cost",
        "filename": "quality_vs_generation_cost",
    },
    {
        "x": COL_LEGAL_REF_SIM,
        "xlabel": "Legal reference similarity (%)",
        "title": "Quality vs. legal-reference similarity",
        "filename": "quality_vs_legal_reference_similarity",
    },
)

STATS_BOX_LOCATIONS = {
    "upper left": (0.04, 0.96, "left", "top"),
    "upper right": (0.96, 0.96, "right", "top"),
    "lower left": (0.04, 0.04, "left", "bottom"),
    "lower right": (0.96, 0.04, "right", "bottom"),
}

__all__ = [
    "ADJUST_TEXT_AVAILABLE",
    "COL_ANSWER_TOKENS",
    "COL_GEN_COST",
    "COL_LEGAL_REF_SIM",
    "COL_SCORE",
    "DEFAULT_EXCLUDE_FROM_COST_PLOT",
    "DEFAULT_EXCLUDE_MODELS",
    "DEFAULT_OUTPUT_DIRECTORY",
    "DEFAULT_PANELS",
    "STATS_BOX_LOCATIONS",
    "make_plots",
]


def _casefold_set(values):
    """Normalised, case-insensitive model-name set, accepting ``None`` as empty.

    Names go through ``_short`` first, so an exclusion may be spelled the way
    the summary table displays it ("Qwen3.6-35B-A3B-FP8"), the way
    ``summary_to_numeric`` renames it ("qwen3.6-35b-a3b"), or with the endpoint
    prefix still attached ("Qwen/Qwen3.6-35B-A3B-FP8"). Without this, an
    exclusion copied straight out of the displayed table would silently match
    nothing for any model covered by ``MODEL_ALIASES``.
    """
    return {_short(str(value)).casefold() for value in (values or ())}


def _warn_unknown_models(names, available, parameter):
    """Report exclusion names that match no model, so a typo cannot pass unseen.

    ``available`` holds normalised names, so each candidate is normalised the
    same way before being looked up; the name is reported back as the caller
    spelled it.
    """
    unknown = sorted(
        str(name)
        for name in (names or ())
        if _short(str(name)).casefold() not in available
    )
    if unknown:
        print(f"Warning: these {parameter} entries matched no model:")
        for name in unknown:
            print(f"  - {name}")


def _prepare_plot_df(
    summary_df,
    exclude_models,
    exclude_from_cost_plot,
    *,
    verbose,
):
    """Parse the summary and apply the global model exclusions.

    Both exclusion lists are checked against the full model set here -- before
    anything is dropped -- so an unmatched name is reported whichever list it
    came from.
    """
    plot_df = summary_to_numeric(summary_df).copy()
    plot_df.index = plot_df.index.astype(str)
    plot_df.index.name = summary_df.index.name or "Model"

    available = {model.casefold() for model in plot_df.index}
    if verbose:
        _warn_unknown_models(exclude_models, available, "exclude_models")
        _warn_unknown_models(
            exclude_from_cost_plot, available, "exclude_from_cost_plot"
        )

    excluded = _casefold_set(exclude_models)
    plot_df = plot_df.loc[
        ~plot_df.index.str.casefold().isin(excluded)
    ].copy()

    if plot_df.empty:
        raise ValueError("No models remain after applying exclude_models.")

    if verbose:
        print(f"Models available for plotting: {len(plot_df)}")
        print(plot_df.index.tolist())

    return plot_df


def _validate_panels(plot_df, panels):
    """Validate panel definitions and required summary columns."""
    if not panels:
        raise ValueError("panels must contain at least one panel definition.")

    required_panel_keys = {"x", "xlabel", "title", "filename"}
    for number, panel in enumerate(panels, start=1):
        missing_keys = required_panel_keys.difference(panel)
        if missing_keys:
            raise KeyError(
                f"Panel {number} is missing keys: {sorted(missing_keys)}"
            )

    required_columns = {COL_SCORE, *(panel["x"] for panel in panels)}
    missing_columns = required_columns.difference(plot_df.columns)
    if missing_columns:
        raise KeyError(
            "The following required columns are missing from the summary: "
            + ", ".join(sorted(missing_columns))
        )

    filenames = [panel["filename"] for panel in panels]
    if len(filenames) != len(set(filenames)):
        raise ValueError("Every panel must have a unique filename.")


def _validate_limits(limits, label):
    """Coerce one user-supplied ``(low, high)`` pair to floats, or raise."""
    try:
        low, high = limits
        low, high = float(low), float(high)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{label} must be a (low, high) pair of numbers, got {limits!r}."
        ) from error
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError(f"{label} must be finite, got {limits!r}.")
    if low >= high:
        raise ValueError(
            f"{label} must be increasing, got low={low} and high={high}."
        )
    return low, high


def _resolve_x_limits(x_limits, panels):
    """Map user x limits onto panels, keyed by panel filename or x column.

    Returns ``{filename: (low, high)}`` covering only the panels the caller
    supplied limits for; the rest keep matplotlib's autoscaling.
    """
    if x_limits is None:
        return {}
    if not hasattr(x_limits, "items"):
        raise TypeError(
            "x_limits must be a mapping keyed by panel filename or x column, "
            f"e.g. {{'{panels[0]['filename']}': (0, 22)}}; got {x_limits!r}."
        )

    by_filename = {panel["filename"]: panel["filename"] for panel in panels}
    by_column = {panel["x"]: panel["filename"] for panel in panels}

    resolved = {}
    for key, limits in x_limits.items():
        filename = by_filename.get(key, by_column.get(key))
        if filename is None:
            raise KeyError(
                f"x_limits key {key!r} matches no panel. Valid keys: "
                + ", ".join(sorted({*by_filename, *by_column}))
            )
        resolved[filename] = _validate_limits(limits, f"x_limits[{key!r}]")
    return resolved


def _score_limits(plot_df, *, score_padding_fraction, minimum_score_padding):
    score_values = (
        plot_df[COL_SCORE]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if score_values.empty:
        raise ValueError("The Score column has no numeric observations.")

    score_range = score_values.max() - score_values.min()
    padding = max(minimum_score_padding, score_range * score_padding_fraction)
    return score_values.min() - padding, score_values.max() + padding


def _panel_data(
    plot_df,
    x_col,
    *,
    exclude_from_cost_plot,
    remove_nonpositive_costs,
):
    """Return the observations that should appear in one panel."""
    panel_df = (
        plot_df[[x_col, COL_SCORE]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if x_col == COL_GEN_COST:
        if remove_nonpositive_costs:
            panel_df = panel_df.loc[panel_df[x_col] > 0].copy()

        excluded_cost_models = _casefold_set(exclude_from_cost_plot)
        panel_df = panel_df.loc[
            ~panel_df.index.str.casefold().isin(excluded_cost_models)
        ].copy()

    return panel_df


def _panel_stats(plot_df, panel_df, panel):
    """Statistics for exactly the observations shown in a panel."""
    x_col = panel["x"]
    rho = np.nan
    p_value = np.nan
    if (
        len(panel_df) >= 3
        and panel_df[x_col].nunique() >= 2
        and panel_df[COL_SCORE].nunique() >= 2
    ):
        correlation = spearmanr(
            panel_df[x_col],
            panel_df[COL_SCORE],
            nan_policy="omit",
        )
        rho = float(correlation.statistic)
        p_value = float(correlation.pvalue)

    dropped = sorted(set(plot_df.index).difference(panel_df.index))
    return {
        "panel": panel["filename"],
        "predictor": x_col,
        "n": len(panel_df),
        "spearman": rho,
        "p_value": p_value,
        "dropped": ", ".join(dropped),
    }


def _draw_quality_panel(
    ax,
    panel,
    panel_df,
    stats,
    *,
    model_colors,
    score_limits,
    show_ylabel,
    x_limits,
    label_models,
    avoid_label_overlap,
    use_log_scale_for_cost,
    point_size,
    point_alpha,
    label_fontsize,
    correlation_fontsize,
    show_stats_box,
    stats_box_location,
    adjust_text_kwargs,
):
    """Draw one configured panel on an existing axis."""
    x_col = panel["x"]
    labels = []

    for model, row in panel_df.iterrows():
        x_value = row[x_col]
        y_value = row[COL_SCORE]

        ax.scatter(
            x_value,
            y_value,
            s=point_size,
            color=model_colors[model],
            edgecolor="white",
            linewidth=0.6,
            alpha=point_alpha,
            zorder=3,
        )

        if label_models:
            if avoid_label_overlap and ADJUST_TEXT_AVAILABLE:
                labels.append(
                    ax.text(
                        x_value,
                        y_value,
                        model,
                        fontsize=label_fontsize,
                        alpha=0.9,
                        zorder=4,
                    )
                )
            else:
                ax.annotate(
                    model,
                    xy=(x_value, y_value),
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

    if show_stats_box and stats["n"] >= 3:
        stats_x, stats_y, stats_ha, stats_va = STATS_BOX_LOCATIONS[
            stats_box_location
        ]
        correlation_text = (
            rf"$\rho_s$ = {stats['spearman']:.2f}"
            f"\nn = {stats['n']}"
            f"\np = {stats['p_value']:.3g}"
        )
        ax.text(
            stats_x,
            stats_y,
            correlation_text,
            transform=ax.transAxes,
            ha=stats_ha,
            va=stats_va,
            fontsize=correlation_fontsize,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "0.8",
                "alpha": 0.9,
            },
            zorder=5,
        )

    ax.set_title(panel["title"], fontsize=12, pad=10)
    ax.set_xlabel(panel["xlabel"], fontsize=10)
    ax.set_ylabel("Essay score (%)" if show_ylabel else "", fontsize=10)
    ax.set_ylim(score_limits)
    ax.grid(True, alpha=0.25)

    if x_col == COL_GEN_COST and use_log_scale_for_cost:
        if not panel_df.empty and (panel_df[x_col] > 0).all():
            ax.set_xscale("log")
            ax.set_xlabel("Total generation cost (USD, log scale)", fontsize=10)
        else:
            print(
                "Cost axis was not changed to log scale because the displayed "
                "costs contain zero or negative values."
            )

    # Applied after the scale change so a log axis keeps the requested range.
    if x_limits is not None:
        if ax.get_xscale() == "log" and x_limits[0] <= 0:
            print(
                f"x limits {x_limits} were not applied to the cost panel "
                "because a log axis cannot start at or below zero."
            )
        else:
            ax.set_xlim(x_limits)


def _save_figure(
    fig,
    filename,
    *,
    output_directory,
    output_format,
    save_dpi,
    verbose,
):
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


def make_plots(
    summary_df,
    *,
    exclude_models=DEFAULT_EXCLUDE_MODELS,
    exclude_from_cost_plot=DEFAULT_EXCLUDE_FROM_COST_PLOT,
    label_models=True,
    avoid_label_overlap=True,
    use_log_scale_for_cost=False,
    remove_nonpositive_costs=True,
    save_figures=False,
    output_directory=DEFAULT_OUTPUT_DIRECTORY,
    output_format="pdf",
    individual_figsize=(9, 7),
    combined_figsize=(19, 6.5),
    figure_dpi=200,
    save_dpi=300,
    make_individual_plots=True,
    make_combined_plot=True,
    show_plots=True,
    combined_title="Essay quality relationships by model",
    combined_title_y=1.02,
    combined_filename="essay_quality_relationships_combined",
    panels=None,
    style="whitegrid",
    context="notebook",
    palette="husl",
    point_size=80,
    point_alpha=0.9,
    label_fontsize=6,
    correlation_fontsize=9,
    show_stats_box=True,
    stats_box_location="lower left",
    x_limits=None,
    y_limits=None,
    score_padding_fraction=0.08,
    minimum_score_padding=2.0,
    adjust_text_kwargs=None,
    verbose=True,
):
    """Create standalone and combined essay-quality scatter plots.

    Parameters
    ----------
    summary_df:
        Per-model DataFrame returned by ``essay_evaluation.make_summary``.
        LaTeX-formatted cells such as ``mean +/- error`` are parsed using
        ``essay_correlation.summary_to_numeric``.
    exclude_models:
        Model names removed from every panel. Matching is case-insensitive and
        uses the normalised name (``summary_to_numeric`` strips the endpoint
        prefix and merges the Qwen FP8/API alias). Names matching no model are
        reported when ``verbose``.
    exclude_from_cost_plot:
        Additional models removed only from the generation-cost panel, useful
        when free/self-hosted and paid endpoints are not cost-comparable.
        Matched and reported exactly like ``exclude_models``.
    label_models, avoid_label_overlap:
        Control point labels. ``adjustText`` is used when installed; otherwise
        labels fall back to fixed offsets without breaking the plot.
    use_log_scale_for_cost:
        Put generation cost on a log axis after filtering invalid values.
    remove_nonpositive_costs:
        Treat zero and negative generation costs as unavailable.
    save_figures, output_directory, output_format, save_dpi:
        Optional saving controls. Every created figure is saved when enabled.
    make_individual_plots, make_combined_plot:
        Control which of the three standalone figures and combined panel are
        created. Both default to true, producing four figures in total.
    show_plots:
        Call ``plt.show()`` after creating and optionally saving the figures.
    panels:
        Optional sequence of mappings with ``x``, ``xlabel``, ``title``, and
        ``filename``. ``None`` selects answer length, generation cost, and
        legal reference similarity; an empty sequence is an error.
    palette:
        Seaborn palette for the per-model point colors. Colors are assigned by
        sorted model name, so a model keeps its color across notebooks that
        plot the same set of models even though their Score ordering differs.
    x_limits, y_limits:
        Fixed axis ranges. ``y_limits`` is one ``(low, high)`` pair shared by
        every panel; ``x_limits`` is a mapping keyed by panel filename or by
        the panel's x column, holding one pair per panel, and panels left out
        keep autoscaling. Pass the same values in two notebooks to make their
        figures directly comparable -- by default each notebook scales to its
        own data, so the same axis spans different ranges in each. A log cost
        axis ignores an x range starting at or below zero, and says so.
    stats_box_location:
        Location of the Spearman/n/p box: ``upper left``, ``upper right``,
        ``lower left``, or ``lower right``.
    show_stats_box:
        Show the Spearman/n/p annotation box. Statistics are still calculated
        and returned in ``result["stats"]`` when this is false.

    Returns
    -------
    dict
        ``figures`` and ``axes`` keyed by panel filename plus ``combined``;
        ``stats`` as a per-panel DataFrame; ``plot_df`` and ``panel_data`` for
        inspection; and ``saved_paths`` for files actually written.
    """
    if not make_individual_plots and not make_combined_plot:
        raise ValueError(
            "Enable make_individual_plots, make_combined_plot, or both."
        )
    if figure_dpi <= 0 or save_dpi <= 0:
        raise ValueError("figure_dpi and save_dpi must be positive.")
    if score_padding_fraction < 0 or minimum_score_padding < 0:
        raise ValueError("Score padding values must be non-negative.")
    if stats_box_location not in STATS_BOX_LOCATIONS:
        raise ValueError(
            "stats_box_location must be one of: "
            + ", ".join(STATS_BOX_LOCATIONS)
        )

    # `None` means "use the defaults"; an empty sequence is a mistake and is
    # rejected by _validate_panels rather than silently redrawing the defaults.
    panels = tuple(
        dict(panel)
        for panel in (DEFAULT_PANELS if panels is None else panels)
    )
    plot_df = _prepare_plot_df(
        summary_df,
        exclude_models,
        exclude_from_cost_plot,
        verbose=verbose,
    )
    _validate_panels(plot_df, panels)

    if label_models and avoid_label_overlap and not ADJUST_TEXT_AVAILABLE:
        print(
            "Note: adjustText is not installed. Labels will use fixed offsets. "
            "Install `adjustText` for automatic overlap reduction."
        )

    sns.set_theme(style=style, context=context)
    # Assign colors by sorted model name, not by row order. Rows arrive sorted
    # by Score, which differs between experiments, so keying on position would
    # give the same model a different color in the no-RAG and law-RAG figures.
    model_order = sorted(plot_df.index, key=str.casefold)
    colors = sns.color_palette(palette, n_colors=len(model_order))
    model_colors = dict(zip(model_order, colors))
    # Explicit limits win; otherwise each axis is scaled to this notebook's own
    # data, which is why two experiments otherwise end up with different ranges.
    score_limits = (
        _validate_limits(y_limits, "y_limits")
        if y_limits is not None
        else _score_limits(
            plot_df,
            score_padding_fraction=score_padding_fraction,
            minimum_score_padding=minimum_score_padding,
        )
    )
    panel_x_limits = _resolve_x_limits(x_limits, panels)

    panel_data = {}
    stats_rows = []
    for panel in panels:
        data = _panel_data(
            plot_df,
            panel["x"],
            exclude_from_cost_plot=exclude_from_cost_plot,
            remove_nonpositive_costs=remove_nonpositive_costs,
        )
        panel_data[panel["filename"]] = data
        stats_rows.append(_panel_stats(plot_df, data, panel))

    stats_df = pd.DataFrame(stats_rows).set_index("panel")
    stats_by_panel = {
        row["panel"]: row
        for row in stats_rows
    }

    figures = {}
    axes = {}
    saved_paths = {}
    output_directory = Path(output_directory)

    if make_individual_plots:
        for panel in panels:
            key = panel["filename"]
            fig, ax = plt.subplots(figsize=individual_figsize, dpi=figure_dpi)
            _draw_quality_panel(
                ax,
                panel,
                panel_data[key],
                stats_by_panel[key],
                model_colors=model_colors,
                score_limits=score_limits,
                show_ylabel=True,
                x_limits=panel_x_limits.get(key),
                label_models=label_models,
                avoid_label_overlap=avoid_label_overlap,
                use_log_scale_for_cost=use_log_scale_for_cost,
                point_size=point_size,
                point_alpha=point_alpha,
                label_fontsize=label_fontsize,
                correlation_fontsize=correlation_fontsize,
                show_stats_box=show_stats_box,
                stats_box_location=stats_box_location,
                adjust_text_kwargs=adjust_text_kwargs,
            )
            fig.tight_layout()
            figures[key] = fig
            axes[key] = ax

            if save_figures:
                saved_paths[key] = _save_figure(
                    fig,
                    key,
                    output_directory=output_directory,
                    output_format=output_format,
                    save_dpi=save_dpi,
                    verbose=verbose,
                )

    if make_combined_plot:
        fig, combined_axes = plt.subplots(
            nrows=1,
            ncols=len(panels),
            figsize=combined_figsize,
            dpi=figure_dpi,
            sharey=True,
            squeeze=False,
        )
        combined_axes = combined_axes.ravel()

        for panel_number, (ax, panel) in enumerate(zip(combined_axes, panels)):
            key = panel["filename"]
            _draw_quality_panel(
                ax,
                panel,
                panel_data[key],
                stats_by_panel[key],
                model_colors=model_colors,
                score_limits=score_limits,
                show_ylabel=panel_number == 0,
                x_limits=panel_x_limits.get(key),
                label_models=label_models,
                avoid_label_overlap=avoid_label_overlap,
                use_log_scale_for_cost=use_log_scale_for_cost,
                point_size=point_size,
                point_alpha=point_alpha,
                label_fontsize=label_fontsize,
                correlation_fontsize=correlation_fontsize,
                show_stats_box=show_stats_box,
                stats_box_location=stats_box_location,
                adjust_text_kwargs=adjust_text_kwargs,
            )

        if combined_title:
            fig.suptitle(combined_title, fontsize=15, y=combined_title_y)
        fig.tight_layout()
        figures["combined"] = fig
        axes["combined"] = combined_axes

        if save_figures:
            saved_paths["combined"] = _save_figure(
                fig,
                combined_filename,
                output_directory=output_directory,
                output_format=output_format,
                save_dpi=save_dpi,
                verbose=verbose,
            )

    if show_plots:
        plt.show()

    return {
        "figures": figures,
        "axes": axes,
        "stats": stats_df,
        "plot_df": plot_df,
        "panel_data": panel_data,
        "saved_paths": saved_paths,
    }

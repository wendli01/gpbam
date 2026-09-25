"""Essay quality against verbatim statute recall, on both recitation datasets.

One dumbbell per model: essay Score on the vertical axis, and the model's two
Article Recitation scores -- ``GPBam Laws`` (the statutes the exam cases
actually cite) and ``Most cited Laws`` (the 100 most-cited German norms) -- as
the two ends of a horizontal segment. Both ends share a y value, so the segment
length is the gap between what a model recalls of the benchmark's own statutes
and of the common ones, and the whole figure reads as one relationship measured
two ways.

Run from ``experiments/``::

    python analysis/plot_recitation_vs_essay.py

Writes ``analysis/out/recitation_vs_essay_norag.{pdf,png}`` and the numbers
behind the figure to ``analysis/out/recitation_vs_essay_norag.csv``.

Method
------
Essay Score is the no-RAG score of :mod:`plot_cost_performance` -- per-essay
median across the judges, x100, meaned over the 81 cases -- so this figure and
the price/performance one always report the same y. Recitation score is the mean
ROUGE-L F1 over the 100 articles of a dataset, x100. Models are matched on the
last path segment of the identifier, lower-cased, with the FP8/quantisation
suffixes stripped: the recitation run used the API alias
``qwen/qwen3.6-35b-a3b`` where the essay run used the self-hosted
``Qwen/Qwen3.6-35B-A3B-FP8``.

A model with essays but no recitation run is named on stdout and left off.
The agent-harness row (Claude-Opus-5) is drawn hollow: see AGENT_HARNESS.

Spearman rho and its percentile bootstrap CI (10,000 resamples of model pairs,
seed 0) follow ``src/notebook_eval/essay_correlation``, so the coefficients here
are the ones in the correlation table.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from figure_common import _place_labels, apply_theme, boot_ci
from plot_cost_performance import RETIRED, build_table

RECITATION = Path("zubaers_result/article_recitation/article_recitation.csv")
OUT = Path("analysis/out")

# dataset -> (colour, label). "GPBam Laws" is the primary one: it is the set the
# essays are actually graded against, and the one the correlation table reports.
DATASETS = {
    "GPBam Laws": ("#2F6B57", "GPBam statutes"),
    "Most cited Laws": ("#C08A2E", "Most-cited statutes"),
}

MARKER_AREA = 85.0

# Rows produced by the Claude Code agent harness rather than by an API call
# through AnswerGenerator. Drawn hollow, exactly as plot_cost_performance draws
# its extrapolated row, because both axes come from that harness: the essays are
# the 48-case agent run and the recitations the 200-prompt agent run. Nothing
# here is extrapolated -- recitation is full coverage -- but the point is not
# effort-matched with the rest, and an unmarked dot in the top-right corner
# would claim more than the run supports. See
# analysis/additional_models/opus5_agent/README.md.
AGENT_HARNESS = {"Claude-Opus-5"}


def _key(model):
    """Match key: last path segment, lower-case, quantisation suffixes dropped."""
    s = model.split("/")[-1].lower().replace("_", "-")
    for suffix in ("-fp8-dynamic", "-fp8-block", "-fp8"):
        s = s.replace(suffix, "")
    return s.replace("-a3b", "")


def build_data(recitation=RECITATION):
    """Per-model essay Score plus one recitation score per dataset."""
    # The main table's model set, not the cost figure's. build_table defaults to
    # the latter, which additionally drops three models that have no OpenRouter
    # listing -- a price question, irrelevant here, and it used to make this
    # figure's rho disagree with the associations table (n=29 vs 32). Cost
    # columns go unused, so no price is required.
    essays = build_table(exclude=RETIRED, require_price=False)
    essays["key"] = essays["model"].map(_key)

    # Models recited after the published sweep live in sibling recitation_*.csv
    # files rather than in the main CSV -- the 0731 re-generation, and the two
    # FAU-served models that were never recited at all. Same 200 prompts, same
    # ROUGE-L scoring, so they concatenate.
    frames = [pd.read_csv(recitation)]
    for extra in sorted(Path(recitation).parent.glob("recitation_*.csv")):
        e = pd.read_csv(extra, usecols=["model", "score", "dataset"])
        frames.append(e[e["score"].notna()])
    rec = pd.concat(frames, ignore_index=True)
    rec["key"] = rec["model"].map(_key)
    wide = rec.groupby(["key", "dataset"])["score"].mean().unstack() * 100

    missing = set(DATASETS) - set(wide.columns)
    if missing:
        raise KeyError(f"recitation datasets absent from the CSV: {sorted(missing)}")

    out = essays.set_index("key").join(wide[list(DATASETS)], how="inner")
    # A model with essays but no recitation run simply is not on this figure --
    # both axes need both numbers. Named rather than raised on, because the essay
    # table now carries rows that were never sent through the 200 recitation
    # prompts, and dropping them silently would understate n without saying so.
    unmatched = sorted(set(essays["key"]) - set(out.index))
    if unmatched:
        print(f"no recitation run, left off the figure: {', '.join(unmatched)}")
    return out.reset_index()


def _stats(tab):
    """Spearman rho with CI for each dataset, and the agreement between them."""
    y = tab["score"].to_numpy()
    rows = {}
    for name in DATASETS:
        x = tab[name].to_numpy()
        rows[name] = (spearmanr(x, y).statistic, *boot_ci(x, y))
    pair = spearmanr(tab[list(DATASETS)[0]], tab[list(DATASETS)[1]]).statistic
    return rows, pair


def make_figure(tab, stats, path):
    apply_theme()
    rows, pair = stats
    names = list(DATASETS)

    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=200)
    fig.subplots_adjust(left=.115, right=.985, bottom=.145, top=.905)

    # The connector first, so both ends sit on top of it.
    for _, row in tab.iterrows():
        ax.plot([row[names[0]], row[names[1]]], [row["score"]] * 2,
                color="#9AA0A5", linewidth=1.0, alpha=.7, zorder=2)

    for name in names:
        colour, label = DATASETS[name]
        agent = tab["short"].isin(AGENT_HARNESS)
        ax.scatter(tab.loc[~agent, name], tab.loc[~agent, "score"], s=MARKER_AREA,
                   color=colour, marker="o", edgecolor="#22282B", linewidth=.9,
                   alpha=.95, zorder=3, label=label)
        if agent.any():
            ax.scatter(tab.loc[agent, name], tab.loc[agent, "score"], s=MARKER_AREA,
                       facecolor="white", marker="o", edgecolor=colour,
                       linewidth=2.0, alpha=.95, zorder=3)
        # Least-squares trend, fitted in the plotted (log) space and drawn only
        # across the data it was fitted on. Decorative: rho is rank-based and
        # identical either way.
        slope, intercept = np.polyfit(np.log10(tab[name]), tab["score"], 1)
        xs = np.geomspace(tab[name].min(), tab[name].max(), 2)
        ax.plot(xs, slope * np.log10(xs) + intercept, color=colour,
                linestyle="--", linewidth=1.6, alpha=.55, zorder=1)

    # Log x: two thirds of the models sit below 20%, and the top of the range is
    # sparse, so the ticks are hand-placed rather than decade-spaced. They must
    # reach past the largest point -- recitation now runs to ~90 -- or the best
    # models sit beyond the last labelled gridline with nothing to read them
    # against.
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(
        matplotlib.ticker.FixedLocator([8, 10, 15, 20, 30, 50, 70, 90]))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Article Recitation  (ROUGE-L F1, %, log scale)")
    ax.set_ylabel("Essay Score, no retrieval  (%)")
    ax.set_title("Verbatim statute recall predicts essay quality", pad=14)
    ax.grid(True, alpha=.3)
    ax.set_axisbelow(True)
    ax.margins(x=.08, y=.09)

    series_leg = ax.legend(loc="upper left", fontsize=12, frameon=True,
                           framealpha=.92, title="Recitation set",
                           title_fontsize=12)
    ax.add_artist(series_leg)

    lines = [
        rf"Spearman $\rho$ (n={len(tab)})",
        *[f"  {DATASETS[n][1]}: {rows[n][0]:.2f} "
          f"[{rows[n][1]:.2f}, {rows[n][2]:.2f}]" for n in names],
        rf"the two sets agree at $\rho$ = {pair:.2f}",
    ]
    if tab["short"].isin(AGENT_HARNESS).any():
        who = ", ".join(sorted(tab.loc[tab["short"].isin(AGENT_HARNESS), "short"]))
        lines.append(f"hollow: agent harness, not an API run -- {who}")
    stats_leg = ax.legend(
        handles=[Line2D([], [], linestyle="", label=l) for l in lines],
        loc="lower right", fontsize=11, frameon=True, framealpha=.92,
        handlelength=0, handletextpad=0, labelspacing=.35,
    )

    fig.canvas.draw()
    boxes = [leg.get_window_extent(fig.canvas.get_renderer())
             for leg in (series_leg, stats_leg)]
    # The connectors are obstacles too, so no name is struck through by one.
    for _, row in tab.iterrows():
        (x0, y0), (x1, _) = (ax.transData.transform((row[n], row["score"]))
                             for n in names)
        boxes.append(matplotlib.transforms.Bbox([[x0, y0 - 3], [x1, y0 + 3]]))
    # Anchor each name on the middle of its dumbbell; both ends are obstacles.
    mid = tab[names].mean(axis=1)
    crowded = _place_labels(
        ax, mid, tab["score"], tab["short"], np.full(len(tab), MARKER_AREA),
        obstacles=boxes, fontsize=9.5,
        extra_markers=[(row[n], row["score"], MARKER_AREA)
                       for _, row in tab.iterrows() for n in names],
    )

    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return crowded


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tab = build_data()
    stats = _stats(tab)
    cols = ["short", "vendor", "score", *DATASETS]
    tab[cols].to_csv(OUT / "recitation_vs_essay_norag.csv", index=False)
    crowded = make_figure(tab, stats, OUT / "recitation_vs_essay_norag")
    if crowded:
        print(f"note: {crowded} of {len(tab)} names overlap another name at "
              f"this figure size")

    rows, pair = stats
    for name, (rho, lo, hi) in rows.items():
        print(f"{name:16s} rho = {rho:.3f}  [{lo:.2f}, {hi:.2f}]")
    print(f"datasets agree at rho = {pair:.3f}")
    print(f"\nwrote {OUT / 'recitation_vs_essay_norag.pdf'} (+ .png, .csv)")


if __name__ == "__main__":
    main()

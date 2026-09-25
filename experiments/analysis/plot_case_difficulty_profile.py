"""Per-case scores against case difficulty, for the no-RAG essays.

Three figures, all sharing one x axis: the 81 cases ordered from the one the
field scores highest on to the one it scores lowest on, where "the field" is
the mean per-case judge median over every model. They answer a question the
table mean cannot -- whether a model is uniformly strong, or strong on most
cases and absent on a few.

    case_difficulty_profile_norag    the top five, as dots, with trend lines
    case_difficulty_spaghetti_norag  every model, trend lines only
    case_difficulty_refsim_norag     every model, statutory-reference
                                     similarity on the same case ordering

Dots, in the first figure, not lines. The agent-harness Claude-Opus-5 row
covers only part of the 81 cases, and a line would run straight through the
ones it never answered, inventing scores that were never measured. In the
trend-line figures its line is dashed for the same reason. The coverage in
the figure notes is counted from the data, not written down here.

Run from ``experiments/``::

    python analysis/plot_case_difficulty_profile.py

Method
------
The judge y value is the per-case median over the three re-judged seats, x100 --
the statistic the main table averages, so a model's dots average to its table
row. The Opus judging is read from ``opus5_agent/scores.csv``, the live pass;
the copy in the re-judged CSV is an earlier one and is dropped.

Reference similarity is ``refex_now`` from ``legal_ref_recomputed.csv``, the
Jaccard overlap between the statutes an essay cites and those the official
solution cites, recomputed for every row with one extractor. Cases 34, 73 and
74 are NaN throughout -- their solutions have no extractable reference, so
there is nothing to agree with -- and are dropped rather than smoothed over.

The third figure keeps the judge-difficulty ordering and keeps the colour keyed
to the *judge* mean, so both trend-line figures encode the same thing the same
way. Where the colour ordering stops being vertical, the two metrics disagree
about a model.

Case difficulty is the mean over all models, so each model contributes to the
ordering of its own x axis -- real circularity, but one row in thirty-odd. The
downward slope is partly built in for the same reason; what the figures are for
is each line's departure from a steady decline, not the decline itself.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out"
REJUDGED = ROOT / "experiments/zubaers_result/essay_writing/without_rag_0731/no_rag_ji2_rejudged.csv"
OPUS = ROOT / "experiments/analysis/additional_models/opus5_agent/scores.csv"
REFSIM = ROOT / "experiments/analysis/deepseek_rerun/results/legal_ref_recomputed.csv"

N_MODELS = 5
WINDOW = 15          # rolling-mean window, in cases
AGENT_HARNESS = {"claude-opus-5"}
# The checkpoint NHR@FAU retired on 2026-08-01, superseded by -0731 and not
# reported in the table; and the row whose 81 calls all errored.
RETIRED = {"DeepSeek-V4-Flash", "mistral-small-3.1-24b-instruct"}
SHORT = {
    "claude-opus-5": "Claude-Opus-5",
    "gemini-3.7-flash": "Gemini-3.7-Flash",
    "DeepSeek-V4-Flash-0731": "DeepSeek-V4-Flash",
    "qwen3.5-397b-a17b": "Qwen3.5-397B",
    "mistral-large-2512": "Mistral-Large",
    "gpt-oss-120b": "GPT-oss-120B",
    "llama-3.1-8b-instruct": "Llama-3.1-8B",
}


def _short(s):
    return s.str.split("/").str[-1]


def load_judge():
    df = pd.read_csv(REJUDGED)
    seats = [c for c in df.columns if c.startswith("score_")]
    df["med"] = df[seats].median(axis=1) * 100
    df["short"] = _short(df["model"])
    # The published Opus judging here is an earlier pass; take the live one so
    # this figure and the table cannot disagree.
    df = df[~df["short"].str.contains("opus", case=False)]
    df = df[~df["short"].isin(RETIRED)]

    opus = pd.read_csv(OPUS).rename(columns={"median": "med"})
    opus["short"] = "claude-opus-5"
    return pd.concat([df[["index", "short", "med"]],
                      opus[["index", "short", "med"]]], ignore_index=True)


def load_refsim():
    df = pd.read_csv(REFSIM)
    df["short"] = _short(df["model"])
    df = df[~df["short"].isin(RETIRED)]
    df["med"] = df["refex_now"] * 100
    # Cases with no extractable reference in the solution carry no signal.
    return df.dropna(subset=["med"])[["index", "short", "med"]]


def spaghetti(long, rank, colour_by, n_cases, fname, ylabel, title, note):
    """Every model as a trend line, no dots, coloured by `colour_by`."""
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    norm = matplotlib.colors.Normalize(vmin=colour_by.min(), vmax=colour_by.max())
    cmap = plt.get_cmap("viridis")

    long = long.copy()
    long["rank"] = long["index"].map(rank)
    for m in colour_by.index:
        sub = long[long["short"] == m].sort_values("rank")
        if sub.empty:
            continue
        trend = sub["med"].rolling(WINDOW, min_periods=5, center=True).mean()
        partial = m in AGENT_HARNESS
        ax.plot(sub["rank"], trend, color=cmap(norm(colour_by[m])),
                lw=2.4 if partial else 1.5, alpha=.95 if partial else .8,
                ls="--" if partial else "-", zorder=3 if partial else 2)

    for m in list(colour_by.head(3).index) + [colour_by.index[-1]]:
        sub = long[long["short"] == m].sort_values("rank")
        if sub.empty:
            continue
        trend = sub["med"].rolling(WINDOW, min_periods=5, center=True).mean().dropna()
        if trend.empty:
            continue
        ax.annotate(f" {SHORT.get(m, m)}", xy=(sub['rank'].max(), trend.iloc[-1]),
                    va="center", fontsize=7.5, color=cmap(norm(colour_by[m])))

    ax.set_xlabel("Case, ordered by mean judge score across all models  "
                  "(easiest for the field $\\rightarrow$ hardest)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, n_cases * 1.16)
    ax.set_ylim(0, None)
    ax.grid(axis="y", alpha=.25, lw=.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                      pad=.02, fraction=.035)
    cb.set_label("Model mean judge score", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(title, loc="left", fontsize=11)
    ax.annotate(note, xy=(0, -0.16), xycoords="axes fraction",
                fontsize=7.5, color="0.35")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {OUT / fname}.pdf")


def profile(long, rank, order, n_cases, agent_n):
    """The top five, as dots, with trend lines."""
    models = list(order.head(N_MODELS).index)
    long = long[long["short"].isin(models)].copy()
    long["rank"] = long["index"].map(rank)

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    colours = plt.get_cmap("tab10").colors
    lanes = np.linspace(-2.2, 2.2, len(models))
    for i, m in enumerate(models):
        sub = long[long["short"] == m].sort_values("rank")
        colour, hollow = colours[i], m in AGENT_HARNESS
        ax.scatter(sub["rank"], sub["med"] + lanes[i], s=26, alpha=.85, zorder=3,
                   label=f"{SHORT.get(m, m)} ({order[m]:.1f})",
                   facecolor="white" if hollow else colour,
                   edgecolor=colour, linewidth=1.6 if hollow else .6)
        ax.plot(sub["rank"],
                sub["med"].rolling(WINDOW, min_periods=5, center=True).mean(),
                color=colour, lw=2.2, alpha=.55, zorder=2)

    ax.set_xlabel("Case, ordered by mean score across all models  "
                  "(easiest for the field $\\rightarrow$ hardest)")
    ax.set_ylabel("Judge median for the case")
    ax.set_ylim(-8, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlim(0, n_cases + 1)
    ax.grid(axis="y", alpha=.25, lw=.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    leg = ax.legend(title="Model (mean)", frameon=False, fontsize=8,
                    title_fontsize=8, loc="lower left", ncol=2)
    leg._legend_box.align = "left"
    ax.set_title("Per-case scores against case difficulty", loc="left", fontsize=11)
    ax.annotate(f"hollow = agent harness, {agent_n} of {n_cases} cases;  "
                f"lines are a {WINDOW}-case rolling mean over observed cases",
                xy=(0, -0.22), xycoords="axes fraction", fontsize=7.5, color="0.35")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"case_difficulty_profile_norag.{ext}", dpi=200,
                    bbox_inches="tight")
    long[["index", "rank", "short", "med"]].to_csv(
        OUT / "case_difficulty_profile_norag.csv", index=False)
    print(f"wrote {OUT / 'case_difficulty_profile_norag.pdf'}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    judge = load_judge()

    difficulty = judge.groupby("index")["med"].mean().sort_values(ascending=False)
    rank = pd.Series(np.arange(1, len(difficulty) + 1), index=difficulty.index)
    order = judge.groupby("short")["med"].mean().sort_values(ascending=False)
    n = len(difficulty)
    agent_n = int(judge[judge["short"].isin(AGENT_HARNESS)]
                  .groupby("short")["index"].nunique().max())

    spaghetti(judge, rank, order, n, "case_difficulty_spaghetti_norag",
              f"Judge median, {WINDOW}-case rolling mean",
              "Every model against case difficulty",
              f"dashed = agent harness, {agent_n} of {n} cases")
    profile(judge, rank, order, n, agent_n)

    ref = load_refsim()
    # Same ordering, same colours: only the y quantity changes.
    ref_order = order[order.index.isin(ref["short"].unique())]
    spaghetti(ref, rank, ref_order, n, "case_difficulty_refsim_norag",
              f"Legal ref. sim., {WINDOW}-case rolling mean",
              "Statutory-reference similarity, cases ordered by judge difficulty",
              "dashed = agent harness;  colour is the model's judge mean, as above;  "
              "cases 34, 73, 74 dropped (no reference in the solution)")

    both = (judge.rename(columns={"med": "judge"})
            .merge(ref.rename(columns={"med": "ref"}), on=["index", "short"]))
    both.to_csv(OUT / "case_difficulty_refsim_norag.csv", index=False)
    print(f"cases {n}, difficulty {difficulty.min():.1f} to {difficulty.max():.1f}")


if __name__ == "__main__":
    main()

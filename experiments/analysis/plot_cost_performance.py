"""Price/performance scatter for the no-RAG essay-writing table.

One point per model: what the 81 essays cost to generate (x) against the score
they earned (y). Bubble area is verbosity -- the total number of completion
tokens the model produced across the 81 essays -- and colour is the vendor
family. Every point carries its model name, so no per-model legend is needed.

Run from ``experiments/``::

    python analysis/plot_cost_performance.py

Writes ``analysis/out/cost_performance_norag.{pdf,png}`` and the numbers behind
the figure to ``analysis/out/cost_performance_norag.csv``.

Cost
----
``total_cost`` in the result CSV is the provider-reported spend and exists only
for the OpenRouter-served models; the self-hosted ones (NHR@FAU, InnKube, local
vLLM) report nothing, or report a literal $0. For every row without a positive
reported cost the spend is imputed as

    prompt_tokens * price_in + completion_tokens * price_out

with list prices from the OpenRouter model catalogue (``/api/v1/models``,
retrieved 2026-08-17). This prices self-hosted runs at what the same generation
would have cost as an API call, which is the only way to put hosted and
self-hosted models on one axis. Imputed models are marked ``*`` after their
name; the three models with no catalogue entry of their own are priced by the
nearest sibling listed in ``PRICES`` and carry that note in the CSV.

Partial rows
------------
Both axes are corpus totals, so a model with fewer than 81 essays is placed by
its per-essay mean times 81 and drawn hollow, with the shortfall named in the
figure. No row is in that position any more -- ``anthropic/claude-opus-5`` was
the last one and now answers all 81 -- but the machinery stays, because it is
what makes a short row visibly short rather than silently mixed in.

``anthropic/claude-opus-5`` is still special on the x axis for a different
reason: the agent harness reports no usage record at all, so its tokens are
estimated rather than measured (see additional_models/opus5_agent/README.md)
and its price is imputed, which the CSV records and the figure marks.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from figure_common import MARKER, VENDORS, _place_labels, apply_theme, vendor_palette

# Overridable so the figures can be built on a different judge panel without
# forking this module: the three plot_* scripts all chain off build_table(), so
# this one path is the only thing that has to move.
RESULTS = Path(os.environ.get(
    "GPBAM_RESULTS",
    "zubaers_result/essay_writing/without_rag/ji2/no_rag_ji2_result.csv"))
OUT = Path("analysis/out")

# Out of the main table entirely: a broken run and a retired checkpoint. See
# RETIRED in deepseek_rerun/scripts/rebuild_tables.py, which documents why.
# Any figure showing the main table's model set excludes exactly these two.
RETIRED = (
    "mistralai/mistral-small-3.1-24b-instruct",
    "deepseek-ai/DeepSeek-V4-Flash",
)

# Additionally outside the *paired* set this figure plots (see
# src/notebook_eval/essay_plots.py). These three ARE in the main table, so a
# figure that means to show the main table's models must not inherit them --
# pass `exclude=RETIRED`. plot_recitation_vs_essay.py does exactly that; it was
# silently inheriting this list, which is why its rho differed from the
# associations table's at n=29 against n=32.
UNPAIRED = (
    "utter-project/EuroLLM-22B-Instruct-2512",
    "Microsoft/Phi-4-mini-instruct",
    "RedHatAI/gemma-4-31B-it-FP8-block",
)

EXCLUDE = RETIRED + UNPAIRED

# model id -> (short name, vendor family). Mirrors Table `tab:models`.
MODELS = {
    "anthropic/claude-haiku-4.5": ("claude-haiku-4.5", "Anthropic"),
    # In the main table, and so in any figure that shows its model set; not in
    # the paired set this cost figure plots, and with no OpenRouter listing.
    "utter-project/EuroLLM-22B-Instruct-2512": ("EuroLLM-22B", "Other"),
    "Microsoft/Phi-4-mini-instruct": ("Phi-4-mini", "Microsoft"),
    "RedHatAI/gemma-4-31B-it-FP8-block": ("gemma-4-31B", "Google"),
    "deepseek/deepseek-chat-v3-0324": ("deepseek-chat-v3", "DeepSeek"),
    "deepseek/deepseek-r1-0528": ("deepseek-r1", "DeepSeek"),
    "deepseek/deepseek-v3.2": ("deepseek-v3.2", "DeepSeek"),
    "deepseek-ai/DeepSeek-V4-Flash": ("DeepSeek-V4-Flash", "DeepSeek"),
    "deepseek-ai/DeepSeek-V4-Flash-0731": ("DeepSeek-V4-Flash-0731", "DeepSeek"),
    "google/gemini-2.5-flash-lite": ("gemini-2.5-fl-lite", "Google"),
    "google/gemini-3.7-flash": ("gemini-3.7-flash", "Google"),
    "anthropic/claude-opus-5": ("Claude-Opus-5", "Anthropic"),
    "soofi-s-isar-preview": ("Soofi-S-Isar", "Soofi"),
    "ibm-granite/granite-4.1-3b": ("granite-4.1-3b", "IBM"),
    "meta-llama/llama-3.1-8b-instruct": ("llama-3.1-8b", "Meta"),
    "meta-llama/llama-3.3-70b-instruct": ("llama-3.3-70b", "Meta"),
    "meta-llama/llama-4-maverick": ("llama-4-maverick", "Meta"),
    "GaleneAI/Magistral-Small-2509-FP8-Dynamic": ("Magistral-Small", "Mistral"),
    "mistralai/Ministral-3-14B-Reasoning-2512": ("Ministral-3-14B-R", "Mistral"),
    "RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8": ("Mistral-Small-3.2", "Mistral"),
    "mistralai/Mistral-Medium-3.5-128B": ("Mistral-Med-3.5", "Mistral"),
    "mistralai/mistral-large-2512": ("mistral-large", "Mistral"),
    "openai/gpt-4o-mini-2024-07-18": ("gpt-4o-mini", "OpenAI"),
    "openai/gpt-5-nano": ("gpt-5-nano", "OpenAI"),
    "openai/gpt-5-mini": ("gpt-5-mini", "OpenAI"),
    "openai/gpt-oss-120b": ("gpt-oss-120b", "OpenAI"),
    "qwen/qwen3-32b": ("qwen3-32b", "Alibaba"),
    "Qwen/Qwen3.6-35B-A3B-FP8": ("Qwen3.6-35B", "Alibaba"),
    "qwen3-next-80b-a3b-instruct": ("qwen3-next-80b", "Alibaba"),
    "qwen/qwen3.5-122b-a10b": ("qwen3.5-122b", "Alibaba"),
    "qwen/qwen3-235b-a22b-thinking-2507": ("qwen3-235b-think", "Alibaba"),
    "qwen/qwen3.5-397b-a17b": ("qwen3.5-397b", "Alibaba"),
    "windprak/open_steuerllm": ("steuerllm", "windprak"),
}

# USD per 1M tokens (input, output), OpenRouter catalogue, retrieved 2026-08-17.
# `note` is empty for an exact catalogue match and names the stand-in otherwise.
PRICES = {
    "Qwen/Qwen3.6-35B-A3B-FP8": (0.14, 1.00, "qwen/qwen3.6-35b-a3b"),
    # These are two checkpoints, not one. The published essays were generated
    # before NHR@FAU swapped the weights on 2026-08-01, so they are 0423 and take
    # the undated slug's price; the -0731 row is the re-generation and takes the
    # dated slug. Pricing both at 0731, as this line used to, overstated the
    # published row by 1.7x on both directions.
    "deepseek-ai/DeepSeek-V4-Flash": (0.0812, 0.1624, "deepseek/deepseek-v4-flash"),
    "deepseek-ai/DeepSeek-V4-Flash-0731": (0.14, 0.28, "deepseek/deepseek-v4-flash-0731"),
    "openai/gpt-oss-120b": (0.03, 0.17, "openai/gpt-oss-120b"),
    "qwen3-next-80b-a3b-instruct": (0.10, 1.10, "qwen/qwen3-next-80b-a3b-instruct"),
    "RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8": (
        0.0938, 0.25, "mistralai/mistral-small-3.2-24b-instruct"),
    "mistralai/Mistral-Medium-3.5-128B": (1.50, 7.50, "mistralai/mistral-medium-3-5"),
    # No catalogue entry of its own -- priced by the nearest listed sibling.
    "mistralai/Ministral-3-14B-Reasoning-2512": (
        0.20, 0.20, "proxy: mistralai/ministral-14b-2512"),
    "ibm-granite/granite-4.1-3b": (0.05, 0.10, "proxy: ibm-granite/granite-4.1-8b"),
    "GaleneAI/Magistral-Small-2509-FP8-Dynamic": (
        0.0938, 0.25, "proxy: mistralai/mistral-small-3.2-24b-instruct"),
    "windprak/open_steuerllm": (0.08, 0.28, "proxy: qwen/qwen3-32b (28B dense)"),
    # The agent-harness run reports no usage at all, so both its tokens and its
    # price are estimates; see opus5_agent/scripts/estimate_tokens.py.
    "anthropic/claude-opus-5": (5.00, 25.00, "anthropic/claude-opus-5"),
    # Closed-beta preview, no listing of its own. Priced by the nearest listed
    # sibling in size and shape: 30B total / 3.5B active vs Qwen3.6's 35B / 3B.
    "soofi-s-isar-preview": (0.14, 1.00, "proxy: qwen/qwen3.6-35b-a3b"),
    # Priced for the single llama-3.3-70b row whose cost the provider dropped.
    "meta-llama/llama-3.3-70b-instruct": (0.038, 0.12, "meta-llama/llama-3.3-70b-instruct"),
}

SIZE_MIN, SIZE_MAX = 60.0, 720.0  # marker area, linear in total completion tokens

# Both figure axes are per-corpus totals, so a model that answered only part of the
# corpus has to be scaled up to be on them at all. Every row of the published sweep
# has all 81; `claude-opus-5` has 48, and is the only one this touches.
N_CASES = 81


def build_table(path=RESULTS, exclude=EXCLUDE, require_price=True):
    """Per-model score, generation cost and verbosity for the no-RAG run.

    `exclude` defaults to this figure's own set. A caller that wants the main
    table's model set passes `exclude=RETIRED`; the three UNPAIRED models have
    no OpenRouter listing, so it must also pass `require_price=False` and ignore
    the cost columns, which is what a recitation or quality figure does anyway.
    """
    df = pd.read_csv(path)
    df = df[~df["model"].isin(exclude)].copy()

    unknown = set(df["model"]) - set(MODELS)
    if unknown:
        raise KeyError(f"models missing from MODELS: {sorted(unknown)}")

    # Reported score: per-essay median across the judges, x100 (see README).
    df["score"] = df.filter(regex=r"^score_Judge").median(axis=1) * 100

    reported = pd.to_numeric(df["total_cost"], errors="coerce")
    price_in = df["model"].map(lambda m: PRICES.get(m, (np.nan,) * 3)[0])
    price_out = df["model"].map(lambda m: PRICES.get(m, (np.nan,) * 3)[1])
    imputed = (
        df["prompt_tokens"].fillna(0) * price_in / 1e6
        + df["completion_tokens"].fillna(0) * price_out / 1e6
    )
    # Billing is a property of the endpoint, not of the single essay: a model is
    # "billed" if any of its 81 rows carries a positive cost. Within a billed
    # model a $0 row is genuine (the two here are content-filtered, 0 tokens),
    # while a NaN row is a dropped usage record and gets the list price.
    billed = df.groupby("model")["total_cost"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").fillna(0).gt(0).any())
    df["cost"] = np.where(billed, reported.fillna(imputed).fillna(0), imputed)

    g = df.groupby("model")
    # Per-essay means scaled to the full corpus, not raw sums: identical to the sum
    # for the 81-case rows, and the honest extrapolation for a short one.
    out = pd.DataFrame({
        "score": g["score"].mean(),
        "cost_usd": g["cost"].mean() * N_CASES,
        "completion_tokens": g["completion_tokens"].mean() * N_CASES,
        "prompt_tokens": g["prompt_tokens"].mean() * N_CASES,
        "billed": g.apply(lambda d: bool(billed[d.index].iloc[0]), include_groups=False),
        "n_rows_missing_cost": g["total_cost"].apply(
            lambda s: int(pd.to_numeric(s, errors="coerce").isna().sum())),
        "n_rows": g.size(),
    })
    if require_price and out["cost_usd"].isna().any():
        raise ValueError(f"no price for: {list(out.index[out['cost_usd'].isna()])}")

    out["short"] = [MODELS[m][0] for m in out.index]
    out["vendor"] = [MODELS[m][1] for m in out.index]
    out["cost_source"] = np.where(out["billed"], "reported", "imputed")
    out["price_basis"] = [PRICES.get(m, ("", "", ""))[2] for m in out.index]
    # Drawn hollow in the figure: x and bubble size are an extrapolation from fewer
    # than 81 essays, while y is still the mean over the essays that exist.
    out["extrapolated"] = out["n_rows"] < N_CASES
    # "*" flags an imputed cost; the figure has no legend to carry the note.
    out["label"] = out["short"] + np.where(out["cost_source"] == "imputed", "*", "")
    return out.sort_values("score").reset_index()


def _bubble_area(tokens):
    lo, hi = tokens.min(), tokens.max()
    return SIZE_MIN + (tokens - lo) / (hi - lo) * (SIZE_MAX - SIZE_MIN)


def _pareto(tab):
    """Models that no cheaper model outscores -- the price/performance frontier."""
    best, keep = -np.inf, []
    for _, row in tab.sort_values("cost_usd").iterrows():
        if row["score"] > best:
            best = row["score"]
            keep.append(row)
    return pd.DataFrame(keep)


def make_figure(tab, path):
    apply_theme()

    # Colour = vendor family; every model uses the same marker shape.
    palette = vendor_palette()
    areas = _bubble_area(tab["completion_tokens"])

    fig, ax = plt.subplots(figsize=(11.2, 7.6), dpi=200)
    fig.subplots_adjust(left=.095, right=.985, bottom=.115, top=.925)

    front = _pareto(tab)
    ax.step(front["cost_usd"], front["score"], where="post",
            color="#8C8F8A", linestyle="--", linewidth=1.5, zorder=1)

    for (_, row), area in zip(tab.iterrows(), areas):
        hollow = bool(row["extrapolated"])
        ax.scatter(
            row["cost_usd"], row["score"],
            s=area, marker=MARKER,
            facecolor="white" if hollow else palette[row["vendor"]],
            edgecolor=palette[row["vendor"]] if hollow else "#22282B",
            linewidth=2.0 if hollow else 1.0,
            alpha=.92, zorder=3,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Generation cost for all 81 essays  (USD, log scale)")
    ax.set_ylabel("Score  (mean of per-essay judge medians, %)")
    ax.set_title("Price and performance without retrieval", pad=14)
    ax.grid(True, which="both", alpha=.3)
    ax.set_axisbelow(True)
    ax.margins(x=.09, y=.09)

    present = [v for v in VENDORS if v in set(tab["vendor"])]
    vendor_handles = [
        Line2D([], [], linestyle="", marker=MARKER, markerfacecolor=palette[v],
               markeredgecolor="#22282B", markersize=10, label=v)
        for v in present
    ]
    vendor_leg = ax.legend(
        handles=vendor_handles, title="Vendor", loc="lower right",
        fontsize=13, title_fontsize=14, ncol=2, frameon=True, framealpha=.92,
    )
    ax.add_artist(vendor_leg)

    # Size key: three round numbers spanning the observed verbosity range.
    ticks = np.array([2e5, 1.0e6, 2.5e6])
    size_handles = [
        Line2D([], [], linestyle="", marker=MARKER, markerfacecolor="#DDE1E3",
               markeredgecolor="#22282B",
               markersize=np.sqrt(_bubble_area(
                   pd.Series([t, tab["completion_tokens"].min(),
                              tab["completion_tokens"].max()]))[0]),
               label=f"{t / 1e6:.1f}M" if t >= 1e6 else f"{t / 1e3:.0f}k")
        for t in ticks
    ]
    size_leg = ax.legend(
        handles=size_handles, title="Tokens produced\n(all 81 essays)",
        loc="upper left", fontsize=13, title_fontsize=13,
        labelspacing=1.1, borderpad=.8, handletextpad=1.2,
        frameon=True, framealpha=.92,
    )
    ax.add_artist(size_leg)

    notes = ["*  cost imputed from list price"]
    if tab["extrapolated"].any():
        short = ", ".join(f'{r["short"]} ({r["n_rows"]}/{N_CASES} essays)'
                          for _, r in tab[tab["extrapolated"]].iterrows())
        notes.append(f"hollow: x and size extrapolated -- {short}")
    note_leg = ax.legend(
        handles=[Line2D([], [], linestyle="", label=n) for n in notes],
        loc="lower left", fontsize=11, frameon=True, framealpha=.92,
        handlelength=0, handletextpad=0, labelspacing=.35,
    )

    fig.canvas.draw()
    legend_boxes = [leg.get_window_extent(fig.canvas.get_renderer())
                    for leg in (vendor_leg, size_leg, note_leg)]
    _place_labels(ax, tab["cost_usd"], tab["score"], tab["label"], areas,
                  obstacles=legend_boxes)

    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tab = build_table()
    tab.to_csv(OUT / "cost_performance_norag.csv", index=False)
    make_figure(tab, OUT / "cost_performance_norag")
    cols = ["short", "vendor", "score", "cost_usd", "completion_tokens",
            "cost_source", "n_rows"]
    print(tab[cols].to_string(index=False))
    print(f"\nwrote {OUT / 'cost_performance_norag.pdf'} (+ .png, .csv)")


if __name__ == "__main__":
    main()

"""Essay quality against the Artificial Analysis Intelligence Index.

One point per model: the AA Intelligence Index on the horizontal axis, our
no-RAG essay Score on the vertical one. It answers whether a general capability
ranking already predicts who writes a passable German Gutachten, or whether
GPBam measures something the aggregate misses.

Run from ``experiments/``::

    python analysis/plot_aa_index_vs_essay.py            # the table below
    python analysis/plot_aa_index_vs_essay.py --refresh  # re-read from the API

Writes ``analysis/out/aa_index_vs_essay_norag.{pdf,png}`` and the numbers behind
the figure to ``analysis/out/aa_index_vs_essay_norag.csv``.

Where the numbers come from
---------------------------
``AA_INDEX`` below was read off the public Artificial Analysis model pages on
2026-08-17 (index version v4.1.1), one page per model, and is the default source
so the figure rebuilds without an account. ``--refresh`` re-reads the same
models through the free data API instead, which needs a key in the repository
``.env`` as ``artificialanalysis_api=...``; it matches on the recorded slug and
prints every value that has moved since.

Matching the configuration
--------------------------
AA scores a *configuration*, not a model, and lists several per model. Our runs
enable reasoning at ``reasoning_effort='high'`` (``src/llm.py``), so the
reasoning / high-effort variant is the one recorded here wherever it exists --
Claude 4.5 Haiku (Reasoning) rather than (Non-reasoning), GPT-5 mini (high),
DeepSeek V3.2 (Reasoning) at 33 rather than the non-reasoning 25. Models with no
reasoning mode (GPT-4o mini, Llama, Mistral Large 3) keep their single entry.

Two caveats survive that:

* **Effort.** Two points are matched on release rather than on effort, because
  AA lists no high-effort configuration for them: ``DeepSeek-V4-Flash-0731`` and
  ``anthropic/claude-opus-5``, whose only published entry is (Adaptive
  Reasoning, Max Effort) at 63. Max effort never scores below high, so for those
  points the recorded x is an upper bound on where they belong. The figure names
  whichever of them it is actually plotting.
* **Estimates.** AA marks some values "Estimate (independent evaluation
  forthcoming)" -- 10 of the 25 here. Those points are drawn hollow, and the
  free API exposes no flag for them, so ``estimated`` is maintained by hand.

``steuerllm`` is a private fine-tune with no AA entry, so the figure reports
one model fewer than the essay table carries.
"""

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from figure_common import MARKER, _place_labels, apply_theme, boot_ci, vendor_palette
from plot_cost_performance import build_table

OUT = Path("analysis/out")
CACHE = OUT / "aa_models.json"
ENV = Path("../.env")
KEY_NAMES = ("artificialanalysis_api", "ARTIFICIALANALYSIS_API_KEY", "aa_api")
ENDPOINTS = (
    "https://artificialanalysis.ai/api/v2/data/llms/models",
    "https://artificialanalysis.ai/api/v2/language/models",
)
INDEX_FIELD = "artificial_analysis_intelligence_index"

# our model id -> (AA slug, AA variant name, index, AA calls it an estimate)
# Read from https://artificialanalysis.ai/models/<slug> on 2026-08-17, v4.1.1.
AA_INDEX = {
    "anthropic/claude-haiku-4.5": (
        "claude-4-5-haiku-reasoning", "Claude 4.5 Haiku (Reasoning)", 30, False),
    # Matched on release, not on effort. Re-checked on 2026-08-21: the Opus 5 page
    # publishes exactly one configuration, (Adaptive Reasoning, Max Effort) at 63,
    # and no high-effort entry at all. Our essays are at high, and max effort only
    # ever scores higher, so 63 is an upper bound on this point's x -- it can move
    # left when AA lists a high-effort configuration, never right.
    "anthropic/claude-opus-5": (
        "claude-opus-5", "Claude Opus 5 (Adaptive Reasoning, Max Effort)", 63, False),
    "deepseek/deepseek-chat-v3-0324": (
        "deepseek-v3-0324", "DeepSeek V3 0324", 15, False),
    "deepseek/deepseek-r1-0528": (
        "deepseek-r1", "DeepSeek R1 0528 (May '25)", 20, True),
    "deepseek/deepseek-v3.2": (
        "deepseek-v3-2-reasoning", "DeepSeek V3.2 (Reasoning)", 33, True),
    # Two checkpoints, matched on effort rather than on release. The published
    # essays predate the 2026-08-01 weight swap, so they are 0423 at high effort,
    # which is AA's 39 entry -- not the 0731 max-effort 52 this used to claim on
    # both counts. The -0731 row is our re-generation, also at high effort; max
    # cannot be generated through this gateway at all (600 s LiteLLM / ~1800 s
    # proxy caps against a full Gutachten), so neither row is a max-effort point.
    "deepseek-ai/DeepSeek-V4-Flash": (
        "deepseek-v4-flash", "DeepSeek V4 Flash 0423 (Reasoning, High Effort)",
        39, False),
    "deepseek-ai/DeepSeek-V4-Flash-0731": (
        "deepseek-v4-flash-0731", "DeepSeek V4 Flash 0731 (Reasoning, High Effort)",
        52, False),
    # Three rows carried an empty AA cell in the merged results table not because
    # Artificial Analysis has no number for them, but because nobody ever looked:
    # they sit in EXCLUDE in plot_cost_performance.py, so build_data never reached
    # them and the missing-entry assertion never fired. Checked on 2026-08-21:
    # Gemma 4 31B and Phi-4 Mini are both listed and are recorded below; EuroLLM
    # 22B is not (404), which is why it keeps its "--".
    #
    # These two fill the AA column of the merged table, which reads AA_INDEX
    # directly. They do NOT enter the AA figure -- that is built from the paired
    # set, and all three are still excluded from it there.
    #
    # Slugs are the ones the public model pages sit at. They are unverified
    # against the data API, which has no key in this checkout; if they are wrong
    # the --refresh path prints "slug gone from the API" and keeps these values,
    # rather than failing.
    "RedHatAI/gemma-4-31B-it-FP8-block": (
        "gemma-4-31b", "Gemma 4 31B (Reasoning)", 30, False),
    "Microsoft/Phi-4-mini-instruct": (
        "phi-4-mini", "Phi-4 Mini Instruct", 6, False),
    "google/gemini-2.5-flash-lite": (
        "gemini-2-5-flash-lite-reasoning", "Gemini 2.5 Flash-Lite (Reasoning)", 11, True),
    # AA's high-effort entry, which is how we run it.
    "google/gemini-3.7-flash": (
        "gemini-3-7-flash-high", "Gemini 3.7 Flash (High)", 56, True),
    "ibm-granite/granite-4.1-3b": (
        "granite-4-1-3b", "Granite 4.1 3B", 4, True),
    "meta-llama/llama-3.1-8b-instruct": (
        "llama-3-1-instruct-8b", "Llama 3.1 Instruct 8B", 7, True),
    "meta-llama/llama-3.3-70b-instruct": (
        "llama-3-3-instruct-70b", "Llama 3.3 Instruct 70B", 9, True),
    "meta-llama/llama-4-maverick": (
        "llama-4-maverick", "Llama 4 Maverick", 14, False),
    "GaleneAI/Magistral-Small-2509-FP8-Dynamic": (
        "magistral-small-2509", "Magistral Small 1.2", 11, False),
    "mistralai/Ministral-3-14B-Reasoning-2512": (
        "ministral-3-14b", "Ministral 3 14B", 10, True),
    "RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8": (
        "mistral-small-3-2", "Mistral Small 3.2", 11, False),
    "mistralai/Mistral-Medium-3.5-128B": (
        "mistral-medium-3-5", "Mistral Medium 3.5", 30, False),
    "mistralai/mistral-large-2512": (
        "mistral-large-3", "Mistral Large 3", 16, False),
    "openai/gpt-4o-mini-2024-07-18": (
        "gpt-4o-mini", "GPT-4o mini", 7, True),
    "openai/gpt-5-nano": (
        "gpt-5-nano", "GPT-5 nano (high)", 20, True),
    "openai/gpt-5-mini": (
        "gpt-5-mini", "GPT-5 mini (high)", 26, False),
    "openai/gpt-oss-120b": (
        "gpt-oss-120b", "gpt-oss-120b (high)", 24, False),
    "qwen/qwen3-32b": (
        "qwen3-32b-instruct-reasoning", "Qwen3 32B (Reasoning)", 11, False),
    "Qwen/Qwen3.6-35B-A3B-FP8": (
        "qwen3-6-35b-a3b", "Qwen3.6 35B A3B (Reasoning)", 32, False),
    "qwen3-next-80b-a3b-instruct": (
        "qwen3-next-80b-a3b-instruct", "Qwen3 Next 80B A3B Instruct", 14, True),
    "qwen/qwen3.5-122b-a10b": (
        "qwen3-5-122b-a10b", "Qwen3.5 122B A10B (Reasoning)", 33, False),
    "qwen/qwen3-235b-a22b-thinking-2507": (
        "qwen3-235b-a22b-instruct-2507-reasoning",
        "Qwen3 235B A22B 2507 (Reasoning)", 20, False),
    "qwen/qwen3.5-397b-a17b": (
        "qwen3-5-397b-a17b", "Qwen3.5 397B A17B (Reasoning)", 34, False),
}

# No AA entry exists: a private fine-tune, a closed-beta preview, and one model
# Artificial Analysis simply does not list (checked 2026-08-21, 404).
NO_AA_ENTRY = {"windprak/open_steuerllm", "soofi-s-isar-preview",
               "utter-project/EuroLLM-22B-Instruct-2512"}

# Matched on release, not on effort: AA lists no high-effort configuration for
# these, so their recorded index is a max-effort one and their x is an upper
# bound. Named in the figure, because two of the three are the rightmost points
# and a fit is only as good as its leverage.
EFFORT_UPPER_BOUND = {
    "deepseek-ai/DeepSeek-V4-Flash",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
    "anthropic/claude-opus-5",
}

MARKER_AREA = 85.0


# --------------------------------------------------------------------------- #
# Optional API refresh
# --------------------------------------------------------------------------- #
def _api_key():
    """The AA key from the repo .env, with a pointer to the docs if absent."""
    if ENV.exists():
        for line in ENV.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() in KEY_NAMES and value.strip():
                return value.strip().strip('"').strip("'")
    raise SystemExit(
        f"No Artificial Analysis key found. Add one of {KEY_NAMES} to {ENV.resolve()};\n"
        "a free key comes from an account at https://artificialanalysis.ai/data-api"
    )


def fetch_catalogue():
    """The AA model list from the data API, cached for inspection."""
    key = _api_key()
    errors = []
    for url in ENDPOINTS:
        request = urllib.request.Request(url, headers={"x-api-key": key})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            errors.append(f"{url} -> HTTP {exc.code}")
            continue
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(payload, indent=1))
        print(f"fetched {len(payload['data'])} models from {url}")
        return payload["data"]
    raise SystemExit("Artificial Analysis API unreachable: " + "; ".join(errors))


def refreshed_index():
    """`AA_INDEX` re-read from the API, reporting every value that moved."""
    by_slug = {e.get("slug"): e for e in fetch_catalogue()}
    out = {}
    for model, (slug, name, index, estimated) in AA_INDEX.items():
        entry = by_slug.get(slug)
        if entry is None:
            print(f"  slug gone from the API, keeping the recorded value: {slug}")
            out[model] = (slug, name, index, estimated)
            continue
        fresh = (entry.get("evaluations") or {}).get(INDEX_FIELD)
        if fresh is None:
            print(f"  no index in the API response, keeping recorded: {slug}")
            fresh = index
        if fresh != index:
            print(f"  {slug}: {index} -> {fresh}")
        out[model] = (slug, entry.get("name", name), fresh, estimated)
    return out


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build_data(refresh=False):
    """Per-model essay Score joined to the AA Intelligence Index."""
    index = refreshed_index() if refresh else AA_INDEX
    essays = build_table()  # the 26 paired models, no-RAG

    rows, missing = [], []
    for _, row in essays.iterrows():
        if row["model"] in NO_AA_ENTRY:
            continue
        if row["model"] not in index:
            missing.append(row["model"])
            continue
        slug, name, value, estimated = index[row["model"]]
        rows.append({
            "model": row["model"], "short": row["short"], "vendor": row["vendor"],
            "score": row["score"], "aa_index": value, "aa_name": name,
            "aa_slug": slug, "aa_estimated": estimated,
            "effort_upper_bound": row["model"] in EFFORT_UPPER_BOUND,
        })

    if missing:
        raise SystemExit("no AA entry recorded for: " + ", ".join(missing))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(tab, path):
    apply_theme()
    palette = vendor_palette()

    fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=200)
    fig.subplots_adjust(left=.115, right=.985, bottom=.145, top=.905)

    x, y = tab["aa_index"].to_numpy(float), tab["score"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 2)
    ax.plot(xs, slope * xs + intercept, color="#8C8F8A", linestyle="--",
            linewidth=1.6, alpha=.6, zorder=1)

    for _, row in tab.iterrows():
        colour = palette[row["vendor"]]
        ax.scatter(row["aa_index"], row["score"], s=MARKER_AREA, marker=MARKER,
                   facecolor="white" if row["aa_estimated"] else colour,
                   edgecolor=colour if row["aa_estimated"] else "#22282B",
                   linewidth=1.7 if row["aa_estimated"] else .9,
                   alpha=.95, zorder=3)

    ax.set_xlabel("Artificial Analysis Intelligence Index")
    ax.set_ylabel("Essay Score, no retrieval  (%)")
    ax.set_title("General capability against German legal essays", pad=14)
    ax.grid(True, alpha=.3)
    ax.set_axisbelow(True)
    ax.margins(x=.09, y=.09)

    rho = spearmanr(x, y).statistic
    lo, hi = boot_ci(x, y)
    lines = [rf"Spearman $\rho$ = {rho:.2f}  [{lo:.2f}, {hi:.2f}]",
             f"n = {len(tab)} models",
             "hollow: AA calls the index an estimate"]
    ub = tab[tab["effort_upper_bound"]]
    if len(ub):
        lines.append("x is an upper bound (max-effort entry, high-effort run):")
        lines.append("   " + ", ".join(sorted(ub["short"])))
    stats_leg = ax.legend(
        handles=[Line2D([], [], linestyle="", label=line) for line in lines],
        loc="upper left", fontsize=11, frameon=True, framealpha=.92,
        handlelength=0, handletextpad=0, labelspacing=.35,
    )

    fig.canvas.draw()
    box = stats_leg.get_window_extent(fig.canvas.get_renderer())
    crowded = _place_labels(
        ax, tab["aa_index"], tab["score"], tab["short"],
        np.full(len(tab), MARKER_AREA), obstacles=[box], fontsize=9.5,
    )

    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return crowded, rho, (lo, hi)


def main():
    parser = argparse.ArgumentParser(description="Essay Score vs the AA index")
    parser.add_argument("--refresh", action="store_true",
                        help="re-read the recorded models through the AA data API")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tab = build_data(refresh=args.refresh)
    tab.to_csv(OUT / "aa_index_vs_essay_norag.csv", index=False)

    crowded, rho, (lo, hi) = make_figure(tab, OUT / "aa_index_vs_essay_norag")
    show = tab[["short", "aa_name", "aa_index", "aa_estimated", "score"]]
    print(show.sort_values("aa_index").to_string(index=False))
    print(f"\nSpearman rho = {rho:.3f} [{lo:.2f}, {hi:.2f}] over n={len(tab)}"
          f"  ({int(tab['aa_estimated'].sum())} of them AA estimates)")
    if crowded:
        print(f"note: {crowded} of {len(tab)} names overlap another name")
    print(f"wrote {OUT / 'aa_index_vs_essay_norag.pdf'} (+ .png, .csv)")


if __name__ == "__main__":
    main()

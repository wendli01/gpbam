"""Shared drawing machinery for the paper's model-level scatter figures.

Three figures plot one point per model with its name written next to it
(``plot_cost_performance``, ``plot_recitation_vs_essay``,
``plot_aa_index_vs_essay``). They share the label placer, the vendor palette
and the bootstrap CI so that the panels stay visually and statistically
consistent; each script keeps its own data loading and axes.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr

# Colour is the vendor family; one marker shape for every model, so shape is
# free and size is left for a quantity.
# Appended to, never reordered: vendor_palette zips this against tab10, so a new
# name at the end takes the next colour and leaves every existing one where it was.
VENDORS = ("Alibaba", "Anthropic", "DeepSeek", "Google", "IBM", "Meta", "Mistral",
           "OpenAI", "windprak", "Soofi")
MARKER = "o"

N_BOOT, SEED = 10_000, 0


def vendor_palette():
    """Vendor -> colour, stable across the figures."""
    return dict(zip(VENDORS, sns.color_palette("tab10", len(VENDORS))))


def apply_theme():
    """The common look: seaborn whitegrid, large type for a small panel."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 19,
        "axes.titlesize": 21,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
    })


def boot_ci(x, y, alpha=.05):
    """Percentile bootstrap CI for Spearman rho, resampling model pairs jointly.

    Same procedure and seed as ``src/notebook_eval/essay_correlation``, so the
    intervals drawn on a figure match the ones printed in the tables.
    """
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, x.size, size=(N_BOOT, x.size))
    vals = np.array([spearmanr(x[i], y[i]).statistic for i in idx])
    vals = vals[~np.isnan(vals)]
    return tuple(np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def _overlap_area(a, b):
    """Area shared by two display-space boxes, 0 when they are disjoint."""
    dx = min(a.x1, b.x1) - max(a.x0, b.x0)
    dy = min(a.y1, b.y1) - max(a.y0, b.y0)
    return dx * dy if dx > 0 and dy > 0 else 0.0


def _marker_box(ax, x, y, area):
    """Display-space box covering one marker."""
    px, py = ax.transData.transform((x, y))
    r = np.sqrt(area) / 2 * ax.figure.dpi / 72.0
    return matplotlib.transforms.Bbox([[px - r, py - r], [px + r, py + r]])


def _place_labels(ax, xs, ys, labels, areas, obstacles=(), extra_markers=(),
                  fontsize=11.5):
    """Model names next to their anchors, nudged to avoid overlaps.

    ``xs``/``ys``/``labels``/``areas`` describe the labelled anchors; markers
    that must be avoided without carrying a label of their own (the second
    series of a dumbbell, say) go in ``extra_markers`` as ``(x, y, area)``.
    Directly above or below is unambiguous on its own; every other position gets
    a thin leader line, because in a crowded plot a sideways label can otherwise
    sit closer to a neighbour than to its own point.

    Returns the number of names that ended up touching another name -- brushing
    a marker or a connector is tolerable, two names on top of each other is not,
    and a figure too small for its labels is a decision rather than something to
    discover by squinting at the render.
    """
    ax.figure.canvas.draw()
    frame = ax.get_window_extent()
    xs, ys, labels = np.asarray(xs), np.asarray(ys), np.asarray(labels)
    areas = np.asarray(areas)

    # Markers and legends are obstacles, so a label never lands on either.
    placed = list(obstacles)
    placed += [_marker_box(ax, x, y, a) for x, y, a in extra_markers]
    n_fixed = len(placed)
    placed += [_marker_box(ax, x, y, a) for x, y, a in zip(xs, ys, areas)]

    # Crowded points are labelled first: they have the fewest free slots, while
    # an isolated point still finds its preferred one whenever its turn comes.
    xy = np.array([b.get_points().mean(axis=0) for b in placed[n_fixed:]])
    dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    order = np.argsort(-(dist < 130).sum(axis=1), kind="stable")

    compromised, text_boxes = 0, []
    for i_pt in order:
        anchor, label = (xs[i_pt], ys[i_pt]), labels[i_pt]
        pad = np.sqrt(areas[i_pt]) / 2 + 3  # marker radius in points
        # Five rings of eight directions, nearest ring first.
        directions = [
            (0, 1, "center", "bottom"), (0, -1, "center", "top"),
            (1, 0, "left", "center"), (-1, 0, "right", "center"),
            (.7, .7, "left", "bottom"), (.7, -.7, "left", "top"),
            (-.7, .7, "right", "bottom"), (-.7, -.7, "right", "top"),
            (.4, 1, "center", "bottom"), (-.4, 1, "center", "bottom"),
            (.4, -1, "center", "top"), (-.4, -1, "center", "top"),
            (1, .5, "left", "center"), (1, -.5, "left", "center"),
            (-1, .5, "right", "center"), (-1, -.5, "right", "center"),
        ]
        candidates = [
            (ux * (pad + d), uy * (pad + d), ha, va)
            for d in (4, 14, 25, 36, 50)
            for ux, uy, ha, va in directions
        ]
        leader = dict(arrowstyle="-", color="#9AA0A5", linewidth=.7, alpha=.6,
                      shrinkA=1, shrinkB=2)
        # Each candidate is scored, and the first zero-penalty one wins. Owning
        # the label matters far more than a clean gap: a name that reads as its
        # neighbour's is wrong, while a name that grazes another name is merely
        # tight, so a mis-owned slot is penalised out of reach of any overlap.
        best = (np.inf, None)
        for i, (dx, dy, ha, va) in enumerate(candidates):
            txt = ax.annotate(
                label, anchor,
                textcoords="offset points", xytext=(dx, dy), ha=ha, va=va,
                fontsize=fontsize, color="#22282B", zorder=4,
                arrowprops=None if i < 2 else leader,
            )
            box = txt.get_window_extent(ax.figure.canvas.get_renderer())
            centre = box.get_points().mean(axis=0)
            nearest = np.argmin(np.linalg.norm(xy - centre, axis=1))
            overlap = sum(_overlap_area(box, b) for b in placed)
            outside = not frame.containsx(box.x0) or not frame.containsx(box.x1) \
                or not frame.containsy(box.y0) or not frame.containsy(box.y1)
            penalty = overlap + 1e6 * (nearest != i_pt) + 1e6 * outside
            if penalty == 0:
                best = (0., i)
                break
            if penalty < best[0]:
                best = (penalty, i)
            txt.remove()

        if best[1] != i:  # the winner was removed while scanning -- redraw it
            dx, dy, ha, va = candidates[best[1]]
            txt = ax.annotate(
                label, anchor,
                textcoords="offset points", xytext=(dx, dy), ha=ha, va=va,
                fontsize=fontsize, color="#22282B", zorder=4,
                arrowprops=None if best[1] < 2 else leader,
            )
            box = txt.get_window_extent(ax.figure.canvas.get_renderer())
        compromised += any(_overlap_area(box, t) > 0 for t in text_boxes)
        text_boxes.append(box)
        placed.append(box.expanded(1.02, 1.02))

    return compromised

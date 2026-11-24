"""Render the supergraph figure: the same six ANOVA supernodes in three models.

Each panel is one model's supergraph for the prompt "36+59=": six nodes, each drawn
as the activation grid of a representative member neuron over all 10^4 two-digit
prompts, and the supernode edge weights of eq:superedge between them. The three
panels are the untrained 1B student, the 8B teacher, and the 1B student after graph
distillation, so both halves of the claim are in one picture -- the teacher's
routing is sharp, the untrained student's is flat, and the trained student's has
moved onto the teacher's.

Node images come from ``results/heatmap_plotting/{1b,8b,1b_anova}/`` and the edge
weights are transcribed from that run's ``log.txt`` in the same directory, which
prints one 6x6 supernode adjacency per model. Rows of that matrix are targets and
columns are sources, so the arrow runs column -> row.

Usage (from the repository root):
    PYTHONPATH=src python -m experiments.plot_supergraph
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from experiments._paper_plots import FIG_DIR, INK, INK_2, INK_3, RC, save

SRC = "results/heatmap_plotting"

# Supernode order as the log prints it; rows = target, columns = source.
LABELS = ["arg1 range", "arg1 units", "arg2 range", "arg2 units", "sum range", "sum units"]
# Each supernode's category mask at the prompt 36+59= (equiv is modulo 10), as the
# paper's category table defines them. mathtext only -- usetex is off.
NODE = ["$a = 36$", r"$a \equiv 6$", "$b = 59$", r"$b \equiv 9$", "$s = 95$", r"$s \equiv 5$"]

W = {
    "1b": [[0.110, 0.097, 0.000, 0.000, 0.000, 0.000],
           [0.067, 0.152, 0.000, 0.000, 0.000, 0.000],
           [0.004, 0.004, 0.025, 0.007, 0.000, 0.000],
           [0.017, 0.016, 0.048, 0.165, 0.000, 0.000],
           [0.009, 0.020, 0.012, 0.020, 0.023, 0.028],
           [0.008, 0.021, 0.010, 0.024, 0.018, 0.033]],
    "8b": [[0.120, 0.205, 0.000, 0.000, 0.000, 0.000],
           [0.094, 0.376, 0.000, 0.000, 0.000, 0.000],
           [0.001, 0.005, 0.021, 0.050, 0.000, 0.000],
           [0.008, 0.018, 0.080, 0.331, 0.000, 0.000],
           [0.029, 0.113, 0.044, 0.160, 0.232, 0.177],
           [0.071, 0.130, 0.028, 0.127, 0.254, 0.260]],
    "1b_anova": [[0.066, 0.041, 0.000, 0.000, 0.000, 0.000],
                 [0.050, 0.178, 0.000, 0.000, 0.000, 0.000],
                 [0.001, 0.001, 0.008, 0.003, 0.000, 0.000],
                 [0.012, 0.009, 0.035, 0.178, 0.000, 0.000],
                 [0.029, 0.108, 0.021, 0.159, 0.164, 0.122],
                 [0.024, 0.098, 0.020, 0.170, 0.220, 0.195]],
}

PANELS = [("1b", "Llama-3.2-1B, untrained"),
          ("8b", "Llama-3-8B, teacher"),
          ("1b_anova", "Llama-3.2-1B, graph distillation")]

# Node boxes in panel coordinates: (x, y, w, h) of the lower-left corner. Two
# columns; arg1 on the top row, arg2 below it, the two sum grids at the bottom.
SW, SH, GW, GH, DH = 0.40, 0.063, 0.40, 0.190, 0.075
# Six category supernodes, then the two dla supernodes under their sum targets.
# The dla grids are constant in b, so a short wide box loses nothing.
BOX = [(0.03, 0.775, SW, SH), (0.57, 0.775, SW, SH),
       (0.03, 0.640, SW, SH), (0.57, 0.640, SW, SH),
       (0.03, 0.175, GW, GH), (0.57, 0.175, GW, GH),
       (0.03, 0.035, GW, DH), (0.57, 0.035, GW, DH)]

EDGE_MIN = 0.10   # below this the edge is not drawn; stated in the caption
FIG_W, FIG_H = 5.5, 3.55
# One panel's axes, in inches, from the figure size and subplots_adjust below.
AX_W = (0.997 - 0.050) * FIG_W / (3 + 2 * 0.09)
AX_H = (0.90 - 0.015) * FIG_H
MAX_W = 0.376     # largest weight anywhere, so widths compare across panels


def centre(i: int) -> tuple:
    x, y, w, h = BOX[i]
    return x + w / 2, y + h / 2


def edge_style(v: float) -> dict:
    """Width and ink from the weight -- one visual variable, monotone in it."""
    t = min(1.0, v / MAX_W)
    return dict(linewidth=0.45 + 2.1 * t, color=INK if t > 0.45 else INK_2,
                alpha=0.40 + 0.60 * t)


def label(ax, x: float, y: float, v: float) -> None:
    ax.text(x, y, f"{v:.2f}", ha="center", va="center", fontsize=5.0, color=INK_2,
            zorder=6, bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none",
                                alpha=0.88))


def draw_self_loop(ax, i: int, v: float) -> None:
    x, y, w, h = BOX[i]
    cx, top, r = x + w / 2, y + h, 0.05
    ax.add_patch(FancyArrowPatch((cx - r, top + 0.003), (cx + r, top + 0.003),
                                 connectionstyle="arc3,rad=-1.3",
                                 arrowstyle="-|>", mutation_scale=4.5,
                                 shrinkA=0, shrinkB=0, zorder=3, **edge_style(v)))
    # The right-column strips carry their node label at the top-right corner, so
    # their self-loop weight goes to the left of the loop instead.
    side = -1 if i in (1, 3) else 1
    label(ax, cx + side * 0.10, top + 0.042, v)


def shrink_pt(i: int, other: int, pad: float = 2.0) -> float:
    """Points to trim off the edge at node i so it starts outside the node's image.

    A fixed shrink cannot serve both node shapes here: the strips are ~7pt tall
    and ~50pt wide, the sum grids ~50pt square, so a value that clears a strip
    vertically leaves a grid-to-grid edge buried under the two images. This
    measures the node's own half-extent along the chord, in display points.
    """
    x, y, w, h = BOX[i]
    ox, oy = centre(other)
    dx, dy = (ox - (x + w / 2)) * AX_W, (oy - (y + h / 2)) * AX_H
    n = np.hypot(dx, dy) or 1.0
    dx, dy = dx / n, dy / n
    tx = abs((w * AX_W / 2) / dx) if abs(dx) > 1e-9 else np.inf
    ty = abs((h * AX_H / 2) / dy) if abs(dy) > 1e-9 else np.inf
    return min(tx, ty) * 72.0 + pad


def draw_edge(ax, src: int, tgt: int, v: float, rad: float, tpos: float,
              off: float) -> None:
    x0, y0 = centre(src)
    x1, y1 = centre(tgt)
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                 connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=4.5,
                                 shrinkA=shrink_pt(src, tgt),
                                 shrinkB=shrink_pt(tgt, src),
                                 zorder=2, **edge_style(v)))
    mx, my = x0 + (x1 - x0) * tpos, y0 + (y1 - y0) * tpos
    dx, dy = x1 - x0, y1 - y0
    n = np.hypot(dx, dy) or 1.0
    label(ax, mx - dy / n * off, my + dx / n * off, v)


# (rad, position along the chord, label offset) per ordered (source, target) pair,
# set by hand so no two labels land on each other.
# arc3 with a positive rad bulges to the RIGHT of the direction of travel, so a
# reciprocal pair wants the SAME positive rad, not opposite signs: reversing the
# direction already flips which side the arc takes. (Opposite signs stack both arcs,
# and both labels, on each other -- that is how the teacher's 0.18 sum-range edge
# went missing.) The label offset must then be negative, or each label lands beside
# the other direction's arrow.
ARC = {
    (1, 0): (0.30, 0.50, -0.055), (0, 1): (0.30, 0.50, -0.055),
    (3, 2): (0.30, 0.50, -0.055), (2, 3): (0.30, 0.50, -0.055),
    (0, 2): (0.30, 0.50, 0.050), (1, 3): (-0.30, 0.50, -0.050),
    (0, 4): (0.18, 0.50, 0.050), (1, 4): (0.12, 0.34, 0.038),
    (2, 4): (0.14, 0.50, 0.046), (3, 4): (0.10, 0.62, 0.038),
    (0, 5): (-0.10, 0.62, -0.038), (1, 5): (-0.18, 0.50, -0.050),
    (2, 5): (-0.12, 0.34, -0.038), (3, 5): (-0.14, 0.50, -0.046),
    (4, 5): (0.26, 0.50, -0.090), (5, 4): (0.26, 0.50, -0.090),
}


def panel(ax, key: str, title: str, first: bool = False) -> None:
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.01, 1.02)
    ax.axis("off")
    ax.text(0.5, 1.005, title, ha="center", va="bottom", fontsize=7, color=INK,
            transform=ax.transAxes)
    w = W[key]

    for i, (x, y, bw, bh) in enumerate(BOX):
        name = (f"{i}.png" if i < 4 else
                f"{i}_grid.png" if i < 6 else f"{i - 2}_dla.png")
        img = mpimg.imread(os.path.join(SRC, key, name))
        sub = ax.inset_axes([x, y, bw, bh])
        sub.imshow(img, aspect="auto", interpolation="nearest")
        sub.set_xticks([])
        sub.set_yticks([])
        for s in sub.spines.values():
            s.set_edgecolor(INK_3)
            s.set_linewidth(0.4)
        if i < 4:
            outer = (x, "left") if i % 2 == 0 else (x + bw, "right")
            ax.text(outer[0], y + bh + 0.034, NODE[i], ha=outer[1], va="bottom",
                    fontsize=6.0, color=INK)
        elif i < 6:
            ax.text(x + bw / 2, y - 0.028, NODE[i], ha="center", va="top",
                    fontsize=6.0, color=INK)
        else:
            ax.text(x + bw / 2, y - 0.024, "output side", ha="center", va="top",
                    fontsize=5.4, color=INK_2)

    if first:
        for cx, name in ((0.23, "value"), (0.77, "ones digit")):
            ax.text(cx, 0.965, name, ha="center", va="bottom", fontsize=5.4,
                    color=INK_2)

    for tgt in range(6):
        for src in range(6):
            v = w[tgt][src]
            if v < EDGE_MIN:
                continue
            if src == tgt:
                draw_self_loop(ax, tgt, v)
            else:
                draw_edge(ax, src, tgt, v, *ARC.get((src, tgt), (0.14, 0.55, 0.04)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=f"{FIG_DIR}/supergraph.pdf")
    args = ap.parse_args()

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H))
        for i, (ax, (key, title)) in enumerate(zip(axes, PANELS)):
            panel(ax, key, title, first=(i == 0))
        top, bot = 0.90, 0.015
        fig.text(0.010, 0.47, "36 + 59 =", rotation=90, ha="left", va="center",
                 fontsize=7.5, color=INK)
        fig.subplots_adjust(left=0.050, right=0.997, top=top, bottom=bot,
                            wspace=0.09)
        save(fig, args.out)


if __name__ == "__main__":
    main()

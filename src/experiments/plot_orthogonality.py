"""Render the gradient-geometry figure for Appendix A.

Appendix A makes two claims at once -- the graph gradient is nearly orthogonal to
the KD gradient, and it is far smaller -- and prose conveys neither well. This
draws both.

Upper panel: the two gradients to scale, on equal axes so the angle is not
distorted. The KD gradient runs long and horizontal; the graph gradient is a stub
pointing almost straight up. That the graph term is small *and* sideways is the
whole of the appendix in one picture.

Lower panel: the cosines on a [-1, 1] number line, with the same measure taken
between two graph-loss linearisations as a yardstick. Without that reference a
reader has no way to judge whether -0.02 is "close to zero" or merely small.

Numbers are the diagnose_grad output for Llama-3.2-1B vs Meta-Llama-3-8B on
22_add, 8 graph prompts, batch 32, at the student's initial checkpoint. They are
transcribed here rather than re-read from a file because diagnose_grad reports to
stdout; re-run it to refresh them.

Usage:
    python -m experiments.plot_orthogonality
"""

from __future__ import annotations

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

# Validated categorical slots 1-2 (light, --pairs all: every check passes).
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8983"
GRID = "#e9e8e5"

KD_NORM = 50.4056
# variant -> (graph gradient norm, cos with the KD gradient)
VARIANTS = [
    ("no freeze", 3.64671, -0.0236),
    ("attention", 4.059, -0.0103),
    ("RMSNorm", 3.9905, -0.0153),
    ("both", 3.8557, -0.0082),
]
# The same cosine measure between graph-loss variants, as a scale reference.
BETWEEN_LINEARISATIONS = [0.6276, 0.7652, 0.6984]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="latex/graph_distillation/figures/orthogonality.pdf")
    args = ap.parse_args()

    plt.rcParams.update({
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "pdf.fonttype": 42,
    })
    fig, (ax_v, ax_c) = plt.subplots(
        2, 1, figsize=(5.5, 2.45), gridspec_kw={"height_ratios": [1.15, 1]}
    )

    # ---- upper: the two gradients, to scale, equal aspect ------------------
    g_norm, g_cos = VARIANTS[0][1], VARIANTS[0][2]
    theta = math.acos(g_cos)  # radians, just past 90 degrees
    gx, gy = g_norm * math.cos(theta), g_norm * math.sin(theta)

    arrow = dict(width=0.28, head_width=1.5, head_length=1.6, length_includes_head=True)
    ax_v.arrow(0, 0, KD_NORM, 0, color=BLUE, zorder=3, **arrow)
    ax_v.arrow(0, 0, gx, gy, color=ORANGE, zorder=3, **arrow)

    ax_v.add_patch(Arc((0, 0), 5.0, 5.0, theta1=0, theta2=math.degrees(theta),
                       color=INK_3, linewidth=0.8, zorder=2))
    ax_v.annotate(f"{math.degrees(theta):.1f}°", (3.1, 1.5), color=INK_3,
                  fontsize=6.5, ha="left", va="bottom")

    ax_v.annotate(r"$\nabla\mathcal{L}_{\mathrm{KD}}$   $\|\cdot\| = 50.4$",
                  (KD_NORM - 1.5, 1.2), color=INK, fontsize=7, ha="right", va="bottom")
    ax_v.annotate(r"$\nabla\mathcal{L}_{\mathrm{graph}}$   $\|\cdot\| = 3.6$",
                  (1.6, gy + 1.6), color=INK, fontsize=7, ha="left", va="bottom")

    ax_v.set_xlim(-2.5, 56)
    ax_v.set_ylim(-1.6, 9.2)
    ax_v.set_aspect("equal")  # required: an unequal aspect would misdraw the angle
    ax_v.axis("off")

    # ---- lower: cosines on a [-1, 1] line ---------------------------------
    ax_c.axhline(0, color=GRID, linewidth=1.0, zorder=1)
    ax_c.plot([BETWEEN_LINEARISATIONS[0], BETWEEN_LINEARISATIONS[1]], [0, 0],
              color=INK_3, linewidth=3.0, solid_capstyle="butt", alpha=0.35, zorder=2)
    ax_c.annotate("between two graph-loss\nlinearisations",
                  (sum(BETWEEN_LINEARISATIONS[:2]) / 2, 0.10), color=INK_3,
                  fontsize=6, ha="center", va="bottom", linespacing=1.3)

    # The four values coincide to within 0.015, so a vertical offset separates
    # them. It carries no meaning -- there is no y axis -- and the caption says so.
    for (_, _, cos), dy in zip(VARIANTS, (0.15, 0.05, -0.05, -0.15)):
        ax_c.plot([cos], [dy], marker="o", markersize=3.6, color=ORANGE,
                  markeredgecolor="white", markeredgewidth=0.6, zorder=4)
    ax_c.annotate("graph vs. KD\n(all four linearisations)", (0, 0.30), color=INK,
                  fontsize=6.5, ha="center", va="bottom", linespacing=1.3)

    for x, lab in ((-1, "opposite"), (0, "orthogonal"), (1, "identical")):
        ax_c.plot([x, x], [-0.1, 0.1], color=INK_3, linewidth=0.8, zorder=3)
        ax_c.annotate(lab, (x, -0.2), color=INK_2, fontsize=6.5, ha="center", va="top")

    ax_c.set_xlim(-1.12, 1.12)
    ax_c.set_ylim(-0.46, 0.92)
    ax_c.set_yticks([])
    ax_c.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax_c.tick_params(colors=INK_2, length=0, pad=3)
    for side in ("top", "right", "left", "bottom"):
        ax_c.spines[side].set_visible(False)
    ax_c.set_xlabel("cosine between gradients in parameter space", color=INK_2, labelpad=2)

    fig.tight_layout(h_pad=0.4)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=220, bbox_inches="tight", facecolor="white")
    print("wrote", args.out)


if __name__ == "__main__":
    main()

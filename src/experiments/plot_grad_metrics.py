"""Render the per-step gradient-metric figure for Appendix C.

Reads the same four history files as plot_freeze_ablation.py, but plots what
--track-grad-metrics records rather than accuracy.

The figure carries two messages. The first is the takeover: the KD gradient
collapses by more than an order of magnitude over fifteen steps while the graph
gradient stays flat, so their ratio climbs tenfold without anyone choosing that.
The second is a negative result, and it is why all four variants are drawn on
every panel -- they lie almost on top of each other. The linearisations differ by
a third in out-of-distribution accuracy while their gradient summaries are nearly
indistinguishable, so whatever separates them is not visible in aggregate
statistics.

Encoding matches the accuracy figure: hue carries the RMSNorm freeze, line style
the attention freeze.

Usage:
    python -m experiments.plot_grad_metrics
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8983"
GRID = "#e9e8e5"

RUNS = [
    ("22_add_argdla", "no freeze", BLUE, (0, (4, 2))),
    ("22_add_argdla_attn", "attention", BLUE, "solid"),
    ("22_add_argdla_rms", "RMSNorm", ORANGE, (0, (4, 2))),
    ("22_add_argdla_attn+rms", "both", ORANGE, "solid"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default="results/freeze_ablation")
    ap.add_argument("--out", default="latex/graph_distillation/figures/grad_metrics.pdf")
    args = ap.parse_args()

    runs = []
    for stem, label, colour, dash in RUNS:
        path = os.path.join(args.results_dir, stem + ".json")
        if not os.path.exists(path):
            raise SystemExit(f"missing history file: {path}")
        with open(path, encoding="utf-8") as f:
            runs.append((label, json.load(f), colour, dash))
    steps = runs[0][1]["train_step"]

    plt.rcParams.update({
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "axes.titlesize": 7.5, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.3))

    def style(ax, title):
        ax.set_facecolor("white")
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=INK_2, length=2, width=0.8)
        ax.set_title(title, color=INK, pad=4)
        ax.set_xlim(0.4, 15.6)
        ax.set_xticks([1, 5, 10, 15])

    line = dict(linewidth=1.3, solid_capstyle="round", dash_capstyle="round",
                marker="o", markersize=1.9, markeredgewidth=0, zorder=3)

    # (a) the two norms, log scale -- they differ by more than an order of magnitude
    ax = axes[0][0]
    style(ax, "gradient norms")
    for _, h, c, d in runs:
        ax.plot(steps, h["step_kl_gnorm"], color=c, linestyle=d, **line)
        ax.plot(steps, h["step_graph_gnorm"], color=c, linestyle=d, **line)
    ax.set_yscale("log")
    ax.annotate(r"$\|g_{\mathrm{KD}}\|$", (10.5, 12), color=INK, fontsize=6.5)
    ax.annotate(r"$\|g_{\mathrm{graph}}\|$", (10.5, 2.4), color=INK, fontsize=6.5)
    ax.set_ylabel("norm (log)", color=INK_2)

    # (b) their ratio
    ax = axes[0][1]
    style(ax, r"$\|g_{\mathrm{graph}}\| / \|g_{\mathrm{KD}}\|$")
    for _, h, c, d in runs:
        ax.plot(steps, h["step_grad_ratio"], color=c, linestyle=d, **line)

    # (c) cosine, with zero marked
    ax = axes[1][0]
    style(ax, r"$\cos(g_{\mathrm{graph}},\, g_{\mathrm{KD}})$")
    ax.axhline(0, color=INK_3, linewidth=0.9, linestyle=(0, (2, 2)), zorder=1)
    for _, h, c, d in runs:
        ax.plot(steps, h["step_grad_cosine"], color=c, linestyle=d, **line)
    ax.set_ylim(-0.12, 0.12)
    ax.set_xlabel("training step", color=INK_2)

    # (d) sign flips
    ax = axes[1][1]
    style(ax, "fraction of update signs set by the graph term")
    for _, h, c, d in runs:
        ax.plot(steps, h["step_grad_signflip"], color=c, linestyle=d, **line)
    ax.set_xlabel("training step", color=INK_2)

    handles = [Line2D([], [], color=c, linestyle=d, linewidth=1.3, marker="o",
                      markersize=1.9, markeredgewidth=0) for _, _, c, d in RUNS]
    leg = fig.legend(handles, [lab for _, lab, _, _ in RUNS],
                     title="frozen when linearising", loc="lower center", ncol=4,
                     frameon=False, fontsize=6.5, handlelength=2.6,
                     columnspacing=1.4, bbox_to_anchor=(0.5, -0.02))
    leg.get_title().set_fontsize(6.5)
    leg.get_title().set_color(INK_2)
    for t in leg.get_texts():
        t.set_color(INK)

    fig.tight_layout(rect=(0, 0.08, 1, 1), h_pad=1.0, w_pad=1.6)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=220, bbox_inches="tight", facecolor="white")
    print("wrote", args.out)


if __name__ == "__main__":
    main()

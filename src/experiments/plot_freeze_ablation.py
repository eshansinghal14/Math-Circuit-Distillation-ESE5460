"""Render the freeze-ablation training curves for the paper appendix.

Reads the four graph-KD history files written by ``training/graph_kd.py`` and
produces a three-panel figure of accuracy against training step.

Encoding: the four runs are a 2x2 factorial, so the figure encodes it as one --
hue carries the RMSNorm freeze, line style carries the attention freeze. That
needs only two categorical hues instead of four, and it makes the finding legible
directly: the two RMSNorm-frozen runs separate as a pair from the two that are
not. The dashed rule on each panel is the untrained student, which is what makes
the sign change on 33_add visible rather than merely tabulated.

Usage:
    python -m experiments.plot_freeze_ablation
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OOD = ["222_add", "2222_add", "33_add", "21_mult"]

# Validated categorical slots 1-2 (scripts/validate_palette.js, light, --pairs all:
# all checks pass, worst normal-vision dE 33.6, worst CVD dE 24.7, contrast >= 3:1).
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8983"
GRID = "#e9e8e5"

# file stem -> (label, colour, linestyle). Hue = RMSNorm freeze, dash = attention freeze.
RUNS = [
    ("22_add_argdla", "no freeze", BLUE, (0, (4, 2))),
    ("22_add_argdla_attn", "attention", BLUE, "solid"),
    ("22_add_argdla_rms", "RMSNorm", ORANGE, (0, (4, 2))),
    ("22_add_argdla_attn+rms", "both", ORANGE, "solid"),
]


def _load(results_dir: str) -> dict:
    out = {}
    for stem, label, colour, dash in RUNS:
        path = os.path.join(results_dir, stem + ".json")
        if not os.path.exists(path):
            raise SystemExit(f"missing history file: {path}")
        with open(path, encoding="utf-8") as f:
            out[label] = (json.load(f), colour, dash)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default="results/freeze_ablation")
    ap.add_argument("--out", default="latex/graph_distillation/figures/freeze_ablation.pdf")
    args = ap.parse_args()

    runs = _load(args.results_dir)
    any_hist = next(iter(runs.values()))[0]
    steps = any_hist["accuracy_step"]

    def series(hist, panel):
        if panel == "22_add":
            return hist["accuracy"]
        if panel == "ood":
            return [sum(hist[f"accuracy_{o}"][i] for o in OOD) / len(OOD)
                    for i in range(len(hist["accuracy"]))]
        return hist[f"accuracy_{panel}"]

    panels = [
        ("22_add", "in-distribution (22_add)", any_hist["student_baseline"], any_hist["teacher_baseline"]),
        ("ood", "mean out-of-distribution", sum(any_hist[f"accuracy_{o}"][0] for o in OOD) / len(OOD), None),
        ("33_add", "three-digit addition (33_add)", any_hist["accuracy_33_add"][0], None),
    ]

    plt.rcParams.update({
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "axes.titlesize": 7.5, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.05))

    for ax, (panel, title, baseline, teacher) in zip(axes, panels):
        ax.set_facecolor("white")
        ax.yaxis.grid(True, color=GRID, linewidth=0.6, solid_capstyle="butt")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=INK_2, length=2, width=0.8)

        # Reference rules: dashed so they never read as the solid hairline grid.
        ax.axhline(baseline, color=INK_3, linewidth=0.9, linestyle=(0, (2, 2)), zorder=1)
        if teacher is not None:
            ax.axhline(teacher, color=INK_3, linewidth=0.9, linestyle=(0, (1, 2)), zorder=1)

        for label, (hist, colour, dash) in runs.items():
            ax.plot(steps, series(hist, panel), color=colour, linewidth=1.4,
                    linestyle=dash, solid_capstyle="round", dash_capstyle="round",
                    marker="o", markersize=2.2, markeredgewidth=0, zorder=3)

        ax.set_title(title, color=INK, pad=4)
        ax.set_xlabel("training step", color=INK_2)
        ax.set_xlim(-0.5, 15.5)
        ax.set_xticks([0, 5, 10, 15])

    axes[0].set_ylabel("accuracy", color=INK_2)
    # Rule labels sit in whitespace the curves leave empty, checked against the
    # rendered figure rather than assumed.
    axes[0].annotate("teacher", (0.3, panels[0][3]), color=INK_3, fontsize=6,
                     ha="left", va="bottom")
    axes[0].annotate("untrained student", (7.0, panels[0][2]), color=INK_3, fontsize=6,
                     ha="left", va="bottom")
    # No second copy in panel (c): every placement there collides with a curve.
    # Panel (a) establishes the dashed rule and the caption carries it.

    handles = [Line2D([], [], color=c, linestyle=d, linewidth=1.4, marker="o",
                      markersize=2.2, markeredgewidth=0)
               for _, _, c, d in RUNS]
    leg = fig.legend(handles, [label for _, label, _, _ in RUNS],
                     title="frozen when linearising", loc="lower center",
                     ncol=4, frameon=False, fontsize=6.5, handlelength=2.6,
                     columnspacing=1.4, bbox_to_anchor=(0.5, -0.02))
    leg.get_title().set_fontsize(6.5)
    leg.get_title().set_color(INK_2)
    for text in leg.get_texts():
        text.set_color(INK)

    fig.tight_layout(rect=(0, 0.11, 1, 1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=220, bbox_inches="tight", facecolor="white")
    print("wrote", args.out)


if __name__ == "__main__":
    main()

"""Render the structure-control figure for Appendix D: real against scrambled teacher target.

The scrambled arm trains graph distillation at the working lambda against a
teacher target whose rows are permuted by a fixed derangement, so the target
keeps the teacher's numbers and sparsity but says nothing about which supernode
routes to which. The history also logs, without gradient, the student's graph
loss against the *real* target. Four panels: in-distribution and mean held-out
accuracy for the real arm, the scrambled arm and the KD-only control; the graph
loss the scrambled arm trains on beside the loss both arms would have against
the real target; and the KD loss of all three, which coincide.

Usage (from the repository root):
    PYTHONPATH=src python -m experiments.plot_scramble
"""

from __future__ import annotations

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from experiments._paper_plots import (
    BLUE, CONTROL, FIG_DIR, INK, INK_2, INK_3, ORANGE, RC, end_labels, legend, line_with_band,
    load_run, rules, save, style,
)

DOT = (0, (1.2, 1.6))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--real", default="results/lambda_graph_study/22_add_argdla_lambda_g=0.1.json")
    ap.add_argument("--scrambled", default="results/baselines/22_add_argdla_scramble-teacher.json")
    ap.add_argument("--kd", default="results/standard_kd/22_add_standard_kd.json")
    ap.add_argument("--out", default=f"{FIG_DIR}/scramble.pdf")
    args = ap.parse_args()

    real = load_run(args.real, "real target")
    scr = load_run(args.scrambled, "scrambled target")
    kd = load_run(args.kd, "KD only")
    last = int(real.steps[-1])
    lam = real.config["lambda_graph"]

    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.9))
    arms = [(kd, INK_2, CONTROL["linestyle"], "KD only"), (real, BLUE, "solid", "real"), (scr, BLUE, DOT, "scrambled")]

    for ax, key, title in ((axes[0], "accuracy", "in-distribution (22_add)"), (axes[1], "ood", "mean out-of-distribution")):
        style(ax, title)
        ends = []
        for run, colour, ls, lab in arms:
            xs, mean = line_with_band(ax, run, key, colour, last, ls, 1.2 if run is kd else 1.4)
            ends.append([xs[-1], mean[-1], lab])
        rules(ax, real.baseline_of(key), real.teacher if key == "accuracy" else None)
        ax.set_xlim(-1, last + 14)
        ax.set_xticks(range(0, last + 1, 50))
        ax.set_ylim(0, 1.02 if key == "accuracy" else 0.62)
        ax.set_xlabel("training step", color=INK_2)
        end_labels(ax, ends, fontsize=5.5)
    axes[0].set_ylabel("accuracy", color=INK_2)
    axes[0].annotate("untrained student", (last, real.baseline), color=INK_3, fontsize=5.5, ha="right", va="bottom")

    # (c) graph losses
    ax = axes[2]
    style(ax, "graph loss")
    xs, g_real = real.step_series("step_graph_loss")
    _, g_scr = scr.step_series("step_graph_loss")
    _, g_scr_real = scr.step_series("step_graph_loss_real_target")
    ax.plot(xs, g_real, color=BLUE, linewidth=1.2, zorder=3)
    ax.plot(xs, g_scr, color=BLUE, linewidth=1.2, linestyle=DOT, zorder=3)
    ax.plot(xs, g_scr_real, color=ORANGE, linewidth=1.2, linestyle=DOT, zorder=3)
    ax.set_xlim(0, last + 1)
    ax.set_xticks(range(0, last + 1, 50))
    ax.set_ylim(0.2, 0.45)
    ax.set_xlabel("training step", color=INK_2)
    ax.annotate("scrambled arm,\nscrambled target", (last * 0.98, float(np.median(g_scr[-20:])) + 0.03),
                color=INK_2, fontsize=5.2, ha="right", va="bottom")
    ax.annotate("scrambled arm,\nreal target", (last * 0.98, float(np.median(g_scr_real[-20:])) + 0.012),
                color=INK_2, fontsize=5.2, ha="right", va="bottom")
    ax.annotate("real arm", (last * 0.98, float(np.median(g_real[-20:])) - 0.012), color=INK_2, fontsize=5.2,
                ha="right", va="top")

    # (d) KD losses coincide
    ax = axes[3]
    style(ax, "KD loss")
    for run, colour, ls, lab in arms:
        xs, y = run.step_series("step_kl_loss")
        keep = xs <= last
        ax.plot(xs[keep], y[keep], color=colour, linewidth=1.2, linestyle=ls, zorder=3)
    ax.set_xlim(0, last + 1)
    ax.set_xticks(range(0, last + 1, 50))
    ax.set_xlabel("training step", color=INK_2)
    _, k_real = real.step_series("step_kl_loss")
    _, k_scr = scr.step_series("step_kl_loss")

    handles = [Line2D([], [], color=INK_2, linestyle=CONTROL["linestyle"], linewidth=1.2),
               Line2D([], [], color=BLUE, linewidth=1.4),
               Line2D([], [], color=BLUE, linestyle=DOT, linewidth=1.4),
               Line2D([], [], color=ORANGE, linestyle=DOT, linewidth=1.2)]
    labels = [f"KD only ({kd.n} seeds)", f"graph, real target ($\\lambda = {lam:g}$)",
              "graph, scrambled target", "scrambled arm scored on the real target"]
    legend(fig, handles, labels, ncol=4, y=-0.04)
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=0.8)
    save(fig, args.out)

    s = scr.summary(last)
    r = real.summary(last)
    print(f"scrambled: in-dist@{last} {s['accuracy']['at'][0]:.3f} meanOOD {s['ood']['at'][0]:.3f} "
          f"(real {r['accuracy']['at'][0]:.3f} / {r['ood']['at'][0]:.3f}); ratio@1 {scr.hists[0]['step_grad_ratio'][0]:.4f} "
          f"vs real {real.hists[0]['step_grad_ratio'][0]:.4f}; mean ratio {np.mean(scr.hists[0]['step_grad_ratio']):.4f} "
          f"vs {np.mean(real.hists[0]['step_grad_ratio']):.4f}")
    print(f"graph loss scrambled target {g_scr[0]:.3f} -> {g_scr[-1]:.3f}; on real target {g_scr_real[0]:.3f} -> {g_scr_real[-1]:.3f}; "
          f"real arm {g_real[0]:.3f} -> {g_real[-1]:.3f}; KD loss max diff {np.abs(k_real - k_scr).max():.5f}")


if __name__ == "__main__":
    main()

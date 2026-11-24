"""Render the structure-control figure: real against scrambled teacher target.

The scrambled arm trains graph distillation at the working lambda against a
teacher target whose rows are permuted by a fixed derangement, so the target
keeps the teacher's numbers and sparsity but says nothing about which supernode
routes to which. The history also logs, without gradient, the student's graph
loss against the *real* target. Four panels: in-distribution and 21_mult
accuracy for the real arm, the scrambled arm and the KD-only baseline; the
graph loss the scrambled arm trains on beside the loss it would have against
the real target and the real arm's own loss; and the KD loss of all three.
Until the scrambled history exists the figure is drawn with the real arm and
the baseline only and the graph-loss panel carries a "pending" mark.

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
    OOD_MEAN, panel_title,
    BLUE, CONTROL, DOT, FIG_DIR, GRAPH_MAIN, INK_2, INK_3, KD_MAIN, ORANGE, RC, SCRAMBLED_MAIN,
    legend, line_with_band, reference_handles, rules, save, seed_lines, select_runs, style,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=f"{FIG_DIR}/scramble.pdf")
    args = ap.parse_args()

    real = select_runs(*GRAPH_MAIN, label="real target")
    scr = select_runs(*SCRAMBLED_MAIN, label="scrambled target", required=False)
    kd = select_runs(*KD_MAIN, label="KD only")
    last = int(real.steps[-1])
    lam = real.config["lambda_graph"]

    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.75))
    arms = [(kd, INK_2, CONTROL["linestyle"], "KD only"), (real, BLUE, "solid", "real")]
    if scr is not None:
        arms.append((scr, BLUE, DOT, "scrambled"))

    for ax, key, title in ((axes[0], "accuracy", panel_title("accuracy")), (axes[1], "21_mult", panel_title("21_mult"))):
        style(ax, title)
        for run, colour, ls, lab in arms:
            draw = seed_lines if run is scr else line_with_band
            draw(ax, run, key, colour, last, ls, 0.7 if run is scr else (1.2 if run is kd else 1.4))
        rules(ax, real.baseline_of(key), real.teacher_of(key))
        ax.set_xlim(-1, last + 2)
        ax.set_xticks(range(0, last + 1, 50))
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("training step", color=INK_2)
    axes[0].set_ylabel("accuracy", color=INK_2)

    # (c) graph losses: the real arm on its target, the scrambled arm on its own
    # target and, scored without gradient, on the real one.
    ax = axes[2]
    style(ax, "graph loss (JSD)")
    xs, g_real = real.step_series("step_graph_loss")
    ax.plot(xs, g_real, color=BLUE, linewidth=1.2, zorder=3)
    ax.set_xlim(0, last + 1)
    ax.set_xticks(range(0, last + 1, 50))
    ax.set_xlabel("training step", color=INK_2)
    ax.set_ylabel("loss", color=INK_2)
    if scr is not None:
        xs_s, g_scr = scr.step_series("step_graph_loss")
        _, g_scr_real = scr.step_series("step_graph_loss_real_target")
        ax.plot(xs_s, g_scr, color=BLUE, linewidth=1.2, linestyle=DOT, zorder=3)
        ax.plot(xs_s, g_scr_real, color=ORANGE, linewidth=1.2, linestyle=DOT, zorder=3)
    else:
        ax.annotate("scrambled arm: pending", (last * 0.5, float(np.nanmax(g_real)) * 0.9), color=INK_3, ha="center")

    # (d) KD losses
    ax = axes[3]
    style(ax, "KD loss")
    for run, colour, ls, lab in arms:
        xs, y = run.step_series("step_kl_loss")
        keep = xs <= last
        ax.plot(xs[keep], y[keep], color=colour, linewidth=1.2, linestyle=ls, zorder=3)
    ax.set_xlim(0, last + 1)
    ax.set_xticks(range(0, last + 1, 50))
    ax.set_xlabel("training step", color=INK_2)
    ax.set_ylabel("loss", color=INK_2)

    handles = [Line2D([], [], color=INK_2, linestyle=CONTROL["linestyle"], linewidth=1.2),
               Line2D([], [], color=BLUE, linewidth=1.4)]
    labels = [f"KD only ({kd.n} seeds)", f"graph, real target ($\\lambda = {lam:g}$, {real.n} seed{'s' if real.n > 1 else ''})"]
    if scr is not None:
        handles += [Line2D([], [], color=BLUE, linestyle=DOT, linewidth=1.4), Line2D([], [], color=ORANGE, linestyle=DOT, linewidth=1.2)]
        labels += [f"graph, scrambled target ({scr.n} seed{'s' if scr.n > 1 else ''}, per seed)",
                   "scrambled arm on the real target"]
    rh, rl = reference_handles()
    legend(fig, handles + rh, labels + rl, ncol=2, y=-0.10)
    fig.tight_layout(rect=(0, 0.22, 1, 1), w_pad=0.8)
    save(fig, args.out)

    r = real.summary()
    print(f"real: peak in-dist {r['accuracy']['sel'][0]:.3f}@{r['sel_steps']} 21_mult {r['21_mult']['sel'][0]:.3f} meanOOD {r['ood']['sel'][0]:.3f}; "
          f"graph loss {g_real[0]:.3f} -> {g_real[-1]:.3f}")
    if scr is not None:
        s = scr.summary()
        print(f"scrambled: peak in-dist {s['accuracy']['sel'][0]:.3f}@{s['sel_steps']} 21_mult {s['21_mult']['sel'][0]:.3f} last in-dist {s['accuracy']['last'][0]:.3f}; "
              f"loss on scrambled {g_scr[0]:.3f} -> {g_scr[-1]:.3f}, on real {g_scr_real[0]:.3f} -> {g_scr_real[-1]:.3f}")
    else:
        print("scrambled control: pending")


if __name__ == "__main__":
    main()

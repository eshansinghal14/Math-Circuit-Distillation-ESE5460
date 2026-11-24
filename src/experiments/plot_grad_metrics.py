"""Render the per-step gradient-metric figure and tables for the gradient appendix.

Reads what ``--track-grad-metrics`` records for the main graph-distillation run,
the scrambled control when it exists, and the lambda sweep: the KD and graph
gradient norms, their ratio, the cosine between them and the pre-clip gradient
norm. Writes ``figures/grad_metrics.pdf``, ``tables/grad_dynamics.tex`` (run
means per arm) and ``tables/canaries.tex`` (the step-1 scale-invariant checks
across lambda, which share weights and batch and so must agree).

Usage (from the repository root):
    PYTHONPATH=src python -m experiments.plot_grad_metrics
"""

from __future__ import annotations

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from experiments._paper_plots import (
    BLUE, DOT, FIG_DIR, GRAPH_FIXED, GRAPH_MAIN, GRAPH_SWEEP, INK, INK_2, INK_3, KD_PROTOCOL, RC, SCRAMBLED_MAIN,
    legend, save, select_runs, style, write_table,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=f"{FIG_DIR}/grad_metrics.pdf")
    args = ap.parse_args()

    graph = select_runs(*GRAPH_MAIN, label="BOS-free, $\\lambda = 3$")
    scr = select_runs(*SCRAMBLED_MAIN, label="BOS-free, $\\lambda = 3$, scrambled", required=False)
    bos_free = dict(KD_PROTOCOL, **GRAPH_FIXED, scramble_teacher_graph=False)
    sweep = {lam: r for lam in (0.3, 1.0, 3.0, 6.0, 10.0)
             if (r := select_runs(GRAPH_SWEEP, dict(bos_free, lambda_graph=lam), f"BOS-free, $\\lambda = {lam:g}$", required=False)) is not None}
    on = {lam: r for lam in (1.0, 3.0, 10.0)
          if (r := select_runs(GRAPH_SWEEP, dict(KD_PROTOCOL, **GRAPH_FIXED, bos_mode="on", kl_tokens="resp", lambda_graph=lam,
                                                 scramble_teacher_graph=False), f"standard BOS, $\\lambda = {lam:g}$", required=False)) is not None}
    last = int(graph.hists[0]["train_step"][-1])
    lam = graph.config["lambda_graph"]

    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.3))
    line = dict(linewidth=1.1, solid_capstyle="round", dash_capstyle="round", zorder=3)
    thin = dict(linewidth=0.6, alpha=0.45, zorder=2)

    def prep(ax, title):
        style(ax, title)
        ax.set_xlim(0, last + 1)
        ax.set_xticks(range(0, last + 1, 25))

    def draw(ax, key, log=False):
        xs, ys = graph.step_series(key, seeds=True)
        for y in ys:
            ax.plot(xs, y, color=BLUE, **thin)
        with np.errstate(all="ignore"):
            ax.plot(xs, np.nanmean(ys, axis=0), color=BLUE, **line)
        if scr is not None:
            xs_s, y_s = scr.step_series(key)
            ax.plot(xs_s, y_s, color=BLUE, linestyle=DOT, **line)
        if log:
            ax.set_yscale("log")

    ax = axes[0][0]
    prep(ax, "gradient norms (log)")
    draw(ax, "step_kl_gnorm", log=True)
    draw(ax, "step_graph_gnorm", log=True)
    _, kd0 = graph.step_series("step_kl_gnorm")
    _, g0 = graph.step_series("step_graph_gnorm")
    x_lab = int(last * 0.7)
    ax.annotate(r"$\|g_{\mathrm{KD}}\|$", (x_lab, np.nanmedian(kd0[x_lab - 10:x_lab + 10]) * 2.2), color=INK, fontsize=6.5, ha="center")
    ax.annotate(r"$\lambda\,\|g_{\mathrm{graph}}\|$", (x_lab, np.nanmedian(g0[x_lab - 10:x_lab + 10]) * 1.9), color=INK, fontsize=6.5, ha="center")
    ax.set_ylabel("norm", color=INK_2)

    ax = axes[0][1]
    prep(ax, r"$\lambda\|g_{\mathrm{graph}}\| / \|g_{\mathrm{KD}}\|$")
    draw(ax, "step_grad_ratio")
    ax.set_ylim(0, 0.6)

    ax = axes[1][0]
    prep(ax, r"$\cos(g_{\mathrm{graph}},\, g_{\mathrm{KD}})$")
    ax.axhline(0, color=INK_3, linewidth=0.9, linestyle=(0, (2, 2)), zorder=1)
    draw(ax, "step_grad_cosine")
    ax.set_ylim(-0.15, 0.15)
    ax.set_xlabel("training step", color=INK_2)

    ax = axes[1][1]
    prep(ax, "pre-clip gradient norm (log)")
    draw(ax, "step_clip_norm", log=True)
    ax.axhline(1.0, color=INK_3, linewidth=0.9, linestyle=(0, (2, 2)), zorder=1)
    ax.annotate("clip threshold", (last * 0.98, 1.0), color=INK_3, fontsize=6.5, ha="right", va="bottom")
    ax.set_xlabel("training step", color=INK_2)

    handles = [Line2D([], [], color=BLUE, linewidth=1.1)]
    labels = [f"graph distillation ($\\lambda = {lam:g}$, {graph.n} seed{'s' if graph.n > 1 else ''})"]
    if graph.n > 1:
        handles.append(Line2D([], [], color=BLUE, linewidth=0.6, alpha=0.45)); labels.append("individual seeds")
    if scr is not None:
        handles.append(Line2D([], [], color=BLUE, linestyle=DOT, linewidth=1.1)); labels.append("scrambled target")
    legend(fig, handles, labels, ncol=3, y=-0.04)
    fig.tight_layout(rect=(0, 0.08, 1, 1), h_pad=1.0, w_pad=1.6)
    save(fig, args.out)

    # ---- run means per arm ----------------------------------------------------
    rows = []
    arms = [graph] + ([scr] if scr is not None else []) + [r for l, r in sweep.items() if l != lam] + list(on.values())
    for r in arms:
        kd = r.step_series("step_kl_gnorm")[1]; g = r.step_series("step_graph_gnorm")[1]
        ratio = r.step_series("step_grad_ratio")[1]; cos = r.step_series("step_grad_cosine")[1]
        clip = r.step_series("step_clip_norm")[1]
        _, cos_seeds = r.step_series("step_grad_cosine", seeds=True)
        neg = np.nanmean(cos_seeds < 0)
        rows.append(f"    {r.label} & {np.nanmean(kd):.1f} & {np.nanmean(g):.2f} & {np.nanmean(ratio):.3f} & "
                    f"{np.nanmean(cos):+.4f} ({np.nanmean(np.abs(cos)):.3f}) & {neg:.2f} & "
                    f"{np.nanmin(clip):.1f}--{np.nanmax(clip):.1f} \\\\")
        print(f"{r.label}: |gKD| {kd[0]:.1f}->{np.nanmean(kd[-5:]):.1f} | l|gg| {g[0]:.2f}->{np.nanmean(g[-5:]):.2f} | ratio@1 {ratio[0]:.4f} mean {np.nanmean(ratio):.3f} "
              f"max {np.nanmax(ratio):.3f} | cos mean {np.nanmean(cos):+.4f} neg {neg:.2f} | clip {np.nanmin(clip):.1f}-{np.nanmax(clip):.1f}")
    if scr is None:
        rows.insert(1, "    BOS-free, $\\lambda = 3$, scrambled & " + " & ".join(["\\pending{}"] * 6) + " \\\\")
    write_table("grad_dynamics", rows,
                "columns: arm & |g_KD| & lambda|g_graph| & ratio & cos (mean |cos|) & frac steps cos<0 & pre-clip norm range; run means")

    # Step-1 accumulation loss across lambda (same weights and batch at step 1), printed
    # for the Appendix A precision paragraph; no longer tabulated in the paper.
    for lam_, r in sweep.items():
        h = r.hists[0]
        print(f"step 1 lambda={lam_:g}: |gKD| {h['step_kl_gnorm'][0]:.2f} |g|/lam {h['step_graph_gnorm'][0]/lam_:.4f} ratio/lam {h['step_grad_ratio'][0]/lam_:.5f} cos {h['step_grad_cosine'][0]:+.4f}")


if __name__ == "__main__":
    main()

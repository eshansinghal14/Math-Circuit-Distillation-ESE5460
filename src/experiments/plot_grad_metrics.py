"""Render the per-step gradient-metric figure and table for Appendix C.

Reads the same arms as plot_freeze_ablation.py but plots what
``--track-grad-metrics`` records: the KD and graph gradient norms, their ratio,
the cosine between them and the fraction of parameter entries whose update sign
the graph term decides. Writes ``figures/grad_metrics.pdf`` and
``tables/freeze_dynamics.tex`` (run means over the whole run, plus the
step-1 and step-100 values that the text quotes).

The figure carries two messages: the KD gradient collapses by an order of
magnitude over the run while the graph gradient stays within a factor of two,
so their ratio climbs without anyone scheduling it; and the arms lie on top of
one another on every panel, so nothing that separates them, if anything does,
is visible in these aggregates.

Encoding matches the accuracy figure: hue carries the RMSNorm freeze, line style
the attention freeze, aqua the constant-weighting arm.

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

from experiments._paper_plots import FIG_DIR, INK, INK_2, INK_3, RC, legend, save, style, write_table
from experiments.plot_freeze_ablation import load_arms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=f"{FIG_DIR}/grad_metrics.pdf")
    args = ap.parse_args()

    arms = load_arms()
    present = [(l, r, c, ls) for l, r, c, ls in arms if r is not None]
    ref = present[0][1]
    lam = ref.config["lambda_graph"]
    last = int(ref.hists[0]["train_step"][-1])

    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.3))
    line = dict(linewidth=1.1, solid_capstyle="round", dash_capstyle="round", zorder=3)

    def prep(ax, title):
        style(ax, title)
        ax.set_xlim(0, last + 1)
        ax.set_xticks(range(0, last + 1, 25))

    ax = axes[0][0]
    prep(ax, "gradient norms (log)")
    for _, r, c, ls in present:
        xs, kd = r.step_series("step_kl_gnorm")
        _, g = r.step_series("step_graph_gnorm")
        ax.plot(xs, kd, color=c, linestyle=ls, **line)
        ax.plot(xs, g, color=c, linestyle=ls, **line)
    ax.set_yscale("log")
    _, kd0 = ref.step_series("step_kl_gnorm")
    _, g0 = ref.step_series("step_graph_gnorm")
    x_lab = int(last * 0.75)
    ax.annotate(r"$\|g_{\mathrm{KD}}\|$", (x_lab, kd0[x_lab - 1] * 2.4), color=INK, fontsize=6.5, ha="center")
    ax.annotate(r"$\lambda\,\|g_{\mathrm{graph}}\|$", (x_lab, g0[x_lab - 1] * 2.6), color=INK, fontsize=6.5, ha="center")
    ax.set_ylabel("norm", color=INK_2)

    ax = axes[0][1]
    prep(ax, r"$\lambda\|g_{\mathrm{graph}}\| / \|g_{\mathrm{KD}}\|$")
    for _, r, c, ls in present:
        xs, y = r.step_series("step_grad_ratio")
        ax.plot(xs, y, color=c, linestyle=ls, **line)

    ax = axes[1][0]
    prep(ax, r"$\cos(g_{\mathrm{graph}},\, g_{\mathrm{KD}})$")
    ax.axhline(0, color=INK_3, linewidth=0.9, linestyle=(0, (2, 2)), zorder=1)
    for _, r, c, ls in present:
        xs, y = r.step_series("step_grad_cosine")
        ax.plot(xs, y, color=c, linestyle=ls, **line)
    ax.set_ylim(-0.12, 0.12)
    ax.set_xlabel("training step", color=INK_2)

    ax = axes[1][1]
    prep(ax, "fraction of update signs set by the graph term")
    for _, r, c, ls in present:
        xs, y = r.step_series("step_grad_signflip")
        ax.plot(xs, y, color=c, linestyle=ls, **line)
    ax.set_xlabel("training step", color=INK_2)

    handles = [Line2D([], [], color=c, linestyle=ls, linewidth=1.1) for _, _, c, ls in arms]
    labels = [l + ("" if r is not None else " (pending)") for l, r, _, _ in arms]
    legend(fig, handles, labels, ncol=3, title=f"linearisation, $\\lambda = {lam:g}$", y=-0.04)
    fig.tight_layout(rect=(0, 0.11, 1, 1), h_pad=1.0, w_pad=1.6)
    save(fig, args.out)

    # ---- table rows: run means ---------------------------------------------
    rows = []
    stats = {}
    for label, r, _, _ in arms:
        if r is None:
            rows.append(f"    {label} & " + " & ".join(["\\pending{}"] * 5) + " \\\\")
            continue
        kd = r.step_series("step_kl_gnorm")[1]; g = r.step_series("step_graph_gnorm")[1]
        ratio = r.step_series("step_grad_ratio")[1]; cos = r.step_series("step_grad_cosine")[1]
        sf = r.step_series("step_grad_signflip")[1]
        stats[label] = dict(kd=kd, g=g, ratio=ratio, cos=cos, sf=sf)
        rows.append(f"    {label} & {kd.mean():.2f} & {g.mean():.3f} & {ratio.mean():.3f} & "
                    f"{cos.mean():+.4f} ({np.abs(cos).mean():.3f}) & {sf.mean():.4f} \\\\")
    if len(stats) > 1:
        def spread(key):
            v = np.array([s[key].mean() for s in stats.values()])
            return (v.max() - v.min()) / v.mean() * 100
        rows.append("    \\midrule")
        rows.append(f"    spread across arms & {spread('kd'):.1f}\\% & {spread('g'):.1f}\\% & {spread('ratio'):.1f}\\% & --- & {spread('sf'):.1f}\\% \\\\")
    write_table("freeze_dynamics", rows, "columns: |g_KD| & lambda|g_graph| & ratio & cos (mean |cos|) & sign flips; run means over all steps")
    for label, s in stats.items():
        print(f"{label}: |gKD| {s['kd'][0]:.2f}->{s['kd'][-1]:.2f} (x{s['kd'][0]/s['kd'][-1]:.1f}) | l|gg| {s['g'][0]:.3f}->{s['g'][-1]:.3f} "
              f"| ratio {s['ratio'][0]:.4f}->{s['ratio'][-1]:.4f} max {s['ratio'].max():.3f} | cos@1 {s['cos'][0]:+.4f} mean {s['cos'].mean():+.4f} "
              f"range [{s['cos'].min():+.3f},{s['cos'].max():+.3f}] | sf {s['sf'][0]:.4f}->{s['sf'][-1]:.4f}")


if __name__ == "__main__":
    main()

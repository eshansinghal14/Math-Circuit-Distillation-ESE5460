"""Render the linearisation-ablation training curves and table for Appendix C.

The arms are graph-KD runs at the working lambda that differ only in what is
stop-gradiented when the edge attribution is linearised (attention pattern,
RMSNorm denominator, both, neither) plus one arm that keeps the unfrozen
Jacobian but replaces the frac_external member weighting of the supernode
aggregation with a constant. Each arm is a history file; the unfrozen arm is the
lambda-sweep run at the same weight, since it is the identical configuration.
An arm whose file is missing is skipped with a notice and its table row is left
as a reserved slot, so the figure can be regenerated as arms come in.

Produces ``figures/freeze_ablation.pdf``, three panels: in-distribution and mean
out-of-distribution accuracy over the run, and the first forty steps of the
in-distribution curve where the runs differ most (the dip). Writes
``tables/freeze_train.tex``.

Encoding: hue carries the RMSNorm freeze (blue without, orange with) and line
style the attention freeze (dashed without, solid with), so the 2x2 design reads
as two pairs; the constant-weighting arm is the third categorical slot (aqua).
Seed means with a +-1 sd band when a file holds several seeds.

Usage (from the repository root):
    PYTHONPATH=src python -m experiments.plot_freeze_ablation
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from experiments._paper_plots import (
    AQUA, BLUE, FIG_DIR, INK, INK_2, INK_3, OOD, ORANGE, RC, legend,
    line_with_band, load_run, metric_table, pm, rules, save, style,
)

DASH = (0, (4, 2))
# (label, path, colour, linestyle). Hue = RMSNorm freeze, dash = attention freeze.
ARMS = [
    ("unfrozen", "results/lambda_graph_study/22_add_argdla_lambda_g=0.1.json", BLUE, DASH),
    ("attention frozen", "results/freeze_ablation/22_add_argdla_attn-only.json", BLUE, "solid"),
    ("RMSNorm frozen", "results/freeze_ablation/22_add_argdla_rmsnorm-only.json", ORANGE, DASH),
    ("both frozen", "results/freeze_ablation/22_add_argdla_fullfreeze.json", ORANGE, "solid"),
    ("unfrozen, constant weighting", "results/freeze_ablation/22_add_argdla_frac_external=1.json", AQUA, DASH),
]


def load_arms():
    arms = []
    for label, path, colour, ls in ARMS:
        if not os.path.exists(path):
            print(f"arm '{label}' not run yet: {path} missing; leaving its slot reserved")
            arms.append((label, None, colour, ls))
            continue
        arms.append((label, load_run(path, label), colour, ls))
    return arms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--at-step", type=int, default=100)
    ap.add_argument("--out", default=f"{FIG_DIR}/freeze_ablation.pdf")
    args = ap.parse_args()

    arms = load_arms()
    present = [(l, r, c, ls) for l, r, c, ls in arms if r is not None]
    if not present:
        raise SystemExit("no ablation arms found")
    ref = present[0][1]
    last = args.at_step
    baseline, teacher, ood_baseline = ref.baseline, ref.teacher, ref.baseline_of("ood")
    lam = ref.config["lambda_graph"]

    plt.rcParams.update(RC)
    fig, (ax_id, ax_ood, ax_dip) = plt.subplots(1, 3, figsize=(5.5, 2.05))
    for ax, key, title, xmax in ((ax_id, "accuracy", "in-distribution (22_add)", last),
                                 (ax_ood, "ood", "mean out-of-distribution", last),
                                 (ax_dip, "accuracy", "in-distribution, first 40 steps", 40)):
        style(ax, title)
        ends = []
        for label, run, colour, ls in present:
            xs, mean = line_with_band(ax, run, key, colour, xmax, ls, marker=(xmax == 40))
            ends.append([xs[-1], mean[-1], label.replace(", constant weighting", ",\nconst. wt.")])
        rules(ax, baseline if key == "accuracy" else ood_baseline, teacher if key == "accuracy" else None)
        ax.set_xlabel("training step", color=INK_2)
        if xmax == 40:
            ax.set_xlim(-0.5, 46)
            ax.set_xticks([0, 10, 20, 30, 40])
            ax.set_ylim(0.3, 1.02)
            ax.annotate("untrained student", (46, baseline), color=INK_3, fontsize=6, ha="right", va="bottom")
        else:
            ax.set_xlim(-1, last + 3)
            ax.set_xticks(range(0, last + 1, 25))
            ax.set_ylim(0, 1.02 if key == "accuracy" else 0.66)
            if key == "accuracy":
                ax.annotate("teacher", (2, teacher), color=INK_3, fontsize=6, ha="left", va="top")
                ax.annotate("untrained student", (last, baseline), color=INK_3, fontsize=6, ha="right", va="top")
    ax_id.set_ylabel("accuracy", color=INK_2)

    handles = [Line2D([], [], color=c, linestyle=ls, linewidth=1.4) for _, _, c, ls in arms]
    labels = [l + ("" if r is not None else " (pending)") for l, r, _, _ in arms]
    legend(fig, handles, labels, ncol=3, title=f"linearisation, $\\lambda = {lam:g}$", y=-0.06)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    save(fig, args.out)

    # ---- table ---------------------------------------------------------------
    HEAD = {"unfrozen": "unfrozen", "attention frozen": "attention\nfrozen", "RMSNorm frozen": "RMSNorm\nfrozen",
            "both frozen": "both\nfrozen", "unfrozen, constant weighting": "unfrozen,\nconstant wt."}
    metric_table("freeze_train", [(HEAD[l], r) for l, r, _, _ in arms], ref, last)
    S = [(l, r, r.summary(last) if r is not None else None) for l, r, _, _ in arms]
    for label, run, s in S:
        if s is not None:
            print(f"{label}: in@{last} {pm(s['accuracy']['at'])} ood@{last} {pm(s['ood']['at'])} peak {pm(s['ood']['peak'])}@{s['ood']['peak_step']} dip {s['dip']:.3f}@{s['dip_step']} rec {s['recovery']}")


if __name__ == "__main__":
    main()

"""Render the lambda-sweep figure and table for Appendix D.

Reads every graph-KD history in the lambda study directory (every seed each
file holds), orders them by ``lambda_graph``, and adds two references: the
standard-KD control (lambda = 0, trained longer so its peak is not cut off) and
supervised fine-tuning on the same prompts. All three come from the same
trainer stack and eval.

Produces ``figures/lambda_sweep.pdf``, three panels: in-distribution and mean
out-of-distribution accuracy against training step, and peak mean OOD accuracy
against lambda on a log axis with the two references as horizontal rules.
Writes ``tables/lambda.tex``: one row per arm with peak in-distribution, peak
mean OOD (step), mean OOD at step 100, each held-out family at step 100, and the
dip / recovery of the in-distribution curve. Seed mean +- sd where a file holds
several seeds.

Encoding: lambda is ordered, so the graph runs take an ordinal ramp of one hue
(light to dark with increasing lambda); SFT is orange, the control dashed grey.
Curves longer than the graph runs are clipped to share the axis; their later
peaks are in the table and on the right-hand panel.

Usage (from the repository root):
    PYTHONPATH=src python -m experiments.plot_lambda_sweep
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from experiments._paper_plots import (
    CONTROL, FIG_DIR, INK, INK_2, INK_3, OOD, ORANGE, RAMP, RC, RULE_UNTRAINED, Run, end_labels,
    legend, line_with_band, load_run, metric_table, optional_run, pm, rules, save, style,
)


def _load_sweep(results_dir: str) -> list[tuple[float, Run]]:
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f).get("config")
        if not cfg:
            print(f"skipping {os.path.basename(path)}: no config record (pre-fp32 history)")
            continue
        lam = float(cfg["lambda_graph"])
        runs.append((lam, load_run(path, f"$\\lambda = {lam:g}$")))
    if not runs:
        raise SystemExit(f"no fp32-regime histories under {results_dir}")
    runs.sort(key=lambda r: r[0])
    if len(runs) > len(RAMP):
        raise SystemExit(f"{len(runs)} runs but the ordinal ramp has {len(RAMP)} steps")
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default="results/lambda_graph_study")
    ap.add_argument("--kd", default="results/standard_kd/22_add_standard_kd.json")
    ap.add_argument("--sft", default="results/sft/22_add.json")
    ap.add_argument("--at-step", type=int, default=100)
    ap.add_argument("--out", default=f"{FIG_DIR}/lambda_sweep.pdf")
    args = ap.parse_args()

    sweep = _load_sweep(args.results_dir)
    kd = optional_run(args.kd, "KD only ($\\lambda = 0$)")
    sft = optional_run(args.sft, "SFT")
    colours = RAMP[-len(sweep):]
    ref = sweep[0][1]
    baseline, teacher, ood_baseline = ref.baseline, ref.teacher, ref.baseline_of("ood")
    last = args.at_step

    plt.rcParams.update(RC)
    fig, (ax_id, ax_ood, ax_lam) = plt.subplots(1, 3, figsize=(5.5, 2.05))

    # ---- (a), (b): trajectories ---------------------------------------------
    for ax, key in ((ax_id, "accuracy"), (ax_ood, "ood")):
        style(ax, "in-distribution (22_add)" if key == "accuracy" else "mean out-of-distribution")
        ax.set_xlim(-1, last + 12)
        ax.set_xticks(range(0, last + 1, 25))
        ax.set_xlabel("training step", color=INK_2)
        ends = []
        if kd is not None:
            xs, mean = line_with_band(ax, kd, key, INK_2, last, CONTROL["linestyle"], 1.2, zorder=2)
            ends.append([xs[-1], mean[-1], "KD only"])
        if sft is not None:
            xs, mean = line_with_band(ax, sft, key, ORANGE, last, "solid", 1.2, zorder=2)
            ends.append([xs[-1], mean[-1], "SFT"])
        for (lam, run), colour in zip(sweep, colours):
            xs, mean = line_with_band(ax, run, key, colour, last)
            ends.append([xs[-1], mean[-1], f"{lam:g}"])
        rules(ax, baseline if key == "accuracy" else ood_baseline, teacher if key == "accuracy" else None)
        if key == "accuracy":
            ax.set_ylim(0, 1.02)
            ax.annotate("teacher", (2, teacher), color=INK_3, fontsize=6, ha="left", va="top")
        else:
            ax.set_ylim(0, 0.66)
        ax.annotate("untrained student", (last, baseline if key == "accuracy" else ood_baseline),
                    color=INK_3, fontsize=6, ha="right", va="top")
        end_labels(ax, ends)
    ax_id.set_ylabel("accuracy", color=INK_2)

    # ---- (c): peak OOD against lambda ----------------------------------------
    sums = {lam: run.summary(last) for lam, run in sweep}
    lams = [lam for lam, _ in sweep]
    peaks = [sums[lam]["ood"]["peak"][0] for lam in lams]
    errs = [sums[lam]["ood"]["peak"][1] for lam in lams]
    style(ax_lam, "peak mean OOD vs. $\\lambda$")
    ax_lam.set_xscale("log")
    ax_lam.plot(lams, peaks, color=INK_2, linewidth=1.0, zorder=2)
    for lam, pk, err, colour in zip(lams, peaks, errs, colours):
        if err is not None:
            ax_lam.errorbar([lam], [pk], yerr=[err], color=colour, linewidth=0.8, capsize=1.5, zorder=3)
        ax_lam.plot([lam], [pk], marker="o", markersize=5, color=colour, markeredgewidth=0, zorder=4)
    ax_lam.axhline(ood_baseline, **RULE_UNTRAINED)
    x0 = lams[0] * 0.72
    if kd is not None:
        c = kd.summary(kd.steps[-1])
        ax_lam.axhline(c["ood"]["peak"][0], **CONTROL)
        ax_lam.annotate(f"KD only, peak (step {c['ood']['peak_step']})", (x0, c["ood"]["peak"][0] + 0.006),
                        color=INK_2, fontsize=6, ha="left", va="bottom")
    if sft is not None:
        s = sft.summary(last)
        ax_lam.axhline(s["ood"]["peak"][0], color=ORANGE, linewidth=1.0, zorder=1)
        ax_lam.annotate("SFT, peak", (x0, s["ood"]["peak"][0] + 0.006), color=INK_2, fontsize=6, ha="left", va="bottom")
    ax_lam.annotate("untrained student", (x0, ood_baseline), color=INK_3, fontsize=6, ha="left", va="bottom")
    ax_lam.set_xlim(lams[0] * 0.6, lams[-1] * 1.8)
    ax_lam.set_xticks(lams)
    ax_lam.set_xticklabels([f"{lam:g}" for lam in lams])
    ax_lam.set_ylim(0.1, 0.66)
    ax_lam.set_xlabel("$\\lambda_{\\mathrm{graph}}$ (log)", color=INK_2)

    handles = [Line2D([], [], color=c, linewidth=1.4) for c in colours]
    labels = [f"$\\lambda={lam:g}$" for lam in lams]
    if sft is not None:
        handles.insert(0, Line2D([], [], color=ORANGE, linewidth=1.2))
        labels.insert(0, "SFT")
    if kd is not None:
        handles.insert(0, Line2D([], [], color=INK_2, linewidth=1.2, linestyle=CONTROL["linestyle"]))
        labels.insert(0, "KD only ($\\lambda=0$)")
    legend(fig, handles, labels, ncol=len(handles))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save(fig, args.out)

    # ---- table ---------------------------------------------------------------
    def seeds(run):
        return f"{run.n} seed{'s' if run.n > 1 else ''}"

    columns = []
    if sft is not None:
        columns.append((f"SFT\n({seeds(sft)})", sft))
    if kd is not None:
        columns.append((f"KD only, $\\lambda = 0$\n({seeds(kd)}, {int(kd.steps[-1])} steps)", kd))
    for lam, run in sweep:
        columns.append((f"$\\lambda = {lam:g}$\n({seeds(run)})", run))
    metric_table("lambda", columns, ref, last)
    for lam, run in sweep:
        h = run.hists[0]
        print(f"% lambda={lam:g} seed {run.seeds[0]}: step1 |g_KD|={h['step_kl_gnorm'][0]:.3f} "
              f"|g_graph|/lambda={h['step_graph_gnorm'][0] / lam:.3f} cos={h['step_grad_cosine'][0]:+.4f} "
              f"ratio/lambda={h['step_grad_ratio'][0] / lam:.4f} canary={h.get('params_changed_step1')}")


if __name__ == "__main__":
    main()

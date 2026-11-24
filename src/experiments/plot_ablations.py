"""Tabulate and plot the graph-distillation ablations on 22_add.

Three things, all from ``results/anova_study/22_add_anova-sweep.json`` and the
standard-KD sweep, one seed each unless stated:

  lambda sweep      BOS-free training, 100 steps, lambda in {0.3, 1, 3, 6, 10}; the
                    lambda = 10 run stopped early and is marked by its last step.
  supernode grid    50 steps, lambda = 3: neurons per label in {3, 10, 30} against
                    the teacher's pre-selection fraction in {0.003, 0.03, 0.1}.
  standard-BOS ablation   BOS in the training sequence: standard KD (KD on all positions or
                    answer positions) against graph distillation at lambda in
                    {1, 3, 10} with the KD term on answer positions or on all
                    positions except BOS.

Writes ``tables/lambda_sweep.tex``, ``tables/supernode_grid.tex``,
``tables/bos_on.tex`` and ``figures/ablations.pdf`` (accuracy against lambda on
a log axis: in-distribution, mean out-of-distribution and 21_mult, with the
standard-KD baseline as a rule with its seed band).

Usage (from the repository root):
    PYTHONPATH=src python -m experiments.plot_ablations
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from experiments._paper_plots import (
    OOD_MEAN, panel_title,
    ANOVA_LABELS, BLUE, FAM_TEX, FIG_DIR, GRAPH_FIXED, GRAPH_SWEEP, INK_2, INK_3, KD_MAIN, KD_PROTOCOL, KD_SWEEP, ORANGE, RC,
    Run, hdr, legend, pm, reference_handles, save, select_runs, steps_str, style, write_table,
)

LEGACY = dict(KD_PROTOCOL, **GRAPH_FIXED, scramble_teacher_graph=False)


def row_cells(r: Run, ref: Run, until: Optional[int] = None) -> Tuple[List[str], dict]:
    s = r.summary(until)
    cells = [pm(s["accuracy"]["sel"]) + f" ({steps_str(s['sel_steps'])})"]
    for o in ref.ood:
        cells.append(pm(s[o]["sel"], below=ref.baseline_of(o)))
    cells.append(pm(s["ood"]["sel"]))
    cells.append(pm(s["ood"]["peak"]) + f" ({s['ood']['peak_step']})")
    return cells, s


def header(ref: Run, first: List[str]) -> List[str]:
    cols = first + [FAM_TEX[ref.dataset] + " peak (step)"] + [FAM_TEX[o] for o in ref.ood] + [OOD_MEAN, "peak OOD (step)"]
    return ["\\begin{tabular}{" + "l" * len(first) + "c" * (len(cols) - len(first)) + "}", "  \\toprule",
            "  " + " & ".join(hdr(c) if i >= len(first) else c for i, c in enumerate(cols)) + " \\\\", "  \\midrule"]


def ref_rows(ref: Run, nfirst: int) -> List[str]:
    pad = " & ".join(["---"] * (nfirst - 1))
    t = ref.teacher_of
    return [f"  untrained student & {pad} & {ref.baseline:.3f} & " + " & ".join(f"{ref.baseline_of(o):.3f}" for o in ref.ood) + f" & {ref.baseline_of('ood'):.3f} & --- \\\\",
            f"  teacher & {pad} & {t('accuracy'):.3f} & " + " & ".join(f"{t(o):.3f}" for o in ref.ood) + f" & {t('ood'):.3f} & --- \\\\"]


def lambda_sweep(kd: Run) -> List[Tuple[float, Run]]:
    out = []
    for lam in (0.3, 1.0, 3.0, 6.0, 10.0):
        r = select_runs(GRAPH_SWEEP, dict(LEGACY, lambda_graph=lam), f"lambda={lam:g}", required=False)
        if r is not None:
            out.append((lam, r))
    lines = header(kd, ["$\\lambda$", "seeds", "steps run"]) + ref_rows(kd, 3)
    lines.append(f"  0 (standard KD) & {kd.n} & 100 & " + " & ".join(row_cells(kd, kd)[0]) + " \\\\")
    for lam, r in out:
        cells, s = row_cells(r, kd)
        lines.append(f"  {lam:g} & {r.n} & {steps_str(s['last_steps'])} & " + " & ".join(cells) + " \\\\")
        print(f"lambda={lam:g}: n={r.n} last {s['last_steps']} in-dist {pm(s['accuracy']['sel'])} 21_mult {pm(s['21_mult']['sel'])} 222 {pm(s['222_add']['sel'])} meanOOD {pm(s['ood']['sel'])} "
              f"ratio@1 {r.hists[0]['step_grad_ratio'][0]:.4f} ratio mean {np.mean(r.hists[0]['step_grad_ratio']):.3f}")
    lines += ["  \\bottomrule", "\\end{tabular}"]
    write_table("lambda_sweep", lines, "graph-loss weight sweep under the shared protocol; each seed at its own best in-distribution step")
    return out


def supernode_grid(kd: Run) -> None:
    lines = header(kd, ["$n_{\\mathrm{per}}$", "teacher top-$k$"]) + ref_rows(kd, 2)
    lines.append(f"  --- (standard KD, 100 steps) & --- & " + " & ".join(row_cells(kd, kd, until=50)[0]) + " \\\\")
    for npl in (3, 10, 30):
        for tp in (0.003, 0.03, 0.1):
            where = dict(LEGACY, lambda_graph=3.0, steps=50, nodes_per_label=npl, teacher_prop_neurons_per_layer=tp)
            r = select_runs(GRAPH_SWEEP, where, f"npl={npl} tp={tp}", required=False)
            if r is None:
                continue
            cells, s = row_cells(r, kd)
            lines.append(f"  {npl} & {tp:g} & " + " & ".join(cells) + " \\\\")
            print(f"npl={npl} tp={tp}: in-dist {pm(s['accuracy']['sel'])} 21_mult {pm(s['21_mult']['sel'])} 222 {pm(s['222_add']['sel'])} meanOOD {pm(s['ood']['sel'])}")
    lines += ["  \\bottomrule", "\\end{tabular}"]
    write_table("supernode_grid", lines, "neurons per label x teacher pre-selection fraction, 50 steps, lambda = 3; KD row read up to step 50")


def bos_on() -> None:
    on = dict(KD_PROTOCOL, bos_mode="on")
    kd_all = select_runs(KD_SWEEP, dict(on, kl_tokens="all"), "KD, all positions")
    kd_resp = select_runs(KD_SWEEP, dict(on, kl_tokens="resp"), "KD, answer positions")
    ref = kd_all
    lines = header(ref, ["arm", "KD positions", "$\\lambda$"]) + ref_rows(ref, 3)
    for lab, r in (("standard KD", kd_all), ("standard KD", kd_resp)):
        cells, s = row_cells(r, ref)
        lines.append(f"  {lab} & {r.config['kl_tokens']} & --- & " + " & ".join(cells) + " \\\\")
        print(f"BOS on KD {r.config['kl_tokens']}: 21_mult {pm(s['21_mult']['sel'])} in-dist {pm(s['accuracy']['sel'])}")
    lines.append("  \\midrule")
    for kl in ("resp", "all-no-bos"):
        for lam in (1.0, 3.0, 10.0):
            where = dict(on, **GRAPH_FIXED, kl_tokens=kl, lambda_graph=lam, scramble_teacher_graph=False)
            r = select_runs(GRAPH_SWEEP, where, f"graph {kl} {lam}", required=False)
            if r is None:
                continue
            cells, s = row_cells(r, ref)
            lines.append(f"  graph distillation & {kl} & {lam:g} & " + " & ".join(cells) + " \\\\")
            print(f"BOS on graph {kl} lambda={lam:g}: 21_mult {pm(s['21_mult']['sel'])} in-dist {pm(s['accuracy']['sel'])} meanOOD {pm(s['ood']['sel'])}")
    lines += ["  \\bottomrule", "\\end{tabular}"]
    write_table("bos_on", lines, "the standard BOS convention: KD baselines against graph distillation at three lambdas, one seed each")


def figure(sweep: List[Tuple[float, Run]], kd: Run, out: str) -> None:
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.7))
    keys = [(k, panel_title(k)) for k in ("accuracy", "ood", "21_mult")]
    ks = kd.summary()
    for ax, (key, title) in zip(axes, keys):
        style(ax, title)
        # A run that stopped before step 50 has not reached its peak; leave it to the table.
        lams = [lam for lam, r in sweep if r.summary()["last_steps"][0] >= 50]
        vals = [r.summary()[key]["sel"][0] for lam, r in sweep if r.summary()["last_steps"][0] >= 50]
        ax.plot(lams, vals, color=BLUE, marker="o", markersize=3, linewidth=1.3, zorder=4)
        m, sd = ks[key]["sel"]
        ax.axhline(m, color=INK_2, linewidth=1.1, linestyle=(0, (4, 2)), zorder=2)
        if sd is not None:
            ax.axhspan(m - sd, m + sd, color=INK_2, alpha=0.12, linewidth=0, zorder=1)
        ax.axhline(kd.baseline_of(key), color=INK_3, linewidth=0.9, linestyle=(0, (2, 2)), zorder=1)
        t = kd.teacher_of(key)
        if t is not None:
            ax.axhline(t, color=INK_3, linewidth=0.9, linestyle=(0, (1, 2)), zorder=1)
        ax.set_xscale("log")
        ax.set_xticks(lams)
        ax.set_xticklabels([f"{l:g}" for l in lams])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlim(min(lams) / 1.4, max(lams) * 1.4)
        ax.set_xlabel(r"$\lambda$", color=INK_2)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("accuracy", color=INK_2)
    handles = [Line2D([], [], color=BLUE, marker="o", markersize=3, linewidth=1.3),
               (Patch(facecolor=INK_2, alpha=0.12, linewidth=0), Line2D([], [], color=INK_2, linewidth=1.1, linestyle=(0, (4, 2))))]
    labels = ["graph distillation (1 seed per $\\lambda$)", f"standard KD ({kd.n} seeds, mean $\\pm$ 1 sd)"]
    rh, rl = reference_handles()
    legend(fig, handles + rh, labels + rl, ncol=2, y=-0.08)
    fig.tight_layout(rect=(0, 0.18, 1, 1), w_pad=1.0)
    save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=f"{FIG_DIR}/ablations.pdf")
    args = ap.parse_args()
    kd = select_runs(*KD_MAIN, label="standard KD")
    sweep = lambda_sweep(kd)
    supernode_grid(kd)
    bos_on()
    figure(sweep, kd, args.out)


if __name__ == "__main__":
    main()

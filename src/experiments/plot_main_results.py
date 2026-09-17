"""Render the main comparison figure and table: graph distillation against SFT and standard KD.

Reads the graph-KD run at the working lambda, the SFT run and the standard-KD
control (each with every seed its history holds) and draws accuracy against
training step: in-distribution, mean out-of-distribution, and each held-out
family on its own panel. Seed means are drawn as lines with a +-1 sd band where
there is more than one seed.

Writes ``figures/main_results.pdf`` and ``tables/main_results.tex`` (the rows of
the main results table: step-100 accuracy per family with seed mean +- sd, peak
mean OOD, and the untrained student and teacher references).

Encoding: three categorical hues in fixed order -- graph distillation blue, SFT
orange, standard KD grey dashed as the reference method -- the same assignment
every figure in the paper uses. The dashed rule is the untrained student, the
dotted rule the teacher.

Usage (from the repository root):
    PYTHONPATH=src python -m experiments.plot_main_results
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from experiments._paper_plots import (
    BLUE, CONTROL, FIG_DIR, INK, INK_2, INK_3, OOD, ORANGE, RC, Run, end_labels, legend,
    line_with_band, load_run, metric_table, optional_run, pm, rules, save, style,
)


SHORT = {"graph distillation": "graph KD", "SFT": "SFT", "standard KD": "KD", "TinyBERT-style KD": "TinyBERT", "CKA KD": "CKA"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--graph", default="results/lambda_graph_study/22_add_argdla_lambda_g=0.1.json")
    ap.add_argument("--sft", default="results/sft/22_add.json")
    ap.add_argument("--kd", default="results/standard_kd/22_add_standard_kd.json")
    ap.add_argument("--bert", default="results/baselines/22_add_bert_lambda_r=0.1.json")
    ap.add_argument("--cka", default="results/baselines/22_add_cka_lambda_r=0.1.json")
    ap.add_argument("--at-step", type=int, default=100)
    ap.add_argument("--out", default=f"{FIG_DIR}/main_results.pdf")
    args = ap.parse_args()

    graph = load_run(args.graph, "graph distillation")
    sft = load_run(args.sft, "SFT")
    kd = load_run(args.kd, "standard KD")
    bert = optional_run(args.bert, "TinyBERT-style KD")
    cka = optional_run(args.cka, "CKA KD")
    last = args.at_step
    # Output- and representation-level distillation share the grey of the reference
    # method and differ by line style; SFT and graph distillation keep their hues.
    arms = [(kd, INK_2, CONTROL["linestyle"])]
    if bert is not None:
        arms.append((bert, INK_2, (0, (1.2, 1.6))))
    if cka is not None:
        arms.append((cka, INK_2, (0, (4, 1.5, 1, 1.5))))
    arms += [(sft, ORANGE, "solid"), (graph, BLUE, "solid")]
    baseline, teacher = graph.baseline, graph.teacher

    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.6))
    panels = [("accuracy", "in-distribution (22_add)"), ("ood", "mean out-of-distribution")] + \
             [(o, f"held out: {o}") for o in OOD]
    axes = [axes[0][0], axes[0][1], axes[0][2], axes[1][0], axes[1][1], axes[1][2]]
    order = ["accuracy", "ood", "222_add", "2222_add", "33_add", "21_mult"]
    for ax, key in zip(axes, order):
        title = dict(panels)[key]
        style(ax, title)
        ax.set_xlim(-1, last + 14)
        ax.set_xticks(range(0, last + 1, 25))
        ends = []
        for run, colour, ls in arms:
            xs, mean = line_with_band(ax, run, key, colour, last_step=last, linestyle=ls,
                                      linewidth=1.4 if run in (sft, graph) else 1.1)
            ends.append([xs[-1], mean[-1], SHORT[run.label]])
        rules(ax, graph.baseline_of(key), teacher if key == "accuracy" else None)
        if key == "accuracy":
            ax.set_ylim(0, 1.02)
        else:
            top = max(run.series(key, last)[1].max() for run, _, _ in arms)
            ax.set_ylim(0, min(1.02, top * 1.25 + 0.02))
        end_labels(ax, ends, fontsize=5.5)
    axes[0].annotate("teacher", (2, teacher), color=INK_3, fontsize=6, ha="left", va="top")
    axes[0].annotate("untrained student", (last, baseline), color=INK_3, fontsize=6, ha="right", va="top")
    for ax in axes[3:]:
        ax.set_xlabel("training step", color=INK_2)
    axes[0].set_ylabel("accuracy", color=INK_2)
    axes[3].set_ylabel("accuracy", color=INK_2)

    handles = [Line2D([], [], color=c, linestyle=ls, linewidth=1.4) for _, c, ls in arms]
    labels = [f"{run.label} ({run.n} seed{'s' if run.n > 1 else ''})" for run, _, _ in arms]
    legend(fig, handles, labels, ncol=3, y=-0.03)
    fig.tight_layout(rect=(0, 0.09, 1, 1), h_pad=1.2, w_pad=1.0)
    save(fig, args.out)

    # ---- table ---------------------------------------------------------------
    sums = {run.label: run.summary(last) for run, _, _ in arms}

    def seeds(run):
        return f"{run.n} seed{'s' if run.n > 1 else ''}"

    columns = [(f"standard KD\n({seeds(kd)})", kd)]
    if bert is not None:
        columns.append((f"TinyBERT\n({seeds(bert)})", bert))
    if cka is not None:
        columns.append((f"CKA\n({seeds(cka)})", cka))
    columns += [(f"SFT\n({seeds(sft)})", sft),
                (f"graph distillation\n($\\lambda = {graph.config['lambda_graph']:g}$, {seeds(graph)})", graph)]
    metric_table("main_results", columns, graph, last, teacher_column=False)
    for run, _, _ in arms:
        s = sums[run.label]
        print(f"{run.label}: seeds={run.seeds} in-dist@{last} {pm(s['accuracy']['at'])} meanOOD@{last} {pm(s['ood']['at'])} "
              f"peak OOD {pm(s['ood']['peak'])}@{s['ood']['peak_step']} dip {s['dip']:.3f}@{s['dip_step']} recovery {s['recovery']}")


if __name__ == "__main__":
    main()

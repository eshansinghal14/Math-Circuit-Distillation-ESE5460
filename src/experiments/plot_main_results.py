"""Render the main comparison figure and table: graph distillation against every baseline.

Reads the arms of the main 22_add experiment under the shared protocol (bf16,
lr 1e-5, 10 warmup steps, cosine to 0.1, tau = 1, KD on every real position,
BOS-free training): standard KD, KD plus the TinyBERT recipe, KD plus linear
CKA (both at weight 0.3, their best in the sweep), supervised fine-tuning,
graph distillation with the ANOVA supernodes at lambda = 3, and the
scrambled-target control. Draws accuracy against training step: in-distribution,
mean out-of-distribution, and each held-out family on its own panel; seed means
with a +-1 sd band where an arm holds more than one seed.

Writes ``figures/main_results.pdf`` and ``tables/main_results.tex``. The table
reports each seed at its own best in-distribution step (see ``_paper_plots``).
An arm whose history is missing (the scrambled control until it runs) is a
\\pending{} column and is absent from the figure.

Encoding: graph distillation blue, the scrambled control blue dotted, SFT
orange; the three KD-based baselines share the grey of the reference method and
differ in line style (standard KD dashed, TinyBERT dotted, CKA dash-dot). The
dashed rule is the untrained student, the dotted rule the teacher.

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
    OOD_MEAN, panel_title,
    BERT_MAIN, BLUE, CKA_MAIN, CONTROL, DOT, FIG_DIR, GRAPH_MAIN, INK_2, INK_3, KD_MAIN, ORANGE, RC, SCRAMBLED_MAIN,
    SFT_MAIN, legend, line_with_band, metric_table, pm, reference_handles, rules, save, seed_lines, select_runs,
    steps_str, style,
)

DASHDOT = (0, (4, 1.5, 1, 1.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=f"{FIG_DIR}/main_results.pdf")
    args = ap.parse_args()

    graph = select_runs(*GRAPH_MAIN, label="graph distillation")
    kd = select_runs(*KD_MAIN, label="standard KD")
    bert = select_runs(*BERT_MAIN, label="KD + TinyBERT", required=False)
    cka = select_runs(*CKA_MAIN, label="KD + CKA", required=False)
    sft = select_runs(*SFT_MAIN, label="SFT", required=False)
    scr = select_runs(*SCRAMBLED_MAIN, label="scrambled target", required=False)
    last = int(graph.steps[-1])
    arms = [(kd, INK_2, CONTROL["linestyle"], "KD")]
    if bert is not None:
        arms.append((bert, INK_2, DOT, "TinyBERT"))
    if cka is not None:
        arms.append((cka, INK_2, DASHDOT, "CKA"))
    if sft is not None:
        arms.append((sft, ORANGE, "solid", "SFT"))
    arms.append((graph, BLUE, "solid", "graph"))
    if scr is not None:
        arms.append((scr, BLUE, DOT, "scrambled"))

    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.0))
    axes = [axes[0][0], axes[0][1], axes[0][2], axes[1][0], axes[1][1], axes[1][2]]
    order = ["accuracy", "ood"] + graph.ood
    titles = {}
    for ax, key in zip(axes, order):
        style(ax, panel_title(key))
        ax.set_xlim(-1, last + 2)
        ax.set_xticks(range(0, last + 1, 25))
        for run, colour, ls, short in arms:
            draw = seed_lines if run is scr else line_with_band
            draw(ax, run, key, colour, last_step=last, linestyle=ls,
                 linewidth=0.7 if run is scr else (1.4 if run in (graph, sft) else 1.0))
        rules(ax, graph.baseline_of(key), graph.teacher_of(key))
        ax.set_ylim(0, 1.02)
    for ax in axes[3:]:
        ax.set_xlabel("training step", color=INK_2)
    axes[0].set_ylabel("accuracy", color=INK_2)
    axes[3].set_ylabel("accuracy", color=INK_2)

    handles = [Line2D([], [], color=c, linestyle=ls, linewidth=1.4) for _, c, ls, _ in arms]
    labels = [f"{run.label} ({run.n} seed{'s' if run.n > 1 else ''}"
              + (", per seed)" if run is scr else ")") for run, _, _, _ in arms]
    rh, rl = reference_handles()
    legend(fig, handles + rh, labels + rl, ncol=3, y=-0.07)
    fig.tight_layout(rect=(0, 0.14, 1, 1), h_pad=1.2, w_pad=1.0)
    save(fig, args.out)

    # ---- table ---------------------------------------------------------------
    def seeds(run):
        return f"{run.n} seed{'s' if run.n > 1 else ''}"

    columns = [(f"standard KD\n({seeds(kd)})", kd)]
    if bert is not None:
        columns.append((f"KD + TinyBERT\n({seeds(bert)})", bert))
    if cka is not None:
        columns.append((f"KD + CKA\n({seeds(cka)})", cka))
    if sft is not None:
        columns.append((f"SFT\n({seeds(sft)})", sft))
    columns.append((f"graph distillation\n($\\lambda = {graph.config['lambda_graph']:g}$, {seeds(graph)})", graph))
    metric_table("main_results", columns, graph)
    for run, _, _, _ in arms:
        s = run.summary()
        print(f"{run.label}: seeds={run.seeds} sel steps {steps_str(s['sel_steps'])} last {steps_str(s['last_steps'])} "
              f"in-dist {pm(s['accuracy']['sel'])} meanOOD {pm(s['ood']['sel'])} "
              + " ".join(f"{o} {pm(s[o]['sel'])}" for o in run.ood)
              + f" | peak OOD {pm(s['ood']['peak'])}@{s['ood']['peak_step']} | last in-dist {pm(s['accuracy']['last'])}"
              + f" | last 21_mult {pm(s['21_mult']['last'])}")


if __name__ == "__main__":
    main()

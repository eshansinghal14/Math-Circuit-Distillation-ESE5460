"""Render the second-student comparison: graph distillation against standard KD on Llama-3.2-3B.

Reads the two 3B arms under the shared protocol (bf16, lr 1e-5, 10 warmup
steps, cosine to 0.1, tau = 1, KD on every real position, BOS-free training):
standard KD from ``results/standard_kd/3b_22_add-sweep.json`` and graph
distillation with the ANOVA supernodes at lambda = 3 from
``results/anova_study/3b_22_add_anova-final.json``, three seeds each. Nothing
was retuned for the larger student: lambda, the supernode categories and the
neurons per label are those of the 1B experiment, with the categories scored
on the 3B's own neurons.

Writes ``tables/second_student.tex`` (the main-table layout: each seed at its
own best in-distribution step, see ``_paper_plots``), ``tables/second_student_sweep.tex``
(the 3B standard-KD learning-rate sweep that confirms the shared protocol on
this student) and ``figures/second_student.pdf`` (accuracy against training
step: in-distribution, mean out-of-distribution, each held-out family).

Encoding as in the main figure: graph distillation blue, standard KD in the
grey of the reference method, dashed; seed means with a +-1 sd band; the dashed
rule is the untrained student, the dotted rule the teacher.

Usage (from the repository root):
    PYTHONPATH=src python -m experiments.plot_second_student
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from experiments._paper_plots import (
    OOD_MEAN, panel_title,
    BLUE, CONTROL, FAM_TEX, FIG_DIR, GRAPH_MAIN_3B, INK_2, INK_3, KD_MAIN_3B, KD_PROTOCOL, KD_SWEEP_3B, RC, all_runs,
    hdr, legend, line_with_band, metric_table, pm, reference_handles, rules, save, select_runs, steps_str,
    style, write_table,
)
from experiments.plot_kd_sweep import lr_tex


def sweep_table(ref) -> None:
    """``tables/second_student_sweep.tex``: the 3B standard-KD learning-rate sweep."""
    runs = all_runs(KD_SWEEP_3B)
    runs.sort(key=lambda r: r.config.get("learning_rate"))
    cols = ["lr", "seeds", FAM_TEX["22_add"] + " peak (step)"] + [FAM_TEX[o] for o in ref.ood] + [OOD_MEAN, FAM_TEX["22_add"] + " last"]
    lines = ["\\begin{tabular}{lc" + "c" * (len(cols) - 2) + "}", "  \\toprule",
             "  " + " & ".join(hdr(c) for c in cols) + " \\\\", "  \\midrule"]
    for r in runs:
        c = r.config
        s = r.summary()
        marked = all(c.get(k) == v for k, v in KD_PROTOCOL.items())
        row = [lr_tex(c["learning_rate"]), str(r.n), pm(s["accuracy"]["sel"]) + f" ({steps_str(s['sel_steps'])})"]
        row += [pm(s[o]["sel"], below=ref.baseline_of(o)) for o in ref.ood]
        row += [pm(s["ood"]["sel"], below=ref.baseline_of("ood")), pm(s["accuracy"]["last"])]
        line = "  " + " & ".join(row) + " \\\\"
        if marked:
            line = "  \\rowcolor{gray!12} " + line.strip()
        lines.append(line)
    lines += ["  \\bottomrule", "\\end{tabular}"]
    write_table("second_student_sweep", lines,
                "standard-KD learning-rate sweep on the 3B student; each seed at its own best in-distribution step; shaded row is the shared protocol")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=f"{FIG_DIR}/second_student.pdf")
    args = ap.parse_args()

    graph = select_runs(*GRAPH_MAIN_3B, label="graph distillation")
    kd = select_runs(*KD_MAIN_3B, label="standard KD")
    last = int(graph.steps[-1])
    arms = [(kd, INK_2, CONTROL["linestyle"], "KD"), (graph, BLUE, "solid", "graph")]

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
            line_with_band(ax, run, key, colour, last_step=last, linestyle=ls,
                           linewidth=1.4 if run is graph else 1.0)
        rules(ax, graph.baseline_of(key), graph.teacher_of(key))
        ax.set_ylim(0, 1.02)
    for ax in axes[3:]:
        ax.set_xlabel("training step", color=INK_2)
    axes[0].set_ylabel("accuracy", color=INK_2)
    axes[3].set_ylabel("accuracy", color=INK_2)

    handles = [Line2D([], [], color=c, linestyle=ls, linewidth=1.4) for _, c, ls, _ in arms]
    labels = [f"{run.label} ({run.n} seed{'s' if run.n > 1 else ''})" for run, _, _, _ in arms]
    rh, rl = reference_handles(student="untrained 3B student")
    legend(fig, handles + rh, labels + rl, ncol=2, y=-0.06)
    fig.tight_layout(rect=(0, 0.11, 1, 1), h_pad=1.2, w_pad=1.0)
    save(fig, args.out)

    # ---- tables --------------------------------------------------------------
    def seeds(run):
        return f"{run.n} seed{'s' if run.n > 1 else ''}"

    columns = [(f"standard KD\n({seeds(kd)})", kd),
               (f"graph distillation\n($\\lambda = {graph.config['lambda_graph']:g}$, {seeds(graph)})", graph)]
    metric_table("second_student", columns, graph,
                 header_comment="Llama-3.2-3B student; arms as columns, metrics as rows; each seed at its own best in-distribution step, seed mean +- sd")
    sweep_table(graph)
    for run, _, _, _ in arms:
        s = run.summary()
        print(f"{run.label}: seeds={run.seeds} sel steps {steps_str(s['sel_steps'])} last {steps_str(s['last_steps'])} "
              f"in-dist {pm(s['accuracy']['sel'])} meanOOD {pm(s['ood']['sel'])} "
              + " ".join(f"{o} {pm(s[o]['sel'])}" for o in run.ood)
              + f" | peak OOD {pm(s['ood']['peak'])}@{s['ood']['peak_step']} | last in-dist {pm(s['accuracy']['last'])}"
              + " | last " + " ".join(f"{o} {pm(s[o]['last'])}" for o in run.ood))
    print("untrained 3B:", {k: round(graph.baseline_of(k), 3) for k in ["accuracy", "ood"] + graph.ood})


if __name__ == "__main__":
    main()

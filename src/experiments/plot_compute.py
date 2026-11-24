"""Tabulate the compute cost of graph distillation against standard KD.

Reads the three ``--track-flops`` runs in ``results/flops`` (five steps each,
same batch of 32 prompts, same protocol): standard KD, graph distillation with
a cold teacher-target cache (the teacher's attribution graph built on the fly)
and graph distillation with a warm cache (every run in the paper). The counter
covers every matmul, attention and convolution kernel in a training step's
forward and backward passes, teacher forward and both attribution graphs
included.

Puts accuracy on the same axis as cost: each row carries the FLOPs the arm has
spent by the step it is read at, so the graph arm at its held-out peak (step 20)
sits next to standard KD with the cosine schedule stretched to the budget of
equal FLOPs (280 steps), and the graph arm at step 100 next to KD at 1400.
Accuracy follows the paper's convention -- each seed at its own best
in-distribution step within the budget.

Writes ``tables/compute.tex`` and prints the equal-compute arithmetic the text
quotes: the KD-step budget matching each graph step count, the one-time cost of
filling the teacher-target cache, and the per-step wall clock of the teacher's
graph with the cache cold and warm, which the appendix quotes.

Usage (from the repository root):
    PYTHONPATH=src python -m experiments.plot_compute
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Tuple

import numpy as np

from experiments._paper_plots import (
    OOD_MEAN,
    FAM_TEX, GRAPH_MAIN, KD_MAIN, KD_STRETCHED, R, Run, hdr, load_run, pm, select_runs,
    steps_str, write_table,
)

FLOPS = {
    "standard KD": f"{R}/flops/22_add_skd.json",
    "graph distillation, warm teacher cache": f"{R}/flops/22_add_gkd_warm.json",
    "graph distillation, cold teacher cache": f"{R}/flops/22_add_gkd.json",
}

# The graph arm is one 100-step run read at two steps: its held-out peak and the
# end. Each is matched against the KD budget of equal FLOPs (280 and 1400).
GRAPH_READ = (20, 100)


def tflops(x: float) -> str:
    return f"{x / 1e12:.1f}"


def pflops(x: float) -> str:
    return f"{x / 1e15:.2f}" if x < 1e16 else f"{x / 1e15:.0f}"


def ratio(x: float) -> str:
    return f"{x:.1f}" if x < 100 else f"{x:.0f}"


def cells(run: Optional[Run], until: Optional[int]) -> List[str]:
    """The three accuracy columns, or dashes when there is no run.

    Seed count is not a column of its own: the step list in the in-distribution
    cell carries one step per seed, and the sd says the same thing.
    """
    if run is None:
        return ["---"] * 3
    s = run.summary(until=until)
    return [pm(s["accuracy"]["sel"]) + f" ({steps_str(s['sel_steps'])})",
            pm(s["21_mult"]["sel"]), pm(s["ood"]["sel"])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args()

    runs = {name: load_run(path, name) for name, path in FLOPS.items()}
    per_step = {name: float(np.mean(r.hists[0]["step_flops"])) for name, r in runs.items()}
    kd = per_step["standard KD"]
    warm = per_step["graph distillation, warm teacher cache"]
    cold = per_step["graph distillation, cold teacher cache"]

    graph = select_runs(*GRAPH_MAIN, label="graph")
    kd_main = select_runs(*KD_MAIN, label="KD")
    batch = graph.config["batch_size"]

    # (arm, steps, FLOPs per step, run, step to read it at). Sorted by the FLOPs
    # spent, so each graph row lands beside the KD run of matching cost.
    entries: List[Tuple[str, int, float, Optional[Run], Optional[int]]] = [
        ("standard KD", kd_main.config["steps"], kd, kd_main, None),
    ]
    for step in GRAPH_READ:
        entries.append(("graph distillation, warm cache", step, warm, graph, step))
    for steps, spec in KD_STRETCHED.items():
        r = select_runs(*spec, label=f"KD {steps}", required=False)
        if r is not None:
            entries.append(("standard KD, stretched schedule", steps, kd, r, None))
    entries.append(("graph distillation, cold cache", GRAPH_READ[-1], cold, None, None))
    entries.sort(key=lambda e: e[1] * e[2])

    cols = ["arm", "steps", "TFLOPs\n/ step", "$\\times$ KD", "PFLOPs\nspent",
            FAM_TEX["22_add"] + " (step)", FAM_TEX["21_mult"], OOD_MEAN]
    rows = ["\\begin{tabular}{lccccccc}", "  \\toprule",
            "  " + " & ".join(hdr(c) for c in cols) + " \\\\", "  \\midrule"]
    prev: Optional[float] = None
    for arm, steps, per, run, until in entries:
        spent = per * steps
        # A blank line between compute-matched pairs: a row costing over twice
        # the one above it starts a new group.
        if prev is not None and spent > 2 * prev:
            rows.append("  \\addlinespace")
        prev = spent
        row = [arm, str(steps), tflops(per), ratio(per / kd), pflops(spent)] + cells(run, until)
        rows.append("  " + " & ".join(row) + " \\\\")
    rows += ["  \\bottomrule", "\\end{tabular}"]
    write_table("compute", rows,
                f"FLOPs per training step of {batch} prompts from --track-flops (5-step runs); "
                "accuracy at each seed's best in-distribution step within the budget")

    print(f"per step: KD {kd:.3e}  graph warm {warm:.3e} ({warm/kd:.1f}x)  graph cold {cold:.3e} ({cold/kd:.0f}x)")
    print(f"teacher-target build: {(cold - warm):.3e} FLOPs per step of {batch} prompts = {(cold - warm)/batch:.3e} per prompt; "
          f"one pass over a 100-step run's prompts = {(cold - warm) * 100:.3e}")
    for step in GRAPH_READ:
        print(f"graph step {step}: cumulative {warm * step:.3e} = {warm * step / kd:.0f} KD steps")
    print(f"graph held-out peak at step {graph.summary()['ood']['peak_step']}, "
          f"in-distribution selection at {steps_str(graph.summary()['sel_steps'])}")
    # The cache's wall-clock argument, from the two flops runs so the two numbers
    # are measured the same way (--track-flops itself costs 2-3x, so neither is
    # the wall clock of a paper run).
    t_cold, t_warm = (float(np.mean(runs[n].hists[0]["step_graph_time_teacher"]))
                      for n in ("graph distillation, cold teacher cache",
                                "graph distillation, warm teacher cache"))
    print(f"teacher graph per step: {t_cold:.0f}s cold vs {t_warm:.2f}s warm")


if __name__ == "__main__":
    main()

# experiments/

One-off diagnostics. Nothing here is imported by the training or graph-building
code — these answer a specific question once, then sit here so the next person
asking it doesn't rebuild the measurement.

Run them as modules with `src` on the path, same as the rest of the tree:

```
python -m experiments.<name> --help
```

| script | question it answers |
| --- | --- |
| `diagnose_grad.py` | Does the edge freeze change the graph gradient's **direction** or only its **magnitude**? Computes the KL, unfrozen-graph and frozen-graph gradients at one checkpoint and reports norms and pairwise cosines, globally and per parameter group. `cos(graph, KL)` also says how much of the graph term KD already implies. |
| `diagnose_freeze.py` | Is the frozen supergraph a **weaker regression target**? Builds the trainer's supergraph once per prompt per `--modes` entry plus an unfrozen reference, and reports row entropy against uniform, attribution mass on token/BOS nodes, `frac_external` spread, distinct-vs-total supernode labels, and the JSD between each mode's target and the unfrozen one. `--modes` takes `attn-only`, `rms-only` and `frozen`, so one run separates the two freezes. |

Both take `--graph-node-labels`, and it **must match the run being diagnosed**:
omitted means arg-token + DLA supernodes (what `graph_kd` uses when its own
`--graph-node-labels` is empty), supplying labels switches to ANOVA supernodes.
Nothing warns you if they disagree — the numbers just describe a different
objective than the one you trained on.

Paper figures and tables (`PYTHONPATH=src python -m experiments.<name>` from the
repo root; each writes `latex/graph_distillation/figures/<name>.pdf` and, where it
has a table, `latex/graph_distillation/tables/<name>.tex` as a complete tabular
that the paper pulls in with `\tablerows{}`; never edit those by hand):

| script | figure / table |
| --- | --- |
| `plot_main_results.py` | graph distillation vs SFT vs standard KD, all families; `tables/main_results.tex` |
| `plot_lambda_sweep.py` | Appendix D: the λ sweep with both references; `tables/lambda.tex` |
| `plot_freeze_ablation.py` | Appendix C: the linearisation arms; `tables/freeze_train.tex`. A missing arm becomes a `\pending{}` column. |
| `plot_grad_metrics.py` | Appendix C: per-step gradient metrics; `tables/freeze_dynamics.tex` |
| `plot_scramble.py` | Appendix D: real against scrambled routing target, accuracy and losses |

`_paper_plots.py` holds the shared loader (every seed in a history, mean ± sd),
the summary statistics (values at a step, per-seed peaks, dip / recovery) and the
palette.

Each script's module docstring explains how to read its output.

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
| `plot_main_results.py` | Table 1 / Figure 3: standard KD, TinyBERT, CKA, SFT (3 seeds each), graph distillation with ANOVA supernodes at lambda 3, and the scrambled target (pending); `tables/main_results.tex` |
| `plot_scramble.py` | Figure 4: real against scrambled routing target, accuracy and losses |
| `plot_ablations.py` | Appendix B: lambda sweep, supernode-size x pre-selection grid, BOS-on ablation; `tables/{lambda_sweep,supernode_grid,bos_on}.tex`, `figures/ablations.pdf` |
| `plot_kd_sweep.py` | Appendix B: the standard-KD protocol sweep and the TinyBERT/CKA weight sweeps; `tables/kd_sweep.tex`, `tables/rep_sweep.tex` |
| `plot_compute.py` | Section 4.2 (Table 2) and Appendix D: FLOPs per step for KD vs graph distillation (warm and cold teacher cache), and each arm's accuracy at a matched FLOP budget -- the graph arm at steps 20 and 100 against KD with its schedule stretched to 280 and 1400; `tables/compute.tex` |
| `plot_grad_metrics.py` | Appendix C: per-step gradient metrics, run means per arm and step-1 canaries; `tables/grad_dynamics.tex`, `tables/canaries.tex` |
| `plot_second_student.py` | Section 4.3 and Appendix B: graph distillation vs standard KD on the Llama-3.2-3B student (3 seeds each, nothing retuned) and the 3B KD learning-rate sweep; `tables/second_student.tex`, `tables/second_student_sweep.tex`, `figures/second_student.pdf` |

`_paper_plots.py` holds the shared loader and the run specs. Histories hold every
run under `runs`, keyed `#<digest>|seed=<n>`; `select_runs(path, where, label)`
picks the runs whose recorded config matches `where` (identity is the config
content, so a run stored under two key spellings counts once) and NaN-pads a
seed whose evaluation stopped early. Never read a file's top level, which is
whichever run wrote last. Tables report each seed at its own best in-distribution
step (`summary()["sel"]`), because every arm peaks and then eases back at a
different step per seed. A spec whose history is missing (`required=False`)
becomes a `\pending{}` column, so the paper builds before every run has landed.

| `inspect_graphs.py` | Where does the student's supergraph differ from the teacher's when it is right vs wrong, and which construction of the graph target is worth training on? Scores a split under teacher forcing, buckets prompts (right-confident, right-unsure, wrong-with-carry, wrong-without), builds both models' graphs for a few per bucket and derives five constructions from the same attributions (`normalised`, `raw-signed`, `gold-signed` and `gold-path` — edges re-oriented by each target neuron's effect on the gold logit so member sums add coherently — and `composition`, the per-block split of inbound mass over arg blocks / sum blocks / tokens), each scored with rel-mse, cosine, row JSD and mass-weighted row JSD. `summary.json` carries per-bucket means and `controls`: same-prompt vs shuffled-prompt distance, teacher-vs-teacher and student-vs-student across prompts, right-vs-wrong gap, a membership noise floor (`--noise-drop-members`) and an optional pool noise floor (`--noise-pool-prop`). `--kd-checkpoint` graphs a standard-KD student as a third model (its distance to the teacher and to the untrained student, `kd_minus_student` in the controls). `--student-checkpoint` runs it on a trained student. `--dataset` takes any N-ary addition set (`222_add` works with `sum range`, `sum units`, `tokens`, `argN ...`). GPU. |

Each script's module docstring explains how to read its output.

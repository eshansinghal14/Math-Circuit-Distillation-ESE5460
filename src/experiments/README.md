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
| `diagnose_freeze.py` | Is the frozen supergraph a **weaker regression target**? Builds the trainer's supergraph twice per prompt, frozen and unfrozen, and reports row entropy against uniform, attribution mass on token/BOS nodes, `frac_external` spread, distinct-vs-total supernode labels, and the cross-mode JSD. |

Each script's module docstring explains how to read its output.

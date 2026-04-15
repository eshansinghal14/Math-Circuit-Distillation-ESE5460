"""Ablate random flattened-MLP neuron subsets at many fractions; measure accuracy drop.

Run from ``src/``::

  python -m experiments.random_ablation_vs_fraction.run \\
    --dataset ../datasets/2d_add_test_20.json --n-points 25 --batch-size 32

  # Sweep fractions 0 … 0.3 only:
  python -m experiments.random_ablation_vs_fraction.run \\
    --dataset ../datasets/2d_add_test_20.json --n-points 20 --batch-size 32 --max-frac 0.3

Outputs go under ``experiments/random_ablation_vs_fraction/results/``.
If ``random_ablation_vs_fraction.json`` already exists there, new points are **appended**
and the scatter plot uses **all** stored points.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from neuron_distillation.ablation import (  # noqa: E402
    apply_activation_ablation_hooks,
    remove_ablation_hooks,
)
from plotting import _NEURIPS_RC  # noqa: E402
from utils import (  # noqa: E402
    EVAL_MAX_NEW_TOKENS,
    eval_model,
    load_model,
    mlp_flatten_dim_from_model,
    test_model,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Random ablation vs neuron fraction")
    p.add_argument("--dataset", type=str, required=True, help="Math JSON (dict or q_str/a_str list)")
    p.add_argument(
        "--n-points",
        type=int,
        required=True,
        metavar="N",
        help="Number of (fraction, accuracy) points along the sweep",
    )
    p.add_argument(
        "--max-frac",
        type=float,
        default=1.0,
        metavar="F",
        help="Largest fraction of neurons ablated in the sweep (default: 1). Sweep is linspace 0..F.",
    )
    p.add_argument("--batch-size", type=int, required=True, metavar="N", help="Eval batch size")
    p.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="HF model id (default: 1B Llama)",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.n_points < 2:
        raise SystemExit("--n-points must be >= 2")
    if not (0.0 < args.max_frac <= 1.0):
        raise SystemExit("--max-frac must be in (0, 1]")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    buf_path = os.path.join(out_dir, "_eval_buffer.json")
    dataset_path = os.path.abspath(args.dataset)

    model, tokenizer = load_model(args.model_name)
    total = mlp_flatten_dim_from_model(model)
    fracs = np.linspace(0.0, float(args.max_frac), args.n_points)

    test_model(
        model, tokenizer, dataset_path, buf_path,
        batch_size=args.batch_size, max_new_tokens=EVAL_MAX_NEW_TOKENS, log=False,
    )
    baseline = eval_model(buf_path, log=False)
    print(
        f"  baseline accuracy: {baseline:.4f} ({total} flat MLP neurons); "
        f"sweep 0 … {float(args.max_frac):.4f} ({args.n_points} points)",
    )

    points: list[dict] = []
    n_frac = len(fracs)
    for i, frac in enumerate(fracs):
        k = int(round(float(frac) * total))
        k = max(0, min(k, total))
        if k == 0:
            acc = baseline
        else:
            g = torch.Generator()
            g.manual_seed(args.seed + i * 10_007)
            idx = torch.randperm(total, generator=g)[:k].long()
            handles = apply_activation_ablation_hooks(model, idx)
            try:
                test_model(
                    model, tokenizer, dataset_path, buf_path,
                    batch_size=args.batch_size, max_new_tokens=EVAL_MAX_NEW_TOKENS, log=False,
                )
                acc = eval_model(buf_path, log=False)
            finally:
                remove_ablation_hooks(handles)
        drop = baseline - acc
        points.append(
            {"fraction_ablated": float(frac), "accuracy": acc, "performance_drop": drop},
        )
        print(
            f"  [{i + 1}/{n_frac}] frac={float(frac):.4f}  "
            f"k={k}/{total}  acc={acc:.4f}  drop={drop:.4f}",
        )

    json_path = os.path.join(out_dir, "random_ablation_vs_fraction.json")
    prior: list[dict] = []
    if os.path.isfile(json_path):
        try:
            with open(json_path, encoding="utf-8") as f:
                old = json.load(f)
            if isinstance(old, dict) and isinstance(old.get("points"), list):
                prior = old["points"]
        except (json.JSONDecodeError, OSError):
            prior = []

    all_points = prior + points
    payload = {
        "model_name": args.model_name,
        "dataset": dataset_path,
        "baseline_accuracy": baseline,
        "total_flat_mlp_neurons": total,
        "max_frac": float(args.max_frac),
        "seed": args.seed,
        "points": all_points,
        "n_points_this_run": len(points),
        "n_points_total": len(all_points),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    xs = [pt["fraction_ablated"] for pt in all_points]
    ys = [pt["performance_drop"] for pt in all_points]
    png_path = os.path.join(out_dir, "random_ablation_vs_fraction.png")
    with plt.rc_context(_NEURIPS_RC):
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.scatter(xs, ys, s=18, alpha=0.85, c="tab:blue", edgecolors="none")
        ax.set_xlabel("Fraction of neurons ablated")
        ax.set_ylabel("Performance drop (baseline − accuracy)")
        ax.set_title("Random ablation vs. fraction")
        fig.tight_layout()
        fig.savefig(png_path)
        plt.close(fig)

    print(f"Wrote {json_path} ({len(points)} new → {len(all_points)} total points)")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()

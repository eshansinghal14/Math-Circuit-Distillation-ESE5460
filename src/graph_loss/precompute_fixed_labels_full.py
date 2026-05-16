"""Pre-compute a frozen neuron→label mapping by running ANOVA on **every** neuron.

Unlike :mod:`precompute_fixed_labels_fast` (which first runs forward passes on
all dataset prompts to filter neurons by ``source_vector_norm`` and then runs
ANOVA on the union), this script skips the filtering phase entirely.  It uses
the pre-computed MLP input cache to compute activations for *every*
``(layer, token_pos, neuron_id)`` combination directly via::

    activation_neuron_i(prompt_j) = silu(x_j @ W_gate_i^T) * (x_j @ W_up_i^T)

where ``x_j`` is the cached residual-stream input to the MLP at the layer of
interest for prompt ``j``.

Trade-offs vs. the fast script:

* No forward-pass filtering pass — much simpler.
* ANOVA grid covers all neurons (16 × 5 × 8192 = 655 k rows for 1B), so this
  takes longer (~20-30 min on an A100) but explores the full neuron space.
* Same final output format ``{layer:neuron_id → label}`` — drop-in compatible.

Usage (run once before training)::

    python -m graph_loss.precompute_fixed_labels_full \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --dataset 22_add_tight \\
        --mlp-input-cache /content/mlp_input_cache \\
        --output /content/fixed_labels_full_1b.json \\
        --anova-nodes-per-label 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _ensure_src_on_path() -> None:
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _build_full_neuron_locations(
    n_layers: int, n_positions: int, d_mlp: int
) -> torch.Tensor:
    """Return ``[n_layers * n_positions * d_mlp, 3]`` covering every triple."""
    layers = torch.arange(n_layers, dtype=torch.long)
    positions = torch.arange(n_positions, dtype=torch.long)
    neurons = torch.arange(d_mlp, dtype=torch.long)
    grid = torch.cartesian_prod(layers, positions, neurons)
    return grid.contiguous()


def main() -> None:
    _ensure_src_on_path()

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model ID for the student model",
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset name (e.g. '22_add_tight') or explicit path to a JSON file. "
             "_resolve_dataset_path will search for *_all.json, *_train.json, etc.",
    )
    parser.add_argument(
        "--mlp-input-cache", required=True,
        help="Path to the pre-computed MLP input cache root directory "
             "(created by precompute_mlp_inputs.py). Required — fails fast if missing.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSON path for the fixed label mapping",
    )
    parser.add_argument(
        "--anova-nodes-per-label", type=int, default=3,
        help="Maximum number of neurons to assign per ANOVA label category (default: 3)",
    )
    parser.add_argument(
        "--anova-range-radius", type=int, default=0,
        help="Radius around target arg for range-label rules (0 = exact match)",
    )
    parser.add_argument(
        "--sum-min-specificity", type=float, default=0.0,
        help="Minimum ANOVA specificity score a neuron must have to receive any label",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"],
        help="Model dtype (default: bfloat16)",
    )
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    # ── Imports (deferred so sys.path is already set) ─────────────────────────
    from utils.hf_models import load_student_model_for_distillation
    from graph_loss.hf_adapter import HFLlamaGraphAdapter
    from graph_loss.neuron_activation_heatmap import (
        _resolve_dataset_path,
        build_neuron_activation_write_result,
    )
    from graph_loss.precompute_mlp_inputs import (
        load_mlp_input_cache,
        mlp_input_cache_dir,
        mlp_input_cache_exists,
    )
    from graph_loss.anova_node_labels import ANOVA_LABEL_CATEGORIES, label_activation_heatmaps

    # ── Load student model ────────────────────────────────────────────────────
    logger.info("Loading student model: %s  (device=%s  dtype=%s)", args.model, device, dtype)
    student_model, tokenizer = load_student_model_for_distillation(
        student_source=None,
        student_model_id=args.model,
        device=device,
    )
    student_model = student_model.to(dtype=dtype)
    adapter = HFLlamaGraphAdapter(student_model, tokenizer, device)
    model_name = getattr(adapter.model.config, "_name_or_path", args.model)
    logger.info(
        "Model loaded: %d layers, d_mlp=%d, d_model=%d",
        adapter.n_layers,
        adapter.d_mlp,
        adapter.d_model,
    )

    # ── Resolve dataset ───────────────────────────────────────────────────────
    dataset_path = _resolve_dataset_path(args.dataset)
    logger.info("Dataset resolved to: %s", dataset_path)

    # ── Load MLP input cache (REQUIRED) ───────────────────────────────────────
    if not mlp_input_cache_exists(args.mlp_input_cache, model_name, dataset_path):
        expected_dir = mlp_input_cache_dir(args.mlp_input_cache, model_name, dataset_path)
        logger.error(
            "MLP input cache not found at %r for model=%r dataset=%r. "
            "Expected directory: %r. Run precompute_mlp_inputs.py first.",
            args.mlp_input_cache,
            model_name,
            dataset_path,
            expected_dir,
        )
        sys.exit(1)

    mlp_input_cache = load_mlp_input_cache(args.mlp_input_cache, model_name, dataset_path)
    cache_meta = mlp_input_cache["meta"]
    n_cache_prompts = int(cache_meta["n_prompts"])
    n_cache_positions = int(cache_meta["n_positions"])
    logger.info(
        "Loaded MLP input cache: %d prompts, %d positions",
        n_cache_prompts,
        n_cache_positions,
    )

    # ── Build full neuron_locations tensor ────────────────────────────────────
    neuron_locations = _build_full_neuron_locations(
        n_layers=adapter.n_layers,
        n_positions=n_cache_positions,
        d_mlp=adapter.d_mlp,
    )
    n_total_neurons = int(neuron_locations.shape[0])
    logger.info(
        "Built full neuron_locations tensor: shape=%s "
        "(layers=%d × positions=%d × d_mlp=%d = %d rows)",
        tuple(neuron_locations.shape),
        adapter.n_layers,
        n_cache_positions,
        adapter.d_mlp,
        n_total_neurons,
    )

    # ── Build activation heatmaps (ANOVA grid) ────────────────────────────────
    logger.info(
        "Starting ANOVA on %d total neurons — this is the slow step, "
        "expect ~20-30 min on an A100.",
        n_total_neurons,
    )
    result = build_neuron_activation_write_result(
        adapter,
        dataset_path,
        neuron_locations,
        mlp_input_cache=mlp_input_cache,
        include_w_down_vectors=False,
    )
    logger.info(
        "Activation result shape: %s  arg_values dims: %s",
        tuple(result.activations.shape),
        [len(v) for v in result.arg_values],
    )

    # ── ANOVA labeling ────────────────────────────────────────────────────────
    logger.info(
        "Running ANOVA labeling (target_args=None — global across all arg values, "
        "anova_range_radius=%d) ...",
        args.anova_range_radius,
    )
    label_results = label_activation_heatmaps(
        result.activations,
        result.arg_values,
        target_args=None,
        anova_range_radius=args.anova_range_radius,
    )

    # ── Category loop: pick top anova_nodes_per_label per category ────────────
    # Same ranking logic as precompute_fixed_labels_fast.py: rank by
    # category_specificity, keep top-K per category, never overwrite an
    # existing assignment for a (layer, neuron_id) key.
    fixed_labels: dict[str, str] = {}
    category_summary: list[tuple[str, int, int, float]] = []

    for category in ANOVA_LABEL_CATEGORIES:
        all_scored: list[tuple[int, float]] = [
            (row_idx, float(label_result.category_specificity[category]))
            for row_idx, label_result in enumerate(label_results)
            if category in label_result.category_specificity
            and label_result.category_scores.get(category, 0.0) > 0.0
            and label_result.category_specificity.get(category, float("-inf"))
            > args.sum_min_specificity
        ]
        sorted_rows = sorted(all_scored, key=lambda x: x[1], reverse=True)
        top_rows = sorted_rows[: args.anova_nodes_per_label]

        if not top_rows:
            logger.info("  ANOVA category %-35s no qualifying neurons", f"{category!r}:")
            category_summary.append((category, 0, 0, float("nan")))
            continue

        newly_labeled = 0
        for row_idx, _score in top_rows:
            layer = int(neuron_locations[row_idx, 0].item())
            neuron_id = int(neuron_locations[row_idx, 2].item())
            key = f"{layer}:{neuron_id}"
            if key not in fixed_labels:
                fixed_labels[key] = label_results[row_idx].categories[category]
                newly_labeled += 1

        best_score = float(top_rows[0][1])
        logger.info(
            "  ANOVA category %-35s top=%d  new_labels=%d  best_specificity=%.4f",
            f"{category!r}:",
            len(top_rows),
            newly_labeled,
            best_score,
        )
        category_summary.append((category, len(top_rows), newly_labeled, best_score))

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(
        "Fixed label map: %d neurons labeled  (from full neuron set of %d triples)",
        len(fixed_labels),
        n_total_neurons,
    )
    label_counts: Counter = Counter(fixed_labels.values())
    for lbl, cnt in sorted(label_counts.items()):
        logger.info("  %-40s : %d neurons", lbl, cnt)

    logger.info("Per-category summary (category | top | newly_labeled | best_specificity):")
    for category, n_top, n_new, best in category_summary:
        if n_top == 0:
            logger.info("  %-35s : skipped (no qualifying neurons)", category)
        else:
            logger.info(
                "  %-35s : top=%d  new=%d  best_specificity=%.4f",
                category,
                n_top,
                n_new,
                best,
            )

    # ── Save ──────────────────────────────────────────────────────────────────
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(fixed_labels, f, indent=2)
    logger.info("Saved fixed labels to %s", args.output)


if __name__ == "__main__":
    main()

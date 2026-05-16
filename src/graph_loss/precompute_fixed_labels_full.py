"""Pre-compute a frozen neuron→label mapping by running ANOVA on **every** neuron.

Unlike :mod:`precompute_fixed_labels_fast` (which first runs forward passes on
all dataset prompts to filter neurons by ``source_vector_norm`` and then runs
ANOVA on the filtered union), this script skips the filtering phase entirely.

It uses the pre-computed MLP input cache to compute activations for *every*
``(layer, neuron_id)`` combination via a vectorized GPU matrix multiply::

    activation_neuron_i(prompt_j, pos_p) = silu(x_jp @ W_gate_i^T) * (x_jp @ W_up_i^T)

where ``x_jp`` is the cached residual-stream input to the MLP at layer ``l``,
token position ``p``, for prompt ``j``.

Processing is done one layer at a time.  For each layer the activation tensor
is ``[n_valid_prompts, d_mlp]`` (< 100 MB on GPU for a 1B model), so there is
no CPU OOM risk.  The ANOVA grid per layer is ``[d_mlp, n_arg1, n_arg2]``
(< 350 MB for d_mlp=8192, 100×100 arg grid).

Across token positions, labels are aggregated by taking the best specificity
score so that the final neuron label reflects the position where its
discriminative signal is strongest.

Same output format as :mod:`precompute_fixed_labels_fast`:
``{"{layer}:{neuron_id}": "<label>"}`` — drop-in compatible.

Usage (run once before training)::

    python -m graph_loss.precompute_fixed_labels_full \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --dataset 22_add_tight_all \\
        --mlp-input-cache /content/local_caches/mlp-input-cache \\
        --output /content/fixed_labels_full.json \\
        --anova-nodes-per-label 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from typing import Dict, List

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _ensure_src_on_path() -> None:
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _run_gpu_anova_all_neurons(
    adapter,
    mlp_input_cache: dict,
    arg_values: List[List[int]],
    device: torch.device,
    anova_range_radius: int,
    anova_nodes_per_label: int,
    sum_min_specificity: float,
) -> Dict[str, str]:
    """Layer-by-layer GPU-vectorized ANOVA over all neurons.

    For each layer ``l`` and token position ``p``:
      1. Load cached MLP inputs ``x`` of shape ``[n_valid, d_model]`` to GPU.
      2. Compute ``acts = silu(x @ W_gate^T) * (x @ W_up^T)`` → ``[n_valid, d_mlp]``
         in a single batched matrix multiply (fast on GPU).
      3. Scatter-add into ANOVA grid ``[d_mlp, n_arg1, n_arg2]`` using the
         pre-computed flat argument-grid indices.
      4. Divide by counts to get mean activation per (neuron, arg-combo) cell.
      5. Run ``label_activation_heatmaps`` on the grid (CPU, iterates over
         d_mlp neurons — typically 8192 — with a ~40-rule ANOVA per neuron).
      6. Track the highest specificity score per (neuron_id, category) across
         all token positions.

    After all positions are done for a layer, assign labels to the top-K neurons
    per category and free GPU memory before moving to the next layer.

    Peak GPU memory per layer: ≈ 2 × d_mlp × n_valid × 4 bytes
                                 + d_mlp × n_flat × 4 bytes
    For d_mlp=8192, n_valid=10k, n_flat=10k:  ≈ 970 MB  (fits on any GPU ≥ 4 GB)
    """
    from graph_loss.anova_node_labels import ANOVA_LABEL_CATEGORIES, label_activation_heatmaps

    cache_meta = mlp_input_cache["meta"]
    cache_layer_inputs: List[torch.Tensor] = mlp_input_cache["layer_inputs"]
    cache_args: List[List[int]] = cache_meta["numeric_args_by_prompt"]
    n_cache_prompts = int(cache_meta["n_prompts"])
    n_cache_positions = int(cache_meta["n_positions"])

    grid_shape = tuple(len(v) for v in arg_values)
    n_grid_dims = len(arg_values)
    arg_to_idx = [{v: i for i, v in enumerate(vals)} for vals in arg_values]

    n_flat = 1
    for s in grid_shape:
        n_flat *= s

    # Compute stride for row-major flat index
    strides = [1] * n_grid_dims
    for dim in range(n_grid_dims - 2, -1, -1):
        strides[dim] = strides[dim + 1] * grid_shape[dim + 1]

    # Build valid prompt mask and flat grid index (done once, reused every layer)
    valid_mask = torch.zeros(n_cache_prompts, dtype=torch.bool)
    flat_indices = torch.zeros(n_cache_prompts, dtype=torch.long)
    for j, args in enumerate(cache_args):
        if len(args) != n_grid_dims:
            continue
        ok = True
        flat = 0
        for dim, val in enumerate(args):
            if val not in arg_to_idx[dim]:
                ok = False
                break
            flat += arg_to_idx[dim][val] * strides[dim]
        if ok:
            valid_mask[j] = True
            flat_indices[j] = flat

    valid_idx = valid_mask.nonzero(as_tuple=True)[0]  # [n_valid]
    n_valid = int(valid_idx.shape[0])
    logger.info("Valid prompts for ANOVA grid: %d / %d", n_valid, n_cache_prompts)
    if n_valid == 0:
        raise ValueError(
            "No valid prompts found — check that the MLP cache was built with the "
            "same dataset as --dataset."
        )

    flat_idx_dev = flat_indices[valid_idx].to(device)  # [n_valid] on GPU

    # Counts per grid cell — same for all neurons, layers, positions
    counts_flat = torch.zeros(n_flat, dtype=torch.float32, device=device)
    counts_flat.scatter_add_(0, flat_idx_dev, torch.ones(n_valid, dtype=torch.float32, device=device))
    counts_grid = counts_flat.reshape(grid_shape).clamp(min=1.0)  # [*grid_shape]

    n_layers = adapter.n_layers
    d_mlp = adapter.d_mlp

    fixed_labels: Dict[str, str] = {}

    for layer_idx in range(n_layers):
        logger.info("━━ Layer %d / %d ━━", layer_idx + 1, n_layers)

        hf_layer = adapter.layers[layer_idx]
        W_gate = hf_layer.mlp.gate_proj.weight.to(device=device, dtype=torch.float32)  # [d_mlp, d_model]
        W_up   = hf_layer.mlp.up_proj.weight.to(device=device, dtype=torch.float32)   # [d_mlp, d_model]

        # layer_inputs: list entry is [n_prompts, n_positions, d_model]
        # Slice valid prompts on CPU to keep transfer cost low, then move to GPU per-position.
        layer_inputs_cpu = cache_layer_inputs[layer_idx][valid_idx]  # [n_valid, n_pos, d_model]

        # Accumulate activations across all (non-BOS) positions into one grid,
        # then call label_activation_heatmaps ONCE per layer instead of once per
        # position.  This is ~n_positions× faster with equivalent label quality.
        layer_grid_flat = torch.zeros(d_mlp, n_flat, dtype=torch.float32, device=device)
        n_active_positions = 0

        for pos_idx in range(n_cache_positions):
            if pos_idx == 0:
                continue  # BOS / padding — skip
            x = layer_inputs_cpu[:, pos_idx, :].to(device=device, dtype=torch.float32)
            acts = F.silu(x @ W_gate.T) * (x @ W_up.T)  # [n_valid, d_mlp]
            layer_grid_flat.scatter_add_(
                1,
                flat_idx_dev.unsqueeze(0).expand(d_mlp, -1),
                acts.T.contiguous().float(),
            )
            n_active_positions += 1
            del x, acts

        if n_active_positions == 0:
            n_active_positions = 1

        # Mean over positions, then mean over prompt counts per grid cell
        grid = (layer_grid_flat / n_active_positions).reshape(d_mlp, *grid_shape) / counts_grid
        del layer_grid_flat

        # Single ANOVA call per layer
        label_results = label_activation_heatmaps(
            grid.cpu(),
            arg_values,
            target_args=None,
            anova_range_radius=anova_range_radius,
        )
        del grid
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Assign labels: top-K per category
        n_labeled_this_layer = 0
        for cat in ANOVA_LABEL_CATEGORIES:
            scored = [
                (neuron_id, float(lr.category_specificity[cat]))
                for neuron_id, lr in enumerate(label_results)
                if cat in lr.category_specificity
                and lr.category_scores.get(cat, 0.0) > 0.0
                and lr.category_specificity.get(cat, float("-inf")) > sum_min_specificity
            ]
            scored.sort(key=lambda kv: kv[1], reverse=True)
            for neuron_id, _spec in scored[:anova_nodes_per_label]:
                key = f"{layer_idx}:{neuron_id}"
                if key not in fixed_labels:
                    fixed_labels[key] = label_results[neuron_id].categories[cat]
                    n_labeled_this_layer += 1

        del W_gate, W_up, layer_inputs_cpu, label_results
        if device.type == "cuda":
            torch.cuda.empty_cache()

        logger.info(
            "  Layer %d: %d new labels this layer, %d total so far",
            layer_idx + 1,
            n_labeled_this_layer,
            len(fixed_labels),
        )

    return fixed_labels


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
        help="Dataset name (e.g. '22_add_tight_all') or path to a JSON file",
    )
    parser.add_argument(
        "--mlp-input-cache", required=True,
        help="Path to the pre-computed MLP input cache root directory "
             "(created by precompute_mlp_inputs.py). Fails fast if missing.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSON path for the fixed label mapping",
    )
    parser.add_argument(
        "--anova-nodes-per-label", type=int, default=3,
        help="Maximum neurons to assign per ANOVA label category (default: 3)",
    )
    parser.add_argument(
        "--anova-range-radius", type=int, default=0,
        help="Radius around target arg for range-label rules (0 = exact match)",
    )
    parser.add_argument(
        "--sum-min-specificity", type=float, default=0.0,
        help="Minimum ANOVA specificity a neuron must have to receive any label",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"],
        help="Model dtype for loading (activations are computed in float32)",
    )
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    # ── Deferred imports (sys.path already adjusted above) ────────────────────
    from utils.hf_models import load_student_model_for_distillation
    from graph_loss.hf_adapter import HFLlamaGraphAdapter
    from graph_loss.neuron_activation_heatmap import (
        _resolve_dataset_path,
        _parse_numeric_args,
    )
    from utils.dataset_json import load_prompt_answer_json
    from graph_loss.precompute_mlp_inputs import (
        load_mlp_input_cache,
        mlp_input_cache_dir,
        mlp_input_cache_exists,
    )

    # ── Load student model ────────────────────────────────────────────────────
    logger.info(
        "Loading student model: %s  (device=%s  dtype=%s)", args.model, device, dtype
    )
    student_model, tokenizer = load_student_model_for_distillation(
        student_source=None,
        student_model_id=args.model,
        device=device,
    )
    student_model = student_model.to(dtype=dtype)
    adapter = HFLlamaGraphAdapter(student_model, tokenizer, device)
    model_name = getattr(adapter.model.config, "_name_or_path", args.model)
    logger.info(
        "Model: %d layers, d_mlp=%d, d_model=%d",
        adapter.n_layers,
        adapter.d_mlp,
        adapter.d_model,
    )

    # ── Resolve dataset and build arg_values ──────────────────────────────────
    dataset_path = _resolve_dataset_path(args.dataset)
    logger.info("Dataset: %s", dataset_path)
    samples = list(load_prompt_answer_json(dataset_path).items())
    numeric_args_by_sample: List[List[int]] = []
    for prompt, _answer in samples:
        try:
            numeric_args_by_sample.append(_parse_numeric_args(prompt))
        except ValueError:
            pass
    if not numeric_args_by_sample:
        raise ValueError(f"No parseable prompts in dataset: {dataset_path}")
    n_arg_dims = len(numeric_args_by_sample[0])
    arg_values: List[List[int]] = [
        sorted({args[dim] for args in numeric_args_by_sample if len(args) == n_arg_dims})
        for dim in range(n_arg_dims)
    ]
    logger.info(
        "Dataset arg grid: %s (total cells: %d)",
        [len(v) for v in arg_values],
        int(__import__("math").prod(len(v) for v in arg_values)),
    )

    # ── Load MLP input cache (REQUIRED) ───────────────────────────────────────
    if not mlp_input_cache_exists(args.mlp_input_cache, model_name, dataset_path):
        expected_dir = mlp_input_cache_dir(args.mlp_input_cache, model_name, dataset_path)
        logger.error(
            "MLP input cache not found. Expected: %r\n"
            "Run:  python -m graph_loss.precompute_mlp_inputs \\\n"
            "          --model %s \\\n"
            "          --dataset %s \\\n"
            "          --cache-dir %s",
            expected_dir,
            args.model,
            args.dataset,
            args.mlp_input_cache,
        )
        sys.exit(1)

    mlp_input_cache = load_mlp_input_cache(args.mlp_input_cache, model_name, dataset_path)
    n_cache_prompts = int(mlp_input_cache["meta"]["n_prompts"])
    n_cache_positions = int(mlp_input_cache["meta"]["n_positions"])
    logger.info(
        "MLP input cache: %d prompts, %d positions", n_cache_prompts, n_cache_positions
    )

    # ── Run GPU-vectorized ANOVA layer by layer ───────────────────────────────
    logger.info(
        "Starting full-coverage GPU ANOVA: %d layers × %d positions × d_mlp=%d …",
        adapter.n_layers,
        n_cache_positions,
        adapter.d_mlp,
    )
    fixed_labels = _run_gpu_anova_all_neurons(
        adapter=adapter,
        mlp_input_cache=mlp_input_cache,
        arg_values=arg_values,
        device=device,
        anova_range_radius=args.anova_range_radius,
        anova_nodes_per_label=args.anova_nodes_per_label,
        sum_min_specificity=args.sum_min_specificity,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(
        "Done. %d neurons labeled (from %d layers × d_mlp=%d)",
        len(fixed_labels),
        adapter.n_layers,
        adapter.d_mlp,
    )
    label_counts: Counter = Counter(fixed_labels.values())
    for lbl, cnt in sorted(label_counts.items()):
        logger.info("  %-45s : %d", lbl, cnt)

    # ── Save ──────────────────────────────────────────────────────────────────
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(fixed_labels, f, indent=2)
    logger.info("Saved → %s", args.output)


if __name__ == "__main__":
    main()

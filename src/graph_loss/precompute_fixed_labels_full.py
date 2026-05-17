"""Pre-compute a frozen neuron→label mapping by running ANOVA on **every** neuron.

Unlike :mod:`precompute_fixed_labels_fast` (which first runs forward passes on
all dataset prompts to filter neurons by ``source_vector_norm`` and then runs
ANOVA on the filtered union), this script skips the filtering phase entirely.

It runs forward passes over all dataset prompts, captures the residual-stream
input to each MLP layer in CPU RAM, then computes activations for *every*
``(layer, neuron_id)`` combination via a vectorized GPU matrix multiply::

    activation_neuron_i(prompt_j, pos_p) = silu(x_jp @ W_gate_i^T) * (x_jp @ W_up_i^T)

where ``x_jp`` is the captured residual-stream input to the MLP at layer ``l``,
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


@torch.no_grad()
def _collect_mlp_inputs_in_memory(
    adapter,
    prompts: List[str],
    numeric_args_by_prompt: List[List[int]],
    *,
    batch_size: int = 8,
) -> dict:
    """Forward-pass all prompts and collect MLP residual-stream inputs in CPU RAM.

    Uses ``register_forward_pre_hook`` on each MLP layer to capture the
    residual-stream tensor entering the MLP (shape ``[bs, seq_len, d_model]``)
    without storing anything to disk.

    Returns a dict with the same structure expected by
    ``_run_gpu_anova_all_neurons``:
      - ``meta``: n_prompts, n_layers, d_model, d_mlp, n_positions,
                  numeric_args_by_prompt
      - ``layer_inputs``: list of ``[n_prompts, n_positions, d_model]`` float16
                          CPU tensors, one per layer.
    """
    from graph_loss.neuron_activation_heatmap import _tokenize_prompt_batch

    n_prompts = len(prompts)
    n_layers = adapter.n_layers
    d_model = adapter.d_model

    first_ids, _ = _tokenize_prompt_batch(adapter, prompts[:1])
    n_positions = first_ids.shape[1]

    layer_inputs = [
        torch.zeros(n_prompts, n_positions, d_model, dtype=torch.float16)
        for _ in range(n_layers)
    ]

    for batch_start in range(0, n_prompts, batch_size):
        batch_prompts = prompts[batch_start : batch_start + batch_size]
        input_ids, _ = _tokenize_prompt_batch(adapter, batch_prompts)
        bs, seq_len = input_ids.shape
        store_len = min(seq_len, n_positions)

        captured: Dict[int, torch.Tensor] = {}
        handles = []
        for layer_idx, layer in enumerate(adapter.layers):
            def _pre_hook(_module, inputs, *, idx=layer_idx):
                captured[idx] = inputs[0].detach().cpu().to(torch.float16)
            handles.append(layer.mlp.register_forward_pre_hook(_pre_hook))

        try:
            adapter.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
            )
        finally:
            for h in handles:
                h.remove()

        for layer_idx in range(n_layers):
            acts = captured[layer_idx]  # [bs, seq_len, d_model]
            layer_inputs[layer_idx][batch_start : batch_start + bs, :store_len, :] = (
                acts[:, :store_len, :]
            )

        logger.info(
            "  Collected %d / %d prompts",
            min(batch_start + batch_size, n_prompts),
            n_prompts,
        )

    return {
        "meta": {
            "n_prompts": n_prompts,
            "n_layers": n_layers,
            "d_model": d_model,
            "d_mlp": adapter.d_mlp,
            "n_positions": n_positions,
            "numeric_args_by_prompt": numeric_args_by_prompt,
        },
        "layer_inputs": layer_inputs,
    }


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

        # Per-position best specificity / label, aggregated across positions
        best_specificity: Dict[str, List[float]] = {
            cat: [float("-inf")] * d_mlp for cat in ANOVA_LABEL_CATEGORIES
        }
        best_label: Dict[str, List[str]] = {
            cat: [""] * d_mlp for cat in ANOVA_LABEL_CATEGORIES
        }

        for pos_idx in range(n_cache_positions):
            x = layer_inputs_cpu[:, pos_idx, :].to(device=device, dtype=torch.float32)  # [n_valid, d_model]

            gate_pre = x @ W_gate.T  # [n_valid, d_mlp]
            up_pre   = x @ W_up.T   # [n_valid, d_mlp]
            acts = F.silu(gate_pre) * up_pre  # [n_valid, d_mlp]

            if pos_idx == 0:
                # BOS / padding token — activations are noisy; zero them out
                acts.zero_()

            # Scatter-add into ANOVA grid [d_mlp, n_flat]
            grid_flat = torch.zeros(d_mlp, n_flat, dtype=torch.float32, device=device)
            grid_flat.scatter_add_(
                1,
                flat_idx_dev.unsqueeze(0).expand(d_mlp, -1),  # [d_mlp, n_valid]
                acts.T.contiguous().float(),
            )

            # Average: [d_mlp, *grid_shape]
            grid = grid_flat.reshape(d_mlp, *grid_shape) / counts_grid

            # ANOVA (CPU-bound Python loop over d_mlp neurons)
            label_results = label_activation_heatmaps(
                grid.cpu(),
                arg_values,
                target_args=None,
                anova_range_radius=anova_range_radius,
            )

            # Update best-across-positions tracking
            for neuron_id, lr in enumerate(label_results):
                for cat in ANOVA_LABEL_CATEGORIES:
                    if cat not in lr.category_specificity:
                        continue
                    score = float(lr.category_scores.get(cat, 0.0))
                    spec  = float(lr.category_specificity[cat])
                    if score > 0.0 and spec > sum_min_specificity and spec > best_specificity[cat][neuron_id]:
                        best_specificity[cat][neuron_id] = spec
                        best_label[cat][neuron_id] = lr.categories.get(cat, "")

            del x, gate_pre, up_pre, acts, grid_flat, grid
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Assign labels: top-K per category
        n_labeled_this_layer = 0
        for cat in ANOVA_LABEL_CATEGORIES:
            scored = [
                (nid, best_specificity[cat][nid])
                for nid in range(d_mlp)
                if best_specificity[cat][nid] > float("-inf") and best_label[cat][nid]
            ]
            scored.sort(key=lambda kv: kv[1], reverse=True)
            for neuron_id, _spec in scored[:anova_nodes_per_label]:
                key = f"{layer_idx}:{neuron_id}"
                if key not in fixed_labels:
                    fixed_labels[key] = best_label[cat][neuron_id]
                    n_labeled_this_layer += 1

        del W_gate, W_up, layer_inputs_cpu, best_specificity, best_label
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
        "--batch-size", type=int, default=8,
        help="Number of prompts per forward pass when collecting MLP inputs (default: 8)",
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
    prompts: List[str] = []
    numeric_args_by_prompt: List[List[int]] = []
    for prompt, _answer in samples:
        try:
            numeric_args_by_prompt.append(_parse_numeric_args(prompt))
            prompts.append(prompt)
        except ValueError:
            pass
    if not prompts:
        raise ValueError(f"No parseable prompts in dataset: {dataset_path}")
    n_arg_dims = len(numeric_args_by_prompt[0])
    arg_values: List[List[int]] = [
        sorted({args[dim] for args in numeric_args_by_prompt if len(args) == n_arg_dims})
        for dim in range(n_arg_dims)
    ]
    logger.info(
        "Dataset arg grid: %s (total cells: %d)",
        [len(v) for v in arg_values],
        int(__import__("math").prod(len(v) for v in arg_values)),
    )

    # ── Collect MLP inputs in memory ─────────────────────────────────────────
    logger.info(
        "Collecting MLP inputs for %d prompts (batch_size=%d) …",
        len(prompts),
        args.batch_size,
    )
    mlp_input_cache = _collect_mlp_inputs_in_memory(
        adapter,
        prompts,
        numeric_args_by_prompt,
        batch_size=args.batch_size,
    )
    n_cache_prompts = int(mlp_input_cache["meta"]["n_prompts"])
    n_cache_positions = int(mlp_input_cache["meta"]["n_positions"])
    logger.info("Collected %d prompts, %d positions", n_cache_prompts, n_cache_positions)

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

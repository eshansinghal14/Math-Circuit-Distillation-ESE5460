"""Pre-compute a frozen neuron→label mapping via forward-only passes on all dataset prompts.

Unlike ``precompute_fixed_labels.py`` (which runs full Jacobian attribution on N=32
training prompts), this script:

1. Runs forward-only passes (no Jacobians, no ``torch.enable_grad()``) on **every**
   prompt in a dataset (e.g. 10k from ``_all.json``).
2. Collects top-K neurons per layer per prompt by
   ``source_vector_norm = |activation| × ‖W_out_row‖``
   (same metric as ``build_graph`` uses, but computed efficiently without building the
   full ``source_vectors`` tensor).
3. Takes the union across all prompts — any neuron that appears in any prompt's top-K
   is a candidate for labeling.
4. Runs ANOVA on the union using the pre-computed MLP input cache.
5. Saves the resulting ``{layer:neuron_id → label}`` JSON — same format as the existing
   script.

Usage (run once before training)::

    python -m graph_loss.precompute_fixed_labels_fast \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --dataset 22_add_tight \\
        --mlp-input-cache /content/mlp_input_cache \\
        --output /content/fixed_labels_fast_1b.json \\
        --prop-neurons-per-layer 5e-4 \\
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
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _ensure_src_on_path() -> None:
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _precompute_out_row_norms(adapter, device: torch.device, dtype: torch.dtype) -> dict[int, torch.Tensor]:
    """Pre-compute ‖W_out_row‖ per neuron per layer (constant across prompts)."""
    norms: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for layer_idx in range(adapter.n_layers):
            _, _, out_rows, _, _ = adapter._layer_weights(
                layer_idx, device=device, dtype=dtype
            )
            # out_rows: [d_mlp, d_model] — norm along d_model axis
            norms[layer_idx] = out_rows.norm(dim=-1)  # [d_mlp]
    logger.info("Pre-computed W_out row norms for %d layers", adapter.n_layers)
    return norms


@torch.no_grad()
def _collect_top_k_neurons(
    adapter,
    prompts: list[str],
    prop: float,
    out_row_norms: dict[int, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    min_frequency: int = 1,
) -> set[tuple[int, int, int]]:
    """Forward pass every prompt and collect the union of top-K (layer, pos, neuron) triples.

    Returns a set of ``(layer_idx, token_pos, neuron_id)`` tuples that appear in
    at least ``min_frequency`` prompts' top-K sets.  Raising ``min_frequency``
    (e.g. to 5-10) dramatically reduces the union size when processing many prompts,
    keeping only neurons that fire consistently across the dataset.
    """
    neuron_counts: Counter = Counter()
    d_mlp = adapter.d_mlp
    n_prompts = len(prompts)

    for i, prompt in enumerate(prompts):
        if i == 0 or (i + 1) % 500 == 0 or (i + 1) == n_prompts:
            logger.info(
                "[%d/%d] forward pass | unique neurons so far: %d",
                i + 1,
                n_prompts,
                len(neuron_counts),
            )

        try:
            input_ids = adapter.ensure_tokenized(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.debug("  Tokenisation failed for %r: %s", prompt[:40], exc)
            continue

        input_batch = input_ids.unsqueeze(0)
        mlp_inputs: dict[int, torch.Tensor] = {}
        handles: list = []

        for layer_idx, layer in enumerate(adapter.layers):
            def _pre_hook(_module, inputs, *, idx: int = layer_idx) -> None:
                mlp_inputs[idx] = inputs[0]

            handles.append(layer.mlp.register_forward_pre_hook(_pre_hook))

        try:
            with adapter.autocast_context(dtype):
                adapter.model(
                    input_ids=input_batch,
                    attention_mask=torch.ones_like(input_batch),
                    output_hidden_states=False,
                    use_cache=False,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("  Forward pass failed for %r: %s", prompt[:40], exc)
            continue
        finally:
            for h in handles:
                h.remove()

        n_pos = int(input_ids.numel())

        for layer_idx in range(adapter.n_layers):
            if layer_idx not in mlp_inputs:
                continue

            layer_input = mlp_inputs[layer_idx].squeeze(0)  # [n_pos, d_model]

            gate_rows, up_rows, _, gate_bias, up_bias = adapter._layer_weights(
                layer_idx, device=device, dtype=layer_input.dtype
            )
            gate_pre = layer_input @ gate_rows.T + gate_bias   # [n_pos, d_mlp]
            up_pre   = layer_input @ up_rows.T + up_bias        # [n_pos, d_mlp]
            neuron_acts = F.silu(gate_pre) * up_pre             # [n_pos, d_mlp]

            # source_vector_norm = |activation| * ‖W_out_row‖
            # (equivalent to norm(neuron_act * W_out_row, dim=-1) without the big tensor)
            layer_out_norms = out_row_norms[layer_idx].to(
                device=layer_input.device, dtype=layer_input.dtype
            )  # [d_mlp]
            flat_norms = (neuron_acts.abs() * layer_out_norms).reshape(-1)  # [n_pos * d_mlp]

            total = int(flat_norms.numel())
            k = min(total, max(1, int(total * prop)))
            keep = torch.topk(flat_norms, k).indices  # flat indices [k]

            # Decode: flat_idx = token_pos * d_mlp + neuron_id
            keep_pos = (keep // d_mlp).tolist()
            keep_nid = (keep % d_mlp).tolist()
            for pos, nid in zip(keep_pos, keep_nid):
                neuron_counts[(layer_idx, pos, nid)] += 1

        mlp_inputs.clear()

    # Keep only neurons that appear in at least min_frequency prompts
    all_neurons = {k for k, v in neuron_counts.items() if v >= min_frequency}
    logger.info(
        "Frequency filter (min=%d): %d → %d unique neurons",
        min_frequency, len(neuron_counts), len(all_neurons),
    )
    return all_neurons


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
             "(created by precompute_mlp_inputs.py)",
    )
    parser.add_argument(
        "--activation-write-cache", default=None,
        help="(Unused — kept for CLI parity with precompute_fixed_labels.py)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSON path for the fixed label mapping",
    )
    parser.add_argument(
        "--prop-neurons-per-layer", type=float, default=5e-4,
        help="Fraction of (token_pos × neuron_id) pairs to keep per layer per prompt "
             "(default: 5e-4, same as training)",
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
    from graph_loss.precompute_mlp_inputs import load_mlp_input_cache, mlp_input_cache_exists
    from graph_loss.anova_node_labels import ANOVA_LABEL_CATEGORIES, label_activation_heatmaps
    from utils import load_prompt_answer_json

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
    logger.info("Model loaded: %d layers, d_mlp=%d, d_model=%d", adapter.n_layers, adapter.d_mlp, adapter.d_model)

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset_path = _resolve_dataset_path(args.dataset)
    logger.info("Dataset resolved to: %s", dataset_path)
    samples = list(load_prompt_answer_json(dataset_path).items())
    prompts = [p for p, _ in samples]
    logger.info("Loaded %d prompts", len(prompts))

    # ── Pre-compute W_out row norms (constant, computed once) ─────────────────
    out_row_norms = _precompute_out_row_norms(adapter, device, dtype)

    # ── Forward-only passes: collect union of top-K neurons ───────────────────
    logger.info(
        "Running forward-only passes on %d prompts  (prop_neurons=%.2e) ...",
        len(prompts),
        args.prop_neurons_per_layer,
    )
    all_neurons = _collect_top_k_neurons(
        adapter,
        prompts,
        prop=args.prop_neurons_per_layer,
        out_row_norms=out_row_norms,
        device=device,
        dtype=dtype,
    )
    logger.info(
        "Union: %d unique (layer, token_pos, neuron_id) triples from %d prompts",
        len(all_neurons),
        len(prompts),
    )

    if not all_neurons:
        logger.error("No neurons selected — check --prop-neurons-per-layer and the dataset.")
        sys.exit(1)

    # [N, 3] tensor sorted for reproducibility
    neuron_locations = torch.tensor(sorted(all_neurons), dtype=torch.long)
    logger.info("Neuron location tensor: %s", tuple(neuron_locations.shape))

    # ── Load MLP input cache ──────────────────────────────────────────────────
    mlp_input_cache = None
    if mlp_input_cache_exists(args.mlp_input_cache, model_name, dataset_path):
        mlp_input_cache = load_mlp_input_cache(args.mlp_input_cache, model_name, dataset_path)
        cache_meta = mlp_input_cache["meta"]
        logger.info(
            "Loaded MLP input cache: %d prompts, %d positions",
            cache_meta["n_prompts"],
            cache_meta["n_positions"],
        )
    else:
        logger.warning(
            "MLP input cache not found at %s for model=%r dataset=%r — "
            "ANOVA will run live forward passes (slow). "
            "Run precompute_mlp_inputs.py first for best performance.",
            args.mlp_input_cache,
            model_name,
            dataset_path,
        )

    # ── Build activation heatmaps (ANOVA grid) ────────────────────────────────
    logger.info("Building activation write result for %d neurons ...", len(neuron_locations))
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
    logger.info("Running ANOVA labeling (target_args=None — global across all arg values) ...")
    label_results = label_activation_heatmaps(
        result.activations,
        result.arg_values,
        target_args=None,   # global: use all arg values, not a single prompt's args
        anova_range_radius=args.anova_range_radius,
    )

    # ── Category loop: pick top anova_nodes_per_label per category ────────────
    # Mirrors the full_search branch in build_super_graph (graph.py) but simplified:
    # uses category_specificity as the ranking score for ALL categories (including
    # "sum range"/"sum units"), since we have no per-prompt target_args to compute
    # the cosine-score alternative ranking.
    fixed_labels: dict[str, str] = {}

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
            continue

        newly_labeled = 0
        for row_idx, score in top_rows:
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

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(
        "Fixed label map: %d neurons labeled  (from union of %d unique triples across %d prompts)",
        len(fixed_labels),
        len(all_neurons),
        len(prompts),
    )
    label_counts: Counter = Counter(fixed_labels.values())
    for lbl, cnt in sorted(label_counts.items()):
        logger.info("  %-40s : %d neurons", lbl, cnt)

    # ── Save ──────────────────────────────────────────────────────────────────
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(fixed_labels, f, indent=2)
    logger.info("Saved fixed labels to %s", args.output)


if __name__ == "__main__":
    main()

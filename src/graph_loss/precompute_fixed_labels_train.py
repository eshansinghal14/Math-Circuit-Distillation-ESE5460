"""Pre-compute a frozen neuron→label mapping conditioned on per-training-prompt target_args.

Unlike :mod:`precompute_fixed_labels_full` which runs ANOVA with ``target_args=None``
(producing globally-best labels like "arg1 11-11"), this script:

1. Builds the mean-activation grid from the **10k MLP cache** (dense statistics).
2. For each unique ``(a1, a2)`` pair in the **training** dataset (e.g. the 192
   prompts from ``22_add_tight_5000_train.json``), runs vectorized ANOVA with
   ``target_args=(a1, a2)``, producing labels like "arg1 22-22" that exactly
   match what the teacher produces for those prompts.
3. Unions all per-prompt labeled neurons (first assignment wins — highest
   specificity score because prompts are iterated in sorted order).

This fixes the teacher-student alignment regression caused by global
``target_args=None`` labels that don't match the teacher's per-prompt labels.

Same output format as :mod:`precompute_fixed_labels_full`:
``{"{layer}:{neuron_id}": "<label>"}`` — drop-in compatible.

Usage (run once before training)::

    python -m graph_loss.precompute_fixed_labels_train \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --train-dataset "$DRIVE/datasets/22_add_tight_5000_train.json" \\
        --mlp-input-cache /content/local_caches/mlp-input-cache \\
        --output /content/fixed_labels_1b_train.json \\
        --anova-nodes-per-label 3 \\
        --anova-range-radius 0 \\
        --sum-min-specificity 1e-2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _ensure_src_on_path() -> None:
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _find_mlp_cache(cache_root: str, model_name: str) -> dict:
    """Auto-detect the MLP input cache for *model_name* under *cache_root*.

    Scans ``<cache_root>/<model_slug>/*/meta.pt`` and returns the first hit.
    Falls back to scanning all immediate subdirectories of *cache_root*.
    """
    import hashlib
    import re

    def _model_slug(name: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")[:48]
        digest = hashlib.sha1(name.encode()).hexdigest()[:8]
        return f"{safe}_{digest}"

    def _try_dir(d: str):
        meta_path = os.path.join(d, "meta.pt")
        if os.path.isfile(meta_path):
            return meta_path
        return None

    model_slug = _model_slug(model_name)
    candidates = [os.path.join(cache_root, model_slug)]

    # Also try raw model_name components in case slug differs
    if os.path.isdir(cache_root):
        for subdir in sorted(os.listdir(cache_root)):
            full = os.path.join(cache_root, subdir)
            if os.path.isdir(full):
                candidates.append(full)

    for candidate_dir in candidates:
        if not os.path.isdir(candidate_dir):
            continue
        for inner in sorted(os.listdir(candidate_dir)):
            inner_path = os.path.join(candidate_dir, inner)
            if os.path.isdir(inner_path):
                meta_path = _try_dir(inner_path)
                if meta_path is not None:
                    logger.info("Found MLP cache at: %s", inner_path)
                    meta = torch.load(meta_path, map_location="cpu", weights_only=True)
                    n_layers = int(meta["n_layers"])
                    layer_inputs = []
                    for i in range(n_layers):
                        layer_pt = os.path.join(inner_path, f"layer_{i}.pt")
                        layer_inputs.append(
                            torch.load(layer_pt, map_location="cpu", weights_only=True)
                        )
                    return {"meta": meta, "layer_inputs": layer_inputs}

    raise FileNotFoundError(
        f"No MLP input cache (meta.pt) found under {cache_root!r} for model {model_name!r}. "
        "Run precompute_mlp_inputs.py first."
    )


def _run_per_prompt_gpu_anova(
    adapter,
    mlp_input_cache: dict,
    cache_arg_values: List[List[int]],
    train_target_args: List[Tuple[int, ...]],
    device: torch.device,
    anova_range_radius: int,
    anova_nodes_per_label: int,
    sum_min_specificity: float,
) -> Dict[str, str]:
    """Layer-by-layer GPU ANOVA conditioned on per-training-prompt target_args.

    For each layer:
      1. Build the mean-activation grid ``[d_mlp, n_flat]`` from the MLP cache
         (same as :func:`precompute_fixed_labels_full._run_gpu_anova_all_neurons`).
      2. For each unique ``(a1, a2)`` in *train_target_args*, build ANOVA rules with
         ``target_args=(a1, a2)`` and score all neurons in one vectorized matmul.
      3. Union results: first-assignment-wins across (a1, a2) pairs.

    Produces labels like "arg1 22-22" that exactly match the teacher's per-prompt
    labels, fixing the S3.1 alignment regression.
    """
    import numpy as np  # noqa: F401 (used via local alias throughout this function)
    from graph_loss.anova_node_labels import (
        ANOVA_LABEL_CATEGORIES,
        BASE_ANOVA_LABEL_CATEGORIES,
        CATEGORY_COMPONENTS,
        build_anova_basis_rules,
    )
    from graph_loss.precompute_fixed_labels_full import _vectorized_anova_scores

    cache_meta = mlp_input_cache["meta"]
    cache_layer_inputs: List[torch.Tensor] = mlp_input_cache["layer_inputs"]
    cache_args: List[List[int]] = cache_meta["numeric_args_by_prompt"]
    n_cache_prompts = int(cache_meta["n_prompts"])
    n_cache_positions = int(cache_meta["n_positions"])

    n_grid_dims = len(cache_arg_values)
    grid_shape = tuple(len(v) for v in cache_arg_values)
    arg_to_idx = [{v: i for i, v in enumerate(vals)} for vals in cache_arg_values]

    n_flat = 1
    for s in grid_shape:
        n_flat *= s

    # Row-major strides for flat index computation
    strides = [1] * n_grid_dims
    for dim in range(n_grid_dims - 2, -1, -1):
        strides[dim] = strides[dim + 1] * grid_shape[dim + 1]

    # Build valid-prompt mask and flat grid index (reused every layer)
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
            "same (or superset) dataset as the training set."
        )

    flat_idx_dev = flat_indices[valid_idx].to(device)  # [n_valid] on GPU

    # Per-cell prompt counts — same for all neurons, layers, positions
    counts_flat = torch.zeros(n_flat, dtype=torch.float32, device=device)
    counts_flat.scatter_add_(
        0, flat_idx_dev, torch.ones(n_valid, dtype=torch.float32, device=device)
    )
    counts_grid = counts_flat.reshape(grid_shape).clamp(min=1.0)

    n_layers = adapter.n_layers
    d_mlp = adapter.d_mlp

    # Deduplicate target_args while preserving order
    seen_targets: set = set()
    unique_targets: List[Tuple[int, ...]] = []
    for t in train_target_args:
        if t not in seen_targets:
            seen_targets.add(t)
            unique_targets.append(t)
    logger.info(
        "Unique (a1, a2) pairs from training set: %d", len(unique_targets)
    )

    fixed_labels: Dict[str, str] = {}

    for layer_idx in range(n_layers):
        t0 = time.time()
        logger.info("━━ Layer %d / %d ━━", layer_idx + 1, n_layers)

        hf_layer = adapter.layers[layer_idx]
        W_gate = hf_layer.mlp.gate_proj.weight.to(device=device, dtype=torch.float32)
        W_up   = hf_layer.mlp.up_proj.weight.to(device=device, dtype=torch.float32)
        layer_inputs_cpu = cache_layer_inputs[layer_idx][valid_idx]  # [n_valid, n_pos, d_model]

        # ── Build mean-activation grid for this layer: [d_mlp, n_flat] ───────
        layer_grid_flat = torch.zeros(d_mlp, n_flat, dtype=torch.float32, device=device)
        n_active_positions = 0
        for pos_idx in range(n_cache_positions):
            if pos_idx == 0:
                continue  # skip BOS
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
        grid_flat = layer_grid_flat / n_active_positions / counts_grid.reshape(1, -1)
        del layer_grid_flat, W_gate, W_up, layer_inputs_cpu

        n_labeled_this_layer = 0

        # ── For each unique training (a1, a2), run vectorized ANOVA ──────────
        for pair_idx, target_args in enumerate(unique_targets):
            if pair_idx == 0 or (pair_idx + 1) % 20 == 0:
                logger.info(
                    "  [layer %d] Processing (%d/%d): target_args=%s",
                    layer_idx + 1,
                    pair_idx + 1,
                    len(unique_targets),
                    target_args,
                )

            rules = build_anova_basis_rules(
                cache_arg_values,
                target_args=target_args,
                anova_range_radius=anova_range_radius,
            )
            if not rules:
                continue

            masks_gpu = torch.stack(
                [rule.mask.float().reshape(-1).to(device) for rule in rules], dim=0
            )  # [n_rules, n_flat]

            category_rule_indices: Dict[str, List[int]] = defaultdict(list)
            for ri, rule in enumerate(rules):
                category_rule_indices[rule.category].append(ri)

            # scores: [d_mlp, n_rules]
            scores_np = _vectorized_anova_scores(grid_flat.detach(), masks_gpu)
            del masks_gpu

            # ── Per-category: best rule label and score for each neuron ──────
            cat_scores: Dict[str, "np.ndarray"] = {}
            cat_labels: Dict[str, List[str]] = {}
            for category, rule_idxs in category_rule_indices.items():
                sub = scores_np[:, rule_idxs]               # [d_mlp, n_cat_rules]
                best_local = sub.argmax(axis=1)              # [d_mlp]
                cat_scores[category] = sub.max(axis=1)       # [d_mlp]
                cat_labels[category] = [
                    rules[rule_idxs[best_local[nid]]].label for nid in range(d_mlp)
                ]

            # Combined categories
            if "arg1 units" in cat_scores and "arg2 units" in cat_scores:
                cat_scores["arg1 units and arg2 units"] = np.minimum(
                    cat_scores["arg1 units"], cat_scores["arg2 units"]
                )
                cat_labels["arg1 units and arg2 units"] = [
                    f"{cat_labels['arg1 units'][n]} and {cat_labels['arg2 units'][n]}"
                    for n in range(d_mlp)
                ]
            if "arg1 range" in cat_scores and "arg2 range" in cat_scores:
                cat_scores["arg1 range and arg2 range"] = np.minimum(
                    cat_scores["arg1 range"], cat_scores["arg2 range"]
                )
                cat_labels["arg1 range and arg2 range"] = [
                    f"{cat_labels['arg1 range'][n]} and {cat_labels['arg2 range'][n]}"
                    for n in range(d_mlp)
                ]

            # ── Specificity filter + top-K assignment ─────────────────────────
            for category in ANOVA_LABEL_CATEGORIES:
                if category not in cat_scores:
                    continue
                cat_arr = cat_scores[category]
                excluded = {category} | CATEGORY_COMPONENTS.get(category, set())
                competitor = np.zeros(d_mlp, dtype=np.float32)
                for comp in BASE_ANOVA_LABEL_CATEGORIES:
                    if comp not in excluded and comp in cat_scores:
                        competitor = np.maximum(competitor, cat_scores[comp])
                specificity = cat_arr - competitor

                scored = [
                    (nid, float(specificity[nid]))
                    for nid in range(d_mlp)
                    if float(cat_arr[nid]) > 0.0 and float(specificity[nid]) > sum_min_specificity
                ]
                scored.sort(key=lambda kv: kv[1], reverse=True)
                for neuron_id, _spec in scored[:anova_nodes_per_label]:
                    key = f"{layer_idx}:{neuron_id}"
                    if key not in fixed_labels:
                        fixed_labels[key] = cat_labels[category][neuron_id]
                        n_labeled_this_layer += 1

        del grid_flat
        if device.type == "cuda":
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        logger.info(
            "  Layer %d done in %.1fs — %d new labels this layer, %d total so far",
            layer_idx + 1,
            elapsed,
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
        "--train-dataset", required=True,
        help="Path to training JSON (e.g. 22_add_tight_5000_train.json) — "
             "used to extract the (a1, a2) target_args for each training prompt",
    )
    parser.add_argument(
        "--mlp-input-cache", required=True,
        help="Path to the pre-computed MLP input cache root directory "
             "(created by precompute_mlp_inputs.py for the full 10k dataset). "
             "The cache is auto-detected by scanning model-slug subdirectories.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSON path for the fixed label mapping",
    )
    parser.add_argument(
        "--anova-nodes-per-label", type=int, default=3,
        help="Maximum neurons to assign per ANOVA label category per (a1, a2) prompt (default: 3)",
    )
    parser.add_argument(
        "--anova-range-radius", type=int, default=0,
        help="Radius around target arg for range-label rules (0 = exact match)",
    )
    parser.add_argument(
        "--sum-min-specificity", type=float, default=1e-2,
        help="Minimum ANOVA specificity a neuron must have to receive any label (default: 1e-2)",
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

    # ── Deferred imports ───────────────────────────────────────────────────────
    from utils.hf_models import load_student_model_for_distillation
    from graph_loss.hf_adapter import HFLlamaGraphAdapter
    from graph_loss.anova_node_labels import parse_numeric_args
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
    model_name = getattr(adapter.model.config, "_name_or_path", args.model)
    logger.info(
        "Model: %d layers, d_mlp=%d, d_model=%d",
        adapter.n_layers,
        adapter.d_mlp,
        adapter.d_model,
    )

    # ── Load training dataset and parse per-prompt (a1, a2) ───────────────────
    if not os.path.isfile(args.train_dataset):
        raise FileNotFoundError(f"Training dataset not found: {args.train_dataset}")

    train_samples = load_prompt_answer_json(args.train_dataset)
    train_targets: List[Tuple[int, ...]] = []
    for prompt in train_samples.keys():
        try:
            train_targets.append(parse_numeric_args(prompt))
        except ValueError:
            pass

    if not train_targets:
        raise ValueError(f"No parseable prompts in training dataset: {args.train_dataset}")

    unique_targets = list(dict.fromkeys(train_targets))
    logger.info(
        "Training dataset: %d prompts → %d unique (a1, a2) pairs",
        len(train_samples),
        len(unique_targets),
    )

    # ── Auto-detect and load MLP input cache ──────────────────────────────────
    logger.info(
        "Auto-detecting MLP input cache under %r for model %r …",
        args.mlp_input_cache,
        model_name,
    )
    mlp_input_cache = _find_mlp_cache(args.mlp_input_cache, model_name)

    # Reconstruct cache_arg_values from the cache metadata
    cache_meta = mlp_input_cache["meta"]
    cache_arg_values: List[List[int]] = cache_meta["arg_values"]
    n_cache_prompts = int(cache_meta["n_prompts"])
    n_cache_positions = int(cache_meta["n_positions"])
    logger.info(
        "MLP input cache: %d prompts, %d positions, arg grid %s (total cells: %d)",
        n_cache_prompts,
        n_cache_positions,
        [len(v) for v in cache_arg_values],
        int(__import__("math").prod(len(v) for v in cache_arg_values)),
    )

    # ── Run per-prompt GPU-vectorized ANOVA layer by layer ───────────────────
    logger.info(
        "Starting per-prompt GPU ANOVA: %d layers × %d positions × d_mlp=%d × %d prompt-pairs …",
        adapter.n_layers,
        n_cache_positions,
        adapter.d_mlp,
        len(unique_targets),
    )
    fixed_labels = _run_per_prompt_gpu_anova(
        adapter=adapter,
        mlp_input_cache=mlp_input_cache,
        cache_arg_values=cache_arg_values,
        train_target_args=unique_targets,
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
    logger.info("Label breakdown (top 30):")
    for lbl, cnt in sorted(label_counts.items())[:30]:
        logger.info("  %-50s : %d", lbl, cnt)
    logger.info("  … (%d distinct labels total)", len(label_counts))

    # ── Save ──────────────────────────────────────────────────────────────────
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(fixed_labels, f, indent=2)
    logger.info("Saved → %s", args.output)


if __name__ == "__main__":
    main()

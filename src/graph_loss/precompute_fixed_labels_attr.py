"""Pre-compute a frozen neuron→label mapping using attribution-aligned neuron selection.

Unlike :mod:`precompute_fixed_labels_train` (which selects neurons by ANOVA explained
variance), this script selects neurons by ``source_vector_norm = |activation| × ‖W_out_row‖``
— the **same metric** used by the student supergraph during training via
``prop_neurons_per_layer``.

This guarantees that the precomputed labeled neurons are exactly the ones the student will
select at runtime, fixing the S3.1 alignment issue where only ~3/9 expected labeled neurons
were found per prompt due to ANOVA vs attribution selection mismatch.

MLP cache format (from :mod:`precompute_mlp_inputs`):
    ``layer_{i}.pt``  shape ``[n_prompts, n_positions, d_model]`` (bfloat16)
    These are the **residual-stream inputs** to the MLP (before gate/up projections).
    To get gated neuron activations: ``act = silu(x @ W_gate.T) * (x @ W_up.T)``.
    To get source_vector_norm for neuron i: ``|act[:, i]| * ‖W_out[:, i]‖₂``.

Algorithm:
  Phase 1 — For each unique (a1, a2) in the training dataset, retrieve the matching
             MLP-cache entry (no new forward passes) and compute per-neuron
             ``source_norm = max_pos(|act| × ‖W_out_row‖)``.  Select the top
             ``int(n_pos × d_mlp × prop_neurons_per_layer)`` neurons (flat across
             positions, same as training).  Build a union across all training prompts
             and record the (a1, a2) where each neuron scored highest.

  Phase 2 — Build the mean-activation ANOVA grid from the full 10k cache (same as
             :mod:`precompute_fixed_labels_train`).  For each union neuron look up its
             best ANOVA label using the (a1, a2) where it scored highest — producing
             labels like "arg1 22-22" that match the teacher's per-prompt context.

Same output format as :mod:`precompute_fixed_labels_train`:
``{"{layer}:{neuron_id}": "<label>"}`` — drop-in compatible.

Usage (run once before training)::

    python -m graph_loss.precompute_fixed_labels_attr \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --train-dataset "$DRIVE/datasets/22_add_tight_5000_train.json" \\
        --mlp-input-cache /content/local_caches/mlp-input-cache \\
        --output /content/fixed_labels_1b_attr.json \\
        --prop-neurons-per-layer 1e-3 \\
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
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _ensure_src_on_path() -> None:
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


# ---------------------------------------------------------------------------
# MLP cache loader (same auto-detect logic as precompute_fixed_labels_train)
# ---------------------------------------------------------------------------

def _find_mlp_cache(cache_root: str, model_name: str) -> dict:
    """Auto-detect the MLP input cache for *model_name* under *cache_root*.

    Scans ``<cache_root>/<model_slug>/*/meta.pt`` and returns the first hit.
    """
    import hashlib
    import re

    def _model_slug(name: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")[:48]
        digest = hashlib.sha1(name.encode()).hexdigest()[:8]
        return f"{safe}_{digest}"

    def _try_dir(d: str) -> Optional[str]:
        meta_path = os.path.join(d, "meta.pt")
        return meta_path if os.path.isfile(meta_path) else None

    model_slug = _model_slug(model_name)
    candidates = [os.path.join(cache_root, model_slug)]

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


# ---------------------------------------------------------------------------
# Per-neuron label assignment from precomputed ANOVA score tables
# ---------------------------------------------------------------------------

def _label_union_neurons_from_tables(
    tables: dict,
    union_neurons: Set[int],
    best_target: Dict[int, Tuple[int, ...]],
    layer_idx: int,
    arg_to_idx: List[Dict[int, int]],
    fixed_labels: Dict[str, str],
    sum_min_specificity: float,
) -> int:
    """Assign the best ANOVA label to each union neuron given its best (a1, a2).

    For each neuron in *union_neurons* the function looks up its explained-variance
    scores from the precomputed *tables* (indexed by its best (a1, a2) pair), computes
    specificity (score minus max competitor base-category score), and assigns the
    ANOVA category with the highest specificity that exceeds *sum_min_specificity*.

    One neuron → at most one label (the globally best specificity category).

    Args:
        tables:             Output of :func:`_build_fast_anova_tables`.
        union_neurons:      Set of neuron IDs to label in this layer.
        best_target:        Mapping neuron_id → (a1, a2) for the training prompt
                            where that neuron's source_norm was highest.
        layer_idx:          Current layer (used to build the ``"L:N"`` key).
        arg_to_idx:         ``[{a1_val: grid_idx}, {a2_val: grid_idx}]``.
        fixed_labels:       Running label dict modified in place.
        sum_min_specificity: Minimum specificity threshold for label assignment.

    Returns:
        Number of new neuron labels added.
    """
    from graph_loss.anova_node_labels import (
        ANOVA_LABEL_CATEGORIES,
        BASE_ANOVA_LABEL_CATEGORIES,
        CATEGORY_COMPONENTS,
    )

    scores_a1    = tables["scores_a1"]     # [d_mlp, n_arg1] numpy
    scores_a2    = tables["scores_a2"]     # [d_mlp, n_arg2]
    scores_sum   = tables["scores_sum"]    # [d_mlp, n_sums]
    scores_u1    = tables["scores_u1"]     # [d_mlp, 10]
    scores_u2    = tables["scores_u2"]     # [d_mlp, 10]
    scores_su    = tables["scores_su"]     # [d_mlp, 10]
    scores_carry = tables["scores_carry"]  # [d_mlp] or None
    has_carry    = tables["has_carry"]
    min_sum      = tables["min_sum"]
    n_sums       = tables["n_sums"]

    n_labeled = 0

    for neuron_id in sorted(union_neurons):
        key = f"{layer_idx}:{neuron_id}"
        if key in fixed_labels:
            continue

        target_args = best_target.get(neuron_id)
        if target_args is None:
            continue

        a1, a2 = int(target_args[0]), int(target_args[1])
        a1_idx = arg_to_idx[0].get(a1, -1)
        a2_idx = arg_to_idx[1].get(a2, -1)
        s      = a1 + a2
        s_idx  = s - min_sum
        u1, u2, su = a1 % 10, a2 % 10, s % 10

        # Scalar ANOVA scores for this neuron at its best (a1, a2) context
        cs_a1r = float(scores_a1[neuron_id, a1_idx]) if 0 <= a1_idx < scores_a1.shape[1] else 0.0
        cs_a2r = float(scores_a2[neuron_id, a2_idx]) if 0 <= a2_idx < scores_a2.shape[1] else 0.0
        cs_sr  = float(scores_sum[neuron_id, s_idx])  if 0 <= s_idx  < n_sums               else 0.0
        cs_u1  = float(scores_u1[neuron_id, u1])
        cs_u2  = float(scores_u2[neuron_id, u2])
        cs_su  = float(scores_su[neuron_id, su])

        label_a1r = f"arg1 {a1}-{a1}"
        label_a2r = f"arg2 {a2}-{a2}"
        label_u1  = f"arg1 units {u1}"
        label_u2  = f"arg2 units {u2}"
        label_sr  = f"sum {s}-{s}"
        label_su  = f"sum units {su}"

        base_cs: Dict[str, float] = {
            "arg1 range": cs_a1r,
            "arg1 units": cs_u1,
            "arg2 range": cs_a2r,
            "arg2 units": cs_u2,
            "sum range":  cs_sr,
            "sum units":  cs_su,
        }
        if has_carry and scores_carry is not None:
            base_cs["carry"] = float(scores_carry[neuron_id])

        best_specificity = -float("inf")
        best_label: Optional[str] = None

        for category in ANOVA_LABEL_CATEGORIES:
            if category == "arg1 range":
                cat_score, label_str = cs_a1r, label_a1r
            elif category == "arg1 units":
                cat_score, label_str = cs_u1, label_u1
            elif category == "arg2 range":
                cat_score, label_str = cs_a2r, label_a2r
            elif category == "arg2 units":
                cat_score, label_str = cs_u2, label_u2
            elif category == "arg1 units and arg2 units":
                cat_score = min(cs_u1, cs_u2)
                label_str = f"{label_u1} and {label_u2}"
            elif category == "arg1 range and arg2 range":
                cat_score = min(cs_a1r, cs_a2r)
                label_str = f"{label_a1r} and {label_a2r}"
            elif category == "carry":
                if not has_carry or scores_carry is None:
                    continue
                cat_score = float(scores_carry[neuron_id])
                label_str = "carry"
            elif category == "sum range":
                cat_score, label_str = cs_sr, label_sr
            elif category == "sum units":
                cat_score, label_str = cs_su, label_su
            else:
                continue

            if cat_score <= 0.0:
                continue

            excluded    = {category} | CATEGORY_COMPONENTS.get(category, set())
            competitor  = 0.0
            for comp in BASE_ANOVA_LABEL_CATEGORIES:
                if comp not in excluded and comp in base_cs:
                    competitor = max(competitor, base_cs[comp])
            specificity = cat_score - competitor

            if specificity <= sum_min_specificity:
                continue

            if specificity > best_specificity:
                best_specificity = specificity
                best_label       = label_str

        if best_label is not None:
            fixed_labels[key] = best_label
            n_labeled += 1

    return n_labeled


# ---------------------------------------------------------------------------
# Main computation: attribution union + ANOVA labeling, layer by layer
# ---------------------------------------------------------------------------

def _run_attr_union_and_label(
    adapter,
    mlp_input_cache: dict,
    cache_arg_values: List[List[int]],
    train_target_args: List[Tuple[int, ...]],
    device: torch.device,
    prop_neurons_per_layer: float,
    anova_range_radius: int,
    anova_nodes_per_label: int,
    sum_min_specificity: float,
) -> Dict[str, str]:
    """Layer-by-layer attribution selection + ANOVA labeling.

    For each layer:
      1. **Attribution pass** — for each unique training (a1, a2) pair, load the
         corresponding cached MLP residual-stream input, compute gated activations,
         score each neuron by ``|act| × ‖W_out_row‖``, and select the top
         ``int(n_pos × d_mlp × prop_neurons_per_layer)`` flat entries (same logic
         as the student supergraph at training time).  Track the best (a1, a2) per
         neuron across all training prompts.

      2. **ANOVA grid** — accumulate the mean-activation grid
         ``[d_mlp, n_flat]`` from all valid cache prompts (identical to
         :mod:`precompute_fixed_labels_train`).

      3. **Score tables** — call :func:`_build_fast_anova_tables` once per layer
         (7 batched matmuls, radius-0 path).

      4. **Label** — for each union neuron look up its best ANOVA label using its
         best (a1, a2) context via :func:`_label_union_neurons_from_tables`.

    Args:
        adapter:                 HFLlamaGraphAdapter (model weights only).
        mlp_input_cache:         Loaded cache dict (meta + layer_inputs list).
        cache_arg_values:        ``[arg1_vals, arg2_vals]`` from cache meta.
        train_target_args:       Deduplicated list of (a1, a2) from training set.
        device:                  Compute device.
        prop_neurons_per_layer:  Fraction of (n_pos × d_mlp) to keep per prompt —
                                 must match the value used in distillation.
        anova_range_radius:      Radius for ANOVA range-label rules (0 = fast path).
        anova_nodes_per_label:   Kept for CLI compatibility; not used as hard limit
                                 since attribution already controls union size.
        sum_min_specificity:     Minimum specificity threshold for label assignment.

    Returns:
        ``{"{layer}:{neuron_id}": "<label>"}`` mapping.
    """
    from graph_loss.precompute_fixed_labels_full import _build_fast_anova_tables

    cache_meta         = mlp_input_cache["meta"]
    cache_layer_inputs = mlp_input_cache["layer_inputs"]  # list[Tensor[n_cache, n_pos, d_model]]
    cache_args         = cache_meta["numeric_args_by_prompt"]  # List[List[int]]
    n_cache_prompts    = int(cache_meta["n_prompts"])
    n_cache_positions  = int(cache_meta["n_positions"])

    n_grid_dims = len(cache_arg_values)
    grid_shape  = tuple(len(v) for v in cache_arg_values)
    arg_to_idx  = [{v: i for i, v in enumerate(vals)} for vals in cache_arg_values]

    n_flat = 1
    for s in grid_shape:
        n_flat *= s

    # Row-major strides for flat index
    strides = [1] * n_grid_dims
    for dim in range(n_grid_dims - 2, -1, -1):
        strides[dim] = strides[dim + 1] * grid_shape[dim + 1]

    # Map each (a1, a2) → list of cache prompt indices (for attribution lookups)
    target_to_cache_idx: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for j, args in enumerate(cache_args):
        if len(args) == n_grid_dims:
            target_to_cache_idx[tuple(args)].append(j)

    # Build valid_mask + flat_indices for ANOVA grid accumulation (reused every layer)
    valid_mask  = torch.zeros(n_cache_prompts, dtype=torch.bool)
    flat_indices = torch.zeros(n_cache_prompts, dtype=torch.long)
    for j, args in enumerate(cache_args):
        if len(args) != n_grid_dims:
            continue
        ok   = True
        flat = 0
        for dim, val in enumerate(args):
            if val not in arg_to_idx[dim]:
                ok = False
                break
            flat += arg_to_idx[dim][val] * strides[dim]
        if ok:
            valid_mask[j]   = True
            flat_indices[j] = flat

    valid_idx   = valid_mask.nonzero(as_tuple=True)[0]  # [n_valid]
    n_valid     = int(valid_idx.shape[0])
    logger.info("Valid prompts for ANOVA grid: %d / %d", n_valid, n_cache_prompts)
    if n_valid == 0:
        raise ValueError(
            "No valid prompts found — check that the MLP cache was built with the "
            "same (or superset) dataset as the training set."
        )

    flat_idx_dev  = flat_indices[valid_idx].to(device)  # [n_valid] on device
    counts_flat   = torch.zeros(n_flat, dtype=torch.float32, device=device)
    counts_flat.scatter_add_(0, flat_idx_dev, torch.ones(n_valid, dtype=torch.float32, device=device))
    counts_grid   = counts_flat.reshape(grid_shape).clamp(min=1.0)

    # Coverage check
    n_covered = sum(1 for t in train_target_args if t in target_to_cache_idx)
    logger.info(
        "Training targets covered by MLP cache: %d / %d unique (a1, a2) pairs",
        n_covered, len(train_target_args),
    )
    if n_covered == 0:
        raise ValueError(
            "No training (a1, a2) pairs found in MLP cache. "
            "Ensure the cache was built with the full dataset that includes all "
            "training arg combinations."
        )

    n_layers = adapter.n_layers
    d_mlp    = adapter.d_mlp

    use_fast = (anova_range_radius == 0 and n_grid_dims == 2)
    if use_fast:
        logger.info(
            "Using fast vectorized ANOVA (radius=0, 2-D grid): "
            "score tables precomputed per layer, O(1) lookup per union neuron."
        )
    else:
        logger.warning(
            "anova_range_radius=%d or non-2-D grid: only the fast path (radius=0, 2-D) "
            "is implemented.  Skipping ANOVA for non-fast cases.",
            anova_range_radius,
        )

    fixed_labels: Dict[str, str] = {}

    for layer_idx in range(n_layers):
        t0 = time.time()
        logger.info("━━ Layer %d / %d ━━", layer_idx + 1, n_layers)

        hf_layer = adapter.layers[layer_idx]

        # Load weights — cast to float32 for numerical stability
        W_gate = hf_layer.mlp.gate_proj.weight.to(device=device, dtype=torch.float32)  # [d_mlp, d_model]
        W_up   = hf_layer.mlp.up_proj.weight.to(device=device, dtype=torch.float32)    # [d_mlp, d_model]
        W_out  = hf_layer.mlp.down_proj.weight.to(device=device, dtype=torch.float32)  # [d_model, d_mlp]

        # W_out_norms[i] = ‖W_out[:, i]‖₂  (norm of the output-projection row for neuron i)
        # This is identical to out_rows.norm(dim=-1) used in precompute_fixed_labels_fast.
        w_out_norms = W_out.norm(dim=0)  # [d_mlp]

        layer_inputs_cpu = cache_layer_inputs[layer_idx]  # [n_cache, n_pos, d_model], CPU bfloat16

        # ── Phase 1: Attribution-based neuron selection ───────────────────────
        # For each unique training (a1, a2), retrieve its cached MLP input and
        # compute source_vector_norm = |silu(W_gate x) * (W_up x)| * ‖W_out_row‖.
        # Select top-(n_pos × d_mlp × prop) neurons (flat, same as training).
        # Track: union_neurons, best_score[nid], best_target[nid].

        best_score  = torch.full((d_mlp,), -1.0, dtype=torch.float32, device=device)
        best_target: Dict[int, Tuple[int, ...]] = {}
        union_neurons: Set[int] = set()

        n_attrs_processed = 0
        for target_args in train_target_args:
            cache_indices = target_to_cache_idx.get(target_args, [])
            if not cache_indices:
                continue

            # Use the first matching cached prompt for this (a1, a2) pair.
            # All cached prompts with the same args have the same question structure
            # and nearly identical MLP activations.
            cache_idx = cache_indices[0]

            # x: [n_pos, d_model] — residual-stream input to this MLP layer
            x = layer_inputs_cpu[cache_idx].to(device=device, dtype=torch.float32)  # [n_pos, d_model]

            # Compute LLaMA gated MLP activations: silu(W_gate x) * (W_up x)
            acts = F.silu(x @ W_gate.T) * (x @ W_up.T)  # [n_pos, d_mlp]

            # Skip BOS token (position 0) — mirrors the ANOVA grid accumulation logic
            if acts.shape[0] > 1:
                acts = acts[1:]  # [n_pos-1, d_mlp]

            # source_vector_norm = |act[pos, i]| * ‖W_out[:, i]‖
            src_norms = acts.abs() * w_out_norms  # [n_pos-1, d_mlp]

            # Top-k flat across (positions × neurons) — identical to training's
            # flat_source_norms selection in replacement_model.setup_attribution
            flat  = src_norms.reshape(-1)                               # [n_pos-1 * d_mlp]
            k     = max(1, int(flat.numel() * prop_neurons_per_layer))
            topk_vals, topk_flat_idx = torch.topk(flat, min(k, flat.numel()))

            # Decode flat index → (position, neuron_id)
            topk_neuron_ids = (topk_flat_idx % d_mlp)  # [k]

            # Max source_norm per neuron across positions (for best-prompt tracking)
            max_per_neuron = src_norms.max(dim=0).values  # [d_mlp]

            # Update best_score / best_target for each neuron that entered the top-k
            selected = topk_neuron_ids.unique()
            for nid_t in selected:
                nid   = int(nid_t.item())
                score = float(max_per_neuron[nid].item())
                if score > float(best_score[nid].item()):
                    best_score[nid]  = score
                    best_target[nid] = target_args
                union_neurons.add(nid)

            n_attrs_processed += 1
            del x, acts, src_norms, flat, topk_vals, topk_flat_idx

        n_union = len(union_neurons)
        logger.info(
            "  [layer %d] Attribution union: %d neurons  "
            "(%d / %d training targets had cache entries)",
            layer_idx + 1, n_union, n_attrs_processed, len(train_target_args),
        )

        if n_union == 0:
            logger.warning("  [layer %d] No union neurons — skipping ANOVA.", layer_idx + 1)
            del W_gate, W_up, W_out, w_out_norms, layer_inputs_cpu, best_score
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        # ── Phase 2: Build ANOVA mean-activation grid [d_mlp, n_flat] ────────
        # Identical to the grid-building loop in precompute_fixed_labels_train.
        t_grid = time.time()
        layer_grid_flat   = torch.zeros(d_mlp, n_flat, dtype=torch.float32, device=device)
        n_active_positions = 0
        for pos_idx in range(n_cache_positions):
            if pos_idx == 0:
                continue  # skip BOS
            x = layer_inputs_cpu[valid_idx, pos_idx, :].to(device=device, dtype=torch.float32)
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
        del layer_grid_flat, W_gate, W_up, W_out, layer_inputs_cpu, best_score
        t_grid_done = time.time()
        logger.info(
            "  [layer %d] ANOVA grid built in %.1fs", layer_idx + 1, t_grid_done - t_grid
        )

        # ── Phase 3: ANOVA score tables + label union neurons ────────────────
        n_labeled_this_layer = 0

        if use_fast:
            tables = _build_fast_anova_tables(grid_flat, cache_arg_values, device)
            del grid_flat
            if device.type == "cuda":
                torch.cuda.empty_cache()

            n_labeled_this_layer = _label_union_neurons_from_tables(
                tables=tables,
                union_neurons=union_neurons,
                best_target=best_target,
                layer_idx=layer_idx,
                arg_to_idx=arg_to_idx,
                fixed_labels=fixed_labels,
                sum_min_specificity=sum_min_specificity,
            )
        else:
            # Non-fast path (radius > 0 or non-2-D): not implemented.
            # Fall back to skipping ANOVA for this layer.
            logger.warning(
                "  [layer %d] Non-fast ANOVA path not implemented — "
                "no labels assigned for this layer.",
                layer_idx + 1,
            )
            del grid_flat

        if device.type == "cuda":
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        logger.info(
            "  Layer %d done in %.1fs — %d new labels this layer  "
            "(union=%d neurons, %d total labels so far)",
            layer_idx + 1, elapsed, n_labeled_this_layer, n_union, len(fixed_labels),
        )

    return fixed_labels


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

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
             "used to extract the (a1, a2) target_args for each training prompt.",
    )
    parser.add_argument(
        "--mlp-input-cache", required=True,
        help="Path to the pre-computed MLP input cache root directory "
             "(created by precompute_mlp_inputs.py for the full 10k dataset). "
             "The cache is auto-detected by scanning model-slug subdirectories.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSON path for the fixed label mapping.",
    )
    parser.add_argument(
        "--prop-neurons-per-layer", type=float, default=1e-3,
        help="Fraction of (n_pos × d_mlp) flat entries to keep per prompt per layer "
             "(default: 1e-3).  MUST match the prop_neurons_per_layer used during "
             "distillation training to guarantee alignment.",
    )
    parser.add_argument(
        "--anova-nodes-per-label", type=int, default=3,
        help="Kept for CLI compatibility with other precompute scripts.  "
             "In this attribution-aligned script, attribution controls selection; "
             "each union neuron is assigned its single best ANOVA label.",
    )
    parser.add_argument(
        "--anova-range-radius", type=int, default=0,
        help="Radius around target arg for range-label rules (0 = exact match, "
             "only the fast vectorized path is supported).",
    )
    parser.add_argument(
        "--sum-min-specificity", type=float, default=1e-2,
        help="Minimum ANOVA specificity a neuron must have to receive a label (default: 1e-2).",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"],
        help="Model dtype for loading (activations are computed in float32).",
    )
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype  = dtype_map[args.dtype]
    device = torch.device(args.device)

    # ── Deferred imports ───────────────────────────────────────────────────────
    from utils.hf_models import load_student_model_for_distillation
    from graph_loss.hf_adapter import HFLlamaGraphAdapter
    from graph_loss.anova_node_labels import parse_numeric_args
    from utils.dataset_json import load_prompt_answer_json

    # ── Load student model (weights only — no forward passes needed) ──────────
    logger.info(
        "Loading student model: %s  (device=%s  dtype=%s)", args.model, device, dtype
    )
    student_model, tokenizer = load_student_model_for_distillation(
        student_source=None,
        student_model_id=args.model,
        device=device,
    )
    student_model = student_model.to(dtype=dtype)
    adapter    = HFLlamaGraphAdapter(student_model, tokenizer, device)
    model_name = getattr(adapter.model.config, "_name_or_path", args.model)
    logger.info(
        "Model: %d layers, d_mlp=%d, d_model=%d",
        adapter.n_layers, adapter.d_mlp, adapter.d_model,
    )

    # ── Load training dataset → unique (a1, a2) pairs ─────────────────────────
    if not os.path.isfile(args.train_dataset):
        raise FileNotFoundError(f"Training dataset not found: {args.train_dataset}")

    train_samples  = load_prompt_answer_json(args.train_dataset)
    train_targets: List[Tuple[int, ...]] = []
    for prompt in train_samples.keys():
        try:
            train_targets.append(parse_numeric_args(prompt))
        except ValueError:
            pass

    if not train_targets:
        raise ValueError(f"No parseable prompts in training dataset: {args.train_dataset}")

    unique_targets: List[Tuple[int, ...]] = list(dict.fromkeys(train_targets))
    logger.info(
        "Training dataset: %d prompts → %d unique (a1, a2) pairs",
        len(train_samples), len(unique_targets),
    )

    # ── Auto-detect and load MLP input cache ──────────────────────────────────
    logger.info(
        "Auto-detecting MLP input cache under %r for model %r …",
        args.mlp_input_cache, model_name,
    )
    mlp_input_cache = _find_mlp_cache(args.mlp_input_cache, model_name)

    cache_meta       = mlp_input_cache["meta"]
    cache_arg_values = cache_meta["arg_values"]  # [[a1_vals], [a2_vals]]
    n_cache_prompts  = int(cache_meta["n_prompts"])
    n_cache_positions = int(cache_meta["n_positions"])
    logger.info(
        "MLP input cache: %d prompts, %d positions, arg grid %s (total cells: %d)",
        n_cache_prompts, n_cache_positions,
        [len(v) for v in cache_arg_values],
        int(__import__("math").prod(len(v) for v in cache_arg_values)),
    )

    # Expected output size estimate (informational)
    k_per_prompt = max(1, int(n_cache_positions * adapter.d_mlp * args.prop_neurons_per_layer))
    logger.info(
        "Estimated top-k per prompt per layer: %d  "
        "(prop=%.2e × n_pos=%d × d_mlp=%d)",
        k_per_prompt, args.prop_neurons_per_layer, n_cache_positions, adapter.d_mlp,
    )
    logger.info(
        "Max union size per layer (before dedup): %d  "
        "Expected labeled neurons (rough): %d",
        k_per_prompt * len(unique_targets),
        k_per_prompt * len(unique_targets) * adapter.n_layers // 4,  # ~75% dedup estimate
    )

    # ── Run attribution-aligned labeling layer by layer ───────────────────────
    logger.info(
        "Starting attribution-aligned label precompute: "
        "%d layers × %d unique training pairs × prop_neurons=%.2e …",
        adapter.n_layers, len(unique_targets), args.prop_neurons_per_layer,
    )
    fixed_labels = _run_attr_union_and_label(
        adapter               = adapter,
        mlp_input_cache       = mlp_input_cache,
        cache_arg_values      = cache_arg_values,
        train_target_args     = unique_targets,
        device                = device,
        prop_neurons_per_layer= args.prop_neurons_per_layer,
        anova_range_radius    = args.anova_range_radius,
        anova_nodes_per_label = args.anova_nodes_per_label,
        sum_min_specificity   = args.sum_min_specificity,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(
        "Done. %d neurons labeled  (from %d layers × d_mlp=%d, "
        "%d unique training pairs, prop=%.2e)",
        len(fixed_labels), adapter.n_layers, adapter.d_mlp,
        len(unique_targets), args.prop_neurons_per_layer,
    )
    label_counts: Counter = Counter(fixed_labels.values())
    logger.info("Label breakdown (top 30 by count):")
    for lbl, cnt in sorted(label_counts.most_common(30)):
        logger.info("  %-50s : %d", lbl, cnt)
    logger.info("  … (%d distinct labels total)", len(label_counts))

    # Category-level breakdown
    cat_counts: Counter = Counter()
    for lbl in fixed_labels.values():
        if "units and" in lbl:
            cat_counts["arg1 units and arg2 units"] += 1
        elif "range and" in lbl:
            cat_counts["arg1 range and arg2 range"] += 1
        elif lbl.startswith("arg1 units"):
            cat_counts["arg1 units"] += 1
        elif lbl.startswith("arg2 units"):
            cat_counts["arg2 units"] += 1
        elif lbl.startswith("arg1"):
            cat_counts["arg1 range"] += 1
        elif lbl.startswith("arg2"):
            cat_counts["arg2 range"] += 1
        elif lbl.startswith("sum units"):
            cat_counts["sum units"] += 1
        elif lbl.startswith("sum"):
            cat_counts["sum range"] += 1
        elif lbl == "carry":
            cat_counts["carry"] += 1
        else:
            cat_counts["other"] += 1
    logger.info("Category breakdown:")
    for cat, cnt in sorted(cat_counts.most_common()):
        logger.info("  %-40s : %d", cat, cnt)

    # ── Save ──────────────────────────────────────────────────────────────────
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(fixed_labels, f, indent=2)
    logger.info("Saved → %s", args.output)


if __name__ == "__main__":
    main()

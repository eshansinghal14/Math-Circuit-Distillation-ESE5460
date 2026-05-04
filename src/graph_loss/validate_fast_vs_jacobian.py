"""Validate fast (DLA) vs full (Jacobian) attribution on held-out prompts.

Step 3 of the distillation pipeline: compare
  (a) fast=True  — single forward pass, DLA readout only, no neuron→neuron edges
  (b) fast=False — full Jacobian attribution, real neuron→neuron edges

Metrics reported per prompt:
  - Number of supernodes in each mode
  - Supernode membership overlap (Jaccard similarity between best-matched pairs)
  - DLA cosine similarity between best-matched supernodes
  - Supernode adjacency edge density comparison

Usage (Colab):
    import sys; sys.path.insert(0, '/content/Math-Circuit-Distillation-ESE5460/src')
    from graph_loss.validate_fast_vs_jacobian import run_validation
    run_validation(teacher_model, prompts=['13+27=', '45+67=', '22+88='])
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from graph_loss.align import compute_supernode_dla
from graph_loss.attribution.attribute import attribute
from graph_loss.graph import SuperGraph, build_super_graph, extract_supernode_members


logger = logging.getLogger(__name__)


def _jaccard(a: set, b: set) -> float:
    union = len(a | b)
    return len(a & b) / union if union > 0 else 0.0


def _compare_prompt(
    prompt: str,
    model: Any,
    *,
    top_k_logits: int = 1000,
    prop_neurons_per_layer: float = 1e-3,
    batch_size: int = 4,
) -> dict:
    """Run both attribution modes on one prompt and return comparison metrics."""
    W_U = model.unembed.W_U
    n_vocab = model.cfg.d_vocab

    # ------------------------------------------------------------------ fast
    logger.info("  [fast]  building graph …")
    graph_fast = attribute(
        prompt, model,
        top_k_logits=top_k_logits,
        prop_neurons_per_layer=prop_neurons_per_layer,
        batch_size=batch_size,
        fast=True,
    )
    with torch.no_grad():
        sg_fast: SuperGraph = build_super_graph(
            graph_fast, model, cluster_method="ablation"
        )
    members_fast = extract_supernode_members(sg_fast, graph_fast, model)
    dlas_fast: dict[int, torch.Tensor] = {
        m["cluster_id"]: F.normalize(compute_supernode_dla(m, W_U)[:n_vocab], dim=0)
        for m in members_fast
    }
    neuron_sets_fast = [set(sn) for sn in sg_fast.supernodes]

    # ------------------------------------------------------------------ full
    logger.info("  [full]  building graph (Jacobian) …")
    graph_full = attribute(
        prompt, model,
        top_k_logits=top_k_logits,
        prop_neurons_per_layer=prop_neurons_per_layer,
        batch_size=batch_size,
        fast=False,
    )
    with torch.no_grad():
        sg_full: SuperGraph = build_super_graph(
            graph_full, model, cluster_method="ablation"
        )
    members_full = extract_supernode_members(sg_full, graph_full, model)
    dlas_full: dict[int, torch.Tensor] = {
        m["cluster_id"]: F.normalize(compute_supernode_dla(m, W_U)[:n_vocab], dim=0)
        for m in members_full
    }
    neuron_sets_full = [set(sn) for sn in sg_full.supernodes]

    n_fast = len(neuron_sets_fast)
    n_full = len(neuron_sets_full)

    # ------------------------------------------------------------------ membership overlap
    # For each full supernode, find best-matching fast supernode by Jaccard
    best_jaccard: list[float] = []
    best_dla_cos: list[float] = []

    for i, full_ns in enumerate(neuron_sets_full):
        full_dla = dlas_full.get(i)
        best_j, best_jac = -1, 0.0
        for j, fast_ns in enumerate(neuron_sets_fast):
            jac = _jaccard(full_ns, fast_ns)
            if jac > best_jac:
                best_jac, best_j = jac, j
        best_jaccard.append(best_jac)

        if best_j >= 0 and full_dla is not None and best_jac > 0:
            fast_dla = dlas_fast.get(best_j)
            if fast_dla is not None:
                cos = float(F.cosine_similarity(full_dla.unsqueeze(0), fast_dla.unsqueeze(0)))
                best_dla_cos.append(cos)

    # ------------------------------------------------------------------ edge structure
    # Fast mode: neuron→neuron edges are all zero by design
    # Full mode: real Jacobian-derived neuron→neuron edges
    adj_fast = sg_fast.supernode_adjacency_matrix
    adj_full = sg_full.supernode_adjacency_matrix
    nnz_fast = float((adj_fast.abs() > 1e-6).float().mean())
    nnz_full = float((adj_full.abs() > 1e-6).float().mean())

    # Neuron→neuron edge density in the raw graph
    n2n_fast = float(graph_fast.adjacency_matrix[
        :graph_fast.n_neurons, :graph_fast.n_neurons
    ].abs().mean())
    n2n_full = float(graph_full.adjacency_matrix[
        :graph_full.n_neurons, :graph_full.n_neurons
    ].abs().mean())

    return {
        "prompt": prompt,
        "n_supernodes_fast": n_fast,
        "n_supernodes_full": n_full,
        "mean_jaccard_overlap": sum(best_jaccard) / len(best_jaccard) if best_jaccard else 0.0,
        "mean_dla_cosine_sim": sum(best_dla_cos) / len(best_dla_cos) if best_dla_cos else 0.0,
        "pct_full_sn_with_any_overlap": sum(j > 0 for j in best_jaccard) / len(best_jaccard) if best_jaccard else 0.0,
        "supernode_adj_density_fast": nnz_fast,
        "supernode_adj_density_full": nnz_full,
        "neuron2neuron_edge_mean_fast": n2n_fast,
        "neuron2neuron_edge_mean_full": n2n_full,
    }


def run_validation(
    model: Any,
    prompts: list[str],
    *,
    top_k_logits: int = 1000,
    prop_neurons_per_layer: float = 1e-3,
    batch_size: int = 4,
    verbose: bool = True,
) -> list[dict]:
    """Run fast vs Jacobian comparison on each prompt and print a summary table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = []
    for prompt in prompts:
        logger.info("Validating prompt: %r", prompt)
        row = _compare_prompt(
            prompt, model,
            top_k_logits=top_k_logits,
            prop_neurons_per_layer=prop_neurons_per_layer,
            batch_size=batch_size,
        )
        results.append(row)
        if verbose:
            print(f"\nPrompt: {prompt}")
            print(f"  Supernodes  — fast: {row['n_supernodes_fast']:3d}  full: {row['n_supernodes_full']:3d}")
            print(f"  Supernode membership overlap (mean Jaccard): {row['mean_jaccard_overlap']:.3f}")
            print(f"  DLA cosine similarity (matched pairs):       {row['mean_dla_cosine_sim']:.3f}")
            print(f"  % full supernodes matched (Jaccard > 0):     {row['pct_full_sn_with_any_overlap']:.1%}")
            print(f"  Supernode adj density  — fast: {row['supernode_adj_density_fast']:.4f}  full: {row['supernode_adj_density_full']:.4f}")
            print(f"  Neuron→neuron edge mean — fast: {row['neuron2neuron_edge_mean_fast']:.4f}  full: {row['neuron2neuron_edge_mean_full']:.4f}")

    if verbose and results:
        print("\n" + "="*60)
        print("Summary across prompts")
        print("="*60)
        for key in [
            "mean_jaccard_overlap",
            "mean_dla_cosine_sim",
            "pct_full_sn_with_any_overlap",
            "neuron2neuron_edge_mean_full",
        ]:
            vals = [r[key] for r in results]
            print(f"  {key:45s}: {sum(vals)/len(vals):.3f}")

    return results

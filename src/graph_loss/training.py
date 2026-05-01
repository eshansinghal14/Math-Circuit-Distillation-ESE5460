"""Training-time graph auxiliary loss helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from graph_loss.align import align_supernodes
from graph_loss.attribution.attribute import attribute
from graph_loss.graph import (
    SuperGraph,
    build_super_graph,
    extract_supernode_members,
    normalize_matrix,
    prune_graph,
)
from graph_loss.hf_adapter import (
    HFLlamaGraphAdapter,
    extract_hf_supernode_members,
)
from graph_loss.loss import compute_graph_loss


@dataclass
class GraphAuxConfig:
    lambda_graph: float = 0.1
    graph_dtype: torch.dtype | None = None
    top_k_logits: int | None = 20
    prop_neurons_per_layer: float = 0.1
    teacher_graph_batch_size: int = 512
    student_graph_batch_size: int = 1
    verbose: bool = False
    graph_prune: bool = False
    graph_node_threshold: float = 0.8
    graph_edge_threshold: float = 0.98
    graph_similarity_threshold: float = 0.7
    graph_max_fan_out: int = 4
    graph_node_weight: float = 0.1  # weight of DLA node loss within L_graph


def _aggregate_supergraph_adjacency(graph, supernodes: list[list[int]]) -> SuperGraph:
    """Aggregate a differentiable graph adjacency using fixed supernode membership."""
    adj_matrix_norm = normalize_matrix(graph.adjacency_matrix)
    num_supernodes = len(supernodes)
    supernode_adj_matrix = torch.zeros(
        num_supernodes,
        num_supernodes,
        dtype=adj_matrix_norm.dtype,
        device=adj_matrix_norm.device,
    )
    for t in range(num_supernodes):
        total_input = torch.abs(adj_matrix_norm[:, supernodes[t]]).sum(dim=0)
        internal_input = torch.abs(adj_matrix_norm[supernodes[t]][:, supernodes[t]]).sum(dim=0)
        frac_external = (total_input - internal_input) / total_input.clamp(min=1e-10)
        for s in range(num_supernodes):
            sum_A = adj_matrix_norm[supernodes[t]][:, supernodes[s]].sum(dim=1)
            supernode_adj_matrix[t, s] = (
                (frac_external * sum_A).sum(dim=0)
                / frac_external.sum(dim=0).clamp(min=1e-10)
            )
    return SuperGraph(
        supernode_adjacency_matrix=supernode_adj_matrix,
        supernodes=supernodes,
    )


def compute_prompt_graph_loss(
    *,
    prompt: str,
    teacher_graph_model: Any,
    student_adapter: HFLlamaGraphAdapter,
    config: GraphAuxConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if config.verbose:
        print(f"  [graph] building teacher graph for prompt: {prompt!r}")
    teacher_graph = attribute(
        prompt=prompt,
        model=teacher_graph_model,
        top_k_logits=config.top_k_logits,
        prop_neurons_per_layer=config.prop_neurons_per_layer,
        batch_size=config.teacher_graph_batch_size,
        verbose=config.verbose,
    )
    if config.verbose:
        print(f"  [graph] building student graph for prompt: {prompt!r}")
    student_graph = student_adapter.build_graph(
        prompt,
        top_k_logits=config.top_k_logits,
        prop_neurons_per_layer=config.prop_neurons_per_layer,
        batch_size=config.student_graph_batch_size,
        dtype=config.graph_dtype,
        verbose=config.verbose,
        # Keep graph loss trainable through student activations/write vectors
        # without retaining a full second-order graph for every attribution edge.
        create_graph=False,
        detach_result=False,
    )

    teacher_prune_result = None
    student_prune_result = None
    if config.graph_prune:
        if config.verbose:
            print("  [graph] pruning teacher/student graphs")
        teacher_prune_result = prune_graph(
            teacher_graph,
            node_threshold=config.graph_node_threshold,
            edge_threshold=config.graph_edge_threshold,
        )
        student_prune_result = prune_graph(
            student_graph,
            node_threshold=config.graph_node_threshold,
            edge_threshold=config.graph_edge_threshold,
        )
        teacher_graph = teacher_graph.apply_prune_result(teacher_prune_result)
        student_graph = student_graph.apply_prune_result(student_prune_result)

    if config.verbose:
        print("  [graph] building teacher supergraph")
    supergraph_start = time.perf_counter()
    teacher_supergraph = build_super_graph(
        teacher_graph,
        teacher_graph_model,
        prune_result=teacher_prune_result,
    )
    if config.verbose:
        print(
            "  [graph] teacher supergraph complete: "
            f"{len(teacher_supergraph.supernodes)} supernodes in "
            f"{time.perf_counter() - supergraph_start:.2f}s",
        )
        print(
            "  [graph] building student supergraph",
        )
    supergraph_start = time.perf_counter()
    with torch.no_grad():
        student_supergraph_structure = build_super_graph(
            student_graph,
            student_adapter,
            prune_result=student_prune_result,
        )
    student_supergraph = _aggregate_supergraph_adjacency(
        student_graph,
        student_supergraph_structure.supernodes,
    )
    if config.verbose:
        print(
            "  [graph] student supergraph complete: "
            f"{len(student_supergraph.supernodes)} supernodes in "
            f"{time.perf_counter() - supergraph_start:.2f}s",
        )

    if config.verbose:
        print("  [graph] aligning supernodes and computing graph loss")
    teacher_members = extract_supernode_members(
        teacher_supergraph,
        teacher_graph,
        teacher_graph_model,
    )
    student_members = extract_hf_supernode_members(
        student_supergraph,
        student_graph,
        student_adapter,
        detach=False,
    )
    alignment = align_supernodes(
        teacher_members,
        student_members,
        teacher_graph_model.unembed.W_U.detach().to(
            device=student_graph.adjacency_device,
            dtype=config.graph_dtype or student_graph.adjacency_matrix.dtype,
        ),
        student_adapter.W_U.to(
            device=student_graph.adjacency_device,
            dtype=config.graph_dtype or student_graph.adjacency_matrix.dtype,
        ),
        similarity_threshold=config.graph_similarity_threshold,
        max_fan_out=config.graph_max_fan_out,
    )

    teacher_ids = list(range(len(teacher_supergraph.supernodes)))
    student_ids = list(range(len(student_supergraph.supernodes)))
    graph_loss, loss_breakdown = compute_graph_loss(
        teacher_supergraph.supernode_adjacency_matrix.detach().to(
            device=student_supergraph.supernode_adjacency_matrix.device,
            dtype=student_supergraph.supernode_adjacency_matrix.dtype,
        ),
        student_supergraph.supernode_adjacency_matrix,
        alignment.mapping,
        teacher_ids,
        student_ids,
        teacher_dla=alignment.teacher_dla,
        student_dla=alignment.student_dla,
        node_weight=config.graph_node_weight,
    )

    metrics = {
        "teacher_supernodes": len(teacher_ids),
        "student_supernodes": len(student_ids),
        "teacher_graph_neurons": int(teacher_graph.n_neurons),
        "student_graph_neurons": int(student_graph.n_neurons),
        "aligned_teacher_supernodes": sum(
            1 for teacher_id in teacher_ids if alignment.mapping.get(teacher_id)
        ),
        "mean_alignment_similarity": (
            sum(alignment.best_sim.values()) / len(alignment.best_sim)
            if alignment.best_sim
            else 0.0
        ),
        **loss_breakdown,
    }
    return graph_loss, metrics


def compute_batch_graph_loss(
    *,
    prompts: list[str],
    teacher_graph_model: Any,
    student_adapter: HFLlamaGraphAdapter,
    config: GraphAuxConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    losses = []
    metric_sums: dict[str, float] = {}
    for prompt in prompts:
        prompt_loss, prompt_metrics = compute_prompt_graph_loss(
            prompt=prompt,
            teacher_graph_model=teacher_graph_model,
            student_adapter=student_adapter,
            config=config,
        )
        losses.append(prompt_loss)
        for key, value in prompt_metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)

    if not losses:
        return torch.tensor(0.0, device=device), {}

    loss = torch.stack(losses).mean()
    denom = float(len(losses))
    metrics = {key: value / denom for key, value in metric_sums.items()}
    metrics["graph_prompts"] = float(len(losses))
    return loss, metrics


def backward_batch_graph_loss(
    *,
    prompts: list[str],
    teacher_graph_model: Any,
    student_adapter: HFLlamaGraphAdapter,
    config: GraphAuxConfig,
    device: torch.device,
    loss_scale: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute and backprop graph loss one prompt at a time.

    This keeps peak graph-loss memory bounded by a single problem instead of
    retaining every prompt's attribution graph until the batch backward call.
    """
    if not prompts:
        return torch.tensor(0.0, device=device), {}

    metric_sums: dict[str, float] = {}
    detached_losses = []
    denom = float(len(prompts))
    graph_backward_prompts = 0
    for prompt in prompts:
        prompt_loss, prompt_metrics = compute_prompt_graph_loss(
            prompt=prompt,
            teacher_graph_model=teacher_graph_model,
            student_adapter=student_adapter,
            config=config,
        )
        detached_losses.append(prompt_loss.detach())
        scaled_prompt_loss = (loss_scale / denom) * prompt_loss
        if scaled_prompt_loss.requires_grad:
            scaled_prompt_loss.backward()
            graph_backward_prompts += 1
        for key, value in prompt_metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
        del prompt_loss, scaled_prompt_loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    loss = torch.stack(detached_losses).mean()
    metrics = {key: value / denom for key, value in metric_sums.items()}
    metrics["graph_prompts"] = float(len(prompts))
    metrics["graph_backward_prompts"] = float(graph_backward_prompts)
    return loss, metrics

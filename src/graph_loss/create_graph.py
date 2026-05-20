"""Unified graph creation pipeline: build_graph → prune → build_super_graph."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from graph_loss.graph import (
    Graph,
    PruneResult,
    SuperGraph,
    build_super_graph,
    prune_graph,
)
from graph_loss.hf_adapter import HFLlamaGraphAdapter


@dataclass
class GraphPipelineResult:
    raw_graph: Graph                  # original built graph (before pruning)
    graph: Graph                      # graph used for supergraph (pruned if prune=True)
    prune_result: PruneResult | None  # None if prune=False
    supergraph: SuperGraph


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _count_nonzero_edges(matrix: torch.Tensor) -> int:
    return int(matrix.count_nonzero().item())


def _density(nonzero_edges: int, n_rows: int, n_cols: int) -> float:
    if n_rows == 0 or n_cols == 0:
        return 0.0
    return nonzero_edges / float(n_rows * n_cols)


def _format_top_logit_targets(graph: Graph, limit: int = 5) -> str:
    top_targets = sorted(
        zip(graph.logit_targets, graph.logit_probabilities.tolist(), strict=False),
        key=lambda item: item[1],
        reverse=True,
    )[:limit]
    if not top_targets:
        return "none"
    return ", ".join(
        f"{target.token_str!r}:{prob:.4g}" for target, prob in top_targets
    )


def _log_graph_summary(graph: Graph, *, logger: logging.Logger, stage: str) -> None:
    total_nodes = graph.n_nodes
    total_edges = int(graph.adjacency_matrix.count_nonzero().item())
    neuron_edges = int(
        graph.adjacency_matrix[: graph.n_neurons, : graph.n_neurons].count_nonzero().item()
    )

    logger.info("%s graph summary", stage)
    logger.info(
        "  nodes: neurons=%d tokens=%d logits=%d total=%d",
        graph.n_neurons,
        graph.n_tokens,
        graph.n_logits,
        total_nodes,
    )
    logger.info(
        "  adjacency: shape=%s dtype=%s device=%s",
        graph.adjacency_shape,
        graph.adjacency_matrix.dtype,
        graph.adjacency_device,
    )
    logger.info(
        "  edges: total=%d density=%.6f neuron_to_neuron=%d neuron_density=%.6f",
        total_edges,
        _density(total_edges, total_nodes, total_nodes),
        neuron_edges,
        _density(neuron_edges, graph.n_neurons, graph.n_neurons),
    )
    logger.info(
        "  edge_weight_mass: abs_total=%.6f abs_neuron_to_neuron=%.6f",
        float(graph.adjacency_matrix.abs().sum().item()),
        float(graph.adjacency_matrix[: graph.n_neurons, : graph.n_neurons].abs().sum().item()),
    )
    logger.info(
        "  logits: vocab_size=%d top_targets=%s",
        graph.vocab_size,
        _format_top_logit_targets(graph),
    )


def _log_prune_summary(
    graph: Graph,
    prune_result: PruneResult,
    *,
    node_threshold: float,
    edge_threshold: float,
    logger: logging.Logger,
) -> None:
    node_mask = prune_result.node_mask
    edge_mask = prune_result.edge_mask
    adjacency_matrix = graph.adjacency_matrix
    adjacency_nonzero = adjacency_matrix != 0
    effective_edge_mask = edge_mask & adjacency_nonzero & node_mask[:, None] & node_mask[None, :]

    kept_neurons = int(node_mask[: graph.n_neurons].sum().item())
    kept_tokens = int(node_mask[graph.n_neurons : graph.n_neurons + graph.n_tokens].sum().item())
    kept_logits = int(node_mask[-graph.n_logits :].sum().item()) if graph.n_logits else 0
    kept_total = int(node_mask.sum().item())
    kept_edges = int(effective_edge_mask.sum().item())
    kept_neuron_edges = int(
        effective_edge_mask[: graph.n_neurons, : graph.n_neurons].sum().item()
    )
    total_edges = _count_nonzero_edges(graph.adjacency_matrix)

    logger.info("Pruned graph summary")
    logger.info(
        "  thresholds: node_threshold=%.4f edge_threshold=%.4f",
        node_threshold,
        edge_threshold,
    )
    logger.info(
        "  kept_nodes: neurons=%d/%d tokens=%d/%d logits=%d/%d total=%d/%d",
        kept_neurons,
        graph.n_neurons,
        kept_tokens,
        graph.n_tokens,
        kept_logits,
        graph.n_logits,
        kept_total,
        graph.n_nodes,
    )
    logger.info(
        "  kept_edges: total=%d/%d density=%.6f neuron_to_neuron=%d density_neuron=%.6f",
        kept_edges,
        total_edges,
        _density(kept_edges, kept_total, kept_total),
        kept_neuron_edges,
        _density(kept_neuron_edges, kept_neurons, kept_neurons),
    )
    logger.info(
        "  reductions: neurons_removed=%d edges_removed=%d node_retention=%.2f%% edge_retention=%.2f%%",
        graph.n_neurons - kept_neurons,
        total_edges - kept_edges,
        100.0 * kept_total / max(graph.n_nodes, 1),
        100.0 * kept_edges / max(total_edges, 1),
    )
    logger.info(
        "  cumulative_score_stats: min=%.6f median=%.6f max=%.6f",
        float(prune_result.cumulative_scores.min().item()),
        float(prune_result.cumulative_scores.median().item()),
        float(prune_result.cumulative_scores.max().item()),
    )


def _log_supergraph_summary(
    graph: Graph,
    supergraph: SuperGraph,
    *,
    logger: logging.Logger,
) -> None:
    cluster_sizes = [len(cluster) for cluster in supergraph.supernodes]
    supernode_count = len(cluster_sizes)
    covered_neurons = sum(cluster_sizes)
    omitted_neurons = graph.n_neurons - covered_neurons
    super_edges = _count_nonzero_edges(supergraph.supernode_adjacency_matrix)

    logger.info("Supergraph summary")
    logger.info(
        "  structure: supernodes=%d super_adjacency_shape=%s edges=%d density=%.6f",
        supernode_count,
        tuple(supergraph.supernode_adjacency_matrix.shape),
        super_edges,
        _density(super_edges, supernode_count, supernode_count),
    )
    if cluster_sizes:
        cluster_sizes_tensor = torch.tensor(cluster_sizes, dtype=torch.float32)
        top_sizes = sorted(cluster_sizes, reverse=True)[:5]
        logger.info(
            "  clusters: covered_neurons=%d/%d omitted_neurons=%d singletons=%d min=%d max=%d mean=%.2f median=%.2f top5=%s",
            covered_neurons,
            graph.n_neurons,
            omitted_neurons,
            sum(size == 1 for size in cluster_sizes),
            min(cluster_sizes),
            max(cluster_sizes),
            float(cluster_sizes_tensor.mean().item()),
            float(cluster_sizes_tensor.median().item()),
            top_sizes,
        )
    else:
        logger.info(
            "  clusters: covered_neurons=0/%d omitted_neurons=%d no supernodes were formed",
            graph.n_neurons,
            omitted_neurons,
        )
    logger.info(
        "  reduction_vs_neurons: original_neurons=%d reduced_to=%d reduction=%.2f%%",
        graph.n_neurons,
        supernode_count,
        100.0 * (graph.n_neurons - supernode_count) / max(graph.n_neurons, 1),
    )


def _log_pipeline_comparison(
    graph: Graph,
    supergraph: SuperGraph,
    *,
    logger: logging.Logger,
    prune_result: PruneResult | None = None,
) -> None:
    total_edges = int(graph.adjacency_matrix.count_nonzero().item())
    super_edges = _count_nonzero_edges(supergraph.supernode_adjacency_matrix)

    logger.info("Pipeline comparison")
    if prune_result is not None:
        kept_edges = int(
            (
                prune_result.edge_mask
                & (graph.adjacency_matrix != 0)
                & prune_result.node_mask[:, None]
                & prune_result.node_mask[None, :]
            ).sum().item()
        )
        kept_neurons = int(prune_result.node_mask[: graph.n_neurons].sum().item())
        logger.info(
            "  build_to_prune: neurons %d -> %d, edges %d -> %d",
            graph.n_neurons,
            kept_neurons,
            total_edges,
            kept_edges,
        )
    logger.info(
        "  build_to_supergraph: neurons %d -> supernodes %d, edges %d -> %d",
        graph.n_neurons,
        len(supergraph.supernodes),
        total_edges,
        super_edges,
    )


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_prune_result(path: str, prune_result: PruneResult) -> None:
    torch.save(
        {
            "node_mask": prune_result.node_mask,
            "edge_mask": prune_result.edge_mask,
            "cumulative_scores": prune_result.cumulative_scores,
        },
        path,
    )


def save_supergraph(
    path: str,
    supergraph: SuperGraph,
    logit_token_ids: torch.Tensor | None = None,
) -> None:
    data = {
        "supernode_adjacency_matrix": supergraph.supernode_adjacency_matrix,
        "supernodes": supergraph.supernodes,
        "node_labels": supergraph.node_labels,
        "supernode_labels": supergraph.supernode_labels,
        "supernode_heatmap_pdf_paths": supergraph.supernode_heatmap_pdf_paths,
    }
    if logit_token_ids is not None:
        data["logit_token_ids"] = logit_token_ids.cpu()
    torch.save(data, path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def create_graph(
    adapter: HFLlamaGraphAdapter,
    prompt: str | torch.Tensor | list[int],
    *,
    # build_graph params
    attribution_targets=None,
    top_k_logits: int | None = 20,
    prop_neurons_per_layer: float = 0.1,
    batch_size: int = 512,
    dtype: torch.dtype | None = None,
    verbose: bool = False,
    build_create_graph: bool = False,
    detach_result: bool | None = None,
    skip_logit_attribution: bool = False,
    # prune params
    prune: bool = False,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    # supergraph params
    dataset: str | None = None,
    activation_forward_batch_size: int = 32,
    mlp_input_cache: dict | None = None,
    model_name: str | None = None,
    supernode_heatmap_output_dir: str | None = None,
    anova_nodes_per_label: int = 10,
    anova_range_radius: int = 0,
    sum_min_specificity: float = 0.0,
    no_grad_supergraph: bool = False,
    logger: logging.Logger | None = None,
) -> GraphPipelineResult:
    """Run the full graph creation pipeline: build_graph → [prune] → build_super_graph.

    Args:
        adapter: Loaded HFLlamaGraphAdapter wrapping the model.
        prompt: Input prompt string, token tensor, or token ID list.
        attribution_targets: Optional attribution targets for build_graph.
        top_k_logits: Number of top logits to attribute to (None = all).
        prop_neurons_per_layer: Fraction of neurons to select per layer.
        batch_size: Attribution batch size.
        dtype: Optional dtype override for build_graph.
        verbose: Verbose logging during attribution.
        build_create_graph: PyTorch autograd create_graph flag (training use).
        detach_result: Whether to detach the adjacency from the grad graph.
        skip_logit_attribution: Skip logit attribution phase.
        prune: Whether to prune the graph before building the supergraph.
        node_threshold: Cumulative node influence fraction to retain.
        edge_threshold: Cumulative edge influence fraction to retain.
        dataset: Dataset name/path for activation-write supergraph labeling.
        activation_forward_batch_size: Batch size for dataset forward passes.
        mlp_input_cache: Pre-built MLP input cache. Built from dataset if None
            and both dataset and model_name are provided.
        model_name: HuggingFace model identifier string.
        supernode_heatmap_output_dir: Directory for per-supernode PDF heatmaps.
        anova_nodes_per_label: Max neurons per ANOVA label supernode.
        anova_range_radius: Radius for target-centered ANOVA range masks.
        sum_min_specificity: Min ANOVA specificity for sum-range/sum-units supernodes.
        no_grad_supergraph: Wrap build_super_graph in torch.no_grad() (training use).
        logger: Optional logger; creates a module-level one if not provided.

    Returns:
        GraphPipelineResult with raw_graph, graph, prune_result, and supergraph.
    """
    _logger = logger or logging.getLogger(__name__)

    _logger.info("Running attribution graph build")
    raw_graph = adapter.build_graph(
        prompt,
        attribution_targets=attribution_targets,
        top_k_logits=top_k_logits,
        prop_neurons_per_layer=prop_neurons_per_layer,
        batch_size=batch_size,
        dtype=dtype,
        verbose=verbose,
        create_graph=build_create_graph,
        detach_result=detach_result,
        skip_logit_attribution=skip_logit_attribution,
    )
    _log_graph_summary(raw_graph, logger=_logger, stage="Built")

    prune_result = None
    graph = raw_graph
    if prune:
        _logger.info("Running prune_graph")
        prune_result = prune_graph(
            graph,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
        )
        _log_prune_summary(
            graph,
            prune_result,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
            logger=_logger,
        )
        _logger.info("Applying prune masks to graph")
        graph = graph.apply_prune_result(prune_result)

    if mlp_input_cache is None and dataset is not None and model_name is not None:
        from graph_loss.neuron_activation_heatmap import _resolve_dataset_path
        from graph_loss.precompute_mlp_inputs import build_mlp_input_cache as _build_mlp_cache
        dataset_path = _resolve_dataset_path(dataset)
        _logger.info("Building MLP input cache for dataset: %s", dataset)
        mlp_input_cache = _build_mlp_cache(
            adapter,
            dataset_path,
            model_name,
            batch_size=activation_forward_batch_size,
        )
        n_prompts = int(mlp_input_cache.get("meta", {}).get("n_prompts", 0))
        _logger.info("  Built MLP cache: %d prompts", n_prompts)

    _logger.info("Running build_super_graph")
    if supernode_heatmap_output_dir:
        _logger.info("Supernode heatmap output directory: %s", supernode_heatmap_output_dir)

    def _run_build_super_graph() -> SuperGraph:
        return build_super_graph(
            graph,
            adapter,
            prune_result=prune_result,
            dataset=dataset,
            activation_forward_batch_size=activation_forward_batch_size,
            mlp_input_cache=mlp_input_cache,
            model_name=model_name,
            supernode_heatmap_output_dir=supernode_heatmap_output_dir,
            anova_nodes_per_label=anova_nodes_per_label,
            anova_range_radius=anova_range_radius,
            sum_min_specificity=sum_min_specificity,
        )

    if no_grad_supergraph:
        with torch.no_grad():
            supergraph = _run_build_super_graph()
    else:
        supergraph = _run_build_super_graph()

    _log_supergraph_summary(graph, supergraph, logger=_logger)
    _log_pipeline_comparison(graph, supergraph, logger=_logger, prune_result=prune_result)

    return GraphPipelineResult(
        raw_graph=raw_graph,
        graph=graph,
        prune_result=prune_result,
        supergraph=supergraph,
    )

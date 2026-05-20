"""Unified graph creation pipeline: ANOVA label → build_graph → build_super_graph."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from graph_loss.anova_node_labels import parse_numeric_args
from graph_loss.attribution.attribute import _attribute_from_context, setup_attribution
from graph_loss.graph import (
    Graph,
    SuperGraph,
    build_super_graph,
    select_anova_supernodes,
)
from graph_loss.hf_adapter import HFLlamaGraphAdapter
from graph_loss.neuron_activation_heatmap import (
    ActivationWriteResult,
    build_neuron_activation_write_result,
    label_neurons_layer_by_layer,
)


@dataclass
class GraphPipelineResult:
    graph: Graph
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
            "  clusters: covered_neurons=0/%d omitted_neurons=%d no supernodes formed",
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
) -> None:
    total_edges = int(graph.adjacency_matrix.count_nonzero().item())
    super_edges = _count_nonzero_edges(supergraph.supernode_adjacency_matrix)
    logger.info("Pipeline comparison")
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
    # attribution params
    attribution_targets=None,
    top_k_logits: int | None = 20,
    prop_neurons_per_layer: float = 0.1,
    batch_size: int = 512,
    dtype: torch.dtype | None = None,
    verbose: bool = False,
    build_create_graph: bool = False,
    detach_result: bool | None = None,
    skip_logit_attribution: bool = False,
    # ANOVA / supergraph params
    dataset: str | None = None,
    activation_forward_batch_size: int = 32,
    mlp_input_cache: dict | None = None,
    model_name: str | None = None,
    supernode_heatmap_output_dir: str | None = None,
    anova_nodes_per_label: int = 10,
    anova_range_radius: int = 0,
    sum_min_specificity: float = 0.0,
    labelling_layer_batch_size: int = 1,
    no_grad_supergraph: bool = False,
    logger: logging.Logger | None = None,
) -> GraphPipelineResult:
    """Run the ANOVA-first graph creation pipeline.

    New order: neuron pre-selection → ANOVA label all candidates →
    select top-K per label → build attribution graph for selected neurons →
    aggregate edges into supergraph.

    Args:
        adapter: Loaded HFLlamaGraphAdapter wrapping the model.
        prompt: Input prompt string, token tensor, or token ID list.
        attribution_targets: Optional attribution targets.
        top_k_logits: Number of top logits to attribute to (None = all).
        prop_neurons_per_layer: Fraction of neurons to pre-select per layer.
        batch_size: Attribution batch size.
        dtype: Optional dtype override.
        verbose: Verbose logging during attribution.
        build_create_graph: PyTorch autograd create_graph flag (training use).
        detach_result: Whether to detach the adjacency from the grad graph.
        skip_logit_attribution: Skip logit attribution phase.
        dataset: Dataset name/path for activation-grid ANOVA labeling.
        activation_forward_batch_size: Batch size for dataset forward passes.
        mlp_input_cache: Pre-built MLP input cache.
        model_name: HuggingFace model identifier string.
        supernode_heatmap_output_dir: Directory for per-supernode PDF heatmaps.
        anova_nodes_per_label: Max neurons per ANOVA label supernode.
        anova_range_radius: Radius for target-centered ANOVA range masks.
        sum_min_specificity: Min ANOVA specificity for sum-range/sum-units supernodes.
        no_grad_supergraph: Wrap build_super_graph in torch.no_grad() (training use).
        logger: Optional logger; creates a module-level one if not provided.

    Returns:
        GraphPipelineResult with the ANOVA-filtered attribution graph and supergraph.
    """
    _logger = logger or logging.getLogger(__name__)

    # Step 1: Tokenize and run the initial forward pass to pre-select neuron candidates
    # by gradient-norm (prop_neurons_per_layer fraction per layer).
    input_ids = adapter.ensure_tokenized(prompt)
    _logger.info("Running setup_attribution (neuron pre-selection by gradient norm)")
    ctx = setup_attribution(adapter, input_ids, prop_neurons_per_layer, dtype)
    _logger.info("  Pre-selected neurons: %d", ctx.n_neurons)

    # Step 2: Build MLP input cache if not supplied.
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

    # Step 3: ANOVA-label all pre-selected neurons, one layer at a time.
    # Activations are computed and discarded per layer — peak memory is
    # O(n_layer_neurons × grid_cells) instead of O(N × grid_cells).
    if dataset is None:
        raise ValueError(
            "A dataset is required for ANOVA labeling. Pass --dataset."
        )
    decoded_prompt = adapter.tokenizer.decode(input_ids.detach().cpu().tolist())
    target_args = parse_numeric_args(decoded_prompt)
    _logger.info("ANOVA-labeling %d pre-selected neurons (layer-by-layer)", ctx.n_neurons)
    label_results = label_neurons_layer_by_layer(
        adapter,
        ctx.neuron_locations,
        mlp_input_cache,
        target_args=target_args,
        anova_range_radius=anova_range_radius,
        labelling_layer_batch_size=labelling_layer_batch_size,
    )

    # Step 4: Select top-K neurons per ANOVA label.
    # Sum categories use DLA-KL scoring (source_vectors @ W_U) rather than graph influence.
    _logger.info("Selecting ANOVA supernodes (top-%d per label)", anova_nodes_per_label)
    selected_row_indices, raw_supernodes, supernode_labels, raw_node_labels, raw_sum_member_scores = (
        select_anova_supernodes(
            label_results,
            anova_nodes_per_label=anova_nodes_per_label,
            sum_min_specificity=sum_min_specificity,
            strict=True,
            source_vectors=ctx.source_vectors,
            W_U=adapter.W_U,
            tokenizer=adapter.tokenizer,
            target_args=target_args,
        )
    )
    _logger.info("  ANOVA selected %d unique neurons", len(selected_row_indices))

    # Step 5: Build a boolean keep_mask and remap supernode indices.
    # After ctx.filter(keep_mask), filtered neuron j = original neuron selected_row_indices[j].
    keep_mask = torch.zeros(ctx.n_neurons, dtype=torch.bool, device=adapter.device)
    for idx in selected_row_indices:
        keep_mask[idx] = True

    old_to_new = {old: new for new, old in enumerate(selected_row_indices)}
    supernodes = [[old_to_new[idx] for idx in sn] for sn in raw_supernodes]
    node_labels = {old_to_new[old]: labels for old, labels in raw_node_labels.items()}
    sum_member_scores = {
        cat: {old_to_new[old]: scores for old, scores in cat_scores.items() if old in old_to_new}
        for cat, cat_scores in raw_sum_member_scores.items()
    }
    filtered_label_results = {old_to_new[old]: label_results[old] for old in selected_row_indices}

    # Step 6: Filter the attribution context to ANOVA-selected neurons only.
    filtered_ctx = ctx.filter(keep_mask)
    _logger.info("  Filtered context: %d neurons", filtered_ctx.n_neurons)

    # Step 7: Build the attribution graph for the ANOVA-selected neurons.
    _logger.info("Running edge attribution on ANOVA-filtered neurons")
    graph = _attribute_from_context(
        filtered_ctx,
        attribution_targets=attribution_targets,
        top_k_logits=top_k_logits,
        batch_size=batch_size,
        create_graph=build_create_graph,
        detach_result=detach_result,
        skip_logit_attribution=skip_logit_attribution,
        verbose=verbose,
    )
    _log_graph_summary(graph, logger=_logger, stage="Built (ANOVA-filtered)")

    # Step 8: Compute activation grids only for the M supergraph neurons (M << N).
    _logger.info(
        "Building activation-write result for %d supergraph neurons", filtered_ctx.n_neurons
    )
    filtered_awr = build_neuron_activation_write_result(
        adapter,
        dataset,
        filtered_ctx.neuron_locations,
        mlp_input_cache=mlp_input_cache,
    )

    # Step 9: Aggregate the adjacency matrix into a supergraph.
    _logger.info("Running build_super_graph")
    if supernode_heatmap_output_dir:
        _logger.info("Supernode heatmap output directory: %s", supernode_heatmap_output_dir)

    def _run_build_super_graph() -> SuperGraph:
        return build_super_graph(
            graph,
            supernodes=supernodes,
            supernode_labels=supernode_labels,
            node_labels=node_labels,
            supernode_heatmap_output_dir=supernode_heatmap_output_dir,
            activation_write_result=filtered_awr,
            sum_member_scores=sum_member_scores,
            filtered_label_results=filtered_label_results,
            W_U=adapter.W_U,
            tokenizer=adapter.tokenizer,
        )

    if no_grad_supergraph:
        with torch.no_grad():
            supergraph = _run_build_super_graph()
    else:
        supergraph = _run_build_super_graph()

    _log_supergraph_summary(graph, supergraph, logger=_logger)
    _log_pipeline_comparison(graph, supergraph, logger=_logger)

    return GraphPipelineResult(graph=graph, supergraph=supergraph)

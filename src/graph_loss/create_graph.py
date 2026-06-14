"""Unified graph creation pipeline: ANOVA label → build_graph → build_super_graph."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch

from graph_loss.anova_node_labels import parse_numeric_args
from graph_loss.attribution.attribute import _attribute_from_context, setup_attribution
from graph_loss.attribution.context import HFAttributionContext
from graph_loss.graph import (
    Graph,
    SuperGraph,
    build_super_graph,
    select_anova_supernodes,
    select_arg_supernodes,
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
    target_position_logits: torch.Tensor | None = None  # logits at target position; used for student DLA selection


@dataclass
class GraphSharedContext:
    """Phase-1 results for a sequence, shared across multiple target positions.

    Contains the attribution context and pre-computed ANOVA + token supernodes.
    The DLA supernode is position-specific and computed per-call in
    create_graph_at_position().
    """
    # Full-sequence attribution context (unfiltered, all pre-selected neurons).
    ctx: "HFAttributionContext"
    # ANOVA supernodes in unfiltered ctx-space neuron indices (empty if no ANOVA).
    anova_raw_supernodes: list
    anova_supernode_labels: list
    # Arg-token supernodes in unfiltered ctx-space neuron indices (empty if not requested).
    token_raw_supernodes: list
    token_supernode_labels: list
    # Sorted union of ANOVA + token neuron indices before DLA merge (ctx-space).
    anova_token_selected_indices: list
    # Per-neuron label dict in ctx-space (remapped to filtered-ctx in create_graph_at_position).
    raw_node_labels: dict
    # ANOVA label results passed to select_anova_supernodes in create_graph_at_position.
    label_results: dict
    raw_sum_member_scores: dict
    target_args: list
    # Pipeline params forwarded to create_graph_at_position.
    nodes_per_label: int
    # Precomputed activation write result (None in training, set by CLI path).
    activation_write_result: object
    supernode_heatmap_output_dir: str | None


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
# Phase-split helpers for multi-position graph loss
# ---------------------------------------------------------------------------

def build_shared_context(
    adapter: "HFLlamaGraphAdapter",
    input_ids: torch.Tensor,
    *,
    prop_neurons_per_layer: float = 0.1,
    dtype: torch.dtype | None = None,
    mlp_input_cache: dict | None = None,
    model_name: str | None = None,
    dataset: str | None = None,
    nodes_per_label: int = 10,
    anova_range_radius: int = 0,
    anova_neuron_chunk: int = 256,
    node_labels: list[str] | None = None,
    batch_size: int = 512,
    cache_batch_size: int = 32,
    supernode_heatmap_output_dir: str | None = None,
    refresh_mlp_cache: bool = False,
    logger: logging.Logger | None = None,
) -> GraphSharedContext:
    """Phase 1 of the graph pipeline: forward pass, ANOVA labeling, token supernodes.

    Does NOT include DLA supernodes (position-specific). Call
    create_graph_at_position() to add DLA and build the attribution graph at a
    specific sequence position.

    Args:
        adapter: Loaded HFLlamaGraphAdapter wrapping the model.
        input_ids: Tokenized input sequence (1-D tensor).
        See create_graph() for remaining argument documentation.

    Returns:
        GraphSharedContext holding the attribution context and pre-computed
        ANOVA + token supernodes in unfiltered ctx-space indices.
    """
    _logger = logger or logging.getLogger(__name__)

    _logger.info("Running setup_attribution (neuron pre-selection by gradient norm)")
    ctx = setup_attribution(adapter, input_ids, prop_neurons_per_layer, dtype)
    _logger.info("  Pre-selected neurons: %d", ctx.n_neurons)

    # An empty list means "no labels requested" => no ANOVA. Only a non-empty
    # list of labels (or "all") should trigger ANOVA labeling. Using bool()
    # instead of `is not None` prevents an empty list (e.g. from `labels or []`
    # in the CoT/compare-tokens path) from wrongly demanding a dataset.
    need_anova = bool(node_labels)
    label_results: dict = {}
    target_args: list = []

    if need_anova:
        if mlp_input_cache is None and dataset is None:
            raise ValueError(
                "--dataset is required when --graph-node-labels is specified and no "
                "mlp_input_cache is pre-supplied."
            )

        decoded = adapter.tokenizer.decode(input_ids.detach().cpu().tolist())
        target_args = parse_numeric_args(decoded)

        if mlp_input_cache is None and model_name is not None:
            from graph_loss.precompute_mlp_inputs import build_mlp_input_cache as _build_mlp_cache
            from utils import load_split as _load_split
            _all_data = _load_split(dataset, "all")
            _logger.info("Building MLP input cache for %s/all dataset", dataset)
            mlp_input_cache = _build_mlp_cache(adapter, dataset, model_name, data_dict=_all_data, batch_size=cache_batch_size, refresh=refresh_mlp_cache)
            _logger.info("  Built MLP cache: %d prompts", int(mlp_input_cache.get("meta", {}).get("n_prompts", 0)))

        if mlp_input_cache is not None:
            _cache_args = mlp_input_cache.get("meta", {}).get("numeric_args_by_prompt", [[]])
            _cache_n_dims = len(_cache_args[0]) if _cache_args else 0
            if _cache_n_dims != len(target_args):
                _logger.warning(
                    "Dataset has %d-arg prompts but current prompt has %d args — "
                    "arg%d+ ANOVA rules will not be generated. "
                    "Pass --dataset with a %d-arg dataset.",
                    _cache_n_dims, len(target_args), _cache_n_dims + 1, len(target_args),
                )

        _logger.info("ANOVA-labeling %d pre-selected neurons (layer-by-layer)", ctx.n_neurons)
        label_results = label_neurons_layer_by_layer(
            adapter,
            ctx.neuron_locations,
            mlp_input_cache,
            target_args=target_args,
            anova_range_radius=anova_range_radius,
            anova_neuron_chunk=anova_neuron_chunk,
        )
    else:
        # Still parse target_args so create_graph_at_position has them for DLA calls.
        decoded = adapter.tokenizer.decode(input_ids.detach().cpu().tolist())
        target_args = parse_numeric_args(decoded)

    # Select ANOVA supernodes (DLA excluded here; it is position-specific).
    if need_anova:
        _logger.info("Selecting ANOVA supernodes (top-%d per label)", nodes_per_label)
        anova_selected, anova_raw_supernodes, anova_supernode_labels, raw_node_labels, raw_sum_member_scores = (
            select_anova_supernodes(
                label_results,
                nodes_per_label=nodes_per_label,
                strict=True,
                source_vectors=ctx.source_vectors,
                W_U=adapter.W_U,
                tokenizer=adapter.tokenizer,
                target_args=target_args,
                allowed_labels=(None if "all" in node_labels else set(node_labels)) if node_labels is not None else set(),
                include_dla_node=False,  # DLA is position-specific; handled in create_graph_at_position
                model_logits=None,
            )
        )
        _logger.info("  ANOVA selected %d unique neurons", len(anova_selected))
    else:
        anova_selected: list = []
        anova_raw_supernodes: list = []
        anova_supernode_labels: list = []
        raw_node_labels: dict = {}
        raw_sum_member_scores: dict = {}

    # Select arg-token supernodes when no ANOVA labels requested.
    token_raw_supernodes: list = []
    token_supernode_labels: list = []
    all_selected = set(anova_selected)

    if not need_anova:
        _logger.info("Computing arg-token supernodes for %d token positions", ctx.n_tokens)
        token_raw_supernodes, token_supernode_labels = select_arg_supernodes(
            ctx,
            adapter.tokenizer,
            input_ids,
            nodes_per_token=nodes_per_label,
        )
        for sn in token_raw_supernodes:
            all_selected.update(sn)
        for sn, label in zip(token_raw_supernodes, token_supernode_labels):
            label_str = label[0] if label else "arg"
            for idx in sn:
                raw_node_labels.setdefault(idx, [])
                if label_str not in raw_node_labels[idx]:
                    raw_node_labels[idx].append(label_str)
        _logger.info(
            "  Arg-nodes: %d token supernodes; total unique neurons so far: %d",
            len(token_raw_supernodes),
            len(all_selected),
        )

    anova_token_selected_indices = sorted(all_selected)

    # Precompute activation write result if needed (CLI heatmap path only; None in training).
    need_awr = need_anova or (supernode_heatmap_output_dir is not None)
    activation_write_result = None
    if need_awr and anova_token_selected_indices:
        # Build a temporary filtered ctx to compute activation write results for ANOVA neurons.
        tmp_keep_mask = torch.zeros(ctx.n_neurons, dtype=torch.bool, device=adapter.device)
        for idx in anova_token_selected_indices:
            tmp_keep_mask[idx] = True
        tmp_filtered_ctx = ctx.filter(tmp_keep_mask)
        _logger.info("Building activation-write result for %d neurons", tmp_filtered_ctx.n_neurons)
        activation_write_result = build_neuron_activation_write_result(
            adapter,
            tmp_filtered_ctx.neuron_locations,
            mlp_input_cache=mlp_input_cache,
        )

    return GraphSharedContext(
        ctx=ctx,
        anova_raw_supernodes=anova_raw_supernodes,
        anova_supernode_labels=anova_supernode_labels,
        token_raw_supernodes=token_raw_supernodes,
        token_supernode_labels=token_supernode_labels,
        anova_token_selected_indices=anova_token_selected_indices,
        raw_node_labels=raw_node_labels,
        label_results=label_results,
        raw_sum_member_scores=raw_sum_member_scores,
        target_args=target_args,
        nodes_per_label=nodes_per_label,
        activation_write_result=activation_write_result,
        supernode_heatmap_output_dir=supernode_heatmap_output_dir,
    )


def create_graph_at_position(
    shared: GraphSharedContext,
    *,
    target_position: int = -1,
    dla_model_logits: torch.Tensor | None = None,
    attribution_targets=None,
    top_k_logits: float | None = 0.95,
    temperature: float = 2.0,
    batch_size: int = 512,
    build_create_graph: bool = False,
    detach_result: bool | None = None,
    skip_logit_attribution: bool = False,
    no_grad_supergraph: bool = False,
    verbose: bool = False,
    logger: logging.Logger | None = None,
) -> GraphPipelineResult:
    """Phase 2: compute DLA supernode + attribution graph at a specific sequence position.

    Reuses the shared attribution context and pre-computed ANOVA + token supernodes
    from build_shared_context(). Only the DLA supernode and the attribution graph
    itself are position-specific.

    Args:
        shared: Result of build_shared_context() for this sequence.
        target_position: Sequence position to attribute from (default -1 = last token).
        include_dla_node: Whether to compute and add the DLA supernode at target_position.
        dla_model_logits: Optional [d_vocab] logit vector for DLA selection. If None and
            include_dla_node is True, uses the model's own logits at target_position.
        See create_graph() for remaining argument documentation.
    """
    _logger = logger or logging.getLogger(__name__)
    ctx = shared.ctx
    adapter = ctx.adapter

    # --- Compute DLA supernode when no ANOVA labels were provided ---
    dla_raw_supernodes: list = []
    dla_supernode_labels: list = []
    dla_raw_node_labels: dict = {}
    dla_selected: list = []

    if not shared.anova_raw_supernodes:
        ref_logits = (
            dla_model_logits
            if dla_model_logits is not None
            else ctx.logits[0, target_position].detach()
        )
        _logger.info("Selecting DLA supernode at position %d", target_position)
        dla_selected, dla_raw_supernodes, dla_supernode_labels, dla_raw_node_labels, _ = (
            select_anova_supernodes(
                shared.label_results,
                nodes_per_label=shared.nodes_per_label,
                strict=True,
                source_vectors=ctx.source_vectors,
                W_U=adapter.W_U,
                tokenizer=adapter.tokenizer,
                target_args=shared.target_args,
                allowed_labels=set(),
                include_dla_node=True,
                model_logits=ref_logits,
            )
        )
        _logger.info("  DLA selected %d neurons", len(dla_selected))

    # --- Merge all neuron sets and remap to filtered-ctx indices ---
    all_selected_set = set(shared.anova_token_selected_indices) | set(dla_selected)
    selected_row_indices = sorted(all_selected_set)
    old_to_new = {old: new for new, old in enumerate(selected_row_indices)}

    # Combine all raw supernodes (still in ctx-space) and remap.
    all_raw_supernodes = (
        shared.anova_raw_supernodes
        + shared.token_raw_supernodes
        + dla_raw_supernodes
    )
    all_supernode_labels = (
        shared.anova_supernode_labels
        + shared.token_supernode_labels
        + dla_supernode_labels
    )
    supernodes = [[old_to_new[idx] for idx in sn] for sn in all_raw_supernodes]

    # Map filtered_ctx indices back to activation_write_result indices.
    # activation_write_result was built from anova_token_selected_indices only,
    # so DLA neurons (added later) have no entry and must not be looked up.
    awr_index_map: dict[int, int] = {
        old_to_new[ctx_idx]: awr_idx
        for awr_idx, ctx_idx in enumerate(shared.anova_token_selected_indices)
        if ctx_idx in old_to_new
    }

    # Merge per-neuron label dicts.
    merged_node_labels: dict = dict(shared.raw_node_labels)
    for idx, labels in dla_raw_node_labels.items():
        merged_node_labels.setdefault(idx, [])
        for lbl in labels:
            if lbl not in merged_node_labels[idx]:
                merged_node_labels[idx].append(lbl)

    node_labels_filtered = {
        old_to_new[old]: labels
        for old, labels in merged_node_labels.items()
        if old in old_to_new
    }
    sum_member_scores = {
        cat: {old_to_new[old]: scores for old, scores in cat_scores.items() if old in old_to_new}
        for cat, cat_scores in shared.raw_sum_member_scores.items()
    }
    filtered_label_results = {
        old_to_new[old]: shared.label_results[old]
        for old in selected_row_indices
        if old in shared.label_results
    }

    # --- Filter context and run attribution ---
    keep_mask = torch.zeros(ctx.n_neurons, dtype=torch.bool, device=adapter.device)
    for idx in selected_row_indices:
        keep_mask[idx] = True
    filtered_ctx = ctx.filter(keep_mask)
    # Free the unselected neurons' large tensors — only filtered_ctx is needed from here.
    ctx.source_vectors = None
    ctx.target_encoders = None
    ctx.neuron_activations = None
    _logger.info("  Filtered context: %d neurons", filtered_ctx.n_neurons)

    _logger.info("Running edge attribution on filtered neurons (position=%d)", target_position)
    graph = _attribute_from_context(
        filtered_ctx,
        attribution_targets=attribution_targets,
        top_k_logits=top_k_logits,
        temperature=temperature,
        batch_size=batch_size,
        create_graph=build_create_graph,
        detach_result=detach_result,
        skip_logit_attribution=skip_logit_attribution,
        verbose=verbose,
        target_position=target_position,
    )
    _log_graph_summary(graph, logger=_logger, stage="Built (filtered)")

    # --- Build supergraph ---
    _logger.info("Running build_super_graph")

    def _run_build_super_graph() -> "SuperGraph":
        return build_super_graph(
            graph,
            supernodes=supernodes,
            supernode_labels=all_supernode_labels,
            node_labels=node_labels_filtered,
            supernode_heatmap_output_dir=shared.supernode_heatmap_output_dir,
            activation_write_result=shared.activation_write_result,
            awr_index_map=awr_index_map,
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
    target_position_logits = shared.ctx.logits[0, target_position].detach()
    return GraphPipelineResult(graph=graph, supergraph=supergraph, target_position_logits=target_position_logits)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def create_graph(
    adapter: HFLlamaGraphAdapter,
    prompt: str | torch.Tensor | list[int],
    *,
    # attribution params
    attribution_targets=None,
    top_k_logits: float | None = 0.95,
    temperature: float = 2.0,
    prop_neurons_per_layer: float = 0.1,
    batch_size: int = 512,
    dtype: torch.dtype | None = None,
    verbose: bool = False,
    build_create_graph: bool = False,
    detach_result: bool | None = None,
    skip_logit_attribution: bool = False,
    # target position for attribution (default -1 = last token, existing behavior)
    target_position: int = -1,
    # ANOVA / supergraph params
    mlp_input_cache: dict | None = None,
    model_name: str | None = None,
    dataset: str | None = None,
    refresh_mlp_cache: bool = False,
    cache_batch_size: int = 32,
    supernode_heatmap_output_dir: str | None = None,
    nodes_per_label: int = 10,
    anova_range_radius: int = 0,
    anova_neuron_chunk: int = 256,
    node_labels: list[str] | None = None,
    dla_model_logits: torch.Tensor | None = None,
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
        top_k_logits: Cumulative probability threshold in (0, 1]. Selects the fewest
            top logits summing to this fraction, capped at 10.
        temperature: Softmax temperature for computing logit probabilities.
        prop_neurons_per_layer: Fraction of neurons to pre-select per layer.
        batch_size: Attribution batch size.
        dtype: Optional dtype override.
        verbose: Verbose logging during attribution.
        build_create_graph: PyTorch autograd create_graph flag (training use).
        detach_result: Whether to detach the adjacency from the grad graph.
        skip_logit_attribution: Skip logit attribution phase.
        mlp_input_cache: Pre-built MLP input cache.
        model_name: HuggingFace model identifier string (used as MLP cache key).
        supernode_heatmap_output_dir: Directory for per-supernode PDF heatmaps.
        nodes_per_label: Max neurons per ANOVA label supernode.
        anova_range_radius: Radius for target-centered ANOVA range masks.
        node_labels: Whitelist of ANOVA label names to include (e.g. ['arg1 range',
            'sum units']). Only supernodes whose category is in this list are created.
            If None (omitted), no ANOVA supernodes are created.
        include_dla_node: If True, create an additional "dla" supernode containing the
            top ``nodes_per_label`` neurons whose DLA distribution (write vector
            projected through W_U) best matches the reference output distribution by
            KL divergence.  The reference is ``dla_model_logits`` when provided,
            otherwise the model's own forward-pass logits.
        dla_model_logits: Optional ``[d_vocab]`` logit vector to use as the reference
            distribution for DLA supernode selection instead of the model's own output.
            Pass the teacher's logits here during student training so the student's DLA
            supernode is selected against the teacher's (correct, stable) output
            distribution rather than the student's own (wrong early in training and
            non-stationary across steps).
        include_arg_nodes: If True, create one ``"arg:TOKEN"`` supernode per token
            position in the prompt.  Each supernode contains the top
            ``nodes_per_label`` neurons (from the pre-ANOVA candidate pool)
            whose activation is most concentrated on that token's embedding, identified
            by back-propagating each neuron's activation to the token embeddings and
            selecting by normalised embedding-gradient mass (= minimum KL divergence
            from the delta distribution at that position).
        no_grad_supergraph: Wrap build_super_graph in torch.no_grad() (training use).
        logger: Optional logger; creates a module-level one if not provided.

    Returns:
        GraphPipelineResult with the ANOVA-filtered attribution graph and supergraph.
    """
    _logger = logger or logging.getLogger(__name__)
    input_ids = adapter.ensure_tokenized(prompt)

    shared = build_shared_context(
        adapter,
        input_ids,
        prop_neurons_per_layer=prop_neurons_per_layer,
        dtype=dtype,
        mlp_input_cache=mlp_input_cache,
        model_name=model_name,
        dataset=dataset,
        nodes_per_label=nodes_per_label,
        anova_range_radius=anova_range_radius,
        anova_neuron_chunk=anova_neuron_chunk,
        node_labels=node_labels,
        batch_size=batch_size,
        cache_batch_size=cache_batch_size,
        supernode_heatmap_output_dir=supernode_heatmap_output_dir,
        refresh_mlp_cache=refresh_mlp_cache,
        logger=_logger,
    )

    return create_graph_at_position(
        shared,
        target_position=target_position,
        dla_model_logits=dla_model_logits,
        attribution_targets=attribution_targets,
        top_k_logits=top_k_logits,
        temperature=temperature,
        batch_size=batch_size,
        build_create_graph=build_create_graph,
        detach_result=detach_result,
        skip_logit_attribution=skip_logit_attribution,
        no_grad_supergraph=no_grad_supergraph,
        verbose=verbose,
        logger=_logger,
    )

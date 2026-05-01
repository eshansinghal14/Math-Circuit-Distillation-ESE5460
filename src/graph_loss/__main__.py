import argparse
import logging

import torch
from huggingface_hub import login

from graph_loss.align import compute_supernode_dla
from graph_loss.attribution.attribute import attribute
from graph_loss.graph import (
    Graph,
    PruneResult,
    SuperGraph,
    build_super_graph,
    extract_supernode_members,
    prune_graph,
)
from graph_loss.replacement_model import TransformerLensReplacementModel
from graph_loss.utils import (
    add_graph_build_args,
    add_graph_prune_args,
    resolve_torch_dtype,
)
from utils import HF_READ_TOKEN


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


def _decode_vocab_token(model: TransformerLensReplacementModel, vocab_idx: int) -> str:
    token = model.tokenizer.decode([vocab_idx])
    return token.replace("\n", "\\n").replace("\r", "\\r")


@torch.no_grad()
def _log_supernode_top_dla_logits(
    supergraph: SuperGraph,
    graph: Graph,
    model: TransformerLensReplacementModel,
    *,
    logger: logging.Logger,
    top_k: int = 10,
) -> None:
    members = extract_supernode_members(supergraph, graph, model)
    W_U = model.unembed.W_U.to(device=graph.adjacency_device)

    logger.info("Supernode top DLA logits")
    if not members:
        logger.info("  no supernodes")
        return

    for member in members:
        cluster_id = int(member["cluster_id"])
        dla = compute_supernode_dla(member, W_U).detach().float().cpu()
        k = min(top_k, int(dla.numel()))
        values, indices = torch.topk(dla, k=k)
        formatted = ", ".join(
            f"{_decode_vocab_token(model, int(idx.item()))!r}:{float(value.item()):.6g}"
            for value, idx in zip(values, indices, strict=True)
        )
        logger.info(
            "  supernode %d (size=%d): %s",
            cluster_id,
            int(member["size"]),
            formatted,
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
    prune_result: PruneResult = None,
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


def _save_prune_result(path: str, prune_result: PruneResult) -> None:
    torch.save(
        {
            "node_mask": prune_result.node_mask,
            "edge_mask": prune_result.edge_mask,
            "cumulative_scores": prune_result.cumulative_scores,
        },
        path,
    )


def _save_supergraph(path: str, supergraph: SuperGraph) -> None:
    torch.save(
        {
            "supernode_adjacency_matrix": supergraph.supernode_adjacency_matrix,
            "supernodes": supergraph.supernodes,
        },
        path,
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Build, prune, and summarize neuron attribution graphs."
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--prompt", required=True, help="Prompt to analyze")
    parser.add_argument(
        "--graph_output_path",
        help="Optional path to save the graph (.pt)",
    )
    add_graph_build_args(parser)
    add_graph_prune_args(parser)
    parser.add_argument(
        "--prune_output_path",
        help="Optional path to save prune masks and cumulative scores (.pt)",
    )
    parser.add_argument(
        "--supergraph_output_path",
        help="Optional path to save the supergraph (.pt)",
    )
    parser.add_argument(
        "--cossim_eps",
        type=float,
        default=0.1,
        help="Deprecated fallback clustering epsilon; use --embedding_eps and --computation_eps",
    )
    parser.add_argument(
        "--embedding_sigma",
        type=float,
        default=1.5,
        help="Gaussian smoothing sigma for numeric-token embedding supernode clustering",
    )
    parser.add_argument(
        "--embedding_eps",
        type=float,
        default=0.1,
        help="Angular distance epsilon for numeric-token embedding supernode clustering",
    )
    parser.add_argument(
        "--computation_sigma",
        type=float,
        default=1.5,
        help="Gaussian smoothing sigma for final-token computation supernode clustering",
    )
    parser.add_argument(
        "--computation_eps",
        type=float,
        default=0.1,
        help="Angular distance epsilon for final-token computation supernode clustering",
    )
    parser.add_argument(
        "--dataset",
        help=(
            "Optional dataset prefix, filename, or path for activation-write "
            "output-neuron clustering and per-cluster PDF heatmaps"
        ),
    )
    parser.add_argument(
        "--activation_forward_batch_size",
        type=int,
        default=32,
        help="Batch size for dataset activation-write forward passes",
    )
    parser.add_argument(
        "--activation-write-cache-path",
        "--activation_write_cache_path",
        dest="activation_write_cache_path",
        help=(
            "Optional cache root for dataset activation-write results. "
            "A model-name folder is created inside this path."
        ),
    )

    args = parser.parse_args()

    dtype = resolve_torch_dtype(args.dtype)

    if HF_READ_TOKEN:
        logger.info("Authenticating with Hugging Face token")
        login(HF_READ_TOKEN)

    logger.info("Loading model: %s", args.model)
    model = TransformerLensReplacementModel.from_pretrained(
        args.model,
        dtype=dtype,
    )

    logger.info("Running attribution graph build")
    graph = attribute(
        prompt=args.prompt,
        model=model,
        top_k_logits=args.top_k_logits,
        prop_neurons_per_layer=args.prop_neurons_per_layer,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )
    _log_graph_summary(graph, logger=logger, stage="Built")

    if args.graph_output_path:
        logger.info("Saving graph to %s", args.graph_output_path)
        graph.to_pt(args.graph_output_path)

    prune_result = None
    if args.prune:
        logger.info("Running prune_graph")
        prune_result = prune_graph(
            graph,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
        )
        _log_prune_summary(
            graph,
            prune_result,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            logger=logger,
        )
        if args.prune_output_path:
            logger.info("Saving prune result to %s", args.prune_output_path)
            _save_prune_result(args.prune_output_path, prune_result)
            
        logger.info("Applying prune masks to graph")
        graph = graph.apply_prune_result(prune_result)

    logger.info("Running build_super_graph")
    supergraph = build_super_graph(
        graph,
        model,
        prune_result=prune_result,
        cossim_eps=args.cossim_eps,
        embedding_sigma=args.embedding_sigma,
        embedding_eps=args.embedding_eps,
        computation_sigma=args.computation_sigma,
        computation_eps=args.computation_eps,
        dataset=args.dataset,
        activation_forward_batch_size=args.activation_forward_batch_size,
        activation_write_cache_path=args.activation_write_cache_path,
        model_name=args.model,
    )
    _log_supergraph_summary(
        graph,
        supergraph,
        logger=logger,
    )
    _log_supernode_top_dla_logits(supergraph, graph, model, logger=logger)
    if args.supergraph_output_path:
        logger.info("Saving supergraph to %s", args.supergraph_output_path)
        _save_supergraph(args.supergraph_output_path, supergraph)

    _log_pipeline_comparison(graph, supergraph, logger=logger, prune_result=prune_result)
    logger.info("Done")


if __name__ == "__main__":
    main()

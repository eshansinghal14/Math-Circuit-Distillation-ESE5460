import argparse
import logging
import math
import os
import warnings

import torch
from huggingface_hub import login

from graph_loss.attribution.attribute import attribute
from graph_loss.frontend_export import (
    default_frontend_output_dir,
    export_supergraph_frontend,
)
from graph_loss.graph import (
    Graph,
    PruneResult,
    SuperGraph,
    build_super_graph,
    compute_neuron_logit_influence,
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
            "node_labels": supergraph.node_labels,
            "supernode_labels": supergraph.supernode_labels,
            "supernode_heatmap_pdf_paths": supergraph.supernode_heatmap_pdf_paths,
        },
        path,
    )


@torch.no_grad()
def _compute_neuron_dla_on_logit_targets(
    graph: Graph,
    model: TransformerLensReplacementModel,
) -> torch.Tensor:
    if graph.n_logits == 0:
        return torch.empty((graph.n_neurons, 0), dtype=torch.float32)

    target_token_ids = graph.logit_token_ids.to(device=graph.adjacency_device)
    W_U_targets = model.unembed.W_U.to(device=graph.adjacency_device)[:, target_token_ids]
    result = torch.empty(
        graph.n_neurons,
        graph.n_logits,
        dtype=W_U_targets.dtype,
        device=graph.adjacency_device,
    )
    w_out_cache = {}

    for node_id in range(graph.n_neurons):
        layer = int(graph.neuron_locations[node_id, 0].item())
        neuron_id = int(graph.neuron_locations[node_id, 2].item())
        if layer not in w_out_cache:
            old_mlp = model.blocks[layer].mlp.old_mlp
            w_out_cache[layer] = model._row_oriented_weight(
                old_mlp.W_out.to(device=graph.adjacency_device, dtype=W_U_targets.dtype)
            )

        activation = graph.neuron_activations[node_id].to(
            device=graph.adjacency_device,
            dtype=W_U_targets.dtype,
        )
        result[node_id] = activation * (w_out_cache[layer][neuron_id] @ W_U_targets)

    return result


def _save_influence_calibration_plot(
    *,
    layers: list[int],
    spearman_values: list[float],
    output_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(layers, spearman_values, alpha=0.65, s=14)
    ax.set_xlabel("Neuron layer")
    ax.set_ylabel("Spearman rho(graph influence, ablation delta)")
    ax.set_title("Neuron Influence Calibration")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def _compute_neuron_logit_prob_deltas(
    graph: Graph,
    model: TransformerLensReplacementModel,
    *,
    batch_size: int,
    logger: logging.Logger,
) -> torch.Tensor:
    if graph.n_logits == 0:
        return torch.empty((graph.n_neurons, 0), dtype=torch.float32)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    input_ids = graph.input_tokens.to(model.cfg.device)
    target_token_ids = graph.logit_token_ids.to(device=input_ids.device)
    if target_token_ids.numel() and int(target_token_ids.max().item()) >= model.cfg.d_vocab:
        raise ValueError("Ablation calibration only supports real vocabulary logit targets")

    baseline_logits = model(input_ids)
    baseline_probs = torch.softmax(baseline_logits[0, -1], dim=-1)[target_token_ids]

    device = input_ids.device
    dtype = baseline_logits.dtype
    neuron_locations = graph.neuron_locations.to(device=device)
    neuron_activations = graph.neuron_activations.to(device=device, dtype=dtype)
    source_vectors = torch.empty(
        graph.n_neurons,
        model.cfg.d_model,
        device=device,
        dtype=dtype,
    )
    w_out_cache = {}
    for node_id in range(graph.n_neurons):
        layer = int(neuron_locations[node_id, 0].item())
        neuron_id = int(neuron_locations[node_id, 2].item())
        if layer not in w_out_cache:
            old_mlp = model.blocks[layer].mlp.old_mlp
            w_out_cache[layer] = model._row_oriented_weight(
                old_mlp.W_out.to(device=device, dtype=dtype)
            )
        source_vectors[node_id] = neuron_activations[node_id] * w_out_cache[layer][neuron_id]

    deltas = torch.empty(
        graph.n_neurons,
        graph.n_logits,
        dtype=torch.float32,
        device="cpu",
    )
    total_batches = math.ceil(graph.n_neurons / batch_size)

    def make_ablation_hook(
        batch_indices: torch.Tensor,
        positions: torch.Tensor,
        vectors: torch.Tensor,
    ):
        def hook_fn(acts: torch.Tensor, hook):
            acts_out = acts.clone()
            acts_out[batch_indices, positions] -= vectors.to(device=acts.device, dtype=acts.dtype)
            return acts_out

        return hook_fn

    for batch_idx, start in enumerate(range(0, graph.n_neurons, batch_size), start=1):
        end = min(start + batch_size, graph.n_neurons)
        batch_locations = neuron_locations[start:end]
        batch_vectors = source_vectors[start:end]
        hooks = []
        for layer in batch_locations[:, 0].unique().tolist():
            layer_idx = int(layer)
            layer_mask = batch_locations[:, 0] == layer_idx
            batch_indices = torch.where(layer_mask)[0].to(device=device)
            hooks.append(
                (
                    f"blocks.{layer_idx}.{model.feature_output_hook}",
                    make_ablation_hook(
                        batch_indices,
                        batch_locations[layer_mask, 1].to(device=device),
                        batch_vectors[layer_mask],
                    ),
                )
            )

        ablated_logits = model.run_with_hooks(
            input_ids.expand(end - start, -1),
            fwd_hooks=hooks,
        )
        ablated_probs = torch.softmax(ablated_logits[:, -1], dim=-1)[:, target_token_ids]
        deltas[start:end] = (baseline_probs.unsqueeze(0) - ablated_probs).detach().float().cpu()

        logger.info(
            "  ablation calibration batch %d/%d: neurons %d-%d",
            batch_idx,
            total_batches,
            start,
            end - 1,
        )

    return deltas


def _run_influence_calibration(
    graph: Graph,
    model: TransformerLensReplacementModel,
    *,
    batch_size: int,
    output_path: str,
    logger: logging.Logger,
) -> None:
    import importlib

    try:
        scipy_stats = importlib.import_module("scipy.stats")
    except ImportError as exc:
        raise ImportError("scipy is required for --test-influence-calibration") from exc
    spearmanr = scipy_stats.spearmanr

    graph_influence = compute_neuron_logit_influence(graph).detach().float().cpu()
    ablation_delta = _compute_neuron_logit_prob_deltas(
        graph,
        model,
        batch_size=batch_size,
        logger=logger,
    )
    layers = [int(row[0].item()) for row in graph.neuron_locations.detach().cpu()]
    spearman_values = []

    for node_id in range(graph.n_neurons):
        if graph.n_logits < 2:
            spearman_values.append(float("nan"))
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho = spearmanr(
                graph_influence[node_id].numpy(),
                ablation_delta[node_id].numpy(),
                nan_policy="omit",
            ).statistic
        spearman_values.append(float(rho))

    valid_values = [value for value in spearman_values if not math.isnan(value)]
    logger.info(
        "Influence calibration Spearman values: valid=%d/%d mean=%.6g",
        len(valid_values),
        len(spearman_values),
        sum(valid_values) / len(valid_values) if valid_values else float("nan"),
    )
    logger.info("Saving influence calibration scatter plot to %s", output_path)
    _save_influence_calibration_plot(
        layers=layers,
        spearman_values=spearman_values,
        output_path=output_path,
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Build, prune, and summarize neuron attribution graphs."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Base model architecture name (e.g. 'gpt2'). Used by TransformerLens to build the model config.",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Local path to a saved HuggingFace model whose weights override the base --model weights.",
    )
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
        "--frontend-output-dir",
        "--frontend_output_dir",
        dest="frontend_output_dir",
        default=str(default_frontend_output_dir()),
        help=(
            "Directory for the static frontend assets and generated supergraph JSON. "
            "Defaults to graph_loss/frontend_assets."
        ),
    )
    parser.add_argument(
        "--frontend-slug",
        "--frontend_slug",
        dest="frontend_slug",
        help="Optional slug for the generated frontend graph_data/<slug>.json file",
    )
    parser.add_argument(
        "--no-frontend",
        dest="export_frontend",
        action="store_false",
        help="Skip exporting static frontend JSON for the generated supergraph",
    )
    parser.set_defaults(export_frontend=True)
    parser.add_argument(
        "--test-influence-calibration",
        dest="test_influence_calibration",
        action="store_true",
        help=(
            "Build the graph, then compare each neuron's graph logit influence "
            "against zero-ablation logit probability deltas with Spearman correlation "
            "instead of building a supergraph"
        ),
    )
    parser.add_argument(
        "--influence-calibration-output-path",
        "--influence_calibration_output_path",
        dest="influence_calibration_output_path",
        default="influence_calibration_spearman.png",
        help="Path to save the influence calibration scatter plot",
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
    parser.add_argument(
        "--supernode-heatmap-output-dir",
        "--supernode_heatmap_output_dir",
        dest="supernode_heatmap_output_dir",
        default="supernode_heatmaps",
        help=(
            "Directory for per-supernode activation heatmap PDFs when dataset "
            "activation clustering is used"
        ),
    )
    parser.add_argument(
        "--anova-nodes-per-label",
        "--anova_nodes_per_label",
        dest="anova_nodes_per_label",
        type=int,
        default=10,
        help="Maximum positive-variance ANOVA nodes to include per label supernode",
    )
    parser.add_argument(
        "--anova-range-radius",
        "--anova_range_radius",
        dest="anova_range_radius",
        type=int,
        default=0,
        help=(
            "Radius around the target arg1/arg2 values for ANOVA range basis masks. "
            "Use 0 for exact target-value masks."
        ),
    )
    parser.add_argument(
        "--sum-min-specificity",
        "--sum_min_specificity",
        dest="sum_min_specificity",
        type=float,
        default=0.0,
        help="Minimum ANOVA specificity for a neuron to be included in a sum range/sum units supernode (neurons are ranked by DLA cosine similarity).",
    )

    args = parser.parse_args()
    if args.supernode_heatmap_output_dir:
        args.supernode_heatmap_output_dir = os.path.abspath(args.supernode_heatmap_output_dir)

    dtype = resolve_torch_dtype(args.dtype)

    if HF_READ_TOKEN:
        logger.info("Authenticating with Hugging Face token")
        login(HF_READ_TOKEN)

    if args.model_path:
        from transformers import AutoModelForCausalLM
        logger.info("Loading weights from local path: %s", args.model_path)
        hf_model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype)
        logger.info("Building TransformerLens model with architecture: %s", args.model)
        model = TransformerLensReplacementModel.from_pretrained(
            args.model,
            dtype=dtype,
            hf_model=hf_model,
        )
    else:
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
        batch_size=args.attribution_batch_size,
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

    if args.test_influence_calibration:
        logger.info("Running influence calibration test")
        _run_influence_calibration(
            graph,
            model,
            batch_size=args.activation_forward_batch_size,
            output_path=args.influence_calibration_output_path,
            logger=logger,
        )
        logger.info("Done")
        return

    logger.info("Running build_super_graph")
    logger.info(
        "Supernode heatmap output directory: %s",
        args.supernode_heatmap_output_dir,
    )
    supergraph = build_super_graph(
        graph,
        model,
        prune_result=prune_result,
        dataset=args.dataset,
        activation_forward_batch_size=args.activation_forward_batch_size,
        activation_write_cache_path=args.activation_write_cache_path,
        model_name=args.model,
        supernode_heatmap_output_dir=args.supernode_heatmap_output_dir,
        anova_nodes_per_label=args.anova_nodes_per_label,
        anova_range_radius=args.anova_range_radius,
        sum_min_specificity=args.sum_min_specificity,
    )
    _log_supergraph_summary(
        graph,
        supergraph,
        logger=logger,
    )
    if args.supergraph_output_path:
        logger.info("Saving supergraph to %s", args.supergraph_output_path)
        _save_supergraph(args.supergraph_output_path, supergraph)

    if args.export_frontend:
        logger.info("Exporting supergraph frontend to %s", args.frontend_output_dir)
        graph_data_path = export_supergraph_frontend(
            graph,
            supergraph,
            output_dir=args.frontend_output_dir,
            slug=args.frontend_slug,
            model_name=args.model,
            tokenizer=getattr(model, "tokenizer", None),
        )
        logger.info("Saved frontend graph data: %s", graph_data_path)
        logger.info("Open %s to view the visualization", os.path.join(args.frontend_output_dir, "index.html"))

    _log_pipeline_comparison(graph, supergraph, logger=logger, prune_result=prune_result)
    logger.info("Done")


if __name__ == "__main__":
    main()

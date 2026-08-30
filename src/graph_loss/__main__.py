import argparse
import logging
import os

from graph_loss.create_graph import (
    GraphPipelineResult,
    create_graph,
)
from graph_loss.frontend_assets.frontend_export import (
    default_frontend_output_dir,
    export_supergraph_frontend,
)
from graph_loss.hf_adapter import HFLlamaGraphAdapter
from graph_loss.utils import add_graph_build_args
from utils import DIR_ROOT, load_model


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Build and summarize neuron attribution graphs."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model name (e.g. 'meta-llama/Meta-Llama-3.1-8B-Instruct').",
    )
    parser.add_argument("--prompt", required=True, help="Prompt to analyze")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name under datasets/ used to build the MLP activation cache (e.g. '22_add', '22_add3'). Required when --graph-node-labels is specified.",
    )
    add_graph_build_args(parser)
    parser.add_argument(
        "--freeze-attention",
        dest="freeze_attention",
        action="store_true",
        help=(
            "Stop-gradient the attention pattern during edge attribution, so edges "
            "reflect only the direct residual-stream path rather than also including "
            "paths where a source node shifts where the target attends. Matches the "
            "published circuit-tracing linearisation. Forces an eager attention kernel, "
            "which is slower and uses more memory than SDPA."
        ),
    )
    parser.add_argument(
        "--freeze-rms-norm",
        dest="freeze_rms_norm",
        action="store_true",
        help=(
            "Stop-gradient the RMSNorm reciprocal-norm scale during edge attribution, "
            "dropping the rank-1 Jacobian term that otherwise subtracts the component "
            "of each edge along the activation direction. No speed penalty."
        ),
    )
    parser.add_argument(
        "--cache-batch-size",
        "--cache_batch_size",
        dest="cache_batch_size",
        type=int,
        default=32,
        help="Prompt batch size when building the MLP activation cache (default: 32).",
    )
    parser.add_argument(
        "--refresh-mlp-cache",
        "--refresh_mlp_cache",
        dest="refresh_mlp_cache",
        action="store_true",
        help="Delete and rebuild the MLP input cache even if one already exists.",
    )
    parser.add_argument(
        "--frontend-slug",
        "--frontend_slug",
        dest="frontend_slug",
        help="Optional slug for the generated frontend graph_data/<slug>.json file",
    )
    parser.add_argument(
        "--supernode-heatmap-output-dir",
        "--supernode_heatmap_output_dir",
        dest="supernode_heatmap_output_dir",
        default=os.path.join(DIR_ROOT, "supernode_heatmaps"),
        help=(
            "Directory for per-supernode activation heatmap PDFs when dataset "
            "activation clustering is used"
        ),
    )
    args = parser.parse_args()
    if args.graph_node_labels and args.dataset is None:
        parser.error("--dataset is required when --graph-node-labels is specified")
    if args.supernode_heatmap_output_dir:
        if not os.path.isabs(args.supernode_heatmap_output_dir):
            args.supernode_heatmap_output_dir = os.path.join(DIR_ROOT, args.supernode_heatmap_output_dir)

    frontend_output_dir = str(default_frontend_output_dir())

    logger.info("Loading model: %s", args.model)
    hf_model, tokenizer = load_model(args.model)
    hf_model = hf_model.eval()
    device = next(hf_model.parameters()).device
    adapter = HFLlamaGraphAdapter(hf_model, tokenizer, device)

    result: GraphPipelineResult = create_graph(
        adapter,
        args.prompt,
        top_k_logits=args.top_k_logits,
        temperature=args.temperature,
        prop_neurons_per_layer=args.prop_neurons_per_layer,
        batch_size=args.attribution_batch_size,
        verbose=args.verbose,
        model_name=args.model,
        dataset=args.dataset,
        refresh_mlp_cache=args.refresh_mlp_cache,
        cache_batch_size=args.cache_batch_size,
        supernode_heatmap_output_dir=args.supernode_heatmap_output_dir,
        nodes_per_label=args.nodes_per_label,
        anova_range_radius=args.anova_range_radius,
        anova_neuron_chunk=args.anova_neuron_chunk,
        node_labels=args.graph_node_labels,
        freeze_attention=args.freeze_attention,
        freeze_rms_norm=args.freeze_rms_norm,
        logger=logger,
    )

    logger.info("Exporting supergraph frontend to %s", frontend_output_dir)
    graph_data_path = export_supergraph_frontend(
        result.graph,
        result.supergraph,
        output_dir=frontend_output_dir,
        slug=args.frontend_slug,
        model_name=args.model,
        tokenizer=adapter.tokenizer,
    )
    logger.info("Saved frontend graph data: %s", graph_data_path)
    logger.info(
        "Open %s to view the visualization",
        os.path.join(frontend_output_dir, "index.html"),
    )

    logger.info("Done")


if __name__ == "__main__":
    main()

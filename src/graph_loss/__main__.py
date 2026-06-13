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
    add_graph_build_args(parser)
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
        supernode_heatmap_output_dir=args.supernode_heatmap_output_dir,
        nodes_per_label=args.nodes_per_label,
        anova_range_radius=args.anova_range_radius,
        node_labels=args.graph_node_labels,
        use_heatmap_arg_nodes=args.use_heatmap_arg_nodes,
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

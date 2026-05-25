import argparse
import logging
import os

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

from graph_loss.create_graph import (
    GraphPipelineResult,
    create_graph,
    save_supergraph,
)
from graph_loss.frontend_export import (
    default_frontend_output_dir,
    export_supergraph_frontend,
)
from graph_loss.hf_adapter import HFLlamaGraphAdapter
from graph_loss.utils import (
    add_graph_build_args,
    resolve_torch_dtype,
)
from utils import HF_READ_TOKEN


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Build and summarize neuron attribution graphs via ANOVA-first pipeline."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model name (e.g. 'meta-llama/Meta-Llama-3.1-8B-Instruct').",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Local path to a saved HuggingFace model whose weights override the base --model weights.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available, else cpu).",
    )
    parser.add_argument("--prompt", required=True, help="Prompt to analyze")
    parser.add_argument(
        "--graph_output_path",
        help="Optional path to save the graph (.pt)",
    )
    add_graph_build_args(parser)
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
        "--dataset",
        required=True,
        help=(
            "Dataset prefix, filename, or path for ANOVA activation-grid labeling "
            "and per-cluster PDF heatmaps"
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
    args = parser.parse_args()
    if args.supernode_heatmap_output_dir:
        args.supernode_heatmap_output_dir = os.path.abspath(args.supernode_heatmap_output_dir)

    dtype = resolve_torch_dtype(args.dtype)

    if HF_READ_TOKEN:
        logger.info("Authenticating with Hugging Face token")
        login(HF_READ_TOKEN)

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.model_path:
        logger.info("Loading weights from local path: %s", args.model_path)
        hf_model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype)
    else:
        logger.info("Loading model: %s", args.model)
        hf_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    hf_model = hf_model.to(device).eval()
    adapter = HFLlamaGraphAdapter(hf_model, tokenizer, device)

    result: GraphPipelineResult = create_graph(
        adapter,
        args.prompt,
        top_k_logits=args.top_k_logits,
        temperature=args.temperature,
        prop_neurons_per_layer=args.prop_neurons_per_layer,
        batch_size=args.attribution_batch_size,
        verbose=args.verbose,
        include_token_nodes=args.include_token_nodes,
        include_logit_nodes=args.include_logit_nodes,
        dataset=args.dataset,

        model_name=args.model,
        supernode_heatmap_output_dir=args.supernode_heatmap_output_dir,
        anova_nodes_per_label=args.anova_nodes_per_label,
        anova_range_radius=args.anova_range_radius,
        sum_min_specificity=args.sum_min_specificity,
        logger=logger,
    )

    if args.graph_output_path:
        logger.info("Saving graph to %s", args.graph_output_path)
        result.graph.to_pt(args.graph_output_path)

    if args.supergraph_output_path:
        logger.info("Saving supergraph to %s", args.supergraph_output_path)
        save_supergraph(args.supergraph_output_path, result.supergraph)

    if args.export_frontend:
        logger.info("Exporting supergraph frontend to %s", args.frontend_output_dir)
        graph_data_path = export_supergraph_frontend(
            result.graph,
            result.supergraph,
            output_dir=args.frontend_output_dir,
            slug=args.frontend_slug,
            model_name=args.model,
            tokenizer=adapter.tokenizer,
        )
        logger.info("Saved frontend graph data: %s", graph_data_path)
        logger.info(
            "Open %s to view the visualization",
            os.path.join(args.frontend_output_dir, "index.html"),
        )

    logger.info("Done")


if __name__ == "__main__":
    main()

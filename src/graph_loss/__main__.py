import argparse
import logging

import torch

from graph_loss.attribution.attribute import attribute
from graph_loss.replacement_model import TransformerLensReplacementModel


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Minimal CLI for neuron attribution graph generation."
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--prompt", required=True, help="Prompt to analyze")
    parser.add_argument(
        "--graph_output_path",
        required=True,
        help="Where to save the graph (.pt)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16", "fp32", "bf16", "fp16"],
        default="float32",
        help="Model dtype",
    )
    parser.add_argument(
        "--logit_min_prob",
        type=float,
        default=1e-5,
        help="Only include logit nodes with probability >= this threshold",
    )
    parser.add_argument(
        "--prop_neurons_per_layer",
        type=float,
        default=0.1,
        help="Fraction of neurons to keep per layer",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for attribution backward passes",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show attribution progress",
    )

    args = parser.parse_args()

    dtype_mapping = {
        "fp32": "float32",
        "bf16": "bfloat16",
        "fp16": "float16",
    }
    dtype_name = dtype_mapping.get(args.dtype, args.dtype)
    dtype = getattr(torch, dtype_name)

    logging.info("Loading model: %s", args.model)
    model = TransformerLensReplacementModel.from_pretrained(
        args.model,
        dtype=dtype,
    )

    logging.info("Running attribution")
    graph = attribute(
        prompt=args.prompt,
        model=model,
        logit_min_prob=args.logit_min_prob,
        prop_neurons_per_layer=args.prop_neurons_per_layer,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    logging.info("Saving graph to %s", args.graph_output_path)
    graph.to_pt(args.graph_output_path)
    logging.info("Done")


if __name__ == "__main__":
    main()

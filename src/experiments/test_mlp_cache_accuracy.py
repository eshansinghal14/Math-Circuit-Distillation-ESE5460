"""Test MLP cache accuracy: compare supernode heatmaps with and without MLP cache.

Runs the graph_loss pipeline (attribute → build_super_graph) on a
checkpoint model twice for a single prompt:

  output_dir/normal_pass/  : ground-truth heatmaps via full forward pass
  output_dir/cached_pass/  : heatmaps recomputed from the original base model's
                             cached residual-stream inputs using the checkpoint weights

Usage (from src/):
    python -m experiments.test_mlp_cache_accuracy \
        --model-id meta-llama/Llama-3.2-1B-Instruct \
        --checkpoint-path /path/to/checkpoint_step_0010/student_model \
        --prompt "5+3=" \
        --dataset add_tight \
        --output-dir /tmp/cache_accuracy_test \
        --device cpu --dtype bfloat16
"""

from __future__ import annotations

import argparse
import logging
import os

import torch
from transformers import AutoModelForCausalLM

from graph_loss.create_graph import create_graph
from graph_loss.hf_adapter import HFLlamaGraphAdapter
from graph_loss.neuron_activation_heatmap import _resolve_dataset_path
from graph_loss.precompute_mlp_inputs import build_mlp_input_cache
from graph_loss.utils import add_graph_build_args, resolve_torch_dtype
from utils.config import HF_READ_TOKEN
from utils.hf_models import load_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare labeled-supernode activation heatmaps: "
            "normal forward pass vs MLP-cache pass on a checkpoint model."
        )
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="HuggingFace model ID — original base model and checkpoint architecture",
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="HF checkpoint dir of the updated model to evaluate",
    )
    parser.add_argument("--prompt", required=True, help="Single prompt to attribute (e.g. '5+3=')")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root dir; normal_pass/ and cached_pass/ subdirs are created inside it",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for base HF model during cache build",
    )
    add_graph_build_args(parser)
    # Supergraph / heatmap args
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name or path — used by build_super_graph to compute activation heatmaps",
    )
    parser.add_argument(
        "--activation-forward-batch-size",
        dest="activation_forward_batch_size",
        type=int,
        default=32,
        help="Batch size for activation-write forward passes",
    )
    parser.add_argument(
        "--activation-write-cache-path",
        dest="activation_write_cache_path",
        default=None,
        help="Optional cache root for activation-write results",
    )
    parser.add_argument(
        "--anova-nodes-per-label",
        dest="anova_nodes_per_label",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--anova-range-radius",
        dest="anova_range_radius",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--sum-min-specificity",
        dest="sum_min_specificity",
        type=float,
        default=0.0,
    )
    args = parser.parse_args()

    if HF_READ_TOKEN:
        from huggingface_hub import login
        login(HF_READ_TOKEN)

    dtype = resolve_torch_dtype(args.dtype)
    normal_dir = os.path.join(args.output_dir, "normal_pass")
    cached_dir = os.path.join(args.output_dir, "cached_pass")
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(cached_dir, exist_ok=True)

    # Resolve dataset path once so both build_mlp_input_cache and build_super_graph use
    # the same absolute path (consistent cache keying).
    dataset_path = _resolve_dataset_path(args.dataset)

    # Load the updated checkpoint model.
    logger.info("Loading checkpoint model from %s", args.checkpoint_path)
    hf_ckpt = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, torch_dtype=dtype)
    hf_ckpt = hf_ckpt.to(args.device)
    hf_ckpt.eval()
    _, ckpt_tokenizer = load_model(args.model_id)
    checkpoint_adapter = HFLlamaGraphAdapter(hf_ckpt, ckpt_tokenizer, args.device)

    # Build MLP cache from the original base HF model + the dataset.
    logger.info("Building/loading MLP cache from base model: %s", args.model_id)
    base_hf, tokenizer = load_model(args.model_id)
    base_hf = base_hf.to(args.device)
    adapter = HFLlamaGraphAdapter(base_hf, tokenizer, args.device)
    mlp_cache = build_mlp_input_cache(
        adapter,
        dataset_path,
        args.model_id,
    )
    del base_hf, adapter

    common_kwargs = dict(
        top_k_logits=args.top_k_logits,
        prop_neurons_per_layer=args.prop_neurons_per_layer,
        batch_size=args.attribution_batch_size,
        verbose=args.verbose,
        dataset=dataset_path,
        activation_forward_batch_size=args.activation_forward_batch_size,
        activation_write_cache_path=args.activation_write_cache_path,
        model_name=args.model_id,
        anova_nodes_per_label=args.anova_nodes_per_label,
        anova_range_radius=args.anova_range_radius,
        sum_min_specificity=args.sum_min_specificity,
    )

    # Pass 1: ground truth — full forward pass on checkpoint, no MLP cache.
    logger.info("Pass 1: normal forward pass → %s", normal_dir)
    create_graph(
        checkpoint_adapter,
        args.prompt,
        supernode_heatmap_output_dir=normal_dir,
        **common_kwargs,
    )

    # Pass 2: cached — original base model's residual-stream inputs + checkpoint weights.
    logger.info("Pass 2: cached pass → %s", cached_dir)
    create_graph(
        checkpoint_adapter,
        args.prompt,
        mlp_input_cache=mlp_cache,
        supernode_heatmap_output_dir=cached_dir,
        **common_kwargs,
    )

    logger.info("Done. Heatmaps saved to:\n  %s\n  %s", normal_dir, cached_dir)


if __name__ == "__main__":
    main()

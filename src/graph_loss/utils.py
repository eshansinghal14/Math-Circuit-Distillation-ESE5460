import argparse
import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any

import torch


DTYPE_CHOICES = ["float32", "bfloat16", "float16", "fp32", "bf16", "fp16"]


@dataclass
class ActivationWriteResult:
    activations: torch.Tensor  # [n_neurons, *grid_shape]
    arg_values: list            # list[list[int]], one list per arg dimension


def convert_nnsight_config_to_transformerlens(cfg):
    """Identity pass-through for HFGraphConfig objects."""
    return cfg


@dataclass
class UnifiedConfig:
    """Minimal LLaMA config used by the graph-loss pipeline."""

    n_layers: int
    d_model: int
    d_head: int
    n_heads: int
    d_mlp: int
    d_vocab: int

    tokenizer_name: str
    model_name: str
    original_architecture: str = "LlamaForCausalLM"
    n_key_value_heads: int | None = None
    dtype: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "UnifiedConfig":
        """Create from dictionary."""
        return cls(
            n_layers=config_dict["n_layers"],
            d_model=config_dict["d_model"],
            d_head=config_dict["d_head"],
            n_heads=config_dict["n_heads"],
            d_mlp=config_dict["d_mlp"],
            d_vocab=config_dict["d_vocab"],
            tokenizer_name=config_dict["tokenizer_name"],
            model_name=config_dict["model_name"],
            original_architecture=config_dict["original_architecture"],
            n_key_value_heads=config_dict.get("n_key_value_heads"),
            dtype=config_dict.get("dtype"),
        )


def add_graph_build_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dtype",
        choices=DTYPE_CHOICES,
        default="float32",
        help="Model dtype",
    )
    parser.add_argument(
        "--top_k_logits",
        type=float,
        default=0.95,
        help="Cumulative probability threshold in (0, 1]. Selects the fewest top logits "
             "summing to this fraction, capped at 10.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="Softmax temperature for computing logit probabilities (default: 2.0).",
    )
    parser.add_argument(
        "--prop_neurons_per_layer",
        type=float,
        default=0.1,
        help="Fraction of neurons to keep per layer",
    )
    parser.add_argument(
        "--attribution_batch_size",
        type=int,
        default=512,
        help="Batch size for attribution backward passes",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show attribution progress",
    )
    parser.add_argument(
        "--include-token-nodes",
        "--include_token_nodes",
        dest="include_token_nodes",
        action="store_true",
        default=False,
        help=(
            "Include token embedding nodes in the attribution graph adjacency matrix. "
            "By default token nodes are excluded (neuron-only graph)."
        ),
    )
    parser.add_argument(
        "--include-logit-nodes",
        "--include_logit_nodes",
        dest="include_logit_nodes",
        action="store_true",
        default=False,
        help=(
            "Include logit target nodes in the attribution graph adjacency matrix. "
            "By default logit nodes are excluded (neuron-only graph)."
        ),
    )
    parser.add_argument(
        "--anova-nodes-per-label",
        "--anova_nodes_per_label",
        dest="anova_nodes_per_label",
        type=int,
        default=10,
        help="Maximum positive-variance ANOVA nodes to include per label supernode.",
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
        help="Minimum ANOVA specificity for sum range/sum units supernodes.",
    )
    parser.add_argument(
        "--graph-node-labels",
        "--graph_node_labels",
        nargs="+",
        default=None,
        metavar="LABEL",
        dest="graph_node_labels",
        help=(
            "Whitelist of ANOVA supernode label names to include when building the "
            "supergraph. E.g. --graph-node-labels 'arg1 range' 'sum units'. "
            "If omitted, all labels are included."
        ),
    )


def resolve_torch_dtype(dtype: str) -> torch.dtype:
    dtype_mapping = {
        "fp32": "float32",
        "bf16": "bfloat16",
        "fp16": "float16",
    }
    dtype_name = dtype_mapping.get(dtype, dtype)
    return getattr(torch, dtype_name)

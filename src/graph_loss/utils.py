import argparse
import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any

import torch


DTYPE_CHOICES = ["float32", "bfloat16", "float16"]


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
        "--nodes-per-label",
        "--nodes_per_label",
        dest="nodes_per_label",
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
        "--anova-neuron-chunk",
        "--anova_neuron_chunk",
        dest="anova_neuron_chunk",
        type=int,
        default=None,
        help="Neurons processed per ANOVA batch (reduce to avoid GPU OOM on large grids; default: all at once).",
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
            "Pass 'all' to include every ANOVA label category. "
            "If omitted, no ANOVA supernodes are created."
        ),
    )



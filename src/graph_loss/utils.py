import argparse
from dataclasses import dataclass
from typing import Any

import torch


DTYPE_CHOICES = ["float32", "bfloat16", "float16", "fp32", "bf16", "fp16"]


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


def convert_nnsight_config_to_transformerlens(config):
    """Normalize a TransformerLens-style LLaMA config into `UnifiedConfig`."""

    if isinstance(config, UnifiedConfig):
        return config

    config_dict = config.to_dict() if hasattr(config, "to_dict") else vars(config)

    model_name = config_dict.get("model_name") or getattr(config, "model_name", None)
    if model_name is None:
        model_name = config_dict.get("name") or getattr(config, "name", None)
    if model_name is None:
        model_name = config_dict.get("name_or_path") or getattr(config, "name_or_path", "llama")

    tokenizer_name = config_dict.get("tokenizer_name") or getattr(config, "tokenizer_name", None)
    if tokenizer_name is None:
        tokenizer_name = config_dict.get("name_or_path") or getattr(config, "name_or_path", model_name)

    return UnifiedConfig(
        n_layers=config_dict["n_layers"],
        d_model=config_dict["d_model"],
        d_head=config_dict["d_head"],
        n_heads=config_dict["n_heads"],
        d_mlp=config_dict["d_mlp"],
        d_vocab=config_dict["d_vocab"],
        tokenizer_name=tokenizer_name,
        model_name=model_name,
        original_architecture="LlamaForCausalLM",
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
        type=int,
        default=20,
        help="If set, include exactly this many highest-probability logit nodes",
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


def add_graph_prune_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--node_threshold",
        type=float,
        default=0.8,
        help="Cumulative node influence threshold for pruning",
    )
    parser.add_argument(
        "--edge_threshold",
        type=float,
        default=0.98,
        help="Cumulative edge influence threshold for pruning",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Whether to apply pruning before building supergraph",
    )


def add_supergraph_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-3,
        help="Cosine distance threshold for supernode clustering",
    )
    parser.add_argument(
        "--min_cum_logit_influence",
        type=float,
        default=0.9,
        help="Minimum cumulative logit influence norm required to form a supernode",
    )


def resolve_torch_dtype(dtype: str) -> torch.dtype:
    dtype_mapping = {
        "fp32": "float32",
        "bf16": "bfloat16",
        "fp16": "float16",
    }
    dtype_name = dtype_mapping.get(dtype, dtype)
    return getattr(torch, dtype_name)
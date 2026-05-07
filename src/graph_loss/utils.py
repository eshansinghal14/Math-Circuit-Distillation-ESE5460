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
    activations: torch.Tensor
    w_down_vectors: torch.Tensor
    arg_values: list[list[int]]

    def __iter__(self):
        yield self.activations
        yield self.w_down_vectors


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


def resolve_torch_dtype(dtype: str) -> torch.dtype:
    dtype_mapping = {
        "fp32": "float32",
        "bf16": "bfloat16",
        "fp16": "float16",
    }
    dtype_name = dtype_mapping.get(dtype, dtype)
    return getattr(torch, dtype_name)


def safe_cache_segment(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return safe.strip("._") or "model"


def activation_write_cache_file(
    cache_root: str,
    model_name: str,
    dataset: str,
    *,
    n_layers: int,
    n_pos: int,
    d_mlp: int,
    neuron_locations: torch.Tensor | None = None,
) -> str:
    model_cache_dir = os.path.join(cache_root, safe_cache_segment(model_name))
    os.makedirs(model_cache_dir, exist_ok=True)

    key = hashlib.sha256()
    key.update(b"activation-write-cache-v4")
    key.update(dataset.encode("utf-8"))
    key.update(f"n_layers={int(n_layers)};n_pos={int(n_pos)};d_mlp={int(d_mlp)}".encode("utf-8"))
    if neuron_locations is not None:
        locations = neuron_locations.detach().cpu().to(dtype=torch.long).contiguous()
        key.update(f";kept_neurons={int(locations.shape[0])}".encode("utf-8"))
        key.update(locations.numpy().tobytes())
    return os.path.join(model_cache_dir, f"activation_write_{key.hexdigest()[:16]}.pt")


def load_activation_write_cache(
    path: str,
    expected_neuron_count: int,
) -> ActivationWriteResult:
    data = torch.load(path, weights_only=False, map_location="cpu")
    activations = data["activations"].detach().cpu()
    if activations.shape[0] != expected_neuron_count:
        raise ValueError(
            f"Cached activations contain {activations.shape[0]} neurons, "
            f"expected {expected_neuron_count}: {path}"
        )
    arg_values = data.get("arg_values")
    if arg_values is None:
        raise ValueError(f"Cached activations are missing arg_values: {path}")
    return ActivationWriteResult(
        activations=activations,
        w_down_vectors=torch.empty((expected_neuron_count, 0), dtype=torch.float32),
        arg_values=[[int(value) for value in values] for values in arg_values],
    )


def save_activation_write_cache(path: str, result: ActivationWriteResult) -> None:
    torch.save(
        {
            "activations": result.activations.detach().to(dtype=torch.float16).cpu(),
            "arg_values": result.arg_values,
        },
        path,
    )


def activation_arg_values_from_shape(activations: torch.Tensor) -> list[list[int]]:
    return [list(range(int(dim_size))) for dim_size in activations.shape[1:]]

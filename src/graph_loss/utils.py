from dataclasses import dataclass
from typing import Any


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
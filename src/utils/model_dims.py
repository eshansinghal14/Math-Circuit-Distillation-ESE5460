from .config import HF_READ_TOKEN


def mlp_flatten_dim_from_model(model) -> int:
    """Total flattened MLP width (``layers x intermediate_size``) for Llama-style causal LMs."""
    cfg = model.config
    return int(cfg.num_hidden_layers) * int(cfg.intermediate_size)


def mlp_flatten_dim_from_pretrained_id(model_id: str) -> int:
    """Same as :func:`mlp_flatten_dim_from_model` but from a HuggingFace id (no model load)."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, token=HF_READ_TOKEN or None)
    return int(cfg.num_hidden_layers) * int(cfg.intermediate_size)

"""Stop-gradient context manager for attribution-graph edge computation.

The circuit-tracing methodology computes edges with attention patterns and
normalisation denominators held fixed, so the model is locally linear in the node
activations and the gradient-activation product is exactly the direct-path
contribution.  Differentiating through the stock Hugging Face forward instead
picks up two extra terms:

* attention -- paths where the source node perturbs a query/key and shifts the
  attention pattern, i.e. changes *where* the target attends rather than what it
  reads;
* RMSNorm -- with ``r = 1/sqrt(mean(x^2) + eps)`` the true Jacobian is
  ``r*I - (r^3/d) x x^T``; freezing keeps only ``r*I``, so differentiating
  through the denominator subtracts the component along ``x`` from every edge.

Both freezes are backward-pass only.  Detaching a multiplicative factor leaves
the forward values bit-identical, so activations, read/write vectors, neuron
pre-selection and supernode membership are all unaffected -- only the edge
gradients change.

Freezing attention requires an attention implementation that materialises the
pattern.  The default SDPA kernel never forms it, so the context manager
temporarily swaps the model onto a registered eager variant that detaches the
softmax output; this is slower and uses more memory than SDPA.
"""

from __future__ import annotations

import contextlib
import logging
import types

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_FROZEN_ATTN_NAME = "graph_loss_frozen_eager"
_MISSING = object()


def _frozen_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """``eager_attention_forward`` with the attention pattern stop-gradiented.

    Values are identical to the unfrozen kernel; only the backward pass differs,
    so attention becomes a fixed linear mixing of the value vectors.
    """
    from transformers.models.llama.modeling_llama import repeat_kv

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # The freeze: the pattern is a constant w.r.t. everything upstream.
    attn_weights = attn_weights.detach()

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _register_frozen_attention() -> None:
    """Register the frozen kernel *and* its mask builder under the same name.

    Both registries must know the name. ``masking_utils`` skips mask creation
    entirely for an unrecognised implementation (it checks ``_global_mapping``
    directly), which would hand the eager kernel ``attention_mask=None`` and
    silently make attention bidirectional instead of causal. ``register`` is a
    classmethod writing to ``_global_mapping``, so plain ``__setitem__`` (which
    only updates the local mapping) would not satisfy that check.
    """
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if _FROZEN_ATTN_NAME not in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS.register(_FROZEN_ATTN_NAME, _frozen_eager_attention_forward)
    if _FROZEN_ATTN_NAME not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
        ALL_MASK_ATTENTION_FUNCTIONS.register(_FROZEN_ATTN_NAME, eager_mask)


def _frozen_rms_norm_forward(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """``LlamaRMSNorm.forward`` with the reciprocal-norm scale stop-gradiented."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    scale = torch.rsqrt(variance + module.variance_epsilon).detach()
    return module.weight * (hidden_states * scale).to(input_dtype)


def _rms_norm_modules(model: nn.Module) -> list[nn.Module]:
    """RMSNorm-shaped submodules, matched structurally rather than by class."""
    return [
        m
        for m in model.modules()
        if hasattr(m, "variance_epsilon") and hasattr(m, "weight") and not hasattr(m, "bias")
    ]


@contextlib.contextmanager
def frozen_graph_edges(
    model: nn.Module,
    *,
    freeze_attention: bool = False,
    freeze_rms_norm: bool = False,
):
    """Temporarily stop-gradient attention patterns and/or RMSNorm denominators.

    Scoped to ``model`` and to the ``with`` block, so the ordinary training
    forward/backward is untouched.  A no-op when both flags are False.
    """
    if not freeze_attention and not freeze_rms_norm:
        yield
        return

    prev_attn = _MISSING
    patched_norms: list[tuple[nn.Module, object]] = []
    try:
        if freeze_attention:
            _register_frozen_attention()
            prev_attn = model.config._attn_implementation
            model.config._attn_implementation = _FROZEN_ATTN_NAME

        if freeze_rms_norm:
            norms = _rms_norm_modules(model)
            if not norms:
                logger.warning(
                    "freeze_rms_norm=True but no RMSNorm-shaped modules were found on %s; "
                    "normalisation denominators are NOT frozen.",
                    type(model).__name__,
                )
            for m in norms:
                patched_norms.append((m, m.__dict__.get("forward", _MISSING)))
                m.forward = types.MethodType(_frozen_rms_norm_forward, m)

        yield
    finally:
        for m, prev_forward in patched_norms:
            if prev_forward is _MISSING:
                m.__dict__.pop("forward", None)
            else:
                m.forward = prev_forward
        if prev_attn is not _MISSING:
            model.config._attn_implementation = prev_attn

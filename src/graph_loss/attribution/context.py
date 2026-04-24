"""Attribution context for neuron-level graph construction."""

from __future__ import annotations

import contextlib
import weakref
from functools import partial
from typing import TYPE_CHECKING, Callable

import torch
from transformer_lens.hook_points import HookPoint

if TYPE_CHECKING:
    from graph_loss.replacement_model import TransformerLensReplacementModel


class AttributionContext:
    """Cache residual activations and build attribution rows for neuron targets."""

    def __init__(
        self,
        *,
        logits: torch.Tensor,
        token_vectors: torch.Tensor,
        neuron_locations: torch.Tensor,
        neuron_activations: torch.Tensor,
        target_encoders: torch.Tensor,
        source_vectors: torch.Tensor,
        layer_capture_stats: list[dict[str, float | int]],
    ) -> None:
        self.logits = logits
        self.token_vectors = token_vectors
        self.neuron_locations = neuron_locations
        self.neuron_activations = neuron_activations
        self.target_encoders = target_encoders
        self.source_vectors = source_vectors
        self.layer_capture_stats = layer_capture_stats

        self.n_layers = int(neuron_locations[:, 0].max().item()) + 1 if len(neuron_locations) else 0
        self.n_pos = token_vectors.shape[0]

        self._resid_activations: list[torch.Tensor | None] = [None] * (self.n_layers + 1)
        self._batch_buffer: torch.Tensor | None = None
        self._row_size = len(neuron_locations) + self.n_pos

    @property
    def source_node_count(self) -> int:
        return self._row_size

    def _caching_hooks(self, feature_input_hook: str) -> list[tuple[str, Callable]]:
        proxy = weakref.proxy(self)

        def _cache(acts: torch.Tensor, hook: HookPoint, *, layer: int) -> torch.Tensor:
            proxy._resid_activations[layer] = acts
            return acts

        hooks = [
            (f"blocks.{layer}.{feature_input_hook}", partial(_cache, layer=layer))
            for layer in range(self.n_layers)
        ]
        hooks.append(("unembed.hook_pre", partial(_cache, layer=self.n_layers)))
        return hooks

    def _compute_score_hook(
        self,
        hook_name: str,
        output_vecs: torch.Tensor,
        positions: torch.Tensor,
        write_index: torch.Tensor | slice,
    ) -> tuple[str, Callable]:
        proxy = weakref.proxy(self)

        def _hook_fn(grads: torch.Tensor, hook: HookPoint) -> None:
            # Avoid allocating huge grads[:, positions] tensor to prevent CUDA OOM
            scores = torch.empty(
                grads.shape[0], 
                len(positions), 
                device=grads.device, 
                dtype=output_vecs.dtype
            )
            for p in torch.unique(positions):
                mask = positions == p
                grad_p = grads[:, p, :].to(output_vecs.dtype)
                out_p = output_vecs[mask]
                scores[:, mask] = torch.matmul(grad_p, out_p.T)
            
            proxy._batch_buffer[write_index] += scores

        return hook_name, _hook_fn

    def _make_attribution_hooks(self, feature_output_hook: str) -> list[tuple[str, Callable]]:
        source_layers = self.neuron_locations[:, 0]
        source_positions = self.neuron_locations[:, 1]

        neuron_hooks = []
        for layer in range(self.n_layers):
            layer_mask = source_layers == layer
            if not layer_mask.any():
                continue

            row_indices = torch.where(layer_mask)[0]
            neuron_hooks.append(
                self._compute_score_hook(
                    f"blocks.{layer}.{feature_output_hook}",
                    self.source_vectors[row_indices],
                    source_positions[row_indices],
                    row_indices,
                )
            )

        token_start = len(self.neuron_locations)
        token_positions = torch.arange(self.n_pos, device=self.token_vectors.device, dtype=torch.long)
        token_hook = [
            self._compute_score_hook(
                "hook_embed",
                self.token_vectors,
                token_positions,
                slice(token_start, token_start + self.n_pos),
            )
        ]
        return neuron_hooks + token_hook

    @contextlib.contextmanager
    def install_hooks(self, model: "TransformerLensReplacementModel"):
        with model.hooks(
            fwd_hooks=self._caching_hooks(model.feature_input_hook),  # type: ignore[arg-type]
            bwd_hooks=self._make_attribution_hooks(model.feature_output_hook),  # type: ignore[arg-type]
        ):
            yield

    def compute_batch(
        self,
        layers: torch.Tensor,
        positions: torch.Tensor,
        inject_values: torch.Tensor,
        retain_graph: bool = True,
    ) -> torch.Tensor:
        batch_size = self._resid_activations[0].shape[0]  # type: ignore[index]
        self._batch_buffer = torch.zeros(
            self._row_size,
            batch_size,
            dtype=inject_values.dtype,
            device=inject_values.device,
        )

        batch_idx = torch.arange(len(layers), device=layers.device)

        def _inject(grads, *, batch_indices, pos_indices, values):
            grads_out = grads.clone().to(values.dtype)
            grads_out.index_put_((batch_indices, pos_indices), values)
            return grads_out.to(grads.dtype)

        handles = []
        layers_in_batch = layers.unique().tolist()

        for layer in layers_in_batch:
            layer_idx = int(layer)
            mask = layers == layer_idx
            if not mask.any():
                continue
            fn = partial(
                _inject,
                batch_indices=batch_idx[mask],
                pos_indices=positions[mask],
                values=inject_values[mask],
            )
            handles.append(self._resid_activations[layer_idx].register_hook(fn))  # type: ignore[union-attr]

        try:
            last_layer = max(int(layer) for layer in layers_in_batch)
            self._resid_activations[last_layer].backward(  # type: ignore[union-attr]
                gradient=torch.zeros_like(self._resid_activations[last_layer]),  # type: ignore[arg-type]
                retain_graph=retain_graph,
            )
        finally:
            for handle in handles:
                handle.remove()

        buf, self._batch_buffer = self._batch_buffer, None
        return buf.T[: len(layers)]

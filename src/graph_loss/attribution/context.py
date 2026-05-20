"""HF adapter attribution context — precomputed state for the graph pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from graph_loss.hf_adapter import HFLlamaGraphAdapter


class HFAttributionContext:
    """Holds precomputed neuron data from the initial forward pass.

    Analogous to the TL ``AttributionContext`` but built from
    ``HFLlamaGraphAdapter`` using PyTorch native hooks.  Created by
    ``setup_attribution`` in ``attribution.attribute``; consumed by
    ``attribute`` to run the gradient-based edge scoring phases.
    """

    def __init__(
        self,
        adapter: "HFLlamaGraphAdapter",
        input_ids: torch.Tensor,
        neuron_locations: torch.Tensor,
        neuron_activations: torch.Tensor,
        target_encoders: torch.Tensor,
        source_vectors: torch.Tensor,
        source_layer_by_node: torch.Tensor,
        embed_out: torch.Tensor,
        logits: torch.Tensor,
        dtype: torch.dtype | None = None,
    ):
        self.adapter = adapter
        self.input_ids = input_ids
        self.neuron_locations = neuron_locations
        self.neuron_activations = neuron_activations
        self.target_encoders = target_encoders
        self.source_vectors = source_vectors
        self.source_layer_by_node = source_layer_by_node
        self.embed_out = embed_out
        self.logits = logits
        self.dtype = dtype

    @property
    def n_neurons(self) -> int:
        return int(self.neuron_locations.shape[0])

    @property
    def n_tokens(self) -> int:
        return int(self.input_ids.numel())

    @property
    def source_count(self) -> int:
        return self.n_neurons + self.n_tokens

    def _expanded_forward(self, batch_len: int):
        """Run a batch-expanded forward pass for gradient-based attribution.

        Returns ``(chunk_out, chunk_mlp_inputs, chunk_mlp_outputs, chunk_embed)``
        with ``requires_grad=True`` set on embed and each MLP output so that
        ``torch.autograd.grad`` can differentiate back to those nodes.
        """
        adapter = self.adapter
        chunk_mlp_inputs: dict[int, torch.Tensor] = {}
        chunk_mlp_outputs: dict[int, torch.Tensor] = {}
        chunk_embed_out: torch.Tensor | None = None
        chunk_handles = []

        def chunk_embed_hook(_module, _inputs, output):
            nonlocal chunk_embed_out
            if not output.requires_grad:
                output = output.detach().requires_grad_(True)
            chunk_embed_out = output
            return output

        chunk_handles.append(adapter.embed_tokens.register_forward_hook(chunk_embed_hook))

        for layer_idx, layer in enumerate(adapter.layers):
            def _pre(m, inputs, *, idx=layer_idx):
                chunk_mlp_inputs[idx] = inputs[0]

            def _out(m, _inputs, output, *, idx=layer_idx):
                if not output.requires_grad:
                    output = output.detach().requires_grad_(True)
                chunk_mlp_outputs[idx] = output
                return output

            chunk_handles.append(layer.mlp.register_forward_pre_hook(_pre))
            chunk_handles.append(layer.mlp.register_forward_hook(_out))

        try:
            with adapter.autocast_context(self.dtype):
                chunk_out = adapter.model(
                    input_ids=self.input_ids.expand(batch_len, -1),
                    attention_mask=torch.ones(
                        batch_len,
                        self.input_ids.numel(),
                        dtype=torch.long,
                        device=adapter.device,
                    ),
                    output_hidden_states=True,
                    use_cache=False,
                )
        finally:
            for h in chunk_handles:
                h.remove()

        if chunk_embed_out is None:
            raise RuntimeError("Embedding hook did not capture chunk token embeddings.")
        if len(chunk_mlp_inputs) != adapter.n_layers or len(chunk_mlp_outputs) != adapter.n_layers:
            raise RuntimeError(
                f"Expected {adapter.n_layers} chunk layer captures, got "
                f"{len(chunk_mlp_inputs)} inputs and {len(chunk_mlp_outputs)} outputs."
            )
        return chunk_out, chunk_mlp_inputs, chunk_mlp_outputs, chunk_embed_out

    def _source_scores_from_grads(
        self, grads: tuple[torch.Tensor | None, ...], batch_len: int
    ) -> torch.Tensor:
        """Convert a grad tuple from ``torch.autograd.grad`` into a
        ``[batch_len, source_count]`` attribution score matrix.

        ``grads`` layout: one entry per MLP layer (index 0..n_layers-1) then
        the embedding grad at index n_layers.
        """
        adapter = self.adapter
        rows = torch.zeros(
            batch_len,
            self.source_count,
            dtype=self.source_vectors.dtype,
            device=adapter.device,
        )
        for layer_idx in range(adapter.n_layers):
            layer_mask = self.source_layer_by_node == layer_idx
            if not layer_mask.any():
                continue
            grad = grads[layer_idx]
            if grad is None:
                continue
            layer_indices = torch.where(layer_mask)[0]
            locs = self.neuron_locations[layer_mask]
            grad_vecs = grad[:, locs[:, 1], :].to(self.source_vectors.dtype)
            scores = (
                grad_vecs * self.source_vectors[layer_indices].unsqueeze(0)
            ).sum(dim=-1)
            col_indices = layer_indices.unsqueeze(0).expand(batch_len, -1)
            rows = rows.scatter_add(1, col_indices, scores)

        token_grad = grads[adapter.n_layers]
        if token_grad is not None:
            token_vectors = self.embed_out.squeeze(0).to(self.source_vectors.dtype)
            token_scores = (
                token_grad.to(self.source_vectors.dtype) * token_vectors.unsqueeze(0)
            ).sum(dim=-1)
            token_indices = torch.arange(
                self.n_neurons,
                self.source_count,
                device=adapter.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(batch_len, -1)
            rows = rows.scatter_add(1, token_indices, token_scores)

        return rows

    def compute_neuron_batch(
        self, start: int, end: int, *, create_graph: bool
    ) -> torch.Tensor:
        """Return ``[batch_len, source_count]`` attribution rows for neuron
        targets in ``[start, end)``.

        Runs one expanded forward pass, then back-differentiates the
        target-encoder dot-product for each row to get edge scores.
        """
        batch_len = end - start
        _, chunk_mlp_inputs, chunk_mlp_outputs, chunk_embed = self._expanded_forward(batch_len)
        terms = []
        for batch_idx, row_idx in enumerate(range(start, end)):
            layer_idx = int(self.neuron_locations[row_idx, 0].item())
            pos_idx = int(self.neuron_locations[row_idx, 1].item())
            terms.append(
                (
                    chunk_mlp_inputs[layer_idx][batch_idx, pos_idx].to(self.target_encoders.dtype)
                    * self.target_encoders[row_idx].detach()
                ).sum()
            )
        source_tensors = [chunk_mlp_outputs[i] for i in range(self.adapter.n_layers)] + [chunk_embed]
        grads = torch.autograd.grad(
            torch.stack(terms).sum(),
            source_tensors,
            retain_graph=False,
            create_graph=create_graph,
            allow_unused=True,
        )
        return self._source_scores_from_grads(grads, batch_len)

    def compute_logit_batch(
        self,
        start: int,
        end: int,
        logit_vectors: torch.Tensor,
        *,
        create_graph: bool,
    ) -> torch.Tensor:
        """Return ``[batch_len, source_count]`` attribution rows for logit
        targets in ``[start, end)``.
        """
        batch_len = end - start
        chunk_out, _chunk_mlp_inputs, chunk_mlp_outputs, chunk_embed = self._expanded_forward(batch_len)
        lv = logit_vectors[start:end].to(
            device=self.adapter.device,
            dtype=chunk_out.hidden_states[-1].dtype,
        )
        terms = (chunk_out.hidden_states[-1][:, -1, :] * lv.detach()).sum(dim=-1)
        source_tensors = [chunk_mlp_outputs[i] for i in range(self.adapter.n_layers)] + [chunk_embed]
        grads = torch.autograd.grad(
            terms.sum(),
            source_tensors,
            retain_graph=False,
            create_graph=create_graph,
            allow_unused=True,
        )
        return self._source_scores_from_grads(grads, batch_len)

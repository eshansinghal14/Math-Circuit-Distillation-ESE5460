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

    def filter(self, keep_mask: torch.Tensor) -> "HFAttributionContext":
        """Return a new context containing only neurons where keep_mask is True."""
        keep_mask = keep_mask.to(device=self.neuron_locations.device, dtype=torch.bool)
        return HFAttributionContext(
            adapter=self.adapter,
            input_ids=self.input_ids,
            neuron_locations=self.neuron_locations[keep_mask],
            neuron_activations=self.neuron_activations[keep_mask],
            target_encoders=self.target_encoders[keep_mask],
            source_vectors=self.source_vectors[keep_mask],
            source_layer_by_node=self.source_layer_by_node[keep_mask],
            embed_out=self.embed_out,
            logits=self.logits,
            dtype=self.dtype,
        )

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

    def compute_embedding_grad_norms_batch(self, start: int, end: int) -> torch.Tensor:
        """Return ``[batch_len, n_pos]`` per-position L2-norms of ∂f/∂E[p].

        For each neuron ``row_idx`` in ``[start, end)``, treats the scalar
        objective ``f = mlp_input[L][pos] @ target_encoder[row_idx]`` (the
        neuron's linearised pre-activation), back-differentiates to the token
        embedding output ``chunk_embed``, and returns the L2-norm of the
        resulting gradient at each sequence position.

        Each batch item's MLP inputs depend only on ``chunk_embed[batch_idx, :]``
        (standard Transformer batching), so ``token_grad[batch_idx, q, :]`` is
        exactly ``∂f_{neuron_{batch_idx}}/∂E[q]`` — no cross-batch contamination.

        Used by ``select_arg_supernodes`` to identify which token positions most
        drive each neuron's activation.
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
            create_graph=False,
            allow_unused=True,
        )

        # Embedding gradient is the last element: [batch_len, seq_len, d_model]
        token_grad = grads[self.adapter.n_layers]
        if token_grad is None:
            return torch.zeros(
                batch_len, self.n_tokens, dtype=torch.float32, device=self.adapter.device
            )

        # Per-position L2 norm: [batch_len, seq_len]
        return token_grad.detach().float().norm(dim=-1)

    def compute_embedding_grad_norms_fast(self) -> torch.Tensor:
        """Single-forward grouped-VJP replacement for compute_embedding_grad_norms_batch.

        Runs ``_expanded_forward(1)`` exactly once, then groups all neurons by
        ``(layer_idx, pos_idx)`` and issues one batched ``torch.autograd.grad``
        call per group using ``is_grads_batched=True``.  This avoids the ~26
        redundant forward passes that the chunk loop would otherwise perform.

        When the model is in training mode (parameters require grad), we
        temporarily freeze all parameters so the autograd graph only spans
        the activation path (embed → mlp_inputs).  This matches the frozen-
        teacher case and avoids retaining the full training graph across the
        ~80 grouped VJP calls.

        Returns:
            all_norms: ``[n_neurons, n_pos]`` float32 tensor on ``adapter.device``.
                       Numerically identical to the result of the chunked loop.
        """
        device = self.adapter.device
        n_neurons = self.n_neurons
        n_pos = self.n_tokens

        # Temporarily freeze model parameters so the autograd graph is thin
        # (activation-only: embed → mlp_inputs, no weight gradient nodes).
        params_requiring_grad = [
            p for p in self.adapter.model.parameters() if p.requires_grad
        ]
        for p in params_requiring_grad:
            p.requires_grad_(False)
        try:
            _, chunk_mlp_inputs, _, chunk_embed = self._expanded_forward(1)
        finally:
            for p in params_requiring_grad:
                p.requires_grad_(True)

        all_norms = torch.zeros(n_neurons, n_pos, dtype=torch.float32, device=device)

        # Build groups: (layer_idx, pos_idx) -> [neuron_idx, ...]
        groups: dict[tuple[int, int], list[int]] = {}
        for neuron_idx in range(n_neurons):
            layer_idx = int(self.neuron_locations[neuron_idx, 0].item())
            pos_idx = int(self.neuron_locations[neuron_idx, 1].item())
            key = (layer_idx, pos_idx)
            if key not in groups:
                groups[key] = []
            groups[key].append(neuron_idx)

        group_list = list(groups.items())

        for group_num, ((layer_idx, pos_idx), group_idxs) in enumerate(group_list):
            is_last = group_num == len(group_list) - 1

            # outputs: [d_mlp] — single vector shared by all k neurons in this group
            outputs = chunk_mlp_inputs[layer_idx][0, pos_idx, :]
            # grad_outputs: [k, d_mlp] — one encoder per neuron
            grad_outputs = self.target_encoders[group_idxs].detach()

            grads = torch.autograd.grad(
                outputs,
                chunk_embed,
                grad_outputs=grad_outputs,
                is_grads_batched=True,
                retain_graph=not is_last,
                create_graph=False,
                allow_unused=True,
            )

            embed_grad = grads[0]
            if embed_grad is None:
                # rows for group_idxs remain zeros (already initialised)
                continue

            # embed_grad: [k, 1, seq_len, d_model] -> [k, seq_len, d_model] -> [k, seq_len]
            norms = embed_grad.squeeze(1).float().norm(dim=-1)
            all_norms[group_idxs, :] = norms.to(device=device)

        return all_norms

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

"""Hugging Face LLaMA graph attribution helpers used during distillation."""

from __future__ import annotations

import contextlib
from typing import Any

import torch
import torch.nn.functional as F

from graph_loss.attribution.targets import AttributionTargets, TargetSpec
from graph_loss.graph import Graph


class HFLlamaGraphAdapter:
    """Small adapter that exposes graph attribution over a HF LLaMA model.

    This intentionally mirrors the TransformerLens graph path at the tensor level
    instead of converting the training model to TransformerLens, so student graph
    loss can backpropagate into the existing HF model used by distillation.
    """

    def __init__(self, model, tokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.layers = model.model.layers
        self.embed_tokens = model.model.embed_tokens
        self.lm_head = model.lm_head
        self.config = model.config

    @property
    def n_layers(self) -> int:
        return int(self.config.num_hidden_layers)

    @property
    def d_model(self) -> int:
        return int(self.config.hidden_size)

    @property
    def d_mlp(self) -> int:
        return int(self.config.intermediate_size)

    @property
    def d_vocab(self) -> int:
        return int(self.config.vocab_size)

    @property
    def W_U(self) -> torch.Tensor:
        return self.lm_head.weight.T

    @property
    def cfg(self) -> "_HFGraphConfig":
        """Lazy TransformerLens-shaped config shim.

        ``graph.py`` paths inherited from the TL pipeline read ``model.cfg.device``,
        ``model.cfg.d_model``, etc.  Building one ``_HFGraphConfig`` and caching it
        on first access lets that code work unchanged against this HF adapter.
        """
        cached = getattr(self, "_cfg_cache", None)
        if cached is None:
            cached = _HFGraphConfig(self)
            self._cfg_cache = cached
        return cached

    @property
    def blocks(self) -> "_HFBlockList":
        """TransformerLens-shaped block list shim.

        ``graph.py``'s full_search path reads ``model.blocks[layer].mlp.old_mlp.W_out``
        to get the down-projection weight for each MLP layer.  This shim lazily wraps
        the HF ``model.layers`` so those accesses work unchanged.
        """
        cached = getattr(self, "_blocks_cache", None)
        if cached is None:
            cached = _HFBlockList(self.layers)
            self._blocks_cache = cached
        return cached

    @property
    def unembed(self) -> "_HFUnembed":
        """TransformerLens-shaped unembed shim.

        ``graph.py``'s full_search path reads ``model.unembed.W_U``.
        Returns a lightweight wrapper around ``lm_head.weight.T``.
        """
        cached = getattr(self, "_unembed_cache", None)
        if cached is None:
            cached = _HFUnembed(self.lm_head)
            self._unembed_cache = cached
        return cached

    def autocast_context(self, dtype: torch.dtype | None):
        if dtype is None or self.device.type != "cuda":
            return contextlib.nullcontext()
        if dtype not in (torch.float16, torch.bfloat16):
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def ensure_tokenized(self, prompt: str | torch.Tensor | list[int]) -> torch.Tensor:
        if isinstance(prompt, str):
            tokens = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"].squeeze(0)
        elif isinstance(prompt, torch.Tensor):
            tokens = prompt.squeeze()
        elif isinstance(prompt, list):
            tokens = torch.tensor(prompt, dtype=torch.long).squeeze()
        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt)}")
        if tokens.ndim != 1:
            raise ValueError(f"Prompt tokens must be 1-D, got {tuple(tokens.shape)}")
        if tokens.numel() == 0:
            raise ValueError(
                "Tokenizer produced 0 tokens from the prompt. "
                "Ensure the prompt is non-empty and contains tokenizable text."
            )
        return tokens.to(self.device)

    @staticmethod
    def _row_oriented_weight(
        weight: torch.Tensor, rows: int | None = None, cols: int | None = None
    ) -> torch.Tensor:
        """Return weight in (rows, cols) orientation.

        When ``rows``/``cols`` are omitted (graph.py's ``full_search`` path passes
        only the weight), the matrix is returned as-is — the caller already loaded
        ``down_proj.weight`` which is (d_model, d_mlp) in HF, so ``_HFOldMLP``
        exposes it as ``W_out`` and this call is a no-op identity.
        """
        if rows is None or cols is None:
            return weight
        if weight.shape == (rows, cols):
            return weight
        if weight.shape == (cols, rows):
            return weight.T
        raise ValueError(
            f"Unsupported weight shape {tuple(weight.shape)} for row shape {(rows, cols)}",
        )

    @staticmethod
    def _silu_derivative(x: torch.Tensor) -> torch.Tensor:
        sigma = torch.sigmoid(x)
        return sigma + x * sigma * (1 - sigma)

    def _layer_weights(self, layer_idx: int, *, device, dtype):
        mlp = self.layers[layer_idx].mlp
        gate_rows = self._row_oriented_weight(
            mlp.gate_proj.weight.to(device=device, dtype=dtype),
            self.d_mlp,
            self.d_model,
        )
        up_rows = self._row_oriented_weight(
            mlp.up_proj.weight.to(device=device, dtype=dtype),
            self.d_mlp,
            self.d_model,
        )
        out_rows = self._row_oriented_weight(
            mlp.down_proj.weight.to(device=device, dtype=dtype),
            self.d_mlp,
            self.d_model,
        )
        gate_bias = getattr(mlp.gate_proj, "bias", None)
        up_bias = getattr(mlp.up_proj, "bias", None)
        if gate_bias is None:
            gate_bias = torch.zeros(self.d_mlp, device=device, dtype=dtype)
        else:
            gate_bias = gate_bias.to(device=device, dtype=dtype)
        if up_bias is None:
            up_bias = torch.zeros(self.d_mlp, device=device, dtype=dtype)
        else:
            up_bias = up_bias.to(device=device, dtype=dtype)
        return gate_rows, up_rows, out_rows, gate_bias, up_bias

    def _compute_layer_neuron_data(
        self,
        layer_idx: int,
        mlp_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_rows, up_rows, out_rows, gate_bias, up_bias = self._layer_weights(
            layer_idx,
            device=mlp_input.device,
            dtype=mlp_input.dtype,
        )
        gate_pre = mlp_input @ gate_rows.T + gate_bias
        up_pre = mlp_input @ up_rows.T + up_bias
        gate_act = F.silu(gate_pre)
        neuron_activations = gate_act * up_pre
        gate_grad = self._silu_derivative(gate_pre)
        target_encoders = (
            gate_act.unsqueeze(-1) * up_rows.unsqueeze(0)
            + (up_pre * gate_grad).unsqueeze(-1) * gate_rows.unsqueeze(0)
        )
        source_vectors = neuron_activations.unsqueeze(-1) * out_rows.unsqueeze(0)
        return neuron_activations, target_encoders, source_vectors

    def build_graph(
        self,
        prompt: str | torch.Tensor | list[int],
        *,
        attribution_targets: list[str] | list[TargetSpec] | torch.Tensor | None = None,
        top_k_logits: int | None = 20,
        prop_neurons_per_layer: float = 0.1,
        batch_size: int = 512,
        dtype: torch.dtype | None = None,
        verbose: bool = False,
        create_graph: bool = False,
        detach_result: bool | None = None,
        fast: bool = False,
        skip_logit_attribution: bool = False,
    ) -> Graph:
        if not (0.0 < prop_neurons_per_layer <= 1.0):
            raise ValueError("prop_neurons_per_layer must be in (0, 1]")
        input_ids = self.ensure_tokenized(prompt)
        input_batch = input_ids.unsqueeze(0)
        mlp_inputs: dict[int, torch.Tensor] = {}
        mlp_outputs: dict[int, torch.Tensor] = {}
        embed_out: torch.Tensor | None = None
        handles = []

        def embed_hook(_module, _inputs, output):
            nonlocal embed_out
            if fast:
                embed_out = output
            else:
                if not output.requires_grad:
                    output = output.detach().requires_grad_(True)
                embed_out = output
            return output

        handles.append(self.embed_tokens.register_forward_hook(embed_hook))

        for layer_idx, layer in enumerate(self.layers):
            def pre_hook(_module, inputs, *, idx=layer_idx):
                mlp_inputs[idx] = inputs[0]

            def out_hook(_module, _inputs, output, *, idx=layer_idx):
                if fast:
                    mlp_outputs[idx] = output
                else:
                    if not output.requires_grad:
                        output = output.detach().requires_grad_(True)
                    mlp_outputs[idx] = output
                return output

            handles.append(layer.mlp.register_forward_pre_hook(pre_hook))
            handles.append(layer.mlp.register_forward_hook(out_hook))

        try:
            # Use no_grad (not inference_mode): activations must be usable in later matmuls
            # with parameter tensors that may still have requires_grad=True.
            fwd_ctx = torch.no_grad() if fast else contextlib.nullcontext()
            with fwd_ctx, self.autocast_context(dtype):
                out = self.model(
                    input_ids=input_batch,
                    attention_mask=torch.ones_like(input_batch, device=self.device),
                    output_hidden_states=True,
                    use_cache=False,
                )
        finally:
            for handle in handles:
                handle.remove()

        if embed_out is None:
            raise RuntimeError("Embedding hook did not capture token embeddings.")
        if len(mlp_inputs) != self.n_layers or len(mlp_outputs) != self.n_layers:
            raise RuntimeError(
                f"Expected {self.n_layers} layer captures, got "
                f"{len(mlp_inputs)} inputs and {len(mlp_outputs)} outputs.",
            )

        logits = out.logits
        final_hidden = out.hidden_states[-1]
        n_pos = int(input_ids.numel())
        positions = torch.arange(n_pos, device=self.device, dtype=torch.long)
        neuron_ids = torch.arange(self.d_mlp, device=self.device, dtype=torch.long)

        neuron_locations = []
        neuron_activations = []
        target_encoders = []
        source_vectors = []
        source_layer_by_node = []

        for layer_idx in range(self.n_layers):
            layer_input = mlp_inputs[layer_idx].squeeze(0)
            layer_acts, layer_target_encoders, layer_source_vectors = (
                self._compute_layer_neuron_data(layer_idx, layer_input)
            )
            flat_source_norms = layer_source_vectors.norm(dim=-1).reshape(-1)
            k = max(1, int(flat_source_norms.numel() * prop_neurons_per_layer))
            k = min(k, flat_source_norms.numel())
            keep = torch.topk(flat_source_norms, k, dim=0).indices
            layer_locations = torch.stack(
                [
                    torch.full((n_pos * self.d_mlp,), layer_idx, device=self.device, dtype=torch.long),
                    positions.repeat_interleave(self.d_mlp),
                    neuron_ids.repeat(n_pos),
                ],
                dim=1,
            )
            neuron_locations.append(layer_locations[keep])
            neuron_activations.append(layer_acts.reshape(-1)[keep])
            target_encoders.append(layer_target_encoders.reshape(-1, self.d_model)[keep])
            source_vectors.append(layer_source_vectors.reshape(-1, self.d_model)[keep])
            source_layer_by_node.extend([layer_idx] * int(keep.numel()))

        neuron_locations_t = torch.cat(neuron_locations, dim=0)
        neuron_activations_t = torch.cat(neuron_activations, dim=0)
        target_encoders_t = torch.cat(target_encoders, dim=0)
        source_vectors_t = torch.cat(source_vectors, dim=0)
        source_layer_by_node_t = torch.tensor(source_layer_by_node, device=self.device, dtype=torch.long)

        targets = AttributionTargets(
            attribution_targets=attribution_targets,
            logits=logits[0, -1],
            unembed_proj=self.W_U.to(dtype=dtype) if dtype is not None else self.W_U,
            tokenizer=self.tokenizer,
            top_k_logits=top_k_logits,
        )

        n_neurons = int(neuron_locations_t.shape[0])
        n_tokens = n_pos
        n_logits = len(targets)
        total_nodes = n_neurons + n_tokens + n_logits
        source_count = n_neurons + n_tokens
        logit_start = n_neurons + n_tokens

        if fast:
            adjacency = torch.zeros(
                total_nodes,
                total_nodes,
                dtype=source_vectors_t.dtype,
                device=self.device,
            )
            lv = targets.logit_vectors.to(
                device=self.device,
                dtype=source_vectors_t.dtype,
            )
            adjacency[logit_start : logit_start + n_logits, :n_neurons] = lv @ source_vectors_t.T
            graph = Graph(
                input_string=self.tokenizer.decode(input_ids.detach().cpu().tolist()),
                input_tokens=input_ids,
                neuron_locations=neuron_locations_t,
                adjacency_matrix=adjacency,
                cfg=_HFGraphConfig(self),
                neuron_activations=neuron_activations_t,
                logit_targets=targets.logit_targets,
                logit_probabilities=targets.logit_probabilities,
                vocab_size=targets.vocab_size,
                attribution_mode="fast",
                neuron_write_vectors=source_vectors_t.clone(),
            )
            if detach_result is None:
                detach_result = not create_graph
            if detach_result:
                graph = detach_graph(graph)
            return graph

        def _source_scores_from_grads(grads, batch_len: int) -> torch.Tensor:
            # Build rows differentiably via scatter_add (out-of-place) so the
            # gradient through source_vectors_t / embed_out propagates back to
            # the student model parameters.  In-place setitem on a freshly-zeroed
            # leaf tensor silently drops grad tracking, which previously made the
            # edge loss gradient identically zero.
            rows = torch.zeros(
                batch_len,
                source_count,
                dtype=source_vectors_t.dtype,
                device=self.device,
            )
            for layer_idx in range(self.n_layers):
                layer_mask = source_layer_by_node_t == layer_idx
                if not layer_mask.any():
                    continue
                grad = grads[layer_idx]
                if grad is None:
                    continue
                layer_indices = torch.where(layer_mask)[0]
                locs = neuron_locations_t[layer_mask]
                grad_vecs = grad[:, locs[:, 1], :].to(source_vectors_t.dtype)
                scores = (
                    grad_vecs * source_vectors_t[layer_indices].unsqueeze(0)
                ).sum(dim=-1)  # [batch_len, n_layer_neurons]
                col_indices = layer_indices.unsqueeze(0).expand(batch_len, -1)
                rows = rows.scatter_add(1, col_indices, scores)
            token_grad = grads[-1]
            if token_grad is not None:
                token_vectors = embed_out.squeeze(0).to(source_vectors_t.dtype)
                token_scores = (
                    token_grad.to(source_vectors_t.dtype) * token_vectors.unsqueeze(0)
                ).sum(dim=-1)  # [batch_len, n_pos]
                token_indices = torch.arange(
                    n_neurons,
                    source_count,
                    device=self.device,
                    dtype=torch.long,
                ).unsqueeze(0).expand(batch_len, -1)
                rows = rows.scatter_add(1, token_indices, token_scores)
            return rows

        def _expanded_forward(batch_len: int):
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

            chunk_handles.append(self.embed_tokens.register_forward_hook(chunk_embed_hook))
            for layer_idx, layer in enumerate(self.layers):
                def pre_hook(_module, inputs, *, idx=layer_idx):
                    chunk_mlp_inputs[idx] = inputs[0]

                def out_hook(_module, _inputs, output, *, idx=layer_idx):
                    if not output.requires_grad:
                        output = output.detach().requires_grad_(True)
                    chunk_mlp_outputs[idx] = output
                    return output

                chunk_handles.append(layer.mlp.register_forward_pre_hook(pre_hook))
                chunk_handles.append(layer.mlp.register_forward_hook(out_hook))

            try:
                with self.autocast_context(dtype):
                    chunk_out = self.model(
                        input_ids=input_ids.expand(batch_len, -1),
                        attention_mask=torch.ones(
                            batch_len,
                            input_ids.numel(),
                            dtype=torch.long,
                            device=self.device,
                        ),
                        output_hidden_states=True,
                        use_cache=False,
                    )
            finally:
                for handle in chunk_handles:
                    handle.remove()

            if chunk_embed_out is None:
                raise RuntimeError("Embedding hook did not capture chunk token embeddings.")
            if len(chunk_mlp_inputs) != self.n_layers or len(chunk_mlp_outputs) != self.n_layers:
                raise RuntimeError(
                    f"Expected {self.n_layers} chunk layer captures, got "
                    f"{len(chunk_mlp_inputs)} inputs and {len(chunk_mlp_outputs)} outputs.",
                )
            return chunk_out, chunk_mlp_inputs, chunk_mlp_outputs, chunk_embed_out


        def source_scores_neuron_chunk(start_idx: int, end_idx: int) -> torch.Tensor:
            batch_len = end_idx - start_idx
            _, chunk_mlp_inputs, chunk_mlp_outputs, chunk_embed = _expanded_forward(batch_len)
            terms = []
            for batch_idx, row_idx in enumerate(range(start_idx, end_idx)):
                layer_idx = int(neuron_locations_t[row_idx, 0].item())
                pos_idx = int(neuron_locations_t[row_idx, 1].item())
                terms.append((
                    chunk_mlp_inputs[layer_idx][batch_idx, pos_idx].to(target_encoders_t.dtype)
                    * target_encoders_t[row_idx].detach()
                ).sum())
            source_tensors = [chunk_mlp_outputs[idx] for idx in range(self.n_layers)] + [chunk_embed]
            grads = torch.autograd.grad(
                torch.stack(terms).sum(),
                source_tensors,
                retain_graph=False,
                create_graph=create_graph,
                allow_unused=True,
            )
            return _source_scores_from_grads(grads, batch_len)

        def source_scores_logit_chunk(start_idx: int, end_idx: int) -> torch.Tensor:
            batch_len = end_idx - start_idx
            chunk_out, _chunk_mlp_inputs, chunk_mlp_outputs, chunk_embed = _expanded_forward(batch_len)
            logit_vecs = targets.logit_vectors[start_idx:end_idx].to(
                device=self.device,
                dtype=chunk_out.hidden_states[-1].dtype,
            )
            terms = (
                chunk_out.hidden_states[-1][:, -1, :] * logit_vecs.detach()
            ).sum(dim=-1)
            source_tensors = [chunk_mlp_outputs[idx] for idx in range(self.n_layers)] + [chunk_embed]
            grads = torch.autograd.grad(
                terms.sum(),
                source_tensors,
                retain_graph=False,
                create_graph=create_graph,
                allow_unused=True,
            )
            return _source_scores_from_grads(grads, batch_len)

        # Build adjacency via cat (out-of-place) so gradients from
        # source_scores_*_chunk → source_vectors_t → down_proj.weight propagate
        # back to model parameters.  In-place setitem on a torch.zeros leaf
        # tensor silently breaks the autograd chain, which is what made the
        # edge loss gradient identically zero in earlier runs.
        neuron_row_chunks = []
        for start in range(0, n_neurons, max(1, batch_size)):
            end = min(start + max(1, batch_size), n_neurons)
            if verbose:
                print(f"    [graph] neuron rows {start}:{end} / {n_neurons}")
            neuron_row_chunks.append(source_scores_neuron_chunk(start, end))
        if neuron_row_chunks:
            neuron_rows = torch.cat(neuron_row_chunks, dim=0)
        else:
            neuron_rows = torch.zeros(
                0, source_count, dtype=source_vectors_t.dtype, device=self.device,
            )

        if not skip_logit_attribution:
            logit_row_chunks = []
            for start in range(0, n_logits, max(1, batch_size)):
                end = min(start + max(1, batch_size), n_logits)
                if verbose:
                    print(f"    [graph] logit rows {start}:{end} / {n_logits}")
                logit_row_chunks.append(source_scores_logit_chunk(start, end))
            if logit_row_chunks:
                logit_rows_partial = torch.cat(logit_row_chunks, dim=0)
            else:
                logit_rows_partial = torch.zeros(
                    0, source_count, dtype=source_vectors_t.dtype, device=self.device,
                )
        else:
            logit_rows_partial = torch.zeros(
                n_logits, source_count, dtype=source_vectors_t.dtype, device=self.device,
            )

        # Pad neuron and logit row blocks (which span source_count cols) out to
        # total_nodes cols by concatenating zero columns for the logit-target columns.
        n_logit_cols = total_nodes - source_count
        if n_logit_cols > 0:
            neuron_rows = torch.cat(
                [
                    neuron_rows,
                    torch.zeros(
                        n_neurons,
                        n_logit_cols,
                        dtype=neuron_rows.dtype,
                        device=self.device,
                    ),
                ],
                dim=1,
            )
            logit_rows = torch.cat(
                [
                    logit_rows_partial,
                    torch.zeros(
                        n_logits,
                        n_logit_cols,
                        dtype=logit_rows_partial.dtype,
                        device=self.device,
                    ),
                ],
                dim=1,
            )
        else:
            logit_rows = logit_rows_partial

        # Token rows have no outgoing-from-tokens entries in this adjacency layout.
        token_rows = torch.zeros(
            n_tokens,
            total_nodes,
            dtype=neuron_rows.dtype,
            device=self.device,
        )

        adjacency = torch.cat([neuron_rows, token_rows, logit_rows], dim=0)

        # Always populate neuron_write_vectors in full mode: this lets the
        # supergraph clustering pre-populate prob_delta_by_neuron via DLA
        # (write_vector @ W_U_logits) instead of needing the adjacency
        # [logits, neurons] block.  Critical when skip_logit_attribution=True.
        graph = Graph(
            input_string=self.tokenizer.decode(input_ids.detach().cpu().tolist()),
            input_tokens=input_ids,
            neuron_locations=neuron_locations_t,
            adjacency_matrix=adjacency,
            cfg=_HFGraphConfig(self),
            neuron_activations=neuron_activations_t,
            logit_targets=targets.logit_targets,
            logit_probabilities=targets.logit_probabilities,
            vocab_size=targets.vocab_size,
            neuron_write_vectors=source_vectors_t.detach().clone(),
        )
        if detach_result is None:
            detach_result = not create_graph
        if detach_result:
            graph = detach_graph(graph)
        return graph


class _HFOldMLP:
    """Shim for ``model.blocks[layer].mlp.old_mlp``.

    TransformerLens naming vs HF LLaMA shapes:

      TL attr   shape (TL)        HF attr            shape (HF)
      --------  ----------------  -----------------  -----------------
      W_gate    [d_mlp, d_model]  gate_proj.weight   [d_mlp, d_model]  ← same
      W_in      [d_mlp, d_model]  up_proj.weight     [d_mlp, d_model]  ← same
      W_out     [d_mlp, d_model]  down_proj.weight   [d_model, d_mlp]  ← TRANSPOSED

    ``W_out[neuron_id]`` must return the d_model-dimensional write vector for that
    neuron; indexing is on the d_mlp axis (first dim in TL convention).
    """

    def __init__(self, hf_mlp):
        self.W_gate = hf_mlp.gate_proj.weight   # already (d_mlp, d_model)
        self.W_in   = hf_mlp.up_proj.weight      # already (d_mlp, d_model)
        # down_proj.weight is (d_model, d_mlp) in HF → transpose to (d_mlp, d_model)
        self.W_out  = hf_mlp.down_proj.weight.T


class _HFMLP:
    """Shim for ``model.blocks[layer].mlp``."""

    def __init__(self, hf_mlp):
        self.old_mlp = _HFOldMLP(hf_mlp)


class _HFBlock:
    """Shim for a single ``model.blocks[layer]``."""

    def __init__(self, hf_layer):
        self.mlp = _HFMLP(hf_layer.mlp)


class _HFBlockList:
    """Shim for ``model.blocks`` — indexable list of ``_HFBlock`` objects."""

    def __init__(self, hf_layers):
        self._layers = hf_layers

    def __getitem__(self, idx: int) -> _HFBlock:
        return _HFBlock(self._layers[idx])

    def __len__(self) -> int:
        return len(self._layers)


class _HFUnembed:
    """Shim for ``model.unembed``.

    TransformerLens exposes ``model.unembed.W_U`` (shape [d_model, d_vocab]).
    HF stores the transposed version in ``lm_head.weight`` ([d_vocab, d_model]).
    """

    def __init__(self, lm_head):
        self._lm_head = lm_head

    @property
    def W_U(self) -> "torch.Tensor":
        return self._lm_head.weight.T


class _HFGraphConfig:
    def __init__(self, adapter: HFLlamaGraphAdapter):
        self.n_layers = adapter.n_layers
        self.d_model = adapter.d_model
        self.d_mlp = adapter.d_mlp
        self.d_vocab = adapter.d_vocab
        self.d_head = int(getattr(adapter.config, "head_dim", 0) or adapter.d_model // adapter.config.num_attention_heads)
        self.n_heads = int(adapter.config.num_attention_heads)
        self.n_key_value_heads = getattr(adapter.config, "num_key_value_heads", None)
        self.device = adapter.device
        name = getattr(adapter.config, "_name_or_path", None) or getattr(adapter.config, "name_or_path", "llama")
        self.model_name = name
        self.tokenizer_name = name

    def to_dict(self) -> dict[str, Any]:
        return vars(self)


def detach_graph(graph: Graph) -> Graph:
    nw = graph.neuron_write_vectors
    if nw is not None:
        nw = nw.detach()
    return Graph(
        input_string=graph.input_string,
        input_tokens=graph.input_tokens.detach(),
        neuron_locations=graph.neuron_locations.detach(),
        adjacency_matrix=graph.adjacency_matrix.detach(),
        cfg=graph.cfg,
        neuron_activations=graph.neuron_activations.detach(),
        logit_targets=graph.logit_targets,
        logit_probabilities=graph.logit_probabilities.detach(),
        vocab_size=graph.vocab_size,
        attribution_mode=graph.attribution_mode,
        neuron_write_vectors=nw,
    )



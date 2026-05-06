"""Hugging Face LLaMA graph attribution helpers used during distillation."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from graph_loss.attribution.targets import AttributionTargets, TargetSpec
from graph_loss.graph import Graph, SuperGraph


@dataclass
class HFGraphBuildConfig:
    top_k_logits: int | None = 20
    prop_neurons_per_layer: float = 0.1
    batch_size: int = 512


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
        return tokens.to(self.device)

    @staticmethod
    def _row_oriented_weight(weight: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
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

    def compute_supernode_ablation_prob_deltas(
        self,
        prompt: str | torch.Tensor | list[int],
        supernodes: list[list[int]],
        neuron_locations_t: torch.Tensor,
        logit_token_ids: torch.Tensor,
        dtype: torch.dtype | None = None,
        ablation_batch_size: int = 8,
    ) -> torch.Tensor:
        """TRUE hook-based supernode ablation prob-deltas (NO grad).

        Used as the cross-model ALIGNMENT signal: this is the principled
        per-supernode causal ablation — masks member neurons to zero at the
        ``down_proj`` input via batched forward pre-hooks, runs the full
        student forward, measures softmax probability change at the answer
        position.  Cross-layer effects propagate naturally because the
        masked activations flow through all subsequent layers and attention.

        Same operational semantics as the teacher's cached
        ``supernode_prob_deltas`` (TransformerLens hook ablation), so cosine
        similarity between the two is a meaningful similarity metric.

        Wrapped entirely in ``torch.no_grad()`` — we don't backprop through
        these forwards (~5–6 chunks per prompt × ~30–50 supernodes), which
        would explode activation memory.  The cheap differentiable
        approximation in ``compute_supernode_dla_approx_prob_deltas_with_grad``
        provides the gradient pathway for the loss.

        Returns a ``[n_supernodes, n_logits]`` float32 tensor.
        """
        input_ids = self.ensure_tokenized(prompt)
        device = self.device
        logit_ids = logit_token_ids.to(device=device, dtype=torch.long)
        n_logits = int(logit_ids.numel())
        n_sn = len(supernodes)

        if n_sn == 0:
            return torch.empty((0, n_logits), device=device, dtype=torch.float32)

        input_ids_2d = input_ids.unsqueeze(0)
        S = input_ids.shape[0]
        d_mlp = self.d_mlp

        with torch.no_grad():
            with self.autocast_context(dtype):
                baseline_logits = self.model(
                    input_ids=input_ids_2d,
                    attention_mask=torch.ones_like(input_ids_2d),
                    use_cache=False,
                ).logits
            baseline_probs = torch.softmax(
                baseline_logits[0, -1].float(), dim=-1
            )[logit_ids]

            supernode_layer_neurons: list[dict[int, list[int]]] = []
            for members in supernodes:
                layer_groups: dict[int, list[int]] = {}
                for m in members:
                    layer_idx = int(neuron_locations_t[m, 0].item())
                    neuron_id = int(neuron_locations_t[m, 2].item())
                    layer_groups.setdefault(layer_idx, []).append(neuron_id)
                supernode_layer_neurons.append(layer_groups)

            delta_chunks: list[torch.Tensor] = []

            for chunk_start in range(0, n_sn, ablation_batch_size):
                chunk_end = min(chunk_start + ablation_batch_size, n_sn)
                chunk = supernode_layer_neurons[chunk_start:chunk_end]
                B = len(chunk)

                layer_to_mask: dict[int, torch.Tensor] = {}
                for b, layer_groups in enumerate(chunk):
                    for layer_idx, neuron_ids in layer_groups.items():
                        if layer_idx not in layer_to_mask:
                            layer_to_mask[layer_idx] = torch.ones(
                                B, d_mlp, device=device, dtype=baseline_logits.dtype
                            )
                        if neuron_ids:
                            idx = torch.tensor(
                                neuron_ids, device=device, dtype=torch.long
                            )
                            layer_to_mask[layer_idx][b] = layer_to_mask[layer_idx][b].scatter(
                                0, idx, 0.0
                            )

                handles = []
                for layer_idx, mask in layer_to_mask.items():
                    def make_hook(_mask):
                        def _hook(_module, args):
                            x = args[0]
                            m = _mask.to(x.dtype).unsqueeze(1)
                            return (x * m,)
                        return _hook
                    handle = self.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(
                        make_hook(mask)
                    )
                    handles.append(handle)

                try:
                    input_ids_b = input_ids_2d.expand(B, S)
                    attn_b = torch.ones_like(input_ids_b)
                    with self.autocast_context(dtype):
                        ablated_logits = self.model(
                            input_ids=input_ids_b,
                            attention_mask=attn_b,
                            use_cache=False,
                        ).logits
                    ablated_probs = torch.softmax(
                        ablated_logits[:, -1].float(), dim=-1
                    )[:, logit_ids]
                    delta = baseline_probs.unsqueeze(0) - ablated_probs
                    delta_chunks.append(delta)
                finally:
                    for h in handles:
                        h.remove()

        return torch.cat(delta_chunks, dim=0).to(torch.float32)

    def compute_supernode_ablation_prob_deltas_chunk_with_grad(
        self,
        prompt: str | torch.Tensor | list[int],
        supernodes_chunk: list[list[int]],
        neuron_locations_t: torch.Tensor,
        logit_token_ids: torch.Tensor,
        baseline_probs_detached: torch.Tensor,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """TRUE hook-based supernode ablation prob-deltas, WITH grad, for a
        small chunk of supernodes.

        Used by the "true" graph_grad_mode pipeline.  Caller is responsible
        for slicing supernodes into chunks small enough to fit in memory and
        calling ``backward()`` on the resulting per-chunk loss term so the
        autograd graph is freed before the next chunk is processed.

        Args:
            supernodes_chunk: ``B`` supernodes (each a list of neuron indices)
                to ablate in a single batched forward pass.
            baseline_probs_detached: pre-computed ``softmax(baseline_logits)``
                indexed by ``logit_token_ids``, shape ``[n_logits]``.  Detached
                so that no gradient flows through the baseline path (KL
                already handles that signal).

        Returns:
            ``[B, n_logits]`` differentiable prob_delta tensor.
        """
        input_ids = self.ensure_tokenized(prompt)
        device = self.device
        logit_ids = logit_token_ids.to(device=device, dtype=torch.long)
        n_logits = int(logit_ids.numel())
        B = len(supernodes_chunk)

        if B == 0:
            return torch.empty((0, n_logits), device=device, dtype=torch.float32)

        input_ids_2d = input_ids.unsqueeze(0)
        S = input_ids.shape[0]
        d_mlp = self.d_mlp

        layer_to_mask: dict[int, torch.Tensor] = {}
        for b, members in enumerate(supernodes_chunk):
            for m in members:
                layer_idx = int(neuron_locations_t[m, 0].item())
                neuron_id = int(neuron_locations_t[m, 2].item())
                if layer_idx not in layer_to_mask:
                    layer_to_mask[layer_idx] = torch.ones(
                        B, d_mlp, device=device,
                        dtype=baseline_probs_detached.dtype,
                    )
                layer_to_mask[layer_idx][b, neuron_id] = 0.0

        handles = []
        for layer_idx, mask in layer_to_mask.items():
            def make_hook(_mask):
                def _hook(_module, args):
                    x = args[0]
                    m = _mask.to(x.dtype).unsqueeze(1)
                    return (x * m,)
                return _hook
            handle = self.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(
                make_hook(mask)
            )
            handles.append(handle)

        try:
            input_ids_b = input_ids_2d.expand(B, S)
            attn_b = torch.ones_like(input_ids_b)
            with self.autocast_context(dtype):
                ablated_logits = self.model(
                    input_ids=input_ids_b,
                    attention_mask=attn_b,
                    use_cache=False,
                ).logits
            ablated_probs = torch.softmax(
                ablated_logits[:, -1].float(), dim=-1
            )[:, logit_ids]
            delta = baseline_probs_detached.unsqueeze(0) - ablated_probs
        finally:
            for h in handles:
                h.remove()

        return delta.to(torch.float32)

    def compute_baseline_probs(
        self,
        prompt: str | torch.Tensor | list[int],
        logit_token_ids: torch.Tensor,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Compute baseline softmax probs over ``logit_token_ids`` (no grad).

        Helper used by "true" graph_grad_mode: we detach baseline so it doesn't
        couple gradients across all chunked ablation backwards.  KL distillation
        already supplies the gradient signal that pushes baseline toward
        teacher behavior.
        """
        input_ids = self.ensure_tokenized(prompt)
        device = self.device
        logit_ids = logit_token_ids.to(device=device, dtype=torch.long)
        with torch.no_grad():
            with self.autocast_context(dtype):
                logits = self.model(
                    input_ids=input_ids.unsqueeze(0),
                    attention_mask=torch.ones_like(input_ids).unsqueeze(0),
                    use_cache=False,
                ).logits
            return torch.softmax(logits[0, -1].float(), dim=-1)[logit_ids]

    def compute_supernode_dla_approx_prob_deltas_with_grad(
        self,
        prompt: str | torch.Tensor | list[int],
        supernodes: list[list[int]],
        neuron_locations_t: torch.Tensor,
        logit_token_ids: torch.Tensor,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Cheap DIFFERENTIABLE approximation of supernode ablation prob-deltas.

        Used as the GRADIENT signal for the alignment-based loss.  Same
        underlying intuition as true ablation, but only captures the
        direct-residual path:

            1. One forward pass with grad → ``final_resid`` at last position
               (the residual stream value going into the LN + lm_head).
            2. For each supernode S, compute the total "direct contribution"
               of its members to ``final_resid``:
                   ``total_source = Σ_member (a_i × W_out_i)``
               where ``a_i`` is the post-gating MLP activation and ``W_out_i``
               is the corresponding row of ``down_proj.weight``.  This sum is
               what the supernode would contribute via the residual stream
               ignoring downstream cross-layer effects.
            3. ``ablated_resid = final_resid - total_source``
            4. ``ablated_probs = softmax(lm_head(norm(ablated_resid)))``
            5. ``prob_delta = baseline_probs - ablated_probs``

        Cost: ONE grad-enabled forward pass + per-supernode (norm + lm_head +
        softmax + a few mat-vecs).  The single forward holds the backward
        graph; per-supernode math is tiny.  Total memory ~2-3 GB peak, vs
        the ~80 GB required for true-ablation-with-grad.

        Faithfulness: misses cross-layer corrections that true ablation
        captures.  But once supernodes are correctly matched (by the no-grad
        true-ablation signal), the LOSS gradient just needs to point roughly
        toward the matched teacher's prob_delta — the linear direct-path
        component is the dominant gradient signal.

        Returns ``[n_supernodes, n_logits]`` differentiable through every
        student parameter that affects ``a_i`` (upstream layers) or
        ``W_out_i`` (down_proj.weight).
        """
        input_ids = self.ensure_tokenized(prompt)
        device = self.device
        logit_ids = logit_token_ids.to(device=device, dtype=torch.long)
        n_logits = int(logit_ids.numel())
        n_sn = len(supernodes)

        if n_sn == 0:
            return torch.empty((0, n_logits), device=device, dtype=torch.float32)

        input_ids_2d = input_ids.unsqueeze(0)

        # Capture down_proj inputs (= post-gating MLP activations) per layer
        # via forward pre-hooks.  These tensors retain grad through the
        # forward pass below.
        mlp_inputs: dict[int, torch.Tensor] = {}
        handles = []
        for layer_idx, layer in enumerate(self.layers):
            def pre_hook(_module, inputs, *, idx=layer_idx):
                mlp_inputs[idx] = inputs[0]  # [1, S, d_mlp]
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(pre_hook))

        try:
            with self.autocast_context(dtype):
                out = self.model(
                    input_ids=input_ids_2d,
                    attention_mask=torch.ones_like(input_ids_2d),
                    output_hidden_states=True,
                    use_cache=False,
                )
            baseline_logits = out.logits  # [1, S, V]
            final_resid = out.hidden_states[-1][0, -1]  # [d_model]
        finally:
            for h in handles:
                h.remove()

        baseline_probs = torch.softmax(
            baseline_logits[0, -1].float(), dim=-1
        )[logit_ids]  # [n_logits]

        # The norm and lm_head used to project final_resid → logits.
        norm_fn = self.model.model.norm
        head_fn = self.lm_head

        # Group members by layer once up front.
        supernode_layer_members: list[dict[int, list[int]]] = []
        for members in supernodes:
            by_layer: dict[int, list[int]] = {}
            for m in members:
                layer_idx = int(neuron_locations_t[m, 0].item())
                neuron_id = int(neuron_locations_t[m, 2].item())
                by_layer.setdefault(layer_idx, []).append(neuron_id)
            supernode_layer_members.append(by_layer)

        deltas: list[torch.Tensor] = []
        for by_layer in supernode_layer_members:
            if not by_layer:
                deltas.append(
                    torch.zeros(
                        n_logits, device=device, dtype=baseline_probs.dtype
                    )
                )
                continue
            total_source = torch.zeros_like(final_resid)
            for layer_idx, neuron_ids in by_layer.items():
                if layer_idx not in mlp_inputs:
                    continue
                # a_vec: [k] vector of post-gating activations at last token
                # for the k members of this supernode in this layer.
                acts_layer = mlp_inputs[layer_idx][0, -1]  # [d_mlp]
                idx_t = torch.tensor(
                    neuron_ids, device=device, dtype=torch.long
                )
                a_vec = acts_layer.index_select(0, idx_t)  # [k]
                # W_out cols: [d_model, k] — down_proj.weight is [d_model, d_mlp]
                w_out_cols = self.layers[layer_idx].mlp.down_proj.weight.index_select(
                    1, idx_t
                )  # [d_model, k]
                # Direct contribution to residual: w_out_cols @ a_vec → [d_model]
                total_source = total_source + (w_out_cols.to(a_vec.dtype) @ a_vec)
            ablated_resid = final_resid - total_source
            ablated_logits = head_fn(
                norm_fn(ablated_resid.unsqueeze(0).unsqueeze(0))
            )[0, 0]  # [V]
            ablated_probs = torch.softmax(ablated_logits.float(), dim=-1)[logit_ids]
            deltas.append(baseline_probs - ablated_probs)

        return torch.stack(deltas, dim=0).to(torch.float32)

    def compute_logit_focus_vector_with_grad(
        self,
        prompt: str | torch.Tensor | list[int],
        neuron_locations_t: torch.Tensor,
        logit_token_ids: torch.Tensor,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Compute a differentiable [n_logits] logit-focus vector for all selected neurons.

        For each selected neuron i at (layer_idx, pos_idx, neuron_id):

            DLA_i = a_i · W_out[neuron_id] @ W_U[:, logit_token_ids]

        Returns ``Σ_i |DLA_i|`` — an un-normalised [n_logits] vector representing the
        total absolute DLA influence that all selected neurons exert on each of the
        top-K logit targets for this prompt.

        Callers should normalise to a distribution before computing KL loss.

        The result is differentiable through:
          - ``a_i`` (gate/up_proj activations, via SiLU-gated MLP forward)
          - ``W_out[neuron_id]`` (down_proj.weight)

        One extra forward pass is performed (no Jacobian, no batch expansion).
        """
        input_ids = self.ensure_tokenized(prompt)
        device = self.device

        mlp_inputs: dict[int, torch.Tensor] = {}
        handles = []

        for layer_idx, layer in enumerate(self.layers):
            def pre_hook(_module, inputs, *, idx=layer_idx):
                mlp_inputs[idx] = inputs[0]
            handles.append(layer.mlp.register_forward_pre_hook(pre_hook))

        input_ids_2d = input_ids.unsqueeze(0)
        try:
            with self.autocast_context(dtype):
                _ = self.model(
                    input_ids=input_ids_2d,
                    attention_mask=torch.ones_like(input_ids_2d),
                    output_hidden_states=False,
                    use_cache=False,
                )
        finally:
            for handle in handles:
                handle.remove()

        logit_ids = logit_token_ids.to(device=device, dtype=torch.long)
        n_logits = int(logit_ids.numel())
        W_U = self.model.lm_head.weight  # [vocab_size, hidden_size]
        act_dtype = next(iter(mlp_inputs.values())).dtype if mlp_inputs else W_U.dtype
        # [n_logits, hidden_size]
        W_U_logits = W_U[logit_ids].to(dtype=act_dtype)

        focus_parts: list[torch.Tensor] = []

        for layer_idx in range(self.n_layers):
            if layer_idx not in mlp_inputs:
                continue
            layer_mask = neuron_locations_t[:, 0] == layer_idx
            if not layer_mask.any():
                continue

            locs = neuron_locations_t[layer_mask]    # [K, 3]
            poses = locs[:, 1]                        # [K]
            neurons = locs[:, 2]                      # [K]

            layer_input = mlp_inputs[layer_idx].squeeze(0)  # [seq_len, d_model]
            gate_rows, up_rows, out_rows, gate_bias, up_bias = self._layer_weights(
                layer_idx, device=device, dtype=layer_input.dtype
            )

            pos_inputs = layer_input[poses]                           # [K, d_model]
            gate_pre = pos_inputs @ gate_rows.T + gate_bias           # [K, d_mlp]
            up_pre = pos_inputs @ up_rows.T + up_bias                 # [K, d_mlp]
            gate_act = F.silu(gate_pre)
            all_acts = gate_act * up_pre                              # [K, d_mlp]

            k_range = torch.arange(locs.shape[0], device=device)
            a_i = all_acts[k_range, neurons]                          # [K]
            w_out_i = out_rows[neurons]                               # [K, d_model]

            # source_vecs[k] = a_i[k] * w_out_i[k],  shape [K, d_model]
            source_vecs = a_i.unsqueeze(-1) * w_out_i                 # [K, d_model]

            # DLA = source_vecs @ W_U_logits.T → [K, n_logits]
            dla = source_vecs @ W_U_logits.to(source_vecs.dtype).T

            focus_parts.append(dla.abs())                             # [K, n_logits]

        if not focus_parts:
            zero = torch.zeros(n_logits, device=device, dtype=W_U.dtype)
            return zero

        return torch.cat(focus_parts, dim=0).sum(dim=0)   # [n_logits]

    def compute_supernode_dlas_with_grad(
        self,
        prompt: str | torch.Tensor | list[int],
        supernodes: list[list[int]],
        neuron_locations_t: torch.Tensor,
        n_vocab: int,
        dtype: torch.dtype | None = None,
    ) -> dict[int, torch.Tensor]:
        """Compute differentiable DLA for each supernode.
        
        Runs a standard forward pass (with autograd enabled) to capture intermediate
        activations, and projects the specified neurons through W_out and W_U.
        This provides a differentiable functional mapping from the student's structural 
        activations to the vocab space, which trains the student model directly.
        """
        input_ids = self.ensure_tokenized(prompt)
        device = self.device
        
        mlp_inputs = {}
        handles = []
        
        for layer_idx, layer in enumerate(self.layers):
            def pre_hook(_module, inputs, *, idx=layer_idx):
                mlp_inputs[idx] = inputs[0]
            handles.append(layer.mlp.register_forward_pre_hook(pre_hook))
            
        input_ids_2d = input_ids.unsqueeze(0)
        try:
            with self.autocast_context(dtype):
                _ = self.model(
                    input_ids=input_ids_2d,
                    attention_mask=torch.ones_like(input_ids_2d),
                    output_hidden_states=False,
                    use_cache=False,
                )
        finally:
            for handle in handles:
                handle.remove()
                
        W_U = self.model.lm_head.weight  # [vocab_size, hidden_size]
        
        layer_neuron_acts = {}
        for layer_idx in range(self.n_layers):
            if layer_idx not in mlp_inputs:
                continue
            layer_input = mlp_inputs[layer_idx].squeeze(0) # [seq_len, hidden_size]
            gate_rows, up_rows, out_rows, gate_bias, up_bias = self._layer_weights(
                layer_idx, device=device, dtype=layer_input.dtype
            )
            
            # Standard MLP forward to maintain gradient graph
            gate_pre = layer_input @ gate_rows.T + gate_bias
            up_pre = layer_input @ up_rows.T + up_bias
            gate_act = F.silu(gate_pre)
            neuron_activations = gate_act * up_pre  # [seq_len, d_mlp]
            
            layer_neuron_acts[layer_idx] = {
                'acts': neuron_activations,
                'out_rows': out_rows
            }
            
        supernode_dlas = {}

        for i, sn_indices in enumerate(supernodes):
            if not sn_indices:
                supernode_dlas[i] = torch.zeros(n_vocab, device=device, dtype=W_U.dtype)
                continue

            locations = neuron_locations_t[sn_indices]  # [K, 3] (layer, pos, neuron)

            # Accumulate source write-vectors in d_model space using out-of-place + so that
            # the computation graph is intact through every addition.  A single W_U matmul
            # at the end avoids repeated vocab-space in-place ops that can silently detach
            # gradients after the first iteration.
            sn_total_source = torch.zeros(self.d_model, device=device, dtype=W_U.dtype)
            for layer_idx in torch.unique(locations[:, 0]):
                mask = locations[:, 0] == layer_idx
                locs = locations[mask]  # [M, 3]
                poses = locs[:, 1]
                neurons = locs[:, 2]
                acts = layer_neuron_acts[layer_idx.item()]['acts'][poses, neurons]  # [M]
                out_w = layer_neuron_acts[layer_idx.item()]['out_rows'][neurons]    # [M, hidden_size]
                sn_sum_source = (acts.unsqueeze(-1) * out_w).sum(dim=0)            # [hidden_size]
                sn_total_source = sn_total_source + sn_sum_source.to(sn_total_source.dtype)

            supernode_dlas[i] = (W_U @ sn_total_source)[:n_vocab]

        return supernode_dlas


class _HFGraphConfig:
    def __init__(self, adapter: HFLlamaGraphAdapter):
        self.n_layers = adapter.n_layers
        self.d_model = adapter.d_model
        self.d_mlp = adapter.d_mlp
        self.d_vocab = adapter.d_vocab
        self.d_head = int(getattr(adapter.config, "head_dim", 0) or adapter.d_model // adapter.config.num_attention_heads)
        self.n_heads = int(adapter.config.num_attention_heads)
        self.n_key_value_heads = getattr(adapter.config, "num_key_value_heads", None)
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


def extract_hf_supernode_members(
    supergraph: SuperGraph,
    graph: Graph,
    adapter: HFLlamaGraphAdapter,
    *,
    detach: bool,
) -> list[dict]:
    result = []
    w_out_cache = {}
    for cluster_id, members in enumerate(supergraph.supernodes):
        acts = []
        w_outs = []
        for node_id in members:
            layer_idx = int(graph.neuron_locations[node_id, 0].item())
            neuron_id = int(graph.neuron_locations[node_id, 2].item())
            if layer_idx not in w_out_cache:
                mlp = adapter.layers[layer_idx].mlp
                W_out = adapter._row_oriented_weight(
                    mlp.down_proj.weight,
                    adapter.d_mlp,
                    adapter.d_model,
                ).to(device=graph.adjacency_device, dtype=graph.adjacency_matrix.dtype)
                w_out_cache[layer_idx] = W_out
            act = graph.neuron_activations[node_id].unsqueeze(0)
            w_row = w_out_cache[layer_idx][neuron_id]
            if detach:
                act = act.detach()
                w_row = w_row.detach()
            acts.append(act)
            w_outs.append(w_row)
        result.append(
            {
                "cluster_id": cluster_id,
                "activations": acts,
                "w_out_rows": w_outs,
                "size": len(members),
            },
        )
    return result

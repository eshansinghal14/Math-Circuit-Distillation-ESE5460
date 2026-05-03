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
            if not output.requires_grad:
                output = output.detach().requires_grad_(True)
            embed_out = output
            return output

        handles.append(self.embed_tokens.register_forward_hook(embed_hook))

        for layer_idx, layer in enumerate(self.layers):
            def pre_hook(_module, inputs, *, idx=layer_idx):
                mlp_inputs[idx] = inputs[0]

            def out_hook(_module, _inputs, output, *, idx=layer_idx):
                if not output.requires_grad:
                    output = output.detach().requires_grad_(True)
                mlp_outputs[idx] = output
                return output

            handles.append(layer.mlp.register_forward_pre_hook(pre_hook))
            handles.append(layer.mlp.register_forward_hook(out_hook))

        try:
            with self.autocast_context(dtype):
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
        adjacency = torch.zeros(
            total_nodes,
            total_nodes,
            dtype=source_vectors_t.dtype,
            device=self.device,
        )

        def _source_scores_from_grads(grads, batch_len: int) -> torch.Tensor:
            rows = torch.zeros(batch_len, source_count, dtype=source_vectors_t.dtype, device=self.device)
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
                rows[:, layer_indices] = (
                    grad_vecs * source_vectors_t[layer_indices].unsqueeze(0)
                ).sum(dim=-1)
            token_grad = grads[-1]
            if token_grad is not None:
                token_vectors = embed_out.squeeze(0).to(source_vectors_t.dtype)
                rows[:, n_neurons:source_count] = (
                    token_grad.to(source_vectors_t.dtype) * token_vectors.unsqueeze(0)
                ).sum(dim=-1)
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

        for start in range(0, n_neurons, max(1, batch_size)):
            end = min(start + max(1, batch_size), n_neurons)
            if verbose:
                print(f"    [graph] neuron rows {start}:{end} / {n_neurons}")
            adjacency[start:end, :source_count] = source_scores_neuron_chunk(start, end)

        logit_start = n_neurons + n_tokens
        for start in range(0, n_logits, max(1, batch_size)):
            end = min(start + max(1, batch_size), n_logits)
            if verbose:
                print(f"    [graph] logit rows {start}:{end} / {n_logits}")
            adjacency[logit_start + start:logit_start + end, :source_count] = source_scores_logit_chunk(start, end)

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
        )
        if detach_result is None:
            detach_result = not create_graph
        if detach_result:
            graph = detach_graph(graph)
        return graph

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
            sn_dla = torch.zeros(n_vocab, device=device, dtype=W_U.dtype)
            if not sn_indices:
                supernode_dlas[i] = sn_dla
                continue
                
            locations = neuron_locations_t[sn_indices] # [K, 3] (layer, pos, neuron)
            
            for layer_idx in torch.unique(locations[:, 0]):
                mask = locations[:, 0] == layer_idx
                locs = locations[mask] # [M, 3]
                
                poses = locs[:, 1]
                neurons = locs[:, 2]
                
                acts = layer_neuron_acts[layer_idx.item()]['acts'][poses, neurons] # [M]
                out_w = layer_neuron_acts[layer_idx.item()]['out_rows'][neurons] # [M, hidden_size]
                
                sn_source_vectors = acts.unsqueeze(-1) * out_w # [M, hidden_size]
                sn_sum_source = sn_source_vectors.sum(dim=0) # [hidden_size]
                
                sn_dla_full = W_U @ sn_sum_source.to(W_U.dtype) # [vocab_size]
                
                if sn_dla_full.shape[0] >= n_vocab:
                    sn_dla += sn_dla_full[:n_vocab]
                else:
                    sn_dla[:sn_dla_full.shape[0]] += sn_dla_full
                
            supernode_dlas[i] = sn_dla
            
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

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
            if not prompt:
                raise ValueError(
                    "Prompt is an empty string. Pass a non-empty value to --prompt."
                )
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
                f"Tokenizer produced 0 tokens from prompt {prompt!r:.100}. "
                "Ensure the prompt is non-empty and contains tokenizable text."
            )
        tokens = tokens.to(self.device)
        bos_token_id = self.tokenizer.bos_token_id
        if bos_token_id is None:
            raise ValueError("LLaMA tokenizer must define bos_token_id.")
        if int(tokens[0].item()) != int(bos_token_id):
            tokens = torch.cat(
                [torch.tensor([bos_token_id], device=self.device, dtype=tokens.dtype), tokens]
            )
        return tokens

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
        # Zero BOS position (index 0) without inplace ops: inplace would corrupt
        # the saved neuron_activations.unsqueeze(-1) view ([n_pos, d_mlp, 1]) that
        # autograd holds for the grad w.r.t. out_rows, triggering AsStridedBackward0 error.
        n_pos = neuron_activations.shape[0]
        if n_pos > 0:
            bos_mask = torch.ones(n_pos, dtype=neuron_activations.dtype, device=neuron_activations.device)
            bos_mask[0] = 0.0
            neuron_activations = neuron_activations * bos_mask.view(-1, 1)
            target_encoders = target_encoders * bos_mask.view(-1, 1, 1)
            source_vectors = source_vectors * bos_mask.view(-1, 1, 1)
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
        skip_logit_attribution: bool = False,
    ) -> Graph:
        from graph_loss.attribution.attribute import attribute
        return attribute(
            self,
            prompt,
            attribution_targets=attribution_targets,
            top_k_logits=top_k_logits,
            prop_neurons_per_layer=prop_neurons_per_layer,
            batch_size=batch_size,
            dtype=dtype,
            verbose=verbose,
            create_graph=create_graph,
            detach_result=detach_result,
            skip_logit_attribution=skip_logit_attribution,
        )


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
        neuron_write_vectors=nw,
    )



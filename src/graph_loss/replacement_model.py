import torch
import torch.nn.functional as F
from torch import nn
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint

from graph_loss.attribution.context import AttributionContext
from utils import get_default_device


class ReplacementMLP(nn.Module):
    """Wrap a TransformerLens MLP so we have stable hook points."""

    def __init__(self, old_mlp: nn.Module):
        super().__init__()
        self.old_mlp = old_mlp
        self.hook_in = HookPoint()
        self.hook_out = HookPoint()

    def forward(self, x):
        x = self.hook_in(x)
        mlp_out = self.old_mlp(x)
        return self.hook_out(mlp_out)


class ReplacementUnembed(nn.Module):
    """Wrap the unembed layer so final residuals can be cached cleanly."""

    def __init__(self, old_unembed: nn.Module):
        super().__init__()
        self.old_unembed = old_unembed
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()

    @property
    def W_U(self):
        return self.old_unembed.W_U

    @property
    def b_U(self):
        return self.old_unembed.b_U

    def forward(self, x):
        x = self.hook_pre(x)
        x = self.old_unembed(x)
        return self.hook_post(x)


class TransformerLensReplacementModel(HookedTransformer):
    """TransformerLens model configured for neuron-level attribution graphs."""

    backend: str
    feature_input_hook: str
    feature_output_hook: str

    @classmethod
    def from_config(
        cls,
        config: HookedTransformerConfig,
        **kwargs,
    ) -> "TransformerLensReplacementModel":
        model = cls(config, **kwargs)
        model._configure_replacement_model()
        return model

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: torch.device | None = None,
        dtype: torch.dtype | None = torch.float32,
        **kwargs,
    ) -> "TransformerLensReplacementModel":
        if device is None:
            device = get_default_device()

        model = super().from_pretrained(
            model_name,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            device=device,
            dtype=dtype,
            **kwargs,
        )
        model._configure_replacement_model()
        return model

    def _configure_replacement_model(self):
        self.backend = "transformerlens"
        self.feature_input_hook = "mlp.hook_in"
        self.feature_output_hook = "mlp.hook_out"
        self.zero_positions = slice(0, 1)

        for block in self.blocks:
            if not isinstance(block.mlp, ReplacementMLP):  # type: ignore[arg-type]
                block.mlp = ReplacementMLP(block.mlp)  # type: ignore[assignment]

        if not isinstance(self.unembed, ReplacementUnembed):
            self.unembed = ReplacementUnembed(self.unembed)  # type: ignore[assignment]

        self._configure_gradient_flow()
        self._deduplicate_attention_buffers()
        self.setup()

    def _configure_gradient_flow(self):
        def stop_gradient(acts, hook):
            return acts.detach()

        for block in self.blocks:
            block.attn.hook_pattern.add_hook(stop_gradient, is_permanent=True)  # type: ignore[attr-defined]
            block.ln1.hook_scale.add_hook(stop_gradient, is_permanent=True)  # type: ignore[attr-defined]
            block.ln2.hook_scale.add_hook(stop_gradient, is_permanent=True)  # type: ignore[attr-defined]

        self.ln_final.hook_scale.add_hook(stop_gradient, is_permanent=True)  # type: ignore[attr-defined]

        for param in self.parameters():
            param.requires_grad = False

        def enable_gradient(acts, hook):
            acts.requires_grad = True
            return acts

        self.hook_embed.add_hook(enable_gradient, is_permanent=True)  # type: ignore[attr-defined]

    def _deduplicate_attention_buffers(self):
        """Share duplicated attention buffers across layers."""

        attn_buffers = {}

        for block in self.blocks:
            attn_buffers[block.attn.attn_type] = block.attn.mask  # type: ignore[attr-defined]
            if hasattr(block.attn, "rotary_sin"):
                attn_buffers["rotary_sin"] = block.attn.rotary_sin  # type: ignore[attr-defined]
                attn_buffers["rotary_cos"] = block.attn.rotary_cos  # type: ignore[attr-defined]

        for block in self.blocks:
            block.attn.mask = attn_buffers[block.attn.attn_type]  # type: ignore[attr-defined]
            if hasattr(block.attn, "rotary_sin"):
                block.attn.rotary_sin = attn_buffers["rotary_sin"]  # type: ignore[attr-defined]
                block.attn.rotary_cos = attn_buffers["rotary_cos"]  # type: ignore[attr-defined]

    def ensure_tokenized(self, prompt: str | torch.Tensor | list[int]) -> torch.Tensor:
        """Convert prompt to a 1-D tensor of token ids and prepend a special token if needed."""

        if isinstance(prompt, str):
            tokens = self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).input_ids.squeeze(0)  # type: ignore[assignment]
        elif isinstance(prompt, torch.Tensor):
            tokens = prompt.squeeze()
        elif isinstance(prompt, list):
            tokens = torch.tensor(prompt, dtype=torch.long).squeeze()
        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt)}")

        if tokens.ndim > 1:
            raise ValueError(f"Tensor must be 1-D, got shape {tokens.shape}")

        tokens = tokens.to(self.cfg.device)

        bos_token_id = self.tokenizer.bos_token_id  # type: ignore[attr-defined]
        if bos_token_id is None:
            raise ValueError("LLaMA tokenizer must define bos_token_id.")
        if int(tokens[0].item()) == int(bos_token_id):
            return tokens

        tokens = torch.cat(
            [torch.tensor([bos_token_id], device=tokens.device, dtype=tokens.dtype), tokens]
        )
        return tokens.to(self.cfg.device)

    def _stack_layer_cache(self, cache: dict[str, torch.Tensor], suffix: str) -> torch.Tensor:
        layer_entries: list[tuple[int, torch.Tensor]] = []
        for hook_name, value in cache.items():
            if not hook_name.endswith(suffix):
                continue
            layer_entries.append((int(hook_name.split(".")[1]), value))

        layer_entries.sort(key=lambda item: item[0])
        if len(layer_entries) != self.cfg.n_layers:
            raise RuntimeError(
                f"Expected {self.cfg.n_layers} cached layers for {suffix}, got {len(layer_entries)}"
            )
        return torch.stack([value for _, value in layer_entries], dim=0)

    def _row_oriented_weight(self, weight: torch.Tensor) -> torch.Tensor:
        d_model = self.cfg.d_model
        d_mlp = self.cfg.d_mlp

        if weight.shape == (d_mlp, d_model):
            return weight
        if weight.shape == (d_model, d_mlp):
            return weight.T
        raise ValueError(
            f"Unsupported MLP weight shape {tuple(weight.shape)} for model dims "
            f"(d_model={d_model}, d_mlp={d_mlp})"
        )

    def _get_bias(
        self,
        module: nn.Module,
        name: str,
        size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        bias = getattr(module, name, None)
        if bias is None:
            return torch.zeros(size, device=device, dtype=dtype)
        return bias.to(device=device, dtype=dtype)

    @staticmethod
    def _silu_derivative(x: torch.Tensor) -> torch.Tensor:
        sigma = torch.sigmoid(x)
        return sigma + x * sigma * (1 - sigma)

    def _compute_gated_layer_data(
        self,
        old_mlp: nn.Module,
        mlp_input: torch.Tensor,
        out_rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_weight = old_mlp.W_gate
        up_weight = old_mlp.W_in
        gate_rows = self._row_oriented_weight(
            gate_weight.to(device=mlp_input.device, dtype=mlp_input.dtype)
        )
        up_rows = self._row_oriented_weight(
            up_weight.to(device=mlp_input.device, dtype=mlp_input.dtype)
        )
        gate_bias = self._get_bias(
            old_mlp,
            "b_gate",
            gate_rows.shape[0],
            mlp_input.device,
            mlp_input.dtype,
        )
        up_bias = self._get_bias(
            old_mlp,
            "b_in",
            up_rows.shape[0],
            mlp_input.device,
            mlp_input.dtype,
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

    def _compute_layer_neuron_data(
        self,
        layer: int,
        mlp_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        old_mlp = self.blocks[layer].mlp.old_mlp  # type: ignore[attr-defined]
        out_rows = self._row_oriented_weight(
            old_mlp.W_out.to(device=mlp_input.device, dtype=mlp_input.dtype)
        )

        neuron_activations, target_encoders, source_vectors = self._compute_gated_layer_data(
            old_mlp, mlp_input, out_rows
        )

        neuron_activations[self.zero_positions] = 0
        target_encoders[self.zero_positions] = 0
        source_vectors[self.zero_positions] = 0
        return neuron_activations, target_encoders, source_vectors

    @torch.no_grad()
    def setup_attribution(self, inputs: str | torch.Tensor, prop_neurons_per_layer: float = 0.1):
        """Precompute prompt-specific neuron source vectors and target encoders."""

        if isinstance(inputs, str):
            tokens = self.ensure_tokenized(inputs)
        else:
            tokens = inputs.squeeze()

        assert isinstance(tokens, torch.Tensor), "Tokens must be a tensor"
        assert tokens.ndim == 1, "Tokens must be a 1D tensor"

        mlp_in_cache, mlp_in_caching_hooks, _ = self.get_caching_hooks(
            lambda name: name.endswith(self.feature_input_hook)
        )
        logits = self.run_with_hooks(tokens, fwd_hooks=mlp_in_caching_hooks)
        mlp_inputs = self._stack_layer_cache(mlp_in_cache, self.feature_input_hook).squeeze(1)

        neuron_locations = []
        neuron_activations = []
        target_encoders = []
        source_vectors = []
        layer_capture_stats = []

        n_pos = tokens.shape[0]
        positions = torch.arange(n_pos, device=tokens.device, dtype=torch.long)
        neuron_ids = torch.arange(self.cfg.d_mlp, device=tokens.device, dtype=torch.long)

        for layer in range(self.cfg.n_layers):
            layer_input = mlp_inputs[layer]
            layer_acts, layer_target_encoders, layer_source_vectors = self._compute_layer_neuron_data(
                layer, layer_input
            )
            layer_source_norms = layer_source_vectors.norm(dim=-1)

            layer_locations = torch.stack(
                [
                    torch.full((n_pos * self.cfg.d_mlp,), layer, device=tokens.device, dtype=torch.long),
                    positions.repeat_interleave(self.cfg.d_mlp),
                    neuron_ids.repeat(n_pos),
                ],
                dim=1,
            )

            flat_source_norms = layer_source_norms.reshape(-1)
            keep_neurons = torch.topk(
                flat_source_norms,
                int(flat_source_norms.numel() * prop_neurons_per_layer),
                dim=0,
            ).indices
            selected_norm_sum = float(flat_source_norms[keep_neurons].sum().item())
            total_norm_sum = float(layer_source_norms.sum().item())
            layer_capture_stats.append(
                {
                    "layer": layer,
                    "selected_neurons": int(keep_neurons.numel()),
                    "total_neurons": int(layer_source_norms.numel()),
                    "selected_residual_write_norm": selected_norm_sum,
                    "total_residual_write_norm": total_norm_sum,
                    "captured_fraction": selected_norm_sum / max(total_norm_sum, 1e-12),
                }
            )

            neuron_locations.append(layer_locations[keep_neurons])
            neuron_activations.append(layer_acts.reshape(-1)[keep_neurons])
            target_encoders.append(layer_target_encoders.reshape(-1, self.cfg.d_model)[keep_neurons])
            source_vectors.append(layer_source_vectors.reshape(-1, self.cfg.d_model)[keep_neurons])

        token_vectors = self.W_E[tokens].detach()

        return AttributionContext(
            logits=logits,
            token_vectors=token_vectors,
            neuron_locations=torch.cat(neuron_locations, dim=0),
            neuron_activations=torch.cat(neuron_activations, dim=0),
            target_encoders=torch.cat(target_encoders, dim=0),
            source_vectors=torch.cat(source_vectors, dim=0),
            layer_capture_stats=layer_capture_stats,
            n_layers=self.cfg.n_layers,
        )

    def __del__(self):
        self.reset_hooks(including_permanent=True)
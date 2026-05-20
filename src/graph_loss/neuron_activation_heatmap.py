"""Neuron activation heatmaps (2D) and logit influence heatmaps (1D) for MLP neurons."""

from __future__ import annotations

import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
import torch.nn.functional as F

from graph_loss.utils import ActivationWriteResult
from utils import default_datasets_dir


def _resolve_dataset_path(dataset_name: str) -> str:
    expanded = os.path.expanduser(dataset_name)
    if os.path.isfile(expanded):
        return os.path.abspath(expanded)

    root = default_datasets_dir()
    candidates = []
    if os.path.dirname(expanded):
        candidates.append(os.path.abspath(expanded))
    else:
        base = os.path.basename(expanded)
        candidates.extend(
            [
                os.path.join(root, base),
                os.path.join(root, f"{base}.json"),
                os.path.join(root, f"{base}_all.json"),
                os.path.join(root, f"{base}_test.json"),
                os.path.join(root, f"{base}_train.json"),
            ],
        )

    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        f"Could not resolve dataset {dataset_name!r}. Tried: {', '.join(candidates)}",
    )


def _parse_numeric_args(prompt: str) -> tuple[int, ...]:
    left = prompt.split("=", 1)[0]
    values = re.findall(r"-?\d+", left)
    if not values:
        raise ValueError(f"No numeric arguments found in prompt {prompt!r}")
    return tuple(int(value) for value in values)


def _select_mlp_neuron_weights(
    model,
    weight: torch.Tensor,
    neuron_ids: torch.Tensor,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    d_model = model.cfg.d_model
    d_mlp = model.cfg.d_mlp
    neuron_ids = neuron_ids.to(device=weight.device, dtype=torch.long)

    if weight.shape == (d_mlp, d_model):
        rows = weight.index_select(0, neuron_ids)
    elif weight.shape == (d_model, d_mlp):
        rows = weight.index_select(1, neuron_ids).T
    else:
        raise ValueError(
            f"Unsupported MLP weight shape {tuple(weight.shape)} for model dims "
            f"(d_model={d_model}, d_mlp={d_mlp})",
        )
    return rows.to(device=device, dtype=dtype)


def _select_mlp_neuron_biases(
    module,
    name: str,
    neuron_ids: torch.Tensor,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    bias = getattr(module, name, None)
    if bias is None:
        return torch.zeros((int(neuron_ids.numel()),), device=device, dtype=dtype)
    neuron_ids = neuron_ids.to(device=bias.device, dtype=torch.long)
    return bias.index_select(0, neuron_ids).to(device=device, dtype=dtype)


@torch.no_grad()
def compute_activation_grid_from_mlp_cache(
    model,
    kept_neuron_locations: torch.Tensor,
    mlp_input_cache: dict,
) -> ActivationWriteResult:
    """Build a 2D activation grid from an in-memory MLP cache — no disk I/O, fully vectorized.

    Neurons are grouped by *layer* so that each layer's CPU tensor is transferred
    to the GPU exactly once per call.  The previous (layer, token_pos) grouping
    forced a strided non-contiguous copy of the full ~800 MB CPU layer tensor for
    every group, causing CPU cache thrashing: ~64 ms/group × up to 128 groups ×
    16 training prompts/step ≈ 130 s of wasted memory latency per training step.
    """
    meta = mlp_input_cache.get("meta", {})
    layer_inputs = mlp_input_cache.get("layer_inputs", {})
    numeric_args_list = meta.get("numeric_args_by_prompt", [])

    if not numeric_args_list:
        raise ValueError("MLP input cache missing numeric_args_by_prompt")

    n_cache_prompts = len(numeric_args_list)
    n_dims = len(numeric_args_list[0])

    arg_values = [
        sorted({args[dim] for args in numeric_args_list})
        for dim in range(n_dims)
    ]
    arg_to_idx = [{v: i for i, v in enumerate(vals)} for vals in arg_values]
    grid_shape = tuple(len(vals) for vals in arg_values)
    grid_cells = 1
    for s in grid_shape:
        grid_cells *= s

    flat_indices = torch.zeros(n_cache_prompts, dtype=torch.long)
    stride = 1
    for dim in range(n_dims - 1, -1, -1):
        dim_indices = torch.tensor(
            [arg_to_idx[dim][args[dim]] for args in numeric_args_list], dtype=torch.long
        )
        flat_indices = flat_indices + dim_indices * stride
        stride *= grid_shape[dim]

    n_kept = int(kept_neuron_locations.shape[0])
    activation_grid = torch.full((n_kept, grid_cells), float("nan"), dtype=torch.float32)

    # Group by LAYER only.  This lets us load each layer's CPU tensor to the GPU
    # once (one contiguous DMA transfer) instead of once per (layer, token_pos)
    # group.  Slicing by token_pos happens on the GPU after the transfer — fast,
    # because GPU memory bandwidth is ~2 TB/s vs ~50 GB/s for CPU with poor
    # cache locality on large strided tensors.
    layer_to_neurons: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for loc_idx in range(n_kept):
        layer = int(kept_neuron_locations[loc_idx, 0].item())
        token_pos = int(kept_neuron_locations[loc_idx, 1].item())
        neuron_id = int(kept_neuron_locations[loc_idx, 2].item())
        layer_to_neurons[layer].append((loc_idx, token_pos, neuron_id))

    device = model.cfg.device

    for layer, layer_members in layer_to_neurons.items():
        if layer >= len(layer_inputs):
            continue
        layer_tensor = layer_inputs[layer]  # [n_cache_prompts, n_positions, d_model] on CPU
        n_positions = int(layer_tensor.shape[1])

        # One contiguous CPU→GPU transfer for the entire layer tensor.
        # Previously we called layer_tensor[:, token_pos, :].to(device) per group,
        # which forced PyTorch to walk the full (~800 MB) non-contiguous tensor on CPU
        # to produce a 40 MB contiguous copy before the GPU transfer — O(800 MB) of
        # DRAM reads per group instead of O(40 MB).
        layer_tensor_gpu = layer_tensor.to(device=device)  # [n_cache_prompts, n_positions, d_model]

        # Sub-group by token_pos within this layer; all slicing now happens on GPU.
        pos_to_members: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for loc_idx, token_pos, neuron_id in layer_members:
            if token_pos < n_positions:
                pos_to_members[token_pos].append((loc_idx, neuron_id))

        for token_pos, pos_members in pos_to_members.items():
            # Contiguous slice on GPU — negligible cost at 2 TB/s.
            mlp_input_at_pos = layer_tensor_gpu[:, token_pos, :].contiguous()

            loc_indices = [loc_idx for loc_idx, _ in pos_members]
            neuron_ids = torch.tensor(
                [nid for _, nid in pos_members], dtype=torch.long, device=device
            )

            old_mlp = model.blocks[layer].mlp.old_mlp
            gate_w = _select_mlp_neuron_weights(
                model, old_mlp.W_gate, neuron_ids, device=device, dtype=mlp_input_at_pos.dtype
            )
            up_w = _select_mlp_neuron_weights(
                model, old_mlp.W_in, neuron_ids, device=device, dtype=mlp_input_at_pos.dtype
            )
            gate_b = _select_mlp_neuron_biases(
                old_mlp, "b_gate", neuron_ids, device=device, dtype=mlp_input_at_pos.dtype
            )
            up_b = _select_mlp_neuron_biases(
                old_mlp, "b_in", neuron_ids, device=device, dtype=mlp_input_at_pos.dtype
            )

            # [n_cache_prompts, n_group_neurons]
            acts = F.silu(mlp_input_at_pos @ gate_w.T + gate_b) * (mlp_input_at_pos @ up_w.T + up_b)
            acts = acts.detach().float().cpu()  # [n_prompts, n_group_neurons]

            # Each prompt maps 1-to-1 to a grid cell; write directly without averaging.
            for group_i, loc_idx in enumerate(loc_indices):
                activation_grid[loc_idx, flat_indices] = acts[:, group_i]

        del layer_tensor_gpu  # free GPU memory before loading the next layer

    activations = activation_grid.reshape(n_kept, *grid_shape)

    return ActivationWriteResult(
        activations=activations,
        arg_values=arg_values,
    )


@torch.no_grad()
def build_neuron_activation_write_result(
    model,
    dataset_name: str,
    neuron_locations: list[tuple[int, int, int]] | torch.Tensor,
    *,
    mlp_input_cache: dict | None = None,
) -> ActivationWriteResult:
    """Return a 2D activation grid for each requested neuron.

    If ``mlp_input_cache`` is provided it is used directly.  Otherwise a
    temporary MLP input cache is built from the dataset and discarded after use.
    """
    dataset_path = _resolve_dataset_path(dataset_name)

    if isinstance(neuron_locations, torch.Tensor):
        kept_neuron_locations = neuron_locations.detach().cpu()
    else:
        kept_neuron_locations = torch.tensor(neuron_locations, dtype=torch.long)

    if mlp_input_cache is None:
        from graph_loss.precompute_mlp_inputs import build_mlp_input_cache
        model_name = getattr(model.cfg, "model_name", "model")
        mlp_input_cache = build_mlp_input_cache(model, dataset_path, model_name)

    try:
        return compute_activation_grid_from_mlp_cache(model, kept_neuron_locations, mlp_input_cache)
    except Exception as exc:
        raise RuntimeError(f"Failed to compute activations from MLP cache: {exc}") from exc


HEATMAP_VALUE_LABEL = "activation"


def save_supernode_activation_heatmap_pdf(
    activation_grids: torch.Tensor,
    arg_values: list[list[int]],
    members: list[int],
    neuron_locations: torch.Tensor,
    *,
    output_path: str,
    title: str,
    member_labels: dict[int, list[str]] | None = None,
    member_number_unembed: dict[int, tuple[list[int], torch.Tensor]] | None = None,
    member_specificity: dict[int, float] | None = None,
    member_norm_props: dict[int, float] | None = None,
) -> str:
    """Save one 2D activation heatmap page per neuron in a supernode, with optional 1D logit influence side panel."""
    if len(arg_values) != 2:
        raise ValueError(
            f"save_supernode_activation_heatmap_pdf expects 2D activation grids, "
            f"got {len(arg_values)} arg dimensions"
        )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    activation_grids = activation_grids.detach().float().cpu()
    if activation_grids.shape[0] != len(members):
        raise ValueError(
            f"Expected one activation grid per member, got {activation_grids.shape[0]} "
            f"grids for {len(members)} members"
        )
    locations_cpu = neuron_locations.detach().cpu()
    member_locations = {}
    for member_idx, graph_neuron_idx in enumerate(members):
        layer = int(locations_cpu[graph_neuron_idx, 0].item())
        token_pos = int(locations_cpu[graph_neuron_idx, 1].item())
        neuron_id = int(locations_cpu[graph_neuron_idx, 2].item())
        member_locations[member_idx] = (
            f"idx={graph_neuron_idx} layer={layer} token={token_pos} neuron={neuron_id}",
            neuron_id,
        )

    xs = arg_values[0]
    ys = arg_values[1]

    with PdfPages(output_path) as pdf:
        for member_idx, activation_grid in enumerate(activation_grids):
            location_text, neuron_id = member_locations[member_idx]
            graph_neuron_idx = int(members[member_idx])
            labels = member_labels.get(graph_neuron_idx, []) if member_labels is not None else []
            label_text = f"\nANOVA labels: {', '.join(labels)}" if labels else ""
            norm_prop = (
                member_norm_props.get(graph_neuron_idx) if member_norm_props is not None else None
            )
            norm_text = f"  ({norm_prop * 100:.2f}% of total residual norm)" if norm_prop is not None else ""
            page_title = f"{title}{label_text}\nNeuron {neuron_id} ({location_text}){norm_text}"
            number_unembed = (
                member_number_unembed.get(graph_neuron_idx)
                if member_number_unembed is not None
                else None
            )

            if number_unembed is None:
                fig, ax = plt.subplots(figsize=(8, 6))
                side_ax = None
            else:
                fig, axes = plt.subplots(
                    1,
                    2,
                    figsize=(14, 6),
                    gridspec_kw={"width_ratios": [2.0, 1.0]},
                )
                ax, side_ax = axes

            if torch.isnan(activation_grid).all():
                ax.text(0.5, 0.5, "No valid activations", ha="center", va="center")
                ax.set_axis_off()
            else:
                heatmap = activation_grid.T.contiguous()
                image = ax.imshow(
                    heatmap.numpy(),
                    origin="lower",
                    aspect="auto",
                    extent=[min(xs), max(xs), min(ys), max(ys)],
                )
                fig.colorbar(image, ax=ax, label=HEATMAP_VALUE_LABEL)
                ax.set_xlabel("arg 1")
                ax.set_ylabel("arg 2")

            ax.set_title(page_title)

            if side_ax is not None and number_unembed is not None:
                number_values, unembed_values = number_unembed
                unembed_heatmap = unembed_values.detach().float().cpu().unsqueeze(0)
                image = side_ax.imshow(
                    unembed_heatmap.numpy(),
                    origin="lower",
                    aspect="auto",
                    extent=[min(number_values), max(number_values), 0, 1],
                )
                fig.colorbar(image, ax=side_ax, label="W_out @ W_U")
                side_ax.set_xlabel("number token")
                side_ax.set_yticks([])
                side_ax.set_title("number-token unembed")

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        if member_specificity:
            specificity_items = [
                (int(member), float(value))
                for member, value in member_specificity.items()
            ]
            if specificity_items:
                specificity_members = [member for member, _value in specificity_items]
                specificity_values = [value for _member, value in specificity_items]
                fig, ax = plt.subplots(figsize=(10, 4))
                xs_bar = list(range(len(specificity_values)))
                ax.bar(xs_bar, specificity_values)
                ax.set_xticks(xs_bar)
                ax.set_xticklabels([str(member) for member in specificity_members], rotation=90)
                ax.set_xlabel("graph neuron index")
                ax.set_ylabel("ranking score")
                ax.set_title(f"{title}\nSorted ANOVA ranking score")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    return output_path

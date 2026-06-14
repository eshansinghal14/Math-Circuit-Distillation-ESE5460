"""Neuron activation heatmaps (2D) and logit influence heatmaps (1D) for MLP neurons."""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
import torch.nn.functional as F

from graph_loss.utils import ActivationWriteResult


def _parse_numeric_args(prompt: str) -> tuple[int, ...]:
    left = prompt.split("=", 1)[0]
    values = re.findall(r"-?\d+", left)
    if not values:
        raise ValueError(f"No numeric arguments found in prompt {prompt!r}")
    return tuple(int(value) for value in values)



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

    # Group by layer so we transfer each layer's CPU tensor to the GPU once.
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
        layer_tensor = layer_inputs[layer]  # [n_prompts, n_positions, d_model] on CPU
        n_positions = int(layer_tensor.shape[1])

        # Filter out-of-range token positions before the GPU transfer.
        layer_members = [(li, tp, nid) for li, tp, nid in layer_members if tp < n_positions]
        if not layer_members:
            continue

        # One contiguous CPU→GPU transfer for the whole layer.
        layer_tensor_gpu = layer_tensor.to(device=device)  # [n_prompts, n_positions, d_model]
        dtype = layer_tensor_gpu.dtype

        # Gather all neuron ids for this layer at once.
        hf_mlp = model.layers[layer].mlp
        all_neuron_ids = torch.tensor(
            [nid for _, _, nid in layer_members], dtype=torch.long, device=device
        )  # [n_layer_kept]
        token_pos_idx = torch.tensor(
            [tp for _, tp, _ in layer_members], dtype=torch.long, device=device
        )  # [n_layer_kept]
        loc_indices = torch.tensor(
            [li for li, _, _ in layer_members], dtype=torch.long
        )  # [n_layer_kept], CPU for indexing activation_grid

        # Select weights for all kept neurons in the layer at once.
        gate_w = hf_mlp.gate_proj.weight[all_neuron_ids].to(dtype=dtype)  # [n_layer_kept, d_model]
        up_w = hf_mlp.up_proj.weight[all_neuron_ids].to(dtype=dtype)      # [n_layer_kept, d_model]
        gate_bias_full = getattr(hf_mlp.gate_proj, "bias", None)
        up_bias_full = getattr(hf_mlp.up_proj, "bias", None)
        gate_b = (
            gate_bias_full[all_neuron_ids].to(dtype=dtype)
            if gate_bias_full is not None
            else torch.zeros(len(layer_members), device=device, dtype=dtype)
        )  # [n_layer_kept]
        up_b = (
            up_bias_full[all_neuron_ids].to(dtype=dtype)
            if up_bias_full is not None
            else torch.zeros(len(layer_members), device=device, dtype=dtype)
        )  # [n_layer_kept]

        # One matmul covers all positions and all neurons in the layer simultaneously.
        # gate_out[p, pos, j] = layer_tensor_gpu[p, pos, :] · gate_w[j, :]
        gate_out = layer_tensor_gpu @ gate_w.T + gate_b  # [n_prompts, n_positions, n_layer_kept]
        up_out   = layer_tensor_gpu @ up_w.T   + up_b   # [n_prompts, n_positions, n_layer_kept]

        # Pick the right token position per neuron via advanced indexing.
        # j_idx and token_pos_idx broadcast to select acts[p, token_pos_j, j] for each j.
        j_idx = torch.arange(len(layer_members), device=device)
        neuron_acts = F.silu(gate_out[:, token_pos_idx, j_idx]) * up_out[:, token_pos_idx, j_idx]
        # [n_prompts, n_layer_kept]

        # Scatter all neurons into activation_grid at once.
        # activation_grid[loc_idx_j, flat_idx_p] = neuron_acts[p, j]
        neuron_acts_cpu = neuron_acts.detach().float().cpu()  # [n_prompts, n_layer_kept]
        activation_grid[loc_indices[:, None], flat_indices[None, :]] = neuron_acts_cpu.T

        del layer_tensor_gpu, gate_out, up_out, neuron_acts  # free GPU memory before next layer

    activations = activation_grid.reshape(n_kept, *grid_shape)

    return ActivationWriteResult(
        activations=activations,
        arg_values=arg_values,
    )


@torch.no_grad()
def label_neurons_layer_by_layer(
    model,
    neuron_locations: torch.Tensor,
    mlp_input_cache: dict,
    *,
    target_args: tuple[int, ...] | None = None,
    anova_range_radius: int = 0,
    anova_neuron_chunk: int = 256,
) -> list:
    """ANOVA-label N neurons across all model layers with pipelined H2D transfers and one D2H flush.

    All CPU→GPU transfers are queued simultaneously via non_blocking=True so the
    DMA engine can pipeline them.  Per-neuron activations stay on GPU until a single D2H transfer
    at the end, eliminating per-layer host synchronisation points.
    Returns a list[NodeLabel] in the same order as neuron_locations.
    """
    from graph_loss.anova_node_labels import (
        NodeLabel,
        build_anova_basis_rules,
        build_gpu_anova_state,
        gpu_label_activation_heatmaps,
    )

    meta = mlp_input_cache.get("meta", {})
    layer_inputs = mlp_input_cache.get("layer_inputs", {})
    numeric_args_list = meta.get("numeric_args_by_prompt", [])
    if not numeric_args_list:
        raise ValueError("MLP input cache missing numeric_args_by_prompt")

    n_prompts = len(numeric_args_list)
    n_dims = len(numeric_args_list[0])
    arg_values = [sorted({args[dim] for args in numeric_args_list}) for dim in range(n_dims)]
    arg_to_idx = [{v: i for i, v in enumerate(vals)} for vals in arg_values]
    grid_shape = tuple(len(vals) for vals in arg_values)
    grid_cells = 1
    for s in grid_shape:
        grid_cells *= s

    flat_indices = torch.zeros(n_prompts, dtype=torch.long)
    stride = 1
    for dim in range(n_dims - 1, -1, -1):
        dim_indices = torch.tensor(
            [arg_to_idx[dim][args[dim]] for args in numeric_args_list], dtype=torch.long
        )
        flat_indices = flat_indices + dim_indices * stride
        stride *= grid_shape[dim]

    n_kept = int(neuron_locations.shape[0])
    label_results: list = [None] * n_kept

    layer_to_neurons: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for loc_idx, (layer, token_pos, neuron_id) in enumerate(neuron_locations.tolist()):
        layer_to_neurons[int(layer)].append((loc_idx, int(token_pos), int(neuron_id)))

    device = model.cfg.device
    empty = NodeLabel(labels=[], scores={}, categories={}, category_scores={}, category_specificity={})
    sorted_layers = sorted(layer_to_neurons.keys())

    # Move flat_indices to GPU once; reused for every layer's scatter operation.
    flat_indices_gpu = flat_indices.to(device=device)

    # Build ANOVA rules + GPU state once; masks and indicator matrices are reused every batch.
    anova_rules = build_anova_basis_rules(
        arg_values,
        target_args=target_args,
        anova_range_radius=anova_range_radius,
    )
    gpu_anova_state = build_gpu_anova_state(anova_rules, device, grid_shape=grid_shape)

    valid_batch_layers = [l for l in sorted_layers if l < len(layer_inputs)]
    if not valid_batch_layers:
        return [empty] * n_kept

    total_neurons_labeled = 0

    # Process one layer at a time; run ANOVA immediately per-layer so grids never accumulate.
    for layer in valid_batch_layers:
        layer_tensor_gpu = layer_inputs[layer].to(device=device)  # [P, n_positions, d]
        n_positions = int(layer_tensor_gpu.shape[1])
        dtype = layer_tensor_gpu.dtype
        d = layer_tensor_gpu.shape[-1]

        # Group this layer's neurons by token_pos so all neurons at the same position
        # share one [P, d] input slice, enabling a single bmm per layer.
        layer_group_map: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for li, tp, nid in layer_to_neurons[layer]:
            if tp < n_positions:
                layer_group_map[tp].append((li, nid))

        if not layer_group_map:
            del layer_tensor_gpu
            continue

        token_positions = list(layer_group_map.keys())
        n_groups = len(token_positions)
        N_max = max(len(v) for v in layer_group_map.values())

        X_batch = torch.stack(
            [layer_tensor_gpu[:, tp, :] for tp in token_positions], dim=0
        )  # [n_groups, P, d]
        del layer_tensor_gpu

        W_gate = torch.zeros(n_groups, N_max, d, device=device, dtype=dtype)
        W_up   = torch.zeros(n_groups, N_max, d, device=device, dtype=dtype)
        b_gate = torch.zeros(n_groups, N_max,    device=device, dtype=dtype)
        b_up   = torch.zeros(n_groups, N_max,    device=device, dtype=dtype)

        hf_mlp = model.layers[layer].mlp
        for g_idx, tp in enumerate(token_positions):
            neurons = layer_group_map[tp]
            n_g = len(neurons)
            nids = torch.tensor([nid for _, nid in neurons], dtype=torch.long, device=device)
            W_gate[g_idx, :n_g] = hf_mlp.gate_proj.weight[nids].to(dtype=dtype)
            W_up[g_idx, :n_g]   = hf_mlp.up_proj.weight[nids].to(dtype=dtype)
            gate_bias_full = getattr(hf_mlp.gate_proj, "bias", None)
            up_bias_full   = getattr(hf_mlp.up_proj,   "bias", None)
            if gate_bias_full is not None:
                b_gate[g_idx, :n_g] = gate_bias_full[nids].to(dtype=dtype)
            if up_bias_full is not None:
                b_up[g_idx, :n_g] = up_bias_full[nids].to(dtype=dtype)

        gate_out = torch.bmm(X_batch, W_gate.permute(0, 2, 1)) + b_gate.unsqueeze(1)
        up_out   = torch.bmm(X_batch, W_up.permute(0, 2, 1))   + b_up.unsqueeze(1)
        del X_batch, W_gate, W_up, b_gate, b_up

        neuron_acts_batch = F.silu(gate_out) * up_out  # [n_groups, P, N_max]
        del gate_out, up_out

        n_labeled_layer = 0
        for g_idx, tp in enumerate(token_positions):
            neurons = layer_group_map[tp]
            n_g = len(neurons)
            acts_g = neuron_acts_batch[g_idx, :, :n_g].float()  # [P, n_g]
            for c_start in range(0, n_g, anova_neuron_chunk):
                chunk_neurons = neurons[c_start:c_start + anova_neuron_chunk]
                acts_c = acts_g[:, c_start:c_start + anova_neuron_chunk]  # [P, chunk]
                n_c = acts_c.shape[1]
                grid = torch.full((n_c, grid_cells), float("nan"), dtype=torch.float32, device=device)
                grid[:, flat_indices_gpu] = acts_c.T
                chunk_labels = gpu_label_activation_heatmaps(grid, gpu_anova_state, anova_rules)
                del grid
                for label_j, (li, _) in enumerate(chunk_neurons):
                    label_results[li] = chunk_labels[label_j]
                n_labeled_layer += n_c
        del neuron_acts_batch
        total_neurons_labeled += n_labeled_layer

    logger.info(
        "  ANOVA labeled %d layers (%d neurons)",
        len(valid_batch_layers),
        total_neurons_labeled,
    )

    return [lbl if lbl is not None else empty for lbl in label_results]


@torch.no_grad()
def build_neuron_activation_write_result(
    model,
    neuron_locations: list[tuple[int, int, int]] | torch.Tensor,
    *,
    mlp_input_cache: dict | None = None,
    dataset: str = "22_add",
) -> ActivationWriteResult:
    """Return an activation grid for each requested neuron.

    If ``mlp_input_cache`` is provided it is used directly.  Otherwise a
    temporary MLP input cache is built from ``dataset``.
    """
    if isinstance(neuron_locations, torch.Tensor):
        kept_neuron_locations = neuron_locations.detach().cpu()
    else:
        kept_neuron_locations = torch.tensor(neuron_locations, dtype=torch.long)

    if mlp_input_cache is None:
        from graph_loss.precompute_mlp_inputs import build_mlp_input_cache
        from utils import load_split
        all_data = load_split(dataset, "all")
        model_name = getattr(model.cfg, "model_name", "model")
        mlp_input_cache = build_mlp_input_cache(model, dataset, model_name, data_dict=all_data)

    try:
        return compute_activation_grid_from_mlp_cache(model, kept_neuron_locations, mlp_input_cache)
    except Exception as exc:
        raise RuntimeError(f"Failed to compute activations from MLP cache: {exc}") from exc


HEATMAP_VALUE_LABEL = "activation"

_ARG_SINGLE_RE = re.compile(r"^arg(\d+)\s+(?:range|units)(?:\s+\d+)?$")
_ARG_JOINT_RE = re.compile(r"^arg(\d+)\s+(?:range|units)\s+and\s+arg(\d+)\s+(?:range|units)$")
_ARG_COMBO_SUM_RE = re.compile(r"^((?:arg\d+\s+)+)sum\s+range$")


def _parse_category_dims(category: str | None, n_dims: int) -> list[int]:
    """Return sorted dim indices (0-based) that are active for this ANOVA category.

    The returned list drives heatmap dimensionality: len 1 → line plot, 2 → imshow, 3 → 3-D scatter.
    Remaining dims are averaged out before plotting.
    """
    if category is None or n_dims <= 1:
        return list(range(n_dims))
    if category in ("sum range", "sum units"):
        return list(range(n_dims))
    if category == "carry":
        return [i for i in [0, 1] if i < n_dims]
    m = _ARG_SINGLE_RE.match(category)
    if m:
        d = int(m.group(1)) - 1
        return [d] if d < n_dims else list(range(n_dims))
    m = _ARG_JOINT_RE.match(category)
    if m:
        dims = sorted({int(m.group(1)) - 1, int(m.group(2)) - 1})
        valid = [d for d in dims if d < n_dims]
        return valid if len(valid) >= 2 else list(range(n_dims))
    # Handles "arg1 arg2 sum range", "arg1 arg3 sum range", "arg1 arg2 arg3 sum range", etc.
    m = _ARG_COMBO_SUM_RE.match(category)
    if m:
        dims = sorted({int(x) - 1 for x in re.findall(r"arg(\d+)", m.group(1))})
        valid = [d for d in dims if d < n_dims]
        return valid if valid else list(range(n_dims))
    return list(range(n_dims))


def save_supernode_activation_heatmap_pdf(
    activation_grids: torch.Tensor,
    arg_values: list[list[int]],
    members: list[int],
    neuron_locations: torch.Tensor,
    *,
    output_path: str,
    title: str,
    supernode_category: str | None = None,
    member_labels: dict[int, list[str]] | None = None,
    member_number_unembed: dict[int, tuple[list[int], torch.Tensor]] | None = None,
    member_specificity: dict[int, float] | None = None,
    member_norm_props: dict[int, float] | None = None,
    member_var_spec: dict[int, tuple[float, float]] | None = None,
    member_dla_kl: dict[int, float] | None = None,
) -> str:
    """Save one activation heatmap page per neuron: 1-D line, 2-D imshow, or 3-D scatter based on supernode_category."""
    n_dims = len(arg_values)
    if n_dims < 1:
        raise ValueError(
            f"save_supernode_activation_heatmap_pdf expects at least 1 arg dimension, got {n_dims}"
        )

    plot_dims = _parse_category_dims(supernode_category, n_dims)

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
            score_parts = []
            if member_var_spec is not None:
                vs = member_var_spec.get(graph_neuron_idx)
                if vs is not None:
                    score_parts.append(f"var={vs[0]:.3f} spec={vs[1]:.3f}")
            if member_dla_kl is not None:
                kl = member_dla_kl.get(graph_neuron_idx)
                if kl is not None:
                    score_parts.append(f"dla_kl={kl:.3f}")
            score_line = ("\n" + "  ".join(score_parts)) if score_parts else ""
            page_title = f"{title}{label_text}\nNeuron {neuron_id} ({location_text}){norm_text}{score_line}"

            number_unembed = (
                member_number_unembed.get(graph_neuron_idx)
                if member_number_unembed is not None
                else None
            )

            if len(plot_dims) == 1:
                # 1-D heatmap strip: average over all other dims, display as 1-row imshow.
                d = plot_dims[0]
                grid_1d = activation_grid
                for dim in sorted([i for i in range(n_dims) if i != d], reverse=True):
                    grid_1d = torch.nanmean(grid_1d, dim=dim)
                x_vals = arg_values[d]
                fig, ax = plt.subplots(figsize=(8, 2))
                if torch.isnan(grid_1d).all():
                    ax.text(0.5, 0.5, "No valid activations", ha="center", va="center")
                    ax.set_axis_off()
                else:
                    strip = grid_1d.unsqueeze(0).numpy()  # (1, N)
                    image = ax.imshow(
                        strip,
                        origin="lower",
                        aspect="auto",
                        extent=[min(x_vals), max(x_vals), -0.5, 0.5],
                        cmap="viridis",
                    )
                    fig.colorbar(image, ax=ax, label=HEATMAP_VALUE_LABEL)
                    ax.set_xlabel(f"arg{d + 1}")
                    ax.set_yticks([])
                ax.set_title(page_title)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                continue

            if len(plot_dims) >= 3:
                # 3-D scatter: use first three plot dims, average any extras.
                d0, d1, d2 = plot_dims[0], plot_dims[1], plot_dims[2]
                grid_3d = activation_grid
                for dim in sorted([i for i in range(n_dims) if i not in {d0, d1, d2}], reverse=True):
                    grid_3d = torch.nanmean(grid_3d, dim=dim)
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
                xs_3d, ys_3d, zs_3d = arg_values[d0], arg_values[d1], arg_values[d2]
                xs_g, ys_g, zs_g = np.meshgrid(xs_3d, ys_3d, zs_3d, indexing="ij")
                c_flat = grid_3d.numpy().ravel()
                valid = ~np.isnan(c_flat)
                if number_unembed is None:
                    fig = plt.figure(figsize=(10, 8))
                    ax3d = fig.add_subplot(111, projection="3d")
                    side_ax_3d = None
                else:
                    fig = plt.figure(figsize=(16, 8))
                    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
                    side_ax_3d = fig.add_subplot(1, 2, 2)
                if valid.any():
                    sc = ax3d.scatter(
                        xs_g.ravel()[valid], ys_g.ravel()[valid], zs_g.ravel()[valid],
                        c=c_flat[valid], cmap="viridis", alpha=0.7,
                    )
                    fig.colorbar(sc, ax=ax3d, label=HEATMAP_VALUE_LABEL, shrink=0.6)
                else:
                    ax3d.text(0.5, 0.5, 0.5, "No valid activations", ha="center", va="center")
                ax3d.set_xlabel(f"arg{d0 + 1}")
                ax3d.set_ylabel(f"arg{d1 + 1}")
                ax3d.set_zlabel(f"arg{d2 + 1}")
                ax3d.set_title(page_title)
                if side_ax_3d is not None and number_unembed is not None:
                    number_values, unembed_values = number_unembed
                    unembed_heatmap = unembed_values.detach().float().cpu().unsqueeze(0)
                    image = side_ax_3d.imshow(
                        unembed_heatmap.numpy(),
                        origin="lower",
                        aspect="auto",
                        extent=[min(number_values), max(number_values), 0, 1],
                    )
                    fig.colorbar(image, ax=side_ax_3d, label="W_out @ W_U")
                    side_ax_3d.set_xlabel("number token")
                    side_ax_3d.set_yticks([])
                    side_ax_3d.set_title("number-token unembed")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                continue

            # 2-D heatmap: average over any non-plot dims, then imshow.
            d0, d1 = plot_dims[0], plot_dims[1]
            grid_2d = activation_grid
            for dim in sorted([i for i in range(n_dims) if i not in {d0, d1}], reverse=True):
                grid_2d = torch.nanmean(grid_2d, dim=dim)
            xs_2d = arg_values[d0]
            ys_2d = arg_values[d1]

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

            if torch.isnan(grid_2d).all():
                ax.text(0.5, 0.5, "No valid activations", ha="center", va="center")
                ax.set_axis_off()
            else:
                heatmap = grid_2d.T.contiguous()
                image = ax.imshow(
                    heatmap.numpy(),
                    origin="lower",
                    aspect="auto",
                    extent=[min(xs_2d), max(xs_2d), min(ys_2d), max(ys_2d)],
                )
                fig.colorbar(image, ax=ax, label=HEATMAP_VALUE_LABEL)
                ax.set_xlabel(f"arg{d0 + 1}")
                ax.set_ylabel(f"arg{d1 + 1}")

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


def save_dla_heatmap_pdf(
    members: list[int],
    neuron_locations: torch.Tensor,
    member_number_unembed: dict[int, tuple[list[int], torch.Tensor]],
    *,
    output_path: str,
    title: str,
    member_labels: dict[int, list[str]] | None = None,
    member_norm_props: dict[int, float] | None = None,
) -> str:
    """Save a 1-D DLA-influence bar chart per neuron (W_out @ W_U over token IDs 0–200).

    Used for DLA supernodes where no 2-D arg1×arg2 activation grid is needed.
    Each page shows one neuron's write-vector projected through W_U for every
    single-token number representation in [0, 200].
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    locations_cpu = neuron_locations.detach().cpu()
    member_locations: dict[int, tuple[str, int]] = {}
    for member_idx, graph_neuron_idx in enumerate(members):
        layer = int(locations_cpu[graph_neuron_idx, 0].item())
        token_pos = int(locations_cpu[graph_neuron_idx, 1].item())
        neuron_id = int(locations_cpu[graph_neuron_idx, 2].item())
        member_locations[member_idx] = (
            f"idx={graph_neuron_idx} layer={layer} token={token_pos} neuron={neuron_id}",
            neuron_id,
        )

    with PdfPages(output_path) as pdf:
        for member_idx, graph_neuron_idx in enumerate(members):
            location_text, neuron_id = member_locations[member_idx]
            labels = (
                member_labels.get(graph_neuron_idx, []) if member_labels is not None else []
            )
            label_text = f"\nANOVA labels: {', '.join(labels)}" if labels else ""
            norm_prop = (
                member_norm_props.get(graph_neuron_idx)
                if member_norm_props is not None
                else None
            )
            norm_text = (
                f"  ({norm_prop * 100:.2f}% of total residual norm)"
                if norm_prop is not None
                else ""
            )
            page_title = (
                f"{title}{label_text}\nNeuron {neuron_id} ({location_text}){norm_text}"
            )

            nu = member_number_unembed.get(graph_neuron_idx)

            fig, ax = plt.subplots(figsize=(12, 3))
            if nu is not None:
                number_values, unembed_values = nu
                vals = unembed_values.detach().float().cpu().numpy()
                ax.bar(number_values, vals, width=0.9, color="steelblue")
                ax.set_xlabel("number token (0–200)")
                ax.set_ylabel("W_out @ W_U")
                ax.set_xlim(min(number_values) - 0.5, max(number_values) + 0.5)
                ax.grid(True, alpha=0.3, axis="y")
            else:
                ax.text(0.5, 0.5, "No DLA data available", ha="center", va="center")
                ax.set_axis_off()
            ax.set_title(page_title)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    return output_path

"""Neuron activation heatmaps (2D) and logit influence heatmaps (1D) for MLP neurons."""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

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
    labelling_layer_batch_size: int = 4,
) -> list:
    """ANOVA-label N neurons in layer-batches with pipelined H2D transfers and one D2H flush per batch.

    All CPU→GPU transfers within a batch are queued simultaneously via non_blocking=True so the
    DMA engine can pipeline them.  Per-neuron activations stay on GPU until a single D2H transfer
    at the end of each batch, eliminating per-layer host synchronisation points.
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
    for loc_idx in range(n_kept):
        layer = int(neuron_locations[loc_idx, 0].item())
        token_pos = int(neuron_locations[loc_idx, 1].item())
        neuron_id = int(neuron_locations[loc_idx, 2].item())
        layer_to_neurons[layer].append((loc_idx, token_pos, neuron_id))

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
    gpu_anova_state = build_gpu_anova_state(anova_rules, device)

    for batch_start in range(0, len(sorted_layers), labelling_layer_batch_size):
        batch_layers = sorted_layers[batch_start : batch_start + labelling_layer_batch_size]
        valid_batch_layers = [l for l in batch_layers if l < len(layer_inputs)]
        if not valid_batch_layers:
            continue

        # Queue ALL H2D transfers for this batch simultaneously with non_blocking=True.
        # The DMA engine can pipeline them, hiding latency while GPU kernels run for
        # already-transferred layers.
        layer_tensors_gpu: dict[int, torch.Tensor] = {
            l: layer_inputs[l].to(device=device, non_blocking=True)
            for l in valid_batch_layers
        }

        # Group neurons by (layer, token_pos): neurons in the same group share an identical
        # input vector X^l[:, τ, :], so all groups can be computed with one bmm.
        # n_groups = unique (layer, tp) pairs — O(B_layers × unique_tps), not O(n_total_neurons),
        # so X_batch: [n_groups, P, d] is tiny compared to the previous [n_neurons, P, d] attempt.
        group_map: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for layer in valid_batch_layers:
            n_positions = int(layer_tensors_gpu[layer].shape[1])
            for li, tp, nid in layer_to_neurons[layer]:
                if tp < n_positions:
                    group_map[(layer, tp)].append((li, nid))

        if not group_map:
            for t in layer_tensors_gpu.values():
                del t
            continue

        groups = list(group_map.keys())
        n_groups = len(groups)
        N_max = max(len(v) for v in group_map.values())

        # Extract one [P, d] slice per group from the already-transferred layer tensors.
        dtype = next(iter(layer_tensors_gpu.values())).dtype
        d = next(iter(layer_tensors_gpu.values())).shape[-1]

        X_batch = torch.stack(
            [layer_tensors_gpu[layer][:, tp, :] for (layer, tp) in groups], dim=0
        )  # [n_groups, P, d]
        del layer_tensors_gpu

        # Build padded weight matrices: [n_groups, N_max, d].
        # Padding is zero so padded outputs are silu(0)*0 = 0; sliced away before grids.
        W_gate = torch.zeros(n_groups, N_max, d, device=device, dtype=dtype)
        W_up   = torch.zeros(n_groups, N_max, d, device=device, dtype=dtype)
        b_gate = torch.zeros(n_groups, N_max,    device=device, dtype=dtype)
        b_up   = torch.zeros(n_groups, N_max,    device=device, dtype=dtype)

        for g_idx, (layer, tp) in enumerate(groups):
            neurons = group_map[(layer, tp)]
            n_g = len(neurons)
            nids = torch.tensor([nid for _, nid in neurons], dtype=torch.long, device=device)
            hf_mlp = model.layers[layer].mlp
            W_gate[g_idx, :n_g] = hf_mlp.gate_proj.weight[nids].to(dtype=dtype)
            W_up[g_idx, :n_g]   = hf_mlp.up_proj.weight[nids].to(dtype=dtype)
            gate_bias_full = getattr(hf_mlp.gate_proj, "bias", None)
            up_bias_full   = getattr(hf_mlp.up_proj,   "bias", None)
            if gate_bias_full is not None:
                b_gate[g_idx, :n_g] = gate_bias_full[nids].to(dtype=dtype)
            if up_bias_full is not None:
                b_up[g_idx, :n_g] = up_bias_full[nids].to(dtype=dtype)

        # Single bmm: [n_groups, P, d] x [n_groups, d, N_max] -> [n_groups, P, N_max]
        gate_out = torch.bmm(X_batch, W_gate.permute(0, 2, 1)) + b_gate.unsqueeze(1)
        up_out   = torch.bmm(X_batch, W_up.permute(0, 2, 1))   + b_up.unsqueeze(1)
        del X_batch, W_gate, W_up, b_gate, b_up

        neuron_acts_batch = F.silu(gate_out) * up_out  # [n_groups, P, N_max]
        del gate_out, up_out

        # Scatter each group's activations into GPU-resident grids.
        batch_grids_gpu: list[torch.Tensor] = []
        batch_loc_indices: list[int] = []
        for g_idx, g in enumerate(groups):
            neurons = group_map[g]
            n_g = len(neurons)
            acts = neuron_acts_batch[g_idx, :, :n_g].float()  # [P, n_g]
            grid = torch.full((n_g, grid_cells), float("nan"), dtype=torch.float32, device=device)
            grid[:, flat_indices_gpu] = acts.T
            batch_grids_gpu.append(grid)
            batch_loc_indices.extend(li for li, _ in neurons)

        del neuron_acts_batch

        # Run ANOVA matmuls on GPU; only [N, R] scores are transferred to CPU.
        batch_grid_gpu = torch.cat(batch_grids_gpu, dim=0)  # [N_batch, grid_cells]
        del batch_grids_gpu
        batch_labels = gpu_label_activation_heatmaps(batch_grid_gpu, gpu_anova_state, anova_rules)
        del batch_grid_gpu
        for label_j, loc_idx in enumerate(batch_loc_indices):
            label_results[loc_idx] = batch_labels[label_j]

        logger.info(
            "  ANOVA labeled layers %s (%d groups, %d neurons)",
            valid_batch_layers,
            n_groups,
            len(batch_loc_indices),
        )

    return [lbl if lbl is not None else empty for lbl in label_results]


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
    member_var_spec: dict[int, tuple[float, float]] | None = None,
    member_dla_kl: dict[int, float] | None = None,
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

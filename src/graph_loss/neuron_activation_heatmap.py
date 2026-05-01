"""Plot a specific MLP neuron's activations over a numeric dataset."""

from __future__ import annotations

import argparse
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
import torch.nn.functional as F
from huggingface_hub import login

from graph_loss.utils import ActivationWriteResult, DTYPE_CHOICES, resolve_torch_dtype
from utils import HF_READ_TOKEN, default_datasets_dir, load_prompt_answer_json


@dataclass
class ActivationAccumulator:
    total: float = 0.0
    count: int = 0

    def add(self, value: float) -> None:
        self.total += value
        self.count += 1

    @property
    def mean(self) -> float:
        return self.total / self.count


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


def _validate_neuron_location(
    model,
    *,
    layer: int,
    token_pos: int,
    neuron_id: int,
) -> tuple[int, int, int]:
    if not (0 <= layer < model.cfg.n_layers):
        raise ValueError(f"--layer must be in [0, {model.cfg.n_layers}), got {layer}")
    if token_pos < 0:
        raise ValueError(f"--token-pos must be non-negative, got {token_pos}")
    if not (0 <= neuron_id < model.cfg.d_mlp):
        raise ValueError(f"--neuron-id must be in [0, {model.cfg.d_mlp}), got {neuron_id}")
    return layer, token_pos, neuron_id


def _select_mlp_neuron_weight(
    model,
    weight: torch.Tensor,
    neuron_id: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    d_model = model.cfg.d_model
    d_mlp = model.cfg.d_mlp

    if weight.shape == (d_mlp, d_model):
        row = weight[neuron_id]
    elif weight.shape == (d_model, d_mlp):
        row = weight[:, neuron_id]
    else:
        raise ValueError(
            f"Unsupported MLP weight shape {tuple(weight.shape)} for model dims "
            f"(d_model={d_model}, d_mlp={d_mlp})",
        )
    return row.to(device=device, dtype=dtype)


def _select_mlp_neuron_bias(
    module,
    name: str,
    neuron_id: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    bias = getattr(module, name, None)
    if bias is None:
        return torch.zeros((), device=device, dtype=dtype)
    return bias[neuron_id].to(device=device, dtype=dtype)


def _tokenize_prompt_batch(model, prompts: list[str]) -> tuple[torch.Tensor, list[int]]:
    tokenizer = model.tokenizer
    tokenized = [
        tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
        for prompt in prompts
    ]
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        raise ValueError("LLaMA tokenizer must define bos_token_id.")

    tokenized = [
        ids
        if int(ids[0].item()) == int(bos_token_id)
        else torch.cat([torch.tensor([bos_token_id], dtype=ids.dtype), ids])
        for ids in tokenized
    ]
    lengths = [int(ids.numel()) for ids in tokenized]
    max_len = max(lengths)

    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad_token_id is None:
        pad_token_id = bos_token_id

    input_ids = torch.full(
        (len(tokenized), max_len),
        int(pad_token_id),
        dtype=tokenized[0].dtype,
        device=model.cfg.device,
    )
    for row_idx, ids in enumerate(tokenized):
        input_ids[row_idx, : ids.numel()] = ids.to(model.cfg.device)
    return input_ids, lengths


@torch.no_grad()
def _activations_for_location_batch(
    model,
    prompts: list[str],
    *,
    layer: int,
    token_pos: int,
    neuron_id: int,
) -> list[float]:
    if not prompts:
        return []

    input_ids, lengths = _tokenize_prompt_batch(model, prompts)
    max_len = input_ids.shape[1]
    if token_pos >= max_len:
        return [float("nan")] * len(prompts)
    valid_at_pos = [token_pos < length for length in lengths]

    target_hook_name = f"blocks.{layer}.{model.feature_input_hook}"
    mlp_input_at_pos = None

    def cache_target_position(acts: torch.Tensor, hook) -> torch.Tensor:
        nonlocal mlp_input_at_pos
        mlp_input_at_pos = acts[:, token_pos, :].detach()
        return acts

    model.run_with_hooks(input_ids, fwd_hooks=[(target_hook_name, cache_target_position)])
    if mlp_input_at_pos is None:
        raise RuntimeError(f"Did not cache target hook {target_hook_name!r}")

    old_mlp = model.blocks[layer].mlp.old_mlp
    gate_weight = _select_mlp_neuron_weight(
        model,
        old_mlp.W_gate,
        neuron_id,
        device=input_ids.device,
        dtype=mlp_input_at_pos.dtype,
    )
    up_weight = _select_mlp_neuron_weight(
        model,
        old_mlp.W_in,
        neuron_id,
        device=input_ids.device,
        dtype=mlp_input_at_pos.dtype,
    )
    gate_bias = _select_mlp_neuron_bias(
        old_mlp,
        "b_gate",
        neuron_id,
        device=input_ids.device,
        dtype=mlp_input_at_pos.dtype,
    )
    up_bias = _select_mlp_neuron_bias(
        old_mlp,
        "b_in",
        neuron_id,
        device=input_ids.device,
        dtype=mlp_input_at_pos.dtype,
    )
    gate_pre = mlp_input_at_pos @ gate_weight + gate_bias
    up_pre = mlp_input_at_pos @ up_weight + up_bias
    activations = F.silu(gate_pre) * up_pre
    if token_pos == 0:
        activations.zero_()

    out = activations.detach().float().cpu().tolist()
    return [
        float(value) if is_valid else float("nan")
        for value, is_valid in zip(out, valid_at_pos, strict=True)
    ]


@torch.no_grad()
def build_neuron_activation_write_result(
    model,
    dataset_name: str,
    neuron_locations: list[tuple[int, int, int]] | torch.Tensor,
    *,
    forward_batch_size: int = 32,
    limit: int | None = None,
    log_interval: int = 100,
) -> ActivationWriteResult:
    """Return activation grids and down-projection vectors for the requested neurons."""
    logger = logging.getLogger(__name__)
    if forward_batch_size <= 0:
        raise ValueError("forward_batch_size must be positive")

    if isinstance(neuron_locations, torch.Tensor):
        locations = [
            (int(row[0].item()), int(row[1].item()), int(row[2].item()))
            for row in neuron_locations.detach().cpu()
        ]
    else:
        locations = [
            (int(layer), int(token_pos), int(neuron_id))
            for layer, token_pos, neuron_id in neuron_locations
        ]

    for layer, token_pos, neuron_id in locations:
        _validate_neuron_location(
            model,
            layer=layer,
            token_pos=token_pos,
            neuron_id=neuron_id,
        )

    dataset_path = _resolve_dataset_path(dataset_name)
    samples = list(load_prompt_answer_json(dataset_path).items())
    if limit is not None:
        samples = samples[:limit]

    prompts = []
    numeric_args_by_prompt = []
    skipped = 0
    expected_n_args = None
    for prompt, _answer in samples:
        try:
            numeric_args = _parse_numeric_args(prompt)
        except ValueError:
            skipped += 1
            continue
        if expected_n_args is None:
            expected_n_args = len(numeric_args)
            if expected_n_args not in (1, 2, 3):
                raise ValueError(
                    "Only 1D, 2D, and 3D activation heatmaps are supported; "
                    f"first parsed prompt has {expected_n_args} numeric args: {prompt!r}",
                )
        if len(numeric_args) != expected_n_args:
            skipped += 1
            continue
        prompts.append(prompt)
        numeric_args_by_prompt.append(numeric_args)

    if not prompts:
        raise ValueError(f"Dataset has no samples: {dataset_path}")

    arg_values = [
        sorted({numeric_args[dim] for numeric_args in numeric_args_by_prompt})
        for dim in range(len(numeric_args_by_prompt[0]))
    ]
    arg_to_idx = [
        {value: idx for idx, value in enumerate(values)}
        for values in arg_values
    ]
    grid_shape = tuple(len(values) for values in arg_values)
    d_model = int(model.cfg.d_model)
    activation_sums = torch.zeros((len(locations), *grid_shape), dtype=torch.float32)
    activation_counts = torch.zeros((len(locations), *grid_shape), dtype=torch.float32)
    w_down_vectors = torch.empty((len(locations), d_model), dtype=torch.float32)
    if not locations:
        return ActivationWriteResult(
            activations=torch.full((0, *grid_shape), float("nan"), dtype=torch.float32),
            w_down_vectors=w_down_vectors,
            arg_values=arg_values,
        )

    location_groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    w_out_cache: dict[int, torch.Tensor] = {}
    for location_idx, (layer, token_pos, neuron_id) in enumerate(locations):
        location_groups[(layer, token_pos)].append((location_idx, neuron_id))
        if layer not in w_out_cache:
            old_mlp = model.blocks[layer].mlp.old_mlp
            w_out_cache[layer] = model._row_oriented_weight(old_mlp.W_out.to(device=model.cfg.device))
        w_down_vectors[location_idx] = w_out_cache[layer][neuron_id].detach().float().cpu()

    for batch_start in range(0, len(prompts), forward_batch_size):
        batch_prompts = prompts[batch_start:batch_start + forward_batch_size]
        batch_numeric_args = numeric_args_by_prompt[
            batch_start:batch_start + len(batch_prompts)
        ]
        input_ids, lengths = _tokenize_prompt_batch(model, batch_prompts)
        max_len = input_ids.shape[1]
        active_groups = {
            group_key: group_members
            for group_key, group_members in location_groups.items()
            if group_key[1] < max_len
        }
        if not active_groups:
            continue

        cached_inputs: dict[tuple[int, int], torch.Tensor] = {}
        hooks = []
        for layer, token_pos in active_groups:
            hook_name = f"blocks.{layer}.{model.feature_input_hook}"

            def cache_target_position(
                acts: torch.Tensor,
                hook,
                *,
                key: tuple[int, int] = (layer, token_pos),
            ) -> torch.Tensor:
                cached_inputs[key] = acts[:, key[1], :].detach()
                return acts

            hooks.append((hook_name, cache_target_position))

        model.run_with_hooks(input_ids, fwd_hooks=hooks)
        for (layer, token_pos), group_members in active_groups.items():
            mlp_input_at_pos = cached_inputs.get((layer, token_pos))
            if mlp_input_at_pos is None:
                raise RuntimeError(
                    f"Did not cache target hook for layer={layer}, token_pos={token_pos}"
                )

            old_mlp = model.blocks[layer].mlp.old_mlp
            valid_at_pos = torch.tensor(
                [token_pos < length for length in lengths],
                dtype=torch.bool,
                device=mlp_input_at_pos.device,
            )
            for location_idx, neuron_id in group_members:
                gate_weight = _select_mlp_neuron_weight(
                    model,
                    old_mlp.W_gate,
                    neuron_id,
                    device=mlp_input_at_pos.device,
                    dtype=mlp_input_at_pos.dtype,
                )
                up_weight = _select_mlp_neuron_weight(
                    model,
                    old_mlp.W_in,
                    neuron_id,
                    device=mlp_input_at_pos.device,
                    dtype=mlp_input_at_pos.dtype,
                )
                gate_bias = _select_mlp_neuron_bias(
                    old_mlp,
                    "b_gate",
                    neuron_id,
                    device=mlp_input_at_pos.device,
                    dtype=mlp_input_at_pos.dtype,
                )
                up_bias = _select_mlp_neuron_bias(
                    old_mlp,
                    "b_in",
                    neuron_id,
                    device=mlp_input_at_pos.device,
                    dtype=mlp_input_at_pos.dtype,
                )
                gate_pre = mlp_input_at_pos @ gate_weight + gate_bias
                up_pre = mlp_input_at_pos @ up_weight + up_bias
                activations = F.silu(gate_pre) * up_pre
                if token_pos == 0:
                    activations.zero_()

                for numeric_args, activation, is_valid in zip(
                    batch_numeric_args,
                    activations.detach().float().cpu(),
                    valid_at_pos.detach().cpu(),
                    strict=True,
                ):
                    if not bool(is_valid.item()):
                        continue
                    grid_idx = tuple(
                        arg_to_idx[dim][arg_value]
                        for dim, arg_value in enumerate(numeric_args)
                    )
                    activation_sums[(location_idx, *grid_idx)] += float(activation.item())
                    activation_counts[(location_idx, *grid_idx)] += 1.0

        processed = batch_start + len(batch_prompts)
        if log_interval and processed % log_interval == 0:
            logger.info("Processed %d activation samples", processed)

    activations = activation_sums / activation_counts.clamp(min=1.0)
    activations[activation_counts == 0] = float("nan")
    logger.info(
        "Built activation grid for %d neurons with arg dims %s from %s (skipped=%d)",
        len(locations),
        grid_shape,
        dataset_path,
        skipped,
    )
    return ActivationWriteResult(
        activations=activations,
        w_down_vectors=w_down_vectors,
        arg_values=arg_values,
    )


@torch.no_grad()
def build_neuron_activation_write_matrix(
    model,
    dataset_name: str,
    neuron_locations: list[tuple[int, int, int]] | torch.Tensor,
    *,
    forward_batch_size: int = 32,
    limit: int | None = None,
    log_interval: int = 100,
) -> torch.Tensor:
    """Return flattened activation-grid * W_down matrices with shape [neurons, grid_points, d_model]."""
    result = build_neuron_activation_write_result(
        model,
        dataset_name,
        neuron_locations,
        forward_batch_size=forward_batch_size,
        limit=limit,
        log_interval=log_interval,
    )
    flat_activations = torch.nan_to_num(result.activations.detach().float()).flatten(start_dim=1)
    return flat_activations[:, :, None] * result.w_down_vectors[:, None, :]


def save_cluster_activation_heatmap_pdfs(
    activations: torch.Tensor,
    arg_values: list[list[int]],
    assignments: torch.Tensor,
    neuron_indices: torch.Tensor,
    neuron_locations: torch.Tensor,
    *,
    output_dir: str = ".",
) -> list[str]:
    """Save one PDF per cluster with one activation heatmap page per neuron."""
    os.makedirs(output_dir, exist_ok=True)
    saved_paths: list[str] = []
    if activations.numel() == 0 or assignments.numel() == 0:
        return saved_paths

    activation_values = activations.detach().float().cpu()
    assignments_cpu = assignments.detach().cpu()
    neuron_indices_cpu = neuron_indices.detach().cpu()
    locations_cpu = neuron_locations.detach().cpu()

    def save_activation_page(
        pdf: PdfPages,
        activation_grid: torch.Tensor,
        *,
        title: str,
    ) -> None:
        if torch.isnan(activation_grid).all():
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No valid activations", ha="center", va="center")
            ax.set_axis_off()
            ax.set_title(title)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            return

        n_args = len(arg_values)
        if n_args == 1:
            xs = arg_values[0]
            heatmap = activation_grid.unsqueeze(0)
            fig, ax = plt.subplots(figsize=(8, 2.5))
            image = ax.imshow(
                heatmap.numpy(),
                origin="lower",
                aspect="auto",
                extent=[min(xs), max(xs), 0, 1],
            )
            fig.colorbar(image, ax=ax, label="activation")
            ax.set_xlabel("arg 1")
            ax.set_yticks([])
            ax.set_title(title)
        elif n_args == 2:
            xs = arg_values[0]
            ys = arg_values[1]
            heatmap = activation_grid.T.contiguous()
            fig, ax = plt.subplots(figsize=(8, 6))
            image = ax.imshow(
                heatmap.numpy(),
                origin="lower",
                aspect="auto",
                extent=[min(xs), max(xs), min(ys), max(ys)],
            )
            fig.colorbar(image, ax=ax, label="activation")
            ax.set_xlabel("arg 1")
            ax.set_ylabel("arg 2")
            ax.set_title(title)
        else:
            points = []
            for x_idx, x in enumerate(arg_values[0]):
                for y_idx, y in enumerate(arg_values[1]):
                    for z_idx, z in enumerate(arg_values[2]):
                        activation = activation_grid[x_idx, y_idx, z_idx]
                        if activation == activation:
                            points.append((x, y, z, float(activation.item())))

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            if points:
                xs, ys, zs, activation_colors = zip(*points, strict=True)
                scatter = ax.scatter(xs, ys, zs, c=activation_colors, cmap="viridis")
                fig.colorbar(scatter, ax=ax, label="activation")
            ax.set_xlabel("arg 1")
            ax.set_ylabel("arg 2")
            ax.set_zlabel("arg 3")
            ax.set_title(title)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    for cluster_id in torch.unique(assignments_cpu).tolist():
        cluster_rows = torch.where(assignments_cpu == int(cluster_id))[0].tolist()
        output_path = os.path.join(output_dir, f"cluster_{int(cluster_id)}.pdf")
        with PdfPages(output_path) as pdf:
            for row_idx in cluster_rows:
                graph_neuron_idx = int(neuron_indices_cpu[row_idx].item())
                layer = int(locations_cpu[graph_neuron_idx, 0].item())
                token_pos = int(locations_cpu[graph_neuron_idx, 1].item())
                neuron_id = int(locations_cpu[graph_neuron_idx, 2].item())

                title = (
                    f"Neuron {neuron_id} activation "
                    f"(cluster={int(cluster_id)}, graph_neuron_idx={graph_neuron_idx}, "
                    f"layer={layer}, token={token_pos}, neuron={neuron_id})"
                )
                save_activation_page(
                    pdf,
                    activation_values[row_idx],
                    title=title,
                )
        saved_paths.append(output_path)

    return saved_paths


def save_supernode_activation_heatmap_pdf(
    activation_grids: torch.Tensor,
    arg_values: list[list[int]],
    members: list[int],
    neuron_locations: torch.Tensor,
    *,
    output_path: str,
    title: str,
    member_title_suffixes: list[str] | None = None,
) -> str:
    """Save one activation heatmap page per neuron in a supernode."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    activation_grids = activation_grids.detach().float().cpu()
    if activation_grids.shape[0] != len(members):
        raise ValueError(
            f"Expected one activation grid per member, got {activation_grids.shape[0]} "
            f"grids for {len(members)} members"
        )
    if member_title_suffixes is not None and len(member_title_suffixes) != len(members):
        raise ValueError(
            f"Expected one title suffix per member, got {len(member_title_suffixes)} "
            f"suffixes for {len(members)} members"
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
            page_title = f"{title}\nNeuron {neuron_id} ({location_text})"
            if member_title_suffixes is not None:
                page_title = f"{page_title}; {member_title_suffixes[member_idx]}"
            if torch.isnan(activation_grid).all():
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.text(0.5, 0.5, "No valid activations", ha="center", va="center")
                ax.set_axis_off()
            else:
                n_args = len(arg_values)
                if n_args == 1:
                    xs = arg_values[0]
                    heatmap = activation_grid.unsqueeze(0)
                    fig, ax = plt.subplots(figsize=(8, 2.5))
                    image = ax.imshow(
                        heatmap.numpy(),
                        origin="lower",
                        aspect="auto",
                        extent=[min(xs), max(xs), 0, 1],
                    )
                    fig.colorbar(image, ax=ax, label="activation")
                    ax.set_xlabel("arg 1")
                    ax.set_yticks([])
                elif n_args == 2:
                    xs = arg_values[0]
                    ys = arg_values[1]
                    heatmap = activation_grid.T.contiguous()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    image = ax.imshow(
                        heatmap.numpy(),
                        origin="lower",
                        aspect="auto",
                        extent=[min(xs), max(xs), min(ys), max(ys)],
                    )
                    fig.colorbar(image, ax=ax, label="activation")
                    ax.set_xlabel("arg 1")
                    ax.set_ylabel("arg 2")
                else:
                    points = []
                    for x_idx, x in enumerate(arg_values[0]):
                        for y_idx, y in enumerate(arg_values[1]):
                            for z_idx, z in enumerate(arg_values[2]):
                                activation = activation_grid[x_idx, y_idx, z_idx]
                                if activation == activation:
                                    points.append((x, y, z, float(activation.item())))

                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111, projection="3d")
                    if points:
                        xs, ys, zs, activation_colors = zip(*points, strict=True)
                        scatter = ax.scatter(xs, ys, zs, c=activation_colors, cmap="viridis")
                        fig.colorbar(scatter, ax=ax, label="activation")
                    ax.set_xlabel("arg 1")
                    ax.set_ylabel("arg 2")
                    ax.set_zlabel("arg 3")

            ax.set_title(page_title)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    return output_path


def _plot_1d(
    values_by_arg: dict[tuple[int, ...], ActivationAccumulator],
    *,
    output_path: str,
    title: str,
) -> None:
    xs = sorted(arg[0] for arg in values_by_arg)
    heatmap = torch.tensor([[values_by_arg[(x,)].mean for x in xs]], dtype=torch.float32)
    fig, ax = plt.subplots(figsize=(8, 2.5))
    image = ax.imshow(
        heatmap.numpy(),
        origin="lower",
        aspect="auto",
        extent=[min(xs), max(xs), 0, 1],
    )
    fig.colorbar(image, ax=ax, label="activation")
    ax.set_xlabel("arg 1")
    ax.set_yticks([])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_2d(
    values_by_arg: dict[tuple[int, ...], ActivationAccumulator],
    *,
    output_path: str,
    title: str,
) -> None:
    xs = sorted({arg[0] for arg in values_by_arg})
    ys = sorted({arg[1] for arg in values_by_arg})
    x_to_idx = {value: idx for idx, value in enumerate(xs)}
    y_to_idx = {value: idx for idx, value in enumerate(ys)}

    heatmap = torch.full((len(ys), len(xs)), float("nan"), dtype=torch.float32)
    for (x, y), stats in values_by_arg.items():
        if stats.count:
            heatmap[y_to_idx[y], x_to_idx[x]] = stats.mean

    plt.figure(figsize=(8, 6))
    plt.imshow(
        heatmap.numpy(),
        origin="lower",
        aspect="auto",
        extent=[min(xs), max(xs), min(ys), max(ys)],
    )
    plt.colorbar(label="activation")
    plt.xlabel("arg 1")
    plt.ylabel("arg 2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_3d(
    values_by_arg: dict[tuple[int, ...], ActivationAccumulator],
    *,
    output_path: str,
    title: str,
) -> None:
    points = []
    for (x, y, z), stats in values_by_arg.items():
        if stats.count:
            points.append((x, y, z, stats.mean))
    if not points:
        raise ValueError("No valid 3D points to plot.")

    xs, ys, zs, activations = zip(*points, strict=True)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(xs, ys, zs, c=activations, cmap="viridis")
    fig.colorbar(scatter, ax=ax, label="activation")
    ax.set_xlabel("arg 1")
    ax.set_ylabel("arg 2")
    ax.set_zlabel("arg 3")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_neuron_activation_heatmap(args: argparse.Namespace) -> str:
    logger = logging.getLogger(__name__)
    if args.forward_batch_size <= 0:
        raise ValueError("--forward-batch-size must be positive")

    dataset_path = _resolve_dataset_path(args.dataset_name)
    data = load_prompt_answer_json(dataset_path)
    samples = list(data.items())
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError(f"Dataset has no samples: {dataset_path}")

    dtype = resolve_torch_dtype(args.dtype)
    if HF_READ_TOKEN:
        logger.info("Authenticating with Hugging Face token")
        login(HF_READ_TOKEN)

    from graph_loss.replacement_model import TransformerLensReplacementModel

    logger.info("Loading model: %s", args.model)
    model = TransformerLensReplacementModel.from_pretrained(args.model, dtype=dtype)
    model.eval()

    layer, token_pos, neuron_id = _validate_neuron_location(
        model,
        layer=args.layer,
        token_pos=args.token_pos,
        neuron_id=args.neuron_id,
    )
    logger.info(
        "Using neuron location layer=%d token=%d neuron=%d",
        layer,
        token_pos,
        neuron_id,
    )

    values_by_arg: dict[tuple[int, ...], ActivationAccumulator] = defaultdict(ActivationAccumulator)
    skipped = 0
    expected_n_args = None
    processed_valid = 0
    batch: list[tuple[str, tuple[int, ...]]] = []

    def flush_batch() -> None:
        nonlocal processed_valid, skipped, batch
        if not batch:
            return
        prompts = [prompt for prompt, _numeric_args in batch]
        activations = _activations_for_location_batch(
            model,
            prompts,
            layer=layer,
            token_pos=token_pos,
            neuron_id=neuron_id,
        )
        for (_prompt, numeric_args), activation in zip(batch, activations, strict=True):
            if activation == activation:
                values_by_arg[numeric_args].add(activation)
            else:
                skipped += 1
        processed_valid += len(batch)
        if args.log_interval and processed_valid % args.log_interval == 0:
            logger.info("Processed %d valid samples", processed_valid)
        batch = []

    for prompt, _answer in samples:
        try:
            numeric_args = _parse_numeric_args(prompt)
        except ValueError:
            skipped += 1
            continue
        if expected_n_args is None:
            expected_n_args = len(numeric_args)
            if expected_n_args not in (1, 2, 3):
                raise ValueError(
                    "Only 1D, 2D, and 3D plots are supported; "
                    f"first parsed prompt has {expected_n_args} numeric args: {prompt!r}",
                )
        if len(numeric_args) != expected_n_args:
            skipped += 1
            continue
        batch.append((prompt, numeric_args))
        if len(batch) == args.forward_batch_size:
            flush_batch()
    flush_batch()

    if not values_by_arg:
        raise ValueError("No valid activations collected.")

    output_path = args.output_path
    if output_path is None:
        dataset_stem = os.path.splitext(os.path.basename(dataset_path))[0]
        output_path = (
            f"layer_{layer}_token_{token_pos}_neuron_{neuron_id}"
            f"_{dataset_stem}_activation_heatmap.png"
        )

    title = (
        f"Neuron {neuron_id} activation "
        f"(layer={layer}, token={token_pos}, neuron={neuron_id})"
    )
    n_args = len(next(iter(values_by_arg)))
    if n_args == 1:
        _plot_1d(values_by_arg, output_path=output_path, title=title)
    elif n_args == 2:
        _plot_2d(values_by_arg, output_path=output_path, title=title)
    else:
        _plot_3d(values_by_arg, output_path=output_path, title=title)

    logger.info("Saved heatmap to %s (skipped=%d)", output_path, skipped)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot a specific MLP neuron's activations over a numeric dataset.",
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--layer", type=int, required=True, help="Layer index")
    parser.add_argument("--token-pos", type=int, required=True, help="Token position")
    parser.add_argument("--neuron-id", type=int, required=True, help="Per-layer MLP neuron id")
    parser.add_argument("--dataset-name", required=True, help="Dataset prefix, filename, or path")
    parser.add_argument("--output-path", help="Path for the output heatmap PNG")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit")
    parser.add_argument("--log-interval", type=int, default=100, help="Progress log interval")
    parser.add_argument(
        "--forward-batch-size",
        type=int,
        default=32,
        help="Batch size for activation forward passes",
    )
    parser.add_argument(
        "--dtype",
        choices=DTYPE_CHOICES,
        default="float32",
        help="Model dtype",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    plot_neuron_activation_heatmap(build_parser().parse_args())


if __name__ == "__main__":
    main()

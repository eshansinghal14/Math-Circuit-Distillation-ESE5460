"""Plot a graph-indexed neuron's activations over a numeric dataset."""

from __future__ import annotations

import argparse
import logging
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from huggingface_hub import login

from graph_loss.utils import add_graph_build_args, resolve_torch_dtype
from utils import HF_READ_TOKEN, default_datasets_dir, load_prompt_answer_json


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
def _resolve_graph_neuron_location(
    model,
    prompt: str,
    neuron_idx: int,
    prop_neurons_per_layer: float,
) -> tuple[int, int, int]:
    ctx = model.setup_attribution(prompt, prop_neurons_per_layer=prop_neurons_per_layer)
    if neuron_idx < 0 or neuron_idx >= int(ctx.neuron_locations.shape[0]):
        raise IndexError(
            f"neuron_idx={neuron_idx} out of range for reference graph "
            f"with {int(ctx.neuron_locations.shape[0])} neurons",
        )
    layer, token_pos, neuron_id = ctx.neuron_locations[neuron_idx].tolist()
    return int(layer), int(token_pos), int(neuron_id)


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
    max_len = max(int(ids.numel()) for ids in tokenized)
    if token_pos >= max_len:
        return [float("nan")] * len(prompts)

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
    valid_at_pos = []
    for row_idx, ids in enumerate(tokenized):
        ids = ids.to(model.cfg.device)
        input_ids[row_idx, : ids.numel()] = ids
        valid_at_pos.append(token_pos < int(ids.numel()))

    mlp_in_cache, mlp_in_caching_hooks, _ = model.get_caching_hooks(
        lambda name: name.endswith(model.feature_input_hook),
    )
    model.run_with_hooks(input_ids, fwd_hooks=mlp_in_caching_hooks)
    mlp_inputs = model._stack_layer_cache(mlp_in_cache, model.feature_input_hook)

    old_mlp = model.blocks[layer].mlp.old_mlp
    gate_rows = model._row_oriented_weight(
        old_mlp.W_gate.to(device=input_ids.device, dtype=mlp_inputs.dtype),
    )
    up_rows = model._row_oriented_weight(
        old_mlp.W_in.to(device=input_ids.device, dtype=mlp_inputs.dtype),
    )
    gate_bias = model._get_bias(
        old_mlp,
        "b_gate",
        gate_rows.shape[0],
        input_ids.device,
        mlp_inputs.dtype,
    )
    up_bias = model._get_bias(
        old_mlp,
        "b_in",
        up_rows.shape[0],
        input_ids.device,
        mlp_inputs.dtype,
    )
    layer_inputs_at_pos = mlp_inputs[layer, :, token_pos]
    gate_pre = layer_inputs_at_pos @ gate_rows.T + gate_bias
    up_pre = layer_inputs_at_pos @ up_rows.T + up_bias
    activations = F.silu(gate_pre) * up_pre
    if token_pos == 0:
        activations.zero_()

    out = activations[:, neuron_id].detach().float().cpu().tolist()
    return [
        float(value) if is_valid else float("nan")
        for value, is_valid in zip(out, valid_at_pos, strict=True)
    ]


def _plot_1d(
    values_by_arg: dict[tuple[int, ...], list[float]],
    *,
    output_path: str,
    title: str,
) -> None:
    xs = sorted(arg[0] for arg in values_by_arg)
    ys = [
        sum(values_by_arg[(x,)]) / max(len(values_by_arg[(x,)]), 1)
        for x in xs
    ]
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("arg 1")
    plt.ylabel("activation")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_2d(
    values_by_arg: dict[tuple[int, ...], list[float]],
    *,
    output_path: str,
    title: str,
) -> None:
    xs = sorted({arg[0] for arg in values_by_arg})
    ys = sorted({arg[1] for arg in values_by_arg})
    x_to_idx = {value: idx for idx, value in enumerate(xs)}
    y_to_idx = {value: idx for idx, value in enumerate(ys)}

    heatmap = torch.full((len(ys), len(xs)), float("nan"), dtype=torch.float32)
    for (x, y), vals in values_by_arg.items():
        if vals:
            heatmap[y_to_idx[y], x_to_idx[x]] = sum(vals) / len(vals)

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
    values_by_arg: dict[tuple[int, ...], list[float]],
    *,
    output_path: str,
    title: str,
) -> None:
    points = []
    for (x, y, z), vals in values_by_arg.items():
        if vals:
            points.append((x, y, z, sum(vals) / len(vals)))
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

    reference_prompt = samples[0][0]
    layer, token_pos, neuron_id = _resolve_graph_neuron_location(
        model,
        reference_prompt,
        args.neuron_idx,
        args.prop_neurons_per_layer,
    )
    logger.info(
        "Resolved graph neuron %d on reference prompt to layer=%d token=%d neuron=%d",
        args.neuron_idx,
        layer,
        token_pos,
        neuron_id,
    )

    values_by_arg: dict[tuple[int, ...], list[float]] = defaultdict(list)
    skipped = 0
    expected_n_args = None
    valid_samples = []
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
        valid_samples.append((prompt, numeric_args))

    for start in range(0, len(valid_samples), args.forward_batch_size):
        batch = valid_samples[start:start + args.forward_batch_size]
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
                values_by_arg[numeric_args].append(activation)
            else:
                skipped += 1
        if args.log_interval and (start + len(batch)) % args.log_interval == 0:
            logger.info("Processed %d/%d valid samples", start + len(batch), len(valid_samples))

    if not values_by_arg:
        raise ValueError("No valid activations collected.")

    output_path = args.output_path
    if output_path is None:
        dataset_stem = os.path.splitext(os.path.basename(dataset_path))[0]
        output_path = f"neuron_{args.neuron_idx}_{dataset_stem}_activation_heatmap.png"

    title = (
        f"Neuron {args.neuron_idx} activation "
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
        description="Plot a graph-indexed neuron's activations over a numeric dataset.",
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--neuron-idx", type=int, required=True, help="Graph neuron index")
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
    add_graph_build_args(parser)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    plot_neuron_activation_heatmap(build_parser().parse_args())


if __name__ == "__main__":
    main()

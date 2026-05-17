"""Pre-compute and cache MLP residual-stream inputs for all prompts in a dataset.

Running this once stores the hidden state entering each MLP layer at every token
position, across every prompt in the dataset.  Subsequent calls to
``build_neuron_activation_write_result`` can then compute any neuron's SwiGLU
activation from the cache with a cheap matmul instead of a forward pass.

Cache layout on disk::

    <cache_dir>/<model_slug>/<dataset_slug>/
        meta.pt          – n_prompts, n_layers, d_model, d_mlp, arg_values, positions
        layer_<i>.pt     – shape [n_prompts, n_positions, d_model]  (bfloat16)

Usage::

    python -m graph_loss.precompute_mlp_inputs \
        --model  meta-llama/Meta-Llama-3.1-8B-Instruct \
        --dataset 22_add_tight_5000_train.json \
        --cache-dir /content/mlp_input_cache \
        --dtype bfloat16 \
        --batch-size 8
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import tempfile
from collections import defaultdict

import torch
from huggingface_hub import login

from graph_loss.neuron_activation_heatmap import _resolve_dataset_path, _tokenize_prompt_batch, _parse_numeric_args
from graph_loss.replacement_model import TransformerLensReplacementModel
from graph_loss.utils import DTYPE_CHOICES, resolve_torch_dtype
from utils import HF_READ_TOKEN, load_prompt_answer_json

logger = logging.getLogger(__name__)


def _model_slug(model_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("._-")[:48]
    digest = hashlib.sha1(model_name.encode()).hexdigest()[:8]
    return f"{safe}_{digest}"


def _dataset_slug(dataset_path: str) -> str:
    base = os.path.splitext(os.path.basename(dataset_path))[0]
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._-")[:48]
    return safe


def mlp_input_cache_dir(cache_root: str, model_name: str, dataset_path: str) -> str:
    return os.path.join(cache_root, _model_slug(model_name), _dataset_slug(dataset_path))


def mlp_input_cache_exists(cache_root: str, model_name: str, dataset_path: str) -> bool:
    d = mlp_input_cache_dir(cache_root, model_name, dataset_path)
    return os.path.isfile(os.path.join(d, "meta.pt"))


def load_mlp_input_cache(cache_root: str, model_name: str, dataset_path: str) -> dict:
    """Load the full cache.  Returns a dict with keys:

    - ``meta``: dict with n_prompts, n_layers, d_model, d_mlp, arg_values, positions
    - ``layer_inputs``: list of [n_prompts, n_positions, d_model] tensors (one per layer)
    """
    d = mlp_input_cache_dir(cache_root, model_name, dataset_path)
    meta = torch.load(os.path.join(d, "meta.pt"), map_location="cpu", weights_only=True)
    n_layers = meta["n_layers"]
    layer_inputs = []
    for i in range(n_layers):
        layer_inputs.append(
            torch.load(os.path.join(d, f"layer_{i}.pt"), map_location="cpu", weights_only=True)
        )
    return {"meta": meta, "layer_inputs": layer_inputs}


@torch.no_grad()
def build_mlp_input_cache(
    adapter,
    dataset_path: str,
    model_name: str,
    *,
    cache_root: str | None = None,
    batch_size: int = 8,
    overwrite: bool = True,
) -> dict:
    """Build an MLP-input cache from a live HF model (via ``HFLlamaGraphAdapter``).

    This is the training-time counterpart to ``precompute_mlp_inputs`` (which
    requires a TransformerLens model).  Uses PyTorch forward pre-hooks to
    capture the residual-stream input entering each MLP layer, writes to disk
    via memory-mapped arrays (to avoid OOM on large datasets), then loads and
    returns the full cache dict.

    Args:
        adapter: ``HFLlamaGraphAdapter`` wrapping the current student model.
        dataset_path: Resolved absolute path to the dataset JSON.
        model_name: HuggingFace model name string (cache key).
        cache_root: Root directory for the on-disk cache.  If ``None``, a
            temporary directory is created automatically.
        batch_size: Prompts per batched forward pass (default 8).
        overwrite: Rebuild even if a valid cache already exists (default
            ``True`` — caller knows the model weights have changed).

    Returns:
        Cache dict with keys ``"meta"`` and ``"layer_inputs"`` (same format
        as ``load_mlp_input_cache``).
    """
    import numpy as np

    _tmp_dir: str | None = None
    if cache_root is None:
        _tmp_dir = tempfile.mkdtemp(prefix="mlp_cache_")
        cache_root = _tmp_dir

    cache_dir = mlp_input_cache_dir(cache_root, model_name, dataset_path)
    meta_path = os.path.join(cache_dir, "meta.pt")
    if os.path.isfile(meta_path) and not overwrite:
        return load_mlp_input_cache(cache_root, model_name, dataset_path)

    os.makedirs(cache_dir, exist_ok=True)
    samples = list(load_prompt_answer_json(dataset_path).items())

    n_layers = adapter.n_layers
    d_model = adapter.d_model

    prompts, numeric_args_by_prompt = [], []
    expected_n_args = None
    for prompt, _ in samples:
        try:
            nargs = _parse_numeric_args(prompt)
        except ValueError:
            continue
        if expected_n_args is None:
            expected_n_args = len(nargs)
        if len(nargs) != expected_n_args:
            continue
        prompts.append(prompt)
        numeric_args_by_prompt.append(nargs)

    n_prompts = len(prompts)
    logger.info(
        "Building MLP input cache for %d prompts, %d layers, d_model=%d",
        n_prompts,
        n_layers,
        d_model,
    )

    first_ids, _ = _tokenize_prompt_batch(adapter, prompts[:1])
    n_positions = first_ids.shape[1]

    mmap_paths = [os.path.join(cache_dir, f"layer_{i}.npy") for i in range(n_layers)]
    mmaps: list[np.memmap] = [
        np.memmap(p, dtype="float16", mode="w+", shape=(n_prompts, n_positions, d_model))
        for p in mmap_paths
    ]

    prompt_idx = 0
    for batch_start in range(0, n_prompts, batch_size):
        batch_prompts = prompts[batch_start : batch_start + batch_size]
        input_ids, _ = _tokenize_prompt_batch(adapter, batch_prompts)
        batch_n_pos = input_ids.shape[1]
        bs = len(batch_prompts)

        captured: dict[int, torch.Tensor] = {}
        handles = []

        for layer_idx, layer in enumerate(adapter.layers):
            def _pre_hook(_module, inputs, *, idx=layer_idx):
                captured[idx] = inputs[0].detach().cpu().to(torch.float16)

            handles.append(layer.mlp.register_forward_pre_hook(_pre_hook))

        try:
            adapter.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
            )
        finally:
            for h in handles:
                h.remove()

        store_len = min(batch_n_pos, n_positions)
        for layer_idx in range(n_layers):
            acts = captured[layer_idx]  # [bs, batch_n_pos, d_model]
            mmaps[layer_idx][batch_start : batch_start + bs, :store_len, :] = (
                acts[:, :store_len, :].numpy()
            )

        prompt_idx += bs
        if prompt_idx % max(batch_size * 10, 50) == 0 or prompt_idx == n_prompts:
            logger.info("  Cached %d / %d prompts", prompt_idx, n_prompts)

    logger.info("Flushing memmaps and converting to .pt ...")
    for i, (mmap, mmap_path) in enumerate(zip(mmaps, mmap_paths)):
        mmap.flush()
        del mmap
        arr = np.memmap(
            mmap_path, dtype="float16", mode="r", shape=(n_prompts, n_positions, d_model)
        )
        buf = torch.from_numpy(np.array(arr)).to(torch.bfloat16)
        torch.save(buf, os.path.join(cache_dir, f"layer_{i}.pt"))
        del buf
        os.remove(mmap_path)
        logger.info("  Saved layer_%d.pt  shape=(%d, %d, %d)", i, n_prompts, n_positions, d_model)

    arg_values = [
        sorted({args[dim] for args in numeric_args_by_prompt})
        for dim in range(expected_n_args or 0)
    ]
    meta = {
        "n_prompts": n_prompts,
        "n_layers": n_layers,
        "d_model": d_model,
        "d_mlp": adapter.d_mlp,
        "n_positions": n_positions,
        "arg_values": arg_values,
        "numeric_args_by_prompt": numeric_args_by_prompt,
        "model_name": model_name,
        "dataset_path": dataset_path,
    }
    torch.save(meta, meta_path)
    logger.info("MLP input cache written to %s", cache_dir)

    return load_mlp_input_cache(cache_root, model_name, dataset_path)


@torch.no_grad()
def precompute_mlp_inputs(
    model: TransformerLensReplacementModel,
    dataset_path: str,
    cache_root: str,
    model_name: str,
    *,
    batch_size: int = 8,
    limit: int | None = None,
    overwrite: bool = False,
) -> str:
    """Forward-pass all prompts and cache MLP hidden-state inputs.

    Returns the path to the cache directory.
    """
    cache_dir = mlp_input_cache_dir(cache_root, model_name, dataset_path)
    meta_path = os.path.join(cache_dir, "meta.pt")

    if os.path.isfile(meta_path) and not overwrite:
        logger.info("MLP input cache already exists at %s — skipping (use --overwrite to rebuild)", cache_dir)
        return cache_dir

    os.makedirs(cache_dir, exist_ok=True)
    samples = list(load_prompt_answer_json(dataset_path).items())
    if limit is not None:
        samples = samples[:limit]

    n_layers = int(model.cfg.n_layers)
    d_model = int(model.cfg.d_model)

    # Collect numeric arg values for the heatmap grid dimensions.
    prompts, numeric_args_by_prompt = [], []
    expected_n_args = None
    for prompt, _ in samples:
        try:
            nargs = _parse_numeric_args(prompt)
        except ValueError:
            continue
        if expected_n_args is None:
            expected_n_args = len(nargs)
        if len(nargs) != expected_n_args:
            continue
        prompts.append(prompt)
        numeric_args_by_prompt.append(nargs)

    n_prompts = len(prompts)
    logger.info("Pre-computing MLP inputs for %d prompts, %d layers, d_model=%d", n_prompts, n_layers, d_model)

    import numpy as np

    # Determine sequence length from first batch.
    first_ids, _ = _tokenize_prompt_batch(model, prompts[:1])
    n_positions = first_ids.shape[1]

    # Use numpy memmaps so each layer is written directly to disk per batch —
    # no large in-RAM buffers (32 layers × 10k × 4096 × bf16 ≈ 78 GB would OOM).
    mmap_paths = [os.path.join(cache_dir, f"layer_{i}.npy") for i in range(n_layers)]
    mmaps: list[np.memmap] = [
        np.memmap(p, dtype="float16", mode="w+", shape=(n_prompts, n_positions, d_model))
        for p in mmap_paths
    ]

    prompt_idx = 0
    for batch_start in range(0, n_prompts, batch_size):
        batch_prompts = prompts[batch_start:batch_start + batch_size]
        input_ids, lengths = _tokenize_prompt_batch(model, batch_prompts)
        batch_n_pos = input_ids.shape[1]

        cached: dict[int, torch.Tensor] = {}

        def make_hook(layer_idx: int):
            def hook(acts: torch.Tensor, hook=None) -> torch.Tensor:
                cached[layer_idx] = acts.detach().cpu().to(torch.float16)
                return acts
            return hook

        hooks = [(f"blocks.{i}.{model.feature_input_hook}", make_hook(i)) for i in range(n_layers)]
        model.run_with_hooks(input_ids, fwd_hooks=hooks)

        store_len = min(batch_n_pos, n_positions)
        bs = len(batch_prompts)
        for layer_idx in range(n_layers):
            acts = cached[layer_idx]  # [bs, batch_n_pos, d_model]
            mmaps[layer_idx][batch_start:batch_start + bs, :store_len, :] = (
                acts[:, :store_len, :].numpy()
            )

        prompt_idx += bs
        if prompt_idx % max(batch_size * 10, 50) == 0 or prompt_idx == n_prompts:
            logger.info("  Cached %d / %d prompts", prompt_idx, n_prompts)

    # Flush memmaps to disk, then convert to .pt (bfloat16) for load_mlp_input_cache.
    logger.info("Flushing memmaps and converting to .pt ...")
    for i, (mmap, mmap_path) in enumerate(zip(mmaps, mmap_paths)):
        mmap.flush()
        del mmap
        arr = np.memmap(mmap_path, dtype="float16", mode="r", shape=(n_prompts, n_positions, d_model))
        buf = torch.from_numpy(np.array(arr)).to(torch.bfloat16)
        torch.save(buf, os.path.join(cache_dir, f"layer_{i}.pt"))
        del buf
        os.remove(mmap_path)
        logger.info("  Saved layer_%d.pt  shape=(%d, %d, %d)", i, n_prompts, n_positions, d_model)

    arg_values = [
        sorted({args[dim] for args in numeric_args_by_prompt})
        for dim in range(expected_n_args or 0)
    ]
    meta = {
        "n_prompts": n_prompts,
        "n_layers": n_layers,
        "d_model": d_model,
        "d_mlp": int(model.cfg.d_mlp),
        "n_positions": n_positions,
        "arg_values": arg_values,
        "numeric_args_by_prompt": numeric_args_by_prompt,
        "model_name": model_name,
        "dataset_path": dataset_path,
    }
    torch.save(meta, meta_path)

    logger.info("MLP input cache written to %s", cache_dir)
    return cache_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-compute MLP residual-stream inputs for all dataset prompts."
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset JSON filename under datasets/ or an explicit path",
    )
    parser.add_argument("--cache-dir", required=True, help="Root directory for the cache")
    parser.add_argument("--dtype", default="bfloat16", choices=DTYPE_CHOICES)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()

    if HF_READ_TOKEN:
        login(HF_READ_TOKEN)

    dtype = resolve_torch_dtype(args.dtype)
    logger.info("Loading model: %s", args.model)
    model = TransformerLensReplacementModel.from_pretrained(args.model, dtype=dtype)
    model.eval()

    dataset_path = _resolve_dataset_path(args.dataset)
    precompute_mlp_inputs(
        model,
        dataset_path,
        cache_root=args.cache_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        limit=args.limit,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

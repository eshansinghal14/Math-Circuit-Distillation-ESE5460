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

from transformers import AutoModelForCausalLM, AutoTokenizer
from graph_loss.hf_adapter import HFLlamaGraphAdapter
from graph_loss.neuron_activation_heatmap import _parse_numeric_args
from graph_loss.utils import DTYPE_CHOICES, resolve_torch_dtype
from utils import HF_READ_TOKEN, default_datasets_dir, load_prompt_answer_json


def _resolve_dataset_path(dataset_name: str) -> str:
    expanded = os.path.expanduser(dataset_name)
    if os.path.isfile(expanded):
        return os.path.abspath(expanded)
    root = default_datasets_dir()
    candidates = [
        os.path.join(root, os.path.basename(expanded)),
        os.path.join(root, f"{os.path.basename(expanded)}.json"),
        os.path.join(root, f"{os.path.basename(expanded)}_all.json"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(f"Could not resolve dataset {dataset_name!r}. Tried: {', '.join(candidates)}")

logger = logging.getLogger(__name__)


def _tokenize_prompt_batch(adapter, prompts: list[str]) -> tuple[torch.Tensor, list[int]]:
    tokenizer = adapter.tokenizer
    tokenized = [
        tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
        for prompt in prompts
    ]
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is not None:
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
        device=adapter.cfg.device,
    )
    for row_idx, ids in enumerate(tokenized):
        input_ids[row_idx, : ids.numel()] = ids.to(adapter.cfg.device)
    return input_ids, lengths


def _model_slug(model_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("._-")[:48]
    digest = hashlib.sha1(model_name.encode()).hexdigest()[:8]
    return f"{safe}_{digest}"


def _dataset_slug(dataset_path: str) -> str:
    base = os.path.splitext(os.path.basename(dataset_path))[0]
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._-")[:48]
    return safe


def load_mlp_cache_dir(cache_dir: str) -> dict:
    """Load an MLP input cache from a directory containing meta.pt + layer_i.pt."""
    meta = torch.load(os.path.join(cache_dir, "meta.pt"), map_location="cpu", weights_only=True)
    n_layers = meta["n_layers"]
    layer_inputs = [
        torch.load(os.path.join(cache_dir, f"layer_{i}.pt"), map_location="cpu", weights_only=True)
        for i in range(n_layers)
    ]
    return {"meta": meta, "layer_inputs": layer_inputs}


@torch.no_grad()
def build_mlp_input_cache(
    adapter,
    dataset_path: str,
    model_name: str,
    *,
    data_dict: dict | None = None,
    batch_size: int = 64,
) -> dict:
    """Build or load an MLP-input cache stored under the system temp directory.

    Checks for an existing cache keyed by model and dataset.  If found, loads
    and returns it immediately.  Otherwise captures residual-stream inputs
    entering each MLP layer via forward pre-hooks, saves the result to temp,
    and returns it.

    Args:
        adapter: ``HFLlamaGraphAdapter`` wrapping the current model.
        dataset_path: Resolved path to the dataset JSON (cache key).
        model_name: HuggingFace model name string (cache key).
        data_dict: Optional pre-loaded ``{prompt: answer}`` dict.
        batch_size: Prompts per batched forward pass (default 64).

    Returns:
        Cache dict with keys ``"meta"`` and ``"layer_inputs"``.
    """
    cache_dir = os.path.join(
        tempfile.gettempdir(), "mlp_cache",
        _model_slug(model_name), _dataset_slug(dataset_path),
    )
    meta_path = os.path.join(cache_dir, "meta.pt")
    if os.path.isfile(meta_path):
        return load_mlp_cache_dir(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)
    if data_dict is not None:
        samples = list(data_dict.items())
    else:
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

    # Pre-allocate in-RAM bfloat16 buffers.
    # Memory cost: n_layers × n_prompts × n_positions × d_model × 2 B.
    # For a 1B student: 16 × 10k × 20 × 2048 × 2 ≈ 12.8 GB — fine on A100.
    layer_inputs: list[torch.Tensor] = [
        torch.zeros((n_prompts, n_positions, d_model), dtype=torch.bfloat16)
        for _ in range(n_layers)
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
                captured[idx] = inputs[0].detach().cpu().to(torch.bfloat16)

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
            acts = captured[layer_idx]  # [bs, batch_n_pos, d_model] bfloat16 on CPU
            layer_inputs[layer_idx][batch_start : batch_start + bs, :store_len, :] = (
                acts[:, :store_len, :]
            )

        prompt_idx += bs
        if prompt_idx % max(batch_size * 10, 50) == 0 or prompt_idx == n_prompts:
            logger.info("  Cached %d / %d prompts", prompt_idx, n_prompts)

    # Write .pt files once to disk for optional reuse across restarts.
    logger.info("Saving MLP input cache to %s ...", cache_dir)
    for i, tensor in enumerate(layer_inputs):
        torch.save(tensor, os.path.join(cache_dir, f"layer_{i}.pt"))
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

    return {"meta": meta, "layer_inputs": layer_inputs}


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
    parser.add_argument("--dtype", default="bfloat16", choices=DTYPE_CHOICES)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()

    if HF_READ_TOKEN:
        login(HF_READ_TOKEN)

    dtype = resolve_torch_dtype(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model: %s", args.model)
    hf_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    hf_model = hf_model.to(device)
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    adapter = HFLlamaGraphAdapter(hf_model, tokenizer, device)

    dataset_path = _resolve_dataset_path(args.dataset)
    build_mlp_input_cache(
        adapter,
        dataset_path,
        args.model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

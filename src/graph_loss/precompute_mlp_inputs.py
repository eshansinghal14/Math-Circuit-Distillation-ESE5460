"""Pre-compute and cache MLP residual-stream inputs for all prompts in a dataset."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import shutil
import tempfile

import torch

from graph_loss.hf_adapter import HFLlamaGraphAdapter
from graph_loss.neuron_activation_heatmap import _parse_numeric_args
from utils import load_model, load_split

logger = logging.getLogger(__name__)


def _model_slug(model_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("._-")[:48]
    digest = hashlib.sha1(model_name.encode()).hexdigest()[:8]
    return f"{safe}_{digest}"


def _dataset_slug(dataset_key: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", dataset_key).strip("._-")[:48]
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


def _tokenize_prompt_batch(adapter, prompts: list[str]) -> tuple[torch.Tensor, list[int]]:
    tokenizer = adapter.tokenizer
    tokenized = [
        tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
        for prompt in prompts
    ]
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is not None:
        tokenized = [
            ids if int(ids[0].item()) == int(bos_token_id)
            else torch.cat([torch.tensor([bos_token_id], dtype=ids.dtype), ids])
            for ids in tokenized
        ]
    lengths = [int(ids.numel()) for ids in tokenized]
    max_len = max(lengths)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or bos_token_id
    input_ids = torch.full(
        (len(tokenized), max_len), int(pad_token_id),
        dtype=tokenized[0].dtype, device=adapter.cfg.device,
    )
    for row_idx, ids in enumerate(tokenized):
        input_ids[row_idx, :ids.numel()] = ids.to(adapter.cfg.device)
    return input_ids, lengths


@torch.no_grad()
def build_mlp_input_cache(
    adapter,
    dataset_key: str,
    model_name: str,
    *,
    data_dict: dict,
    batch_size: int = 64,
    refresh: bool = False,
) -> dict:
    """Build or load an MLP-input cache stored under the system temp directory.

    Checks for an existing cache keyed by model and dataset. If found, loads
    and returns it immediately. Otherwise captures residual-stream inputs
    entering each MLP layer via forward pre-hooks, saves batch-by-batch to
    disk (no upfront full allocation), then concatenates and returns.
    """
    cache_dir = os.path.join(
        tempfile.gettempdir(), "mlp_cache",
        _model_slug(model_name), _dataset_slug(dataset_key),
    )
    if refresh and os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
        logger.info("Deleted existing MLP cache at %s", cache_dir)
    meta_path = os.path.join(cache_dir, "meta.pt")
    if os.path.isfile(meta_path):
        return load_mlp_cache_dir(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)
    n_layers = adapter.n_layers
    d_model = adapter.d_model

    # Parse all prompts; skip only those that have no numeric args at all.
    prompts: list[str] = []
    numeric_args_by_prompt: list[tuple[int, ...]] = []
    for prompt, _ in data_dict.items():
        try:
            nargs = _parse_numeric_args(prompt)
        except ValueError:
            continue
        prompts.append(prompt)
        numeric_args_by_prompt.append(nargs)

    n_prompts = len(prompts)
    if n_prompts == 0:
        raise ValueError(f"No parseable prompts found in dataset '{dataset_key}'")

    logger.info("Building MLP input cache: %d prompts, %d layers, d_model=%d", n_prompts, n_layers, d_model)

    first_ids, _ = _tokenize_prompt_batch(adapter, prompts[:1])
    n_positions = first_ids.shape[1]

    # Write each batch's layer activations to a temporary subdir, then concatenate
    # layer-by-layer so we never hold all layers × all prompts in RAM at once.
    batches_dir = os.path.join(cache_dir, "_batches")
    os.makedirs(batches_dir, exist_ok=True)

    n_batches = 0
    prompt_idx = 0
    for batch_start in range(0, n_prompts, batch_size):
        batch_prompts = prompts[batch_start:batch_start + batch_size]
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
            adapter.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
        finally:
            for h in handles:
                h.remove()

        store_len = min(batch_n_pos, n_positions)
        for layer_idx in range(n_layers):
            acts = captured[layer_idx][:, :store_len, :]  # [bs, store_len, d_model]
            if store_len < n_positions:
                pad = torch.zeros(bs, n_positions - store_len, d_model, dtype=torch.bfloat16)
                acts = torch.cat([acts, pad], dim=1)
            torch.save(acts, os.path.join(batches_dir, f"b{n_batches}_l{layer_idx}.pt"))

        prompt_idx += bs
        n_batches += 1
        logger.info("  Cached %d / %d prompts", prompt_idx, n_prompts)

    logger.info("Saving MLP input cache to %s (concatenating %d batches)", cache_dir, n_batches)
    for layer_idx in range(n_layers):
        batch_tensors = [
            torch.load(
                os.path.join(batches_dir, f"b{b}_l{layer_idx}.pt"),
                map_location="cpu",
                weights_only=True,
            )
            for b in range(n_batches)
        ]
        torch.save(
            torch.cat(batch_tensors, dim=0),
            os.path.join(cache_dir, f"layer_{layer_idx}.pt"),
        )
    shutil.rmtree(batches_dir)

    n_dims = len(numeric_args_by_prompt[0])
    arg_values = [
        sorted({args[dim] for args in numeric_args_by_prompt if dim < len(args)})
        for dim in range(n_dims)
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
        "dataset_key": dataset_key,
    }
    torch.save(meta, meta_path)
    logger.info("MLP input cache written to %s", cache_dir)
    return load_mlp_cache_dir(cache_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Pre-compute MLP residual-stream inputs for all dataset prompts.")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--dataset", required=True, help="Dataset name under datasets/ (e.g. 22_add)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--refresh", action="store_true", help="Delete and rebuild cache even if it exists")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    device = next(model.parameters()).device
    adapter = HFLlamaGraphAdapter(model, tokenizer, device)
    all_data = load_split(args.dataset, "all")
    build_mlp_input_cache(adapter, args.dataset, args.model, data_dict=all_data, batch_size=args.batch_size, refresh=args.refresh)


if __name__ == "__main__":
    main()

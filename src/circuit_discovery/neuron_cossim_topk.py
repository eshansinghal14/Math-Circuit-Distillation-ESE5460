"""
Compute per-neuron mean pairwise cosine similarity of (pooled) activations across all
problems in ``datasets/2d_add_all.json`` for 1B and 8B, save to JSON, and optionally
build a circuit-discovery-style model with fixed binary masks (top-k by this metric).

Run from ``src``::

    python -m circuit_discovery.neuron_cossim_topk --k-classes 8
    python -m circuit_discovery.neuron_cossim_topk --k-classes 8 --frac-activated 0.05 --circuit-checkpoint path/to/epoch_100.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import torch
from torch import nn

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _src not in sys.path:
    sys.path.insert(0, _src)

from circuit_discovery.utils import _stack_layer_activations, config, llama_1b, llama_8b
from circuit_discovery.models import CircuitDiscoveryModel


def _results_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "..", "results", "circuit-discovery")


def default_cossim_json_path() -> str:
    return os.path.join(_results_dir(), "neuron_mean_pairwise_cossim.json")


def mean_pairwise_cossim_per_neuron(pooled: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean pairwise cosine similarity across problems, per neuron.

    ``pooled`` has shape ``[N, D]`` (one pooled activation vector per problem). For column
    ``d``, let ``v_i`` be the scalar activation of neuron ``d`` on problem ``i``. We use
    ``cos(i,j) = v_i v_j / (|v_i||v_j|)`` and report the mean over all pairs ``i < j``.

    Vectorized: with ``vn_i = v_i / |v_i|``, ``sum_{i<j} vn_i vn_j = ((sum vn)^2 - sum vn^2) / 2``.
    """
    if pooled.dim() != 2:
        raise ValueError(f"Expected [N, D], got {tuple(pooled.shape)}")
    n = pooled.size(0)
    if n < 2:
        raise ValueError("Need at least two problems for pairwise statistics")
    vn = pooled / pooled.abs().clamp(min=eps)
    s = vn.sum(dim=0)
    ss = (vn**2).sum(dim=0)
    pair_sum = (s * s - ss) / 2.0
    num_pairs = n * (n - 1) / 2.0
    return pair_sum / num_pairs


def collect_pooled_activations_all_batches(model_name: str, batch_size: int) -> torch.Tensor:
    """Returns ``[N, D]`` float32 CPU tensor (mean over sequence positions per problem)."""
    from neuron_distillation.activations import NeuronActivationsGenerator

    gen = NeuronActivationsGenerator(model_name, batch_size=batch_size)
    n_ex = gen.ids.shape[0]
    num_batches = (n_ex + batch_size - 1) // batch_size
    chunks: List[torch.Tensor] = []
    for b in range(num_batches):
        batch = gen.generate_batch_activations(b, log=False)
        stacked = _stack_layer_activations(batch["activations"])
        pooled = stacked.mean(dim=1).float().cpu()
        chunks.append(pooled)
    gen.remove_handles()
    return torch.cat(chunks, dim=0)


def build_cossim_record(pooled_1b: torch.Tensor, pooled_8b: torch.Tensor) -> Dict[str, Any]:
    c1 = mean_pairwise_cossim_per_neuron(pooled_1b)
    c2 = mean_pairwise_cossim_per_neuron(pooled_8b)
    return {
        "schema": "neuron_mean_pairwise_cossim_v1",
        "num_problems": int(pooled_1b.size(0)),
        "1b": {
            "dim": int(pooled_1b.size(1)),
            "intermediate_size": int(config["1b"].intermediate_size),
            "num_hidden_layers": int(config["1b"].num_hidden_layers),
            "mean_pairwise_cossim": c1.tolist(),
        },
        "8b": {
            "dim": int(pooled_8b.size(1)),
            "intermediate_size": int(config["8b"].intermediate_size),
            "num_hidden_layers": int(config["8b"].num_hidden_layers),
            "mean_pairwise_cossim": c2.tolist(),
        },
    }


def load_cossim_record(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cossim_record(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


class FixedBinaryNeuronMask(nn.Module):
    """Same binary mask for every class (and forward path), values in ``{0, 1}``."""

    def __init__(self, k_classes: int, activations_dim: int, mask_01: torch.Tensor):
        super().__init__()
        self.k_classes = k_classes
        self.activations_dim = activations_dim
        if mask_01.shape != (activations_dim,):
            raise ValueError(f"mask_01 must be [{activations_dim}], got {tuple(mask_01.shape)}")
        self.register_buffer("mask", mask_01.float().clamp(0.0, 1.0))

    def forward(self, class_probs, activations, mask_temperature):
        del class_probs, mask_temperature
        b = activations.size(0)
        m = self.mask.unsqueeze(0).expand(b, -1)
        masked_activations = activations * m.unsqueeze(1)
        return masked_activations, m

    def class_masks(self, mask_temperature):
        del mask_temperature
        return self.mask.unsqueeze(0).expand(self.k_classes, -1)


def topk_binary_mask(dim: int, cossim: List[float], k: int) -> torch.Tensor:
    """Indices with largest mean pairwise cossim get weight 1."""
    if k <= 0 or k > dim:
        raise ValueError(f"k must be in [1, {dim}], got {k}")
    t = torch.tensor(cossim, dtype=torch.float32)
    _, idx = torch.topk(t, k, largest=True)
    m = torch.zeros(dim, dtype=torch.float32)
    m[idx] = 1.0
    return m


def build_sparse_circuit_model(
    k_classes: int,
    mask_1b: torch.Tensor,
    mask_8b: torch.Tensor,
    mask_temperature: float = 1.0,
    circuit_checkpoint: Optional[str] = None,
) -> CircuitDiscoveryModel:
    from utils import _resolve_ckpt_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CircuitDiscoveryModel(k_classes=k_classes, mask_temperature=mask_temperature).to(device)

    if circuit_checkpoint is not None:
        ckpt_path = _resolve_ckpt_path(circuit_checkpoint)
        ckpt_data = torch.load(ckpt_path, map_location=device)
        state = ckpt_data["model_state_dict"]
        prefix_ok = {k: v for k, v in state.items() if not k.startswith("neuron_masks")}
        model.load_state_dict(prefix_ok, strict=False)

    d1 = mask_1b.numel()
    d8 = mask_8b.numel()
    model.neuron_masks_1b = FixedBinaryNeuronMask(k_classes, d1, mask_1b.to(device)).to(device)
    model.neuron_masks_8b = FixedBinaryNeuronMask(k_classes, d8, mask_8b.to(device)).to(device)
    return model


def ensure_cossim_file(
    out_path: str,
    batch_size: int,
    force: bool,
) -> Dict[str, Any]:
    if os.path.isfile(out_path) and not force:
        print(f"Using existing cossim file (skip recompute): {out_path}")
        return load_cossim_record(out_path)

    print("Computing pooled activations and per-neuron mean pairwise cossim (1b, 8b)...")
    p1 = collect_pooled_activations_all_batches(llama_1b, batch_size)
    p8 = collect_pooled_activations_all_batches(llama_8b, batch_size)
    if p1.size(0) != p8.size(0):
        raise RuntimeError(f"Problem count mismatch 1b={p1.size(0)} vs 8b={p8.size(0)}")
    record = build_cossim_record(p1, p8)
    save_cossim_record(out_path, record)
    print(f"Wrote {out_path}")
    return record


def run(
    k_classes: int,
    cossim_json: str,
    batch_size: int,
    frac_activated: Optional[float],
    circuit_checkpoint: Optional[str],
    force_recompute_cossim: bool,
) -> None:
    os.makedirs(_results_dir(), exist_ok=True)
    record = ensure_cossim_file(cossim_json, batch_size, force=force_recompute_cossim)

    c1 = record["1b"]["mean_pairwise_cossim"]
    c2 = record["8b"]["mean_pairwise_cossim"]
    d1 = len(c1)
    d8 = len(c2)

    if frac_activated is None:
        return

    if not (0.0 < frac_activated <= 1.0):
        raise ValueError("frac_activated must be in (0, 1]")

    k1 = max(1, int(round(frac_activated * d1)))
    k8 = max(1, int(round(frac_activated * d8)))
    print(f"Building binary masks: top-{k1}/{d1} (1b), top-{k8}/{d8} (8b) by mean pairwise cossim")

    mask_1b = topk_binary_mask(d1, c1, k1)
    mask_8b = topk_binary_mask(d8, c2, k8)

    model = build_sparse_circuit_model(
        k_classes=k_classes,
        mask_1b=mask_1b.cpu(),
        mask_8b=mask_8b.cpu(),
        mask_temperature=1.0,
        circuit_checkpoint=circuit_checkpoint,
    )

    tag = f"sparse_binary_frac{frac_activated:g}_k1b{k1}_k8b{k8}_kcls{k_classes}.pt"
    out_pt = os.path.join(_results_dir(), tag)
    payload = {
        "k_classes": k_classes,
        "frac_activated": frac_activated,
        "k_1b": k1,
        "k_8b": k8,
        "dim_1b": d1,
        "dim_8b": d8,
        "mask_1b": mask_1b.cpu().clone(),
        "mask_8b": mask_8b.cpu().clone(),
        "cossim_json": os.path.abspath(cossim_json),
        "circuit_checkpoint": circuit_checkpoint,
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, out_pt)
    print(f"Saved sparse binary circuit model to {out_pt}")


def load_saved_sparse_binary_checkpoint(path: str, device: Optional[str] = None) -> CircuitDiscoveryModel:
    """Load a file produced by ``run(..., frac_activated=...)``."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(path, map_location=dev)
    m = build_sparse_circuit_model(
        k_classes=int(data["k_classes"]),
        mask_1b=data["mask_1b"].to(dev),
        mask_8b=data["mask_8b"].to(dev),
        mask_temperature=1.0,
        circuit_checkpoint=None,
    )
    m.load_state_dict(data["model_state_dict"], strict=True)
    return m


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Neuron pairwise cossim stats + optional sparse binary circuit model")
    p.add_argument("--k-classes", type=int, required=True, help="Number of circuit classes (for saved model)")
    p.add_argument(
        "--cossim-json",
        type=str,
        default=None,
        help=f"Path for neuron cossim JSON (default: {default_cossim_json_path()})",
    )
    p.add_argument("--batch-size", type=int, default=50, help="Batch size for activation generation")
    p.add_argument(
        "--frac-activated",
        type=float,
        default=None,
        help="If set, fraction of neurons to keep (top-k by cossim per tower); builds and saves a .pt model",
    )
    p.add_argument(
        "--circuit-checkpoint",
        type=str,
        default=None,
        help="Optional trained circuit checkpoint to copy problem_encoder/classifier from (masks replaced)",
    )
    p.add_argument(
        "--force-recompute-cossim",
        action="store_true",
        help="Recompute cossim JSON even if it already exists",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    cossim_path = args.cossim_json or default_cossim_json_path()
    run(
        k_classes=args.k_classes,
        cossim_json=cossim_path,
        batch_size=args.batch_size,
        frac_activated=args.frac_activated,
        circuit_checkpoint=args.circuit_checkpoint,
        force_recompute_cossim=args.force_recompute_cossim,
    )


if __name__ == "__main__":
    main()

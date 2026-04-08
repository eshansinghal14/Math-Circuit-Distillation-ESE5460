"""
Compute per-neuron mean pairwise cosine similarity of (pooled) activations across all
problems in ``datasets/<PREFIX>_all.json`` (default prefix ``2d_add``) for 1B and 8B, save
to JSON, and optionally build a :class:`~circuit_discovery.models.CircuitDiscoveryModel`
using standard :class:`~circuit_discovery.models.NeuronMask` modules initialized from a
binary top-k mask (checkpoint format matches training runs so ``load_model_checkpoint`` /
``clustering.py`` work with ``--k-classes 1``).

Run from ``src`` (always builds a **single-class** ``k_classes=1`` model)::

    python -m circuit_discovery.neuron_cossim_topk
    python -m circuit_discovery.neuron_cossim_topk --dataset 2d1d_mult
    python -m circuit_discovery.neuron_cossim_topk --frac-activated 0.05

Gated Llama weights use ``HF_TOKEN`` from ``constants.py`` (see ``circuit_discovery.utils``), same as elsewhere in this repo.
"""

from __future__ import annotations

# Binary-mask pipeline is only defined for one latent class.
K_CLASSES = 1

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import torch

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _src not in sys.path:
    sys.path.insert(0, _src)

from circuit_discovery.utils import _stack_layer_activations, config, llama_1b, llama_8b
from circuit_discovery.models import CircuitDiscoveryModel, neuron_mask_from_binary_mask


def _results_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "..", "results", "circuit-discovery")


def default_cossim_json_path(dataset_prefix: str) -> str:
    safe = dataset_prefix.replace(os.sep, "_").replace("/", "_")
    return os.path.join(_results_dir(), f"neuron_mean_pairwise_cossim_{safe}.json")


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


def collect_pooled_activations_all_batches(
    model_name: str,
    batch_size: int,
    *,
    dataset_prefix: str,
) -> torch.Tensor:
    """Returns ``[N, D]`` float32 CPU tensor (mean over sequence positions per problem)."""
    from neuron_distillation.activations import NeuronActivationsGenerator

    gen = NeuronActivationsGenerator(
        model_name,
        batch_size=batch_size,
        dataset_prefix=dataset_prefix,
    )
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


def build_cossim_record(
    pooled_1b: torch.Tensor,
    pooled_8b: torch.Tensor,
    dataset_prefix: str,
) -> Dict[str, Any]:
    c1 = mean_pairwise_cossim_per_neuron(pooled_1b)
    c2 = mean_pairwise_cossim_per_neuron(pooled_8b)
    return {
        "schema": "neuron_mean_pairwise_cossim_v1",
        "dataset_prefix": dataset_prefix,
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


def mean_cossim_topk_neurons(cossim: List[float], k: int) -> float:
    """Average of the *k* largest per-neuron mean pairwise cossim values (the activated set)."""
    if k <= 0:
        raise ValueError("k must be positive")
    t = torch.tensor(cossim, dtype=torch.float64)
    k = min(k, t.numel())
    topv, _ = torch.topk(t, k, largest=True)
    return float(topv.mean().item())


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
    mask_1b: torch.Tensor,
    mask_8b: torch.Tensor,
    mask_temperature: float = 1.0,
) -> CircuitDiscoveryModel:
    """``CircuitDiscoveryModel`` with standard :class:`NeuronMask` modules (``k_classes=1``)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CircuitDiscoveryModel(k_classes=K_CLASSES, mask_temperature=mask_temperature).to(device)
    T = model.mask_temperature
    model.neuron_masks_1b = neuron_mask_from_binary_mask(mask_1b.to(device), T, k_classes=K_CLASSES).to(device)
    model.neuron_masks_8b = neuron_mask_from_binary_mask(mask_8b.to(device), T, k_classes=K_CLASSES).to(device)
    return model


def ensure_cossim_file(
    out_path: str,
    batch_size: int,
    force: bool,
    *,
    dataset_prefix: str,
) -> Dict[str, Any]:
    if os.path.isfile(out_path) and not force:
        print(f"Using existing cossim file (skip recompute): {out_path}")
        return load_cossim_record(out_path)

    print(
        f"Computing pooled activations and per-neuron mean pairwise cossim "
        f"(1b, 8b) on dataset prefix {dataset_prefix!r}..."
    )
    p1 = collect_pooled_activations_all_batches(
        llama_1b, batch_size, dataset_prefix=dataset_prefix,
    )
    p8 = collect_pooled_activations_all_batches(
        llama_8b, batch_size, dataset_prefix=dataset_prefix,
    )
    if p1.size(0) != p8.size(0):
        raise RuntimeError(f"Problem count mismatch 1b={p1.size(0)} vs 8b={p8.size(0)}")
    record = build_cossim_record(p1, p8, dataset_prefix)
    save_cossim_record(out_path, record)
    print(f"Wrote {out_path}")
    return record


def run(
    cossim_json: str,
    batch_size: int,
    frac_activated: Optional[float],
    force_recompute_cossim: bool,
    *,
    dataset_prefix: str,
) -> None:
    os.makedirs(_results_dir(), exist_ok=True)
    record = ensure_cossim_file(
        cossim_json,
        batch_size,
        force=force_recompute_cossim,
        dataset_prefix=dataset_prefix,
    )

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

    avg_c_1b = mean_cossim_topk_neurons(c1, k1)
    avg_c_8b = mean_cossim_topk_neurons(c2, k8)
    print(f"Mean pairwise cossim (activated / top-k neurons): 1b={avg_c_1b:.6f}, 8b={avg_c_8b:.6f}")

    model = build_sparse_circuit_model(
        mask_1b=mask_1b.cpu(),
        mask_8b=mask_8b.cpu(),
        mask_temperature=1.0,
    )

    safe_ds = dataset_prefix.replace(os.sep, "_").replace("/", "_")
    tag = f"sparse_binary_{safe_ds}_frac{frac_activated:g}_k1b{k1}_k8b{k8}.pt"
    out_pt = os.path.join(_results_dir(), tag)
    # Same keys as training checkpoints so ``utils.load_model_checkpoint`` / clustering work.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    torch.save(
        {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics_log": [],
            "sparse_binary_metadata": {
                "k_classes": K_CLASSES,
                "dataset_prefix": dataset_prefix,
                "frac_activated": frac_activated,
                "k_1b": k1,
                "k_8b": k8,
                "dim_1b": d1,
                "dim_8b": d8,
                "mask_1b": mask_1b.cpu().clone(),
                "mask_8b": mask_8b.cpu().clone(),
                "cossim_json": os.path.abspath(cossim_json),
            },
        },
        out_pt,
    )
    print(f"Saved sparse binary circuit model to {out_pt}")
    print("  Load with: utils.load_model_checkpoint(path, k_classes=1, lr=1e-3) or use --k-classes 1 in clustering.")


def load_saved_sparse_binary_checkpoint(path: str) -> CircuitDiscoveryModel:
    """Thin wrapper: same as ``load_model_checkpoint(path, k_classes=1, ...)`` for sparse exports."""
    from utils import load_model_checkpoint

    model, _, _, _ = load_model_checkpoint(path, k_classes=K_CLASSES, lr=1e-3)
    return model


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Neuron pairwise cossim stats + optional sparse binary circuit model (k_classes=1 only).",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="2d_add",
        metavar="PREFIX",
        help="Dataset family prefix: loads repo datasets/<PREFIX>_all.json (default: 2d_add)",
    )
    p.add_argument(
        "--cossim-json",
        type=str,
        default=None,
        help=(
            "Path for neuron cossim JSON "
            "(default: results/circuit-discovery/neuron_mean_pairwise_cossim_<PREFIX>.json)"
        ),
    )
    p.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=50,
        metavar="N",
        help="Batch size for activation generation (default: 50)",
    )
    p.add_argument(
        "--frac-activated",
        type=float,
        default=None,
        help="If set, fraction of neurons to keep (top-k by cossim per tower); builds and saves a .pt model",
    )
    p.add_argument(
        "--force-recompute-cossim",
        action="store_true",
        help="Recompute cossim JSON even if it already exists",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    ds = args.dataset.strip()
    if not ds:
        raise SystemExit("ERROR: --dataset PREFIX must be non-empty (e.g. 2d_add, 2d1d_mult).")
    cossim_path = args.cossim_json or default_cossim_json_path(ds)
    run(
        cossim_json=cossim_path,
        batch_size=args.batch_size,
        frac_activated=args.frac_activated,
        force_recompute_cossim=args.force_recompute_cossim,
        dataset_prefix=ds,
    )


if __name__ == "__main__":
    main()

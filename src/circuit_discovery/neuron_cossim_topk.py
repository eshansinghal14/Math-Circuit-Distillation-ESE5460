"""
Compute per-neuron mean pairwise cosine similarity of **full token trajectories** (no
mean-pooling over sequence length) across all problems in ``datasets/<PREFIX>_test.json``
(default prefix ``2d_add``) for 1B and 8B, save to JSON, and optionally build a
:class:`~circuit_discovery.models.CircuitDiscoveryModel`
using standard :class:`~circuit_discovery.models.NeuronMask` modules initialized from a
binary top-k mask (checkpoint format matches training runs so ``load_model_checkpoint`` /
``clustering.py`` work with ``--k-classes 1``).

Outputs: cossim JSON under ``<Math Circuit Distillation (ESE 5460)>/<dataset>/`` when
Colab Drive is mounted (else ``results/circuit-discovery/<dataset>/``); sparse ``.pt`` under
``.../<dataset>/frac<value>/`` when ``--frac-activated`` is set.

Run from ``src`` (always builds a **single-class** ``k_classes=1`` model)::

    python -m circuit_discovery.neuron_cossim_topk
    python -m circuit_discovery.neuron_cossim_topk --dataset 2d1d_mult
    python -m circuit_discovery.neuron_cossim_topk --frac-activated 0.05
    python -m circuit_discovery.neuron_cossim_topk --dataset 222_add --res-token 1

Gated Llama weights use ``HF_READ_TOKEN`` from ``constants.py`` (see ``circuit_discovery.utils``), same as elsewhere in this repo.

Activations are accumulated in one pass per model with **streaming** (only the ``[T, D]`` sum tensor is kept, not full ``[N, T, D]``), so RAM scales as ``O(T \\cdot D)`` plus one batch. Use a smaller ``--batch-size`` if GPU memory is tight.
"""

from __future__ import annotations

# Binary-mask pipeline is only defined for one latent class.
K_CLASSES = 1

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _src not in sys.path:
    sys.path.insert(0, _src)

from circuit_discovery.utils import _stack_layer_activations, config, llama_1b, llama_8b
from circuit_discovery.models import CircuitDiscoveryModel, neuron_mask_from_binary_mask
from utils import get_default_device

# Colab: store under Drive when mounted; otherwise repo ``results/circuit-discovery``.
_DRIVE_MCD_ROOT = "/content/drive/My Drive/Math Circuit Distillation (ESE 5460)"


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _neuron_cossim_workspace_root() -> str:
    """``<Drive MCD root>`` if present, else ``<repo>/results/circuit-discovery``."""
    drive = os.path.abspath(_DRIVE_MCD_ROOT)
    if os.path.isdir(drive):
        return drive
    return os.path.join(_repo_root(), "results", "circuit-discovery")


def _dataset_segment(dataset_prefix: str) -> str:
    return dataset_prefix.replace(os.sep, "_").replace("/", "_")


def _dataset_res_segment(dataset_prefix: str, res_token: Optional[int] = None) -> str:
    seg = _dataset_segment(dataset_prefix)
    if res_token is None:
        return seg
    return f"{seg}_{int(res_token)}"


def _dataset_mode_segment(trajectory_space: str) -> str:
    return "residual_write" if trajectory_space == "residual_write" else "activation"


def dataset_cossim_dir(dataset_prefix: str, res_token: Optional[int] = None) -> str:
    """Directory for neuron mean pairwise cossim JSON: ``<workspace>/<dataset[_res_token]>/``."""
    return os.path.join(
        _neuron_cossim_workspace_root(),
        _dataset_res_segment(dataset_prefix, res_token),
    )


def sparse_binary_checkpoint_dir(
    dataset_prefix: str,
    frac_activated: float,
    res_token: Optional[int] = None,
) -> str:
    """Sparse ``.pt`` dir: ``<workspace>/<dataset[_res_token]>/frac<value>/``."""
    seg = _dataset_res_segment(dataset_prefix, res_token)
    frac_folder = f"frac{frac_activated:g}"
    return os.path.join(_neuron_cossim_workspace_root(), seg, frac_folder)


def default_cossim_json_path(
    dataset_prefix: str,
    res_token: Optional[int] = None,
    trajectory_space: str = "activation",
) -> str:
    safe = _dataset_segment(dataset_prefix)
    mode_suffix = "" if trajectory_space == "activation" else f"_{_dataset_mode_segment(trajectory_space)}"
    suffix = "" if res_token is None else f"_restok{int(res_token)}"
    return os.path.join(
        dataset_cossim_dir(dataset_prefix, res_token),
        f"neuron_mean_pairwise_cossim_{safe}{suffix}{mode_suffix}.json",
    )


def mean_pairwise_cossim_from_normalized_sum(
    sum_v: torch.Tensor,
    n: int,
    *,
    chunk_d: int = 65536,
) -> torch.Tensor:
    """Per-neuron mean pairwise cosine from ``s = \\sum_i v_i`` with ``v_i`` unit-norm in ``ℝ^T``.

    ``sum_v`` has shape ``[T, D]``. Computes ``((‖s_d‖^2 - N)/2) / \\binom{N}{2}`` per neuron ``d``.
    Processes columns in ``chunk_d`` blocks to limit peak memory.
    """
    if n < 2:
        raise ValueError("Need at least two problems for pairwise statistics")
    if sum_v.dim() != 2:
        raise ValueError(f"Expected [T, D] sum tensor, got {tuple(sum_v.shape)}")
    num_pairs = n * (n - 1) / 2.0
    _, d = sum_v.shape
    out = torch.empty(d, dtype=torch.float32)
    sum_v = sum_v.float()
    for start in range(0, d, chunk_d):
        end = min(start + chunk_d, d)
        sl = sum_v[:, start:end]
        pair_sum = ((sl * sl).sum(dim=0) - n) / 2.0
        out[start:end] = (pair_sum / num_pairs).to(torch.float32)
    return out


def mean_pairwise_cossim_per_neuron(per_problem: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Same statistic as streaming path, but materializes ``[N, T, D]`` (for tests / small runs)."""
    if per_problem.dim() != 3:
        raise ValueError(f"Expected [N, T, D], got {tuple(per_problem.shape)}")
    n = per_problem.size(0)
    vn = F.normalize(per_problem, p=2, dim=1, eps=eps)
    s = vn.sum(dim=0)
    return mean_pairwise_cossim_from_normalized_sum(s, n)


def stream_normalized_sum_token_activations(
    model_name: str,
    batch_size: int,
    *,
    dataset_prefix: str,
    res_token: Optional[int] = None,
    trajectory_space: str = "activation",
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, int, int, int]:
    """Accumulate ``\\sum_i v_i`` where ``v_i`` is the token-wise L2-normalized trajectory for problem ``i``.

    Returns ``(sum_v, n, T, D)`` with ``sum_v`` on **CPU float32**, shape ``[T, D]``. Memory is
    ``O(T \\cdot D)`` instead of ``O(N \\cdot T \\cdot D)`` for the full activation tensor.
    """
    from neuron_distillation.activations import NeuronActivationsGenerator

    if trajectory_space not in {"activation", "residual_write"}:
        raise ValueError(
            "trajectory_space must be one of {'activation', 'residual_write'}"
        )

    gen = NeuronActivationsGenerator(
        model_name,
        batch_size=batch_size,
        dataset_prefix=dataset_prefix,
        res_token=res_token,
    )
    n_ex = int(gen.ids.shape[0])
    if n_ex < 2:
        gen.remove_handles()
        raise ValueError("Need at least two problems for pairwise statistics")
    num_batches = (n_ex + batch_size - 1) // batch_size
    acc: Optional[torch.Tensor] = None
    n_total = 0
<<<<<<< HEAD
    dev = get_default_device()
=======
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    residual_write_scale: Optional[torch.Tensor] = None
>>>>>>> 71f9ea2a26b8e5db54e17dcb6fe971b88fb73533

    if trajectory_space == "residual_write":
        # For Llama MLPs, each neuron writes a fixed residual direction through down_proj.
        # Per-neuron column norms let us represent residual-write trajectories as
        # activation trajectories scaled per-neuron.
        model = gen.model
        scales = []
        for layer in model.model.layers:
            col_norm = layer.mlp.down_proj.weight.detach().float().norm(dim=0)
            scales.append(col_norm)
        residual_write_scale = torch.cat(scales, dim=0).to(dev)

    for b in range(num_batches):
        batch = gen.generate_batch_activations(b, log=False)
        stacked = _stack_layer_activations(batch["activations"]).float()
        if stacked.device != dev:
            stacked = stacked.to(dev)
        if residual_write_scale is not None:
            if stacked.size(-1) != residual_write_scale.numel():
                gen.remove_handles()
                raise RuntimeError(
                    f"Residual-write scale dim mismatch: activations have D={stacked.size(-1)} "
                    f"but scales have D={residual_write_scale.numel()}"
                )
            stacked = stacked * residual_write_scale.view(1, 1, -1)
        vn = F.normalize(stacked, p=2, dim=1, eps=eps)
        chunk_sum = vn.sum(dim=0)
        if acc is None:
            acc = chunk_sum.cpu().to(torch.float32)
            t0, d0 = acc.shape
        else:
            if chunk_sum.shape != (t0, d0):
                gen.remove_handles()
                raise RuntimeError(
                    f"Inconsistent activation shape across batches: expected {(t0, d0)}, got {tuple(chunk_sum.shape)}"
                )
            acc.add_(chunk_sum.cpu().to(torch.float32))
        n_total += stacked.size(0)
        del batch, stacked, vn, chunk_sum
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    gen.remove_handles()
    assert acc is not None
    if n_total != n_ex:
        raise RuntimeError(f"Internal error: counted {n_total} problems but dataset has {n_ex}")
    return acc, n_total, int(acc.size(0)), int(acc.size(1))


def build_cossim_record(
    *,
    sum_norm_1b: torch.Tensor,
    n_1b: int,
    sum_norm_8b: torch.Tensor,
    n_8b: int,
    dataset_prefix: str,
    res_token: Optional[int],
    trajectory_space: str,
) -> Dict[str, Any]:
    if n_1b != n_8b:
        raise RuntimeError(f"Problem count mismatch 1b={n_1b} vs 8b={n_8b}")
    c1 = mean_pairwise_cossim_from_normalized_sum(sum_norm_1b, n_1b)
    c2 = mean_pairwise_cossim_from_normalized_sum(sum_norm_8b, n_8b)
    return {
        "schema": "neuron_mean_pairwise_cossim_v3",
        "dataset_prefix": dataset_prefix,
        "res_token": (int(res_token) if res_token is not None else None),
        "trajectory_space": trajectory_space,
        "trajectory_mode": (
            "prefix_for_response_token" if res_token is not None else "full_sequence"
        ),
        "num_problems": int(n_1b),
        "streaming_sum_accumulator": True,
        "1b": {
            "dim": int(sum_norm_1b.size(-1)),
            "seq_len": int(sum_norm_1b.size(0)),
            "intermediate_size": int(config["1b"].intermediate_size),
            "num_hidden_layers": int(config["1b"].num_hidden_layers),
            "mean_pairwise_cossim": c1.tolist(),
        },
        "8b": {
            "dim": int(sum_norm_8b.size(-1)),
            "seq_len": int(sum_norm_8b.size(0)),
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
    device = get_default_device()
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
    res_token: Optional[int] = None,
    trajectory_space: str = "activation",
) -> Dict[str, Any]:
    if os.path.isfile(out_path) and not force:
        print(f"Using existing cossim file (skip recompute): {out_path}")
        return load_cossim_record(out_path)

    print(
        f"Streaming per-token {trajectory_space} trajectories (O(T·D) RAM) and per-neuron mean pairwise cossim "
        f"({('full trajectories' if res_token is None else f'prefix up to response token {int(res_token)}')}," 
        f" 1b then 8b) on dataset prefix {dataset_prefix!r}..."
    )
    sum1, n1, _, _ = stream_normalized_sum_token_activations(
        llama_1b,
        batch_size,
        dataset_prefix=dataset_prefix,
        res_token=res_token,
        trajectory_space=trajectory_space,
    )
    sum8, n8, _, _ = stream_normalized_sum_token_activations(
        llama_8b,
        batch_size,
        dataset_prefix=dataset_prefix,
        res_token=res_token,
        trajectory_space=trajectory_space,
    )
    record = build_cossim_record(
        sum_norm_1b=sum1,
        n_1b=n1,
        sum_norm_8b=sum8,
        n_8b=n8,
        dataset_prefix=dataset_prefix,
        res_token=res_token,
        trajectory_space=trajectory_space,
    )
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
    res_token: Optional[int] = None,
    trajectory_space: str = "activation",
) -> None:
    os.makedirs(dataset_cossim_dir(dataset_prefix, res_token), exist_ok=True)
    record = ensure_cossim_file(
        cossim_json,
        batch_size,
        force=force_recompute_cossim,
        dataset_prefix=dataset_prefix,
        res_token=res_token,
        trajectory_space=trajectory_space,
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

    safe_ds = _dataset_segment(dataset_prefix)
    pt_dir = sparse_binary_checkpoint_dir(dataset_prefix, frac_activated, res_token)
    os.makedirs(pt_dir, exist_ok=True)
    res_tok_tag = "" if res_token is None else f"_restok{int(res_token)}"
    tag = f"sparse_binary_{safe_ds}{res_tok_tag}_frac{frac_activated:g}_k1b{k1}_k8b{k8}.pt"
    out_pt = os.path.join(pt_dir, tag)
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
                "res_token": (int(res_token) if res_token is not None else None),
                "trajectory_space": trajectory_space,
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
        help="Dataset family prefix: loads repo datasets/<PREFIX>_test.json (default: 2d_add)",
    )
    p.add_argument(
        "--cossim-json",
        type=str,
        default=None,
        help=(
            "Path for neuron cossim JSON (default: under Math Circuit Distillation (ESE 5460)/"
            "<dataset>/ on Drive when mounted, else results/circuit-discovery/<dataset>/)"
        ),
    )
    p.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=50,
        metavar="N",
        help="Batch size for forward passes (smaller uses less GPU memory per batch; default: 50)",
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
    p.add_argument(
        "--res-token",
        type=int,
        default=None,
        metavar="K",
        help=(
            "If set, use the forward-pass prefix for generating response token K "
            "(1-indexed). Example: --res-token 1 runs on prompt only, so activations "
            "correspond to generating the first answer token."
        ),
    )
    p.add_argument(
        "--trajectory-space",
        type=str,
        choices=["activation", "residual_write"],
        default="activation",
        help=(
            "Vector used for per-neuron token trajectories before cosine. "
            "'activation' uses raw MLP neuron activations; "
            "'residual_write' scales each neuron activation by ||down_proj[:, i]|| "
            "to represent residual-stream write magnitude."
        ),
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    ds = args.dataset.strip()
    if not ds:
        raise SystemExit("ERROR: --dataset PREFIX must be non-empty (e.g. 2d_add, 2d1d_mult).")
    if args.res_token is not None and args.res_token < 1:
        raise SystemExit("ERROR: --res-token must be >= 1")
    cossim_path = args.cossim_json or default_cossim_json_path(
        ds,
        args.res_token,
        args.trajectory_space,
    )
    run(
        cossim_json=cossim_path,
        batch_size=args.batch_size,
        frac_activated=args.frac_activated,
        force_recompute_cossim=args.force_recompute_cossim,
        dataset_prefix=ds,
        res_token=args.res_token,
        trajectory_space=args.trajectory_space,
    )


if __name__ == "__main__":
    main()

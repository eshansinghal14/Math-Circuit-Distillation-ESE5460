import json
import os
import sys

# Project imports expect `src` on sys.path (works when run from repo root or this file directly).
_SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

import argparse
import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

from utils import (
    LLAMA_1B_MODEL_NAME,
    NEURON_CLUSTERING_SUBDIR,
    get_default_device,
    load_model_checkpoint,
    patch_tokenizer_no_special_tokens,
    _stack_layer_activations,
)
from circuit_discovery.utils import parse_equation
from neuron_distillation.activations import NeuronActivationsGenerator

device = get_default_device()


def _save_k_vs_concordance_plot(
    ks: Sequence[int],
    concordance: Sequence[float],
    *,
    plots_dir: str,
    model_name: str,
    subclass: int,
) -> None:
    """Mean Adjusted Rand index between paired k-means runs vs k (higher = more stable partitions)."""
    ks = list(ks)
    y = [float(x) for x in concordance]
    if len(ks) < 1 or len(y) != len(ks):
        return

    plt.figure(figsize=(6, 4))
    plt.plot(ks, y, marker="o")
    plt.xlabel("k (number of clusters)")
    plt.ylabel("Partition concordance (mean ARI)")
    plt.title(f"k-means stability vs k for {model_name}, subclass {subclass}")
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)

    plot_path = os.path.join(plots_dir, f"k_vs_concordance_subclass_{subclass}.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()


def _set_rng_seeds(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _adjusted_rand_index_numpy(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Adjusted Rand index between two cluster assignments (same points). Pure NumPy."""
    a = np.asarray(labels_a, dtype=np.int64).ravel()
    b = np.asarray(labels_b, dtype=np.int64).ravel()
    if a.size != b.size:
        raise ValueError("label arrays must have the same length")
    n = int(a.size)
    if n < 2:
        return 1.0
    a = np.unique(a, return_inverse=True)[1]
    b = np.unique(b, return_inverse=True)[1]
    n_a = int(a.max()) + 1
    n_b = int(b.max()) + 1
    contingency = np.zeros((n_a, n_b), dtype=np.int64)
    np.add.at(contingency, (a, b), 1)
    comb_ij = contingency.astype(np.float64) * (contingency.astype(np.float64) - 1.0) / 2.0
    sum_comb = float(comb_ij.sum())
    a_sum = contingency.sum(axis=1).astype(np.float64)
    b_sum = contingency.sum(axis=0).astype(np.float64)
    sum_comb_a = float((a_sum * (a_sum - 1.0) / 2.0).sum())
    sum_comb_b = float((b_sum * (b_sum - 1.0) / 2.0).sum())
    comb_n = n * (n - 1) / 2.0
    if comb_n <= 0:
        return 1.0
    prod_combs = sum_comb_a * sum_comb_b / comb_n
    mean_comb = (sum_comb_a + sum_comb_b) / 2.0
    denom = mean_comb - prod_combs
    if abs(denom) < 1e-15:
        return 1.0 if abs(sum_comb - prod_combs) < 1e-15 else 0.0
    return (sum_comb - prod_combs) / denom


def partition_concordance_ari(
    x: torch.Tensor,
    k: int,
    *,
    num_iters: int = 100,
    n_pairs: int = 5,
) -> float:
    """Mean ARI between ``n_pairs`` independent pairs of cosine k-means runs (replication concordance)."""
    n_points = x.shape[0]
    if k < 1 or k > n_points:
        return float("nan")
    device = x.device
    aris: List[float] = []
    base = 913_733
    for p in range(n_pairs):
        s1 = base + 10_000 * k + 2 * p
        s2 = s1 + 1
        _set_rng_seeds(s1, device)
        x1 = x.clone()
        ids1, _, _ = _kmeans_cosine(x1, k=k, num_iters=num_iters)
        _set_rng_seeds(s2, device)
        x2 = x.clone()
        ids2, _, _ = _kmeans_cosine(x2, k=k, num_iters=num_iters)
        aris.append(
            _adjusted_rand_index_numpy(
                ids1.detach().cpu().numpy(),
                ids2.detach().cpu().numpy(),
            ),
        )
    return float(np.mean(aris))


def _model_path_segments(model_name: str) -> Tuple[str, ...]:
    """HF model id as nested path segments (e.g. ``org/model`` → two folders)."""
    norm = model_name.replace("\\", "/").strip("/")
    return tuple(p.replace(":", "_") for p in norm.split("/") if p)


def _neuron_clustering_run_dir(model_name: str, output_dir: Optional[str] = None) -> str:
    """``<output_dir>/neuron_clustering/<hf-model-as-dirs>/`` (no ``results/`` prefix).

    ``output_dir`` is the run root you pass via ``--output-dir`` (e.g. a ``frac0.001`` folder).
    If omitted, uses the current working directory (``./neuron_clustering/...``).
    """
    base = (output_dir or ".").strip() or "."
    parts: List[str] = [base, NEURON_CLUSTERING_SUBDIR]
    parts.extend(_model_path_segments(model_name))
    return os.path.abspath(os.path.join(*parts))


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Neuron clustering")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset family prefix, e.g. 2d_add -> datasets/2d_add_all.json and related splits",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help=f"HuggingFace model identifier (e.g. {LLAMA_1B_MODEL_NAME})",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Circuit discovery checkpoint to load",
    )
    parser.add_argument(
        "--k-classes",
        type=int,
        default=8,
        help="Number of circuit classes (must match the trained checkpoint)",
    )
    parser.add_argument(
        "--mask-temperature",
        type=float,
        default=None,
        help="Override mask sigmoid temperature T; omit to use value stored in checkpoint",
    )
    parser.add_argument(
        "--mask-activate-thresh",
        type=float,
        default=0.99,
        help="Mask probability above this counts as neuron active (hard threshold)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=500,
        metavar="N",
        help="Batch size for forward passes when collecting neuron features (default: 500)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Run root directory; outputs go to <output_dir>/neuron_clustering/<hf-id-as-dirs>/ "
            "(e.g. .../frac0.001/neuron_clustering/meta-llama/...). Omit to use cwd.",
        ),
    )
    parser.add_argument(
        "--neuron-slice-chunk",
        type=int,
        default=4096,
        metavar="N",
        help=(
            "When gathering activations per subclass, slice this many neuron columns at a time "
            "(lower peak GPU memory during collection; default: 4096)",
        ),
    )
    parser.add_argument(
        "--cluster-k-max",
        type=int,
        default=8,
        metavar="K1",
        help=(
            "Largest k in the k-sweep, inclusive; sweep always starts at k=1 "
            "(default: 19; with default step 2 → 1,3,...,19).",
        ),
    )
    parser.add_argument(
        "--cluster-k-step",
        type=int,
        default=1,
        metavar="S",
        help="Spacing between k values (default: 2).",
    )
    parser.add_argument(
        "--concordance-pairs",
        type=int,
        default=5,
        metavar="P",
        help=(
            "Pairs of independent k-means runs per k for partition concordance (mean ARI); "
            "higher suggests a more stable choice of k (default: 5).",
        ),
    )
    return parser.parse_args(argv)


def _gather_subclass_activations_chunked(
    activations: torch.Tensor,
    ex_mask: torch.Tensor,
    idx: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Subset activations to neuron indices ``idx`` with chunked column slices (lower peak GPU memory).

    Returns a **CPU** float tensor of shape ``[n_c * T, |idx|]``.
    """
    n_neu = int(idx.numel())
    if n_neu == 0:
        return torch.empty(0, 0)
    sub = activations[ex_mask]
    parts: List[torch.Tensor] = []
    cs = max(1, int(chunk_size))
    for start in range(0, n_neu, cs):
        sl = idx[start : start + cs]
        chunk = sub[:, :, sl].reshape(-1, sl.numel())
        parts.append(chunk.detach().float().cpu())
    return torch.cat(parts, dim=1)


def _normalize_rows_inplace_chunked(
    x: torch.Tensor,
    chunk_rows: int = 4096,
    eps: float = 1e-8,
) -> torch.Tensor:
    """In-place row L2 normalize in row chunks (no full-size second buffer; avoids OOM on ``[N,D]``)."""
    n = x.shape[0]
    if n <= chunk_rows:
        norms = x.norm(dim=-1, keepdim=True).clamp_min(eps)
        x.div_(norms)
        return x
    for i in range(0, n, chunk_rows):
        sl = slice(i, min(i + chunk_rows, n))
        sub = x[sl]
        norms = sub.norm(dim=-1, keepdim=True).clamp_min(eps)
        sub.div_(norms)
    return x


def _kmeans_cosine(x, k, num_iters=20):
    """Lloyd k-means on L2-normalized rows (cosine geometry); cluster sizes unbalanced."""
    N, D = x.shape
    if k > N:
        raise ValueError("k cannot be larger than number of points")

    _normalize_rows_inplace_chunked(x)

    indices = []
    first = torch.randint(0, N, (1,), device=x.device)
    indices.append(first.item())
    for _ in range(1, k):
        centers = x[torch.tensor(indices, device=x.device)]
        sim = x @ centers.t()
        closest_sim, _ = sim.max(dim=1)
        dist = (1.0 - closest_sim.clamp(-1.0, 1.0)).clamp(min=0.0)
        dist = torch.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)

        # Avoid re-selecting already chosen centers.
        if indices:
            dist[torch.tensor(indices, device=x.device)] = 0.0

        dist_sum = dist.sum()
        if not torch.isfinite(dist_sum) or dist_sum.item() <= 0.0:
            remaining = torch.ones(N, device=x.device, dtype=torch.bool)
            remaining[torch.tensor(indices, device=x.device)] = False
            remaining_idx = remaining.nonzero(as_tuple=False).squeeze(1)
            if remaining_idx.numel() == 0:
                break
            next_idx = remaining_idx[torch.randint(0, remaining_idx.numel(), (1,), device=x.device)]
            indices.append(int(next_idx.item()))
        else:
            probs = (dist / dist_sum).float()
            # Sample on CPU to avoid CUDA device-side asserts.
            next_idx = torch.multinomial(probs.detach().cpu(), 1).to(x.device)
            indices.append(int(next_idx.item()))

    centroids = x[torch.tensor(indices, device=x.device)]

    prev_cluster_ids = None
    prev_loss = None
    loss = None

    for _ in range(num_iters):
        sim = x @ centroids.t()
        cluster_ids = sim.argmax(dim=1)

        point_sim = sim[torch.arange(N, device=x.device), cluster_ids]
        point_dists = 1.0 - point_sim.clamp(-1.0, 1.0)
        loss = point_dists.mean().item()

        if prev_cluster_ids is not None and torch.equal(cluster_ids, prev_cluster_ids):
            break

        if prev_loss is not None and abs(loss - prev_loss) < 1e-6:
            break

        prev_cluster_ids = cluster_ids.clone()
        prev_loss = loss

        new_centroids = torch.zeros_like(centroids)
        for j in range(k):
            mask = cluster_ids == j
            if mask.any():
                new_centroids[j] = x[mask].mean(dim=0)
            else:
                rand_idx = torch.randint(0, N, (1,), device=x.device)
                new_centroids[j] = x[rand_idx]

        centroids = F.normalize(new_centroids, p=2, dim=-1, eps=1e-8)

    return cluster_ids, centroids, loss


def _collect_neuron_features_per_subclass(
    batch_size=5,
    save_path=None,
    *,
    dataset_prefix: str,
    neuron_slice_chunk: int = 4096,
):
    activations_generator = NeuronActivationsGenerator(
        model_name,
        batch_size=batch_size,
        dataset_prefix=dataset_prefix,
    )
    num_batches = (activations_generator.ids.shape[0] + batch_size - 1) // batch_size

    k_classes = neuron_masks.size(0)

    indices_per_subclass = {}
    for c in range(k_classes):
        mask = neuron_masks[c]
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        indices_per_subclass[c] = idx

    features_lists = {c: [] for c in indices_per_subclass.keys()}

    for batch_idx in range(num_batches):
        batch = activations_generator.generate_batch_activations(batch_idx, log=True)

        ids, activations_dict = batch["ids"], batch["activations"]
        activations = _stack_layer_activations(activations_dict).to(device)
        if k_classes == 1:
            # Single-class masks do not need problem-dependent routing.
            subclass = torch.zeros(activations.size(0), dtype=torch.long, device=device)
        else:
            full_strs = batch.get("full_strs")
            if full_strs is None or any(s is None for s in full_strs):
                if isinstance(ids, torch.Tensor):
                    input_id_list = ids.tolist()
                else:
                    input_id_list = ids
                prompts = tokenizer.batch_decode(input_id_list, skip_special_tokens=True)
            else:
                prompts = [str(s) for s in full_strs]

            op1, op2, res = parse_equation(prompts, device=device)
            classifier_logits = model.classify_problem(op1, op2, res)
            hard = F.gumbel_softmax(classifier_logits, tau=model.tau, dim=-1, hard=True)
            subclass = hard.argmax(dim=-1)

        for c, idx in indices_per_subclass.items():
            ex_mask = subclass == c
            if not ex_mask.any():
                continue
            # Chunked column slices -> CPU [n_c*T, |idx|]; avoids one huge [n_c,T,|idx|] on GPU.
            flat_c = _gather_subclass_activations_chunked(
                activations, ex_mask, idx, chunk_size=neuron_slice_chunk,
            )
            features_lists[c].append(flat_c)

        del activations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    activations_generator.remove_handles()

    features_per_subclass = {}
    for c, feats in features_lists.items():
        if not feats:
            continue
        # feats: list of CPU [n_b*T, |idx|] per batch; cat -> [N_c*T, |idx|] on CPU.
        combined = torch.cat(feats, dim=0)
        feats_flat = combined.T.contiguous()  # [|idx|, N_c*T] on CPU
        features_per_subclass[c] = feats_flat

    if save_path is not None:
        torch.save(
            {
                "model_name": model_name,
                "dataset_prefix": dataset_prefix,
                "features_per_subclass": {c: v.detach().cpu() for c, v in features_per_subclass.items()},
                "indices_per_subclass": {c: idx.detach().cpu() for c, idx in indices_per_subclass.items()},
            },
            save_path,
        )
        print(f"Saved subclass neuron features to {save_path}")

    return features_per_subclass, indices_per_subclass


def _choose_kmeans_device_dtype(x: torch.Tensor) -> Tuple[torch.device, torch.dtype]:
    """Pick device/dtype so the feature matrix fits in memory.

    Row-normalization is in-place chunked, so peak is ~one tensor + chunk norms.
    Prefer CUDA float32, then CUDA float16, else CPU float32.
    """
    n = x.numel()
    if n == 0:
        return torch.device("cpu"), torch.float32
    bytes_f32 = n * 4
    bytes_f16 = n * 2
    if not torch.cuda.is_available():
        return torch.device("cpu"), torch.float32
    torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info()
    # Leave headroom for k-means matmuls (N×k) and other temps.
    frac = 0.82
    if bytes_f32 < free * frac:
        return torch.device("cuda"), torch.float32
    if bytes_f16 < free * frac:
        return torch.device("cuda"), torch.float16
    warnings.warn(
        "k-means feature matrix is too large for available GPU memory (even in float16); "
        "running k-means on CPU (slower).",
        stacklevel=2,
    )
    return torch.device("cpu"), torch.float32


def load_subclass_features_bundle(
    subclass: int,
    results_dir: str,
    batch_size: int = 5,
    *,
    dataset_prefix: str,
    neuron_slice_chunk: int = 4096,
    subclass_features_path: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load (or collect) neuron feature matrix and global indices for one subclass; used for k-sweeps."""
    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    if subclass_features_path is None:
        subclass_features_path = os.path.join(results_dir, "subclass_features.pt")

    if os.path.exists(subclass_features_path):
        ckpt = torch.load(subclass_features_path, map_location="cpu")
        ckpt_dataset = ckpt.get("dataset_prefix")
        if ckpt_dataset is not None and ckpt_dataset != dataset_prefix:
            raise ValueError(
                f"Cached subclass features were built for dataset {ckpt_dataset!r}, "
                f"not {dataset_prefix!r}: {subclass_features_path}"
            )
        features_per_subclass = {int(c): v for c, v in ckpt["features_per_subclass"].items()}
        indices_per_subclass = {int(c): idx for c, idx in ckpt["indices_per_subclass"].items()}
    else:
        features_per_subclass, indices_per_subclass = _collect_neuron_features_per_subclass(
            batch_size=batch_size,
            save_path=subclass_features_path,
            dataset_prefix=dataset_prefix,
            neuron_slice_chunk=neuron_slice_chunk,
        )

    if subclass not in features_per_subclass:
        raise ValueError(f"No features found for subclass {subclass}")

    x = features_per_subclass[subclass].float()
    km_dev, km_dtype = _choose_kmeans_device_dtype(x)
    x = x.to(device=km_dev, dtype=km_dtype)
    subclass_indices = indices_per_subclass[subclass]
    if subclass_indices.device != km_dev:
        subclass_indices = subclass_indices.to(km_dev)
    return x, subclass_indices


def run_neuron_kmeans(
    k,
    subclass: int,
    batch_size=5,
    num_iters=100,
    log=True,
    subclass_features_path=None,
    results_dir=None,
    *,
    dataset_prefix: str,
    neuron_slice_chunk: int = 4096,
    x_preloaded: Optional[torch.Tensor] = None,
    subclass_indices_preloaded: Optional[torch.Tensor] = None,
    rng_seed: Optional[int] = None,
):
    if results_dir is None:
        results_dir = _neuron_clustering_run_dir(model_name)
    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    if subclass_features_path is None:
        subclass_features_path = os.path.join(results_dir, "subclass_features.pt")

    if x_preloaded is not None:
        x = x_preloaded.clone()
        km_dev = x.device
        subclass_indices = subclass_indices_preloaded
        if subclass_indices is None:
            raise ValueError("subclass_indices_preloaded is required when x_preloaded is set")
        if subclass_indices.device != km_dev:
            subclass_indices = subclass_indices.to(km_dev)
    else:
        if subclass_features_path is not None and os.path.exists(subclass_features_path):
            ckpt = torch.load(subclass_features_path, map_location="cpu")
            ckpt_dataset = ckpt.get("dataset_prefix")
            if ckpt_dataset is not None and ckpt_dataset != dataset_prefix:
                raise ValueError(
                    f"Cached subclass features were built for dataset {ckpt_dataset!r}, "
                    f"not {dataset_prefix!r}: {subclass_features_path}"
                )
            features_per_subclass = {int(c): v for c, v in ckpt["features_per_subclass"].items()}
            indices_per_subclass = {int(c): idx for c, idx in ckpt["indices_per_subclass"].items()}
        else:
            features_per_subclass, indices_per_subclass = _collect_neuron_features_per_subclass(
                batch_size=batch_size,
                save_path=subclass_features_path,
                dataset_prefix=dataset_prefix,
                neuron_slice_chunk=neuron_slice_chunk,
            )

        if subclass not in features_per_subclass:
            raise ValueError(f"No features found for subclass {subclass}")

        x = features_per_subclass[subclass].float()
        km_dev, km_dtype = _choose_kmeans_device_dtype(x)
        x = x.to(device=km_dev, dtype=km_dtype)

        subclass_indices = indices_per_subclass[subclass]
        if subclass_indices.device != km_dev:
            subclass_indices = subclass_indices.to(km_dev)

    if rng_seed is not None:
        _set_rng_seeds(int(rng_seed), km_dev)

    cluster_ids, centroids, loss = _kmeans_cosine(x, k=k, num_iters=num_iters)

    cluster_to_indices = {}
    for j in range(k):
        mask = cluster_ids == j
        if mask.any():
            cluster_to_indices[j] = subclass_indices[mask].cpu()
        else:
            cluster_to_indices[j] = torch.empty(0, dtype=subclass_indices.dtype)

    clusters_path = os.path.join(results_dir, f"clusters/subclass_{subclass}_clusters/k{k}.pt")
    os.makedirs(os.path.dirname(clusters_path), exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "subclass": subclass,
            "k": k,
            "cluster_ids": cluster_ids.cpu(),
            "subclass_indices": subclass_indices.cpu(),
            "cluster_to_indices": cluster_to_indices,
            "loss": loss,
        },
        clusters_path,
    )

    if log:
        print(f"Subclass {subclass}: k-means over neurons completed.")
        print(f"Mean cosine distance to centroids (loss): {loss:.6f}")
        for j in range(k):
            size = int((cluster_ids == j).sum().item())
            print(f"Cluster {j}: size={size}")
        print(f"Saved cluster assignments to {clusters_path}")

    return cluster_ids, centroids, loss
if __name__ == "__main__":

    args = _parse_args(sys.argv[1:])
    if args.batch_size < 1:
        raise SystemExit("ERROR: --batch-size must be a positive integer")
    if args.neuron_slice_chunk < 1:
        raise SystemExit("ERROR: --neuron-slice-chunk must be a positive integer")
    if args.cluster_k_max < 1:
        raise SystemExit("ERROR: --cluster-k-max must be >= 1")
    if args.cluster_k_step < 1:
        raise SystemExit("ERROR: --cluster-k-step must be >= 1")
    if args.concordance_pairs < 1:
        raise SystemExit("ERROR: --concordance-pairs must be >= 1")

    model_name = args.model_name
    k_classes = args.k_classes
    model, _, _, _ = load_model_checkpoint(args.checkpoint_path, k_classes=k_classes, lr=1e-3)
    model.eval()

    if args.mask_temperature is not None:
        model.mask_temperature.fill_(float(args.mask_temperature))

    results_dir = _neuron_clustering_run_dir(model_name, args.output_dir)
    os.makedirs(results_dir, exist_ok=True)

    tokenizer = patch_tokenizer_no_special_tokens(
        AutoTokenizer.from_pretrained(model_name),
    )

    mask_on_threshold = args.mask_activate_thresh
    T = model.mask_temperature
    if model_name == LLAMA_1B_MODEL_NAME:
        neuron_masks = model.neuron_masks_1b.class_masks(T)
    else:
        neuron_masks = model.neuron_masks_8b.class_masks(T)
    neuron_masks = neuron_masks > mask_on_threshold

    print("Active neurons ratio:", torch.mean(torch.mean(neuron_masks.float(), dim=1)).item())
    for i in range(k_classes):
        print(neuron_masks[i].count_nonzero().item())

    k_gs_testing = {}
    k_gs_concordance = {}
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    for subclass in range(k_classes):
        if neuron_masks[subclass].any().item():
            print(f"Processing subclass {subclass}")
            k_gs_testing[subclass] = {}
            k_gs_concordance[subclass] = {}
            x_sub, idx_sub = load_subclass_features_bundle(
                subclass,
                results_dir,
                batch_size=args.batch_size,
                dataset_prefix=args.dataset,
                neuron_slice_chunk=args.neuron_slice_chunk,
            )
            for k in range(1, args.cluster_k_max + 1, args.cluster_k_step):
                conc = partition_concordance_ari(
                    x_sub,
                    k,
                    num_iters=100,
                    n_pairs=args.concordance_pairs,
                )
                k_gs_concordance[subclass][k] = conc
                rng_seed = 40_000 + subclass * 1_000 + k
                _, _, loss = run_neuron_kmeans(
                    k,
                    subclass=subclass,
                    log=False,
                    batch_size=args.batch_size,
                    results_dir=results_dir,
                    dataset_prefix=args.dataset,
                    neuron_slice_chunk=args.neuron_slice_chunk,
                    x_preloaded=x_sub,
                    subclass_indices_preloaded=idx_sub,
                    rng_seed=rng_seed,
                )
                k_gs_testing[subclass][k] = loss
                print(
                    f"Subclass {subclass}, k={k}, loss={loss}, "
                    f"concordance(ARI)={conc:.4f}",
                )

            ks = sorted(int(k) for k in k_gs_testing[subclass].keys())
            losses = [float(k_gs_testing[subclass][k]) for k in ks]
            concs = [float(k_gs_concordance[subclass][k]) for k in ks]

            plt.figure(figsize=(6, 4))
            plt.plot(ks, losses, marker="o")
            plt.xlabel("k (number of clusters)")
            plt.ylabel("Mean cosine distance to centroids (loss)")
            plt.title(f"k-means loss vs k for {model_name}, subclass {subclass}")
            plt.grid(True, alpha=0.3)

            plot_path = os.path.join(plots_dir, f"k_vs_loss_subclass_{subclass}.png")
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()

            _save_k_vs_concordance_plot(
                ks,
                concs,
                plots_dir=plots_dir,
                model_name=model_name,
                subclass=subclass,
            )
            if concs:
                # k=1 gives ARI=1 trivially; prefer max over k>=2 when available.
                candidates = [(kv, cv) for kv, cv in zip(ks, concs) if kv > 1]
                if not candidates:
                    k_best, c_best = ks[0], concs[0]
                else:
                    k_best, c_best = max(candidates, key=lambda t: t[1])
                print(
                    f"Subclass {subclass}: k with max partition concordance (mean ARI, k≥2): "
                    f"{k_best} (ARI={c_best:.4f})",
                )

    out_path = os.path.join(results_dir, "k_gs_testing.json")
    with open(out_path, "w") as f:
        json.dump(k_gs_testing, f, indent=2)
    conc_path = os.path.join(results_dir, "k_gs_concordance.json")
    with open(conc_path, "w") as f:
        json.dump(k_gs_concordance, f, indent=2)

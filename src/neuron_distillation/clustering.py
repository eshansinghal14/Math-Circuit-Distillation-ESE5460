import json
import os
import sys

# Project imports expect `src` on sys.path (works when run from repo root or this file directly).
_SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

import argparse
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

from utils import load_model_checkpoint, _stack_layer_activations
from circuit_discovery.utils import parse_equation
from neuron_distillation.activations import NeuronActivationsGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"


def _safe_model_dir(model_name: str) -> str:
    """Filesystem-safe folder name for a HuggingFace model id."""
    return model_name.replace("/", "_").replace(":", "_")


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Neuron clustering")
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="HuggingFace model identifier (e.g. meta-llama/Llama-3.2-1B)",
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
            "Subdirectory under results/neuron-clustering/ before the model folder "
            "(default: default). Full path: results/neuron-clustering/<this>/<model-name>/",
        ),
    )
    parser.add_argument(
        "--kmeans-device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help=(
            "Device for balanced k-means (cpu avoids large [|neurons|, tokens] tensors on GPU; "
            "cuda is faster if it fits in VRAM)",
        ),
    )
    parser.add_argument(
        "--max-feature-dim",
        type=int,
        default=None,
        metavar="D",
        help=(
            "If set, randomly subsample token positions to D columns before k-means "
            "(reduces [|neurons|, tokens] memory; use when many activated neurons)",
        ),
    )
    parser.add_argument(
        "--feature-subsample-seed",
        type=int,
        default=0,
        help="RNG seed for --max-feature-dim column choice (default: 0)",
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
        default=19,
        metavar="K1",
        help=(
            "Largest k in the k-sweep, inclusive; sweep always starts at k=1 "
            "(default: 19; with default step 2 → 1,3,...,19).",
        ),
    )
    parser.add_argument(
        "--cluster-k-step",
        type=int,
        default=2,
        metavar="S",
        help="Spacing between k values (default: 2).",
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


def _maybe_subsample_feature_columns(
    x: torch.Tensor,
    max_dim: Optional[int],
    seed: int = 0,
) -> torch.Tensor:
    """``x`` is ``[|neurons|, n_tokens]``. Optionally subsample token columns to cap k-means memory."""
    if max_dim is None or x.shape[1] <= max_dim:
        return x
    torch.manual_seed(seed)
    perm = torch.randperm(x.shape[1], device=x.device)[:max_dim]
    return x[:, perm]


def _kmeans_cosine(x, k, num_iters=20):
    N, D = x.shape
    if k > N:
        raise ValueError("k cannot be larger than number of points")

    x = F.normalize(x, p=2, dim=-1, eps=1e-8)

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

    base_cap = N // k
    remainder = N % k
    capacities = torch.full((k,), base_cap, device=x.device, dtype=torch.long)
    if remainder > 0:
        capacities[:remainder] += 1

    prev_cluster_ids = None
    prev_loss = None
    loss = None

    for _ in range(num_iters):
        sim = x @ centroids.t()
        dists = 1.0 - sim.clamp(-1.0, 1.0)

        cluster_ids = torch.full((N,), -1, device=x.device, dtype=torch.long)
        remaining_cap = capacities.clone()

        _, sorted_clusters = torch.sort(dists, dim=1)

        for rank in range(k):
            unassigned = cluster_ids.eq(-1)
            if not unassigned.any():
                break

            cand_clusters = sorted_clusters[unassigned, rank]
            unassigned_idx = unassigned.nonzero(as_tuple=False).squeeze(1)

            for j in range(k):
                if remaining_cap[j] <= 0:
                    continue

                want_j_mask = cand_clusters.eq(j)
                if not want_j_mask.any():
                    continue

                cand_indices = unassigned_idx[want_j_mask]
                take = min(remaining_cap[j].item(), cand_indices.numel())
                if take <= 0:
                    continue

                chosen = cand_indices[:take]
                cluster_ids[chosen] = j
                remaining_cap[j] -= take

        if (cluster_ids == -1).any():
            raise RuntimeError("Balanced k-means assignment failed: some points unassigned")

        if prev_cluster_ids is not None and torch.equal(cluster_ids, prev_cluster_ids):
            break

        point_sim = sim[torch.arange(N, device=x.device), cluster_ids]
        point_dists = 1.0 - point_sim.clamp(-1.0, 1.0)
        loss = point_dists.mean().item()

        if prev_loss is not None and loss is not None:
            if abs(loss - prev_loss) < 1e-6:
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
    neuron_slice_chunk: int = 4096,
):
    activations_generator = NeuronActivationsGenerator(model_name, batch_size=batch_size)
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

        if isinstance(ids, torch.Tensor):
            input_id_list = ids.tolist()
        else:
            input_id_list = ids

        prompts = tokenizer.batch_decode(input_id_list, skip_special_tokens=True)
        activations = _stack_layer_activations(activations_dict).to(device)

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
                "features_per_subclass": {c: v.detach().cpu() for c, v in features_per_subclass.items()},
                "indices_per_subclass": {c: idx.detach().cpu() for c, idx in indices_per_subclass.items()},
            },
            save_path,
        )
        print(f"Saved subclass neuron features to {save_path}")

    return features_per_subclass, indices_per_subclass


def run_neuron_kmeans(
    k,
    subclass: int,
    batch_size=5,
    num_iters=100,
    log=True,
    subclass_features_path=None,
    results_dir=None,
    *,
    kmeans_device: str = "cpu",
    max_feature_dim: Optional[int] = None,
    feature_subsample_seed: int = 0,
    neuron_slice_chunk: int = 4096,
):
    if results_dir is None:
        results_dir = os.path.join(
            "results", "neuron-clustering", "default", _safe_model_dir(model_name),
        )
    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    if subclass_features_path is None:
        subclass_features_path = os.path.join(results_dir, "subclass_features.pt")

    if subclass_features_path is not None and os.path.exists(subclass_features_path):
        ckpt = torch.load(subclass_features_path, map_location="cpu")
        features_per_subclass = {int(c): v for c, v in ckpt["features_per_subclass"].items()}
        indices_per_subclass = {int(c): idx for c, idx in ckpt["indices_per_subclass"].items()}
    else:
        features_per_subclass, indices_per_subclass = _collect_neuron_features_per_subclass(
            batch_size=batch_size,
            save_path=subclass_features_path,
            neuron_slice_chunk=neuron_slice_chunk,
        )

    if subclass not in features_per_subclass:
        raise ValueError(f"No features found for subclass {subclass}")

    x = features_per_subclass[subclass].float()
    x = _maybe_subsample_feature_columns(x, max_feature_dim, feature_subsample_seed)
    if kmeans_device == "cuda" and torch.cuda.is_available():
        km_dev = torch.device("cuda")
    else:
        km_dev = torch.device("cpu")
    x = x.to(km_dev)

    subclass_indices = indices_per_subclass[subclass]
    if subclass_indices.device != km_dev:
        subclass_indices = subclass_indices.to(km_dev)

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


def print_problem_counts_per_class(model, tokenizer, k_classes, device, batch_size=128):
    """Argmax class assignment over `datasets/2d_add_all.json` (same source as activation gen)."""
    dataset_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "datasets", "2d_add_all.json"
    )
    if not os.path.isfile(dataset_path):
        print(f"Warning: dataset not found at {dataset_path}, skipping problems-per-class counts.")
        return
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    n = len(dataset)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0
    counts = torch.zeros(k_classes, dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            rows = [dataset[i]["ids"] for i in range(start, end)]
            max_len = max(len(r) for r in rows)
            batch_rows = [r + [pad_id] * (max_len - len(r)) for r in rows]
            batch_ids = torch.tensor(batch_rows, dtype=torch.long, device=device)
            prompts = tokenizer.batch_decode(batch_ids, skip_special_tokens=True)
            op1, op2, res = parse_equation(prompts, device=device)
            logits = model.classify_problem(op1, op2, res)
            pred = logits.argmax(dim=-1)
            for c in range(k_classes):
                counts[c] += (pred == c).sum()
    print("Problems per class (classifier argmax on full dataset):")
    for c in range(k_classes):
        print(f"  class {c}: {int(counts[c].item())}")
    print(f"  total: {int(counts.sum().item())} (dataset size {n})")


if __name__ == "__main__":

    args = _parse_args(sys.argv[1:])
    if args.batch_size < 1:
        raise SystemExit("ERROR: --batch-size must be a positive integer")
    if args.neuron_slice_chunk < 1:
        raise SystemExit("ERROR: --neuron-slice-chunk must be a positive integer")
    if args.max_feature_dim is not None and args.max_feature_dim < 1:
        raise SystemExit("ERROR: --max-feature-dim must be >= 1 when set")
    if args.cluster_k_max < 1:
        raise SystemExit("ERROR: --cluster-k-max must be >= 1")
    if args.cluster_k_step < 1:
        raise SystemExit("ERROR: --cluster-k-step must be >= 1")

    model_name = args.model_name
    k_classes = args.k_classes
    model, _, _, _ = load_model_checkpoint(args.checkpoint_path, k_classes=k_classes, lr=1e-3)
    model.eval()

    if args.mask_temperature is not None:
        model.mask_temperature.fill_(float(args.mask_temperature))

    out_seg = args.output_dir if args.output_dir is not None else "default"
    results_dir = os.path.join(
        "results", "neuron-clustering", out_seg, _safe_model_dir(model_name),
    )
    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print_problem_counts_per_class(model, tokenizer, k_classes, device)

    mask_on_threshold = args.mask_activate_thresh
    T = model.mask_temperature
    if model_name == "meta-llama/Llama-3.2-1B":
        neuron_masks = model.neuron_masks_1b.class_masks(T)
    else:
        neuron_masks = model.neuron_masks_8b.class_masks(T)
    neuron_masks = neuron_masks > mask_on_threshold

    print("Active neurons ratio:", torch.mean(torch.mean(neuron_masks.float(), dim=1)).item())
    for i in range(k_classes):
        print(neuron_masks[i].count_nonzero().item())

    k_gs_testing = {}
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    for subclass in range(k_classes):
        if neuron_masks[subclass].any().item():
            print(f"Processing subclass {subclass}")
            k_gs_testing[subclass] = {}
            for k in range(1, args.cluster_k_max + 1, args.cluster_k_step):
                _, _, loss = run_neuron_kmeans(
                    k,
                    subclass=subclass,
                    log=False,
                    batch_size=args.batch_size,
                    results_dir=results_dir,
                    kmeans_device=args.kmeans_device,
                    max_feature_dim=args.max_feature_dim,
                    feature_subsample_seed=args.feature_subsample_seed,
                    neuron_slice_chunk=args.neuron_slice_chunk,
                )
                k_gs_testing[subclass][k] = loss
                print(f"Subclass {subclass}, k={k}, loss={loss}")

            ks = sorted(int(k) for k in k_gs_testing[subclass].keys())
            losses = [float(k_gs_testing[subclass][k]) for k in ks]

            plt.figure(figsize=(6, 4))
            plt.plot(ks, losses, marker="o")
            plt.xlabel("k (number of clusters)")
            plt.ylabel("Mean cosine distance to centroids (loss)")
            plt.title(f"k-means loss vs k for {model_name}, subclass {subclass}")
            plt.grid(True, alpha=0.3)

            plot_path = os.path.join(plots_dir, f"k_vs_loss_subclass_{subclass}.png")
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()

    out_path = os.path.join(results_dir, "k_gs_testing.json")
    with open(out_path, "w") as f:
        json.dump(k_gs_testing, f, indent=2)

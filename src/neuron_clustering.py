import json
import torch
import os
import sys
import argparse

import torch.nn.functional as F
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

from utils import (

    load_model_checkpoint,
    _stack_layer_activations,
)
from circuit_discovery.utils import parse_equation
from gen_activations_dataset import NeuronActivationsGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    return parser.parse_args(argv)


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


def _collect_neuron_features_per_subclass(batch_size=5, save_path=None):
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
            # activations: [B, T, D]. Pool *after* token-wise comparisons:
            # first aggregate examples, keep token positions distinct.
            acts_c = activations[ex_mask][:, :, idx]  # [n_c, T, |idx|]
            file_feature_c = acts_c.mean(dim=0)  # [T, |idx|]
            features_lists[c].append(file_feature_c)

    activations_generator.remove_handles()

    features_per_subclass = {}
    for c, feats in features_lists.items():
        if not feats:
            continue
        # feats_tensor: [num_files, T, |idx|]
        feats_tensor = torch.stack(feats, dim=0).to(device)
        # Arrange as [|idx|, num_files*T] so downstream kmeans stays 2D.
        feats_flat = feats_tensor.permute(2, 0, 1).reshape(feats_tensor.size(-1), -1)
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
):
    results_dir = os.path.join("results", "neuron-clustering", model_name)
    os.makedirs(results_dir, exist_ok=True)

    if subclass_features_path is None:
        subclass_features_path = os.path.join(results_dir, "subclass_features.pt")

    if subclass_features_path is not None and os.path.exists(subclass_features_path):
        ckpt = torch.load(subclass_features_path, map_location=device)
        features_per_subclass = {int(c): v.to(device) for c, v in ckpt["features_per_subclass"].items()}
        indices_per_subclass = {int(c): idx.to(device) for c, idx in ckpt["indices_per_subclass"].items()}
    else:
        features_per_subclass, indices_per_subclass = _collect_neuron_features_per_subclass(
            batch_size=batch_size, save_path=subclass_features_path
        )

    if subclass not in features_per_subclass:
        raise ValueError(f"No features found for subclass {subclass}")

    x = features_per_subclass[subclass]
    subclass_indices = indices_per_subclass[subclass]

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

    model_name = args.model_name
    model, _, _, _ = load_model_checkpoint(args.checkpoint_path, k_classes=8, lr=1e-3)
    model.eval()

    results_dir = os.path.join("results", "neuron-clustering", model_name)
    os.makedirs(results_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    threshold = 1e-3
    if model_name == "meta-llama/Llama-3.2-1B":
        neuron_masks = model.neuron_masks_1b.class_masks()
    else:
        neuron_masks = model.neuron_masks_8b.class_masks()
    neuron_masks = neuron_masks > (1 - threshold)

    print("Active neurons ratio:", torch.mean(torch.mean(neuron_masks.float(), dim=1)).item())
    for i in range(8):
        print(neuron_masks[i].count_nonzero().item())

    k_gs_testing = {}
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    for subclass in range(8):
        if neuron_masks[subclass].any().item():
            print(f"Processing subclass {subclass}")
            k_gs_testing[subclass] = {}
            for k in range(1, 20, 2):
                _, _, loss = run_neuron_kmeans(k, subclass=subclass, log=False)
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

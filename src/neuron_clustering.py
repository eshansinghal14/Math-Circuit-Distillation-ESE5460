import io
import json
import torch
import boto3
import os
import sys
import shutil

import torch.nn.functional as F
from transformers import AutoTokenizer

from constants import BUCKET_NAME
from utils import (
    load_model_checkpoint,
    get_model_name,
    _stack_layer_activations,
    list_keys,
    suffix_map,
)
from circuit_discovery.utils import parse_equation


s3 = boto3.client("s3")
device = "cuda" if torch.cuda.is_available() else "cpu"

if os.path.exists("/opt/dlami/nvme"):
    cache_dir_resolved = "/opt/dlami/nvme/activations_cache"
else:
    cache_dir_resolved = "/mnt/activations_cache"
os.makedirs(cache_dir_resolved, exist_ok=True)

model_name = get_model_name(sys.argv)
checkpoint_name = "model_5000"
model, _, _, _ = load_model_checkpoint(checkpoint_name, k_classes=8, lr=1e-3)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)

threshold = 1e-3
if model_name == "meta-llama/Llama-3.2-1B":
    neuron_masks = model.neuron_masks_1b.class_masks()
else:
    neuron_masks = model.neuron_masks_8b.class_masks()
neuron_masks = neuron_masks > (1 - threshold)

active_neuron_indices = neuron_masks.any(dim=0)
print("Active neurons ratio:", torch.mean(torch.mean(neuron_masks.float(), dim=1)).item())


def _has_free_space(path, min_bytes=1024 * 1024 * 100):
    root = path
    if not os.path.isdir(root):
        root = os.path.dirname(root) or "."
    total, used, free = shutil.disk_usage(root)
    return free >= min_bytes


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
        dist = 1 - closest_sim.clamp(-1, 1)
        probs = dist / dist.sum()
        next_idx = torch.multinomial(probs, 1)
        indices.append(next_idx.item())

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
    keys = list_keys(model_name)
    key_map = suffix_map(keys)
    suffixes = list(key_map.keys())

    k_classes = neuron_masks.size(0)

    indices_per_subclass = {}
    for c in range(k_classes):
        mask = neuron_masks[c]
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        indices_per_subclass[c] = idx

    features_lists = {c: [] for c in indices_per_subclass.keys()}

    for i in range(0, len(suffixes), batch_size):
        print(f'processing batch {i}/{len(suffixes)}')
        batch_suffixes = suffixes[i: i + batch_size]

        for suffix in batch_suffixes:
            key = key_map[suffix]
            if model_name == "meta-llama/Llama-3.2-1B":
                local = os.path.join(cache_dir_resolved, f"1b_{suffix}")
            else:
                local = os.path.join(cache_dir_resolved, f"8b_{suffix}")

            if os.path.exists(local):
                batch = torch.load(local, map_location="cpu")
            else:
                if _has_free_space(local, min_bytes=1024 ** 3):
                    s3.download_file(BUCKET_NAME, key, local)
                    batch = torch.load(local, map_location="cpu")
                else:
                    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
                    bytestream = io.BytesIO(obj["Body"].read())
                    batch = torch.load(bytestream, map_location="cpu")

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

            mean_activations = activations.mean(dim=1)

            for c, idx in indices_per_subclass.items():
                ex_mask = subclass == c
                if not ex_mask.any():
                    continue
                acts_c = mean_activations[ex_mask][:, idx]
                file_feature_c = acts_c.mean(dim=0)
                features_lists[c].append(file_feature_c)

    features_per_subclass = {}
    for c, feats in features_lists.items():
        if not feats:
            continue
        feats_tensor = torch.stack(feats, dim=0).to(device)
        features_per_subclass[c] = feats_tensor.t()

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
    subclass_features_path=f"../results/neuron-clustering/{model_name}/subclass_features.pt",
):
    results_dir = os.path.join("..", "results", "neuron-clustering", model_name)
    os.makedirs(results_dir, exist_ok=True)

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

    clusters_path = os.path.join(results_dir, f"subclass_{subclass}_clusters/k{k}.pt")
    os.makedirs(results_dir, exist_ok=True)
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
    for i in range(8):
        print(neuron_masks[i].count_nonzero().item())

    k_gs_testing = {}
    for subclass in range(8):
        if neuron_masks[subclass].any().item():
            print(f"Processing subclass {subclass}")
            k_gs_testing[subclass] = {}
            for k in range(1, 10):
                _, _, loss = run_neuron_kmeans(k, subclass=subclass, log=False)
                k_gs_testing[subclass][k] = loss
                print(f"Subclass {subclass}, k={k}, loss={loss}")

    results_dir = os.path.join("..", "results", "neuron-clustering", model_name)
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "k_gs_testing.json")
    with open(out_path, "w") as f:
        json.dump(k_gs_testing, f, indent=2)

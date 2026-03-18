import os
import torch
from transformers import AutoConfig
from huggingface_hub import login

from constants import HF_TOKEN

if HF_TOKEN:
    login(HF_TOKEN)
llama_1b = "meta-llama/Llama-3.2-1B"
llama_8b = "meta-llama/Meta-Llama-3-8B"

config = {
    "1b": AutoConfig.from_pretrained(llama_1b),
    "8b": AutoConfig.from_pretrained(llama_8b),
}


def parse_equation(probs, device=None):
    op1_list = []
    op2_list = []
    res_list = []

    for prob in probs:
        add_idx = prob.index("+")
        equal_idx = prob.index("=")
        op1_str = prob[:add_idx]
        op2_str = prob[add_idx + 1 : equal_idx]
        res_str = prob[equal_idx + 1 :]

        op1_list.append(int(op1_str))
        op2_list.append(int(op2_str))
        res_list.append(int(res_str))

    op1 = torch.tensor(op1_list, dtype=torch.long, device=device)
    op2 = torch.tensor(op2_list, dtype=torch.long, device=device)
    res = torch.tensor(res_list, dtype=torch.long, device=device)

    return op1, op2, res


def merge_activation_batches(batches):
    merged = {}
    ids_list = []
    for b in batches:
        ids_list.append(b["ids"])
        for layer_idx, t in b["activations"].items():
            merged.setdefault(layer_idx, []).append(t)

    ids_cat = torch.cat(ids_list, dim=0) if ids_list else torch.empty(0, dtype=torch.long)
    for layer_idx, chunks in list(merged.items()):
        merged[layer_idx] = torch.cat(chunks, dim=0)
    return {"ids": ids_cat, "activations": merged}


def _stack_layer_activations(batch_activations):
    if not batch_activations:
        raise ValueError("batch_activations is empty")

    layers = sorted(batch_activations.keys())
    tensors = [batch_activations[i] for i in layers]
    return torch.cat(tensors, dim=-1)


# Pairs of keys to log as "name_1b/8b: val_1b/val_8b"
_1B_8B_PAIRS = [
    ("sim_loss_1b", "sim_loss_8b", "sim_loss_1b/8b"),
    ("frac_activated_1b", "frac_activated_8b", "frac_activated_1b/8b"),
    ("sparsity_1b", "sparsity_8b", "sparsity_1b/8b"),
    ("kl_bernoulli_1b", "kl_bernoulli_8b", "kl_bernoulli_1b/8b"),
    ("mask_cossim_1b_loss", "mask_cossim_8b_loss", "mask_cossim_1b/8b"),
]


def log_epoch_metrics(epoch_metrics):
    parts = []
    skip_keys = set()
    if "epoch" in epoch_metrics:
        parts.append(f"epoch: {int(epoch_metrics['epoch'])}")
        skip_keys.add("epoch")
    if "max_class_usage_entropy" in epoch_metrics:
        skip_keys.add("max_class_usage_entropy")
    if "class_counts" in epoch_metrics:
        skip_keys.add("class_counts")
    for key_1b, key_8b, label in _1B_8B_PAIRS:
        if key_1b in epoch_metrics and key_8b in epoch_metrics:
            v1 = epoch_metrics[key_1b]
            v2 = epoch_metrics[key_8b]
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                parts.append(f"{label}: {v1:.4f}/{v2:.4f}")
            else:
                parts.append(f"{label}: {v1}/{v2}")
            skip_keys.add(key_1b)
            skip_keys.add(key_8b)
    for key, value in epoch_metrics.items():
        if key in skip_keys:
            continue
        if key == "class_usage_entropy" and "max_class_usage_entropy" in epoch_metrics:
            max_ent = epoch_metrics["max_class_usage_entropy"]
            parts.append(f"class_usage_entropy: {value:.4f} (max: {max_ent:.4f})")
        elif isinstance(value, (int, float)):
            parts.append(f"{key}: {value:.4f}")
        else:
            parts.append(f"{key}: {value}")
    print(" - ".join(parts))
    if "class_counts" in epoch_metrics:
        counts = epoch_metrics["class_counts"]
        if isinstance(counts, (list, tuple)):
            print("  class counts: " + " - ".join(str(c) for c in counts))
        else:
            print("  class counts:", counts)

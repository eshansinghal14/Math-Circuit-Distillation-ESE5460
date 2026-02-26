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


def log_epoch_metrics(epoch_metrics):
    parts = []
    for key, value in epoch_metrics.items():
        if isinstance(value, (int, float)):
            parts.append(f"{key}: {value:.4f}")
        else:
            parts.append(f"{key}: {value}")
    print(" - ".join(parts))

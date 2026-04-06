import os
import torch
from transformers import AutoConfig
from huggingface_hub import login

try:
    from constants import HF_TOKEN
except ModuleNotFoundError:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")

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


def tower_grad_balance_weights(loss_1b, loss_8b, parameters, eps=1e-8):
    """Inverse grad-norm weights so neither 1b nor 8b tower dominates the combined gradient.

    Uses ||∇_θ L_k|| over all ``parameters`` for each tower loss separately, then
    w_k ∝ 1 / ||g_k|| (normalized to sum to 1). Falls back to 0.5 / 0.5 if norms are non-finite.
    """
    params = [p for p in parameters if p.requires_grad]
    g1 = torch.autograd.grad(loss_1b, params, retain_graph=True, allow_unused=True)
    g2 = torch.autograd.grad(loss_8b, params, retain_graph=True, allow_unused=True)

    def _flatten_norm(gs):
        s = torch.zeros((), device=loss_1b.device, dtype=torch.float32)
        for g in gs:
            if g is not None:
                s = s + g.detach().float().pow(2).sum()
        return torch.sqrt(s + eps)

    n1 = _flatten_norm(g1)
    n2 = _flatten_norm(g2)
    if not (torch.isfinite(n1) and torch.isfinite(n2)):
        return 0.5, 0.5, float("nan"), float("nan")

    inv1 = 1.0 / (n1 + eps)
    inv2 = 1.0 / (n2 + eps)
    s = inv1 + inv2
    w1 = (inv1 / s).item()
    w2 = (inv2 / s).item()
    return w1, w2, n1.item(), n2.item()


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
    for _k in ("prop_active_neurons_1b_per_class", "prop_active_neurons_8b_per_class"):
        skip_keys.add(_k)
    for _k in ("tower_balance_w1", "tower_balance_w2", "tower_grad_norm_1b", "tower_grad_norm_8b"):
        skip_keys.add(_k)
    skip_keys.add("loss")
    skip_keys.add("loss_unweighted")
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
    if "tower_balance_w1" in epoch_metrics and "tower_balance_w2" in epoch_metrics:
        w1 = epoch_metrics["tower_balance_w1"]
        w2 = epoch_metrics["tower_balance_w2"]
        gn1 = epoch_metrics.get("tower_grad_norm_1b", float("nan"))
        gn2 = epoch_metrics.get("tower_grad_norm_8b", float("nan"))
        print(
            f"  tower grad balance: w_1b={w1:.4f} w_8b={w2:.4f} "
            f"||g||_1b={gn1:.4f} ||g||_8b={gn2:.4f}"
        )
    if "class_counts" in epoch_metrics:
        counts = epoch_metrics["class_counts"]
        pa1 = epoch_metrics.get("prop_active_neurons_1b_per_class")
        pa8 = epoch_metrics.get("prop_active_neurons_8b_per_class")
        if (
            isinstance(counts, (list, tuple))
            and isinstance(pa1, (list, tuple))
            and isinstance(pa8, (list, tuple))
            and len(counts) == len(pa1) == len(pa8)
        ):
            parts_cc = []
            for i, n in enumerate(counts):
                parts_cc.append(f"c{i}: {int(n)}/{pa1[i]:.3f}/{pa8[i]:.3f}")
            print("  class counts (n/act1b/act8b): " + " | ".join(parts_cc))
        elif isinstance(counts, (list, tuple)):
            print("  class counts: " + " - ".join(str(c) for c in counts))
        else:
            print("  class counts:", counts)

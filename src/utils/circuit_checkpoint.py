import glob
import os
import re

import torch

from .config import CIRCUIT_DISCOVERY_CKPT_DIR, REPO_ROOT
from .device import get_default_device


def _resolve_ckpt_path(checkpoint: str) -> str:
    """
    Resolve a checkpoint spec to a local .pt file.

    Accepted forms:
    - absolute/relative filepath to a .pt file
    - "latest"
    - "1500" / "epoch_1500" / "epoch_1500.pt"
    """
    if os.path.exists(checkpoint):
        return checkpoint

    ckpt_root = CIRCUIT_DISCOVERY_CKPT_DIR or os.path.join(
        REPO_ROOT, "results", "circuit-discovery", "checkpoints"
    )
    ckpt_root = os.path.abspath(ckpt_root)

    if checkpoint == "latest":
        cand = glob.glob(os.path.join(ckpt_root, "epoch_*.pt"))
        if not cand:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_root}")

        def _epoch_num(p: str) -> int:
            m = re.search(r"epoch_(\d+)\.pt$", os.path.basename(p))
            return int(m.group(1)) if m else -1

        return max(cand, key=_epoch_num)

    m = re.search(r"(\d+)", checkpoint)
    if m:
        epoch = int(m.group(1))
        cand = os.path.join(ckpt_root, f"epoch_{epoch}.pt")
        if os.path.exists(cand):
            return cand

    raise FileNotFoundError(
        f"Could not resolve checkpoint '{checkpoint}'. "
        f"Provide a path to a .pt file, 'latest', or an epoch like '1500'. "
        f"Looked in {ckpt_root}."
    )


def _extract_circuit_model_state_dict(ckpt_data, ckpt_path: str):
    """Resolve model weights from a .pt file saved in different layouts."""
    if not isinstance(ckpt_data, dict):
        raise TypeError(
            f"Checkpoint {ckpt_path!r} must load to a dict, got {type(ckpt_data)}."
        )
    for key in ("model_state_dict", "state_dict"):
        if key in ckpt_data:
            return ckpt_data[key]
    # torch.save(model.state_dict(), path) - weights only, no wrapper dict
    if any(k.startswith("classifier.") for k in ckpt_data.keys()):
        return ckpt_data
    # Common mistake: neuron cluster / feature files
    if "features_per_subclass" in ckpt_data or "cluster_to_indices" in ckpt_data:
        raise ValueError(
            f"{ckpt_path!r} is a neuron-cluster or feature file, not a circuit-discovery "
            "checkpoint. Pass the circuit training checkpoint (e.g. epoch_*.pt with "
            "model weights from circuit_discovery), not k*.pt under clusters/."
        )
    keys_preview = list(ckpt_data.keys())[:12]
    extra = "..." if len(ckpt_data) > 12 else ""
    raise ValueError(
        "No model weights found: expected keys 'model_state_dict' or 'state_dict', "
        "or a raw state_dict with 'classifier.*' keys (from circuit discovery). "
        f"File has keys: {keys_preview}{extra}"
    )


def load_model_checkpoint(checkpoint, k_classes, lr):
    from circuit_discovery.models import CircuitDiscoveryModel

    device = get_default_device()
    ckpt_path = None
    try:
        ckpt_path = _resolve_ckpt_path(checkpoint)
    except FileNotFoundError:
        ckpt_path = None

    if ckpt_path is None:
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint!r}. "
            "Provide a valid path to a .pt file, 'latest', or an epoch number."
        )

    ckpt_data = torch.load(ckpt_path, map_location=device)

    # Auto-detect k_classes from checkpoint weights
    state = _extract_circuit_model_state_dict(ckpt_data, ckpt_path)
    ckpt_k = None
    if "classifier.classifier.4.weight" in state:
        ckpt_k = state["classifier.classifier.4.weight"].shape[0]

    if ckpt_k is not None and ckpt_k != k_classes:
        raise RuntimeError(
            f"Checkpoint was trained with k_classes={ckpt_k} but you "
            f"requested k_classes={k_classes}. Use a checkpoint that "
            f"matches your experiment, or pass --k-classes {ckpt_k}.\n"
            f"  Checkpoint: {ckpt_path}"
        )

    model = CircuitDiscoveryModel(k_classes=k_classes, mask_temperature=1.0).to(device)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        print(
            "Warning: checkpoint missing keys (using model defaults):",
            incompatible.missing_keys,
        )
    if incompatible.unexpected_keys:
        print("Warning: checkpoint unexpected keys ignored:", incompatible.unexpected_keys)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    opt_state = ckpt_data.get("optimizer_state_dict")
    if opt_state is not None:
        optimizer.load_state_dict(opt_state)

    epoch = ckpt_data.get("epoch", ckpt_data.get("step", 0))
    metrics_log = ckpt_data.get("metrics_log", [])
    return model, optimizer, metrics_log, epoch


def _stack_layer_activations(batch_activations):
    if not batch_activations:
        raise ValueError("batch_activations is empty")

    layers = sorted(batch_activations.keys())
    tensors = [batch_activations[i] for i in layers]
    return torch.cat(tensors, dim=-1)

"""Shared training utilities (model loading, checkpointing, history, curves)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import torch


def make_optimizer(model, lr: float):
    """Return a bitsandbytes 8-bit PagedAdamW if available, else standard AdamW."""
    import importlib
    bnb = None
    try:
        bnb = importlib.import_module("bitsandbytes")
    except ImportError:
        pass
    if bnb is not None:
        return bnb.optim.PagedAdamW8bit(params=model.parameters(), lr=lr)
    from torch.optim import AdamW
    return AdamW(params=model.parameters(), lr=lr, foreach=False)


def save_checkpoint(model, tokenizer, save_dir: str) -> None:
    path = os.path.join(save_dir, "final_checkpoint")
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def save_history(history: Dict[str, Any], save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "training_history.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(history), f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def save_curves(history: Dict[str, List], save_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    steps = history.get("train_step", [])
    ce_series = history.get("step_ce_loss", [])
    acc_series = history.get("accuracy", [])
    acc_steps = history.get("accuracy_step", list(range(1, len(acc_series) + 1)))
    if not steps:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(steps[: len(ce_series)], ce_series, marker="o", markersize=2)
    axes[0].set_title("CE Loss")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(acc_steps[: len(acc_series)], acc_series, marker="o", markersize=2)
    axes[1].set_title("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close(fig)

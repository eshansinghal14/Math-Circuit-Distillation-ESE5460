"""Shared training utilities (model loading, checkpointing, history, curves)."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import torch
import torch.nn.functional as F


def kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
    token_chunk_size: int = 64,
) -> torch.Tensor:
    """KL divergence from teacher to student over all non-padding token positions."""
    t = temperature
    vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
    student_flat = student_logits[..., :vocab].reshape(-1, vocab)
    teacher_flat = teacher_logits[..., :vocab].reshape(-1, vocab)
    valid = attention_mask.reshape(-1).bool().nonzero(as_tuple=False).squeeze(-1)
    if valid.numel() == 0:
        return student_logits.sum() * 0.0

    total = student_logits.new_zeros((), dtype=torch.float32)
    for idx in valid.split(max(1, token_chunk_size)):
        s_chunk = student_flat.index_select(0, idx.to(student_flat.device)).float()
        t_chunk = teacher_flat.index_select(0, idx.to(teacher_flat.device)).to(
            device=s_chunk.device,
            dtype=torch.float32,
        )
        log_p_t = F.log_softmax(t_chunk / t, dim=-1)
        log_q_s = F.log_softmax(s_chunk / t, dim=-1)
        p_t = log_p_t.exp()
        total = total + (p_t * (log_p_t - log_q_s)).sum()

    return total / valid.numel() * (t**2)


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


def save_curves(
    history: Dict[str, List],
    save_dir: str,
    loss_key: str = "step_ce_loss",
    loss_label: str = "CE Loss",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    steps = history.get("train_step", [])
    ce_series = history.get(loss_key, [])
    acc_series = history.get("accuracy", [])
    acc_steps = history.get("accuracy_step", list(range(1, len(acc_series) + 1)))
    extra_acc_keys = sorted(k for k in history if k.startswith("accuracy_") and k != "accuracy_step")
    use_legend = bool(extra_acc_keys)
    if not steps:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(steps[: len(ce_series)], ce_series, marker="o", markersize=2)
    axes[0].set_title(loss_label)
    axes[0].grid(True, alpha=0.3)
    if acc_series:
        axes[1].plot(
            acc_steps[: len(acc_series)], acc_series,
            marker="o", markersize=2,
            label="main" if use_legend else None,
        )
    for key in extra_acc_keys:
        ds_name = key[len("accuracy_"):]
        extra_series = history.get(key, [])
        if extra_series:
            axes[1].plot(
                acc_steps[: len(extra_series)], extra_series,
                marker="o", markersize=2, label=ds_name,
            )
    axes[1].set_title("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    if use_legend:
        axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


def add_kd_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("kd_args")
    group.add_argument("--teacher", type=str, required=True)
    group.add_argument("--temperature", type=float, default=2.0)
    group.add_argument("--kl-token-chunk-size", type=int, default=64, dest="kl_token_chunk_size")
    group.add_argument("--eval-datasets", type=str, nargs="*", default=[], dest="eval_datasets",
                       metavar="DATASET", help="Additional datasets to evaluate on at every eval step.")


def add_standard_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("standard_args")
    group.add_argument("--model", type=str, required=True)
    group.add_argument("--dataset", type=str, required=True)
    group.add_argument("--steps", type=int, default=15)
    group.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    group.add_argument("--lr", type=float, default=1e-6)
    group.add_argument("--save-dir", type=str, default="results/sft")
    group.add_argument("--eval-every-n-steps", type=int, default=1, dest="eval_every_n_steps")
    group.add_argument("--grad-accum-steps", type=int, default=1, dest="grad_accum_steps")
    group.add_argument("--max-eval-tokens", type=int, default=256, dest="max_eval_tokens")
    group.add_argument("--test-limit", type=int, default=None, dest="test_limit")

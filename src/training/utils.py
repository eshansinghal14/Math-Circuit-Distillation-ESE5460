"""Shared training utilities (model loading, checkpointing, history, curves)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import torch

from utils import load_model, parse_response, patch_tokenizer_no_special_tokens

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_student_model(model_id: str) -> Tuple:
    """Load a causal LM + tokenizer ready for SFT training.

    Wraps load_model with training-specific setup: bfloat16, response-only
    tokenizer patching, gradient checkpointing, and disabled KV cache.
    """
    print("\n" + "=" * 60)
    print(f"Loading model: {model_id!r}")
    print("=" * 60)
    model, tokenizer = load_model(model_id)

    tokenizer = patch_tokenizer_no_special_tokens(tokenizer)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.to(device=_DEVICE, dtype=torch.bfloat16)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    return model, tokenizer


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


@torch.no_grad()
def eval_model(model, tokenizer, test_dataset, dataset_name: str, batch_size: int, max_eval_tokens: int) -> float:
    """Greedy-generate on test_dataset and return accuracy using dataset-appropriate answer parsing."""
    model.eval()
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    correct = total = 0
    samples = test_dataset.samples
    try:
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            prompts = [s["formatted_prompt"] for s in batch]
            golds = [s["answer"] for s in batch]
            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False,
            ).to(_DEVICE)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_eval_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            gen_texts = tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            )
            for gen, gold in zip(gen_texts, golds):
                pred = parse_response(gen, dataset_name)
                if isinstance(gold, int):
                    if pred == gold:
                        correct += 1
                else:
                    gold_parsed = parse_response(str(gold), dataset_name)
                    if pred is not None and gold_parsed is not None and pred == gold_parsed:
                        correct += 1
                total += 1
    finally:
        tokenizer.padding_side = original_side
    model.train()
    return correct / max(total, 1)


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

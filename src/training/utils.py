"""Shared training utilities (model loading, checkpointing, history, curves)."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
from typing import Any, Dict, List

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Student precision
# ─────────────────────────────────────────────────────────────────────────────
#
# Every trainer loads the student through load_student, runs its forward under
# student_autocast and optimises it with make_optimizer, so SFT, standard KD and
# graph KD share one precision regime and their curves are comparable. There is
# deliberately no flag for any of it. The regime is the standard AMP split: fp32
# master weights, fp32 .grad, bf16 autocast forward, fp32 Adam moments.
#
# bf16 weights had two separate failures, both silent:
#
# 1. Weight updates were rounded away. bf16 carries ~0.39% relative precision, so
#    at lr=1e-5 an Adam step is well under half a ULP for any weight above ~0.004
#    in magnitude. Only ~15.6% of a N(0, 0.02) linear layer changed per step, and
#    RMSNorm gains (~1.0) never moved at any lr up to 1e-4. Training was a sparse
#    fine-tune of the smallest weights, not a slower version of full fine-tuning.
# 2. The graph gradient was rounded away on accumulation. backward_batch_graph_loss
#    backprops one prompt at a time, so the graph term arrives as ~32 small adds
#    into a .grad already holding the far larger KD gradient; each add lands below
#    half a ULP and is dropped. Retention was 66 / 78 / 91 / 100% at lambda
#    0.03 / 0.1 / 0.3 / 1.0.
#
# The bitsandbytes 8-bit AdamW that make_optimizer used to pick whenever it was
# importable is out for a related reason: it stores the Adam moments in a
# blockwise dynamic 8-bit format whose relative resolution is ~3-13% for a
# typical element, coarser than the 0.5-6% of the gradient the graph term
# contributes. fp32 moments cost ~7.5 GB more on a 1B student and remove the
# question.

STUDENT_DTYPE = torch.float32
AUTOCAST_DTYPE = torch.bfloat16


def load_student(model_name: str):
    """Load the trainable student in fp32 master weights.

    Also disables the KV cache and turns on gradient checkpointing, which every
    trainer wants. The forward must then run under :func:`student_autocast`.
    """
    from utils import load_model

    model, tokenizer = load_model(model_name, dtype=STUDENT_DTYPE)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model, tokenizer


def student_autocast():
    """bf16 autocast for the student forward and for eval.

    Generation needs no gradients, so eval runs under the same context; fp32
    generation would double its cost. A no-op off CUDA, and for a bf16 teacher.
    """
    if not torch.cuda.is_available():
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE)


def make_optimizer(model, lr: float):
    """torch AdamW with fp32 moments. See the precision note at the top of this file."""
    from torch.optim import AdamW
    return AdamW(params=model.parameters(), lr=lr, foreach=False)


def run_config_record(config, model, optimizer) -> Dict[str, Any]:
    """Resolved run configuration for the history JSON.

    History files used to record curves but not what produced them, which made a
    lambda=0.1 run indistinguishable from an ablation. This is every dataclass
    field plus the things the dataclass cannot know: the dtype the student actually
    loaded in, the optimizer class actually constructed, and library versions.
    """
    record: Dict[str, Any] = dataclasses.asdict(config)
    first = next(model.parameters())
    record["student_dtype"] = str(first.dtype).removeprefix("torch.")
    record["student_autocast"] = (
        str(AUTOCAST_DTYPE).removeprefix("torch.") if torch.cuda.is_available() else "none"
    )
    record["student_n_params"] = sum(p.numel() for p in model.parameters())
    record["optimizer_class"] = type(optimizer).__qualname__
    record["torch_version"] = torch.__version__
    try:
        import transformers
        record["transformers_version"] = transformers.__version__
    except ImportError:
        pass
    try:
        import bitsandbytes
        record["bitsandbytes_version"] = bitsandbytes.__version__
    except Exception:
        pass
    record["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return record


def describe_run_setup(record: Dict[str, Any]) -> str:
    """One line for the log: precision regime, optimizer, parameter count."""
    return (
        f"Student: {record['student_dtype']} weights, autocast={record['student_autocast']}"
        f" | optimizer={record['optimizer_class']}"
        f" | {record['student_n_params']:,} params"
    )


class ParamChangeCanary:
    """Fraction of parameter entries that the first optimizer step actually changes.

    The direct test for the bf16 rounding failure above: with fp32 weights every
    entry with a nonzero gradient moves, so the fraction should be near 1.0 and the
    RMSNorm gains near 1.0 too. In bf16 at lr=1e-5 it was ~0.16 overall and 0.0 for
    the gains. Samples a stride of each tensor rather than copying the model, so it
    costs a few MB and is run once.
    """

    _SAMPLE = 8192

    def __init__(self, model) -> None:
        self._before: Dict[str, torch.Tensor] = {}
        self._ndim: Dict[str, int] = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            flat = p.detach().flatten()
            stride = max(1, flat.numel() // self._SAMPLE)
            self._before[name] = flat[::stride].to("cpu", copy=True)
            self._ndim[name] = p.ndim

    def report(self, model) -> Dict[str, float]:
        changed = total = 0
        changed_gain = total_gain = 0
        for name, p in model.named_parameters():
            before = self._before.get(name)
            if before is None:
                continue
            flat = p.detach().flatten()
            stride = max(1, flat.numel() // self._SAMPLE)
            after = flat[::stride].to("cpu")
            n = before.numel()
            c = int((after != before).sum().item())
            changed += c
            total += n
            if self._ndim[name] == 1:
                changed_gain += c
                total_gain += n
        return {
            "params_changed_step1": changed / max(total, 1),
            "norm_gains_changed_step1": changed_gain / max(total_gain, 1),
        }


def log_first_step_canary(report: Dict[str, float], history: Dict[str, Any]) -> None:
    """Print and record the first-step change fractions; warn when they say bf16."""
    for key, val in report.items():
        history[key] = val
    frac = report["params_changed_step1"]
    gains = report["norm_gains_changed_step1"]
    print(f"  step 1 | params changed: {frac:.1%} | norm gains changed: {gains:.1%}")
    if frac < 0.5:
        print(
            "  step 1 | WARN: most parameter entries did not change on the first step. "
            "The update is being rounded away: the student is not in fp32, or the lr "
            "is far below the weight precision. Nothing downstream of this is trustworthy."
        )


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


def save_checkpoint(model, tokenizer, save_dir: str, name: str = "final_checkpoint") -> None:
    path = os.path.join(save_dir, name)
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def maybe_save_periodic_checkpoint(
    model,
    tokenizer,
    save_dir: str,
    step: int,
    every: int,
    last_saved_step: int,
    history: Dict[str, Any] | None = None,
) -> int:
    """Overwrite ``<save_dir>/checkpoint`` once ``every`` steps have passed since the last save.

    Returns the step of the most recent save (unchanged if nothing was written).
    """
    if every <= 0 or step - last_saved_step < every:
        return last_saved_step
    save_checkpoint(model, tokenizer, save_dir, name="checkpoint")
    if history is not None:
        save_history(history, save_dir)
    print(f"  Saved checkpoint at step {step} -> {os.path.join(save_dir, 'checkpoint')}")
    return step


def history_path(save_dir: str) -> str:
    """``<save_dir>/<folder name>.json`` -- the run's history is named after its output folder."""
    folder = os.path.basename(os.path.normpath(os.path.abspath(save_dir)))
    return os.path.join(save_dir, f"{folder}.json")


def save_history(history: Dict[str, Any], save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    path = history_path(save_dir)
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


def add_standard_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("standard_args")
    group.add_argument("--model", type=str, required=True)
    group.add_argument("--dataset", type=str, required=True)
    group.add_argument("--eval-datasets", type=str, nargs="*", default=[], dest="eval_datasets",
                    metavar="DATASET", help="Additional datasets to evaluate on at every eval step.")
    group.add_argument("--steps", type=int, default=15)
    group.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    group.add_argument("--lr", type=float, default=1e-6)
    group.add_argument("--save-dir", type=str, default="results/sft")
    group.add_argument("--eval-every-n-steps", type=int, default=1, dest="eval_every_n_steps")
    group.add_argument("--save-every-n-steps", type=int, default=0, dest="save_every_n_steps",
                    help="Overwrite <save-dir>/checkpoint every N train steps (0 = disable).")
    group.add_argument("--grad-accum-steps", type=int, default=1, dest="grad_accum_steps")
    group.add_argument("--max-eval-tokens", type=int, default=256, dest="max_eval_tokens")
    group.add_argument("--test-limit", type=int, default=None, dest="test_limit")

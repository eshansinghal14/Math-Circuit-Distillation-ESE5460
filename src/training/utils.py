"""Shared training utilities (model loading, checkpointing, history, curves)."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import random
import shutil
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import flop_registry as _TORCH_FLOP_REGISTRY


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


# ─────────────────────────────────────────────────────────────────────────────
# FLOP counting
# ─────────────────────────────────────────────────────────────────────────────


class FlopCounter(TorchDispatchMode):
    """Sum the FLOPs of every matmul, attention and convolution kernel dispatched
    while the mode is active, forward and backward alike.

    Uses torch's own per-op formulas (``torch.utils.flop_counter.flop_registry``,
    the ones ``FlopCounterMode`` applies) but not ``FlopCounterMode`` itself: that
    class also runs a ``ModuleTracker`` whose backward hooks call
    ``_will_engine_execute_node`` on leaf tensors, which raises inside
    ``torch.autograd.grad`` on the detached-leaf MLP outputs the attribution
    forwards differentiate to. Elementwise work (norms, activations, softmax, the
    optimizer update) is not counted, matching the usual model-FLOPs convention.
    A dispatch mode routes every aten op through Python, so keep it active only
    around the compute being measured and expect tens of microseconds per op;
    graph KD dispatches ~1M ops per step (2-3x slower), standard KD ~7k.

    One instance may be entered several times in sequence; ``flops`` accumulates
    across entries until ``reset``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.flops = 0

    def reset(self) -> None:
        self.flops = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        formula = _TORCH_FLOP_REGISTRY.get(getattr(func, "_overloadpacket", None))
        if formula is not None:
            self.flops += int(formula(*args, **kwargs, out_val=out))
        return out


class _NullFlopCounter:
    """Stand-in for FlopCounter when --track-flops is off: no dispatch mode, no cost."""

    flops = 0

    def reset(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False


def step_flop_counter(enabled: bool):
    """A FlopCounter when ``enabled``, otherwise a no-op with the same interface."""
    return FlopCounter() if enabled else _NullFlopCounter()


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


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoints and resume
# ─────────────────────────────────────────────────────────────────────────────
#
# Two kinds of checkpoint are written, both only when --save-every-n-steps > 0;
# by default a run leaves just its history JSON and curves. ``<save_dir>/final_checkpoint``
# is the deliverable: weights and tokenizer only, loadable with from_pretrained. The
# periodic ``<save_dir>/checkpoint`` exists so an interrupted run can continue,
# so next to the weights it also carries ``training_state.pt``: the optimizer
# state (fp32 Adam moments, ~2x the parameter count), the train step, the
# history so the curves continue, and the RNG state. ``--resume`` loads the
# weights from that folder and restores the rest through resume_training_state.
#
# The weights and the training state must come from the same save. They once
# did not: a lambda_graph=0.3 run resumed "at step 15" with weights that were
# clearly further along (KL, grad norm and accuracy all jumped to values the
# run only reached much later), so the ~5 GB weights file and the ~10 GB state
# file in ``checkpoint/`` had been written by different saves -- a crash or a
# lost Drive flush partway through one of them leaves exactly that. Two guards:
#
# 1. The periodic checkpoint is written to ``checkpoint.tmp`` in full and only
#    then swapped into place, so ``checkpoint/`` is always one complete save.
# 2. ``training_state.pt`` carries a per-parameter hash of the weights it was
#    saved with, and resume_training_state refuses to continue unless the
#    weights it just loaded hash to the same values. A stale, partial or
#    precision-rounded weights file fails loudly instead of training on.

TRAINING_STATE_FILE = "training_state.pt"
PERIODIC_CHECKPOINT_NAME = "checkpoint"


def weights_fingerprint(model) -> Dict[str, str]:
    """Per-parameter digest of the raw parameter bytes, keyed by parameter name.

    Hashes the exact bytes (plus shape and dtype), so it changes after any
    optimizer step and after any dtype round trip. One parameter is moved to the
    CPU at a time; ~5 GB of fp32 weights takes a few seconds.
    """
    out: Dict[str, str] = {}
    for name, param in model.named_parameters():
        t = param.detach().contiguous().cpu()
        h = hashlib.blake2b(digest_size=16)
        h.update(f"{tuple(t.shape)}:{t.dtype}".encode())
        h.update(np.ascontiguousarray(t.numpy()).reshape(-1).view(np.uint8))
        out[name] = h.hexdigest()
    return out


def verify_weights_fingerprint(model, expected: Dict[str, str], where: str) -> None:
    """Raise unless ``model``'s parameters hash to ``expected`` (see weights_fingerprint)."""
    got = weights_fingerprint(model)
    changed = [n for n in expected if n in got and got[n] != expected[n]]
    missing = [n for n in expected if n not in got]
    extra = [n for n in got if n not in expected]
    if not (changed or missing or extra):
        return
    lines = [
        f"Refusing to resume: the weights loaded from {where} are not the weights its "
        f"{TRAINING_STATE_FILE} was saved with, so the optimizer state and the model would "
        f"be from different points in training.",
        f"  {len(changed)}/{len(expected)} parameters differ, {len(missing)} missing, "
        f"{len(extra)} unexpected.",
    ]
    for n in (changed + missing + extra)[:5]:
        lines.append(f"    {n}")
    lines.append(
        "  The checkpoint folder is inconsistent (a save that did not finish, or a stale weights "
        "file). Delete it and restart the run, or resume from a checkpoint that passes this check."
    )
    raise RuntimeError("\n".join(lines))


def _rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    model,
    tokenizer,
    save_dir: str,
    name: str = "final_checkpoint",
    *,
    optimizer=None,
    step: int | None = None,
    history: Dict[str, Any] | None = None,
    extra_state: Dict[str, Any] | None = None,
) -> None:
    """Write weights and tokenizer to ``<save_dir>/<name>``.

    When ``optimizer`` is given the folder also gets ``training_state.pt`` with
    the optimizer state, step, history, RNG state and a hash of the weights,
    which is what makes the periodic checkpoint resumable. That whole folder is
    staged as ``<name>.tmp`` and swapped in only once every file is written, so
    an interrupted save leaves the previous complete checkpoint untouched rather
    than new weights next to an old optimizer state. The final checkpoint omits
    the training state. ``extra_state`` is any further trainer state that must
    travel with the optimizer (a representation trainer's projector, say); it is
    stored verbatim under ``"extra"`` and handed back by resume_training_state.
    """
    path = os.path.join(save_dir, name)
    if optimizer is None:
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        return

    stage = path + ".tmp"
    if os.path.isdir(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)
    model.save_pretrained(stage)
    tokenizer.save_pretrained(stage)
    state = {
        "optimizer": optimizer.state_dict(),
        "step": int(step or 0),
        "history": dict(history) if history is not None else {},
        "rng": _rng_state(),
        "weights_fingerprint": weights_fingerprint(model),
        "extra": extra_state,
    }
    with open(os.path.join(stage, TRAINING_STATE_FILE), "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())

    # Swap the staged folder in. The previous checkpoint is only removed after
    # the new one is in place, so at every instant ``path`` is a complete save.
    old = path + ".old"
    if os.path.isdir(old):
        shutil.rmtree(old)
    if os.path.isdir(path):
        os.rename(path, old)
    os.rename(stage, path)
    if os.path.isdir(old):
        shutil.rmtree(old, ignore_errors=True)


def resume_checkpoint_dir(save_dir: str) -> str:
    """The periodic checkpoint folder a resumed run loads its student from."""
    path = os.path.join(save_dir, PERIODIC_CHECKPOINT_NAME)
    if not os.path.isfile(os.path.join(path, TRAINING_STATE_FILE)):
        raise FileNotFoundError(
            f"--resume needs {os.path.join(path, TRAINING_STATE_FILE)}; run with "
            f"--save-every-n-steps > 0 first so a resumable checkpoint exists."
        )
    return path


def resume_training_state(model, optimizer, save_dir: str) -> Tuple[int, Dict[str, Any], Any]:
    """Restore optimizer, RNG and history from ``<save_dir>/checkpoint``.

    Returns ``(step, history, extra)`` where ``extra`` is whatever the saver
    passed as ``extra_state`` (None if nothing). ``model`` must be the student loaded from that
    same folder and ``optimizer`` must wrap its parameters. The model is hashed
    and checked against the fingerprint stored with the optimizer state, so a
    checkpoint whose weights and training state came from different saves is
    rejected instead of resumed.
    """
    ckpt_dir = resume_checkpoint_dir(save_dir)
    path = os.path.join(ckpt_dir, TRAINING_STATE_FILE)
    state = torch.load(path, map_location="cpu", weights_only=False)
    fingerprint = state.get("weights_fingerprint")
    if fingerprint is None:
        raise RuntimeError(
            f"{path} has no weights fingerprint, so it cannot be checked against the weights in "
            f"{ckpt_dir}. It was written before that check existed; restart the run rather than "
            f"resuming from it."
        )
    verify_weights_fingerprint(model, fingerprint, ckpt_dir)
    optimizer.load_state_dict(state["optimizer"])
    _restore_rng_state(state["rng"])
    step = int(state["step"])
    print(f"Resumed optimizer, RNG and history from {path} at step {step}; weights verified")
    return step, state["history"], state.get("extra")


# Fields that legitimately differ between the original launch and a resume.
_RESUME_CONFIG_IGNORED = {
    "resume", "device", "torch_version", "transformers_version", "bitsandbytes_version",
}


def record_resumed_config(
    history: Dict[str, Any], config, model, optimizer,
) -> Dict[str, Tuple[Any, Any]]:
    """Append the resumed run's resolved config to ``history["config_resumed"]``.

    The history only ever recorded the config of the original launch, so a resume
    with a different learning rate, temperature or batch size was invisible in
    the JSON. Returns ``{field: (original, resumed)}`` for every training field
    that differs and prints them as a warning.
    """
    record = run_config_record(config, model, optimizer)
    history.setdefault("config_resumed", []).append(record)
    original = history.get("config") or {}
    diffs = {
        k: (original[k], v) for k, v in record.items()
        if k in original and k not in _RESUME_CONFIG_IGNORED and original[k] != v
    }
    if diffs:
        print("WARNING: resumed run's config differs from the original launch:")
        for k, (before, after) in diffs.items():
            print(f"    {k}: {before!r} -> {after!r}")
    return diffs


def maybe_save_periodic_checkpoint(
    model,
    tokenizer,
    save_dir: str,
    step: int,
    every: int,
    last_saved_step: int,
    history: Dict[str, Any] | None = None,
    optimizer=None,
    extra_state: Dict[str, Any] | None = None,
) -> int:
    """Overwrite ``<save_dir>/checkpoint`` once ``every`` steps have passed since the last save.

    Passing ``optimizer`` makes the checkpoint resumable (see save_checkpoint);
    ``extra_state`` rides along in the training state.
    Returns the step of the most recent save (unchanged if nothing was written).
    """
    if every <= 0 or step - last_saved_step < every:
        return last_saved_step
    save_checkpoint(
        model, tokenizer, save_dir, name=PERIODIC_CHECKPOINT_NAME,
        optimizer=optimizer, step=step, history=history, extra_state=extra_state,
    )
    if history is not None:
        save_history(history, save_dir)
    print(f"  Saved checkpoint at step {step} -> {os.path.join(save_dir, PERIODIC_CHECKPOINT_NAME)}")
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
    losses: Sequence[Tuple[str, str]] = (("step_ce_loss", "CE Loss"),),
) -> None:
    """Write ``<save_dir>/training_curves.png``: one panel per ``(history key, title)``
    in ``losses`` against the train step, then an accuracy panel."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    steps = history.get("train_step", [])
    acc_series = history.get("accuracy", [])
    acc_steps = history.get("accuracy_step", list(range(1, len(acc_series) + 1)))
    extra_acc_keys = sorted(k for k in history if k.startswith("accuracy_") and k != "accuracy_step")
    use_legend = bool(extra_acc_keys)
    if not steps:
        return
    n_panels = len(losses) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))
    for ax, (loss_key, loss_label) in zip(axes, losses):
        series = history.get(loss_key, [])
        ax.plot(steps[: len(series)], series, marker="o", markersize=2)
        ax.set_title(loss_label)
        ax.set_xlabel("train step")
        ax.grid(True, alpha=0.3)
    acc_ax = axes[-1]
    if acc_series:
        acc_ax.plot(
            acc_steps[: len(acc_series)], acc_series,
            marker="o", markersize=2,
            label="main" if use_legend else None,
        )
    for key in extra_acc_keys:
        ds_name = key[len("accuracy_"):]
        extra_series = history.get(key, [])
        if extra_series:
            acc_ax.plot(
                acc_steps[: len(extra_series)], extra_series,
                marker="o", markersize=2, label=ds_name,
            )
    acc_ax.set_title("Accuracy")
    acc_ax.set_xlabel("train step")
    acc_ax.set_ylim(0, 1)
    acc_ax.grid(True, alpha=0.3)
    if use_legend:
        acc_ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


def add_kd_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("kd_args")
    group.add_argument("--teacher", type=str, required=True)
    group.add_argument("--temperature", type=float, default=2.0)
    group.add_argument("--kl-token-chunk-size", type=int, default=64, dest="kl_token_chunk_size")
    group.add_argument(
        "--track-flops", "--track_flops", action="store_true", dest="track_flops",
        help="Count the FLOPs of every matmul, attention and convolution kernel in each "
             "train step's forward and backward passes (teacher forward and, for graph "
             "KD, both attribution graphs included) and record the per-step total as "
             "step_flops in the history. Elementwise ops and the optimizer update are "
             "not counted. Routes every aten op through a Python dispatch mode at tens "
             "of microseconds each: negligible for standard KD (~7k ops per step), but "
             "graph KD launches ~1M ops per step, so expect that step to take 2-3x "
             "longer. FLOPs per step are nearly constant, so a few tracked steps "
             "calibrate a run.",
    )


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
                    help="Overwrite <save-dir>/checkpoint (weights + optimizer state) every N "
                         "train steps. 0 (the default) writes no weights at all, neither the "
                         "periodic checkpoint nor <save-dir>/final_checkpoint; the history JSON "
                         "and curves are always written.")
    group.add_argument("--resume", action="store_true",
                    help="Continue from <save-dir>/checkpoint: loads and verifies the student weights, "
                         "optimizer state, step and history saved by --save-every-n-steps.")
    group.add_argument("--grad-accum-steps", type=int, default=1, dest="grad_accum_steps")
    group.add_argument("--max-eval-tokens", type=int, default=256, dest="max_eval_tokens")
    group.add_argument("--test-limit", type=int, default=None, dest="test_limit")

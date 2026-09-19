"""Shared training utilities (model loading, checkpointing, history, curves)."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import hashlib
import json
import math
import os
import random
import shutil
from typing import Any, Callable, Dict, List, Sequence, Tuple

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
# Every trainer seeds python, numpy and torch with this unless --seeds says otherwise.
DEFAULT_SEED = 42


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


def scheduled_lr(step: int, total_steps: int, peak: float, warmup_steps: int, floor: float) -> float:
    """Learning rate for optimizer step ``step`` (1-based): linear warmup from 0 over
    ``warmup_steps``, then cosine decay from ``peak`` to ``floor * peak`` at ``total_steps``.

    Stateless, so --resume needs nothing beyond the restored step count. Added
    2026-09-17: with a constant 1e-6 and no warmup, standard KD on the answer
    positions overshot to 0.51 in-distribution at step 10 and then cycled
    +-0.03 with a ~10-step period for the rest of the run -- an Adam limit
    cycle, visible in every family in phase -- so every at-step number depended
    on the phase the eval landed in. Warmup removes the overshoot; the decay
    damps the cycle. ``warmup_steps=0`` and ``floor=1`` give the old constant
    rate.
    """
    if warmup_steps > 0 and step <= warmup_steps:
        return peak * step / warmup_steps
    if total_steps <= warmup_steps:
        return peak
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return peak * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def apply_lr_schedule(optimizer, step: int, total_steps: int, peak: float, warmup_steps: int, floor: float) -> float:
    """Set every param group's lr for this step and return it."""
    lr = scheduled_lr(step, total_steps, peak, warmup_steps, floor)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


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
        + (f" | seed={record['seed']}" if "seed" in record else "")
    )


class ParamStepTracker:
    """How far the weights actually move on each optimizer step.

    Two runs can share a learning rate and still take very different sized steps:
    Adam's update is ``lr * m / (sqrt(v) + eps)``, so a gradient whose sign pattern
    is persistent across batches produces near-full-size steps while a noisy one
    produces much smaller ones, and gradient clipping rescales by a factor that
    changes over training. ``|dtheta|`` separates "this objective carries more
    signal" from "this objective just moves the weights further per step".

    Samples a stride of each tensor the way ParamChangeCanary does, so it costs a
    few MB and no host transfer, and is cheap enough to run every step. The sample
    is a fixed stride, so the ratio it reports is an unbiased estimate of the
    whole model's relative movement.

    Usage: ``snapshot()`` before ``optimizer.step()``, ``delta()`` after.
    """

    _SAMPLE = 8192

    def __init__(self, model) -> None:
        self._names = []
        for name, p in model.named_parameters():
            if p.requires_grad:
                self._names.append(name)
        self._before: Dict[str, torch.Tensor] = {}

    def _sample(self, p: torch.Tensor) -> torch.Tensor:
        flat = p.detach().flatten()
        stride = max(1, flat.numel() // self._SAMPLE)
        return flat[::stride]

    def snapshot(self, model) -> None:
        params = dict(model.named_parameters())
        self._before = {
            n: self._sample(params[n]).clone()
            for n in self._names if n in params
        }

    def delta(self, model) -> tuple[float, float]:
        """``(|dtheta|, |dtheta| / |theta|)`` over the sampled entries."""
        if not self._before:
            return float("nan"), float("nan")
        params = dict(model.named_parameters())
        d_sq = 0.0
        p_sq = 0.0
        for n, before in self._before.items():
            p = params.get(n)
            if p is None:
                continue
            after = self._sample(p)
            if after.shape != before.shape:
                continue
            diff = (after.float() - before.float())
            d_sq += float(diff.pow(2).sum().item())
            p_sq += float(after.float().pow(2).sum().item())
        self._before = {}
        if p_sq <= 0:
            return d_sq ** 0.5, float("nan")
        return d_sq ** 0.5, (d_sq ** 0.5) / (p_sq ** 0.5)


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


def kd_position_mask(attention_mask: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """``[B, L]`` mask of the logit positions the KD term is computed on: the
    positions that *predict* an answer token or the EOS after it, i.e.
    ``response_mask`` shifted left by one (logits at position ``p`` predict token
    ``p + 1``), exactly the positions the SFT loss scores.

    The KD term is scored on these positions only. Scoring every non-padding
    position (the behaviour before 2026-09-17) spent most of the loss off the
    task: with BOS in the sequence the largest single term was the two models'
    disagreement over how a document starts, scored at the attention-sink
    position (~2.4 of ~14 nats per sequence), and the prompt positions scored
    the teacher's prior over random operands; the student fell to ~0.01
    in-distribution accuracy within ten steps of KD.
    """
    mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    mask[:, :-1] = response_mask[:, 1:].bool()
    return mask & attention_mask.bool()


def first_answer_token_accuracy(logits: torch.Tensor, input_ids: torch.Tensor, response_mask: torch.Tensor) -> float:
    """Teacher-forced accuracy of the first answer token on a training batch.

    The argmax of the logits at the last prompt position against the gold token
    that follows it: the same prediction greedy decoding makes for its first
    token, taken from the training forward. If this is high while the eval
    accuracy is low, the gap is in generation or parsing; if both are low, the
    model has genuinely lost the answer.
    """
    first = response_mask.int().argmax(dim=1)
    rows = torch.arange(input_ids.size(0), device=input_ids.device)
    pred = logits[rows, first - 1].argmax(dim=-1)
    return float((pred == input_ids[rows, first]).float().mean().item())


def kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
    token_chunk_size: int = 64,
) -> torch.Tensor:
    """KL divergence from teacher to student, averaged over the positions where
    ``attention_mask`` is set; pass :func:`kd_position_mask` for the KD term."""
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
# Two kinds of checkpoint are written: the periodic one only when --save-every-n-steps > 0,
# the final one whenever it is nonzero (-1 means final only);
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
    "anova_cache_device", "teacher_target_cache_dir",
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


# Key under which a history JSON keeps every seed that has written to its folder.
HISTORY_RUNS_KEY = "runs"
# Seeds sharing a history file must have trained the same way; settings that only
# change what is written or measured on the side are not differences.
_SAME_FILE_CONFIG_IGNORED = _RESUME_CONFIG_IGNORED | {
    "seed", "save_dir", "save_every_n_steps", "track_flops", "track_grad_metrics", "graph_verbose",
}


# Keys a trainer may store in the ``shared`` dict run_seeds threads through its
# builds. Everything here is independent of the seed: the frozen teacher, the
# untrained student's and the teacher's baseline accuracies, and (graph KD, ANOVA
# path) the MLP-input caches built from the untrained models.
SHARED_TEACHER = "teacher"
SHARED_BASELINES = "baselines"


def completed_seeds(save_dir: str, steps: int) -> Dict[int, int]:
    """Seeds whose run in ``save_dir``'s history JSON reached ``steps``, mapped to their last step.

    A seed whose entry stops short (a crash after a periodic checkpoint wrote the
    history) is not counted, so run_seeds trains it again.
    """
    path = history_path(save_dir)
    if not os.path.exists(path):
        return {}
    try:
        runs = load_history_runs(path)
    except (OSError, ValueError) as e:
        print(f"WARN: could not read {path} ({e}); treating no seed as done")
        return {}
    done: Dict[int, int] = {}
    for key, run in runs.items():
        train_steps = run.get("train_step") or []
        last = max(train_steps) if train_steps else 0
        if last >= steps:
            done[int(key)] = last
    return done


def run_seeds(
    seeds: Sequence[int],
    resume: bool,
    build_trainer: Callable[[int, Dict[str, Any]], Any],
    *,
    save_dir: str | None = None,
    steps: int | None = None,
    redo: bool = False,
) -> List[Dict[str, Any]]:
    """Train once per seed, in order, and return each run's history.

    A seed that already has a finished run (one that reached ``steps``) in
    ``save_dir``'s history JSON is skipped unless ``redo`` is set, so a seed list
    can be re-issued after an interruption and only the missing seeds run. An
    entry that stopped short of ``steps`` is treated as absent and re-run from
    scratch; use ``--resume`` with that single seed to continue it instead.

    ``build_trainer(seed, shared)`` constructs a fresh trainer for that seed. The
    student and optimizer are rebuilt every time; ``shared`` is one dict that
    lives across the loop, in which a trainer keeps what does not depend on the
    seed (see :func:`shared_teacher`, :func:`run_baselines`), so the teacher is
    loaded and the baselines are evaluated for the first seed only. Each trainer
    is dropped and the CUDA cache emptied before the next seed starts, so a list
    costs no more peak memory than a single run. Seeds are de-duplicated,
    keeping first occurrence. ``--resume`` continues one specific checkpoint and
    so is refused with more than one seed.
    """
    seeds = list(dict.fromkeys(int(s) for s in seeds))
    if not seeds:
        raise ValueError("--seeds needs at least one seed")
    if resume and len(seeds) > 1:
        raise ValueError("--resume continues a single run's checkpoint; pass exactly one seed with it")
    if save_dir is not None and steps is not None and not redo and not resume:
        done = completed_seeds(save_dir, steps)
        skipped = [s for s in seeds if s in done]
        if skipped:
            print(f"Skipping seed(s) {skipped}: already finished ({steps} steps) in {history_path(save_dir)}; "
                  f"pass --redo-seeds to train them again.")
            seeds = [s for s in seeds if s not in done]
        if not seeds:
            print("Every requested seed is already done; nothing to train.")
            return []
    shared: Dict[str, Any] = {}
    histories: List[Dict[str, Any]] = []
    for i, seed in enumerate(seeds, 1):
        if len(seeds) > 1:
            print(f"\n=== seed {seed} ({i}/{len(seeds)}) ===")
        trainer = build_trainer(seed, shared)
        try:
            histories.append(trainer.train())
        finally:
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return histories


def shared_teacher(shared: Dict[str, Any] | None, teacher_name: str, load: Callable[[str], Any]):
    """The frozen, eval-mode, no-cache teacher; loaded once per ``shared`` dict.

    ``load(name)`` returns ``(model, tokenizer)``. With ``shared`` None (a trainer
    built outside run_seeds) the teacher is simply loaded.
    """
    if shared is not None and shared.get(SHARED_TEACHER) is not None:
        print(f"Teacher {teacher_name} reused from the previous seed.")
        return shared[SHARED_TEACHER]
    teacher, _ = load(teacher_name)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    if hasattr(teacher.config, "use_cache"):
        teacher.config.use_cache = False
    if shared is not None:
        shared[SHARED_TEACHER] = teacher
    return teacher


def run_baselines(
    shared: Dict[str, Any] | None,
    history: Dict[str, Any],
    *,
    student: Callable[[], float],
    teacher: Callable[[], float] | None,
    extra: Callable[[], Dict[str, float]],
    teacher_extra: Callable[[], Dict[str, float]] | None = None,
) -> None:
    """Record the step-0 accuracies in ``history``, evaluating them once per ``shared`` dict.

    The untrained student and the teacher are the same weights for every seed,
    and eval is greedy, so the baselines are computed for the first seed and
    copied into every later seed's history. ``teacher`` is None for trainers
    without one (SFT). ``teacher_extra`` scores the teacher on every
    ``--eval-datasets`` entry, recorded as ``teacher_baseline_<dataset>``: the
    ceiling each OOD curve should be read against.
    """
    cached = shared.get(SHARED_BASELINES) if shared is not None else None
    if cached is None:
        print("Evaluating baseline...")
        cached = {
            "student": student(),
            "teacher": teacher() if teacher is not None else None,
            "extra": extra(),
            "teacher_extra": teacher_extra() if teacher_extra is not None else {},
        }
        if shared is not None:
            shared[SHARED_BASELINES] = cached
    else:
        print("Baselines reused from the first seed (untrained student and teacher do not depend on it).")
    history["student_baseline"] = cached["student"]
    history["accuracy"].append(cached["student"])
    history["accuracy_step"].append(0)
    print(f"  Student baseline accuracy: {cached['student']:.4f}")
    if cached["teacher"] is not None:
        history["teacher_baseline"] = cached["teacher"]
        print(f"  Teacher baseline accuracy: {cached['teacher']:.4f}")
    for ds, acc in cached["extra"].items():
        history[f"accuracy_{ds}"].append(acc)
        print(f"  Student baseline [{ds}]: {acc:.4f}")
    for ds, acc in cached.get("teacher_extra", {}).items():
        history[f"teacher_baseline_{ds}"] = acc
        print(f"  Teacher baseline [{ds}]: {acc:.4f}")


def history_seed(history: Dict[str, Any]) -> int:
    """The seed a history was produced with; files predating --seeds ran at DEFAULT_SEED."""
    config = history.get("config")
    if isinstance(config, dict) and config.get("seed") is not None:
        return int(config["seed"])
    return DEFAULT_SEED


def load_history_runs(path: str) -> Dict[str, Dict[str, Any]]:
    """Every run in a history JSON, keyed by seed (as a string).

    A file written before multi-seed support holds one run at top level and no
    ``runs`` key; it comes back as that single run under its seed.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    runs = data.get(HISTORY_RUNS_KEY)
    if isinstance(runs, dict) and runs:
        return dict(runs)
    top = {k: v for k, v in data.items() if k != HISTORY_RUNS_KEY}
    return {str(history_seed(top)): top} if top else {}


def save_history(history: Dict[str, Any], save_dir: str) -> None:
    """Write the run's history to ``history_path(save_dir)``, keeping other seeds.

    Layout: the top level is this run's history, unchanged, so every reader that
    expects one flat history keeps working and sees whichever seed wrote last.
    ``runs`` maps every seed that has written to this folder, this one included,
    to its history; re-running a seed replaces its entry. Runs sharing a file
    should differ only in their seed; a config mismatch is reported, never fatal.
    The file is staged and swapped in so a crash mid-write cannot drop the other
    seeds.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = history_path(save_dir)
    this = {k: v for k, v in dict(history).items() if k != HISTORY_RUNS_KEY}
    seed = str(history_seed(this))
    runs: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(path):
        try:
            runs = load_history_runs(path)
        except (OSError, ValueError) as e:
            print(f"WARN: could not read {path} ({e}); any other seeds it held are not kept")
    this_config = this.get("config")
    if isinstance(this_config, dict):
        for other_seed, other in runs.items():
            other_config = other.get("config")
            if other_seed == seed or not isinstance(other_config, dict):
                continue
            diffs = sorted(
                k for k in set(this_config) | set(other_config)
                if k not in _SAME_FILE_CONFIG_IGNORED and this_config.get(k) != other_config.get(k)
            )
            if diffs:
                print(
                    f"WARN: {path} already holds seed {other_seed} with a different config "
                    f"({', '.join(diffs)}); merging anyway, but seeds sharing a file should "
                    f"differ only in their seed"
                )
    runs[seed] = this
    out = dict(this)
    out[HISTORY_RUNS_KEY] = runs
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _ema(series: Sequence[float], alpha: float) -> List[float]:
    """Exponential moving average seeded with the first value, so the start is not dragged to zero."""
    out: List[float] = []
    m: float | None = None
    for v in series:
        m = float(v) if m is None else alpha * float(v) + (1.0 - alpha) * m
        out.append(m)
    return out


def _seed_sort_key(seed: str):
    return (0, int(seed)) if seed.lstrip("-").isdigit() else (1, seed)


def save_curves(
    history: Dict[str, List],
    save_dir: str,
    losses: Sequence[Tuple[str, str]] = (("step_ce_loss", "CE Loss"),),
    ema_alpha: float = 0.1,
) -> None:
    """Write ``<save_dir>/training_curves.png``: one panel per ``(history key, title)``
    in ``losses`` against the train step, then an accuracy panel.

    Every seed recorded in the folder's history JSON (see save_history) is drawn:
    losses as a faint raw trace under an EMA with smoothing ``ema_alpha``, one
    colour per seed; accuracies one colour per dataset, with the per-seed traces
    faint and their mean over seeds bold once there is more than one seed, and
    the teacher's accuracy on each dataset as a dashed line in that colour.
    ``history`` is the run that just finished and is used alone if the JSON is
    unreadable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    runs: Dict[str, Dict[str, Any]] = {}
    path = history_path(save_dir)
    if os.path.exists(path):
        try:
            runs = load_history_runs(path)
        except (OSError, ValueError):
            runs = {}
    runs[str(history_seed(history))] = dict(history)
    seeds = sorted(runs, key=_seed_sort_key)
    multi = len(seeds) > 1
    if not any(runs[s].get("train_step") for s in seeds):
        return
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    n_panels = len(losses) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))
    for ax, (loss_key, loss_label) in zip(axes, losses):
        for i, seed in enumerate(seeds):
            h = runs[seed]
            series = h.get(loss_key, [])
            steps = h.get("train_step", [])
            if not series or not steps:
                continue
            x = steps[: len(series)]
            series = series[: len(x)]
            c = colors[i % len(colors)]
            ax.plot(x, series, color=c, alpha=0.25, linewidth=0.8)
            ax.plot(
                x, _ema(series, ema_alpha), color=c, linewidth=1.8,
                label=f"seed {seed}" if multi else None,
            )
        ax.set_title(f"{loss_label} (EMA {ema_alpha:g}, raw faint)")
        ax.set_xlabel("train step")
        ax.grid(True, alpha=0.3)
        if multi:
            ax.legend(fontsize=7)
    acc_ax = axes[-1]
    acc_keys = ["accuracy"] + sorted({
        k for h in runs.values() for k in h
        if k.startswith("accuracy_") and k != "accuracy_step"
    })
    use_legend = len(acc_keys) > 1
    teacher_drawn = False
    for j, key in enumerate(acc_keys):
        name = "main" if key == "accuracy" else key[len("accuracy_"):]
        c = colors[j % len(colors)]
        # The teacher's accuracy on the same dataset, dashed in the dataset's
        # colour: the ceiling each curve should be read against.
        t_key = "teacher_baseline" if key == "accuracy" else f"teacher_baseline_{name}"
        t_val = next((runs[s][t_key] for s in seeds if isinstance(runs[s].get(t_key), (int, float))), None)
        if t_val is not None:
            acc_ax.axhline(
                t_val, color=c, linestyle="--", linewidth=1.0, alpha=0.7,
                label=None if teacher_drawn else "teacher (dashed)",
            )
            teacher_drawn = True
        per_seed: List[Tuple[List, List]] = []
        for seed in seeds:
            h = runs[seed]
            series = h.get(key, [])
            if not series:
                continue
            acc_steps = h.get("accuracy_step", list(range(1, len(series) + 1)))
            x = acc_steps[: len(series)]
            series = series[: len(x)]
            per_seed.append((x, series))
            acc_ax.plot(
                x, series, color=c, marker="o", markersize=2,
                alpha=0.35 if multi else 1.0, linewidth=0.8 if multi else 1.5,
                label=None if multi else (name if use_legend else None),
            )
        if multi and per_seed:
            n = min(len(s) for _, s in per_seed)
            mean = [sum(s[i] for _, s in per_seed) / len(per_seed) for i in range(n)]
            acc_ax.plot(
                per_seed[0][0][:n], mean, color=c, linewidth=2.0,
                label=f"{name} (mean of {len(per_seed)} seeds)",
            )
    acc_ax.set_title("Accuracy")
    acc_ax.set_xlabel("train step")
    acc_ax.set_ylim(0, 1)
    acc_ax.grid(True, alpha=0.3)
    if use_legend or multi or teacher_drawn:
        acc_ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


def add_kd_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("kd_args")
    group.add_argument("--teacher", type=str, required=True)
    group.add_argument("--temperature", type=float, default=1.0,
                       help="KD softening temperature, also used for the graph term's logit targets. "
                            "1.0: at 2.0 the teacher's tail mass on non-answer tokens after '=' (a '?') "
                            "rivals the answer's, and the mode-covering forward KL made the student "
                            "emit '?' within five steps (2026-09-17).")
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
    group.add_argument("--lr", type=float, default=5e-7, help="Peak learning rate; see --warmup-steps.")
    group.add_argument("--warmup-steps", type=int, default=10, dest="warmup_steps",
                       help="Linear warmup from 0 to --lr over this many steps, then cosine decay to "
                            "--lr-floor x --lr at --steps. 0 disables the warmup.")
    group.add_argument("--lr-floor", type=float, default=0.3, dest="lr_floor",
                       help="Fraction of --lr the cosine decay reaches at the last step; 1 keeps the "
                            "rate constant after warmup.")
    group.add_argument("--save-dir", type=str, default="results/sft")
    group.add_argument("--eval-every-n-steps", type=int, default=1, dest="eval_every_n_steps")
    group.add_argument("--save-every-n-steps", type=int, default=0, dest="save_every_n_steps",
                    help="N > 0: overwrite <save-dir>/checkpoint (weights + optimizer state) every N "
                         "train steps and write <save-dir>/final_checkpoint at the end. -1: no "
                         "periodic checkpoint, final_checkpoint only. 0 (the default): no weights "
                         "at all. The history JSON and curves are always written.")
    group.add_argument("--resume", action="store_true",
                    help="Continue from <save-dir>/checkpoint: loads and verifies the student weights, "
                         "optimizer state, step and history saved by --save-every-n-steps.")
    group.add_argument("--grad-accum-steps", type=int, default=1, dest="grad_accum_steps")
    group.add_argument("--max-eval-tokens", type=int, default=None, dest="max_eval_tokens",
                    help="Greedy-decoded tokens per eval prompt. Default: 8 for the local arithmetic "
                         "datasets (4-digit answers, worst case one digit per token), 256 for gsm8k/svamp. "
                         "1 truncates any digit-by-digit answer and caps 33_add, which needs two tokens.")
    group.add_argument("--eval-batch-size", type=int, default=256, dest="eval_batch_size",
                    help="Prompts per generate() call during eval. Eval is no-grad greedy decoding of "
                         "short prompts, so it can run far larger batches than training; it used to "
                         "share --batch-size. Default 256.")
    group.add_argument("--test-limit", type=int, default=None, dest="test_limit")
    group.add_argument("--seeds", "--seed", type=int, nargs="+", default=[DEFAULT_SEED], dest="seeds",
                    help="One or more seeds for python, numpy and torch: data order, any sampling, "
                         "and (graph KD) the scramble permutations. The trainer runs once per seed, "
                         "in order, reloading the models in between. Every seed writes into the same "
                         "--save-dir: the history JSON keeps each under \"runs\", the top level being "
                         "the last run, and training_curves.png overlays them. The saved model and "
                         "periodic checkpoint are per folder, so the last seed's overwrite the "
                         "earlier ones. A seed whose run in that folder already reached --steps is "
                         "skipped (see --redo-seeds). --resume needs exactly one seed. Default 42.")
    group.add_argument("--redo-seeds", action="store_true", dest="redo_seeds",
                    help="Train every listed seed even if the folder's history already holds a "
                         "finished run for it, replacing that entry.")

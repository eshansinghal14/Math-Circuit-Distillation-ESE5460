"""Standard Knowledge Distillation (pure KL baseline).

KL is applied at the causal position that predicts the first answer token.
Paper target: 63.6% accuracy on 2-digit addition.

New runs:  results go to  <save-dir>/standard-kl/<run-name|datetime>/
Resume:    add ``--resume``; weights load from ``<run>/student_model/``. Give
           ``--checkpoint-run`` (datetime folder) or omit it to use the most recently
           modified run under ``standard-kl/``. Optimizer state is in ``training_state.pt``.

Examples (from src/)::

  # Dataset prefix (see ``utils``): 2d_add -> datasets/2d_add_train_80.json and _test_20.json
  # --dataset is required.

  # Fresh run, auto-dated folder
  python standard_distillation.py \\
    --dataset 2d_add \\
    --save-dir "/content/drive/MyDrive/Math Circuit Distillation (ESE 5460)/results"

  # Fresh run with a custom folder name
  python standard_distillation.py \\
    --dataset 2d_add \\
    --save-dir "/content/drive/MyDrive/Math Circuit Distillation (ESE 5460)/results" \\
    --run-name my_run

  # Resume a specific run (loads student_model/ + training_state.pt when present)
  python standard_distillation.py \\
    --dataset 2d_add \\
    --save-dir "/content/drive/MyDrive/Math Circuit Distillation (ESE 5460)/results" \\
    --resume \\
    --checkpoint-run "2026-04-07_22-15-56" \\
    --epochs 45

  # Resume from most recently modified run
  python standard_distillation.py \\
    --dataset 2d_add \\
    --save-dir "/content/drive/MyDrive/Math Circuit Distillation (ESE 5460)/results" \\
    --resume \\
    --epochs 20
"""

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime
from functools import partial
from typing import Optional

import subprocess

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from utils import EVAL_MAX_NEW_TOKENS, json_to_prompt_answer_dict, load_model, resolve_train_test_paths

# HuggingFace student weights + tokenizer (overwritten each save)
STUDENT_MODEL_DIR = "student_model"


def _rm_dir(path: str) -> None:
    """Delete a directory tree. Uses shell rm -rf (more reliable on Drive FUSE for large dirs)
    with shutil.rmtree as fallback. Silently ignores missing paths."""
    try:
        result = subprocess.run(["rm", "-rf", path], capture_output=True)
        if result.returncode != 0:
            raise OSError(result.stderr.decode().strip())
    except Exception:
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass


class AddDataset(Dataset):
    """Matches the notebook: tokenize at construction, right-pad to same length."""

    def __init__(self, path: str, tokenizer):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        data = json_to_prompt_answer_dict(raw)

        self.samples = []
        for prompt, answer in data.items():
            answer = str(answer)
            prompt_ids = tokenizer(
                prompt, return_tensors="pt", padding=False, add_special_tokens=False,
            )["input_ids"].squeeze(0)
            answer_ids = tokenizer(
                answer + tokenizer.eos_token,
                return_tensors="pt", padding=False, add_special_tokens=False,
            )["input_ids"].squeeze(0)
            input_ids = torch.cat([prompt_ids, answer_ids])
            self.samples.append({
                "input_ids": input_ids,
                "prompt_len": len(prompt_ids),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(examples, pad_id: int):
    """Right-pad variable-length token sequences and build kl_mask."""
    max_len = max(ex["input_ids"].size(0) for ex in examples)
    B = len(examples)

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long)
    kl_mask = torch.zeros(B, max_len, dtype=torch.float32)

    for i, ex in enumerate(examples):
        ids = ex["input_ids"]
        L = ids.size(0)
        input_ids[i, :L] = ids
        attention_mask[i, :L] = 1
        # KL at the single logit position that predicts the first answer token
        pos = ex["prompt_len"] - 1
        if 0 <= pos < L:
            kl_mask[i, pos] = 1.0

    return {"input_ids": input_ids, "attention_mask": attention_mask, "kl_mask": kl_mask}


def _extract_int_after_equals(text: str) -> Optional[int]:
    m = re.search(r"=\s*(\d+)", text)
    return int(m.group(1)) if m else None


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    test_path: str,
    batch_size: int = 32,
    max_new_tokens: Optional[int] = None,
    debug_decode: int = 0,
    debug_tag: Optional[str] = None,
) -> float:
    if max_new_tokens is None:
        max_new_tokens = EVAL_MAX_NEW_TOKENS
    with open(test_path, "r", encoding="utf-8") as f:
        data = json_to_prompt_answer_dict(json.load(f))
    prompts = list(data.keys())
    answers = list(data.values())

    model.eval()
    correct = total = 0
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for i in range(0, len(prompts), batch_size):
        batch_p = prompts[i : i + batch_size]
        batch_a = answers[i : i + batch_size]
        inputs = tokenizer(batch_p, return_tensors="pt", padding=True).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if debug_decode > 0 and i == 0:
            n = min(debug_decode, len(decoded))
            tag = f" [{debug_tag}]" if debug_tag else ""
            print(f"--- decode debug{tag} (first batch, max_new_tokens={max_new_tokens}) ---")
            for j in range(n):
                text = decoded[j]
                gold = batch_a[j]
                pred = _extract_int_after_equals(text)
                print(f"  gold={gold}  pred={pred}  decoded={text!r}")
            print("---")

        for text, gold in zip(decoded, batch_a):
            pred = _extract_int_after_equals(text)
            if pred == gold:
                correct += 1
            total += 1

    tokenizer.padding_side = original_side
    return correct / max(total, 1)


def save_student_checkpoint(model, tokenizer, run_dir: str) -> None:
    """Write student + tokenizer under ``run_dir/student_model/`` (replaces any previous save)."""
    path = os.path.join(run_dir, STUDENT_MODEL_DIR)
    _rm_dir(path)
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def _training_state_path(save_dir: str) -> str:
    return os.path.join(save_dir, "training_state.pt")


def save_training_state(
    save_dir: str,
    optimizer: torch.optim.Optimizer,
    next_epoch: int,
    best_acc: float,
) -> None:
    """next_epoch = number of epochs already completed (resume will start training at this index)."""
    path = _training_state_path(save_dir)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "next_epoch": next_epoch,
            "best_acc": best_acc,
        },
        path,
    )


def load_training_state(path: str, optimizer: torch.optim.Optimizer, map_location) -> tuple[int, float]:
    # Optimizer dicts are not loadable with weights_only=True (PyTorch 2.6+).
    try:
        chk = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        chk = torch.load(path, map_location=map_location)
    optimizer.load_state_dict(chk["optimizer"])
    return int(chk["next_epoch"]), float(chk["best_acc"])


def _save_curves(history: dict, run_dir: str) -> None:
    """Save KL-loss and accuracy training curves as PNGs into run_dir."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping curve plots.")
        return

    epochs = history.get("epoch", [])
    if not epochs:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history.get("kl_loss", []), marker="o", markersize=3, linewidth=1.5)
    ax1.set_title("KL Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("KL Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history.get("accuracy", []), marker="o", markersize=3,
             linewidth=1.5, color="tab:orange")
    ax2.set_title("Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Standard KL Distillation", fontsize=13)
    fig.tight_layout()

    out = os.path.join(run_dir, "training_curves.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved training curves → {out}")


def _most_recent_run(parent_dir: str) -> Optional[str]:
    """Return the most recently modified subdirectory of parent_dir."""
    if not os.path.isdir(parent_dir):
        return None
    try:
        entries = os.listdir(parent_dir)
    except OSError:
        return None
    best_mtime, best_path = None, None
    for name in entries:
        full = os.path.join(parent_dir, name)
        if not os.path.isdir(full):
            continue
        try:
            mtime = os.path.getmtime(full)
            if best_mtime is None or mtime > best_mtime:
                best_mtime, best_path = mtime, full
        except OSError:
            pass
    return best_path


def _resolve_run_dir(args) -> tuple[str, Optional[str]]:
    """Return ``(run_dir, student_source)``.

    New run: ``run_dir = <save_dir>/standard-kl/<run-name|datetime>/``, ``student_source`` is None.

    Resume: ``--resume`` loads weights from ``<run_dir>/student_model/`` (constant path).
    ``--checkpoint-run`` is the datetime folder (``standard-kl/`` prepended if omitted);
    if omitted, the most recently modified folder under ``<save_dir>/standard-kl/`` is used.
    """
    if not args.resume:
        folder = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(args.save_dir, "standard-kl", folder)
        return run_dir, None

    skl_dir = os.path.join(args.save_dir, "standard-kl")

    if args.checkpoint_run:
        cr = args.checkpoint_run
        if not cr.startswith("standard-kl/"):
            cr = f"standard-kl/{cr}"
        run_dir = os.path.join(args.save_dir, cr)
    else:
        run_dir = _most_recent_run(skl_dir)
        if run_dir is None:
            raise SystemExit(
                f"No run folders found in {skl_dir}.\n"
                "Provide --checkpoint-run <datetime> explicitly."
            )
        print(f"Auto-detected most recent run: {run_dir}")

    student_source = os.path.join(run_dir, STUDENT_MODEL_DIR)
    if not os.path.isdir(student_source):
        raise SystemExit(
            f"Resume expected saved weights at {student_source}. "
            "Train a run first (or place a save_pretrained tree there)."
        )
    print(f"Loading student from {student_source}")
    return run_dir, student_source


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Resolve run directory and checkpoint source ----
    run_dir, student_source = _resolve_run_dir(args)
    os.makedirs(run_dir, exist_ok=True)
    is_resume = student_source is not None

    print(f"Run dir: {run_dir}")
    if not student_source:
        print(f"Loading student from HF: {args.student_model!r}")
    student, tokenizer = load_model(student_source or args.student_model)
    student = student.to("cpu").float().to(device)
    tokenizer.padding_side = "left"

    print("Loading teacher...")
    teacher, _ = load_model(args.teacher_model)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    dataset = AddDataset(args.train_path, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_id=tokenizer.eos_token_id),
    )

    optimizer = AdamW(student.parameters(), lr=args.lr)
    history: dict = {}
    hist_path = os.path.join(run_dir, "training_history.json")
    state_path = _training_state_path(run_dir)
    start_epoch = 0
    best_acc = 0.0

    if is_resume:
        # Always load existing history first (regardless of whether training_state.pt exists)
        if os.path.isfile(hist_path):
            with open(hist_path, "r") as f:
                history = json.load(f)
            if not isinstance(history, dict):
                history = {}
        for key in ("epoch", "kl_loss", "accuracy"):
            if key not in history:
                history[key] = []

        if os.path.isfile(state_path):
            start_epoch, best_acc = load_training_state(state_path, optimizer, device)
            print(
                f"Resumed optimizer state (starting at global epoch {start_epoch + 1}, "
                f"best_acc={best_acc:.4f})"
            )
        else:
            start_epoch = len(history.get("epoch", [])) if history else 0
            best_acc = max(history["accuracy"]) if history.get("accuracy") else 0.0
            print(
                f"No training_state.pt in {run_dir} — warm-starting from student_model "
                f"with new optimizer (continuing from epoch {start_epoch + 1})."
            )

    if not history:
        history = defaultdict(list)

    if start_epoch == 0:
        print("Evaluating baselines...")
        student_base = evaluate(
            student,
            tokenizer,
            args.test_path,
            max_new_tokens=args.eval_max_new_tokens,
            debug_decode=args.debug_decode,
            debug_tag="student baseline",
        )
        teacher_base = evaluate(
            teacher,
            tokenizer,
            args.test_path,
            max_new_tokens=args.eval_max_new_tokens,
            debug_decode=args.debug_decode,
            debug_tag="teacher baseline",
        )
        print(f"  Student baseline accuracy: {student_base:.4f}")
        print(f"  Teacher baseline accuracy: {teacher_base:.4f}")
    else:
        print("Skipping baseline eval (resumed run).")
        student_base = float(history.get("student_baseline", 0.0))
        teacher_base = float(history.get("teacher_baseline", 0.0))

    end_epoch = start_epoch + args.epochs
    print("=" * 60)
    print("Standard KL Distillation")
    print(f"  Run dir:   {run_dir}")
    if is_resume:
        print(f"  Epochs:    +{args.epochs} this run  (epochs {start_epoch + 1}..{end_epoch}, {end_epoch} total)")
        print(f"  Resumed from: {student_source!r}")
    else:
        print(f"  Epochs:    {args.epochs}")
    print(f"  Batch:     {args.batch_size}")
    print(f"  LR:        {args.lr}")
    print(f"  temp:      {args.temperature}")
    print(f"  eval max_new_tokens: {args.eval_max_new_tokens}")
    print("=" * 60)

    T = args.temperature

    for epoch in range(start_epoch, end_epoch):
        student.train()
        epoch_loss, n_steps = 0.0, 0

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            kl_mask = batch["kl_mask"].to(device)

            student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits

            with torch.no_grad():
                teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits

            # KL only at the single answer-logit position (see collate_fn).
            log_p_s = F.log_softmax(student_logits / T, dim=-1)
            p_t = F.softmax(teacher_logits / T, dim=-1)
            kl_per_token = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1)
            loss = (kl_per_token * kl_mask).sum() / kl_mask.sum().clamp_min(1.0) * (T**2)

            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_steps += 1

            if step % 50 == 0:
                print(f"  step {step:04d} | KL {loss.item():.4f}")

        avg_loss = epoch_loss / max(n_steps, 1)
        acc = evaluate(
            student,
            tokenizer,
            args.test_path,
            max_new_tokens=args.eval_max_new_tokens,
            debug_decode=args.debug_decode,
            debug_tag=f"epoch {epoch + 1} student",
        )

        history["epoch"].append(epoch + 1)
        history["kl_loss"].append(avg_loss)
        history["accuracy"].append(acc)

        print(f"Epoch {epoch+1}/{end_epoch}: KL={avg_loss:.4f} Acc={acc:.4f}")

        hist_out = dict(history)
        with open(hist_path, "w") as f:
            json.dump(hist_out, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        if acc > best_acc:
            best_acc = acc
            if args.save_best:
                save_student_checkpoint(student, tokenizer, run_dir)
                print(f"  Saved {STUDENT_MODEL_DIR}/ (new best accuracy)")

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_student_checkpoint(student, tokenizer, run_dir)
            print(f"  Saved {STUDENT_MODEL_DIR}/ (epoch {epoch + 1})")

        save_training_state(run_dir, optimizer, epoch + 1, best_acc)

    hist_out = dict(history)
    hist_out["student_baseline"] = student_base
    hist_out["teacher_baseline"] = teacher_base

    # Write history BEFORE the slow checkpoint save so a Ctrl+C can't lose it
    with open(hist_path, "w") as f:
        json.dump(hist_out, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    _save_curves(hist_out, run_dir)

    save_student_checkpoint(student, tokenizer, run_dir)
    print(f"  Saved {STUDENT_MODEL_DIR}/ (final)")

    print(f"\nDone. Best accuracy: {best_acc:.4f}")
    print(f"Results saved to: {run_dir}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_base = os.path.join(script_dir, "..", "results")

    parser = argparse.ArgumentParser(
        description="Standard KL distillation (baseline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--student-model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--teacher-model", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument(
        "--dataset",
        default=None,
        metavar="PREFIX",
        help="e.g. 2d_add -> datasets/<PREFIX>_train_80.json and _test_20.json (required)",
    )
    parser.add_argument(
        "--datasets-dir",
        default=None,
        help="Directory containing *_train_80.json (default: repo datasets/)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=EVAL_MAX_NEW_TOKENS,
        help=f"Greedy eval: max new tokens after prompt (default {EVAL_MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help=f"Overwrite {STUDENT_MODEL_DIR}/ every N epochs (0 = disable periodic save)",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        help=f"Also overwrite {STUDENT_MODEL_DIR}/ when eval accuracy improves (off by default)",
    )
    # ---- Save / resume args ----
    parser.add_argument(
        "--save-dir",
        default=default_base,
        metavar="DIR",
        help=(
            "Base results directory. New runs create standard-kl/<run-name|datetime>/ inside it. "
            f"Default: {default_base}"
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        metavar="NAME",
        help="Custom folder name for a new run (default: YYYY-MM-DD_HH-MM-SS timestamp).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an existing run; load weights from <run>/student_model/. "
            "Use --checkpoint-run for the datetime folder, or omit to use the most "
            "recent run under standard-kl/."
        ),
    )
    parser.add_argument(
        "--checkpoint-run",
        default=None,
        metavar="DATETIME",
        help=(
            "Datetime folder of the run to resume (e.g. '2026-04-07_22-15-56'). "
            "Loads weights from that run's student_model/. "
            "'standard-kl/' is prepended automatically. Only used with --resume."
        ),
    )
    parser.add_argument(
        "--debug-decode",
        type=int,
        default=0,
        metavar="N",
        help="Print N decode examples per evaluate call for debugging (0=off)",
    )
    args = parser.parse_args()
    try:
        train_p, test_p, _ = resolve_train_test_paths(
            dataset=args.dataset,
            datasets_dir=args.datasets_dir,
        )
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e
    args.train_path = train_p
    args.test_path = test_p
    train(args)


if __name__ == "__main__":
    main()

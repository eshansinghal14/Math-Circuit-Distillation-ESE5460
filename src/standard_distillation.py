"""Standard Knowledge Distillation (pure KL baseline).

Assumes **one token per answer** (typical for 3-digit sums in Llama BPE). KL is only at
the causal step whose logits predict that answer token (index ``answer_start - 1``).

Eval defaults to ``--eval-max-new-tokens 1``.

No internal alignment -- this is the baseline that other experiments compare against.
Paper target: 63.6% accuracy on 2-digit addition.

**Why the student can approach the teacher:** both ``load_model`` calls load **pretrained**
HuggingFace weights, not random init. A 1B student that already speaks English/math will
often land close to an 8B teacher after sequence-level KD on this task; that is not a bug
in the loss unless you intended a different setup (e.g. random init).

Usage (from src/):
  python standard_distillation.py --save-dir /path/to/drive/results/standard-kl
"""

import argparse
import json
import os
import re
from collections import defaultdict
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from utils import EVAL_MAX_NEW_TOKENS, load_model


class AddDataset(Dataset):
    def __init__(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        self.prompts = list(data.keys())
        self.answers = [str(data[p]) for p in self.prompts]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "answer": self.answers[idx]}


def _answer_token_start(prompt: str, full_text: str, tokenizer) -> int:
    """Index in ``full`` token sequence of the first answer token (after shared prompt prefix)."""
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    lcp = 0
    while lcp < len(p_ids) and lcp < len(f_ids) and p_ids[lcp] == f_ids[lcp]:
        lcp += 1
    return lcp


def _kl_mask_single_answer_logit(
    attention_mask: torch.Tensor,
    answer_starts: torch.Tensor,
) -> torch.Tensor:
    """One-hot mask: KL only at logits predicting the first answer token. Shape [B, L]."""
    B, L = attention_mask.shape
    kl = torch.zeros(B, L, dtype=torch.float32)
    for b in range(B):
        real_len = int(attention_mask[b].sum().item())
        a0 = int(answer_starts[b].item())
        if 0 < a0 < real_len:
            i = a0 - 1
            if i < L:
                kl[b, i] = 1.0
    return kl


def collate_fn(examples, tokenizer):
    prompts = [ex["prompt"] for ex in examples]
    answers = [ex["answer"] for ex in examples]
    full_texts = [p + a for p, a in zip(prompts, answers)]

    # Right-pad so token indices align with per-string tokenization (prompt + answer).
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    enc = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True)
    tokenizer.padding_side = old_side

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    answer_starts = torch.tensor(
        [_answer_token_start(p, f, tokenizer) for p, f in zip(prompts, full_texts)],
        dtype=torch.long,
    )
    kl_mask = _kl_mask_single_answer_logit(attention_mask, answer_starts)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "kl_mask": kl_mask,
    }


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
    with open(test_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        prompts = list(data.keys())
        answers = [int(data[p]) for p in prompts]
    else:
        prompts = [d["q_str"] for d in data]
        answers = [int(d["a_str"]) for d in data]

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


def save_checkpoint(model, tokenizer, save_dir: str, tag: str):
    path = os.path.join(save_dir, f"student_{tag}")
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading student...")
    student, tokenizer = load_model(args.student_model)
    student = student.to("cpu").float().to(device)
    tokenizer.padding_side = "left"

    print("Loading teacher...")
    teacher, _ = load_model(args.teacher_model)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    dataset = AddDataset(args.train_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    optimizer = AdamW(student.parameters(), lr=args.lr)
    history = defaultdict(list)
    best_acc = 0.0

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

    print("=" * 60)
    print("Standard KL Distillation")
    print(f"  Epochs:    {args.epochs}")
    print(f"  Batch:     {args.batch_size}")
    print(f"  LR:        {args.lr}")
    print(f"  temp:      {args.temperature}")
    print(f"  eval max_new_tokens: {args.eval_max_new_tokens}")
    print(f"  Save dir:  {args.save_dir}")
    print("=" * 60)

    T = args.temperature

    for epoch in range(args.epochs):
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

        print(f"Epoch {epoch+1}/{args.epochs}: KL={avg_loss:.4f} Acc={acc:.4f}")

        with open(os.path.join(args.save_dir, "training_history.json"), "w") as f:
            json.dump(dict(history), f, indent=2)

        if acc > best_acc:
            best_acc = acc
            save_checkpoint(student, tokenizer, args.save_dir, "best")

        if args.checkpoint_every > 0 and (epoch + 1) % args.checkpoint_every == 0:
            save_checkpoint(student, tokenizer, args.save_dir, f"epoch_{epoch+1}")

    history["student_baseline"] = student_base
    history["teacher_baseline"] = teacher_base

    save_checkpoint(student, tokenizer, args.save_dir, "final")
    with open(os.path.join(args.save_dir, "training_history.json"), "w") as f:
        json.dump(dict(history), f, indent=2)
    print(f"\nDone. Best accuracy: {best_acc:.4f}")
    print(f"Results saved to: {args.save_dir}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Standard KL distillation (baseline)")
    parser.add_argument("--student-model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--teacher-model", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--train-path", default=os.path.join(script_dir, "..", "datasets", "2d_add_train_80.json"))
    parser.add_argument("--test-path", default=os.path.join(script_dir, "..", "datasets", "2d_add_test_20.json"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=1,
        help="Greedy eval: max new tokens after prompt (default 1 for single-token answers)",
    )
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument(
        "--debug-decode",
        type=int,
        default=0,
        metavar="N",
        help="Print N examples (gold, pred, decoded repr) from the first batch of each evaluate (0=off)",
    )
    parser.add_argument("--save-dir", default=os.path.join(script_dir, "..", "results", "standard-kl"))
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

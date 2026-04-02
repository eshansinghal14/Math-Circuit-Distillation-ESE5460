"""Standard Knowledge Distillation (pure KL baseline).

Loss = CE(student, labels) + alpha * KL(student_logits/T, teacher_logits/T)

No internal alignment -- this is the baseline that other experiments compare against.

Usage (from src/):
  python standard_distillation.py --save-dir /path/to/drive/results/standard-kl
"""

import argparse
import json
import os
import re
from collections import defaultdict
from functools import partial
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from utils import load_model


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


def collate_fn(examples, tokenizer):
    prompts = [ex["prompt"] for ex in examples]
    answers = [ex["answer"] for ex in examples]
    full_texts = [p + a for p, a in zip(prompts, answers)]

    enc = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    labels = input_ids.clone()
    prompt_enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    prompt_lens = (prompt_enc["input_ids"] != tokenizer.pad_token_id).sum(dim=1)
    for i, length in enumerate(prompt_lens.tolist()):
        labels[i, :length] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _extract_int_after_equals(text: str) -> Optional[int]:
    m = re.search(r"=\s*(\d+)", text)
    return int(m.group(1)) if m else None


@torch.no_grad()
def evaluate(model, tokenizer, test_path: str, batch_size: int = 32) -> float:
    with open(test_path, "r") as f:
        data = json.load(f)

    # Handle both formats: {"prompt": answer} dict or [{"q_str":..,"a_str":..}] list
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
        outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
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
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    optimizer = AdamW(student.parameters(), lr=args.lr)
    history = defaultdict(list)
    best_acc = 0.0

    print("=" * 60)
    print("Standard KL Distillation")
    print(f"  Epochs:    {args.epochs}")
    print(f"  Batch:     {args.batch_size}")
    print(f"  LR:        {args.lr}")
    print(f"  alpha_kl:  {args.alpha_kl}")
    print(f"  temp:      {args.temperature}")
    print(f"  Save dir:  {args.save_dir}")
    print("=" * 60)

    for epoch in range(args.epochs):
        student.train()
        epoch_ce, epoch_kl, epoch_total, n_steps = 0.0, 0.0, 0.0, 0

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if (labels != -100).sum().item() == 0:
                continue

            student_out = student(
                input_ids=input_ids, attention_mask=attention_mask,
            )
            student_logits = student_out.logits

            with torch.no_grad():
                teacher_out = teacher(
                    input_ids=input_ids, attention_mask=attention_mask,
                )
                teacher_logits = teacher_out.logits

            # Shift for next-token prediction (standard causal LM)
            shift_student = student_logits[..., :-1, :].contiguous()
            shift_teacher = teacher_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten to (B*T, V) and (B*T,)
            B, T_len, V = shift_student.shape
            flat_student = shift_student.view(-1, V)
            flat_teacher = shift_teacher.view(-1, V)
            flat_labels = shift_labels.view(-1)

            # CE loss on answer tokens only
            ce_loss = F.cross_entropy(flat_student.float(), flat_labels, ignore_index=-100)

            # KL loss on answer tokens only (following torchtune ForwardKLLoss)
            mask = (flat_labels != -100)
            num_valid = mask.sum()

            if num_valid > 0:
                teacher_prob = F.softmax(flat_teacher[mask] / args.temperature, dim=-1, dtype=torch.float32)
                student_logprob = F.log_softmax(flat_student[mask] / args.temperature, dim=-1, dtype=torch.float32)
                teacher_logprob = F.log_softmax(flat_teacher[mask] / args.temperature, dim=-1, dtype=torch.float32)
                # True KL(teacher || student) = sum(p * (log(p) - log(q)))
                kl_loss = (teacher_prob * (teacher_logprob - student_logprob)).sum() / num_valid
                kl_loss = kl_loss * (args.temperature ** 2)
            else:
                kl_loss = torch.tensor(0.0, device=device)

            loss = ce_loss + args.alpha_kl * kl_loss

            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            optimizer.step()

            epoch_ce += ce_loss.item()
            epoch_kl += kl_loss.item()
            epoch_total += loss.item()
            n_steps += 1

            if step % 50 == 0:
                print(f"  step {step:04d} | CE {ce_loss.item():.4f} | "
                      f"KL {kl_loss.item():.4f} | Total {loss.item():.4f}")

        avg = lambda v: v / max(n_steps, 1)
        acc = evaluate(student, tokenizer, args.test_path)

        history["epoch"].append(epoch + 1)
        history["ce_loss"].append(avg(epoch_ce))
        history["kl_loss"].append(avg(epoch_kl))
        history["total_loss"].append(avg(epoch_total))
        history["accuracy"].append(acc)

        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"CE={avg(epoch_ce):.4f} KL={avg(epoch_kl):.4f} Acc={acc:.4f}")

        # Incremental history save
        with open(os.path.join(args.save_dir, "training_history.json"), "w") as f:
            json.dump(dict(history), f, indent=2)

        if acc > best_acc:
            best_acc = acc
            save_checkpoint(student, tokenizer, args.save_dir, "best")

        if args.checkpoint_every > 0 and (epoch + 1) % args.checkpoint_every == 0:
            save_checkpoint(student, tokenizer, args.save_dir, f"epoch_{epoch+1}")

    save_checkpoint(student, tokenizer, args.save_dir, "final")
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha-kl", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--save-dir", default=os.path.join(script_dir, "..", "results", "standard-kl"))
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

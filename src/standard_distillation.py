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

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import load_model, test_model, eval_model


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
    results_path = os.path.join(os.path.dirname(test_path), "_std_distill_eval.json")
    test_model(model, tokenizer, test_path, results_path,
               batch_size=batch_size, max_new_tokens=2, log=False)
    return eval_model(results_path)


def save_checkpoint(model, tokenizer, save_dir: str, tag: str):
    path = os.path.join(save_dir, f"student_{tag}")
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    dtype = torch.float16 if device == "cuda" else torch.float32

    print("Loading student...")
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model, torch_dtype=dtype,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Loading teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=dtype,
    ).to(device)
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
                input_ids=input_ids, attention_mask=attention_mask, labels=labels,
            )
            ce_loss = student_out.loss
            student_logits = student_out.logits

            with torch.no_grad():
                teacher_out = teacher(
                    input_ids=input_ids, attention_mask=attention_mask,
                )
                teacher_logits = teacher_out.logits

            T = args.temperature
            kl_loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=-1),
                F.softmax(teacher_logits / T, dim=-1),
                reduction="batchmean",
            ) * (T * T)

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
    parser.add_argument("--test-path", default=os.path.join(script_dir, "..", "datasets", "2d_add_test_20_formatted.json"))
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

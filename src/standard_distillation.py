"""Standard Knowledge Distillation (pure KL baseline).

Loss = T^2 * KL(softmax(student/T) || softmax(teacher/T))

No internal alignment -- this is the baseline that other experiments compare against.
Paper target: 63.6% accuracy on 2-digit addition.

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

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    }


def _extract_int_after_equals(text: str) -> Optional[int]:
    m = re.search(r"=\s*(\d+)", text)
    return int(m.group(1)) if m else None


@torch.no_grad()
def evaluate(model, tokenizer, test_path: str, batch_size: int = 32) -> float:
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

    print("Evaluating baselines...")
    student_base = evaluate(student, tokenizer, args.test_path)
    teacher_base = evaluate(teacher, tokenizer, args.test_path)
    print(f"  Student baseline accuracy: {student_base:.4f}")
    print(f"  Teacher baseline accuracy: {teacher_base:.4f}")

    print("=" * 60)
    print("Standard KL Distillation")
    print(f"  Epochs:    {args.epochs}")
    print(f"  Batch:     {args.batch_size}")
    print(f"  LR:        {args.lr}")
    print(f"  temp:      {args.temperature}")
    print(f"  Save dir:  {args.save_dir}")
    print("=" * 60)

    T = args.temperature

    for epoch in range(args.epochs):
        student.train()
        epoch_loss, n_steps = 0.0, 0

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits

            with torch.no_grad():
                teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits

            loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=-1),
                F.softmax(teacher_logits / T, dim=-1),
                reduction="batchmean",
            ) * (T ** 2)

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
        acc = evaluate(student, tokenizer, args.test_path)

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
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--save-dir", default=os.path.join(script_dir, "..", "results", "standard-kl"))
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

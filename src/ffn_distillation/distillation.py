"""FFN layer-level distillation with CKA alignment.

Loss = CE(student, labels) + lambda * mean(CKA_loss per paired MLP layer)

Hooks the full MLP output of each paired layer in both student and teacher,
then computes CKA between those outputs.  This is the "full MLP layer pairing"
experiment -- a middle ground between pure KL (no internal alignment) and
neuron-cluster CKA (fine-grained internal alignment).

Usage (from src/):
  python -m ffn_distillation.distillation --save-dir /path/to/results
"""

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import AutoModelForCausalLM, AutoTokenizer

from cka_loss import CKALoss
from ffn_distillation.layer_pairing import LayerPairInfo


@dataclass
class FFNDistillationConfig:
    teacher_model: str = "meta-llama/Meta-Llama-3-8B"
    student_model: str = "meta-llama/Llama-3.2-1B"

    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-4
    grad_clip: float = 1.0

    lambda_cka: float = 0.1
    eval_every: int = 1
    checkpoint_every: int = 5
    save_dir: str = "results/ffn-distillation"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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


class MLPActivationCache:
    """Hooks the full MLP output for specified layers."""

    def __init__(self):
        self.activations: Dict[int, torch.Tensor] = {}
        self.hooks: List = []

    def _make_hook(self, layer_idx: int, detach: bool):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output
            self.activations[layer_idx] = act.detach() if detach else act
        return hook

    def register_hooks(self, model, layer_indices: List[int], detach: bool = False):
        self.clear()
        for idx in layer_indices:
            h = model.model.layers[idx].mlp.register_forward_hook(
                self._make_hook(idx, detach=detach)
            )
            self.hooks.append(h)

    def clear(self):
        self.activations.clear()
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


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


class FFNDistillationTrainer:
    """Training loop for FFN layer-level CKA distillation."""

    def __init__(
        self,
        config: FFNDistillationConfig,
        layer_pairs: List[LayerPairInfo],
        train_path: str,
        test_path: str,
    ):
        self.config = config
        self.layer_pairs = layer_pairs
        self.test_path = test_path
        self.device = config.device

        self.student_layer_indices = sorted(set(p.student_layer for p in layer_pairs))
        self.teacher_layer_indices = sorted(set(p.teacher_layer for p in layer_pairs))

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.student_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Models
        dtype = torch.float16 if config.device == "cuda" else torch.float32
        print(f"Loading student: {config.student_model}")
        self.student = AutoModelForCausalLM.from_pretrained(
            config.student_model, torch_dtype=dtype,
        ).to(config.device)
        print(f"Loading teacher: {config.teacher_model}")
        self.teacher = AutoModelForCausalLM.from_pretrained(
            config.teacher_model, torch_dtype=dtype,
        ).to(config.device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Caches
        self.student_cache = MLPActivationCache()
        self.teacher_cache = MLPActivationCache()

        # Loss
        self.cka_loss_fn = CKALoss(efficient=True)

        # Optimizer
        self.optimizer = AdamW(self.student.parameters(), lr=config.learning_rate)

        # Data
        dataset = AddDataset(train_path)
        self.loader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True,
            collate_fn=partial(collate_fn, tokenizer=self.tokenizer),
        )

        self.history: Dict[str, List] = defaultdict(list)

    def _forward_and_loss(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        if (labels != -100).sum().item() == 0:
            return None, {}

        self.teacher_cache.register_hooks(
            self.teacher, self.teacher_layer_indices, detach=True,
        )
        self.student_cache.register_hooks(
            self.student, self.student_layer_indices, detach=False,
        )

        try:
            with torch.no_grad():
                self.teacher(input_ids=input_ids, attention_mask=attention_mask)

            student_out = self.student(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels,
            )
            ce_loss = student_out.loss

            # CKA over each paired layer
            cka_losses = []
            cka_scores = {}
            for pair in self.layer_pairs:
                s_act = self.student_cache.activations.get(pair.student_layer)
                t_act = self.teacher_cache.activations.get(pair.teacher_layer)

                if s_act is None or t_act is None:
                    continue

                loss_i, cka_i = self.cka_loss_fn(s_act.float(), t_act.float())
                cka_losses.append(loss_i)
                cka_scores[(pair.student_layer, pair.teacher_layer)] = cka_i.item()

            if cka_losses:
                cka_loss = torch.stack(cka_losses).mean()
            else:
                cka_loss = torch.tensor(0.0, device=self.device)

            total = ce_loss + self.config.lambda_cka * cka_loss

            metrics = {
                "ce_loss": ce_loss.item(),
                "cka_loss": cka_loss.item(),
                "total_loss": total.item(),
                "mean_cka": (
                    sum(cka_scores.values()) / len(cka_scores)
                    if cka_scores else 0.0
                ),
            }
            return total, metrics

        finally:
            self.student_cache.clear()
            self.teacher_cache.clear()

    def train(self) -> Dict[str, List]:
        cfg = self.config
        os.makedirs(cfg.save_dir, exist_ok=True)

        print("=" * 60)
        print("FFN Layer-Level CKA Distillation")
        print(f"  Epochs:      {cfg.epochs}")
        print(f"  Batch:       {cfg.batch_size}")
        print(f"  LR:          {cfg.learning_rate}")
        print(f"  lambda_cka:  {cfg.lambda_cka}")
        print(f"  Layer pairs: {len(self.layer_pairs)}")
        print(f"  Save dir:    {cfg.save_dir}")
        print("=" * 60)

        best_acc = 0.0

        for epoch in range(cfg.epochs):
            self.student.train()
            agg = defaultdict(float)
            n = 0

            for step, batch in enumerate(self.loader):
                loss, metrics = self._forward_and_loss(batch)
                if loss is None:
                    continue
                if torch.isnan(loss):
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), cfg.grad_clip)
                self.optimizer.step()

                for k, v in metrics.items():
                    agg[k] += v
                n += 1

                if step % 10 == 0:
                    print(f"  step {step:04d} | CE {metrics['ce_loss']:.4f} | "
                          f"CKA_loss {metrics['cka_loss']:.4f} | "
                          f"mean_CKA {metrics['mean_cka']:.4f}")

            avg = {k: v / max(n, 1) for k, v in agg.items()}

            acc = 0.0
            if (epoch + 1) % cfg.eval_every == 0:
                acc = evaluate(self.student, self.tokenizer, self.test_path)
                avg["accuracy"] = acc

            self.history["epoch"].append(epoch + 1)
            for k, v in avg.items():
                self.history[k].append(v)

            # Incremental history save
            with open(os.path.join(cfg.save_dir, "training_history.json"), "w") as f:
                json.dump(dict(self.history), f, indent=2)

            print(f"Epoch {epoch+1}/{cfg.epochs}: "
                  f"CE={avg.get('ce_loss', 0):.4f} CKA_loss={avg.get('cka_loss', 0):.4f} "
                  f"mean_CKA={avg.get('mean_cka', 0):.4f} Acc={acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                save_checkpoint(self.student, self.tokenizer, cfg.save_dir, "best")

            if cfg.checkpoint_every > 0 and (epoch + 1) % cfg.checkpoint_every == 0:
                save_checkpoint(self.student, self.tokenizer, cfg.save_dir, f"epoch_{epoch+1}")

        save_checkpoint(self.student, self.tokenizer, cfg.save_dir, "final")
        print(f"\nDone. Best accuracy: {best_acc:.4f}")
        print(f"Results saved to: {cfg.save_dir}")

        return dict(self.history)

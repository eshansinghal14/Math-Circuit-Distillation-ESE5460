"""KL-only knowledge distillation — student learns from teacher's output distribution."""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional

import torch

from new_utils import DataLoader, eval_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.utils import add_standard_args, kl_loss, make_optimizer, save_checkpoint, save_curves, save_history

from utils import (
    DIR_ROOT,
    PromptAnswerDataset,
    collate_fn,
    load_data,
    load_model,
    seed_all,
)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_GRAD_CLIP = 1.0
_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StandardKDConfig:
    model: str
    teacher: str
    dataset: str
    steps: int = 15
    batch_size: int = 32
    learning_rate: float = 1e-6
    temperature: float = 2.0
    kl_token_chunk_size: int = 64
    max_eval_tokens: int = 256
    save_dir: str = "results/standard_kd"
    eval_every_n_steps: int = 1
    grad_accum_steps: int = 1


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────


class StandardKDTrainer:
    def __init__(
        self,
        config: StandardKDConfig,
        train_data: Dict[str, Any],
        test_data: Dict[str, Any],
    ) -> None:
        self.config = config
        seed_all(_SEED)

        self.model, self.tokenizer = load_model(config.model)
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        self.teacher, _ = load_model(config.teacher)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        if hasattr(self.teacher.config, "use_cache"):
            self.teacher.config.use_cache = False

        dataset = PromptAnswerDataset(config.dataset, train_data, self.tokenizer)
        self.test_dataset = PromptAnswerDataset(config.dataset, test_data, self.tokenizer)
        self.loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=partial(collate_fn, pad_id=self.tokenizer.eos_token_id),
        )

        self.optimizer = make_optimizer(self.model, config.learning_rate)

        self.history: Dict[str, List] = defaultdict(list)
        self._train_step = 0

    def _eval(self) -> float:
        cfg = self.config
        return eval_model(
            self.model, self.tokenizer, self.test_dataset,
            cfg.dataset, cfg.batch_size, cfg.max_eval_tokens,
        )

    def _eval_teacher(self) -> float:
        cfg = self.config
        return eval_model(
            self.teacher, self.tokenizer, self.test_dataset,
            cfg.dataset, cfg.batch_size, cfg.max_eval_tokens,
        )

    def train_epoch(self, *, max_steps: Optional[int] = None) -> Dict[str, float]:
        self.model.train()
        cfg = self.config
        grad_accum = cfg.grad_accum_steps
        total_loss = 0.0
        n_steps = 0
        accum_loss = 0.0
        micro_step = 0

        self.optimizer.zero_grad()
        for batch in self.loader:
            if max_steps is not None and n_steps >= max_steps:
                break
            input_ids = batch["input_ids"].to(_DEVICE)
            attention_mask = batch["attention_mask"].to(_DEVICE)

            student_logits = self.model(input_ids, attention_mask=attention_mask).logits

            with torch.no_grad():
                teacher_logits = self.teacher(input_ids, attention_mask=attention_mask).logits

            loss = kl_loss(
                student_logits, teacher_logits, attention_mask,
                cfg.temperature, cfg.kl_token_chunk_size,
            ) / grad_accum

            if not torch.isfinite(loss):
                micro_step += 1
                if micro_step % grad_accum == 0:
                    self.optimizer.zero_grad()
                continue

            loss.backward()
            accum_loss += float(loss.item())
            micro_step += 1

            if micro_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), _GRAD_CLIP)
                self.optimizer.step()
                self.optimizer.zero_grad()

                self._train_step += 1
                self.history["train_step"].append(self._train_step)
                self.history["step_kl_loss"].append(accum_loss)
                total_loss += accum_loss
                accum_loss = 0.0
                n_steps += 1

        return {"kl_loss": total_loss / max(n_steps, 1)}

    def train(self) -> Dict[str, List]:
        cfg = self.config
        os.makedirs(cfg.save_dir, exist_ok=True)

        print("Evaluating baseline...")
        baseline_acc = self._eval()
        self.history["student_baseline"] = baseline_acc
        self.history["accuracy"].append(baseline_acc)
        self.history["accuracy_step"].append(0)
        print(f"  Student baseline accuracy: {baseline_acc:.4f}")
        teacher_baseline_acc = self._eval_teacher()
        self.history["teacher_baseline"] = teacher_baseline_acc
        print(f"  Teacher baseline accuracy: {teacher_baseline_acc:.4f}")

        sample = self.loader.dataset[0]
        print("─" * 60)
        print("Sample [0]:")
        print(f"  prompt:  {str(sample['prompt'])[:120]!r}")
        print(f"  answer:  {str(sample['answer'])[:80]!r}")
        print(f"  tokens:  {sample['input_ids'].shape[0]} total, {sample['prompt_len']} prompt")
        print("─" * 60)

        print(
            f"KD | student={cfg.model} | teacher={cfg.teacher} | dataset={cfg.dataset}"
            f" | steps={cfg.steps} | lr={cfg.learning_rate} | temp={cfg.temperature}"
        )

        while self._train_step < cfg.steps:
            remaining = cfg.steps - self._train_step
            metrics = self.train_epoch(max_steps=min(cfg.eval_every_n_steps, remaining))
            if not metrics:
                break
            self.history["kl_loss"].append(metrics["kl_loss"])

            acc = self._eval()
            self.history["accuracy"].append(acc)
            self.history["accuracy_step"].append(self._train_step)
            print(f"  Step {self._train_step}/{cfg.steps} | KL={metrics['kl_loss']:.4f} | Acc={acc:.4f}")

        save_history(self.history, cfg.save_dir)
        save_curves(self.history, cfg.save_dir, loss_key="step_kl_loss", loss_label="KL Loss")
        save_checkpoint(self.model, self.tokenizer, cfg.save_dir)
        print(f"Results saved to: {cfg.save_dir}")
        return dict(self.history)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KL-only distillation on GSM8K / SVAMP / local datasets.")
    add_standard_args(parser)
    group = parser.add_argument_group("kd_args")
    group.add_argument("--teacher", type=str, required=True)
    group.add_argument("--temperature", type=float, default=2.0)
    group.add_argument("--kl-token-chunk-size", type=int, default=64, dest="kl_token_chunk_size")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_data, test_data = load_data(args.dataset, test_limit=args.test_limit)
    print(f"Train: {len(train_data)} | Test: {len(test_data)}")
    trainer = StandardKDTrainer(
        StandardKDConfig(
            model=args.model,
            teacher=args.teacher,
            dataset=args.dataset,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            temperature=args.temperature,
            kl_token_chunk_size=args.kl_token_chunk_size,
            save_dir=os.path.join(DIR_ROOT, args.save_dir),
            eval_every_n_steps=args.eval_every_n_steps,
            grad_accum_steps=args.grad_accum_steps,
            max_eval_tokens=args.max_eval_tokens,
        ),
        train_data,
        test_data,
    )
    trainer.train()


if __name__ == "__main__":
    main()

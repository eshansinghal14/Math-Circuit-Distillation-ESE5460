"""Supervised fine-tuning (SFT) on GSM8K / SVAMP / local arithmetic datasets."""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from new_utils import DataLoader, eval_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.utils import make_optimizer, save_checkpoint, save_curves, save_history

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
# Loss
# ─────────────────────────────────────────────────────────────────────────────


def sft_ce_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :]
    labels = input_ids[:, 1:]
    mask = response_mask[:, 1:]
    valid = mask.reshape(-1).bool().nonzero(as_tuple=False).squeeze(-1)
    if valid.numel() == 0:
        return logits.sum() * 0.0
    logits_flat = shifted_logits.reshape(-1, shifted_logits.shape[-1])
    labels_flat = labels.reshape(-1)
    s = logits_flat.index_select(0, valid.to(logits_flat.device)).float()
    l = labels_flat.index_select(0, valid.to(labels_flat.device)).to(s.device)
    return F.cross_entropy(s, l)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SFTConfig:
    model: str
    dataset: str
    steps: int = 15
    batch_size: int = 32
    learning_rate: float = 1e-6
    max_eval_tokens: int = 256
    save_dir: str = "results/sft"
    eval_every_n_steps: int = 1


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────


class SFTTrainer:
    def __init__(
        self,
        config: SFTConfig,
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

    def train_epoch(self, *, max_steps: Optional[int] = None) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_steps = 0

        for batch in self.loader:
            if max_steps is not None and n_steps >= max_steps:
                break
            input_ids = batch["input_ids"].to(_DEVICE)
            attention_mask = batch["attention_mask"].to(_DEVICE)
            response_mask = batch["response_mask"].to(_DEVICE)

            logits = self.model(input_ids, attention_mask=attention_mask).logits
            loss = sft_ce_loss(logits, input_ids, response_mask)

            if not torch.isfinite(loss):
                continue

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), _GRAD_CLIP)
            self.optimizer.step()

            loss_val = float(loss.item())
            self._train_step += 1
            self.history["train_step"].append(self._train_step)
            self.history["step_ce_loss"].append(loss_val)
            total_loss += loss_val
            n_steps += 1
            print(f"  Step {self._train_step}/{self.config.steps} | CE={loss_val:.4f}")

        return {"ce_loss": total_loss / max(n_steps, 1)}

    def train(self) -> Dict[str, List]:
        cfg = self.config
        os.makedirs(cfg.save_dir, exist_ok=True)

        print("Evaluating baseline...")
        baseline_acc = self._eval()
        self.history["student_baseline"] = baseline_acc
        print(f"  Baseline accuracy: {baseline_acc:.4f}")

        sample = self.loader.dataset[0]
        print("─" * 60)
        print("Sample [0]:")
        print(f"  prompt:  {str(sample['prompt'])[:120]!r}")
        print(f"  answer:  {str(sample['answer'])[:80]!r}")
        print(f"  tokens:  {sample['input_ids'].shape[0]} total, {sample['prompt_len']} prompt")
        print("─" * 60)

        print(f"SFT | model={cfg.model} | dataset={cfg.dataset} | steps={cfg.steps} | lr={cfg.learning_rate}")

        while self._train_step < cfg.steps:
            remaining = cfg.steps - self._train_step
            metrics = self.train_epoch(max_steps=min(cfg.eval_every_n_steps, remaining))
            if not metrics:
                break
            self.history["ce_loss"].append(metrics["ce_loss"])

            acc = self._eval()
            self.history["accuracy"].append(acc)
            self.history["accuracy_step"].append(self._train_step)
            print(f"  Step {self._train_step}/{cfg.steps} | Acc={acc:.4f}")

        save_history(self.history, cfg.save_dir)
        save_curves(self.history, cfg.save_dir)
        save_checkpoint(self.model, self.tokenizer, cfg.save_dir)
        print(f"Results saved to: {cfg.save_dir}")
        return dict(self.history)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SFT on GSM8K / SVAMP / local datasets.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/sft",
    )
    parser.add_argument("--eval-every-n-steps", type=int, default=1, dest="eval_every_n_steps")
    parser.add_argument("--max-eval-tokens", type=int, default=256, dest="max_eval_tokens")
    parser.add_argument("--test-limit", type=int, default=None, dest="test_limit")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_data, test_data = load_data(args.dataset, test_limit=args.test_limit)
    print(f"Train: {len(train_data)} | Test: {len(test_data)}")
    trainer = SFTTrainer(
        SFTConfig(
            model=args.model,
            dataset=args.dataset,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            save_dir=os.path.join(DIR_ROOT, args.save_dir),
            eval_every_n_steps=args.eval_every_n_steps,
            max_eval_tokens=args.max_eval_tokens,
        ),
        train_data,
        test_data,
    )
    trainer.train()


if __name__ == "__main__":
    main()

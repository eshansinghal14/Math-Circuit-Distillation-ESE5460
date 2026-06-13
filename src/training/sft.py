"""Supervised fine-tuning (SFT) on GSM8K / SVAMP / local arithmetic datasets."""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from utils import (
    DataLoader,
    DIR_ROOT,
    PromptAnswerDataset,
    collate_fn,
    eval_model,
    load_data,
    load_model,
    seed_all,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.utils import add_standard_args, make_optimizer, save_checkpoint, save_curves, save_history

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
    grad_accum_steps: int = 1
    eval_datasets: List[str] = field(default_factory=list)


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

        self.extra_test_datasets: Dict[str, PromptAnswerDataset] = {}
        for ds in config.eval_datasets:
            _, ds_test_data = load_data(ds)
            self.extra_test_datasets[ds] = PromptAnswerDataset(ds, ds_test_data, self.tokenizer)

        self.optimizer = make_optimizer(self.model, config.learning_rate)

        self.history: Dict[str, List] = defaultdict(list)
        self._train_step = 0

    def _eval_on(self, dataset_name: str, test_dataset: PromptAnswerDataset) -> float:
        cfg = self.config
        return eval_model(self.model, self.tokenizer, test_dataset, dataset_name, cfg.batch_size, cfg.max_eval_tokens)

    def _eval(self) -> float:
        return self._eval_on(self.config.dataset, self.test_dataset)

    def _eval_all_extra(self) -> Dict[str, float]:
        return {ds: self._eval_on(ds, td) for ds, td in self.extra_test_datasets.items()}

    def train_epoch(self, *, max_steps: Optional[int] = None) -> Dict[str, float]:
        self.model.train()
        grad_accum = self.config.grad_accum_steps
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
            response_mask = batch["response_mask"].to(_DEVICE)

            logits = self.model(input_ids, attention_mask=attention_mask).logits
            loss = sft_ce_loss(logits, input_ids, response_mask) / grad_accum

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
                self.history["step_ce_loss"].append(accum_loss)
                total_loss += accum_loss
                print(f"  step {self._train_step} | CE={accum_loss:.4f}")
                accum_loss = 0.0
                n_steps += 1

        return {"ce_loss": total_loss / max(n_steps, 1)}

    def train(self) -> Dict[str, List]:
        cfg = self.config
        os.makedirs(cfg.save_dir, exist_ok=True)

        print("Evaluating baseline...")
        baseline_acc = self._eval()
        self.history["student_baseline"] = baseline_acc
        self.history["accuracy"].append(baseline_acc)
        self.history["accuracy_step"].append(0)
        print(f"  Baseline accuracy: {baseline_acc:.4f}")
        for ds, acc in self._eval_all_extra().items():
            self.history[f"accuracy_{ds}"].append(acc)
            print(f"  Baseline [{ds}]: {acc:.4f}")

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
            extra_accs = self._eval_all_extra()
            for ds, ds_acc in extra_accs.items():
                self.history[f"accuracy_{ds}"].append(ds_acc)
            extra_str = "".join(f" | {ds}={a:.4f}" for ds, a in extra_accs.items())
            print(f"  [eval] step {self._train_step}/{cfg.steps} | Acc={acc:.4f}{extra_str}")

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
    add_standard_args(parser)
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
            grad_accum_steps=args.grad_accum_steps,
            max_eval_tokens=args.max_eval_tokens,
            eval_datasets=args.eval_datasets,
        ),
        train_data,
        test_data,
    )
    trainer.train()


if __name__ == "__main__":
    main()

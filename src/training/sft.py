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
    load_split,
    seed_all,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.utils import (
    DEFAULT_SEED,
    ParamChangeCanary,
    add_standard_args,
    describe_run_setup,
    load_student,
    log_first_step_canary,
    make_optimizer,
    maybe_save_periodic_checkpoint,
    record_resumed_config,
    first_answer_token_accuracy,
    apply_lr_schedule,
    resume_checkpoint_dir,
    run_baselines,
    run_seeds,
    resume_training_state,
    run_config_record,
    save_checkpoint,
    refresh_curves,
    save_curves,
    save_history,
    student_autocast,
)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_GRAD_CLIP = 1.0
_SEED = DEFAULT_SEED


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
    learning_rate: float = 5e-7
    warmup_steps: int = 10
    lr_floor: float = 0.3  # see training.utils.scheduled_lr
    max_eval_tokens: Optional[int] = None  # None -> utils.default_eval_tokens(dataset)
    eval_batch_size: int = 256
    save_dir: str = "results/sft"
    eval_every_n_steps: int = 1
    save_every_n_steps: int = 0
    grad_accum_steps: int = 1
    eval_datasets: List[str] = field(default_factory=list)
    test_limit: Optional[int] = None
    dtype: Optional[str] = None
    resume: bool = False
    seed: int = _SEED


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────


class SFTTrainer:
    def __init__(
        self,
        config: SFTConfig,
        train_data: Dict[str, Any],
        test_data: Dict[str, Any],
        shared: Dict[str, Any] | None = None,
    ) -> None:
        """``shared`` is run_seeds' cross-seed dict (teacher, baselines); None loads everything."""
        self.config = config
        self.shared = shared
        seed_all(config.seed)

        # fp32 master weights with a bf16 autocast forward; see training/utils.py.
        # --resume loads the weights the periodic checkpoint saved instead.
        student_src = resume_checkpoint_dir(config.save_dir) if config.resume else config.model
        self.model, self.tokenizer = load_student(student_src, getattr(config, "dtype", None))

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
            _, ds_test_data = load_data(ds, test_limit=config.test_limit)
            self.extra_test_datasets[ds] = PromptAnswerDataset(ds, ds_test_data, self.tokenizer)

        self.optimizer = make_optimizer(self.model, config.learning_rate)

        self.history: Dict[str, List] = defaultdict(list)
        self._train_step = 0
        self._last_save_step = 0
        if config.resume:
            step, history, _ = resume_training_state(self.model, self.optimizer, config.save_dir)
            self.history = defaultdict(list, history)
            self._train_step = step
            self._last_save_step = step

    def _autocast(self):
        return student_autocast()

    def _eval_on(self, dataset_name: str, test_dataset: PromptAnswerDataset) -> float:
        cfg = self.config
        with self._autocast():
            return eval_model(
                self.model, self.tokenizer, test_dataset, dataset_name,
                cfg.eval_batch_size, cfg.max_eval_tokens,
            )

    def _eval(self) -> float:
        return self._eval_on(self.config.dataset, self.test_dataset)

    def _eval_all_extra(self) -> Dict[str, float]:
        return {ds: self._eval_on(ds, td) for ds, td in self.extra_test_datasets.items()}

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
            response_mask = batch["response_mask"].to(_DEVICE)

            with self._autocast():
                logits = self.model(input_ids, attention_mask=attention_mask).logits
            self._last_tf_acc = first_answer_token_accuracy(logits.detach(), input_ids, response_mask)
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
                canary = ParamChangeCanary(self.model) if self._train_step == 0 else None
                self._last_lr = apply_lr_schedule(
                    self.optimizer, self._train_step + 1, cfg.steps, cfg.learning_rate,
                    cfg.warmup_steps, cfg.lr_floor)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if canary is not None:
                    log_first_step_canary(canary.report(self.model), self.history)

                self._train_step += 1
                self.history["train_step"].append(self._train_step)
                self.history["step_tf_acc"].append(self._last_tf_acc)
                self.history["step_lr"].append(self._last_lr)
                self.history["step_ce_loss"].append(accum_loss)
                total_loss += accum_loss
                print(f"  step {self._train_step} | CE={accum_loss:.4f}")
                self._last_save_step = maybe_save_periodic_checkpoint(
                    self.model, self.tokenizer, self.config.save_dir,
                    self._train_step, self.config.save_every_n_steps,
                    self._last_save_step, self.history,
                    optimizer=self.optimizer,
                )
                accum_loss = 0.0
                n_steps += 1

        return {"ce_loss": total_loss / max(n_steps, 1)}

    def train(self) -> Dict[str, List]:
        cfg = self.config
        os.makedirs(cfg.save_dir, exist_ok=True)

        if cfg.resume:
            # The config record and baselines were taken by the original run and
            # came back with the history; re-running the baseline eval would
            # score the checkpoint, not the untrained student.
            if self.history.get("config"):
                print(describe_run_setup(self.history["config"]))
            record_resumed_config(self.history, cfg, self.model, self.optimizer)
            self.history["resumed_at_step"].append(self._train_step)
            print(f"Resuming at step {self._train_step}/{cfg.steps}; skipping baseline eval.")
        else:
            self.history["config"] = run_config_record(cfg, self.model, self.optimizer)
            print(describe_run_setup(self.history["config"]))

            run_baselines(self.shared, self.history, student=self._eval, teacher=None, extra=self._eval_all_extra)

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
            refresh_curves(self.history, cfg.save_dir, step=self._train_step)

        save_history(self.history, cfg.save_dir)
        save_curves(self.history, cfg.save_dir)
        if cfg.save_every_n_steps != 0:
            save_checkpoint(self.model, self.tokenizer, cfg.save_dir)
        else:
            print("--save-every-n-steps is 0; not writing the trained model")
        print(f"Results saved to: {cfg.save_dir}")
        return dict(self.history)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SFT on GSM8K / SVAMP / local datasets.")
    add_standard_args(parser)
    parser.add_argument(
        "--use-sft-split", action="store_true", dest="use_sft_split",
        help="Fine-tune on datasets/<name>/sft.json instead of train.json. That split is written "
             "by generate_context_dataset.py --sft and is disjoint from both train and test, so a "
             "format-fixing SFT pass leaves every distillation run fresh prompts: the student is "
             "not fitted twice on the same data, and a fine-tuned teacher's soft targets on the "
             "distillation set are not memorised.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_data, test_data = load_data(args.dataset, test_limit=args.test_limit)
    if args.use_sft_split:
        # Fine-tune the answer format on a slice no distillation run ever sees, so
        # the student is not fitted twice on the same prompts and the teacher's
        # soft targets on the distillation set are not memorised.
        train_data = load_split(args.dataset, "sft")
        shared = set(train_data) & set(load_data(args.dataset)[0])
        if shared:
            raise SystemExit(
                f"{len(shared)} sft.json prompts also appear in train.json; "
                "regenerate the dataset with --sft")
        print(f"--use-sft-split: fine-tuning on {len(train_data)} held-out prompts "
              f"(train.json untouched)")
    print(f"Train: {len(train_data)} | Test: {len(test_data)}")
    save_dir = os.path.join(DIR_ROOT, args.save_dir, args.model.split("/")[-1], args.dataset)

    def build(seed: int, shared: Dict[str, Any]):
        return SFTTrainer(
            SFTConfig(
                model=args.model,
                dataset=args.dataset,
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                warmup_steps=args.warmup_steps,
                lr_floor=args.lr_floor,
                save_dir=save_dir,
                eval_every_n_steps=args.eval_every_n_steps,
                save_every_n_steps=args.save_every_n_steps,
                grad_accum_steps=args.grad_accum_steps,
                max_eval_tokens=args.max_eval_tokens,
                eval_batch_size=args.eval_batch_size,
                eval_datasets=args.eval_datasets,
                test_limit=args.test_limit,
                dtype=args.dtype,
                resume=args.resume,
                seed=seed,
            ),
            train_data,
            test_data,
            shared=shared,
        )

    run_seeds(args.seeds, args.resume, build, save_dir=save_dir, steps=args.steps, redo=args.redo_seeds)


if __name__ == "__main__":
    main()

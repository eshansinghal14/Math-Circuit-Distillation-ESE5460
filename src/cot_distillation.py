"""GSM8K CoT KL distillation."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from graph_loss.utils import DTYPE_CHOICES, resolve_torch_dtype
from utils import (
    LLAMA_1B_MODEL_NAME,
    LLAMA_8B_MODEL_NAME,
    get_default_device,
    json_to_prompt_answer_dict,
    load_model,
    load_prompt_answer_json,
    load_student_model_for_distillation,
    patch_tokenizer_no_special_tokens,
    resolve_train_test_paths,
    rm_dir_tree,
    run_hf_benchmark,
    seed_all,
)


# ─────────────────────────────────────────────────────────────────────────────
# KL loss helpers
# ─────────────────────────────────────────────────────────────────────────────


def kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
    token_chunk_size: int = 64,
) -> torch.Tensor:
    t = temperature
    vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
    student_flat = student_logits[..., :vocab].reshape(-1, vocab)
    teacher_flat = teacher_logits[..., :vocab].reshape(-1, vocab)
    valid = attention_mask.reshape(-1).bool().nonzero(as_tuple=False).squeeze(-1)
    if valid.numel() == 0:
        return student_logits.sum() * 0.0

    total = student_logits.new_zeros((), dtype=torch.float32)
    for idx in valid.split(max(1, token_chunk_size)):
        s_chunk = student_flat.index_select(0, idx.to(student_flat.device)).float()
        t_chunk = teacher_flat.index_select(0, idx.to(teacher_flat.device)).to(
            device=s_chunk.device, dtype=torch.float32,
        )
        log_p_t = F.log_softmax(t_chunk / t, dim=-1)
        log_q_s = F.log_softmax(s_chunk / t, dim=-1)
        total = total + (log_p_t.exp() * (log_p_t - log_q_s)).sum()

    return total / valid.numel() * (t ** 2)


def kl_loss_from_student_hidden(
    student_hidden: torch.Tensor,
    lm_head,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
    token_chunk_size: int = 64,
) -> torch.Tensor:
    t = temperature
    vocab = min(int(lm_head.weight.shape[0]), int(teacher_logits.shape[-1]))
    hidden_flat = student_hidden.reshape(-1, student_hidden.shape[-1])
    teacher_flat = teacher_logits[..., :vocab].reshape(-1, vocab)
    valid = attention_mask.reshape(-1).bool().nonzero(as_tuple=False).squeeze(-1)
    if valid.numel() == 0:
        return student_hidden.sum() * 0.0

    total = student_hidden.new_zeros((), dtype=torch.float32)
    for idx in valid.split(max(1, token_chunk_size)):
        h_chunk = hidden_flat.index_select(0, idx.to(hidden_flat.device))
        s_chunk = lm_head(h_chunk)[..., :vocab].float()
        t_chunk = teacher_flat.index_select(0, idx.to(teacher_flat.device)).to(
            device=s_chunk.device, dtype=torch.float32,
        )
        log_p_t = F.log_softmax(t_chunk / t, dim=-1)
        log_q_s = F.log_softmax(s_chunk / t, dim=-1)
        total = total + (log_p_t.exp() * (log_p_t - log_q_s)).sum()

    return total / valid.numel() * (t ** 2)


class AddDataset(Dataset):
    def __init__(self, data: Union[str, Dict[str, Union[int, str]]], tokenizer):
        if isinstance(data, str):
            with open(data, "r", encoding="utf-8") as f:
                raw = json.load(f)
            data = json_to_prompt_answer_dict(raw)
        self.samples = []
        for prompt, answer in data.items():
            answer_text = str(answer) if isinstance(answer, int) else answer
            prompt_ids = tokenizer(
                prompt, return_tensors="pt", padding=False, add_special_tokens=False,
            )["input_ids"].squeeze(0)
            answer_ids = tokenizer(
                answer_text + tokenizer.eos_token,
                return_tensors="pt", padding=False, add_special_tokens=False,
            )["input_ids"].squeeze(0)
            self.samples.append({
                "input_ids": torch.cat([prompt_ids, answer_ids]),
                "prompt": str(prompt),
                "answer": answer,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def collate_fn(examples, pad_id: int) -> Dict[str, Any]:
    max_len = max(ex["input_ids"].size(0) for ex in examples)
    input_ids = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(len(examples), max_len, dtype=torch.long)
    for row, ex in enumerate(examples):
        ids = ex["input_ids"]
        input_ids[row, : ids.size(0)] = ids
        attention_mask[row, : ids.size(0)] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompts": [str(ex["prompt"]) for ex in examples],
        "answers": [ex["answer"] for ex in examples],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Config & run-dir helpers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CoTDistillationConfig:
    teacher_model: str = LLAMA_8B_MODEL_NAME
    student_model: str = LLAMA_1B_MODEL_NAME
    steps: int = 15
    batch_size: int = 32
    grad_accum_steps: int = 1
    learning_rate: float = 1e-6
    temperature: float = 2.0
    kl_token_chunk_size: int = 64
    grad_clip: float = 1.0
    eval_max_new_tokens: int = 256
    eval_batch_size: int = 32
    save_dir: str = "results/cot_distillation"
    step_log_interval: int = 1
    save_interval: int = 0
    benchmark_eval_limit: Optional[int] = 100
    seed: int = 42
    device: torch.device = field(default_factory=get_default_device)


def find_student_source(path: str) -> Optional[str]:
    if os.path.isdir(path) and (
        os.path.isfile(os.path.join(path, "config.json"))
        or os.path.isfile(os.path.join(path, "model.safetensors"))
    ):
        return path
    return None


def resolve_distillation_run_dir(
    save_dir: str,
    *,
    resume: bool,
    checkpoint_run: Optional[str],
) -> Tuple[str, Optional[str]]:
    save_dir = os.path.abspath(save_dir)
    if not resume:
        return save_dir, None
    if checkpoint_run:
        src = (
            os.path.normpath(checkpoint_run)
            if os.path.isabs(checkpoint_run)
            else os.path.join(save_dir, checkpoint_run)
        )
        student_source = find_student_source(src)
        if student_source is None:
            raise SystemExit(f"No student weights found in {src}.")
        print(f"Loading student from {student_source}")
    else:
        student_source = None
    return save_dir, student_source


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────


class CoTDistillationTrainer:
    def __init__(
        self,
        *,
        config: CoTDistillationConfig,
        train_data: Dict[str, Any],
        test_data: Dict[str, Any],
        tokenizer=None,
        student=None,
        resume_step: Optional[int] = None,
    ) -> None:
        self.config = config
        self.test_data = test_data
        self.device = torch.device(config.device)
        self._resume_step = resume_step
        self._resume = resume_step is not None
        seed_all(config.seed)

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(config.student_model)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer = patch_tokenizer_no_special_tokens(self.tokenizer)
        self.tokenizer.padding_side = "right"

        if student is not None:
            self.student = student
        else:
            self.student, self.tokenizer = load_student_model_for_distillation(
                None, config.student_model, self.device,
            )
        self.student = self.student.to(self.device).float()
        if hasattr(self.student.config, "use_cache"):
            self.student.config.use_cache = False
        if hasattr(self.student, "gradient_checkpointing_enable"):
            self.student.gradient_checkpointing_enable()
            print("Enabled student gradient checkpointing.")
        self.student.train()

        print(f"Loading teacher: {config.teacher_model}")
        self.teacher, _ = load_model(config.teacher_model)
        self.teacher = self.teacher.to(self.device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        bnb = None
        try:
            bnb = importlib.import_module("bitsandbytes")
        except ImportError:
            pass
        if bnb is not None:
            self.optimizer = bnb.optim.PagedAdamW8bit(
                params=self.student.parameters(), lr=config.learning_rate,
            )
            print("Using 8-bit Paged AdamW optimizer.")
        else:
            self.optimizer = AdamW(
                params=self.student.parameters(), lr=config.learning_rate,
            )
            print("Using standard AdamW optimizer.")

        self.loader = DataLoader(
            AddDataset(train_data, self.tokenizer),
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=partial(collate_fn, pad_id=self.tokenizer.eos_token_id),
        )

        self.history: Dict[str, List] = defaultdict(list)
        self._train_step = 0
        self._step_log_eval_accuracy = 0.0

    def _evaluate_model(self, model) -> float:
        cfg = self.config
        _, acc = run_hf_benchmark(
            model,
            self.tokenizer,
            "gsm8k",
            results_fname=None,
            batch_size=cfg.eval_batch_size,
            max_new_tokens=cfg.eval_max_new_tokens,
            limit=cfg.benchmark_eval_limit,
            log=False,
        )
        return float(acc)

    def _teacher_logits_for_batch(
        self, batch: Dict[str, Any], input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.teacher(
                input_ids=input_ids, attention_mask=attention_mask,
            ).logits.detach().cpu()

    def _forward_kl(self, batch: Dict[str, Any]) -> tuple[torch.Tensor, float]:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        teacher_logits = self._teacher_logits_for_batch(batch, input_ids, attention_mask)
        self._clear_cuda_cache()

        student_backbone = getattr(self.student, "model", None)
        lm_head = getattr(self.student, "lm_head", None)
        if student_backbone is not None and lm_head is not None:
            student_out = student_backbone(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
            )
            loss = kl_loss_from_student_hidden(
                student_out.last_hidden_state,
                lm_head,
                teacher_logits,
                attention_mask,
                self.config.temperature,
                token_chunk_size=self.config.kl_token_chunk_size,
            )
        else:
            student_logits = self.student(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
            ).logits
            loss = kl_loss(
                student_logits,
                teacher_logits,
                attention_mask,
                self.config.temperature,
                token_chunk_size=self.config.kl_token_chunk_size,
            )
        return loss, float(loss.item())

    def _clear_cuda_cache(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train_epoch(self, epoch: int, *, max_steps: Optional[int] = None) -> Dict[str, float]:
        self.student.train()
        agg_kl = 0.0
        n_steps = 0
        interval_kl = 0.0
        interval_steps = 0
        accum_steps = max(1, self.config.grad_accum_steps)
        pending_accum = 0
        self.optimizer.zero_grad(set_to_none=True)

        for batch in self.loader:
            loss, kl_val = self._forward_kl(batch)
            if not torch.isfinite(loss).item():
                continue

            (loss / accum_steps).backward()
            pending_accum += 1
            should_step = (
                pending_accum >= accum_steps
                or (max_steps is not None and n_steps + 1 >= max_steps)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.config.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                pending_accum = 0
            self._clear_cuda_cache()

            self._train_step += 1
            self.history["train_step"].append(self._train_step)
            self.history["step_kl_loss"].append(kl_val)
            agg_kl += kl_val
            interval_kl += kl_val
            n_steps += 1
            interval_steps += 1

            self._save_history()

            if self.config.save_interval > 0 and self._train_step % self.config.save_interval == 0:
                self._save_checkpoint_at_step(self._train_step)

            if self._train_step == 1 or self._train_step % max(1, self.config.step_log_interval) == 0:
                acc = self._evaluate_model(self.student)
                self.student.train()
                self._step_log_eval_accuracy = acc
                self.history["accuracy"].append(acc)
                kl_avg = interval_kl / max(interval_steps, 1)
                print(f"  step {self._train_step:04d} | KL {kl_avg:.4f} | Acc {acc:.4f}")
                interval_kl = 0.0
                interval_steps = 0
                self._save_history()
                self._save_curves()

            if max_steps is not None and n_steps >= max_steps:
                break

        if pending_accum > 0:
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.config.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        return {"kl_loss": agg_kl / max(n_steps, 1)}

    def train(self) -> Dict[str, List]:
        cfg = self.config
        hist_path = os.path.join(cfg.save_dir, "training_history.json")
        start_epoch = 0
        if self._resume and os.path.isfile(hist_path):
            with open(hist_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                for key, value in loaded.items():
                    self.history[key] = value
            self._train_step = self._resume_step
            train_steps = self.history.get("train_step", [])
            n_keep = sum(1 for s in train_steps if s <= self._resume_step)
            for key in list(self.history.keys()):
                if isinstance(self.history[key], list) and len(self.history[key]) == len(train_steps):
                    self.history[key] = self.history[key][:n_keep]
            start_epoch = len(self.history.get("epoch", []))
            self._step_log_eval_accuracy = (
                float(self.history["accuracy"][-1]) if self.history.get("accuracy") else 0.0
            )
            opt_path = os.path.join(cfg.save_dir, f"step_{self._resume_step}_checkpoint", "optimizer.pt")
            if os.path.isfile(opt_path):
                self.optimizer.load_state_dict(torch.load(opt_path, map_location=self.device))
                print(f"Restored optimizer state from step {self._resume_step}.")
            print(f"Warm-starting from step {self._train_step + 1}.")

        print("=" * 60)
        print("GSM8K CoT KL Distillation")
        print(f"  Run dir:          {cfg.save_dir}")
        print(f"  Steps:            {self._train_step + 1}..{cfg.steps}")
        print(f"  Batch size:       {cfg.batch_size}")
        print(f"  Grad accum steps: {cfg.grad_accum_steps}")
        print(f"  LR:               {cfg.learning_rate}")
        print(f"  Temperature:      {cfg.temperature}")
        print(f"  Eval every:       {cfg.step_log_interval} steps")
        print("=" * 60)

        epoch = start_epoch
        while self._train_step < cfg.steps:
            remaining = cfg.steps - self._train_step
            epoch_metrics = self.train_epoch(epoch, max_steps=remaining)
            if not epoch_metrics and remaining > 0:
                break
            self.history["epoch"].append(epoch + 1)
            for key, value in epoch_metrics.items():
                self.history[key].append(value)
            print(
                f"Pass {epoch + 1}: KL={epoch_metrics.get('kl_loss', float('nan')):.4f}, "
                f"Acc={self._step_log_eval_accuracy:.4f}, Step={self._train_step}/{cfg.steps}"
            )
            epoch += 1

        self._save_history()
        self._save_curves()
        self._save_checkpoint()
        print(f"Results saved to: {cfg.save_dir}")
        return dict(self.history)

    def _save_checkpoint(self) -> None:
        os.makedirs(self.config.save_dir, exist_ok=True)
        self.student.save_pretrained(self.config.save_dir)
        self.tokenizer.save_pretrained(self.config.save_dir)

    def _save_checkpoint_at_step(self, step: int) -> None:
        path = os.path.join(self.config.save_dir, f"step_{step}_checkpoint")
        rm_dir_tree(path)
        os.makedirs(path, exist_ok=True)
        self.student.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
        print(f"  [checkpoint] Saved step {step} → {path}")

    def _save_history(self) -> None:
        os.makedirs(self.config.save_dir, exist_ok=True)
        path = os.path.join(self.config.save_dir, "training_history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(self.history), f, indent=2)
            f.flush()
            os.fsync(f.fileno())

    def _save_curves(self) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        loss_steps = self.history.get("train_step", [])
        if not loss_steps:
            return
        kl_series = self.history.get("step_kl_loss", [])
        acc_series = self.history.get("accuracy", [])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(loss_steps[: len(kl_series)], kl_series, marker="o", markersize=2)
        axes[0].set_title("KL Loss")
        axes[1].plot(list(range(1, len(acc_series) + 1)), acc_series, marker="o", markersize=3)
        axes[1].set_title("Accuracy")
        axes[1].set_ylim(0, 1)
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        os.makedirs(self.config.save_dir, exist_ok=True)
        fig.savefig(os.path.join(self.config.save_dir, "training_curves.png"), dpi=150)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KL distillation for GSM8K CoT.")
    parser.add_argument("--student-model", type=str, default=LLAMA_1B_MODEL_NAME)
    parser.add_argument("--teacher-model", type=str, default=LLAMA_8B_MODEL_NAME)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kl-token-chunk-size", type=int, default=64)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dtype", choices=DTYPE_CHOICES, default="float32")
    parser.add_argument("--test-limit", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "results", "cot_distillation"),
    )
    parser.add_argument("--step-log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=0)
    parser.add_argument("--resume-step", type=int, default=None, dest="resume_step")
    parser.add_argument("--checkpoint-run", default=None, metavar="PATH")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    train_path, test_path, _ = resolve_train_test_paths(dataset="gsm8k", datasets_dir=None)
    run_dir, student_source = resolve_distillation_run_dir(
        os.path.abspath(args.save_dir),
        resume=args.resume_step is not None,
        checkpoint_run=args.checkpoint_run,
    )
    if args.resume_step is not None and not args.checkpoint_run:
        step_ckpt = os.path.join(run_dir, f"step_{args.resume_step}_checkpoint")
        if not os.path.isdir(step_ckpt):
            raise SystemExit(f"No checkpoint found at {step_ckpt}")
        student_source = find_student_source(step_ckpt)
        if student_source is None:
            raise SystemExit(f"No student weights found in step checkpoint {step_ckpt}")
    if args.resume_step is not None:
        print(f"Resuming from step {args.resume_step} checkpoint: {student_source}")
    os.makedirs(run_dir, exist_ok=True)

    train_data = load_prompt_answer_json(train_path)
    test_data = load_prompt_answer_json(test_path)
    if args.test_limit is not None:
        test_data = dict(list(test_data.items())[: args.test_limit])
    print(f"Train: {len(train_data)} examples | Test: {len(test_data)} examples")

    device = get_default_device()
    student, tokenizer = load_student_model_for_distillation(student_source, args.student_model, device)

    trainer = CoTDistillationTrainer(
        config=CoTDistillationConfig(
            teacher_model=args.teacher_model,
            student_model=args.student_model,
            steps=args.steps,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            learning_rate=args.lr,
            temperature=args.temperature,
            kl_token_chunk_size=args.kl_token_chunk_size,
            grad_clip=args.grad_clip,
            eval_max_new_tokens=256,
            eval_batch_size=args.eval_batch_size,
            save_dir=run_dir,
            step_log_interval=args.step_log_interval,
            save_interval=args.save_interval,
            benchmark_eval_limit=args.test_limit,
            seed=args.seed,
            device=device,
        ),
        train_data=train_data,
        test_data=test_data,
        tokenizer=tokenizer,
        student=student,
        resume_step=args.resume_step,
    )
    history = trainer.train()
    if "accuracy" in history and history["accuracy"]:
        print(f"Best accuracy: {max(history['accuracy']):.4f}")


if __name__ == "__main__":
    main()

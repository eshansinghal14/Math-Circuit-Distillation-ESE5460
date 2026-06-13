"""KL + ANOVA-graph distillation — student learns from teacher logits and circuit structure."""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional

import torch

from new_utils import (
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

from graph_loss.hf_adapter import HFLlamaGraphAdapter
from graph_loss.training import GraphAuxConfig, backward_batch_graph_loss
from training.utils import add_kd_args, add_standard_args, kl_loss, make_optimizer, save_checkpoint, save_curves, save_history

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_GRAD_CLIP = 1.0
_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ActivationsAnovaKDConfig:
    model: str
    teacher: str
    dataset: str
    steps: int = 15
    batch_size: int = 32
    learning_rate: float = 1e-6
    temperature: float = 2.0
    kl_token_chunk_size: int = 64
    max_eval_tokens: int = 256
    save_dir: str = "results/activations_anova_kd"
    eval_every_n_steps: int = 1
    grad_accum_steps: int = 1
    eval_datasets: List[str] = field(default_factory=list)
    # graph loss
    lambda_graph: float = 0.1
    graph_node_labels: List[str] = field(default_factory=list)
    teacher_prop_neurons_per_layer: float = 0.1
    student_prop_neurons_per_layer: float = 0.1
    nodes_per_label: int = 10
    anova_range_radius: int = 0
    graph_loss_type: str = "jsd"
    top_k_logits: float = 0.95
    teacher_graph_batch_size: int = 512
    student_graph_batch_size: int = 1
    n_graph_prompts: Optional[int] = None
    graph_verbose: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────


class ActivationsAnovaKDTrainer:
    def __init__(
        self,
        config: ActivationsAnovaKDConfig,
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

        self.extra_test_datasets: Dict[str, PromptAnswerDataset] = {}
        for ds in config.eval_datasets:
            _, ds_test_data = load_data(ds)
            self.extra_test_datasets[ds] = PromptAnswerDataset(ds, ds_test_data, self.tokenizer)

        self.optimizer = make_optimizer(self.model, config.learning_rate)

        self.student_adapter = HFLlamaGraphAdapter(self.model, self.tokenizer, _DEVICE)
        self.teacher_adapter = HFLlamaGraphAdapter(self.teacher, self.tokenizer, _DEVICE)

        self.graph_config = GraphAuxConfig(
            lambda_graph=config.lambda_graph,
            teacher_prop_neurons_per_layer=config.teacher_prop_neurons_per_layer,
            student_prop_neurons_per_layer=config.student_prop_neurons_per_layer,
            top_k_logits=config.top_k_logits,
            temperature=config.temperature,
            teacher_graph_batch_size=config.teacher_graph_batch_size,
            student_graph_batch_size=config.student_graph_batch_size,
            student_nodes_per_label=config.nodes_per_label,
            teacher_nodes_per_label=config.nodes_per_label,
            student_anova_range_radius=config.anova_range_radius,
            graph_loss_type=config.graph_loss_type,
            student_graph_labels=config.graph_node_labels if config.graph_node_labels else None,
            verbose=config.graph_verbose,
        )

        self.history: Dict[str, List] = defaultdict(list)
        self._train_step = 0

    def _eval_on(self, model, dataset_name: str, test_dataset: PromptAnswerDataset) -> float:
        cfg = self.config
        return eval_model(model, self.tokenizer, test_dataset, dataset_name, cfg.batch_size, cfg.max_eval_tokens)

    def _eval(self) -> float:
        return self._eval_on(self.model, self.config.dataset, self.test_dataset)

    def _eval_teacher(self) -> float:
        return self._eval_on(self.teacher, self.config.dataset, self.test_dataset)

    def _eval_all_extra(self) -> Dict[str, float]:
        return {ds: self._eval_on(self.model, ds, td) for ds, td in self.extra_test_datasets.items()}

    def train_epoch(self, *, max_steps: Optional[int] = None) -> Dict[str, float]:
        self.model.train()
        cfg = self.config
        grad_accum = cfg.grad_accum_steps
        use_graph = bool(cfg.graph_node_labels)

        total_kl = 0.0
        total_graph = 0.0
        n_steps = 0
        accum_kl = 0.0
        accum_graph = 0.0
        micro_step = 0

        self.optimizer.zero_grad()
        for batch in self.loader:
            if max_steps is not None and n_steps >= max_steps:
                break
            input_ids = batch["input_ids"].to(_DEVICE)
            attention_mask = batch["attention_mask"].to(_DEVICE)

            # ── KL loss ───────────────────────────────────────────────────────
            student_logits = self.model(input_ids, attention_mask=attention_mask).logits

            with torch.no_grad():
                teacher_logits = self.teacher(input_ids, attention_mask=attention_mask).logits

            kl = kl_loss(
                student_logits, teacher_logits, attention_mask,
                cfg.temperature, cfg.kl_token_chunk_size,
            ) / grad_accum

            kl_finite = torch.isfinite(kl)
            if kl_finite:
                kl.backward()

            # ── Graph loss ────────────────────────────────────────────────────
            graph_val = 0.0
            if use_graph:
                prompts: List[str] = batch["prompts"]
                answers = batch["answers"]
                if cfg.n_graph_prompts is not None and cfg.n_graph_prompts < len(prompts):
                    sel = random.sample(range(len(prompts)), cfg.n_graph_prompts)
                    prompts = [prompts[i] for i in sel]
                    answers = [answers[i] for i in sel]
                graph_loss_tensor, _ = backward_batch_graph_loss(
                    prompts=prompts,
                    answers=answers,
                    student_adapter=self.student_adapter,
                    teacher_adapter=self.teacher_adapter,
                    config=self.graph_config,
                    device=_DEVICE,
                    loss_scale=cfg.lambda_graph / grad_accum,
                )
                graph_val = float(graph_loss_tensor.item())

            micro_step += 1

            if not kl_finite:
                if micro_step % grad_accum == 0:
                    self.optimizer.zero_grad()
                continue

            accum_kl += float(kl.item())
            accum_graph += graph_val

            if micro_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), _GRAD_CLIP)
                self.optimizer.step()
                self.optimizer.zero_grad()

                self._train_step += 1
                self.history["train_step"].append(self._train_step)
                self.history["step_kl_loss"].append(accum_kl)
                self.history["step_graph_loss"].append(accum_graph)
                total_kl += accum_kl
                total_graph += accum_graph
                print(f"  step {self._train_step} | KL={accum_kl:.4f} | Graph={accum_graph:.4f}")
                accum_kl = 0.0
                accum_graph = 0.0
                n_steps += 1

        denom = max(n_steps, 1)
        return {"kl_loss": total_kl / denom, "graph_loss": total_graph / denom}

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
        for ds, acc in self._eval_all_extra().items():
            self.history[f"accuracy_{ds}"].append(acc)
            print(f"  Student baseline [{ds}]: {acc:.4f}")

        sample = self.loader.dataset[0]
        print("─" * 60)
        print("Sample [0]:")
        print(f"  prompt:  {str(sample['prompt'])[:120]!r}")
        print(f"  answer:  {str(sample['answer'])[:80]!r}")
        print(f"  tokens:  {sample['input_ids'].shape[0]} total, {sample['prompt_len']} prompt")
        print("─" * 60)

        print(
            f"ANOVA-KD | student={cfg.model} | teacher={cfg.teacher} | dataset={cfg.dataset}"
            f" | steps={cfg.steps} | lr={cfg.learning_rate} | temp={cfg.temperature}"
            f" | lambda_graph={cfg.lambda_graph} | labels={cfg.graph_node_labels}"
        )

        while self._train_step < cfg.steps:
            remaining = cfg.steps - self._train_step
            metrics = self.train_epoch(max_steps=min(cfg.eval_every_n_steps, remaining))
            if not metrics:
                break
            self.history["kl_loss"].append(metrics["kl_loss"])
            self.history["graph_loss"].append(metrics["graph_loss"])

            acc = self._eval()
            self.history["accuracy"].append(acc)
            self.history["accuracy_step"].append(self._train_step)
            extra_accs = self._eval_all_extra()
            for ds, ds_acc in extra_accs.items():
                self.history[f"accuracy_{ds}"].append(ds_acc)
            extra_str = "".join(f" | {ds}={a:.4f}" for ds, a in extra_accs.items())
            print(f"  [eval] step {self._train_step}/{cfg.steps} | Acc={acc:.4f}{extra_str}")

        save_history(self.history, cfg.save_dir)
        save_curves(self.history, cfg.save_dir, loss_key="step_kl_loss", loss_label="KL Loss")
        save_checkpoint(self.model, self.tokenizer, cfg.save_dir)
        print(f"Results saved to: {cfg.save_dir}")
        return dict(self.history)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="KL + ANOVA-graph distillation on GSM8K / SVAMP / local datasets."
    )
    add_standard_args(parser)
    add_kd_args(parser)
    group = parser.add_argument_group("kd_graph_args")
    # graph loss
    group.add_argument("--lambda-graph", type=float, default=0.1, dest="lambda_graph")
    group.add_argument(
        "--graph-node-labels",
        nargs="+", default=[], dest="graph_node_labels", metavar="LABEL",
        help="ANOVA supernode labels, e.g. 'sum units' 'arg1 range'. Pass 'all' for every category.",
    )
    group.add_argument("--nodes-per-label", type=int, default=10,
                       dest="nodes_per_label")
    group.add_argument("--anova-range-radius", "--anova_range_radius", type=int, default=0,
                       dest="anova_range_radius")
    group.add_argument(
        "--graph-loss-type", type=str, default="jsd", dest="graph_loss_type",
        choices=["jsd", "kld", "mse", "mse-norm", "mse-scale"],
    )
    group.add_argument("--top-k-logits", "--top_k_logits", type=float, default=0.95,
                       dest="top_k_logits")
    group.add_argument("--teacher-prop-neurons", type=float, default=0.1,
                       dest="teacher_prop_neurons_per_layer")
    group.add_argument("--student-prop-neurons", type=float, default=0.1,
                       dest="student_prop_neurons_per_layer")
    group.add_argument("--teacher-graph-batch-size", type=int, default=512,
                       dest="teacher_graph_batch_size")
    group.add_argument("--student-graph-batch-size", type=int, default=1,
                       dest="student_graph_batch_size")
    group.add_argument("--n-graph-prompts", type=int, default=None, dest="n_graph_prompts",
                       help="Max prompts per batch to compute graph loss for (None = all).")
    group.add_argument("--graph-verbose", action="store_true", dest="graph_verbose")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_data, test_data = load_data(args.dataset, test_limit=args.test_limit)
    print(f"Train: {len(train_data)} | Test: {len(test_data)}")
    trainer = ActivationsAnovaKDTrainer(
        ActivationsAnovaKDConfig(
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
            eval_datasets=args.eval_datasets,
            lambda_graph=args.lambda_graph,
            graph_node_labels=args.graph_node_labels,
            nodes_per_label=args.nodes_per_label,
            anova_range_radius=args.anova_range_radius,
            graph_loss_type=args.graph_loss_type,
            top_k_logits=args.top_k_logits,
            teacher_prop_neurons_per_layer=args.teacher_prop_neurons_per_layer,
            student_prop_neurons_per_layer=args.student_prop_neurons_per_layer,
            teacher_graph_batch_size=args.teacher_graph_batch_size,
            student_graph_batch_size=args.student_graph_batch_size,
            n_graph_prompts=args.n_graph_prompts,
            graph_verbose=args.graph_verbose,
        ),
        train_data,
        test_data,
    )
    trainer.train()


if __name__ == "__main__":
    main()

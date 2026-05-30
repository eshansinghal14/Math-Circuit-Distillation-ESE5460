"""CLI for simplified KL + graph-loss distillation."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from graph_loss.utils import DTYPE_CHOICES, resolve_torch_dtype
from utils import (
    EVAL_MAX_NEW_TOKENS,
    LLAMA_1B_MODEL_NAME,
    LLAMA_8B_MODEL_NAME,
    evaluate_prompt_answer_dict,
    get_default_device,
    json_to_prompt_answer_dict,
    load_model,
    load_prompt_answer_json,
    load_student_model_for_distillation,
    patch_tokenizer_no_special_tokens,
    resolve_test_path,
    resolve_train_test_paths,
    rm_dir_tree,
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
) -> torch.Tensor:
    """KL from teacher to student over all non-padding token positions."""
    t = temperature
    vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
    student_logits = student_logits[..., :vocab]
    teacher_logits = teacher_logits[..., :vocab]

    log_p_t = F.log_softmax(teacher_logits.float() / t, dim=-1)
    p_t = log_p_t.exp()
    log_q_s = F.log_softmax(student_logits.float() / t, dim=-1)
    kl_per_token = (p_t * (log_p_t - log_q_s)).sum(dim=-1)
    mask = attention_mask.float()
    return (kl_per_token * mask).sum() / mask.sum().clamp_min(1.0) * (t**2)


class AddDataset(Dataset):
    """Tokenize prompt-answer rows once for masked causal KL."""

    def __init__(self, data: Union[str, Dict[str, Union[int, str]]], tokenizer):
        if isinstance(data, str):
            with open(data, "r", encoding="utf-8") as f:
                raw = json.load(f)
            data = json_to_prompt_answer_dict(raw)
        self.samples = []
        for prompt, answer in data.items():
            # Support both int answers (arithmetic) and str answers (CoT/GSM8K).
            if isinstance(answer, int):
                answer_text = str(answer)
            else:
                answer_text = answer
            prompt_ids = tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                add_special_tokens=False,
            )["input_ids"].squeeze(0)
            answer_ids = tokenizer(
                answer_text + tokenizer.eos_token,
                return_tensors="pt",
                padding=False,
                add_special_tokens=False,
            )["input_ids"].squeeze(0)
            self.samples.append(
                {
                    "input_ids": torch.cat([prompt_ids, answer_ids]),
                    "prompt": str(prompt),
                    "answer": answer,  # preserve original type (int or str)
                },
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def collate_fn(examples, pad_id: int) -> Dict[str, Any]:
    """Right-pad sequences to the same length."""
    max_len = max(ex["input_ids"].size(0) for ex in examples)
    batch_size = len(examples)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

    for row, ex in enumerate(examples):
        ids = ex["input_ids"]
        length = ids.size(0)
        input_ids[row, :length] = ids
        attention_mask[row, :length] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompts": [str(ex["prompt"]) for ex in examples],
        "answers": [ex["answer"] for ex in examples],  # preserve int or str type
    }


# ─────────────────────────────────────────────────────────────────────────────
# Distillation config & run-dir resolution
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DistillationConfig:
    teacher_model: str = LLAMA_8B_MODEL_NAME
    student_model: str = LLAMA_1B_MODEL_NAME
    steps: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    temperature: float = 2.0
    grad_clip: float = 1.0
    lambda_graph: float = 0.1
    lambda_kl: float = 1.0
    graph_dtype: Optional[torch.dtype] = None
    teacher_prop_neurons_per_layer: float = 0.1
    student_prop_neurons_per_layer: float = 0.1
    graph_max_neurons_per_layer: int | None = None
    graph_top_k_logits: float | None = 0.955
    teacher_graph_batch_size: int = 512
    student_graph_batch_size: int = 1
    graph_verbose: bool = False

    student_dataset: Optional[str] = None
    student_anova_range_radius: int = 0
    student_anova_nodes_per_label: int = 10
    student_sum_min_specificity: float = 0.0
    teacher_anova_range_radius: int = 0
    teacher_anova_nodes_per_label: int = 10
    teacher_sum_min_specificity: float = 0.0
    graph_grad_norm_scale: bool = False
    graph_start_step: int = 1
    eval_batch_size: int = 50
    step_log_interval: int = 50
    eval_max_new_tokens: int = EVAL_MAX_NEW_TOKENS
    track_loss_grads: bool = False
    save_dir: str = "results/distillation"
    teacher_data_cache: Optional[str] = None
    seed: int = 42
    device: torch.device = field(default_factory=get_default_device)
    save_interval: int = 0
    label_refresh_interval: int = 0
    label_refresh_n_prompts: int = 8
    mlp_cache_refresh_interval: int = 0
    mlp_cache_batch_size: int = 64
    graph_loss_type: Literal["jsd", "kld", "mse", "mse-norm", "mse-scale"] = "jsd"
    graph_node_labels: Optional[List[str]] = None
    tokens_dla_nodes: bool = False
    compare_n_tokens: Optional[int] = None
    skip_baseline_eval: bool = False
    debug_compare_teacher_graphs: bool = False



def find_student_source(path: str) -> Optional[str]:
    """Return *path* if it is an HF checkpoint directory, otherwise ``None``.

    Checkpoints are saved directly into their folder (``model.safetensors`` +
    ``config.json`` at the root), matching how ``_save_checkpoint_at_step`` works.
    """
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
    runs_subdir: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Return ``(run_dir, student_source)``.

    All training outputs go directly under ``<save_dir>`` (or ``<save_dir>/<runs_subdir>``
    if set). There is no timestamp or extra run subfolder.
    """
    save_dir = os.path.abspath(save_dir)
    sub = (runs_subdir or "").strip().strip("/").strip("\\")
    run_dir = os.path.join(save_dir, sub) if sub else save_dir

    if not resume:
        return run_dir, None

    # Resolve student_source independently of run_dir so that --save-dir (where
    # outputs go) and --checkpoint-run (where weights come from) are always distinct.
    if checkpoint_run:
        src = (
            os.path.normpath(checkpoint_run)
            if os.path.isabs(checkpoint_run)
            else os.path.join(save_dir, checkpoint_run)
        )
        student_source = find_student_source(src)
        if student_source is None:
            raise SystemExit(
                f"No student weights found in {src}.\n"
                "Expected config.json or model.safetensors directly in that folder.\n"
                "Train a run first."
            )
        print(f"Loading student from {student_source}")
    else:
        student_source = None  # caller fills this in via --resume-step

    return run_dir, student_source


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────


class DistillationTrainer:
    def __init__(
        self,
        *,
        config: DistillationConfig,
        train_data: Dict[str, int],
        test_data: Dict[str, int],
        extra_eval_data: Optional[Dict[str, Dict[str, int]]] = None,
        tokenizer=None,
        student=None,
        resume_step: Optional[int] = None,
    ) -> None:
        self.config = config
        self.test_data = test_data
        self.extra_eval_data = extra_eval_data or {}
        self.device = torch.device(config.device)
        self._resume_step = resume_step
        self._resume = resume_step is not None
        self._use_graph = config.lambda_graph > 0.0
        self._graph_start_step = config.graph_start_step
        seed_all(config.seed)

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(config.student_model)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer = patch_tokenizer_no_special_tokens(self.tokenizer)
        self.tokenizer.padding_side = "right"

        self.teacher_data_cache = (
            self._load_teacher_data_cache(config.teacher_data_cache)
            if config.teacher_data_cache
            else None
        )
        if self.teacher_data_cache is not None:
            cached_keys = self.teacher_data_cache._samples
            filtered = {p: a for p, a in train_data.items() if (p, a) in cached_keys}
            n_orig, n_kept = len(train_data), len(filtered)
            print(
                f"Teacher cache filter: {n_kept}/{n_orig} training prompts have cached data "
                f"({100 * n_kept / max(n_orig, 1):.1f}%).",
            )
            if n_kept == 0:
                raise RuntimeError(
                    "No training prompts found in teacher cache. "
                    "Run graph_loss/generate_teacher_data.py first, or remove --teacher-data-cache.",
                )
            self.train_data = filtered
        else:
            self.train_data = train_data

        if student is not None:
            self.student = student
        else:
            self.student, self.tokenizer = load_student_model_for_distillation(
                None,
                config.student_model,
                self.device,
            )
        self.student = self.student.to(self.device).float()
        self.student.train()

        if config.compare_n_tokens is not None and config.teacher_data_cache is not None:
            raise ValueError(
                "--compare-n-tokens is incompatible with --teacher-data-cache. "
                "Remove --teacher-data-cache to use live teacher computation."
            )

        self.teacher = None
        self.teacher_graph_model = None
        self.teacher_graph_adapter = None

        need_live_teacher = (
            self.teacher_data_cache is None
            or config.debug_compare_teacher_graphs
        )
        if need_live_teacher:
            mode_tag = (
                "debug-compare" if config.debug_compare_teacher_graphs
                else "live-teacher" if self._use_graph
                else "kl-only"
            )
            print(f"Loading teacher ({mode_tag} mode): {config.teacher_model}")
            self.teacher, _ = load_model(config.teacher_model)
            self.teacher = self.teacher.to(self.device)
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False
        elif self.teacher_data_cache is not None:
            print(f"Using cached teacher data: {config.teacher_data_cache}")

        self.graph_loss_config = None
        self.student_graph_adapter = None
        if self._use_graph:
            from graph_loss.hf_adapter import HFLlamaGraphAdapter
            from graph_loss.training import GraphAuxConfig

            self.graph_loss_config = GraphAuxConfig(
                lambda_graph=config.lambda_graph,
                graph_dtype=config.graph_dtype,
                teacher_prop_neurons_per_layer=config.teacher_prop_neurons_per_layer,
                student_prop_neurons_per_layer=config.student_prop_neurons_per_layer,
                max_neurons_per_layer=config.graph_max_neurons_per_layer,
                top_k_logits=config.graph_top_k_logits,
                temperature=config.temperature,
                teacher_graph_batch_size=config.teacher_graph_batch_size,
                student_graph_batch_size=config.student_graph_batch_size,
                verbose=config.graph_verbose,
                student_anova_range_radius=config.student_anova_range_radius,
                student_anova_nodes_per_label=config.student_anova_nodes_per_label,
                student_sum_min_specificity=config.student_sum_min_specificity,
                teacher_anova_range_radius=config.teacher_anova_range_radius,
                teacher_anova_nodes_per_label=config.teacher_anova_nodes_per_label,
                teacher_sum_min_specificity=config.teacher_sum_min_specificity,
                dataset=config.student_dataset,
                graph_loss_type=config.graph_loss_type,
                student_graph_labels=config.graph_node_labels,
                tokens_dla_nodes=config.tokens_dla_nodes,
                compare_n_tokens=config.compare_n_tokens,
                debug_compare_teacher_graphs=config.debug_compare_teacher_graphs,
            )

            self.student_graph_adapter = HFLlamaGraphAdapter(
                self.student,
                self.tokenizer,
                self.device,
            )

            if self.teacher is not None and (config.tokens_dla_nodes or config.debug_compare_teacher_graphs or self.teacher_data_cache is None):
                self.teacher_graph_adapter = HFLlamaGraphAdapter(
                    self.teacher,
                    self.tokenizer,
                    self.device,
                )

            if config.student_dataset and not config.label_refresh_interval:
                # Only build here when label_refresh_interval==0; otherwise
                # train() will call _refresh_student_labels() at startup.
                from graph_loss.neuron_activation_heatmap import _resolve_dataset_path
                from graph_loss.precompute_mlp_inputs import build_mlp_input_cache

                dataset_path = _resolve_dataset_path(config.student_dataset)
                self.student.eval()
                mlp_cache = build_mlp_input_cache(
                    self.student_graph_adapter,
                    dataset_path,
                    config.student_model,
                    batch_size=config.mlp_cache_batch_size,
                )
                self.student.train()
                self.graph_loss_config.mlp_input_cache = mlp_cache
                n_prompts = int(mlp_cache.get("meta", {}).get("n_prompts", 0))
                print(f"  Built MLP input cache: {n_prompts} prompts")

        bnb = None
        try:
            bnb = importlib.import_module("bitsandbytes")
        except ImportError:
            pass
        if bnb is not None:
            self.optimizer = bnb.optim.PagedAdamW8bit(
                params=self.student.parameters(),
                lr=config.learning_rate,
            )
            print("Using 8-bit Paged AdamW optimizer to save memory.")
        else:
            self.optimizer = AdamW(
                params=self.student.parameters(),
                lr=config.learning_rate,
            )
            print("Using standard AdamW optimizer.")

        self.loader = DataLoader(
            AddDataset(self.train_data, self.tokenizer),
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=partial(collate_fn, pad_id=self.tokenizer.eos_token_id),
        )

        self.history: Dict[str, List] = defaultdict(list)
        self._train_step = 0
        self._step_log_eval_accuracy = 0.0
        self._step_log_extra_eval_acc: Dict[str, float] = {}

    @staticmethod
    def _load_teacher_data_cache(cache_dir: str):
        from graph_loss.teacher_data_cache import TeacherDataCache

        return TeacherDataCache(cache_dir)

    def _refresh_student_labels(self) -> None:
        """Rebuild MLP input cache from student dataset using current student weights."""
        if self.student_graph_adapter is None or self.graph_loss_config is None:
            return
        if not self.config.student_dataset:
            raise RuntimeError(
                "--label-refresh-interval > 0 requires --student-dataset."
            )
        from graph_loss.neuron_activation_heatmap import _resolve_dataset_path
        from graph_loss.precompute_mlp_inputs import build_mlp_input_cache

        dataset_path = _resolve_dataset_path(self.config.student_dataset)
        self.student.eval()
        mlp_cache = build_mlp_input_cache(
            self.student_graph_adapter,
            dataset_path,
            self.config.student_model,
            batch_size=self.config.mlp_cache_batch_size,
        )
        self.student.train()
        self.graph_loss_config.mlp_input_cache = mlp_cache
        n_prompts = int(mlp_cache["meta"].get("n_prompts", 0))
        print(
            f"  [label-refresh] step={self._train_step} "
            f"MLP cache rebuilt: {n_prompts} prompts, "
            f"{len(mlp_cache['layer_inputs'])} layers"
        )

    def _extra_eval_history_key(self, prefix: str) -> str:
        return f"accuracy_extra_{prefix}"

    def _teacher_logits_for_batch(
        self,
        batch: Dict[str, Any],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.teacher_data_cache is not None:
            try:
                return self.teacher_data_cache.get_batch_logits(
                    prompts=batch["prompts"],
                    answers=batch["answers"],
                    input_ids=input_ids,
                    device=self.device,
                )
            except (KeyError, FileNotFoundError, ValueError) as e:
                raise RuntimeError(
                    "Teacher data cache is enabled but required cached logits could not "
                    "be loaded for this batch. Regenerate the cache for this "
                    "dataset/tokenizer or remove --teacher-data-cache."
                ) from e
        if self.teacher is None:
            raise RuntimeError("No teacher model or teacher cache is configured.")
        with torch.no_grad():
            return self.teacher(input_ids=input_ids, attention_mask=attention_mask).logits

    def _forward_kl(self, batch: Dict[str, Any]) -> tuple[torch.Tensor, Dict[str, float]]:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        teacher_logits = self._teacher_logits_for_batch(batch, input_ids, attention_mask)
        student_logits = self.student(input_ids=input_ids, attention_mask=attention_mask).logits

        loss = kl_loss(
            student_logits,
            teacher_logits,
            attention_mask,
            self.config.temperature,
        )
        metrics: Dict[str, float] = {"kl_loss": float(loss.item())}
        metrics["total_loss"] = float(loss.item())
        if self.config.lambda_kl == 0.0:
            loss = loss.detach() * 0.0
        elif self.config.lambda_kl != 1.0:
            loss = self.config.lambda_kl * loss
        return loss, metrics

    def _backward_graph_loss(
        self,
        batch: Dict[str, Any],
        metrics: Dict[str, float],
        non_graph_loss: torch.Tensor,
        kl_grad_norm: Optional[float] = None,
    ) -> torch.Tensor:
        if not self._use_graph:
            return non_graph_loss
        if self.student_graph_adapter is None:
            raise RuntimeError("Graph loss requested but student graph adapter is missing.")
        if self.graph_loss_config is None:
            raise RuntimeError("Graph loss config was not initialized.")

        use_grad_norm_scale = (
            self.config.graph_grad_norm_scale
            and kl_grad_norm is not None
            and kl_grad_norm > 1e-8
        )

        if use_grad_norm_scale:
            kl_grad_snap: Dict[str, Optional[torch.Tensor]] = {
                n: (p.grad.clone() if p.grad is not None else None)
                for n, p in self.student.named_parameters()
            }

        try:
            from graph_loss.training import backward_batch_graph_loss

            graph_loss, graph_metrics = backward_batch_graph_loss(
                prompts=batch["prompts"],
                student_adapter=self.student_graph_adapter,
                config=self.graph_loss_config,
                device=self.device,
                loss_scale=1.0 if use_grad_norm_scale else self.config.lambda_graph,
                teacher_cache=self.teacher_data_cache,
                teacher_adapter=(
                    self.teacher_graph_adapter
                    if self.teacher_data_cache is None
                    else None
                ),
                answers=batch["answers"],
                debug_teacher_adapter=(
                    self.teacher_graph_adapter
                    if self.config.debug_compare_teacher_graphs
                    else None
                ),
            )
        except RuntimeError:
            exc_path = os.path.join(self.config.save_dir, "exception_checkpoint")
            os.makedirs(exc_path, exist_ok=True)
            self.student.save_pretrained(exc_path)
            self.tokenizer.save_pretrained(exc_path)
            raise

        if use_grad_norm_scale:
            graph_grad_sq = 0.0
            for n, p in self.student.named_parameters():
                if p.grad is None:
                    continue
                kl_g = kl_grad_snap[n]
                graph_only = p.grad if kl_g is None else (p.grad - kl_g)
                graph_grad_sq += graph_only.float().norm().item() ** 2
            graph_grad_norm = graph_grad_sq ** 0.5

            if graph_grad_norm > 1e-8:
                effective_scale = self.config.lambda_graph * kl_grad_norm / graph_grad_norm
                for n, p in self.student.named_parameters():
                    if p.grad is None:
                        continue
                    kl_g = kl_grad_snap[n]
                    graph_only = p.grad if kl_g is None else (p.grad - kl_g)
                    p.grad = (kl_g if kl_g is not None else torch.zeros_like(p.grad)) + effective_scale * graph_only
            else:
                for n, p in self.student.named_parameters():
                    p.grad = kl_grad_snap[n]

            metrics["graph_grad_norm_scale_effective_lambda"] = float(
                self.config.lambda_graph * kl_grad_norm / max(graph_grad_norm, 1e-8)
            )

        _skip_metric_keys = {
            "teacher_supernodes",
            "student_supernodes",
            "aligned_teacher_supernodes",
            "graph_prompts",
            "graph_backward_prompts",
            "edge_loss",
            "student_graph_neurons",
        }
        graph_weighted = self.config.lambda_graph * graph_loss
        total = non_graph_loss.detach() + graph_weighted
        metrics["graph_loss"] = float(graph_loss.item())
        metrics["total_loss"] = float(total.item())
        for key, value in graph_metrics.items():
            if key in _skip_metric_keys:
                continue
            if isinstance(value, (int, float)):
                metrics[f"graph_{key}"] = float(value)
        return total

    def _record_step_metrics(self, epoch: int, batch_step: int, metrics: Dict[str, float]) -> None:
        self._train_step += 1
        self.history["train_step"].append(self._train_step)
        for key, value in metrics.items():
            self.history[f"step_{key}"].append(float(value))

    def _run_training_eval(self) -> float:
        cfg = self.config
        acc = evaluate_prompt_answer_dict(
            self.student,
            self.tokenizer,
            self.test_data,
            batch_size=cfg.eval_batch_size,
            max_new_tokens=cfg.eval_max_new_tokens,
        )
        self.student.train()
        acc_f = float(acc)
        self._step_log_eval_accuracy = acc_f
        self.history["accuracy"].append(acc_f)
        for prefix, data in self.extra_eval_data.items():
            extra_acc = evaluate_prompt_answer_dict(
                self.student,
                self.tokenizer,
                data,
                batch_size=cfg.eval_batch_size,
                max_new_tokens=cfg.eval_max_new_tokens,
            )
            extra_f = float(extra_acc)
            self.history[self._extra_eval_history_key(prefix)].append(extra_f)
            self._step_log_extra_eval_acc[prefix] = extra_f
            self.student.train()
        self._clear_cuda_cache()
        return acc_f

    def _clear_cuda_cache(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _clone_student_grads(self) -> List[Optional[torch.Tensor]]:
        return [
            param.grad.detach().clone() if param.grad is not None else None
            for param in self.student.parameters()
        ]

    def _record_kl_only_grad_metrics(self, metrics: Dict[str, float]) -> None:
        kl_norm_sq = 0.0
        for param in self.student.parameters():
            if param.grad is not None:
                grad = param.grad.detach().float()
                kl_norm_sq += float((grad * grad).sum().item())
        metrics["kl_loss_grad"] = math.sqrt(kl_norm_sq)
        metrics["graph_loss_grad"] = 0.0
        metrics["loss_cossim"] = 0.0

    def _record_loss_grad_metrics(
        self,
        metrics: Dict[str, float],
        kl_grads: List[Optional[torch.Tensor]],
    ) -> None:
        kl_norm_sq = 0.0
        graph_norm_sq = 0.0
        dot = 0.0
        for param, kl_grad in zip(self.student.parameters(), kl_grads, strict=True):
            if kl_grad is not None:
                kl_grad_f = kl_grad.float()
                kl_norm_sq += float((kl_grad_f * kl_grad_f).sum().item())
            if param.grad is None:
                continue
            graph_grad = param.grad.detach()
            if kl_grad is not None:
                graph_grad = graph_grad - kl_grad.to(device=graph_grad.device)
                dot += float((kl_grad.to(device=graph_grad.device).float() * graph_grad.float()).sum().item())
            graph_grad_f = graph_grad.float()
            graph_norm_sq += float((graph_grad_f * graph_grad_f).sum().item())

        kl_norm = math.sqrt(kl_norm_sq)
        graph_norm = math.sqrt(graph_norm_sq)
        metrics["kl_loss_grad"] = kl_norm
        metrics["graph_loss_grad"] = graph_norm
        metrics["loss_cossim"] = (
            dot / (kl_norm * graph_norm)
            if kl_norm > 0.0 and graph_norm > 0.0
            else 0.0
        )

    def _get_grad_norm(self) -> float:
        total_sq = 0.0
        for p in self.student.parameters():
            if p.grad is not None:
                total_sq += p.grad.detach().float().norm().item() ** 2
        return total_sq ** 0.5

    def train_epoch(self, epoch: int, *, max_steps: Optional[int] = None) -> Dict[str, float]:
        self.student.train()
        agg = defaultdict(float)
        n_steps = 0
        skipped_nonfinite = 0
        interval = defaultdict(float)
        interval_steps = 0
        for step, batch in enumerate(self.loader):
            loss, metrics = self._forward_kl(batch)
            if not torch.isfinite(loss).item():
                skipped_nonfinite += 1
                continue
            use_graph_this_step = self._use_graph and (
                self._train_step + 1 >= self._graph_start_step
            )
            self.optimizer.zero_grad(set_to_none=use_graph_this_step)
            if use_graph_this_step:
                loss.backward()
                kl_grads = (
                    self._clone_student_grads()
                    if self.config.track_loss_grads
                    else None
                )
                self._clear_cuda_cache()
                kl_grad_norm = self._get_grad_norm() if self.config.graph_grad_norm_scale else None
                loss = self._backward_graph_loss(batch, metrics, loss, kl_grad_norm=kl_grad_norm)
                if kl_grads is not None:
                    self._record_loss_grad_metrics(metrics, kl_grads)
            else:
                loss.backward()
                if self.config.track_loss_grads:
                    self._record_kl_only_grad_metrics(metrics)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.config.grad_clip)
            self.optimizer.step()
            if use_graph_this_step:
                self._clear_cuda_cache()

            if not torch.isfinite(loss).item():
                skipped_nonfinite += 1
                continue
            self._record_step_metrics(epoch, step, metrics)
            self._save_history()
            self._save_curves()
            for key, value in metrics.items():
                agg[key] += float(value)
                interval[key] += float(value)
            n_steps += 1
            interval_steps += 1

            if self.config.save_interval > 0 and self._train_step % self.config.save_interval == 0:
                self._save_checkpoint_at_step(self._train_step)

            if self._use_graph and (
                (
                    self.config.label_refresh_interval > 0
                    and self._train_step % self.config.label_refresh_interval == 0
                )
                or (
                    self.config.mlp_cache_refresh_interval > 0
                    and self._train_step % self.config.mlp_cache_refresh_interval == 0
                )
            ):
                self._refresh_student_labels()

            if self._train_step == 1 or self._train_step % max(1, self.config.step_log_interval) == 0:
                self._run_training_eval()
                extra_eval_s = ""
                if self._step_log_extra_eval_acc:
                    extra_eval_s = " | extra " + ", ".join(
                        f"{p}={self._step_log_extra_eval_acc[p]:.4f}"
                        for p in sorted(self._step_log_extra_eval_acc.keys())
                    )
                denom = max(interval_steps, 1)
                kl_avg = interval.get("kl_loss", 0.0) / denom
                graph_s = ""
                if self._use_graph and self._train_step < self._graph_start_step:
                    graph_s = f" | Graph [warmup until step {self._graph_start_step}]"
                elif self._use_graph:
                    graph_avg = interval.get("graph_loss", 0.0) / denom
                    graph_s = f" | Graph {graph_avg:.4f}"
                    if self.config.graph_grad_norm_scale:
                        eff_lam_key = "graph_grad_norm_scale_effective_lambda"
                        eff_lam_avg = interval.get(eff_lam_key, 0.0) / denom
                        graph_s += f" | effLambda {eff_lam_avg:.4f}"
                grad_s = ""
                if self.config.track_loss_grads:
                    grad_s = (
                        f" | grad KL {metrics.get('kl_loss_grad', 0.0):.3e}"
                        f" Graph {metrics.get('graph_loss_grad', 0.0):.3e}"
                        f" cos {metrics.get('loss_cossim', 0.0):.4f}"
                    )
                print(
                    f"  step {self._train_step:04d} | KL {kl_avg:.4f}"
                    f"{graph_s} | Acc {self._step_log_eval_accuracy:.4f}"
                    f"{extra_eval_s}{grad_s}",
                )
                interval.clear()
                interval_steps = 0

            if max_steps is not None and n_steps >= max_steps:
                break

        if n_steps == 0:
            print(
                f"  WARNING: no valid optimizer steps this pass "
                f"({skipped_nonfinite} non-finite batch(es)).",
            )
        return {key: value / max(n_steps, 1) for key, value in agg.items()}

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
            # Trim history to only the entries up to resume_step.
            train_steps = self.history.get("train_step", [])
            n_keep = sum(1 for s in train_steps if s <= self._resume_step)
            for key in list(self.history.keys()):
                if isinstance(self.history[key], list) and len(self.history[key]) == len(train_steps):
                    self.history[key] = self.history[key][:n_keep]
            start_epoch = len(self.history.get("epoch", []))
            self._step_log_eval_accuracy = (
                float(self.history["accuracy"][-1])
                if self.history.get("accuracy")
                else 0.0
            )
            opt_path = os.path.join(
                cfg.save_dir, f"step_{self._resume_step}_checkpoint", "optimizer.pt"
            )
            if os.path.isfile(opt_path):
                self.optimizer.load_state_dict(
                    torch.load(opt_path, map_location=self.device)
                )
                print(f"Restored optimizer state from step {self._resume_step}.")
            print(f"Warm-starting from step {self._train_step + 1}.")
        elif cfg.skip_baseline_eval:
            self._step_log_eval_accuracy = 0.0
        else:
            print("Evaluating baselines...")
            student_base = evaluate_prompt_answer_dict(
                self.student,
                self.tokenizer,
                self.test_data,
                batch_size=cfg.eval_batch_size,
                max_new_tokens=cfg.eval_max_new_tokens,
            )
            self.student.train()
            self.history["student_baseline"] = float(student_base)
            print(f"  Student baseline accuracy (train): {student_base:.4f}")
            for prefix, data in self.extra_eval_data.items():
                extra_acc = evaluate_prompt_answer_dict(
                    self.student,
                    self.tokenizer,
                    data,
                    batch_size=cfg.eval_batch_size,
                    max_new_tokens=cfg.eval_max_new_tokens,
                )
                self.student.train()
                self.history[f"student_baseline_{prefix}"] = float(extra_acc)
                print(f"  Student baseline accuracy ({prefix}): {float(extra_acc):.4f}")

            teacher_loaded_for_baseline = False
            if self.teacher is None:
                print(f"  Loading teacher for baseline eval: {cfg.teacher_model}")
                teacher_for_baseline, _ = load_model(cfg.teacher_model)
                teacher_for_baseline = teacher_for_baseline.to(self.device)
                teacher_for_baseline.eval()
                for param in teacher_for_baseline.parameters():
                    param.requires_grad = False
                teacher_loaded_for_baseline = True
            else:
                teacher_for_baseline = self.teacher

            teacher_base = evaluate_prompt_answer_dict(
                teacher_for_baseline,
                self.tokenizer,
                self.test_data,
                batch_size=cfg.eval_batch_size,
                max_new_tokens=cfg.eval_max_new_tokens,
            )
            self.history["teacher_baseline"] = float(teacher_base)
            print(f"  Teacher baseline accuracy (train): {teacher_base:.4f}")
            for prefix, data in self.extra_eval_data.items():
                extra_acc = evaluate_prompt_answer_dict(
                    teacher_for_baseline,
                    self.tokenizer,
                    data,
                    batch_size=cfg.eval_batch_size,
                    max_new_tokens=cfg.eval_max_new_tokens,
                )
                self.history[f"teacher_baseline_{prefix}"] = float(extra_acc)
                print(f"  Teacher baseline accuracy ({prefix}): {float(extra_acc):.4f}")

            if teacher_loaded_for_baseline:
                del teacher_for_baseline
                self._clear_cuda_cache()

            self._step_log_eval_accuracy = float(student_base)

        target_step = int(cfg.steps)
        print("=" * 60)
        title = "KL + Graph Distillation" if self._use_graph else "KL Distillation"
        print(title)
        print(f"  Run dir:          {cfg.save_dir}")
        print(f"  Steps:            {self._train_step + 1}..{target_step}")
        print(f"  Batch size:       {cfg.batch_size}")
        print(f"  LR:               {cfg.learning_rate}")
        print(f"  Temperature:      {cfg.temperature}")
        if self._use_graph:
            print(f"  lambda_graph:     {cfg.lambda_graph}")
            print(f"  teacher prop neurons/layer: {cfg.teacher_prop_neurons_per_layer}")
            print(f"  student prop neurons/layer: {cfg.student_prop_neurons_per_layer}")
        print(f"  Eval every:       {cfg.step_log_interval} training batches")
        if cfg.save_interval > 0:
            print(f"  Save every:       {cfg.save_interval} steps → checkpoint_step_NNNN/")
        if cfg.label_refresh_interval > 0:
            print(f"  Label refresh:    every {cfg.label_refresh_interval} steps (all train prompts)")
        print("=" * 60)

        if self._use_graph and self.config.label_refresh_interval > 0:
            print("  [label-refresh] Initial label computation...")
            self._refresh_student_labels()

        epoch = start_epoch
        while self._train_step < target_step:
            remaining_steps = target_step - self._train_step
            epoch_metrics = self.train_epoch(epoch, max_steps=remaining_steps)
            if not epoch_metrics and remaining_steps > 0:
                break
            self.history["epoch"].append(epoch + 1)
            for key, value in epoch_metrics.items():
                self.history[key].append(value)
            acc_s = f"{self._step_log_eval_accuracy:.4f}"
            graph_s = ""
            if self._use_graph:
                graph_s = f", Graph={epoch_metrics.get('graph_loss', float('nan')):.4f}"
            print(
                f"Pass {epoch + 1}: "
                f"KL={epoch_metrics.get('kl_loss', float('nan')):.4f}"
                f"{graph_s}, Acc={acc_s}, Step={self._train_step}/{target_step}",
            )
            epoch += 1

        self._save_history()
        self._save_curves()
        self._save_checkpoint()
        print(f"Results saved to: {cfg.save_dir}")
        return dict(self.history)

    def _save_checkpoint(self) -> None:
        path = self.config.save_dir
        os.makedirs(path, exist_ok=True)
        self.student.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

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
            print("matplotlib not installed; skipping curve plots.")
            return
        loss_steps = self.history.get("train_step", [])
        if not loss_steps:
            return
        kl_series = self.history.get("step_kl_loss") or self.history.get("kl_loss", [])
        graph_series = self.history.get("step_graph_loss") or self.history.get("graph_loss", [])
        if self._use_graph:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].plot(loss_steps[: len(kl_series)], kl_series, marker="o", markersize=2)
            axes[0].set_title("KL Loss")
            axes[1].plot(
                loss_steps[: len(graph_series)],
                graph_series,
                marker="o",
                markersize=2,
                color="tab:green",
            )
            axes[1].set_title("Graph Loss")
            acc_ax = axes[2]
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(loss_steps[: len(kl_series)], kl_series, marker="o", markersize=2)
            axes[0].set_title("KL Loss")
            acc_ax = axes[1]
        acc_series = self.history.get("accuracy") or []
        acc_x = self.history.get("eval_train_step") or list(range(1, len(acc_series) + 1))
        extra_prefixes = [p for p in self.extra_eval_data if self.history.get(self._extra_eval_history_key(p))]
        use_legend = bool(extra_prefixes)
        if acc_series:
            acc_ax.plot(
                acc_x[: len(acc_series)],
                acc_series,
                marker="o",
                markersize=3,
                label="test" if use_legend else None,
            )
        for prefix in extra_prefixes:
            key = self._extra_eval_history_key(prefix)
            extra_series = self.history.get(key) or []
            if extra_series:
                acc_ax.plot(
                    acc_x[: len(extra_series)],
                    extra_series,
                    marker="o",
                    markersize=3,
                    label=prefix,
                )
        acc_ax.set_title("Accuracy")
        acc_ax.set_ylim(0, 1)
        if use_legend:
            acc_ax.legend(fontsize=7, loc="lower right")
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        os.makedirs(self.config.save_dir, exist_ok=True)
        out = os.path.join(self.config.save_dir, "training_curves.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simplified KL distillation with optional graph-loss auxiliary objective.",
    )
    parser.add_argument("--student-model", type=str, default=LLAMA_1B_MODEL_NAME)
    parser.add_argument("--teacher-model", type=str, default=LLAMA_8B_MODEL_NAME)
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        metavar="PREFIX",
        help="Dataset prefix, e.g. 2d_add for datasets/2d_add_train.json + _test.json.",
    )
    parser.add_argument("--datasets-dir", type=str, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None,
                        help="Use only the first N test examples for evaluation "
                             "(speeds up periodic eval on large test sets like GSM8K).")
    parser.add_argument("--train-start-idx", type=int, default=None, dest="train_start_idx",
                        help="Index in the training dataset to start from (inclusive).")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-graph", type=float, default=0.1)
    parser.add_argument(
        "--lambda-kl",
        type=float,
        default=1.0,
        help=(
            "Scale factor for the KL distillation loss.  Set to 0 to run "
            "graph-loss-only.  Default 1.0."
        ),
    )
    parser.add_argument("--dtype", choices=DTYPE_CHOICES, default="float32")
    parser.add_argument("--teacher-prop-neurons-per-layer", type=float, default=0.1)
    parser.add_argument("--student-prop-neurons-per-layer", type=float, default=0.1)
    parser.add_argument(
        "--graph-max-neurons-per-layer",
        type=int,
        default=None,
        help=(
            "Optional absolute cap on attribution neurons selected per layer. "
            "When unset (default None), behavior is unchanged and the per-layer "
            "neuron count k scales with sequence length. When set, k is capped "
            "so it stops scaling with sequence length, fixing CoT/long-prompt "
            "memory blowup. Only affects the compare-tokens graph path."
        ),
    )
    parser.add_argument("--graph-top-k-logits", type=float, default=0.95, dest="graph_top_k_logits",
                        help="Cumulative probability threshold in (0, 1] for logit nodes. "
                             "Selects the fewest top logits summing to this fraction, capped at 10.")
    parser.add_argument("--teacher-graph-batch-size", type=int, default=512)
    parser.add_argument("--student-graph-batch-size", type=int, default=1)

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--graph-loss-type",
        choices=["jsd", "kld", "mse", "mse-norm", "mse-scale"],
        default="jsd",
        help=(
            "Similarity function for graph loss. "
            "'jsd' (default): symmetric Jensen-Shannon divergence. "
            "'kld': forward KL(teacher||student). "
            "'mse': raw Frobenius MSE on coarsened adjacency values. "
            "'mse-norm': MSE on L1-row-normalised distributions (scale-invariant). "
            "'mse-scale': mse-norm + row-sum magnitude penalty."
        ),
    )
    parser.add_argument(
        "--graph-grad-norm-scale",
        action="store_true",
        default=False,
        help=(
            "Dynamically scale the graph loss contribution so its gradient norm "
            "equals the KL gradient norm before applying lambda_graph.  Prevents "
            "graph gradients from dominating when GraphActive spikes.  "
            "Adds ~1 GB memory overhead to store the KL gradient snapshot."
        ),
    )
    parser.add_argument(
        "--graph-start-step",
        type=int,
        default=1,
        help=(
            "Step (1-indexed, inclusive) at which graph loss is first applied. "
            "Steps before this use KL loss only.  Default 1 (graph loss always on)."
        ),
    )
    parser.add_argument("--eval-max-new-tokens", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=50)
    parser.add_argument("--eval-datasets", nargs="*", default=None, metavar="DATASET")
    parser.add_argument(
        "--track-loss-grads",
        action="store_true",
        help="Log KL and graph gradient norms plus their cosine similarity.",
    )
    parser.add_argument("--teacher-data-cache", type=str, default=None, metavar="PATH")
    parser.add_argument(
        "--student-dataset",
        type=str,
        default=None,
        metavar="DATASET",
        help="Activation dataset for student full_search clustering (e.g. 22_add_tight_all.json).",
    )
    parser.add_argument(
        "--student-anova-range-radius",
        type=int,
        default=0,
        help="Radius around target arg/sum values for the student ANOVA range mask.",
    )
    parser.add_argument(
        "--student-anova-nodes-per-label",
        type=int,
        default=10,
        help="Cap on neurons per labelled student supernode (default 10).",
    )
    parser.add_argument(
        "--student-sum-min-specificity",
        type=float,
        default=0.0,
        help="Minimum ANOVA specificity for student supernodes.",
    )
    parser.add_argument(
        "--teacher-anova-range-radius",
        type=int,
        default=0,
        help="Radius around target arg/sum values for the teacher ANOVA range mask.",
    )
    parser.add_argument(
        "--teacher-anova-nodes-per-label",
        type=int,
        default=10,
        help="Cap on neurons per labelled teacher supernode (default 10).",
    )
    parser.add_argument(
        "--teacher-sum-min-specificity",
        type=float,
        default=0.0,
        help="Minimum ANOVA specificity for teacher supernodes.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "results", "distillation"),
    )
    parser.add_argument("--resume-step", type=int, default=None, dest="resume_step",
                        help="Resume training from a specific step checkpoint.")
    parser.add_argument("--checkpoint-run", default=None, metavar="PATH")
    parser.add_argument("--step-log-interval", type=int, default=50)
    parser.add_argument(
        "--save-interval",
        type=int,
        default=0,
        help="Save a checkpoint every N steps. 0 = final only.",
    )
    parser.add_argument(
        "--label-refresh-interval",
        type=int,
        default=0,
        help=(
            "Re-run ANOVA on --label-refresh-n-prompts training prompts every N steps "
            "to refresh the in-memory fixed-label cache. "
            "0 = never refresh (use precomputed labels only). "
            "Only active when --student-dataset is set."
        ),
    )
    parser.add_argument(
        "--label-refresh-n-prompts",
        type=int,
        default=8,
        help="Number of training prompts (taken from the start of the train set) used for each label refresh.",
    )
    parser.add_argument(
        "--mlp-cache-refresh-interval",
        type=int,
        default=0,
        help=(
            "Rebuild the in-memory MLP input cache every N steps. "
            "0 = never refresh (default).  Requires --student-dataset."
        ),
    )
    parser.add_argument(
        "--mlp-cache-batch-size",
        type=int,
        default=64,
        help=(
            "Batch size for building the MLP input cache (default 64, fast on GPU). "
            "The full dataset (~10k prompts) is always used for ANOVA coverage."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-baseline-eval",
        action="store_true",
        default=False,
        dest="skip_baseline_eval",
        help="Skip the teacher/student baseline accuracy evaluation at the start of training.",
    )
    parser.add_argument(
        "--graph-node-labels",
        nargs="+",
        default=None,
        metavar="LABEL",
        dest="graph_node_labels",
        help=(
            "Whitelist of ANOVA supernode label names to include when building the "
            "student supergraph and computing graph loss. "
            "E.g. --graph-node-labels 'arg1 range' 'sum units'. "
            "If omitted, all labels are included."
        ),
    )
    parser.add_argument(
        "--tokens-dla-nodes",
        action="store_true",
        default=False,
        dest="tokens_dla_nodes",
        help=(
            "Include arg-token supernodes and a DLA supernode in both teacher and student "
            "graphs during distillation. Replaces the old --student-include-dla-node and "
            "--student-include-arg-nodes flags."
        ),
    )
    parser.add_argument(
        "--compare-n-tokens",
        type=int,
        default=None,
        dest="compare_n_tokens",
        help=(
            "Number of top-KL-divergence response tokens to compute graph loss on. "
            "Incompatible with --teacher-data-cache."
        ),
    )
    parser.add_argument(
        "--debug-compare-teacher-graphs",
        action="store_true",
        default=False,
        dest="debug_compare_teacher_graphs",
        help=(
            "Diagnostic: build the teacher graph live for every prompt and compare it to "
            "the cached version. Logs supergraph labels, adjacency matrix stats, per-label "
            "row diffs, and graph loss from each teacher. Requires --teacher-data-cache and "
            "loads the teacher model regardless. Useful for pinpointing differences between "
            "cached and live teacher graphs."
        ),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.dataset is None:
        raise SystemExit("--dataset is required.")
    if args.steps < 0:
        raise SystemExit("--steps must be >= 0")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.eval_batch_size < 1:
        raise SystemExit("--eval-batch-size must be >= 1")
    if args.step_log_interval < 1:
        raise SystemExit("--step-log-interval must be >= 1")
    if args.lambda_graph < 0:
        raise SystemExit("--lambda-graph must be >= 0")
    if not (0.0 < args.teacher_prop_neurons_per_layer <= 1.0):
        raise SystemExit("--teacher-prop-neurons-per-layer must be in (0, 1]")
    if not (0.0 < args.student_prop_neurons_per_layer <= 1.0):
        raise SystemExit("--student-prop-neurons-per-layer must be in (0, 1]")
    if args.teacher_graph_batch_size < 1:
        raise SystemExit("--teacher-graph-batch-size must be >= 1")
    if args.student_graph_batch_size < 1:
        raise SystemExit("--student-graph-batch-size must be >= 1")



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    train_path, test_path, dataset_prefix = resolve_train_test_paths(
        dataset=args.dataset,
        datasets_dir=args.datasets_dir,
    )
    run_dir, student_source = resolve_distillation_run_dir(
        os.path.abspath(args.save_dir),
        resume=args.resume_step is not None,
        checkpoint_run=args.checkpoint_run,
    )
    if args.resume_step is not None and not args.checkpoint_run:
        # checkpoint_run already resolved student_source directly; only build the
        # step path when the user pointed at a run dir without a specific checkpoint.
        step_ckpt = os.path.join(run_dir, f"step_{args.resume_step}_checkpoint")
        if not os.path.isdir(step_ckpt):
            raise SystemExit(f"No checkpoint found at {step_ckpt}")
        student_source = find_student_source(step_ckpt)
        if student_source is None:
            raise SystemExit(f"No student weights found in step checkpoint {step_ckpt}")
    if args.resume_step is not None:
        print(f"Resuming from step {args.resume_step} checkpoint: {student_source}")
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 60)
    print("Configuration")
    print("=" * 60)
    print(f"  dataset (prefix):   {dataset_prefix}")
    print(f"  train_path:         {train_path}")
    print(f"  test_path:          {test_path}")
    print(f"  lambda_graph:       {args.lambda_graph}")
    print(f"  graph_loss_type:    {args.graph_loss_type}")
    print(f"  graph dtype:        {args.dtype}")
    print(f"  teacher cache:      {args.teacher_data_cache or 'none'}")
    print(f"  save_dir:           {run_dir}")
    print("=" * 60)

    train_data = load_prompt_answer_json(train_path)
    test_data = load_prompt_answer_json(test_path)
    if args.train_start_idx is not None:
        train_data = dict(list(train_data.items())[args.train_start_idx :])
    if args.train_limit is not None:
        train_data = dict(list(train_data.items())[: args.train_limit])
    if args.test_limit is not None:
        test_data = dict(list(test_data.items())[: args.test_limit])
    extra_eval_data: Dict[str, Dict[str, int]] = {}
    if args.eval_datasets:
        for eval_dataset in args.eval_datasets:
            eval_test_path, eval_prefix = resolve_test_path(
                dataset=eval_dataset,
                datasets_dir=args.datasets_dir,
            )
            extra_eval_data[eval_prefix] = load_prompt_answer_json(eval_test_path)
    print(f"  Train: {len(train_data)} examples")
    print(f"  Test:  {len(test_data)} examples")

    device = get_default_device()
    student, tokenizer = load_student_model_for_distillation(
        student_source,
        args.student_model,
        device,
    )

    trainer = DistillationTrainer(
        config=DistillationConfig(
            teacher_model=args.teacher_model,
            student_model=args.student_model,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            temperature=args.temperature,
            grad_clip=args.grad_clip,
            lambda_graph=args.lambda_graph,
            lambda_kl=args.lambda_kl,
            graph_dtype=resolve_torch_dtype(args.dtype),
            teacher_prop_neurons_per_layer=args.teacher_prop_neurons_per_layer,
            student_prop_neurons_per_layer=args.student_prop_neurons_per_layer,
            graph_max_neurons_per_layer=args.graph_max_neurons_per_layer,
            graph_top_k_logits=args.graph_top_k_logits if args.graph_top_k_logits > 0 else None,
            teacher_graph_batch_size=args.teacher_graph_batch_size,
            student_graph_batch_size=args.student_graph_batch_size,
            graph_verbose=args.verbose,

            graph_grad_norm_scale=args.graph_grad_norm_scale,
            track_loss_grads=args.track_loss_grads,
            eval_max_new_tokens=(
                args.eval_max_new_tokens
                if args.eval_max_new_tokens is not None
                else EVAL_MAX_NEW_TOKENS
            ),
            eval_batch_size=args.eval_batch_size,
            save_dir=run_dir,
            teacher_data_cache=args.teacher_data_cache,
            student_dataset=args.student_dataset,
            student_anova_range_radius=args.student_anova_range_radius,
            student_anova_nodes_per_label=args.student_anova_nodes_per_label,
            student_sum_min_specificity=args.student_sum_min_specificity,
            teacher_anova_range_radius=args.teacher_anova_range_radius,
            teacher_anova_nodes_per_label=args.teacher_anova_nodes_per_label,
            teacher_sum_min_specificity=args.teacher_sum_min_specificity,
            step_log_interval=args.step_log_interval,
            save_interval=args.save_interval,
            label_refresh_interval=args.label_refresh_interval,
            label_refresh_n_prompts=args.label_refresh_n_prompts,
            mlp_cache_refresh_interval=args.mlp_cache_refresh_interval,
            mlp_cache_batch_size=args.mlp_cache_batch_size,
            graph_loss_type=args.graph_loss_type,
            graph_node_labels=args.graph_node_labels,
            tokens_dla_nodes=args.tokens_dla_nodes,
            compare_n_tokens=args.compare_n_tokens,
            skip_baseline_eval=args.skip_baseline_eval,
            debug_compare_teacher_graphs=args.debug_compare_teacher_graphs,

            seed=args.seed,
            device=device,
        ),
        train_data=train_data,
        test_data=test_data,
        extra_eval_data=extra_eval_data,
        tokenizer=tokenizer,
        student=student,
        resume_step=args.resume_step,
    )
    history = trainer.train()
    if "accuracy" in history and history["accuracy"]:
        print(f"Best accuracy: {max(history['accuracy']):.4f}")


if __name__ == "__main__":
    main()

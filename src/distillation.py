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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from graph_loss.utils import DTYPE_CHOICES, resolve_torch_dtype
from utils import (
    EVAL_MAX_NEW_TOKENS,
    LLAMA_1B_MODEL_NAME,
    LLAMA_8B_MODEL_NAME,
    STUDENT_MODEL_DIR,
    evaluate_prompt_answer_dict,
    get_default_device,
    json_to_prompt_answer_dict,
    load_model,
    load_prompt_answer_json,
    load_student_model_for_distillation,
    most_recent_subdirectory,
    patch_tokenizer_no_special_tokens,
    resolve_test_path,
    resolve_train_test_paths,
    rm_dir_tree,
    seed_all,
)
from utils.config import STUDENT_WEIGHTS_FILE

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

    def __init__(self, data: Union[str, Dict[str, int]], tokenizer):
        if isinstance(data, str):
            with open(data, "r", encoding="utf-8") as f:
                raw = json.load(f)
            data = json_to_prompt_answer_dict(raw)
        self.samples = []
        for prompt, answer in data.items():
            answer_text = str(answer)
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
                    "answer": int(answer),
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
        "answers": [int(ex["answer"]) for ex in examples],
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
    graph_top_k_logits: Optional[int] = 20
    graph_prop_neurons_per_layer: float = 0.1
    teacher_graph_batch_size: int = 512
    student_graph_batch_size: int = 1
    graph_verbose: bool = False
    graph_prune: bool = False
    graph_node_threshold: float = 0.8
    graph_edge_threshold: float = 0.98
    student_activation_forward_batch_size: int = 32
    student_dataset: Optional[str] = None
    student_anova_range_radius: int = 0
    student_anova_nodes_per_label: int = 10
    student_min_specificity: float = 0.0
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
    graph_node_weight: float = 1.0
    graph_edge_weight: float = 0.0
    graph_similarity_threshold: float = 0.7
    graph_max_fan_out: int = 4
    fast_teacher_graph: bool = False
    student_computation_eps: float = 1e-6
    student_embedding_eps: float = 0.0
    student_skip_logit_attribution: bool = False
    align_diagnostic: bool = False
    graph_focus_weight: float = 0.0
    graph_grad_mode: str = "approx"
    graph_true_grad_chunk_size: int = 4
    fast_student_graph: bool = False
    ablation_batch_size: int = 1
    align_by_label: bool = False
    student_activation_write_cache_path: Optional[str] = None
    student_mlp_input_cache_path: Optional[str] = None
    mlp_cache_refresh_interval: int = 0
    mlp_cache_batch_size: int = 64



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
    base = os.path.join(save_dir, sub) if sub else save_dir

    if not resume:
        return base, None

    if checkpoint_run:
        cr = checkpoint_run.replace("\\", "/").strip("/")
        if sub and not cr.startswith(f"{sub}/"):
            cr = f"{sub}/{cr}"
        run_dir = os.path.join(save_dir, cr)
    else:
        hf_here = os.path.join(base, STUDENT_MODEL_DIR)
        wt_here = os.path.join(base, STUDENT_WEIGHTS_FILE)
        if os.path.isdir(hf_here) or os.path.isfile(wt_here):
            run_dir = base
        else:
            run_dir = most_recent_subdirectory(base)
            if run_dir is None:
                raise SystemExit(
                    f"No checkpoints in {base} and no run subfolders.\n"
                    "Train here first or pass --checkpoint-run <path under --save-dir>."
                )
            print(f"Auto-detected most recent run folder: {run_dir}")

    hf_path = os.path.join(run_dir, STUDENT_MODEL_DIR)
    wt_path = os.path.join(run_dir, STUDENT_WEIGHTS_FILE)
    if os.path.isdir(hf_path):
        student_source = hf_path
        print(f"Loading student from {student_source}")
    elif os.path.isfile(wt_path):
        student_source = wt_path
        print(f"Loading student weights from {student_source} (fast checkpoint)")
    else:
        raise SystemExit(
            f"Resume expected saved weights at {hf_path} or {wt_path}. "
            "Train a run first."
        )
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
        resume: bool = False,
    ) -> None:
        self.config = config
        self.test_data = test_data
        self.extra_eval_data = extra_eval_data or {}
        self.device = torch.device(config.device)
        self._resume = resume
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

        self.teacher = None
        self.teacher_graph_model = None
        if self.teacher_data_cache is not None:
            print(f"Using cached teacher data: {config.teacher_data_cache}")
        elif self._use_graph:
            raise RuntimeError(
                "Graph loss requires --teacher-data-cache. "
                "Run graph_loss/generate_teacher_data.py first, "
                "or set --lambda-graph 0 to disable graph loss."
            )
        else:
            print(f"Loading teacher: {config.teacher_model}")
            self.teacher, _ = load_model(config.teacher_model)
            self.teacher = self.teacher.to(self.device)
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False

        self.graph_loss_config = None
        self.student_graph_adapter = None
        if self._use_graph:
            from graph_loss.hf_adapter import HFLlamaGraphAdapter
            from graph_loss.training import GraphAuxConfig

            self.graph_loss_config = GraphAuxConfig(
                lambda_graph=config.lambda_graph,
                graph_dtype=config.graph_dtype,
                top_k_logits=config.graph_top_k_logits,
                prop_neurons_per_layer=config.graph_prop_neurons_per_layer,
                teacher_graph_batch_size=config.teacher_graph_batch_size,
                student_graph_batch_size=config.student_graph_batch_size,
                verbose=config.graph_verbose,
                graph_prune=config.graph_prune,
                graph_node_threshold=config.graph_node_threshold,
                graph_edge_threshold=config.graph_edge_threshold,
                student_activation_forward_batch_size=(
                    config.student_activation_forward_batch_size
                ),
                student_anova_range_radius=config.student_anova_range_radius,
                student_anova_nodes_per_label=config.student_anova_nodes_per_label,
                student_min_specificity=config.student_min_specificity,
                dataset=config.student_dataset,
                student_activation_write_cache_path=config.student_activation_write_cache_path,
            )

            self.student_graph_adapter = HFLlamaGraphAdapter(
                self.student,
                self.tokenizer,
                self.device,
            )

            # Build or load MLP input cache at startup so live_anova never
            # falls back to per-prompt disk I/O during training.
            if config.student_mlp_input_cache_path and os.path.isfile(
                config.student_mlp_input_cache_path
            ):
                mlp_cache = torch.load(
                    config.student_mlp_input_cache_path, map_location="cpu"
                )
                self.graph_loss_config.mlp_input_cache = mlp_cache
                n_prompts = int(mlp_cache.get("meta", {}).get("n_prompts", 0))
                print(
                    f"  Loaded MLP input cache from {config.student_mlp_input_cache_path} "
                    f"({n_prompts} prompts)"
                )
            elif config.student_dataset and not config.label_refresh_interval:
                # Only build here when label_refresh_interval==0; otherwise
                # train() will call _refresh_student_labels() at startup.
                # Use the full dataset path so ANOVA has sufficient statistical
                # coverage (~10k prompts).  On an A100 this takes ~3 seconds.
                from graph_loss.neuron_activation_heatmap import _resolve_dataset_path
                from graph_loss.precompute_mlp_inputs import build_mlp_input_cache

                dataset_path = _resolve_dataset_path(config.student_dataset)
                cache_root = os.path.join("/tmp", "mlp_input_cache")
                self.student.eval()
                mlp_cache = build_mlp_input_cache(
                    self.student_graph_adapter,
                    dataset_path,
                    config.student_model,
                    cache_root=cache_root,
                    batch_size=config.mlp_cache_batch_size,
                )
                self.student.train()
                self.graph_loss_config.mlp_input_cache = mlp_cache
                n_prompts = int(mlp_cache.get("meta", {}).get("n_prompts", 0))
                print(
                    f"  Built MLP input cache at startup: {n_prompts} prompts, "
                    f"{len(mlp_cache.get('layer_inputs', []))} layers"
                )

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

        loader_generator = torch.Generator()
        loader_generator.manual_seed(config.seed)
        self.loader = DataLoader(
            AddDataset(self.train_data, self.tokenizer),
            batch_size=config.batch_size,
            shuffle=True,
            generator=loader_generator,
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
        cache_root = os.path.join("/tmp", "mlp_input_cache")
        self.student.eval()
        mlp_cache = build_mlp_input_cache(
            self.student_graph_adapter,
            dataset_path,
            self.config.student_model,
            cache_root=cache_root,
            batch_size=self.config.mlp_cache_batch_size,
            overwrite=True,
        )
        self.student.train()
        self.graph_loss_config.mlp_input_cache = mlp_cache
        # Invalidate the cross-step activation-write cache: the student weights
        # have changed, so previously computed activation grids are stale.
        self.graph_loss_config.activation_write_result_cache.clear()
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
        if self.teacher_data_cache is None:
            raise RuntimeError("Graph loss requires --teacher-data-cache.")
        if self.graph_loss_config is None:
            raise RuntimeError("Graph loss config was not initialized.")
        self._clear_cuda_cache()
        from graph_loss.training import backward_batch_graph_loss

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

        graph_loss, graph_metrics = backward_batch_graph_loss(
            prompts=batch["prompts"],
            student_adapter=self.student_graph_adapter,
            config=self.graph_loss_config,
            device=self.device,
            loss_scale=1.0 if use_grad_norm_scale else self.config.lambda_graph,
            teacher_cache=self.teacher_data_cache,
            answers=graph_answers,
        )

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

        graph_weighted = self.config.lambda_graph * graph_loss
        total = non_graph_loss.detach() + graph_weighted
        metrics["graph_loss"] = float(graph_loss.item())
        metrics["graph_loss_weighted"] = float(graph_weighted.item())
        metrics["total_loss"] = float(total.item())
        for key, value in graph_metrics.items():
            if isinstance(value, (int, float)):
                metrics[f"graph_{key}"] = float(value)
        return total

    def _record_step_metrics(self, epoch: int, batch_step: int, metrics: Dict[str, float]) -> None:
        self._train_step += 1
        self.history["train_step"].append(self._train_step)
        self.history["train_epoch"].append(epoch + 1)
        self.history["train_batch"].append(batch_step)
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
        self.history["eval_train_step"].append(int(self._train_step))
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
                    weighted_avg = interval.get("graph_loss_weighted", 0.0) / denom
                    n_prompts = interval.get("graph_graph_prompts", 0.0)
                    n_back = interval.get("graph_graph_backward_prompts", 0.0)
                    active_pct = (100.0 * n_back / n_prompts) if n_prompts > 0 else 0.0
                    graph_s = (
                        f" | Graph {graph_avg:.4f}"
                        f" | lambdaGraph {weighted_avg:.4f}"
                        f" | GraphActive {active_pct:.0f}%"
                    )
                    if self.config.graph_grad_norm_scale:
                        eff_lam_key = "graph_grad_norm_scale_effective_lambda"
                        eff_lam_avg = interval.get(eff_lam_key, 0.0) / denom
                        graph_s += f" | effLambda {eff_lam_avg:.4f}"
                    t_sn = interval.get("graph_teacher_supernodes", 0.0) / denom
                    s_sn = interval.get("graph_student_supernodes", 0.0) / denom
                    mapped = interval.get("graph_aligned_teacher_supernodes", 0.0) / denom
                    if t_sn > 0 or s_sn > 0:
                        graph_s += (
                            f" | align T{t_sn:.1f}/S{s_sn:.1f}"
                            f" mapped={mapped:.1f}"
                        )
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
            start_epoch = len(self.history.get("epoch", []))
            self._train_step = len(self.history.get("train_step", []))
            self._step_log_eval_accuracy = (
                float(self.history["accuracy"][-1])
                if self.history.get("accuracy")
                else 0.0
            )
            print(f"Warm-starting from step {self._train_step + 1}.")
        else:
            print("Evaluating baselines...")
            student_base = evaluate_prompt_answer_dict(
                self.student,
                self.tokenizer,
                self.test_data,
                batch_size=cfg.eval_batch_size,
                max_new_tokens=cfg.eval_max_new_tokens,
            )
            self.history["student_baseline"] = float(student_base)
            if self.teacher is not None:
                teacher_base = evaluate_prompt_answer_dict(
                    self.teacher,
                    self.tokenizer,
                    self.test_data,
                    batch_size=cfg.eval_batch_size,
                    max_new_tokens=cfg.eval_max_new_tokens,
                )
            else:
                teacher_base = 0.0
            self.history["teacher_baseline"] = float(teacher_base)
            self._step_log_eval_accuracy = float(student_base)
            print(f"  Student baseline accuracy: {student_base:.4f}")
            if self.teacher is not None:
                print(f"  Teacher baseline accuracy: {teacher_base:.4f}")
            else:
                print("  Teacher baseline accuracy: skipped")

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
            print(f"  graph top_k_logits: {cfg.graph_top_k_logits}")
            print(f"  graph prop neurons/layer: {cfg.graph_prop_neurons_per_layer}")
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
                graph_s = (
                    f", Graph={epoch_metrics.get('graph_loss', float('nan')):.4f}"
                    f", lambdaGraph={epoch_metrics.get('graph_loss_weighted', float('nan')):.4f}"
                )
            print(
                f"Pass {epoch + 1}: "
                f"KL={epoch_metrics.get('kl_loss', float('nan')):.4f}"
                f"{graph_s}, Acc={acc_s}, Step={self._train_step}/{target_step}",
            )
            epoch += 1

        self._save_history()
        self._save_curves()
        self._save_checkpoint()
        print(f"  Saved {STUDENT_MODEL_DIR}/ (final)")
        print(f"Results saved to: {cfg.save_dir}")
        return dict(self.history)

    def _save_checkpoint(self) -> None:
        path = os.path.join(self.config.save_dir, STUDENT_MODEL_DIR)
        rm_dir_tree(path)
        os.makedirs(path, exist_ok=True)
        self.student.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def _save_checkpoint_at_step(self, step: int) -> None:
        step_dir = os.path.join(self.config.save_dir, f"checkpoint_step_{step:04d}")
        path = os.path.join(step_dir, STUDENT_MODEL_DIR)
        rm_dir_tree(path)
        os.makedirs(path, exist_ok=True)
        self.student.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"  [checkpoint] Saved step {step:04d} → {path}")

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
        epochs = self.history.get("epoch", [])
        if not epochs:
            return
        loss_steps = self.history.get("train_step") or epochs
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
        if acc_series:
            acc_ax.plot(acc_x[: len(acc_series)], acc_series, marker="o", markersize=3)
        acc_ax.set_title("Test Accuracy")
        acc_ax.set_ylim(0, 1)
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        os.makedirs(self.config.save_dir, exist_ok=True)
        out = os.path.join(self.config.save_dir, "training_curves.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved training curves to {out}")


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
    parser.add_argument("--graph-top-k-logits", type=int, default=20)
    parser.add_argument("--graph-prop-neurons-per-layer", type=float, default=0.1)
    parser.add_argument("--teacher-graph-batch-size", type=int, default=512)
    parser.add_argument("--student-graph-batch-size", type=int, default=1)
    parser.add_argument(
        "--student-activation-forward-batch-size",
        type=int,
        default=32,
        help="Batch size for student ablation forwards during supernode clustering.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--graph-prune", action="store_true")
    parser.add_argument("--graph-node-threshold", type=float, default=0.8)
    parser.add_argument("--graph-edge-threshold", type=float, default=0.98)
    parser.add_argument("--graph-node-weight", type=float, default=1.0)
    parser.add_argument("--graph-edge-weight", type=float, default=0.0)
    parser.add_argument(
        "--graph-focus-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for Phase-3 logit-focus distribution loss "
            "(KL(teacher_focus || student_focus), no alignment required). "
            "Set >0 to enable, e.g. --graph-focus-weight 1.0. "
            "When this is the only active graph loss term, also set "
            "--graph-node-weight 0 --graph-edge-weight 0."
        ),
    )
    parser.add_argument("--graph-similarity-threshold", type=float, default=0.7)
    parser.add_argument("--graph-max-fan-out", type=int, default=4)
    parser.add_argument("--fast-teacher-graph", action="store_true")
    parser.add_argument(
        "--graph-grad-mode",
        type=str,
        default="approx",
        choices=["approx", "true"],
        help=(
            "Gradient pathway for per-supernode prob-delta loss.  "
            "'approx' (default): cheap differentiable DLA-approximation, "
            "one forward + per-supernode (norm/lm_head/softmax).  Fast, "
            "~95%% directionally faithful.  "
            "'true': full hook-based ablation with chunked sequential "
            "backward, 100%% faithful to teacher's operational space at "
            "~3-5x training cost.  Tune chunk size with --graph-true-grad-chunk-size."
        ),
    )
    parser.add_argument(
        "--graph-true-grad-chunk-size",
        type=int,
        default=4,
        help=(
            "In --graph-grad-mode true, # of supernodes per batched "
            "ablation forward+backward.  Higher = more parallelism, "
            "more peak memory.  Default 4."
        ),
    )
    parser.add_argument(
        "--fast-student-graph",
        action="store_true",
        help=(
            "Skip the [neurons, neurons] Jacobian backward passes when "
            "building the student graph.  Selects neurons + builds a "
            "DLA-only adjacency, then clusters via real ablation.  "
            "Required when running node-loss-only (much faster).  "
            "Must be combined with --graph-edge-weight 0."
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
    parser.add_argument(
        "--align-diagnostic",
        action="store_true",
        help=(
            "Log per-prompt teacher/student supernode cosine-matrix stats "
            "(mean/median/max, fraction of teacher SNs with any match >= 0.3) "
            "to verify alignment quality before/after a fix."
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
        "--graph-align-by-label",
        action="store_true",
        help=(
            "Align teacher↔student supernodes by ANOVA label string instead of "
            "cosine similarity of prob-delta vectors.  Requires teacher cache built "
            "with full_search (labels saved)."
        ),
    )
    parser.add_argument(
        "--student-dataset",
        type=str,
        default=None,
        metavar="DATASET",
        help="Activation dataset for student full_search clustering (e.g. 22_add_tight_all.json).",
    )
    parser.add_argument(
        "--student-activation-write-cache",
        type=str,
        default=None,
        metavar="PATH",
        help="Local path for caching student activation-write grids between steps.",
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
        "--student-min-specificity",
        type=float,
        default=0.0,
        help="Minimum ANOVA specificity for student supernodes.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "results", "distillation"),
    )
    parser.add_argument("--resume", action="store_true")
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
    parser.add_argument("--student-computation-eps", type=float, default=1e-6)
    parser.add_argument("--student-embedding-eps", type=float, default=0.0)
    parser.add_argument(
        "--student-skip-logit-attribution",
        action="store_true",
        help=(
            "Skip Jacobian logit-influence rows in the student graph (saves memory "
            "but alignment signal is sum-of-DLAs, not prob-deltas)."
        ),
    )
    parser.add_argument(
        "--ablation-batch-size",
        type=int,
        default=1,
        help="Batch size for ablation experiments.",
    )
    parser.add_argument(
        "--student-mlp-input-cache",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a pre-built MLP input cache (.pt file).  If provided and the "
            "file exists it is loaded at startup.  If not provided (or file not "
            "found) and --student-dataset is set, the cache is built from the "
            "student model at startup."
        ),
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
    if args.graph_top_k_logits is not None and args.graph_top_k_logits <= 0:
        raise SystemExit("--graph-top-k-logits must be positive")
    if not (0.0 < args.graph_prop_neurons_per_layer <= 1.0):
        raise SystemExit("--graph-prop-neurons-per-layer must be in (0, 1]")
    if args.teacher_graph_batch_size < 1:
        raise SystemExit("--teacher-graph-batch-size must be >= 1")
    if args.student_graph_batch_size < 1:
        raise SystemExit("--student-graph-batch-size must be >= 1")
    if args.student_activation_forward_batch_size < 1:
        raise SystemExit("--student-activation-forward-batch-size must be >= 1")
    if not (0.0 <= args.graph_node_threshold <= 1.0):
        raise SystemExit("--graph-node-threshold must be in [0, 1]")
    if not (0.0 <= args.graph_edge_threshold <= 1.0):
        raise SystemExit("--graph-edge-threshold must be in [0, 1]")


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
        resume=args.resume,
        checkpoint_run=args.checkpoint_run,
    )
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 60)
    print("Configuration")
    print("=" * 60)
    print(f"  dataset (prefix):   {dataset_prefix}")
    print(f"  train_path:         {train_path}")
    print(f"  test_path:          {test_path}")
    print(f"  lambda_graph:       {args.lambda_graph}")
    print(f"  graph node weight:  {args.graph_node_weight}")
    print(f"  graph edge weight:  {args.graph_edge_weight}")
    print(f"  graph focus weight: {args.graph_focus_weight}")
    print(f"  graph grad mode:    {args.graph_grad_mode}", end="")
    if args.graph_grad_mode == "true":
        print(f" (chunk_size={args.graph_true_grad_chunk_size})")
    else:
        print()
    if args.fast_student_graph:
        print(f"  fast student graph: True (node-loss-only fast path)")
    print(f"  graph dtype:        {args.dtype}")
    print(f"  teacher cache:      {args.teacher_data_cache or 'none'}")
    print(f"  save_dir:           {run_dir}")
    print("=" * 60)

    train_data = load_prompt_answer_json(train_path)
    if args.train_limit is not None:
        train_data = dict(list(train_data.items())[: args.train_limit])
    test_data = load_prompt_answer_json(test_path)
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
            graph_top_k_logits=args.graph_top_k_logits,
            graph_prop_neurons_per_layer=args.graph_prop_neurons_per_layer,
            teacher_graph_batch_size=args.teacher_graph_batch_size,
            student_graph_batch_size=args.student_graph_batch_size,
            graph_verbose=args.verbose,
            graph_prune=args.graph_prune,
            graph_node_threshold=args.graph_node_threshold,
            graph_edge_threshold=args.graph_edge_threshold,
            graph_node_weight=args.graph_node_weight,
            graph_edge_weight=args.graph_edge_weight,
            graph_similarity_threshold=args.graph_similarity_threshold,
            graph_max_fan_out=args.graph_max_fan_out,
            fast_teacher_graph=args.fast_teacher_graph,
            student_computation_eps=args.student_computation_eps,
            student_embedding_eps=args.student_embedding_eps,
            student_activation_forward_batch_size=args.student_activation_forward_batch_size,
            student_skip_logit_attribution=args.student_skip_logit_attribution,
            align_diagnostic=args.align_diagnostic,
            graph_focus_weight=args.graph_focus_weight,
            graph_grad_mode=args.graph_grad_mode,
            graph_true_grad_chunk_size=args.graph_true_grad_chunk_size,
            fast_student_graph=args.fast_student_graph,
            ablation_batch_size=args.ablation_batch_size,
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
            align_by_label=args.graph_align_by_label,
            student_dataset=args.student_dataset,
            student_activation_write_cache_path=args.student_activation_write_cache,
            student_anova_range_radius=args.student_anova_range_radius,
            student_anova_nodes_per_label=args.student_anova_nodes_per_label,
            student_min_specificity=args.student_min_specificity,
            step_log_interval=args.step_log_interval,
            save_interval=args.save_interval,
            label_refresh_interval=args.label_refresh_interval,
            label_refresh_n_prompts=args.label_refresh_n_prompts,
            student_mlp_input_cache_path=args.student_mlp_input_cache,
            mlp_cache_refresh_interval=args.mlp_cache_refresh_interval,
            mlp_cache_batch_size=args.mlp_cache_batch_size,

            seed=args.seed,
            device=device,
        ),
        train_data=train_data,
        test_data=test_data,
        extra_eval_data=extra_eval_data,
        tokenizer=tokenizer,
        student=student,
        resume=student_source is not None,
    )
    history = trainer.train()
    if "accuracy" in history and history["accuracy"]:
        print(f"Best accuracy: {max(history['accuracy']):.4f}")


if __name__ == "__main__":
    main()

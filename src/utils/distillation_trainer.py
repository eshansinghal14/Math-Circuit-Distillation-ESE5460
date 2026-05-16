import gc
import importlib
import json
import math
import os
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import STUDENT_MODEL_DIR
from .device import seed_all
from .distillation_batch import AddDataset, collate_fn, masked_kl_loss
from .distillation_run import DistillationConfig
from .eval_inference import evaluate_prompt_answer_dict
from .fs import rm_dir_tree
from .hf_models import load_model, load_student_model_for_distillation, patch_tokenizer_no_special_tokens


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

        teacher_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.teacher = None
        self.teacher_graph_model = None
        if self.teacher_data_cache is not None:
            print(f"Using cached teacher data: {config.teacher_data_cache}")
        elif self._use_graph:
            from graph_loss.replacement_model import TransformerLensReplacementModel

            print(f"Loading teacher graph model: {config.teacher_model}")
            self.teacher_graph_model = TransformerLensReplacementModel.from_pretrained(
                config.teacher_model,
                device=self.device,
                dtype=config.graph_dtype or teacher_dtype,
            )
            self.teacher_graph_model.eval()
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
                graph_gen_batch_size=config.graph_gen_batch_size,
                teacher_graph_batch_size=config.teacher_graph_batch_size,
                student_graph_batch_size=config.student_graph_batch_size,
                verbose=config.graph_verbose,
                graph_prune=config.graph_prune,
                graph_node_threshold=config.graph_node_threshold,
                graph_edge_threshold=config.graph_edge_threshold,
                graph_node_weight=config.graph_node_weight,
                graph_edge_weight=config.graph_edge_weight,
                graph_similarity_threshold=config.graph_similarity_threshold,
                graph_max_fan_out=config.graph_max_fan_out,
                fast_teacher_graph=config.fast_teacher_graph,
                student_computation_eps=config.student_computation_eps,
                student_embedding_eps=config.student_embedding_eps,
                student_activation_forward_batch_size=(
                    config.student_activation_forward_batch_size
                ),
                student_skip_logit_attribution=config.student_skip_logit_attribution,
                align_diagnostic=config.align_diagnostic,
                graph_focus_weight=config.graph_focus_weight,
                graph_grad_mode=config.graph_grad_mode,
                graph_true_grad_chunk_size=config.graph_true_grad_chunk_size,
                fast_student_graph=config.fast_student_graph,
                ablation_batch_size=config.ablation_batch_size,
                align_by_label=config.align_by_label,
                student_cluster_method=config.student_cluster_method,
                student_dataset=config.student_dataset,
                student_mlp_input_cache_path=config.student_mlp_input_cache_path,
                student_activation_write_cache_path=config.student_activation_write_cache_path,
                student_anova_range_radius=config.student_anova_range_radius,
                student_anova_nodes_per_label=config.student_anova_nodes_per_label,
                student_sum_min_specificity=config.student_sum_min_specificity,
                student_fixed_labels_path=config.student_fixed_labels_path,
            )
            # Load the fixed-labels JSON once at startup and attach to the config
            # object so every call to compute_prompt_graph_loss can look it up
            # without re-reading the file.
            if config.student_fixed_labels_path:
                import json as _json
                print(f"[graph] Loading fixed labels from {config.student_fixed_labels_path} …")
                with open(config.student_fixed_labels_path, encoding="utf-8") as _fh:
                    self.graph_loss_config._fixed_labels_cache = _json.load(_fh)
                print(f"[graph] Loaded {len(self.graph_loss_config._fixed_labels_cache)} fixed neuron labels.")
            else:
                self.graph_loss_config._fixed_labels_cache = None

            self.student_graph_adapter = HFLlamaGraphAdapter(
                self.student,
                self.tokenizer,
                self.device,
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
        self._best_eval_accuracy = 0.0

    @staticmethod
    def _load_teacher_data_cache(cache_dir: str):
        from graph_loss.teacher_data_cache import TeacherDataCache

        return TeacherDataCache(cache_dir)

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
        if self.teacher_graph_model is not None:
            with torch.no_grad():
                out = self.teacher_graph_model(input_ids)
                return (out.logits if hasattr(out, "logits") else out).to(self.device)
        if self.teacher is None:
            raise RuntimeError("No teacher model or teacher cache is configured.")
        with torch.no_grad():
            return self.teacher(input_ids=input_ids, attention_mask=attention_mask).logits

    def _forward_kl(self, batch: Dict[str, Any]) -> tuple[torch.Tensor, Dict[str, float]]:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        kl_mask = batch["kl_mask"].to(self.device)
        teacher_logits = self._teacher_logits_for_batch(batch, input_ids, attention_mask)
        student_logits = self.student(input_ids=input_ids, attention_mask=attention_mask).logits

        kl_loss = masked_kl_loss(
            student_logits,
            teacher_logits,
            kl_mask,
            self.config.temperature,
        )
        metrics: Dict[str, float] = {"kl_loss": float(kl_loss.item())}
        metrics["total_loss"] = float(kl_loss.item())
        if self.config.lambda_kl == 0.0:
            # Zero out KL contribution so graph loss is the only gradient signal.
            kl_loss = kl_loss.detach() * 0.0
        elif self.config.lambda_kl != 1.0:
            kl_loss = self.config.lambda_kl * kl_loss
        return kl_loss, metrics

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
        if self.teacher_graph_model is None and self.teacher_data_cache is None:
            raise RuntimeError("Graph loss requires a teacher graph model or teacher cache.")
        if self.graph_loss_config is None:
            raise RuntimeError("Graph loss config was not initialized.")
        self._clear_cuda_cache()
        from graph_loss.training import backward_batch_graph_loss

        use_grad_norm_scale = (
            self.config.graph_grad_norm_scale
            and kl_grad_norm is not None
            and kl_grad_norm > 1e-8
        )

        # When grad-norm scaling is on we snapshot KL grads, run graph backward
        # with loss_scale=1 (no lambda yet), then re-scale the graph-only delta.
        if use_grad_norm_scale:
            kl_grad_snap: Dict[str, Optional[torch.Tensor]] = {
                n: (p.grad.clone() if p.grad is not None else None)
                for n, p in self.student.named_parameters()
            }

        graph_loss, graph_metrics = backward_batch_graph_loss(
            prompts=batch["prompts"],
            teacher_graph_model=self.teacher_graph_model,
            student_adapter=self.student_graph_adapter,
            config=self.graph_loss_config,
            device=self.device,
            loss_scale=1.0 if use_grad_norm_scale else self.config.lambda_graph,
            teacher_cache=self.teacher_data_cache,
            answers=batch.get("answers"),
        )

        if use_grad_norm_scale:
            # Compute graph-only grad norm from the delta, then rescale.
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
                # Graph backward produced no signal; restore KL-only grads.
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

    def _run_training_eval(self, epoch: int, batch_step: int) -> float:
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
        if acc_f > self._best_eval_accuracy:
            self._best_eval_accuracy = acc_f
            if cfg.save_best:
                self._save_checkpoint()
                print(f"  Saved {STUDENT_MODEL_DIR}/ (new best accuracy {acc_f:.4f})")
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
        # Per-interval running sums for diagnostic step prints.  We want the
        # printed Graph/lambdaGraph/active-rate values to reflect *all* batches
        # since the last print, not just the latest batch (whose graph loss is
        # often 0 because that single batch happened to have no matched
        # supernodes — misleading when many earlier batches in the interval
        # did backprop graph gradients).
        interval = defaultdict(float)
        interval_steps = 0
        for step, batch in enumerate(self.loader):
            loss, metrics = self._forward_kl(batch)
            if not torch.isfinite(loss).item():
                skipped_nonfinite += 1
                continue
            # graph_start_step is 1-indexed; _train_step hasn't been incremented yet
            # for the current step, so current step number = _train_step + 1.
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

            # Periodic fixed-label refresh via ANOVA on a small sample of training prompts.
            if (
                self.config.label_refresh_interval > 0
                and self.config.student_cluster_method == "fixed_labels"
                and self._train_step > 0
                and self._train_step % self.config.label_refresh_interval == 0
                and self.config.student_mlp_input_cache_path
                and self.config.student_dataset
            ):
                self._refresh_fixed_labels(self._train_step)

            # Periodic checkpoint save.
            if self.config.save_interval > 0 and self._train_step % self.config.save_interval == 0:
                self._save_checkpoint_at_step(self._train_step)

            if self._train_step == 1 or self._train_step % max(1, self.config.step_log_interval) == 0:
                self._run_training_eval(epoch, step)
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
                    cos = interval.get("graph_mean_alignment_similarity", 0.0) / denom
                    if t_sn > 0 or s_sn > 0:
                        graph_s += (
                            f" | align T{t_sn:.1f}/S{s_sn:.1f}"
                            f" mapped={mapped:.1f} cos={cos:.3f}"
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
            self._best_eval_accuracy = (
                max(self.history["accuracy"]) if self.history.get("accuracy") else 0.0
            )
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
            print(
                f"  Label refresh:    every {cfg.label_refresh_interval} steps "
                f"({cfg.label_refresh_n_prompts} prompts)"
            )
        print("=" * 60)

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

    def _refresh_fixed_labels(self, step: int) -> None:
        """Re-run ANOVA on the *full* neuron set and update _fixed_labels_cache.

        Mirrors :mod:`graph_loss.precompute_fixed_labels_full`: skips any
        forward-pass top-K filtering and instead runs ANOVA over every
        ``(layer, token_pos, neuron_id)`` combination using the pre-computed
        MLP input cache and the student's *current* MLP weights.

        The MLP input cache is stale residual stream + current weights — same
        approximation as before, but the candidate set is no longer constrained
        to neurons that look "important" under a snapshot top-K filter.

        Time per refresh on an A100: ~20-30 min.  Tune
        ``--label-refresh-interval`` accordingly.
        """
        config = self.config
        if self.graph_loss_config is None or self.student_graph_adapter is None:
            return

        if not (
            config.student_cluster_method == "fixed_labels"
            and config.student_mlp_input_cache_path
            and config.student_dataset
        ):
            print(
                "[graph] Warning: --label-refresh-interval has no effect unless "
                "--student-cluster-method=fixed_labels, --student-mlp-input-cache, "
                "and --student-dataset are all set."
            )
            return

        try:
            from graph_loss.anova_node_labels import ANOVA_LABEL_CATEGORIES, label_activation_heatmaps
            from graph_loss.neuron_activation_heatmap import (
                _resolve_dataset_path,
                build_neuron_activation_write_result,
            )
            from graph_loss.precompute_mlp_inputs import (
                load_mlp_input_cache,
                mlp_input_cache_exists,
            )

            dataset_path = _resolve_dataset_path(config.student_dataset)
            _student_model_name = getattr(
                getattr(self.student_graph_adapter.model, "config", None),
                "_name_or_path",
                "unknown_model",
            )
            if mlp_input_cache_exists(
                config.student_mlp_input_cache_path, _student_model_name, dataset_path
            ):
                student_mlp_input_cache = load_mlp_input_cache(
                    config.student_mlp_input_cache_path, _student_model_name, dataset_path
                )
            else:
                import warnings
                from graph_loss.precompute_mlp_inputs import mlp_input_cache_dir
                expected_dir = mlp_input_cache_dir(
                    config.student_mlp_input_cache_path, _student_model_name, dataset_path
                )
                warnings.warn(
                    f"[graph] Step {step}: label refresh skipped — MLP input cache not found at "
                    f"{expected_dir!r}. Set --student-mlp-input-cache to the *parent* of the "
                    f"model-slug directory (e.g. /content/local_caches/mlp-input-cache), not the "
                    f"model-slug subdirectory itself."
                )
                return

            adapter = self.student_graph_adapter
            cache_meta = student_mlp_input_cache["meta"]
            n_cache_positions = int(cache_meta["n_positions"])
            n_layers = adapter.n_layers
            d_mlp = adapter.d_mlp

            # Build full neuron_locations: every (layer, token_pos, neuron_id) triple.
            layers_arange = torch.arange(n_layers, dtype=torch.long)
            positions_arange = torch.arange(n_cache_positions, dtype=torch.long)
            neurons_arange = torch.arange(d_mlp, dtype=torch.long)
            neuron_locations = torch.cartesian_prod(
                layers_arange, positions_arange, neurons_arange
            ).contiguous()
            n_total = int(neuron_locations.shape[0])
            print(
                f"[graph] Step {step}: starting ANOVA on {n_total} total neurons "
                f"(layers={n_layers} × positions={n_cache_positions} × d_mlp={d_mlp}). "
                f"This is the slow step (~20-30 min on A100)."
            )

            # The student's training-mode flag controls dropout etc.; the heatmap
            # builder reads weights only, so eval-mode is a no-op for this model
            # but is still safer (matches precompute_fixed_labels_full semantics).
            self.student.eval()
            try:
                result = build_neuron_activation_write_result(
                    adapter,
                    dataset_path,
                    neuron_locations,
                    mlp_input_cache=student_mlp_input_cache,
                    include_w_down_vectors=False,
                )
            finally:
                self.student.train()

            label_results = label_activation_heatmaps(
                result.activations,
                result.arg_values,
                target_args=None,
                anova_range_radius=config.student_anova_range_radius,
            )

            new_labels: Dict[str, str] = {}
            for category in ANOVA_LABEL_CATEGORIES:
                all_scored = [
                    (row_idx, float(label_result.category_specificity[category]))
                    for row_idx, label_result in enumerate(label_results)
                    if category in label_result.category_specificity
                    and label_result.category_scores.get(category, 0.0) > 0.0
                    and label_result.category_specificity.get(category, float("-inf"))
                    > config.student_sum_min_specificity
                ]
                sorted_rows = sorted(all_scored, key=lambda x: x[1], reverse=True)
                top_rows = sorted_rows[: config.student_anova_nodes_per_label]

                for row_idx, _ in top_rows:
                    layer = int(neuron_locations[row_idx, 0].item())
                    neuron_id = int(neuron_locations[row_idx, 2].item())
                    key = f"{layer}:{neuron_id}"
                    if key not in new_labels:
                        new_labels[key] = label_results[row_idx].categories[category]

            self.graph_loss_config._fixed_labels_cache = new_labels
            print(
                f"[graph] Step {step}: refreshed fixed labels (full) from "
                f"{n_total} candidate neurons → {len(new_labels)} labeled neurons"
            )
        except Exception as exc:
            import warnings

            warnings.warn(
                f"[graph] Step {step}: label refresh failed ({exc!r}), keeping old labels."
            )

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

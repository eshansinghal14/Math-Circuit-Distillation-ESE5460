"""KL + graph distillation — supernodes built automatically (arg-token nodes + DLA node)."""

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

from utils import (
    DataLoader,
    DIR_ROOT,
    PromptAnswerDataset,
    collate_fn,
    eval_model,
    load_data,
    load_model,
    seed_all,
    tokenize_prompt_answer,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_loss.hf_adapter import HFLlamaGraphAdapter
from graph_loss.utils import normalize_node_labels
from graph_loss.training import (
    GraphAuxConfig,
    TeacherTargetCache,
    _compute_teacher_target,
    _teacher_target_entry,
    backward_batch_graph_loss,
    teacher_target_cache_key,
)
from training.utils import (
    eval_batch_size_for,
    DEFAULT_SEED,
    ParamChangeCanary,
    ParamStepTracker,
    add_kd_args,
    add_standard_args,
    describe_run_setup,
    kd_position_mask,
    first_answer_token_accuracy,
    kl_loss,
    load_student,
    log_first_step_canary,
    make_optimizer,
    maybe_save_periodic_checkpoint,
    record_resumed_config,
    apply_lr_schedule,
    resume_checkpoint_dir,
    run_baselines,
    shared_teacher,
    run_seeds,
    resume_training_state,
    run_config_record,
    save_checkpoint,
    refresh_curves,
    save_curves,
    save_history,
    step_flop_counter,
    student_autocast,
)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_GRAD_CLIP = 1.0
_SEED = DEFAULT_SEED


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GraphKDConfig:
    model: str
    teacher: str
    dataset: str
    steps: int = 15
    batch_size: int = 32
    learning_rate: float = 5e-7
    warmup_steps: int = 10
    lr_floor: float = 0.3  # see training.utils.scheduled_lr
    temperature: float = 1.0
    kl_token_chunk_size: int = 64
    max_eval_tokens: Optional[int] = None  # None -> utils.default_eval_tokens(dataset)
    eval_batch_size: Any = 256
    save_dir: str = "results/graph_kd"
    eval_every_n_steps: int = 1
    save_every_n_steps: int = 0
    grad_accum_steps: int = 1
    eval_datasets: List[str] = field(default_factory=list)
    test_limit: Optional[int] = None
    dtype: Optional[str] = None
    resume: bool = False
    seed: int = _SEED
    # graph loss
    lambda_graph: float = 1.0
    lambda_kl: float = 1.0
    teacher_prop_neurons_per_layer: float = 0.003
    student_prop_neurons_per_layer: float = 0.01
    nodes_per_label: int = 10
    graph_loss_type: str = "jsd"
    freeze_attention: bool = False
    freeze_rms_norm: bool = False
    constant_node_weighting: bool = False
    supergraph_aggregation: str = "normalised"
    token_source_columns: bool = False
    token_path_rows: str = "weighted"
    token_path_micro_batch: int = 4
    top_k_logits: float = 0.95
    teacher_graph_batch_size: int = 512
    student_graph_batch_size: int = 1
    n_graph_prompts: Optional[int] = None
    graph_verbose: bool = False
    track_grad_metrics: bool = False
    track_flops: bool = False
    scramble_teacher_graph: bool = False
    graph_node_labels: List[str] = field(default_factory=list)
    anova_range_radius: int = 0
    mlp_cache_batch_size: int = 32
    anova_neuron_chunk: int | None = None
    # Performance only; neither changes any number a run produces.
    anova_cache_device: str = "auto"
    teacher_target_cache_dir: str | None = "cache/teacher_targets"


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────


class GraphKDTrainer:
    def __init__(
        self,
        config: GraphKDConfig,
        train_data: Dict[str, Any],
        test_data: Dict[str, Any],
        shared: Dict[str, Any] | None = None,
    ) -> None:
        """``shared`` is run_seeds' cross-seed dict (teacher, baselines); None loads everything."""
        self.config = config
        self.shared = shared
        seed_all(config.seed)
        # 'tokens' asks for the token-embedding source columns and is not an ANOVA
        # category; with no real category left there is no ANOVA and no MLP-input
        # cache to build. Done here as well as in main() so a config built directly
        # (a notebook) behaves the same.
        config.graph_node_labels, tokens_label = normalize_node_labels(config.graph_node_labels)
        config.token_source_columns = config.token_source_columns or tokens_label

        # fp32 master weights with a bf16 autocast forward, shared with SFT and
        # standard KD so the three are comparable. The graph term is the reason this
        # matters most here: backward_batch_graph_loss backprops one prompt at a
        # time, and in bf16 those ~32 small adds into a .grad already holding the
        # KD gradient were mostly rounded away. See training/utils.py. The teacher
        # is inference-only and stays bf16.
        # --resume loads the weights the periodic checkpoint saved instead.
        student_src = resume_checkpoint_dir(config.save_dir) if config.resume else config.model
        self.model, self.tokenizer = load_student(student_src, getattr(config, "dtype", None))

        self.teacher = shared_teacher(shared, config.teacher, load_model)

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

        self.student_adapter = HFLlamaGraphAdapter(self.model, self.tokenizer, _DEVICE)
        self.teacher_adapter = HFLlamaGraphAdapter(self.teacher, self.tokenizer, _DEVICE)

        student_mlp_cache: dict | None = None
        teacher_mlp_cache: dict | None = None
        if config.graph_node_labels:
            # Built from the untrained student and the teacher, so seed-independent:
            # under run_seeds they are built for the first seed and reused.
            if shared is not None and "mlp_caches" in shared:
                print("MLP-input caches reused from the previous seed.")
                student_mlp_cache, teacher_mlp_cache = shared["mlp_caches"]
            else:
                from graph_loss.precompute_mlp_inputs import build_mlp_input_cache as _build_mlp_cache
                student_dev, teacher_dev = _anova_cache_devices(config.anova_cache_device)
                student_mlp_cache = _build_mlp_cache(
                    self.student_adapter, config.dataset, config.model,
                    data_dict=train_data, batch_size=config.mlp_cache_batch_size,
                    cache_device=student_dev,
                )
                teacher_mlp_cache = _build_mlp_cache(
                    self.teacher_adapter, config.dataset, config.teacher,
                    data_dict=train_data, batch_size=config.mlp_cache_batch_size,
                    cache_device=teacher_dev,
                )
                print(
                    f"MLP-input caches: student {_cache_gb(student_mlp_cache):.2f} GB on {student_dev}, "
                    f"teacher {_cache_gb(teacher_mlp_cache):.2f} GB on {teacher_dev}"
                )
                if shared is not None:
                    shared["mlp_caches"] = (student_mlp_cache, teacher_mlp_cache)

        self.graph_config = GraphAuxConfig(
            lambda_graph=config.lambda_graph,
            graph_dtype=torch.bfloat16,
            teacher_prop_neurons_per_layer=config.teacher_prop_neurons_per_layer,
            student_prop_neurons_per_layer=config.student_prop_neurons_per_layer,
            top_k_logits=config.top_k_logits,
            temperature=config.temperature,
            teacher_graph_batch_size=config.teacher_graph_batch_size,
            student_graph_batch_size=config.student_graph_batch_size,
            student_nodes_per_label=config.nodes_per_label,
            teacher_nodes_per_label=config.nodes_per_label,
            graph_loss_type=config.graph_loss_type,
            freeze_attention=config.freeze_attention,
            freeze_rms_norm=config.freeze_rms_norm,
            constant_node_weighting=config.constant_node_weighting,
            supergraph_aggregation=config.supergraph_aggregation,
            token_source_columns=config.token_source_columns,
            token_path_rows=config.token_path_rows,
            token_path_micro_batch=config.token_path_micro_batch,
            verbose=config.graph_verbose,
            mlp_input_cache=student_mlp_cache,
            teacher_mlp_input_cache=teacher_mlp_cache,
            graph_node_labels=config.graph_node_labels if config.graph_node_labels else None,
            student_anova_range_radius=config.anova_range_radius,
            anova_neuron_chunk=config.anova_neuron_chunk,
            dataset_name=config.dataset,
            scramble_teacher_graph=config.scramble_teacher_graph,
            scramble_seed=config.seed,
        )

        # The teacher's per-prompt target is a pure function of the teacher-side
        # configuration, so it is built once per prompt and shared by every run
        # with that configuration (all seeds, every lambda, every control).
        self._step_tracker = ParamStepTracker(self.model)
        self.teacher_target_cache: TeacherTargetCache | None = None
        cache_dir = config.teacher_target_cache_dir
        if config.supergraph_aggregation == "token-path":
            # token-path targets are a forward and a backward, cheaper to recompute
            # than to round-trip through a growing .pt on Drive.
            cache_dir = None
        if cache_dir and cache_dir.lower() != "none":
            if shared is not None and "teacher_target_cache" in shared:
                self.teacher_target_cache = shared["teacher_target_cache"]
            else:
                if not os.path.isabs(cache_dir):
                    cache_dir = os.path.join(DIR_ROOT, cache_dir)
                key = teacher_target_cache_key(self.graph_config, config.teacher)
                self.teacher_target_cache = TeacherTargetCache(os.path.join(cache_dir, f"{key}.pt"))
                self.teacher_target_cache.validate(self._rebuild_teacher_target)
                if shared is not None:
                    shared["teacher_target_cache"] = self.teacher_target_cache
            self.graph_config.teacher_target_cache = self.teacher_target_cache
            print(
                f"Teacher target cache: {self.teacher_target_cache.path} "
                f"({len(self.teacher_target_cache)} prompts cached)"
            )

        self.history: Dict[str, List] = defaultdict(list)
        self._train_step = 0
        self._last_save_step = 0
        if config.resume:
            step, history, _ = resume_training_state(self.model, self.optimizer, config.save_dir)
            self.history = defaultdict(list, history)
            self._train_step = step
            self._last_save_step = step

    def _autocast(self):
        """bf16 autocast for the student forward; it holds fp32 master weights."""
        return student_autocast()

    def _rebuild_teacher_target(self, prompt: str, answer: Any) -> dict:
        """One prompt's teacher target with the current code (TeacherTargetCache.validate)."""
        supergraph, logit_ids, dla_logits = _compute_teacher_target(
            prompt, answer, self.teacher_adapter, self.graph_config, _DEVICE,
        )
        return _teacher_target_entry(supergraph, logit_ids, dla_logits, answer=answer)

    def _check_sequence_consistency(self, batch, input_ids, attention_mask) -> None:
        """Once per run: the ids the graph term will build on must equal the KD batch row.

        The KD term sees ``batch["input_ids"]``; the graph term re-tokenises
        ``batch["prompts"]`` / ``batch["answers"]`` through tokenize_prompt_answer and
        the attribution adapter prepends BOS when missing; eval_model tokenises with
        add_special_tokens=True. All three must agree token for token, and the
        sequence must start with BOS. A mismatch here once let the KD loss train a
        distribution the student was never evaluated on.
        """
        p_ids, a_ids = tokenize_prompt_answer(self.tokenizer, batch["prompts"][0], str(batch["answers"][0]))
        seq = torch.cat([p_ids, a_ids]).to(input_ids.device)
        n = int(attention_mask[0].sum())
        row = input_ids[0][:n]
        if n != seq.numel() or not torch.equal(row, seq):
            raise RuntimeError(
                "KD batch row and graph-term tokenisation disagree for the first prompt: "
                f"batch {row.tolist()} vs graph {seq.tolist()}"
            )
        adapter_ids = self.student_adapter.ensure_tokenized(batch["prompts"][0])
        if not torch.equal(adapter_ids.cpu(), p_ids.cpu()):
            raise RuntimeError(
                "attribution adapter tokenises the prompt differently from the KD batch: "
                f"adapter {adapter_ids.tolist()} vs batch {p_ids.tolist()}"
            )
        eval_ids = self.tokenizer([batch["prompts"][0]], add_special_tokens=True)["input_ids"][0]
        if list(eval_ids) != p_ids.tolist():
            raise RuntimeError(
                f"eval_model tokenisation differs from the KD batch prompt: eval {eval_ids} vs batch {p_ids.tolist()}"
            )
        print(f"  sequence check passed: KD batch, graph term, adapter and eval agree on {n} tokens "
              f"(BOS id {self.tokenizer.bos_token_id} leads)")

    def _eval_on(self, model, dataset_name: str, test_dataset: PromptAnswerDataset) -> float:
        cfg = self.config
        # A no-op for the bf16 teacher.
        with self._autocast():
            return eval_model(
                model, self.tokenizer, test_dataset, dataset_name,
                eval_batch_size_for(cfg.eval_batch_size,
                                    model is getattr(self, "teacher", None)),
                cfg.max_eval_tokens,
            )

    def _eval(self) -> float:
        return self._eval_on(self.model, self.config.dataset, self.test_dataset)

    def _eval_teacher(self) -> float:
        return self._eval_on(self.teacher, self.config.dataset, self.test_dataset)

    def _eval_all_extra(self) -> Dict[str, float]:
        return {ds: self._eval_on(self.model, ds, td) for ds, td in self.extra_test_datasets.items()}

    def _eval_teacher_all_extra(self) -> Dict[str, float]:
        return {ds: self._eval_on(self.teacher, ds, td) for ds, td in self.extra_test_datasets.items()}

    def train_epoch(self, *, max_steps: Optional[int] = None) -> Dict[str, float]:
        self.model.train()
        cfg = self.config
        grad_accum = cfg.grad_accum_steps

        total_kl = 0.0
        total_graph = 0.0
        n_steps = 0
        accum_kl = 0.0
        accum_graph = 0.0
        accum_kl_gnorm = 0.0
        accum_graph_gnorm = 0.0
        accum_cos = 0.0
        accum_flip = 0.0
        accum_clip = 0.0
        accum_dtheta = accum_dtheta_rel = float("nan")
        accum_aligned = 0.0
        accum_teacher_sn = 0.0
        accum_graph_real = 0.0
        accum_flops = 0
        accum_time_teacher = 0.0
        accum_time_student = 0.0
        accum_cache_hits = 0.0
        accum_graph_prompts = 0.0
        micro_step = 0
        # Matmul / attention FLOPs of each micro-step's forward and backward
        # passes; a no-op unless --track-flops. Entered around the compute only,
        # so the eval and the optimizer step stay outside it.
        flop_counter = step_flop_counter(cfg.track_flops)

        self.optimizer.zero_grad()
        for batch in self.loader:
            if max_steps is not None and n_steps >= max_steps:
                break
            flop_counter.reset()
            input_ids = batch["input_ids"].to(_DEVICE)
            attention_mask = batch["attention_mask"].to(_DEVICE)
            kd_mask = kd_position_mask(attention_mask, batch["response_mask"].to(_DEVICE))

            # With grad_accum == 1 the optimizer has just zeroed .grad, so the
            # gradient after kl.backward() *is* this step's KL gradient and no
            # snapshot is needed. With accumulation it is not, so record the
            # carried-over gradient and subtract it back out below.
            grads_at_start = None
            if cfg.track_grad_metrics and grad_accum > 1:
                grads_at_start = {
                    n: p.grad.detach().clone()
                    for n, p in self.model.named_parameters() if p.grad is not None
                }

            # ── KL loss ───────────────────────────────────────────────────────
            with flop_counter:
                with self._autocast():
                    student_logits = self.model(input_ids, attention_mask=attention_mask).logits
                self._last_tf_acc = first_answer_token_accuracy(
                    student_logits.detach(), input_ids, batch["response_mask"].to(_DEVICE))

                with torch.no_grad():
                    teacher_logits = self.teacher(input_ids, attention_mask=attention_mask).logits

                kl = kl_loss(
                    student_logits, teacher_logits, kd_mask,
                    cfg.temperature, cfg.kl_token_chunk_size,
                ) / grad_accum

                kl_finite = torch.isfinite(kl)
                # ``kl`` stays unscaled so step_kl_loss is comparable across
                # --lambda-kl settings; only the backward carries the weight, and
                # lambda_kl == 1.0 takes the original path unchanged.
                if kl_finite and cfg.lambda_kl != 0.0:
                    (kl if cfg.lambda_kl == 1.0 else kl * cfg.lambda_kl).backward()

            # ── Graph loss ────────────────────────────────────────────────────
            prompts: List[str] = batch["prompts"]
            answers = batch["answers"]
            if self._train_step == 0 and micro_step == 0:
                self._check_sequence_consistency(batch, input_ids, attention_mask)
            if cfg.n_graph_prompts is not None and cfg.n_graph_prompts < len(prompts):
                sel = random.sample(range(len(prompts)), cfg.n_graph_prompts)
                prompts = [prompts[i] for i in sel]
                answers = [answers[i] for i in sel]
            if cfg.track_grad_metrics:
                grads_before = {
                    n: p.grad.detach().clone() if p.grad is not None else None
                    for n, p in self.model.named_parameters()
                }
            with flop_counter:
                graph_loss_tensor, graph_metrics = backward_batch_graph_loss(
                    prompts=prompts,
                    answers=answers,
                    student_adapter=self.student_adapter,
                    teacher_adapter=self.teacher_adapter,
                    config=self.graph_config,
                    device=_DEVICE,
                    loss_scale=cfg.lambda_graph / grad_accum,
                )
            graph_val = float(graph_loss_tensor.item())
            # Supernode alignment counts are the cheapest early warning that the
            # target has quietly degraded. arg:<token> labels collide whenever a
            # prompt repeats a digit ("51+12=" has two '1's) and s_label_to_sid
            # keeps only the last, and any teacher/student label mismatch shrinks
            # the loss to the intersection -- both silently.
            accum_aligned += float(graph_metrics.get("aligned_teacher_supernodes", 0.0))
            accum_teacher_sn += float(graph_metrics.get("teacher_supernodes", 0.0))
            accum_graph_real += float(graph_metrics.get("edge_loss_real_target", 0.0))
            accum_time_teacher += float(graph_metrics.get("graph_time_teacher", 0.0))
            accum_time_student += float(graph_metrics.get("graph_time_student", 0.0))
            accum_cache_hits += float(graph_metrics.get("teacher_cache_hits", 0.0))
            accum_graph_prompts += float(graph_metrics.get("graph_prompts", 0.0))
            if cfg.track_grad_metrics:
                # grads_before is the gradient after the KL backward and before the
                # graph backward, so the graph contribution is exactly the delta
                # across backward_batch_graph_loss. Everything below accumulates as
                # scalars in one pass, so it costs no memory beyond that snapshot.
                # The graph gradient here is already scaled by lambda_graph, since
                # loss_scale carries it -- so these norms compare what actually
                # reaches the optimizer, not the two losses in the abstract.
                dot = kl_sq = graph_sq = 0.0
                flipped = total_elems = 0
                for n, p in self.model.named_parameters():
                    if p.grad is None:
                        continue
                    before = grads_before.get(n)
                    if before is None:
                        before = torch.zeros_like(p.grad)
                    g_kl = before.float()
                    if grads_at_start is not None:
                        start = grads_at_start.get(n)
                        if start is not None:
                            g_kl = g_kl - start.float()
                    g_graph = (p.grad - before).float()
                    dot += float((g_kl * g_graph).sum().item())
                    kl_sq += float((g_kl * g_kl).sum().item())
                    graph_sq += float((g_graph * g_graph).sum().item())
                    # Adam normalises per parameter, so what decides whether the
                    # graph term changes an update is not its norm but whether it
                    # flips the sign. This is the quantity that explains how a term
                    # orders of magnitude smaller in norm still moves the model.
                    flipped += int((torch.sign(g_kl + g_graph) != torch.sign(g_kl)).sum().item())
                    total_elems += g_kl.numel()
                denom = (kl_sq ** 0.5) * (graph_sq ** 0.5)
                accum_kl_gnorm += kl_sq ** 0.5
                accum_graph_gnorm += graph_sq ** 0.5
                accum_cos += dot / denom if denom > 0 else 0.0
                # A "flip" is only defined against a nonzero KD gradient. With
                # --lambda-kl 0 every sign(g_kl) is 0, so sign(g_kl + g_graph)
                # differs wherever the graph gradient is nonzero and the metric
                # would read as the fraction of parameters the graph term touches
                # (~0.70), not as a flip rate. Report NaN instead, as ratio does.
                accum_flip += (flipped / total_elems) if (total_elems and kl_sq > 0) else float("nan")

            micro_step += 1

            if not kl_finite:
                if micro_step % grad_accum == 0:
                    self.optimizer.zero_grad()
                continue

            accum_kl += float(kl.item())
            accum_graph += graph_val
            accum_flops += flop_counter.flops

            if micro_step % grad_accum == 0:
                # The graph loss is backpropagated inside backward_batch_graph_loss
                # with no finiteness check of its own, so guard here: a single
                # non-finite gradient would otherwise poison the Adam moment
                # estimates permanently. clip_grad_norm_ turns an infinite total
                # norm into NaN grads, but those are discarded by the zero_grad
                # below when the step is skipped.
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), _GRAD_CLIP)
                accum_clip = float(total_norm)
                if torch.isfinite(total_norm):
                    canary = ParamChangeCanary(self.model) if self._train_step == 0 else None
                    self._last_lr = apply_lr_schedule(
                        self.optimizer, self._train_step + 1, cfg.steps, cfg.learning_rate,
                        cfg.warmup_steps, cfg.lr_floor)
                    self._step_tracker.snapshot(self.model)
                    self.optimizer.step()
                    accum_dtheta, accum_dtheta_rel = self._step_tracker.delta(self.model)
                    if canary is not None:
                        log_first_step_canary(canary.report(self.model), self.history)
                else:
                    print(
                        f"  step {self._train_step + 1} | WARN: non-finite grad norm "
                        f"({total_norm.item()}); skipping optimizer step"
                    )
                self.optimizer.zero_grad()

                self._train_step += 1
                self.history["train_step"].append(self._train_step)
                self.history["step_kl_loss"].append(accum_kl)
                self.history["step_tf_acc"].append(self._last_tf_acc)
                self.history["step_lr"].append(self._last_lr)
                self.history["step_graph_loss"].append(accum_graph)
                total_kl += accum_kl
                total_graph += accum_graph
                if cfg.track_grad_metrics:
                    mean_cos = accum_cos / grad_accum
                    mean_flip = accum_flip / grad_accum
                    ratio = accum_graph_gnorm / accum_kl_gnorm if accum_kl_gnorm else float("nan")
                    mean_clip = accum_clip
                    for key, val in (
                        ("step_kl_gnorm", accum_kl_gnorm),
                        ("step_graph_gnorm", accum_graph_gnorm),
                        ("step_grad_ratio", ratio),
                        ("step_grad_cosine", mean_cos),
                        ("step_grad_signflip", mean_flip),
                        ("step_clip_norm", mean_clip),
                        ("step_update_norm", accum_dtheta),
                        ("step_update_rel", accum_dtheta_rel),
                    ):
                        self.history[key].append(val)
                    gnorm_str = (
                        f" | |g_KL|={accum_kl_gnorm:.4f} | |g_graph|={accum_graph_gnorm:.4f}"
                        f" | ratio={ratio:.4f} | cos={mean_cos:+.4f}"
                        f" | signflip={mean_flip:.3f} | clip={mean_clip:.3f}"
                        f" | dtheta={accum_dtheta:.3e}"
                    )
                else:
                    gnorm_str = ""
                self.history["step_aligned_supernodes"].append(accum_aligned / grad_accum)
                self.history["step_teacher_supernodes"].append(accum_teacher_sn / grad_accum)
                if cfg.scramble_teacher_graph:
                    # The loss the scrambled run is *not* training on: the student
                    # against the real teacher target, for drift monitoring.
                    self.history["step_graph_loss_real_target"].append(accum_graph_real)
                    self.history["scramble_permutations"] = {
                        str(k): v for k, v in self.graph_config.scramble_permutations.items()
                    }
                    scramble_str = f" | Graph(real target)={accum_graph_real:.4f}"
                else:
                    scramble_str = ""
                if cfg.track_flops:
                    self.history["step_flops"].append(accum_flops)
                    flops_str = f" | FLOPs={accum_flops:.3e}"
                else:
                    flops_str = ""
                self.history["step_graph_time_teacher"].append(accum_time_teacher)
                self.history["step_graph_time_student"].append(accum_time_student)
                self.history["step_teacher_cache_hits"].append(accum_cache_hits)
                hits_str = (
                    f", cache hits {accum_cache_hits:.0f}/{accum_graph_prompts:.0f}"
                    if self.teacher_target_cache is not None else ""
                )
                time_str = (
                    f" | graph {accum_time_teacher + accum_time_student:.0f}s"
                    f" (teacher {accum_time_teacher:.0f}s, student {accum_time_student:.0f}s{hits_str})"
                )
                print(
                    f"  step {self._train_step} | KL={accum_kl:.4f} | "
                    f"Graph={accum_graph:.4f}{scramble_str}{gnorm_str}{flops_str}{time_str}"
                )
                self._last_save_step = maybe_save_periodic_checkpoint(
                    self.model, self.tokenizer, self.config.save_dir,
                    self._train_step, self.config.save_every_n_steps,
                    self._last_save_step, self.history,
                    optimizer=self.optimizer,
                )
                accum_kl = 0.0
                accum_graph = 0.0
                accum_kl_gnorm = 0.0
                accum_graph_gnorm = 0.0
                accum_cos = 0.0
                accum_flip = 0.0
                accum_clip = 0.0
                accum_aligned = 0.0
                accum_teacher_sn = 0.0
                accum_graph_real = 0.0
                accum_flops = 0
                accum_time_teacher = 0.0
                accum_time_student = 0.0
                accum_cache_hits = 0.0
                accum_graph_prompts = 0.0
                n_steps += 1

        denom = max(n_steps, 1)
        return {"kl_loss": total_kl / denom, "graph_loss": total_graph / denom}

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

            run_baselines(
                self.shared, self.history,
                student=self._eval, teacher=self._eval_teacher, extra=self._eval_all_extra,
                teacher_extra=self._eval_teacher_all_extra,
            )

        sample = self.loader.dataset[0]
        print("─" * 60)
        print("Sample [0]:")
        print(f"  prompt:  {str(sample['prompt'])[:120]!r}")
        print(f"  answer:  {str(sample['answer'])[:80]!r}")
        print(f"  tokens:  {sample['input_ids'].shape[0]} total, {sample['prompt_len']} prompt")
        print("─" * 60)

        print(
            f"Graph-KD | student={cfg.model} | teacher={cfg.teacher} | dataset={cfg.dataset}"
            f" | steps={cfg.steps} | lr={cfg.learning_rate} | temp={cfg.temperature}"
            f" | lambda_graph={cfg.lambda_graph} | lambda_kl={cfg.lambda_kl}"
            f" | nodes_per_label={cfg.nodes_per_label}"
            + (" | CONTROL: scrambled teacher graph" if cfg.scramble_teacher_graph else "")
        )

        while self._train_step < cfg.steps:
            remaining = cfg.steps - self._train_step
            metrics = self.train_epoch(max_steps=min(cfg.eval_every_n_steps, remaining))
            if not metrics:
                break
            self.history["kl_loss"].append(metrics["kl_loss"])
            self.history["graph_loss"].append(metrics["graph_loss"])
            if self.teacher_target_cache is not None:
                self.teacher_target_cache.flush()

            acc = self._eval()
            self.history["accuracy"].append(acc)
            self.history["accuracy_step"].append(self._train_step)
            extra_accs = self._eval_all_extra()
            for ds, ds_acc in extra_accs.items():
                self.history[f"accuracy_{ds}"].append(ds_acc)
            extra_str = "".join(f" | {ds}={a:.4f}" for ds, a in extra_accs.items())
            print(f"  [eval] step {self._train_step}/{cfg.steps} | Acc={acc:.4f}{extra_str}")
            refresh_curves(self.history, cfg.save_dir, step=self._train_step,
                           losses=[("step_kl_loss", "KL Loss"), ("step_graph_loss", "Graph Loss")])

        if self.teacher_target_cache is not None:
            self.teacher_target_cache.flush()
        save_history(self.history, cfg.save_dir)
        save_curves(
            self.history, cfg.save_dir,
            losses=[("step_kl_loss", "KL Loss"), ("step_graph_loss", "Graph Loss")],
        )
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
    parser = argparse.ArgumentParser(
        description="KL + graph distillation with automatic arg-token and DLA supernodes."
    )
    add_standard_args(parser)
    add_kd_args(parser)
    group = parser.add_argument_group("kd_graph_args")
    group.add_argument("--lambda-graph", type=float, default=1.0, dest="lambda_graph")
    group.add_argument("--lambda-kl", type=float, default=1.0, dest="lambda_kl",
                       help="Weight on the KD term. 0 trains on the graph loss alone; the KL is "
                            "still computed and logged as step_kl_loss, it just gets no backward.")
    group.add_argument("--nodes-per-label", type=int, default=10, dest="nodes_per_label",
                       help="Neurons per arg-token supernode and DLA supernode.")
    group.add_argument(
        "--graph-loss-type", type=str, default="jsd", dest="graph_loss_type",
        choices=["jsd", "kld", "mse", "mse-norm", "mse-scale", "rel-mse"],
        help="rel-mse: relative squared error on the signed, globally normalised matrices "
             "(pair it with --supergraph-aggregation raw-signed); the others act on "
             "|entries| row-normalised.",
    )
    group.add_argument(
        "--supergraph-aggregation", type=str, default="normalised", dest="supergraph_aggregation",
        choices=["normalised", "raw-signed", "token-path"],
        help="normalised: per-target |inbound| shares with frac_external weighting (pool-size "
             "dependent; only row shape is comparable across models). raw-signed: mean over "
             "target members of the summed raw signed edges, whole matrix divided once by its "
             "|mass| (pool-independent; keeps sign and relative edge strength). token-path: no "
             "supernodes at all -- a distribution over input token positions (see "
             "--token-path-rows).",
    )
    group.add_argument(
        "--token-path-rows", type=str, default="weighted", dest="token_path_rows",
        choices=["weighted", "gold", "all"],
        help="Only with --supergraph-aggregation token-path. 'weighted' (default): one row, the "
             "logit rows combined with the teacher's probabilities -- needs no gold token. 'gold': "
             "one row, attribution to the gold answer's logit. 'all': the full L x T, one row per "
             "logit target, keeping which tokens push toward the wrong candidates.",
    )
    group.add_argument(
        "--token-path-micro-batch", type=int, default=4, dest="token_path_micro_batch",
        help="Prompts per forward inside the token-path attribution. The backward needs the whole "
             "sequence's activations alive, so a 444-token batch of 32 keeps ~36 GB of MLP "
             "intermediates on an 8B teacher, while 4 keeps ~4.5 GB. Raise it for short prompts, "
             "lower it if the graph term runs out of memory. Changes peak memory only, not the "
             "result.",
    )
    group.add_argument(
        "--token-source-columns", action="store_true", dest="token_source_columns",
        help="Append the token-embedding nodes as extra source columns of the supergraph "
             "(present in both models regardless of pre-selection). Same as passing "
             "'tokens' in --graph-node-labels.",
    )
    group.add_argument(
        "--freeze-attention", action="store_true", dest="freeze_attention",
        help="Stop-gradient the attention pattern when computing attribution-graph edges, "
             "so edges reflect only the direct residual-stream path. Forces an eager "
             "attention kernel (slower and more memory than SDPA).",
    )
    group.add_argument(
        "--freeze-rms-norm", action="store_true", dest="freeze_rms_norm",
        help="Stop-gradient the RMSNorm reciprocal-norm scale when computing "
             "attribution-graph edges. No speed penalty.",
    )
    group.add_argument("--top-k-logits", "--top_k_logits", type=float, default=0.95,
                       dest="top_k_logits")
    # Matched pools: 0.003 x (32 x 14336) and 0.01 x (16 x 8192) are both ~1.3-1.4k
    # nodes per position, so the ten most specific members per label are drawn from
    # pools of the same size in both models (and the teacher graph is ~30x cheaper
    # than at the old 0.1).
    group.add_argument("--teacher-prop-neurons", type=float, default=0.003,
                       dest="teacher_prop_neurons_per_layer")
    group.add_argument("--student-prop-neurons", type=float, default=0.01,
                       dest="student_prop_neurons_per_layer")
    group.add_argument("--teacher-graph-batch-size", type=int, default=512,
                       dest="teacher_graph_batch_size")
    group.add_argument("--student-graph-batch-size", type=int, default=128,
                       dest="student_graph_batch_size")
    group.add_argument("--n-graph-prompts", type=int, default=None, dest="n_graph_prompts",
                       help="Max prompts per batch to compute graph loss for (None = all).")
    group.add_argument("--graph-verbose", action="store_true", dest="graph_verbose")
    group.add_argument(
        "--constant-node-weighting", "--constant_node_weighting",
        action="store_true", dest="constant_node_weighting",
        help="Replace frac_external with 1 in the supernode aggregation, so a supernode "
             "edge is a plain mean over its members instead of a frac_external-weighted "
             "one. Applied to teacher and student alike. The RMSNorm freeze already drives "
             "frac_external to ~0.995 with its spread collapsing roughly tenfold, which "
             "makes the weighting nearly inert; this flag isolates that effect from the "
             "linearisation, so it can be run unfrozen to test whether the weighting "
             "rather than the freeze is what matters.",
    )
    group.add_argument(
        "--track-grad-metrics", "--track_grad_metrics", "--track-grad-norms",
        action="store_true", dest="track_grad_metrics",
        help="Per-step gradient diagnostics, also recorded in the history: the KL and "
             "graph gradient norms and their ratio (the graph norm is post-lambda_graph, "
             "so it is what actually reaches the optimizer), cos(graph, KD) -- which "
             "Appendix A measures only at initialisation -- the fraction of parameter "
             "entries whose update sign the graph term flips, and the pre-clip total "
             "gradient norm against the 1.0 clip threshold. --track-grad-norms still works.",
    )
    group.add_argument(
        "--graph-node-labels", "--graph_node_labels",
        nargs="+", default=[], dest="graph_node_labels", metavar="LABEL",
        help="ANOVA supernode labels, e.g. 'sum units' 'arg1 range'. Pass 'all' for every category. "
             "'tokens' adds the token-embedding nodes as source columns of the supergraph "
             "(dataset-independent; equivalent to --token-source-columns).",
    )
    group.add_argument("--anova-range-radius", "--anova_range_radius", type=int, default=0,
                       dest="anova_range_radius")
    group.add_argument(
        "--cache-batch-size", "--cache_batch_size",
        type=int, default=32, dest="cache_batch_size",
        help="Prompt batch size when building the MLP activation cache (default: 32).",
    )
    group.add_argument(
        "--anova-neuron-chunk", "--anova_neuron_chunk",
        type=int, default=None, dest="anova_neuron_chunk",
        help="Neurons processed per ANOVA batch (reduce to avoid GPU OOM on large grids; default: all at once).",
    )
    group.add_argument(
        "--anova-cache-device", "--anova_cache_device",
        choices=["auto", "cpu", "pinned", "cuda"], default="auto", dest="anova_cache_device",
        help="Where the MLP-input caches the ANOVA labeler reads live. The labeler copies "
             "every layer of a cache to the GPU for every prompt it labels, so pageable "
             "CPU memory (cpu) costs seconds per prompt, pinned memory copies at PCIe "
             "speed and cuda makes the copy free. auto (default) puts the student's cache "
             "on the GPU and pins the teacher's.",
    )
    group.add_argument(
        "--teacher-target-cache", "--teacher_target_cache",
        type=str, default="cache/teacher_targets", dest="teacher_target_cache_dir",
        help="Directory (relative to the data root unless absolute) holding one file per "
             "teacher-side configuration with every prompt's teacher target built so "
             "far. A prompt's target is computed once and reused by every later run, "
             "seed or lambda with the same teacher settings; the key hashes the "
             "graph_loss source so code changes never serve stale targets. 'none' "
             "disables it.",
    )
    group.add_argument(
        "--scramble-teacher-graph", "--scramble_teacher_graph",
        action="store_true", dest="scramble_teacher_graph",
        help="Control arm: permute each row of the teacher's supernode adjacency by a "
             "fixed derangement (one per row index, drawn once from the run seed and "
             "reused for every prompt), so the target keeps the teacher's numbers and "
             "per-row sparsity but carries no information about which supernode routes "
             "to which. Supernode selection, the student's graph and the loss are "
             "unchanged. The loss against the real target is still logged as "
             "step_graph_loss_real_target, and the permutations as "
             "scramble_permutations. A scrambled target starts further from the "
             "student, so rescale --lambda-graph to match the real run's step-1 "
             "step_grad_ratio before comparing arms.",
    )
    return parser


def _anova_cache_devices(spec: str) -> tuple[str, str]:
    """``(student, teacher)`` placement for the two MLP-input caches."""
    if spec == "auto":
        return ("cuda" if torch.cuda.is_available() else "cpu", "pinned")
    return (spec, spec)


def _cache_gb(cache: dict | None) -> float:
    if not cache:
        return 0.0
    return sum(t.numel() * t.element_size() for t in cache.get("layer_inputs", [])) / 1e9


def main() -> None:
    args = build_parser().parse_args()
    graph_node_labels, tokens_label = normalize_node_labels(args.graph_node_labels)
    if tokens_label:
        args.token_source_columns = True
    train_data, test_data = load_data(args.dataset, test_limit=args.test_limit)
    print(f"Train: {len(train_data)} | Test: {len(test_data)}")
    save_dir = os.path.join(DIR_ROOT, args.save_dir)

    def build(seed: int, shared: Dict[str, Any]):
        return GraphKDTrainer(
            GraphKDConfig(
                model=args.model,
                teacher=args.teacher,
                dataset=args.dataset,
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                warmup_steps=args.warmup_steps,
                lr_floor=args.lr_floor,
                temperature=args.temperature,
                kl_token_chunk_size=args.kl_token_chunk_size,
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
                lambda_graph=args.lambda_graph,
                lambda_kl=args.lambda_kl,
                nodes_per_label=args.nodes_per_label,
                graph_loss_type=args.graph_loss_type,
                freeze_attention=args.freeze_attention,
                freeze_rms_norm=args.freeze_rms_norm,
                constant_node_weighting=args.constant_node_weighting,
                supergraph_aggregation=args.supergraph_aggregation,
                token_source_columns=args.token_source_columns,
                token_path_rows=args.token_path_rows,
                token_path_micro_batch=args.token_path_micro_batch,
                top_k_logits=args.top_k_logits,
                teacher_prop_neurons_per_layer=args.teacher_prop_neurons_per_layer,
                student_prop_neurons_per_layer=args.student_prop_neurons_per_layer,
                teacher_graph_batch_size=args.teacher_graph_batch_size,
                student_graph_batch_size=args.student_graph_batch_size,
                n_graph_prompts=args.n_graph_prompts,
                graph_verbose=args.graph_verbose,
                track_grad_metrics=args.track_grad_metrics,
                track_flops=args.track_flops,
                scramble_teacher_graph=args.scramble_teacher_graph,
                graph_node_labels=graph_node_labels,
                anova_range_radius=args.anova_range_radius,
                mlp_cache_batch_size=args.cache_batch_size,
                anova_neuron_chunk=args.anova_neuron_chunk,
                anova_cache_device=args.anova_cache_device,
                teacher_target_cache_dir=args.teacher_target_cache_dir,
            ),
            train_data,
            test_data,
            shared=shared,
        )

    run_seeds(args.seeds, args.resume, build, save_dir=save_dir, steps=args.steps, redo=args.redo_seeds)


if __name__ == "__main__":
    main()

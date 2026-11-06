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
    bos_in_eval,
    bos_in_training,
    bos_mode,
    set_bos_mode,
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
    kl_position_mask,
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
    sweep_apply,
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
    kl_tokens: str = "resp"
    max_eval_tokens: Optional[int] = None  # None -> utils.default_eval_tokens(dataset)
    eval_batch_size: Any = 256
    save_dir: str = "results/graph_kd"
    eval_every_n_steps: int = 1
    save_every_n_steps: int = 0
    grad_accum_steps: int = 1
    eval_datasets: List[str] = field(default_factory=list)
    test_limit: Optional[int] = None
    dtype: Optional[str] = None
    bos_mode: str = "on"
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
    token_path_micro_tokens: Optional[int] = 1024
    token_path_top_tokens: int = 0
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
    # Which loss may move the BOS token's embedding row; see --bos-embedding.
    bos_embedding: str = "train"
    # Remove the KD direction from the graph gradient; see --graph-grad-projection.
    graph_grad_projection: str = "none"


BOS_EMBEDDING_MODES = ("train", "freeze", "graph-only")
GRAPH_GRAD_PROJECTIONS = ("none", "orthogonal")


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
            token_path_micro_tokens=config.token_path_micro_tokens,
            token_path_top_tokens=config.token_path_top_tokens,
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

        # The BOS row of the (tied) embedding. Under --bos-mode legacy the KD term
        # never sees BOS as an input, so this row is the one parameter only the graph
        # term reaches through the forward; --bos-embedding tests whether that is
        # where the graph term's effect lives. The row is logged every step either way.
        if config.bos_embedding not in BOS_EMBEDDING_MODES:
            raise ValueError(f"bos_embedding must be one of {BOS_EMBEDDING_MODES}, got {config.bos_embedding!r}")
        self._bos_id = self.tokenizer.bos_token_id
        if self._bos_id is None:
            raise RuntimeError("tokenizer has no BOS token; --bos-embedding and the BOS-row log need one")
        self._bos_row0 = self._bos_weight()[self._bos_id].detach().clone()
        if config.graph_grad_projection not in GRAPH_GRAD_PROJECTIONS:
            raise ValueError(f"graph_grad_projection must be one of {GRAPH_GRAD_PROJECTIONS}, "
                             f"got {config.graph_grad_projection!r}")
        if config.graph_grad_projection != "none":
            # The snapshot taken before the graph backward must be this step's KD
            # gradient alone; with accumulation it would also hold earlier micro-steps'
            # graph gradients. graph-only rewrites the same .grad from its own snapshot.
            if config.grad_accum_steps != 1:
                raise ValueError("--graph-grad-projection needs --grad-accum-steps 1")
            if config.bos_embedding == "graph-only":
                raise ValueError("--graph-grad-projection cannot be combined with --bos-embedding graph-only")

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

    def _bos_weight(self) -> torch.nn.Parameter:
        """The input-embedding matrix; on Llama-3.2 it is also the lm_head (tied)."""
        return self.model.get_input_embeddings().weight

    def _project_graph_grad_off_kd(self, kd_grads: Dict[str, Optional[torch.Tensor]]) -> tuple[float, float]:
        """--graph-grad-projection orthogonal: g <- g_KD + (g_graph - c * g_KD).

        ``kd_grads`` is .grad before the graph backward (host copies), so the graph
        gradient is the delta. c = <g_graph, g_KD> / |g_KD|^2 over the whole model
        (one scalar, fp32), i.e. the graph term keeps only what is orthogonal to the
        KD gradient of this step. Returns (cos before, |g_graph| after projection).
        """
        params = [(n, p) for n, p in self.model.named_parameters() if p.grad is not None]
        dot = kd_sq = gr_sq = 0.0
        for n, p in params:
            kd = kd_grads.get(n)
            kd = torch.zeros_like(p.grad, dtype=torch.float32) if kd is None else kd.to(p.grad.device).float()
            g = p.grad.float() - kd
            dot += float((g * kd).sum())
            kd_sq += float((kd * kd).sum())
            gr_sq += float((g * g).sum())
        if kd_sq == 0.0 or gr_sq == 0.0:
            return 0.0, gr_sq ** 0.5
        c = dot / kd_sq
        cos = dot / (kd_sq ** 0.5 * gr_sq ** 0.5)
        with torch.no_grad():
            for n, p in params:
                kd = kd_grads.get(n)
                if kd is None:
                    continue
                kd = kd.to(p.grad.device).float()
                # g_KD + g_graph - c * g_KD, written back once (one rounding).
                p.grad.copy_(p.grad.float() - c * kd)
        # |g_graph - c g_KD|^2 = |g_graph|^2 - dot^2 / |g_KD|^2
        return cos, max(gr_sq - dot * dot / kd_sq, 0.0) ** 0.5

    def _keep_graph_grad_on_bos_row_only(self, kd_grads: Dict[str, Optional[torch.Tensor]]) -> None:
        """--bos-embedding graph-only: drop the graph gradient everywhere but the BOS row.

        ``kd_grads`` is .grad as it stood before the graph backward. Every parameter
        is put back to it exactly (a copy, so nothing is rounded), except the BOS
        row of the embedding, which keeps KD + graph as accumulated.
        """
        emb = self._bos_weight()
        for n, p in self.model.named_parameters():
            if n not in kd_grads:
                continue
            before = kd_grads[n]
            if p is emb:
                if p.grad is None:
                    continue
                row = p.grad[self._bos_id].clone()
                if before is None:
                    p.grad.zero_()
                else:
                    p.grad.copy_(before)
                p.grad[self._bos_id] = row
            elif before is None:
                p.grad = None
            elif p.grad is not None:
                p.grad.copy_(before)

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
        # Under --bos-mode legacy the training sequence deliberately carries no BOS
        # while eval and the adapter do, so the comparison is against p_ids with the
        # BOS they add stripped. That is the whole point of the mode, and checking
        # the remainder still catches any *other* drift between the four paths.
        bos_id = self.tokenizer.bos_token_id
        lead = bos_in_eval() and not bos_in_training()
        def _strip(ids: list) -> list:
            if lead and ids and bos_id is not None and int(ids[0]) == int(bos_id):
                return list(ids[1:])
            return list(ids)

        adapter_ids = self.student_adapter.ensure_tokenized(batch["prompts"][0])
        if _strip(adapter_ids.tolist()) != p_ids.tolist():
            raise RuntimeError(
                "attribution adapter tokenises the prompt differently from the KD batch: "
                f"adapter {adapter_ids.tolist()} vs batch {p_ids.tolist()}"
            )
        eval_ids = self.tokenizer(
            [batch["prompts"][0]], add_special_tokens=bos_in_eval())["input_ids"][0]
        if _strip(eval_ids) != p_ids.tolist():
            raise RuntimeError(
                f"eval_model tokenisation differs from the KD batch prompt: eval {eval_ids} vs batch {p_ids.tolist()}"
            )
        if lead:
            for name, ids in (("adapter", adapter_ids.tolist()), ("eval", list(eval_ids))):
                if bos_id is not None and (not ids or int(ids[0]) != int(bos_id)):
                    raise RuntimeError(
                        f"--bos-mode legacy expects {name} to lead with BOS (id {bos_id}), got {ids[:4]}"
                    )
        where = {"on": "everywhere", "legacy": "at eval and in the adapter only",
                 "off": "nowhere"}[bos_mode()]
        print(f"  sequence check passed: KD batch, graph term, adapter and eval agree on {n} tokens "
              f"(--bos-mode {bos_mode()}: BOS id {bos_id} leads {where})")

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
        accum_bos_kl_gnorm = 0.0
        accum_bos_graph_gnorm = 0.0
        micro_step = 0
        # Matmul / attention FLOPs of each micro-step's forward and backward
        # passes; a no-op unless --track-flops. Entered around the compute only,
        # so the eval and the optimizer step stay outside it.
        flop_counter = step_flop_counter(cfg.track_flops)

        self.optimizer.zero_grad()
        if _DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats(_DEVICE)
        for batch in self.loader:
            if max_steps is not None and n_steps >= max_steps:
                break
            flop_counter.reset()
            input_ids = batch["input_ids"].to(_DEVICE)
            attention_mask = batch["attention_mask"].to(_DEVICE)
            kd_mask = kl_position_mask(
                attention_mask, batch["response_mask"].to(_DEVICE), cfg.kl_tokens,
                input_ids=input_ids, bos_token_id=self.tokenizer.bos_token_id)

            # With grad_accum == 1 the optimizer has just zeroed .grad, so the
            # gradient after kl.backward() *is* this step's KL gradient and no
            # snapshot is needed. With accumulation it is not, so record the
            # carried-over gradient and subtract it back out below.
            grads_at_start = None
            if cfg.track_grad_metrics and grad_accum > 1:
                grads_at_start = {
                    n: p.grad.detach().to("cpu", copy=True)
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
                # The graph term below builds a double-backward graph over the
                # student; the two B x L x V logit tensors (3.6 GB each in bf16 at
                # a 444-token context) are done and must not sit next to it.
                del student_logits, teacher_logits

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
                # Held on the host: an fp32 copy of every gradient is 4.9 GB on
                # the 1B student, and it would otherwise sit on the GPU through
                # the graph term, whose double-backward graph is the run's peak.
                # Two transfers of that size cost ~2 s against a ~40 s step.
                grads_before = {
                    n: p.grad.detach().to("cpu", copy=True) if p.grad is not None else None
                    for n, p in self.model.named_parameters()
                }
            # graph-only: an on-device copy of the KD gradient, restored after the
            # graph backward everywhere except the BOS row (~2.5 GB in bf16).
            kd_grads = None
            if cfg.bos_embedding == "graph-only":
                kd_grads = {
                    n: p.grad.detach().clone() if p.grad is not None else None
                    for n, p in self.model.named_parameters()
                }
            # Projection: a host copy of the KD gradient (~2.5 GB in bf16); the graph
            # backward is the run's memory peak, so it stays off the GPU.
            proj_kd_grads = None
            if cfg.graph_grad_projection != "none":
                proj_kd_grads = {
                    n: p.grad.detach().to("cpu", copy=True) if p.grad is not None else None
                    for n, p in self.model.named_parameters()
                }
            # Embedding-row gradient before the graph backward, for the BOS-row share.
            emb = self._bos_weight()
            bos_g_before = (emb.grad[self._bos_id].detach().float().clone()
                            if emb.grad is not None else torch.zeros_like(self._bos_row0, dtype=torch.float32))
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
            bos_g_after = (emb.grad[self._bos_id].detach().float()
                           if emb.grad is not None else torch.zeros_like(bos_g_before))
            accum_bos_kl_gnorm += float(bos_g_before.norm())
            accum_bos_graph_gnorm += float((bos_g_after - bos_g_before).norm())
            del bos_g_before, bos_g_after
            if kd_grads is not None:
                self._keep_graph_grad_on_bos_row_only(kd_grads)
                del kd_grads
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
                    # The snapshot lives on the host; bring this tensor back once
                    # and use that copy for both the KL gradient and the delta.
                    before = (torch.zeros_like(p.grad) if before is None
                              else before.to(p.grad.device))
                    g_kl = before.float()
                    if grads_at_start is not None:
                        start = grads_at_start.get(n)
                        if start is not None:
                            g_kl = g_kl - start.to(p.grad.device).float()
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
                del grads_before, grads_at_start

            # After the grad metrics, so step_grad_cosine / ratio stay the raw graph
            # gradient and compare with every other run; the projection's own
            # numbers are logged as step_proj_*.
            if proj_kd_grads is not None:
                self._proj_cos, self._proj_gnorm = self._project_graph_grad_off_kd(proj_kd_grads)
                del proj_kd_grads

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
                emb = self._bos_weight()
                if cfg.bos_embedding == "freeze" and emb.grad is not None:
                    # Out of the clip norm too, so the rest of the update is what an
                    # unfrozen run would take.
                    emb.grad[self._bos_id] = 0
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), _GRAD_CLIP)
                accum_clip = float(total_norm)
                bos_row_before = emb[self._bos_id].detach().float().clone()
                if torch.isfinite(total_norm):
                    canary = ParamChangeCanary(self.model) if self._train_step == 0 else None
                    self._last_lr = apply_lr_schedule(
                        self.optimizer, self._train_step + 1, cfg.steps, cfg.learning_rate,
                        cfg.warmup_steps, cfg.lr_floor)
                    self._step_tracker.snapshot(self.model)
                    self.optimizer.step()
                    if cfg.bos_embedding == "freeze":
                        # A zero gradient still leaves AdamW's decoupled weight decay
                        # (and any momentum), so put the row back exactly.
                        with torch.no_grad():
                            emb[self._bos_id] = self._bos_row0
                    accum_dtheta, accum_dtheta_rel = self._step_tracker.delta(self.model)
                    if canary is not None:
                        log_first_step_canary(canary.report(self.model), self.history)
                else:
                    print(
                        f"  step {self._train_step + 1} | WARN: non-finite grad norm "
                        f"({total_norm.item()}); skipping optimizer step"
                    )
                # Did the BOS row move this step, and how far is it from where it
                # started? In bf16 an Adam step below half a ULP is dropped, so
                # changed_frac says whether the row can move at this lr at all.
                bos_row_after = emb[self._bos_id].detach().float()
                row0 = self._bos_row0.float()
                bos_update_rel = float((bos_row_after - bos_row_before).norm() / bos_row_before.norm().clamp_min(1e-12))
                bos_changed = float((bos_row_after != bos_row_before).float().mean())
                bos_drift_rel = float((bos_row_after - row0).norm() / row0.norm().clamp_min(1e-12))
                self.history["step_bos_row_update_rel"].append(bos_update_rel)
                self.history["step_bos_row_changed_frac"].append(bos_changed)
                self.history["step_bos_row_drift_rel"].append(bos_drift_rel)
                self.history["step_bos_row_kl_gnorm"].append(accum_bos_kl_gnorm)
                self.history["step_bos_row_graph_gnorm"].append(accum_bos_graph_gnorm)
                bos_str = (f" | BOS row: |g_KL|={accum_bos_kl_gnorm:.2e} |g_graph|={accum_bos_graph_gnorm:.2e}"
                           f" moved={bos_changed:.2f} drift={bos_drift_rel:.2e}")
                if cfg.graph_grad_projection != "none":
                    self.history["step_proj_cos"].append(self._proj_cos)
                    self.history["step_proj_graph_gnorm"].append(self._proj_gnorm)
                    bos_str += f" | proj: cos removed={self._proj_cos:+.3f} |g_graph⊥|={self._proj_gnorm:.3e}"
                del bos_row_before, bos_row_after
                accum_bos_kl_gnorm = 0.0
                accum_bos_graph_gnorm = 0.0
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
                # Peak allocation over the step, and what is still resident once
                # the step is over. A run that OOMs after n identical steps is
                # either a peak that finally landed on a long batch (peak varies,
                # resident flat) or a leak (resident climbs step over step); the
                # traceback alone cannot tell the two apart.
                if _DEVICE.type == "cuda":
                    peak_gb = torch.cuda.max_memory_allocated(_DEVICE) / 2**30
                    resident_gb = torch.cuda.memory_allocated(_DEVICE) / 2**30
                    torch.cuda.reset_peak_memory_stats(_DEVICE)
                    self.history["step_peak_mem_gb"].append(peak_gb)
                    self.history["step_resident_mem_gb"].append(resident_gb)
                    mem_str = f" | mem peak {peak_gb:.1f}G resident {resident_gb:.1f}G"
                else:
                    mem_str = ""
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
                    f"Graph={accum_graph:.4f}{scramble_str}{gnorm_str}{bos_str}{flops_str}{time_str}{mem_str}"
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
            + (f" | BOS embedding row: {cfg.bos_embedding}" if cfg.bos_embedding != "train" else "")
            + (f" | graph grad projection: {cfg.graph_grad_projection}"
               if cfg.graph_grad_projection != "none" else "")
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
    group.add_argument("--lambda-graph", type=float, nargs="+", default=[1.0], dest="lambda_graph",
                       help="Weight on the graph term. Several values sweep; see --lr.")
    group.add_argument("--lambda-kl", type=float, nargs="+", default=[1.0], dest="lambda_kl",
                       help="Weight on the KD term. 0 trains on the graph loss alone; the KL is "
                            "still computed and logged as step_kl_loss, it just gets no backward.")
    group.add_argument("--nodes-per-label", type=int, nargs="+", default=[10], dest="nodes_per_label",
                       help="Neurons per arg-token supernode and DLA supernode.")
    group.add_argument(
        "--graph-loss-type", type=str, nargs="+", default=["jsd"], dest="graph_loss_type",
        choices=["jsd", "kld", "mse", "mse-norm", "mse-scale", "rel-mse"],
        help="rel-mse: relative squared error on the signed, globally normalised matrices "
             "(pair it with --supergraph-aggregation raw-signed); the others act on "
             "|entries| row-normalised.",
    )
    group.add_argument(
        "--supergraph-aggregation", type=str, nargs="+", default=["normalised"],
        dest="supergraph_aggregation",
        choices=["normalised", "raw-signed", "token-path"],
        help="normalised: per-target |inbound| shares with frac_external weighting (pool-size "
             "dependent; only row shape is comparable across models). raw-signed: mean over "
             "target members of the summed raw signed edges, whole matrix divided once by its "
             "|mass| (pool-independent; keeps sign and relative edge strength). token-path: no "
             "supernodes at all -- a distribution over input token positions (see "
             "--token-path-rows).",
    )
    group.add_argument(
        "--token-path-rows", type=str, nargs="+", default=["weighted"], dest="token_path_rows",
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
             "result. --token-path-micro-tokens caps the same chunks by padded tokens.",
    )
    group.add_argument(
        "--token-path-micro-tokens", type=int, default=1024, dest="token_path_micro_tokens",
        help="Padded tokens (prompts x longest sequence) per token-path micro-batch, on top of "
             "--token-path-micro-batch; 0 disables the cap. The student's double-backward graph "
             "scales with this, so it is what holds the peak steady across batches of "
             "different length: at 4 x ~450-token prompts the peak swung 66-76 GB and "
             "then overflowed 80, at 1024 tokens (2 such prompts) it is ~15 GB lower. "
             "Irrelevant for 9-token arithmetic prompts (4 x 9 is far under the cap).",
    )
    group.add_argument(
        "--token-path-top-tokens", type=int, nargs="+", default=[0], dest="token_path_top_tokens",
        help="Score only the teacher's top-k attributed prompt positions, plus one "
             "aggregate column for the mass each model puts everywhere else; 0 (the "
             "default) keeps the full profile. A full row is a distribution over every "
             "prompt token, so on a ~450-token context the ~440 near-zero entries set "
             "the scale of the distance and the positions carrying the evidence barely "
             "move it. The residual column is what keeps the loss proper: without it "
             "the student's attribution off the teacher's positions is unconstrained. "
             "Indices come from the teacher, and under --scramble-teacher-graph the "
             "permutation is applied at full width before the top-k is taken.",
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
    group.add_argument("--top-k-logits", "--top_k_logits", type=float, nargs="+", default=[0.95],
                       dest="top_k_logits")
    # Matched pools: 0.003 x (32 x 14336) and 0.01 x (16 x 8192) are both ~1.3-1.4k
    # nodes per position, so the ten most specific members per label are drawn from
    # pools of the same size in both models (and the teacher graph is ~30x cheaper
    # than at the old 0.1).
    group.add_argument("--teacher-prop-neurons", type=float, nargs="+", default=[0.003],
                       dest="teacher_prop_neurons_per_layer")
    group.add_argument("--student-prop-neurons", type=float, nargs="+", default=[0.01],
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
    group.add_argument(
        "--bos-embedding", type=str, nargs="+", default=["train"], choices=list(BOS_EMBEDDING_MODES),
        dest="bos_embedding",
        help="Which loss may move the BOS token's embedding row (tied to its lm_head row on "
             "Llama-3.2). 'train' (default): both, as every recorded run. 'freeze': neither -- the "
             "row is held at its initial value (gradient zeroed before the clip, row restored "
             "after the AdamW step). 'graph-only': the graph gradient is applied to the BOS row "
             "and nowhere else, the KD gradient everywhere as usual. Under --bos-mode legacy the "
             "KD term never feeds BOS in, so the row is the one parameter only the graph term "
             "reaches through the forward; freeze tests whether the graph term's effect needs it, "
             "graph-only whether the row alone carries it. Every run logs the row's per-step "
             "update, drift and gradient shares as step_bos_row_*.",
    )
    group.add_argument(
        "--graph-grad-projection", type=str, nargs="+", default=["none"],
        choices=list(GRAPH_GRAD_PROJECTIONS), dest="graph_grad_projection",
        help="'orthogonal': each step, remove from the graph gradient its component along "
             "that step's KD gradient (one scalar over the whole model, fp32), so the graph "
             "term adds only directions KD is not already taking. Tests whether the graph "
             "term fails under --bos-mode on because its gradient is ~75%% parallel to KD's "
             "(under legacy it is ~orthogonal already). step_grad_cosine/ratio still log the "
             "raw graph gradient; the removed cosine and the remaining norm are step_proj_*. "
             "Needs --grad-accum-steps 1.",
    )
    return parser


def config_from_args(args: argparse.Namespace, seed: int, save_dir: str) -> GraphKDConfig:
    """The run config for one seed, from args already collapsed to a sweep point."""
    graph_node_labels, _ = normalize_node_labels(args.graph_node_labels)
    return GraphKDConfig(
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
        kl_tokens=args.kl_tokens,
        save_dir=save_dir,
        eval_every_n_steps=args.eval_every_n_steps,
        save_every_n_steps=args.save_every_n_steps,
        grad_accum_steps=args.grad_accum_steps,
        max_eval_tokens=args.max_eval_tokens,
        eval_batch_size=args.eval_batch_size,
        eval_datasets=args.eval_datasets,
        test_limit=args.test_limit,
        dtype=args.dtype,
        bos_mode=args.bos_mode,
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
        token_path_micro_tokens=args.token_path_micro_tokens or None,
        token_path_top_tokens=args.token_path_top_tokens,
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
        bos_embedding=args.bos_embedding,
        graph_grad_projection=args.graph_grad_projection,
    )


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
            config_from_args(args, seed, save_dir),
            train_data,
            test_data,
            shared=shared,
        )

    for point in sweep_apply(args):
        set_bos_mode(args.bos_mode)
        if point:
            print()
            print("=== sweep point: " + point + " ===")
        run_seeds(args.seeds, args.resume, build, save_dir=save_dir, steps=args.steps,
                  redo=args.redo_seeds)


if __name__ == "__main__":
    main()

"""GSM8K CoT KL distillation."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import os
import random
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
    FEWSHOT_POOL_SIZE,
    LLAMA_1B_MODEL_NAME,
    LLAMA_8B_MODEL_NAME,
    TRAIN_SPLIT_SIZE,
    build_fewshot_prefix,
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
from utils.answer_parsing import extract_gsm8k_answer


# ─────────────────────────────────────────────────────────────────────────────
# KL loss helpers
# ─────────────────────────────────────────────────────────────────────────────


def kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    t = temperature
    vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
    student_flat = student_logits[..., :vocab].reshape(-1, vocab)
    teacher_flat = teacher_logits[..., :vocab].reshape(-1, vocab)
    valid = attention_mask.reshape(-1).bool().nonzero(as_tuple=False).squeeze(-1)
    if valid.numel() == 0:
        return student_logits.sum() * 0.0
    s = student_flat.index_select(0, valid.to(student_flat.device)).float()
    t_log = teacher_flat.index_select(0, valid.to(teacher_flat.device)).to(device=s.device, dtype=torch.float32)
    log_p_t = F.log_softmax(t_log / t, dim=-1)
    log_q_s = F.log_softmax(s / t, dim=-1)
    return (log_p_t.exp() * (log_p_t - log_q_s)).sum() / valid.numel() * (t ** 2)


def generalized_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    response_mask: torch.Tensor,
    beta: float,
    temperature: float,
) -> torch.Tensor:
    """Generalized JSD distillation loss (GKD, arXiv:2306.13649).

    Faithfully mirrors TRL ``GKDTrainer.generalized_jsd_loss`` (temperature
    scaling, log_softmax, the ``beta==0``/``beta==1`` special cases using
    ``F.kl_div(..., log_target=True)`` with PyTorch's swapped-argument
    convention, and the ``logsumexp`` mixture for ``0 < beta < 1``) but reduces
    over the RESPONSE-mask positions using the same masking style as ``kl_loss``
    (flatten, index_select by mask, mean over valid tokens, ``* temperature**2``).

    With ``beta == 0`` this is *exactly* the value ``kl_loss`` computes:
    forward KL(teacher || student) on the response tokens. ``F.kl_div(input,
    target, log_target=True)`` returns ``target.exp() * (target - input)`` per
    element, so ``F.kl_div(student_log_probs, teacher_log_probs)`` equals
    ``p_teacher * (log p_teacher - log p_student)`` — identical to the
    ``(log_p_t.exp() * (log_p_t - log_q_s))`` term in ``kl_loss``.
    """
    t = temperature
    vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
    student_flat = student_logits[..., :vocab].reshape(-1, vocab)
    teacher_flat = teacher_logits[..., :vocab].reshape(-1, vocab)
    valid = response_mask.reshape(-1).bool().nonzero(as_tuple=False).squeeze(-1)
    if valid.numel() == 0:
        return student_logits.sum() * 0.0
    s = student_flat.index_select(0, valid.to(student_flat.device)).float() / t
    t_in = teacher_flat.index_select(0, valid.to(teacher_flat.device)).to(
        device=s.device, dtype=torch.float32
    ) / t
    student_log_probs = F.log_softmax(s, dim=-1)
    teacher_log_probs = F.log_softmax(t_in, dim=-1)

    if beta == 0.0:
        # KL(teacher || student) — identical to kl_loss's summand.
        per_token = F.kl_div(
            student_log_probs, teacher_log_probs, reduction="none", log_target=True
        )
    elif beta == 1.0:
        # KL(student || teacher) — reverse KL.
        per_token = F.kl_div(
            teacher_log_probs, student_log_probs, reduction="none", log_target=True
        )
    else:
        beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack(
                [
                    student_log_probs + torch.log(beta_t),
                    teacher_log_probs + torch.log(1.0 - beta_t),
                ]
            ),
            dim=0,
        )
        # PyTorch swaps the standard KL argument order; see F.kl_div docs.
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        per_token = beta_t * kl_teacher + (1.0 - beta_t) * kl_student

    return per_token.sum() / valid.numel() * (t ** 2)


def kl_loss_from_student_hidden(
    student_hidden: torch.Tensor,
    lm_head,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    t = temperature
    vocab = min(int(lm_head.weight.shape[0]), int(teacher_logits.shape[-1]))
    hidden_flat = student_hidden.reshape(-1, student_hidden.shape[-1])
    teacher_flat = teacher_logits[..., :vocab].reshape(-1, vocab)
    valid = attention_mask.reshape(-1).bool().nonzero(as_tuple=False).squeeze(-1)
    if valid.numel() == 0:
        return student_hidden.sum() * 0.0
    s = lm_head(hidden_flat.index_select(0, valid.to(hidden_flat.device)))[..., :vocab].float()
    t_log = teacher_flat.index_select(0, valid.to(teacher_flat.device)).to(device=s.device, dtype=torch.float32)
    log_p_t = F.log_softmax(t_log / t, dim=-1)
    log_q_s = F.log_softmax(s / t, dim=-1)
    return (log_p_t.exp() * (log_p_t - log_q_s)).sum() / valid.numel() * (t ** 2)


def ce_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
    hard_labels = teacher_logits[..., :vocab].argmax(dim=-1)
    valid = response_mask.reshape(-1).bool().nonzero(as_tuple=False).squeeze(-1)
    if valid.numel() == 0:
        return student_logits.sum() * 0.0
    student_flat = student_logits[..., :vocab].reshape(-1, vocab)
    labels_flat = hard_labels.reshape(-1)
    s = student_flat.index_select(0, valid.to(student_flat.device)).float()
    l = labels_flat.index_select(0, valid.to(labels_flat.device)).to(s.device)
    return F.cross_entropy(s, l)


def sft_ce_loss(
    student_logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """CE against gold token IDs for SFT (no teacher needed)."""
    logits = student_logits[:, :-1, :]
    labels = input_ids[:, 1:]
    mask = response_mask[:, 1:]
    valid = mask.reshape(-1).bool().nonzero(as_tuple=False).squeeze(-1)
    if valid.numel() == 0:
        return student_logits.sum() * 0.0
    logits_flat = logits.reshape(-1, logits.shape[-1])
    labels_flat = labels.reshape(-1)
    s = logits_flat.index_select(0, valid.to(logits_flat.device)).float()
    l = labels_flat.index_select(0, valid.to(labels_flat.device)).to(s.device)
    return F.cross_entropy(s, l)


def teacher_correct_mask(
    teacher_logits: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer,
    gold_answers: list,
) -> torch.Tensor:
    """Return a [B] bool tensor — True where teacher's greedy response matches the gold answer."""
    B = teacher_logits.shape[0]
    pred_ids = teacher_logits.argmax(dim=-1)  # [B, T]
    correct = torch.zeros(B, dtype=torch.bool)
    for i in range(B):
        resp_ids = pred_ids[i][response_mask[i].bool()]
        decoded = tokenizer.decode(resp_ids.tolist(), skip_special_tokens=True)
        pred_num = extract_gsm8k_answer(decoded)
        gold_num = extract_gsm8k_answer(str(gold_answers[i]))
        if pred_num is not None and gold_num is not None and pred_num == gold_num:
            correct[i] = True
    return correct


class GSM8KDataset(Dataset):
    def __init__(self, data: Union[str, Dict[str, Union[int, str]]], tokenizer, max_response_tokens: Optional[int] = None, fewshot_pool: Optional[Dict] = None, num_fewshot: int = 0):
        if isinstance(data, str):
            with open(data, "r", encoding="utf-8") as f:
                raw = json.load(f)
            data = json_to_prompt_answer_dict(raw)
        self.samples = []
        for prompt, answer in data.items():
            answer_text = str(answer) if isinstance(answer, int) else answer
            if fewshot_pool and num_fewshot > 0:
                full_prompt = build_fewshot_prefix(fewshot_pool, num_fewshot) + prompt
            else:
                full_prompt = prompt
            if getattr(tokenizer, "chat_template", None):
                try:
                    formatted_prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": full_prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    formatted_prompt = full_prompt + "\n\nA:"
            else:
                formatted_prompt = full_prompt + "\n\nA:"
            prompt_ids = tokenizer(
                formatted_prompt, return_tensors="pt", padding=False, add_special_tokens=False,
            )["input_ids"].squeeze(0)
            answer_ids = tokenizer(
                answer_text + tokenizer.eos_token,
                return_tensors="pt", padding=False, add_special_tokens=False,
            )["input_ids"].squeeze(0)
            if max_response_tokens is not None:
                answer_ids = answer_ids[:max_response_tokens]
            self.samples.append({
                "input_ids": torch.cat([prompt_ids, answer_ids]),
                "prompt_len": int(prompt_ids.size(0)),
                "prompt": full_prompt,
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
    response_mask = torch.zeros(len(examples), max_len, dtype=torch.long)
    for row, ex in enumerate(examples):
        ids = ex["input_ids"]
        prompt_len = ex["prompt_len"]
        input_ids[row, : ids.size(0)] = ids
        attention_mask[row, : ids.size(0)] = 1
        response_mask[row, prompt_len : ids.size(0)] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
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
    warmup_ratio: float = 0.0
    kl_weight: float = 1.0
    ce_weight: float = 0.0
    only_correct: bool = False
    temperature: float = 2.0
    grad_clip: float = 1.0
    max_response_tokens: int = 256
    eval_batch_size: int = 32
    save_dir: str = "results/cot_distillation"
    step_log_interval: int = 1
    eval_every_n_steps: int = 1
    save_interval: int = 0
    benchmark_eval_limit: Optional[int] = 100
    skip_baseline_eval: bool = False
    num_fewshot: int = 0
    seed: int = 42
    device: torch.device = field(default_factory=get_default_device)

    # ── Feature 1: native on-policy GKD (arXiv:2306.13649) ──────────────────
    # Defaults below preserve the current pure off-policy forward-KL behavior.
    lmbda: float = 0.0          # P(train on student-generated on-policy seqs); 0 = off-policy.
    beta: float = 0.0           # JSD interp: 0 = forward KL (== kl_loss), 1 = reverse KL.
    gkd_max_new_tokens: int = 256  # max new tokens when generating on-policy.
    seq_kd: bool = False        # when not on-policy this step, train on teacher-generated seqs.

    # ── Feature 2: CoT graph distillation (reuses graph_loss/training.py) ───
    # lambda_graph == 0 (default) loads NO graph machinery: pure KL/GKD path.
    lambda_graph: float = 0.0
    graph_loss_type: Literal["jsd", "kld", "mse", "mse-norm", "mse-scale"] = "kld"
    compare_n_tokens: Optional[int] = 3
    tokens_dla_nodes: bool = False
    graph_dtype: Optional[torch.dtype] = None
    teacher_prop_neurons_per_layer: float = 0.1
    student_prop_neurons_per_layer: float = 0.1
    graph_top_k_logits: Optional[float] = 0.95
    teacher_graph_batch_size: int = 32
    student_graph_batch_size: int = 1
    teacher_nodes_per_label: int = 10
    student_nodes_per_label: int = 10
    graph_start_step: int = 1
    graph_verbose: bool = False

    track_loss_grads: bool = False

    # ── SFT mode ─────────────────────────────────────────────────────────────
    # "student" or "teacher": fine-tune that model on gold GSM8K responses.
    # Forces kl_weight=0, ce_weight=1.0; no frozen teacher is loaded.
    sft_model: Optional[Literal["student", "teacher"]] = None


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
        fewshot_pool: Optional[Dict] = None,
        tokenizer=None,
        student=None,
        resume_step: Optional[int] = None,
    ) -> None:
        self.config = config
        self.fewshot_pool = fewshot_pool or {}
        self.test_data = test_data
        self.device = torch.device(config.device)
        self._resume_step = resume_step
        self._resume = resume_step is not None
        # GKD is active whenever any on-policy/JSD knob departs from its default;
        # otherwise the loss path is byte-for-byte the original forward-KL one.
        self._is_sft = config.sft_model is not None
        if self._is_sft:
            config.kl_weight = 0.0
            config.ce_weight = 1.0
        self._use_gkd = (
            config.beta != 0.0 or config.lmbda > 0.0 or config.seq_kd
        )
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

        sft_target_model = (
            config.teacher_model if config.sft_model == "teacher" else config.student_model
        )
        if student is not None and config.sft_model != "teacher":
            self.student = student
        else:
            self.student, self.tokenizer = load_student_model_for_distillation(
                None, sft_target_model, self.device,
            )
        self.student = self.student.to(device=self.device, dtype=torch.bfloat16)
        if hasattr(self.student.config, "use_cache"):
            self.student.config.use_cache = False
        if hasattr(self.student, "gradient_checkpointing_enable"):
            self.student.gradient_checkpointing_enable()
            print("Enabled student gradient checkpointing.")
        self.student.train()

        if self._is_sft:
            self.teacher = None
        else:
            print(f"Loading teacher: {config.teacher_model}")
            self.teacher, _ = load_model(config.teacher_model)
            self.teacher = self.teacher.to(device=self.device, dtype=torch.bfloat16)
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False

        # ── Feature 2 setup: graph distillation adapters/config (lazy) ──────
        # Only touch graph_loss machinery when lambda_graph > 0 so the default
        # KL/GKD path never imports or builds any graph state.
        self.graph_loss_config = None
        self.student_graph_adapter = None
        self.teacher_graph_adapter = None
        if self._use_graph and not self._is_sft:
            from graph_loss.hf_adapter import HFLlamaGraphAdapter
            from graph_loss.training import GraphAuxConfig

            self.graph_loss_config = GraphAuxConfig(
                lambda_graph=config.lambda_graph,
                graph_dtype=config.graph_dtype,
                teacher_prop_neurons_per_layer=config.teacher_prop_neurons_per_layer,
                student_prop_neurons_per_layer=config.student_prop_neurons_per_layer,
                top_k_logits=config.graph_top_k_logits,
                temperature=config.temperature,
                teacher_graph_batch_size=config.teacher_graph_batch_size,
                student_graph_batch_size=config.student_graph_batch_size,
                verbose=config.graph_verbose,
                teacher_nodes_per_label=config.teacher_nodes_per_label,
                student_nodes_per_label=config.student_nodes_per_label,
                graph_loss_type=config.graph_loss_type,
                tokens_dla_nodes=config.tokens_dla_nodes,
                compare_n_tokens=config.compare_n_tokens,
            )
            self.student_graph_adapter = HFLlamaGraphAdapter(
                self.student, self.tokenizer, self.device,
            )
            self.teacher_graph_adapter = HFLlamaGraphAdapter(
                self.teacher, self.tokenizer, self.device,
            )

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
                params=self.student.parameters(), lr=config.learning_rate, foreach=False,
            )
            print("Using standard AdamW optimizer.")

        warmup_steps = max(1, int(config.warmup_ratio * config.steps))
        def _lr_lambda(current_step: int) -> float:
            if config.warmup_ratio <= 0.0:
                return 1.0
            return min(1.0, current_step / warmup_steps)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda)

        self.loader = DataLoader(
            GSM8KDataset(train_data, self.tokenizer, max_response_tokens=config.max_response_tokens, fewshot_pool=fewshot_pool, num_fewshot=config.num_fewshot),
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=partial(collate_fn, pad_id=self.tokenizer.eos_token_id),
        )

        self.history: Dict[str, List] = defaultdict(list)
        self._train_step = 0
        self._step_log_eval_accuracy = 0.0

    def _evaluate_model(self, model) -> float:
        cfg = self.config
        model.eval()
        with torch.no_grad():
            _, acc = run_hf_benchmark(
                model,
                self.tokenizer,
                "gsm8k",
                results_fname=None,
                batch_size=cfg.eval_batch_size,
                max_new_tokens=cfg.max_response_tokens,
                limit=cfg.benchmark_eval_limit,
                log=False,
                fewshot_pool=self.fewshot_pool or None,
                num_fewshot=cfg.num_fewshot,
            )
        return float(acc)

    def _teacher_logits_for_batch(
        self, batch: Dict[str, Any], input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.teacher(
                input_ids=input_ids, attention_mask=attention_mask,
            ).logits.detach().cpu()

    def _student_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Student logits WITH grad, using the chunk-free backbone+lm_head path when available."""
        student_backbone = getattr(self.student, "model", None)
        lm_head = getattr(self.student, "lm_head", None)
        if student_backbone is not None and lm_head is not None:
            student_out = student_backbone(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
            )
            return lm_head(student_out.last_hidden_state)
        return self.student(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
        ).logits

    def _distill_term(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        """kl_weight * distill + ce_weight * ce.

        Uses the original ``kl_loss`` (forward KL) on the default path and the
        generalized JSD only when GKD is active.  At ``beta == 0`` the two are
        numerically identical, so this preserves current behavior exactly when
        all GKD knobs are at their defaults.
        """
        if self._use_gkd:
            kl = generalized_jsd_loss(
                student_logits, teacher_logits, response_mask, self.config.beta, self.config.temperature,
            )
        else:
            kl = kl_loss(student_logits, teacher_logits, response_mask, self.config.temperature)
        ce = ce_loss(student_logits, teacher_logits, response_mask)
        loss = self.config.kl_weight * kl + self.config.ce_weight * ce
        return loss, float(kl.item()), float(ce.item())

    def _generate_on_policy_batch(
        self, prompts: List[str], generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate continuations from ``generator`` given prompts only.

        Mirrors TRL's ``generate_on_policy_outputs``: left-pad the prompts,
        sample a continuation, then build ``(input_ids, attention_mask,
        response_mask)`` where the response mask is 1 only on freshly generated
        non-pad positions.  Generation runs under ``no_grad``; the loss forward
        pass on these sequences happens later WITH grad.
        """
        tok = self.tokenizer
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        if getattr(tok, "chat_template", None):
            try:
                prompts = [
                    tok.apply_chat_template(
                        [{"role": "user", "content": p}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for p in prompts
                ]
            except Exception:
                prompts = [p + "\n\nA:" for p in prompts]
        else:
            prompts = [p + "\n\nA:" for p in prompts]
        prev_side = tok.padding_side
        tok.padding_side = "left"
        try:
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        finally:
            tok.padding_side = prev_side
        prompt_ids = enc["input_ids"].to(self.device)
        prompt_attn = enc["attention_mask"].to(self.device)
        prompt_len = prompt_ids.shape[1]

        was_training = generator.training
        prev_use_cache = getattr(generator.config, "use_cache", None)
        generator.eval()
        if hasattr(generator.config, "use_cache"):
            generator.config.use_cache = True
        with torch.no_grad():
            full_ids = generator.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_attn,
                do_sample=True,
                temperature=self.config.temperature,
                top_k=0,
                max_new_tokens=self.config.gkd_max_new_tokens,
                pad_token_id=pad_id,
            )
        if hasattr(generator.config, "use_cache") and prev_use_cache is not None:
            generator.config.use_cache = prev_use_cache
        if was_training:
            generator.train()

        non_pad = (full_ids != pad_id).long()
        attention_mask = non_pad.clone()
        response_mask = torch.zeros_like(full_ids)
        response_mask[:, prompt_len:] = 1
        response_mask = response_mask * non_pad
        return full_ids, attention_mask, response_mask

    def _forward_step(self, batch: Dict[str, Any]) -> tuple[torch.Tensor, float, float, float]:
        """Compute the (KL/GKD + CE) loss for one batch.

        With probability ``lmbda`` the batch is replaced by student-generated
        on-policy sequences (GKD); when ``seq_kd`` is set and this is an
        off-policy step, the batch is replaced by teacher-generated sequences.
        Otherwise the fixed dataset batch is used (current behavior).
        """
        on_policy = self.config.lmbda > 0.0 and random.random() < self.config.lmbda
        if on_policy:
            input_ids, attention_mask, response_mask = self._generate_on_policy_batch(
                batch["prompts"], self.student,
            )
        elif self.config.seq_kd:
            input_ids, attention_mask, response_mask = self._generate_on_policy_batch(
                batch["prompts"], self.teacher,
            )
        else:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            response_mask = batch["response_mask"].to(self.device)

        if self._is_sft:
            student_logits = self._student_logits(input_ids, attention_mask)
            loss = sft_ce_loss(student_logits, input_ids, response_mask)
            ce_val = float(loss.item())
            return loss, 0.0, ce_val, ce_val

        teacher_logits = self._teacher_logits_for_batch(batch, input_ids, attention_mask)
        self._clear_cuda_cache()

        if self.config.only_correct and not on_policy and not self.config.seq_kd:
            correct = teacher_correct_mask(
                teacher_logits, response_mask.cpu(), self.tokenizer, batch["answers"],
            ).to(response_mask.device)
            response_mask = response_mask * correct.unsqueeze(1)

        student_logits = self._student_logits(input_ids, attention_mask)
        loss, kl_val, ce_val = self._distill_term(student_logits, teacher_logits, response_mask)
        return loss, kl_val, ce_val, float(loss.item())

    def _backward_graph_loss(self, batch: Dict[str, Any], grad_scale: float) -> float:
        """Reuse graph_loss/training.py's compare-tokens path, exactly as distillation.py.

        ``backward_batch_graph_loss`` dispatches to the compare-tokens routine
        internally when ``config.compare_n_tokens`` is set, and performs the
        graph backward itself (accumulating into the student's existing grads).
        Returns the detached graph-loss value for logging.
        """
        from graph_loss.training import backward_batch_graph_loss

        def _log_graph_progress(done: int, total: int) -> None:
            print(f"    [graph] prompt {done}/{total}", flush=True)

        graph_loss, _graph_metrics = backward_batch_graph_loss(
            prompts=batch["prompts"],
            student_adapter=self.student_graph_adapter,
            config=self.graph_loss_config,
            device=self.device,
            loss_scale=self.config.lambda_graph * grad_scale,
            teacher_cache=None,
            teacher_adapter=self.teacher_graph_adapter,
            answers=batch["answers"],
            on_prompt_done=_log_graph_progress,
        )
        return float(graph_loss.item())

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

    def _clear_cuda_cache(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train_epoch(self, epoch: int, *, max_steps: Optional[int] = None) -> Dict[str, float]:
        self.student.train()
        agg_kl = 0.0
        agg_ce = 0.0
        agg_graph = 0.0
        agg_total = 0.0
        n_steps = 0
        interval_kl = 0.0
        interval_ce = 0.0
        interval_graph = 0.0
        interval_total = 0.0
        interval_steps = 0
        accum_steps = max(1, self.config.grad_accum_steps)
        pending_accum = 0
        self.optimizer.zero_grad(set_to_none=True)
        # per-window accumulators (one window = one optimizer step)
        _win_kl = 0.0; _win_ce = 0.0; _win_graph = 0.0; _win_total = 0.0; _win_n = 0

        for batch in self.loader:
            loss, kl_val, ce_val, total_val = self._forward_step(batch)
            if not torch.isfinite(loss).item():
                continue

            grad_scale = 1.0 / accum_steps
            (loss * grad_scale).backward()
            del loss
            # Feature 2: add the graph-loss gradient on top of the KL/GKD gradient,
            # exactly as distillation.py does (graph backward happens inside the call).
            graph_val = 0.0
            if self._use_graph and self._train_step + 1 >= self._graph_start_step:
                kl_grads = self._clone_student_grads() if self.config.track_loss_grads else None
                self._clear_cuda_cache()
                graph_val = self._backward_graph_loss(batch, grad_scale)
                total_val = total_val + self.config.lambda_graph * graph_val
                if kl_grads is not None:
                    _grad_metrics: Dict[str, float] = {}
                    self._record_loss_grad_metrics(_grad_metrics, kl_grads)
                self._clear_cuda_cache()
            else:
                if self.config.track_loss_grads:
                    _grad_metrics = {}
                    self._record_kl_only_grad_metrics(_grad_metrics)
                else:
                    _grad_metrics = {}
            pending_accum += 1
            _win_kl += kl_val; _win_ce += ce_val; _win_graph += graph_val; _win_total += total_val; _win_n += 1

            should_step = (
                pending_accum >= accum_steps
                or (max_steps is not None and n_steps + 1 >= max_steps)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.config.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                pending_accum = 0

                # average losses over the batches in this optimizer step
                _denom = max(_win_n, 1)
                kl_val = _win_kl / _denom
                ce_val = _win_ce / _denom
                graph_val = _win_graph / _denom
                total_val = _win_total / _denom
                _win_kl = 0.0; _win_ce = 0.0; _win_graph = 0.0; _win_total = 0.0; _win_n = 0

                self._train_step += 1
                self.history["train_step"].append(self._train_step)
                if self.config.kl_weight != 0:
                    self.history["step_kl_loss"].append(kl_val)
                if self.config.ce_weight != 0:
                    self.history["step_ce_loss"].append(ce_val)
                if self._use_graph:
                    self.history["step_graph_loss"].append(graph_val)
                self.history["step_total_loss"].append(total_val)
                agg_kl += kl_val
                agg_ce += ce_val
                agg_graph += graph_val
                agg_total += total_val
                interval_kl += kl_val
                interval_ce += ce_val
                interval_graph += graph_val
                interval_total += total_val
                n_steps += 1
                interval_steps += 1
                self._save_history()

                if self.config.save_interval > 0 and self._train_step % self.config.save_interval == 0:
                    self._save_checkpoint_at_step(self._train_step)

                if self._train_step == 1 or self._train_step % max(1, self.config.step_log_interval) == 0:
                    eval_n = max(1, self.config.eval_every_n_steps)
                    did_eval = self._train_step % eval_n == 0
                    if did_eval:
                        self._clear_cuda_cache()
                        self._step_log_eval_accuracy = self._evaluate_model(self.student)
                        self._clear_cuda_cache()
                        self.student.train()
                        self.history["accuracy"].append(self._step_log_eval_accuracy)
                        self.history["accuracy_step"].append(self._train_step)
                    denom = max(interval_steps, 1)
                    kl_avg = interval_kl / denom
                    ce_avg = interval_ce / denom
                    graph_avg = interval_graph / denom
                    total_avg = interval_total / denom
                    acc_str = f" | Acc {self._step_log_eval_accuracy:.4f}" if did_eval else ""
                    kl_label = "JSD" if self._use_gkd else "KL"
                    kl_str = f" | {kl_label} {kl_avg:.4f}" if self.config.kl_weight != 0 else ""
                    ce_str = f" | CE {ce_avg:.4f}" if self.config.ce_weight != 0 else ""
                    graph_str = f" | Graph {graph_avg:.4f}" if self._use_graph else ""
                    grad_str = ""
                    if self.config.track_loss_grads:
                        grad_str = (
                            f" | grad KL {_grad_metrics.get('kl_loss_grad', 0.0):.3e}"
                            f" Graph {_grad_metrics.get('graph_loss_grad', 0.0):.3e}"
                            f" cos {_grad_metrics.get('loss_cossim', 0.0):.4f}"
                        )
                    print(f"  step {self._train_step:04d} | Total {total_avg:.4f}{kl_str}{ce_str}{graph_str}{acc_str}{grad_str}")
                    interval_kl = 0.0
                    interval_ce = 0.0
                    interval_graph = 0.0
                    interval_total = 0.0
                    interval_steps = 0
                    self._save_curves()

                if max_steps is not None and n_steps >= max_steps:
                    break

            self._clear_cuda_cache()

        if pending_accum > 0:
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.config.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        denom = max(n_steps, 1)
        metrics = {"total_loss": agg_total / denom}
        if self.config.kl_weight != 0:
            metrics["kl_loss"] = agg_kl / denom
        if self.config.ce_weight != 0:
            metrics["ce_loss"] = agg_ce / denom
        if self._use_graph:
            metrics["graph_loss"] = agg_graph / denom
        return metrics

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
            ckpt_dir = os.path.join(cfg.save_dir, f"step_{self._resume_step}_checkpoint")
            opt_path = os.path.join(ckpt_dir, "optimizer.pt")
            if os.path.isfile(opt_path):
                self.optimizer.load_state_dict(torch.load(opt_path, map_location=self.device))
                print(f"Restored optimizer state from step {self._resume_step}.")
            sched_path = os.path.join(ckpt_dir, "scheduler.pt")
            if os.path.isfile(sched_path):
                self.scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))
                print(f"Restored scheduler state from step {self._resume_step}.")
            print(f"Warm-starting from step {self._train_step + 1}.")
        elif cfg.skip_baseline_eval:
            self._step_log_eval_accuracy = 0.0
        else:
            print("Evaluating baselines...")
            student_base = self._evaluate_model(self.student)
            self.student.train()
            self.history["student_baseline"] = student_base
            label = cfg.sft_model if self._is_sft else "student"
            print(f"  {label} baseline accuracy: {student_base:.4f}")
            if not self._is_sft:
                teacher_base = self._evaluate_model(self.teacher)
                self.history["teacher_baseline"] = teacher_base
                print(f"  Teacher baseline accuracy: {teacher_base:.4f}")
            self._step_log_eval_accuracy = student_base

        print("=" * 60)
        mode_bits = []
        if self._is_sft:
            mode_bits.append(f"SFT-{cfg.sft_model}")
        if self._use_gkd:
            mode_bits.append("GKD")
        if self._use_graph:
            mode_bits.append("Graph")
        mode_tag = f" ({'+'.join(mode_bits)})" if mode_bits else ""
        print(f"GSM8K CoT KL Distillation{mode_tag}")
        print(f"  Run dir:          {cfg.save_dir}")
        print(f"  Steps:            {self._train_step + 1}..{cfg.steps}")
        print(f"  Batch size:       {cfg.batch_size}")
        print(f"  Grad accum steps: {cfg.grad_accum_steps}")
        print(f"  LR:               {cfg.learning_rate}")
        print(f"  Temperature:      {cfg.temperature}")
        if self._use_gkd:
            print(f"  lmbda (on-policy):{cfg.lmbda}")
            print(f"  beta (JSD):       {cfg.beta}")
            print(f"  seq_kd:           {cfg.seq_kd}")
        if self._use_graph:
            print(f"  lambda_graph:     {cfg.lambda_graph}")
            print(f"  graph_loss_type:  {cfg.graph_loss_type}")
            print(f"  compare_n_tokens: {cfg.compare_n_tokens}")
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
            kl_label = "JSD" if self._use_gkd else "KL"
            kl_part = f" | {kl_label}={epoch_metrics.get('kl_loss', float('nan')):.4f}" if cfg.kl_weight != 0 else ""
            ce_part = f" | CE={epoch_metrics.get('ce_loss', float('nan')):.4f}" if cfg.ce_weight != 0 else ""
            graph_part = f" | Graph={epoch_metrics.get('graph_loss', float('nan')):.4f}" if self._use_graph else ""
            print(
                f"Pass {epoch + 1}: Total={epoch_metrics.get('total_loss', float('nan')):.4f}"
                f"{kl_part}{ce_part}{graph_part} | Acc={self._step_log_eval_accuracy:.4f} | Step={self._train_step}/{cfg.steps}"
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
        torch.save(self.scheduler.state_dict(), os.path.join(path, "scheduler.pt"))
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
        total_series = self.history.get("step_total_loss", [])
        kl_series = self.history.get("step_kl_loss", [])
        ce_series = self.history.get("step_ce_loss", [])
        acc_series = self.history.get("accuracy", [])
        acc_steps = self.history.get("accuracy_step", list(range(1, len(acc_series) + 1)))

        graph_series = self.history.get("step_graph_loss", [])
        kl_title = "JSD Loss" if self._use_gkd else "KL Loss"
        plots = [("Total Loss", loss_steps[: len(total_series)], total_series, None)]
        if self.config.kl_weight != 0:
            plots.append((kl_title, loss_steps[: len(kl_series)], kl_series, None))
        if self.config.ce_weight != 0:
            plots.append(("CE Loss", loss_steps[: len(ce_series)], ce_series, None))
        if self._use_graph:
            plots.append(("Graph Loss", loss_steps[: len(graph_series)], graph_series, None))
        plots.append(("Accuracy", acc_steps[: len(acc_series)], acc_series, (0, 1)))

        n = len(plots)
        cols = min(n, 2)
        rows = (n + cols - 1) // cols
        fig, axes_grid = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows))
        if n == 1:
            axes_list = [axes_grid]
        elif rows == 1:
            axes_list = list(axes_grid)
        else:
            axes_list = list(axes_grid.flat)

        for ax, (title, x, y, ylim) in zip(axes_list, plots):
            ax.plot(x, y, marker="o", markersize=2)
            ax.set_title(title)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.3)
        for ax in axes_list[n:]:
            ax.set_visible(False)

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
    parser.add_argument("--warmup-ratio", type=float, default=0.0, dest="warmup_ratio")
    parser.add_argument("--kl-weight", type=float, default=1.0, dest="kl_weight")
    parser.add_argument("--ce-weight", type=float, default=0.0, dest="ce_weight")
    parser.add_argument("--only-correct", action="store_true", default=False, dest="only_correct")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--test-limit", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=200)
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "results", "cot_distillation"),
    )
    parser.add_argument("--step-log-interval", type=int, default=1)
    parser.add_argument("--eval-every-n-steps", type=int, default=1, dest="eval_every_n_steps")
    parser.add_argument("--max-response-tokens", type=int, default=256)
    parser.add_argument("--save-interval", type=int, default=0)
    parser.add_argument("--skip-baseline-eval", action="store_true", default=False, dest="skip_baseline_eval")
    parser.add_argument("--num-fewshot", type=int, default=0, dest="num_fewshot",
                        help="Number of few-shot examples prepended to each prompt (drawn from last 473 train examples).")
    parser.add_argument("--resume-step", type=int, default=None, dest="resume_step")
    parser.add_argument("--checkpoint-run", default=None, metavar="PATH")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sft-model", choices=["student", "teacher"], default=None, dest="sft_model",
        help="SFT the specified model on GSM8K gold responses (forces kl_weight=0, ce_weight=1).",
    )
    parser.add_argument(
        "--track-loss-grads",
        action="store_true",
        dest="track_loss_grads",
        help="Log KL and graph gradient norms plus their cosine similarity.",
    )

    # ── Feature 1: native on-policy GKD (arXiv:2306.13649) ──────────────────
    gkd = parser.add_argument_group("GKD (on-policy generalized knowledge distillation)")
    gkd.add_argument(
        "--lmbda", type=float, default=0.0,
        help="Per-step probability of training on STUDENT-generated (on-policy) sequences. "
             "0.0 (default) = pure off-policy KL (current behavior).",
    )
    gkd.add_argument(
        "--beta", type=float, default=0.0,
        help="JSD interpolation: 0 = forward KL (identical to current kl_loss), "
             "1 = reverse KL, 0<beta<1 = generalized JSD. Default 0.0.",
    )
    gkd.add_argument(
        "--gkd-max-new-tokens", type=int, default=256, dest="gkd_max_new_tokens",
        help="Max new tokens to sample when generating on-policy sequences.",
    )
    gkd.add_argument(
        "--seq-kd", action="store_true", default=False, dest="seq_kd",
        help="On off-policy steps, train on TEACHER-generated sequences (sequence-level KD).",
    )

    # ── Feature 2: CoT graph distillation (reuses graph_loss/training.py) ───
    # Flag names mirror distillation.py so usage is consistent.
    graph = parser.add_argument_group("Graph distillation (compare-tokens)")
    graph.add_argument(
        "--lambda-graph", type=float, default=0.0, dest="lambda_graph",
        help="Weight of the graph auxiliary loss. 0.0 (default) loads NO graph machinery.",
    )
    graph.add_argument(
        "--graph-loss-type", choices=["jsd", "kld", "mse", "mse-norm", "mse-scale"],
        default="kld", dest="graph_loss_type",
    )
    graph.add_argument(
        "--compare-n-tokens", type=int, default=3, dest="compare_n_tokens",
        help="Number of top-KL response token positions to compute graph loss on.",
    )
    graph.add_argument(
        "--tokens-dla-nodes", action="store_true", default=False, dest="tokens_dla_nodes",
        help="Include arg-token supernodes and a DLA supernode in both graphs.",
    )
    graph.add_argument("--dtype", choices=DTYPE_CHOICES, default="bfloat16", dest="dtype")
    graph.add_argument("--teacher-prop-neurons-per-layer", type=float, default=0.1)
    graph.add_argument("--student-prop-neurons-per-layer", type=float, default=0.1)
    graph.add_argument(
        "--graph-top-k-logits", type=float, default=0.95, dest="graph_top_k_logits",
        help="Cumulative probability threshold in (0, 1] for logit nodes.",
    )
    graph.add_argument("--teacher-graph-batch-size", type=int, default=32)
    graph.add_argument("--student-graph-batch-size", type=int, default=1)
    graph.add_argument("--teacher-nodes-per-label", type=int, default=10)
    graph.add_argument("--student-nodes-per-label", type=int, default=10)
    graph.add_argument(
        "--graph-start-step", type=int, default=1,
        help="Step (1-indexed, inclusive) at which graph loss is first applied.",
    )
    graph.add_argument("--graph-verbose", action="store_true", default=False, dest="graph_verbose")
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

    all_train = load_prompt_answer_json(train_path)
    all_train_items = list(all_train.items())
    train_data = dict(all_train_items[:TRAIN_SPLIT_SIZE])
    fewshot_pool = dict(all_train_items[TRAIN_SPLIT_SIZE:TRAIN_SPLIT_SIZE + FEWSHOT_POOL_SIZE])

    if args.sft_model == "teacher":
        train_data = dict(all_train_items)

    test_data = load_prompt_answer_json(test_path)
    if args.test_limit is not None:
        test_data = dict(list(test_data.items())[: args.test_limit])
    print(f"Train: {len(train_data)} examples | Fewshot pool: {len(fewshot_pool)} | Test: {len(test_data)} examples")

    device = get_default_device()
    if args.sft_model == "teacher":
        student, tokenizer = None, None
    else:
        student, tokenizer = load_student_model_for_distillation(student_source, args.student_model, device)

    trainer = CoTDistillationTrainer(
        config=CoTDistillationConfig(
            teacher_model=args.teacher_model,
            student_model=args.student_model,
            steps=args.steps,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            learning_rate=args.lr,
            warmup_ratio=args.warmup_ratio,
            kl_weight=args.kl_weight,
            ce_weight=args.ce_weight,
            only_correct=args.only_correct,
            temperature=args.temperature,
            grad_clip=args.grad_clip,
            max_response_tokens=args.max_response_tokens,
            eval_batch_size=args.eval_batch_size,
            save_dir=run_dir,
            step_log_interval=args.step_log_interval,
            eval_every_n_steps=args.eval_every_n_steps,
            save_interval=args.save_interval,
            benchmark_eval_limit=args.test_limit,
            skip_baseline_eval=args.skip_baseline_eval,
            num_fewshot=args.num_fewshot,
            seed=args.seed,
            device=device,
            lmbda=args.lmbda,
            beta=args.beta,
            gkd_max_new_tokens=args.gkd_max_new_tokens,
            seq_kd=args.seq_kd,
            lambda_graph=args.lambda_graph,
            graph_loss_type=args.graph_loss_type,
            compare_n_tokens=args.compare_n_tokens,
            tokens_dla_nodes=args.tokens_dla_nodes,
            graph_dtype=resolve_torch_dtype(args.dtype),
            teacher_prop_neurons_per_layer=args.teacher_prop_neurons_per_layer,
            student_prop_neurons_per_layer=args.student_prop_neurons_per_layer,
            graph_top_k_logits=args.graph_top_k_logits if args.graph_top_k_logits > 0 else None,
            teacher_graph_batch_size=args.teacher_graph_batch_size,
            student_graph_batch_size=args.student_graph_batch_size,
            teacher_nodes_per_label=args.teacher_nodes_per_label,
            student_nodes_per_label=args.student_nodes_per_label,
            graph_start_step=args.graph_start_step,
            graph_verbose=args.graph_verbose,
            sft_model=args.sft_model,
            track_loss_grads=args.track_loss_grads,
        ),
        train_data=train_data,
        test_data=test_data,
        fewshot_pool=fewshot_pool,
        tokenizer=tokenizer,
        student=student,
        resume_step=args.resume_step,
    )
    history = trainer.train()
    if "accuracy" in history and history["accuracy"]:
        print(f"Best accuracy: {max(history['accuracy']):.4f}")


if __name__ == "__main__":
    main()

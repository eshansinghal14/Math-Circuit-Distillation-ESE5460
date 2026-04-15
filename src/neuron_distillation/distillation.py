"""
Neuron-Cluster Circuit Distillation

Extends circuit distillation from layer-level to neuron-cluster-level
alignment.  Instead of hooking individual MLP layers and comparing their
full outputs, this module:

  1. Hooks ``up_proj`` in ALL MLP layers of both student and teacher,
     producing per-layer intermediate activations.
  2. Flattens activations across layers at the last-token position to get
     a single (B, num_layers * intermediate_size) vector per example.
  3. For each paired cluster (discovered via ablation-based importance
     matching), extracts the corresponding neuron subsets and computes
     CKA between student and teacher.
  4. The total loss uses masked per-position KL at answer-token predictions only
     (not the step that predicts EOS after the answer),
     plus ``lambda * L_cluster_align`` (CKA). No ``batchmean`` over the full vocab grid.

Pipeline prerequisites (run before this module):
  - Neuron clustering  (clustering.py)
  - Cluster ablation   (ablation.py)
  - Cluster pairing    (pairing.py)
"""

import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from torch.utils.flop_counter import FlopCounterMode
    import torch.utils.flop_counter as _flop_counter_mod

    def _gqa_sdpa_flop_count(query_shape, key_shape, value_shape, **kwargs):
        """GQA-aware replacement for sdpa_flop_count.

        PyTorch's built-in asserts q_heads == kv_heads, which breaks on models
        using grouped-query attention (e.g. Llama-3-8B: 32 Q heads, 8 KV heads).
        We use q_heads for the FLOP formula since KV is broadcast to match Q.
        """
        b, h_q, s_q, d_q = query_shape
        _b, _h_kv, s_kv, d_k = key_shape
        _b2, _h_kv2, _s_kv, d_v = value_shape
        # QK^T: 2 * b * h_q * s_q * s_kv * d_q
        # softmax(QK^T) @ V: 2 * b * h_q * s_q * d_v * s_kv
        return 2 * b * h_q * s_q * s_kv * (d_q + d_v)

    _flop_counter_mod.sdpa_flop_count = _gqa_sdpa_flop_count

except ImportError:
    FlopCounterMode = None  # type: ignore[misc, assignment]

from cka_loss import leading_eigenvalue_hkh, linear_cka_efficient
from neuron_distillation.pairing import (
    ClusterMapping,
    _load_single_ablation_performance,
    adjust_ablation_drops_for_poly_importance,
    create_cluster_mapping,
    default_random_ablation_poly_json_paths,
)
from utils import (
    EVAL_MAX_NEW_TOKENS,
    STUDENT_MODEL_DIR,
    STUDENT_WEIGHTS_FILE,
    _extract_int_after_equals,
    json_to_prompt_answer_dict,
    load_training_state,
    patch_tokenizer_no_special_tokens,
    rm_dir_tree,
    save_training_state,
    training_state_path,
)


def _seed_all(seed: int) -> None:
    """Set Python / NumPy / PyTorch RNGs (DataLoader shuffle uses the torch generator passed separately)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ClusterPairInfo:
    """A single student<->teacher cluster pair with full neuron indices."""

    subclass: int
    student_cluster_idx: int
    teacher_cluster_idx: int
    student_neuron_indices: torch.Tensor
    teacher_neuron_indices: torch.Tensor
    # Student-side score used for pairing / CKA weighting (residual vs poly when enabled).
    importance: float


@dataclass
class ClusterDistillationConfig:
    teacher_model: str = "meta-llama/Meta-Llama-3-8B"
    student_model: str = "meta-llama/Llama-3.2-1B"
    # ``circuit``: KL + cluster CKA (hooks). ``standard``: KL only (no hooks / no ablation).
    distillation_mode: str = "circuit"

    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    temperature: float = 2.0
    grad_clip: float = 1.0

    lambda_cluster: float = 0.01
    # If in (0, 1], add masked CE with weight ``hard_ce_weight`` and scale KL by ``1 - hard_ce_weight``.
    hard_ce_weight: float = 0.0
    # If set ``(lo, hi)`` (inclusive), build vocab indices from ``str(n)`` for ``n`` in that
    # range; KL sums ``p_t * (log p_t - log q_s)`` only over those vocab columns (full softmax
    # on both sides; mask zeros other dimensions in the per-vocab KL sum).
    kl_mask_range: Optional[Tuple[int, int]] = None
    importance_weighting: bool = True
    # If True, multiply each pair's CKA loss weight by (|student cluster| / full student MLP width).
    cluster_size_weighting: bool = False

    eval_every: int = 1
    # Greedy test accuracy: prompts batched for ``generate`` (independent of training batch size).
    eval_batch_size: int = 50
    # In-epoch step log every N batches.
    step_log_interval: int = 50
    save_every: int = 5
    save_best: bool = False
    eval_max_new_tokens: int = EVAL_MAX_NEW_TOKENS
    # Greedy-eval debug: print first N prompts + top-5 softmax at ``temperature`` (student & teacher).
    eval_print_samples: int = 0
    save_dir: str = "results/cluster-distillation"
    # Count FLOPs (FlopCounterMode) only on epochs where ``epoch_index % N == 0`` (0-based).
    # 1 = every epoch; 0 = never; N>1 = every Nth epoch (0, N, 2N, …). ~10% step overhead when active.
    count_flops_every: int = 1

    # Log global L2 grads for KL vs λ·CKA (``autograd.grad``; ~2× work). When True, the
    # λ·CKA contribution to ``param.grad`` is scaled by ``||g_KL|| / ||g_{λ·CKA}||`` (KL vs
    # scaled CKA term) so the step matches unscaled KL plus that ratio times the CKA grads.
    log_kl_cka_grad_norms: bool = False

    # Reproducible train loader shuffle / batch order (also call :func:`_seed_all` in the trainer).
    seed: int = 42

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def checkpoint_every(self) -> int:
        """Alias for ``save_every`` (older code / notebooks referenced this name)."""
        return self.save_every


def token_ids_for_integer_range(tokenizer, lo: int, hi: int) -> torch.LongTensor:
    """Token ID set for decimal strings ``str(n)``, for each ``n`` in ``[lo, hi]`` inclusive."""
    if lo > hi:
        raise ValueError(f"kl_mask_range requires lo <= hi, got ({lo}, {hi})")
    seen = set()
    for n in range(lo, hi + 1):
        for tid in tokenizer.encode(str(n), add_special_tokens=False):
            seen.add(int(tid))
    return torch.tensor(sorted(seen), dtype=torch.long)


def _masked_kl_loss_restricted(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    kl_mask: torch.Tensor,
    temperature: float,
    restrict_token_ids: Optional[torch.Tensor],
) -> torch.Tensor:
    """KL from full teacher/student distributions, optional vocab mask on the KL sum.

    Uses ``p_t * (log p_t - log q_s)`` per vocab entry (``p_t`` = teacher softmax, ``log q_s`` =
    student log-softmax), then sums over vocab—optionally zeroing non-``restrict_token_ids``
    columns so only those dimensions contribute. Sequence masking ``kl_mask`` is unchanged.
    """
    t = temperature
    log_p_t = F.log_softmax(teacher_logits.float() / t, dim=-1)
    p_t = log_p_t.exp()
    log_q_s = F.log_softmax(student_logits.float() / t, dim=-1)
    kl_per_vocab = p_t * (log_p_t - log_q_s)
    if restrict_token_ids is not None:
        device = kl_per_vocab.device
        r = restrict_token_ids.to(device=device, dtype=torch.long)
        vocab_mask = torch.zeros_like(kl_per_vocab)
        vocab_mask[..., r] = 1.0
        kl_per_vocab = kl_per_vocab * vocab_mask
    kl_per_token = kl_per_vocab.sum(dim=-1)
    return (
        (kl_per_token * kl_mask).sum()
        / kl_mask.sum().clamp_min(1.0)
        * (t**2)
    )


# ---------------------------------------------------------------------------
# Loading cluster pairs
# ---------------------------------------------------------------------------

def load_cluster_pairs(
    student_ablation_path: str,
    teacher_ablation_path: str,
    student_clusters_dir: str,
    teacher_clusters_dir: str,
    class_clusters_student: List[int],
    class_clusters_teacher: List[int],
    top_k_per_subclass: Optional[int] = None,
    importance_vs_poly: bool = True,
    student_poly_json: Optional[str] = None,
    teacher_poly_json: Optional[str] = None,
    student_model_name: Optional[str] = None,
    teacher_model_name: Optional[str] = None,
) -> List[ClusterPairInfo]:
    """Build a flat list of cluster pairs with neuron indices and importance.

    Args:
        student_ablation_path: ``ablation_performance.json`` for the student.
        teacher_ablation_path: ``ablation_performance.json`` for the teacher.
        student_clusters_dir: Dir containing
            ``subclass_<N>_clusters/k<K>.pt`` for the student model.
        teacher_clusters_dir: Same for the teacher model.
        class_clusters_student: List of *k* values per subclass for student.
        class_clusters_teacher: Same for teacher.
        top_k_per_subclass: Keep only the top-k most important student
            clusters per subclass.
        importance_vs_poly: If True (default), replace raw drops with
            ``drop − poly(|C|/D)`` (signed residual) using default or given poly JSON paths.
        student_poly_json / teacher_poly_json: Optional overrides for poly files.
        student_model_name / teacher_model_name: HF ids (required if poly adjustment on).

    Returns:
        List of :class:`ClusterPairInfo` sorted by importance (descending).
    """
    delta_s = _load_single_ablation_performance(student_ablation_path)
    delta_t = _load_single_ablation_performance(teacher_ablation_path)

    if importance_vs_poly:
        if not student_model_name or not teacher_model_name:
            raise ValueError(
                "load_cluster_pairs(..., importance_vs_poly=True) requires "
                "student_model_name and teacher_model_name.",
            )
        sp = student_poly_json or default_random_ablation_poly_json_paths()[0]
        tp = teacher_poly_json or default_random_ablation_poly_json_paths()[1]
        delta_s, delta_t = adjust_ablation_drops_for_poly_importance(
            delta_s,
            delta_t,
            student_clusters_dir,
            teacher_clusters_dir,
            class_clusters_student,
            class_clusters_teacher,
            sp,
            tp,
            student_model_name,
            teacher_model_name,
        )

    mappings: List[ClusterMapping] = create_cluster_mapping(
        delta_s, delta_t, top_k_student=top_k_per_subclass,
    )

    pairs: List[ClusterPairInfo] = []

    for m in mappings:
        sc = m.subclass
        s_k = class_clusters_student[sc]
        t_k = class_clusters_teacher[sc]

        s_cluster_path = os.path.join(
            student_clusters_dir,
            f"subclass_{sc}_clusters/k{s_k}.pt",
        )
        t_cluster_path = os.path.join(
            teacher_clusters_dir,
            f"subclass_{sc}_clusters/k{t_k}.pt",
        )

        if not os.path.exists(s_cluster_path) or not os.path.exists(t_cluster_path):
            continue

        s_ckpt = torch.load(s_cluster_path, map_location="cpu")
        t_ckpt = torch.load(t_cluster_path, map_location="cpu")

        s_c2i = s_ckpt["cluster_to_indices"]
        t_c2i = t_ckpt["cluster_to_indices"]

        s_idx_key = m.student_cluster_idx
        t_idx_key = m.teacher_cluster_idx

        if s_idx_key not in s_c2i or t_idx_key not in t_c2i:
            continue

        s_neuron_idx = s_c2i[s_idx_key]
        t_neuron_idx = t_c2i[t_idx_key]

        if not isinstance(s_neuron_idx, torch.Tensor):
            s_neuron_idx = torch.tensor(s_neuron_idx, dtype=torch.long)
        if not isinstance(t_neuron_idx, torch.Tensor):
            t_neuron_idx = torch.tensor(t_neuron_idx, dtype=torch.long)

        if s_neuron_idx.numel() == 0 or t_neuron_idx.numel() == 0:
            continue

        pairs.append(ClusterPairInfo(
            subclass=sc,
            student_cluster_idx=m.student_cluster_idx,
            teacher_cluster_idx=m.teacher_cluster_idx,
            student_neuron_indices=s_neuron_idx,
            teacher_neuron_indices=t_neuron_idx,
            importance=m.student_importance,
        ))

    pairs.sort(key=lambda p: p.importance, reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# Activation cache (hooks up_proj across all layers)
# ---------------------------------------------------------------------------

class ClusterActivationCache:
    """Hooks ``up_proj`` in every MLP layer and provides flattened activations.

    Neuron indices from clustering are defined over the flattened vector
    ``[layer_0.up_proj_out | layer_1.up_proj_out | ...]`` so we must
    collect all layers and concatenate.
    """

    def __init__(self):
        self.layer_activations: Dict[int, torch.Tensor] = {}
        self.hooks: List = []

    def _make_hook(self, layer_idx: int, detach: bool):
        def hook(module, _input, output):
            self.layer_activations[layer_idx] = output.detach() if detach else output
        return hook

    def register_hooks(self, model, detach: bool = False):
        """Register forward hooks on ``up_proj`` in every transformer block.

        Args:
            model: A HuggingFace causal LM (Llama-style).
            detach: If True, stored activations are detached from the
                computation graph (use for teacher).
        """
        self.clear()
        for i, layer in enumerate(model.model.layers):
            h = layer.mlp.up_proj.register_forward_hook(
                self._make_hook(i, detach=detach)
            )
            self.hooks.append(h)

    def get_flattened_last_token(
        self, attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Concatenate all per-layer activations at the last non-pad token.

        Returns:
            ``(B, num_layers * intermediate_size)`` in float32.
        """
        layers = sorted(self.layer_activations.keys())
        parts = []
        for i in layers:
            act = self.layer_activations[i]  # (B, S, D_intermediate)
            if attention_mask is not None:
                last_idx = attention_mask.sum(dim=1).long() - 1  # (B,)
                last_tok = act[torch.arange(act.size(0), device=act.device), last_idx]
            else:
                last_tok = act[:, -1, :]  # (B, D_intermediate)
            parts.append(last_tok.float())
        return torch.cat(parts, dim=-1)

    def get_flattened_per_token(self) -> torch.Tensor:
        """Concatenate all per-layer activations, keeping the token dimension.

        MLP/FFN operates independently per token, so each position carries
        unique information.  This returns per-token activations so that CKA
        can be computed at each token position separately and then averaged,
        matching how neuron clustering defined clusters (across all tokens).

        Returns:
            ``(B, T, num_layers * intermediate_size)`` in float32.
        """
        layers = sorted(self.layer_activations.keys())
        parts = []
        for i in layers:
            act = self.layer_activations[i].float()  # (B, T, D_intermediate)
            parts.append(act)
        return torch.cat(parts, dim=-1)  # (B, T, D_total)

    def clear(self):
        self.layer_activations.clear()
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ---------------------------------------------------------------------------
# Grad norms for KL vs CKA terms (optional logging)
# ---------------------------------------------------------------------------

def _grad_tuple_global_l2_norm(grads: Tuple[Optional[torch.Tensor], ...]) -> float:
    """Global L2 norm of a tuple of per-parameter gradients (``None`` ignored)."""
    sq = 0.0
    for g in grads:
        if g is not None:
            sq += float(g.detach().float().pow(2).sum().item())
    return sq ** 0.5


def kc_lam1_metric_key(subclass: int, student_c: int, teacher_c: int) -> str:
    """JSON-safe key for per-cluster-pair ``λ_max(K_c)`` (subclass, student cluster, teacher cluster)."""
    return f"s{subclass}_us{student_c}_ut{teacher_c}"


# ---------------------------------------------------------------------------
# Cluster-level CKA alignment loss
# ---------------------------------------------------------------------------

class ClusterAlignmentLoss(nn.Module):
    """Importance-weighted CKA over paired neuron clusters.

    Computes CKA **per token position** and averages across tokens.
    This respects the fact that MLP/FFN operates independently at each
    position, so per-token alignment is more informative than aligning
    a pre-averaged representation.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        student_acts: torch.Tensor,
        teacher_acts: torch.Tensor,
        cluster_pairs: List[ClusterPairInfo],
        attention_mask: Optional[torch.Tensor] = None,
        importance_weighting: bool = True,
        cluster_size_weighting: bool = False,
        compute_kc_leading_eigenvalue: bool = False,
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        """
        Args:
            student_acts: ``(B, T, D_student_total)`` per-token up_proj activations.
            teacher_acts: ``(B, T, D_teacher_total)`` same for teacher.
            cluster_pairs: List of :class:`ClusterPairInfo`.
            attention_mask: ``(B, T)`` binary mask; 1 = real token, 0 = pad.
            importance_weighting: Scale each pair's loss by its importance.
            cluster_size_weighting: If True, multiply each pair's weight by
                ``|student cluster| / D_student_total`` (flattened MLP width).
            compute_kc_leading_eigenvalue: If True, compute per-pair mean
                ``λ_max(K_c)`` for student cluster activations with
                ``K_c = H X X^T H`` (column centering ``H``); used for logging only.

        Returns:
            ``(total_loss, cka_scores, kc_lam1_scores)`` where ``kc_lam1_scores`` maps
            the same keys as ``cka_scores`` to mean ``λ_max(K_c)`` over valid tokens, or
            is empty when ``compute_kc_leading_eigenvalue`` is False.
        """
        device = student_acts.device
        B, T, d_student_total = student_acts.shape

        if attention_mask is not None:
            valid_mask = attention_mask.bool()  # (B, T)
        else:
            valid_mask = torch.ones(B, T, dtype=torch.bool, device=device)

        # Find token positions where ALL examples are non-pad so CKA gets
        # the full batch at each position.
        all_valid = valid_mask.all(dim=0)  # (T,)
        valid_positions = all_valid.nonzero(as_tuple=False).squeeze(-1)

        if valid_positions.numel() == 0:
            return (
                torch.tensor(0.0, device=device, requires_grad=True),
                {},
                {},
            )

        pair_losses = []
        pair_weights = []
        cka_scores: Dict[Tuple[int, int, int], float] = {}
        kc_lam1_scores: Dict[Tuple[int, int, int], float] = {}

        for pair in cluster_pairs:
            s_idx = pair.student_neuron_indices.to(device)
            t_idx = pair.teacher_neuron_indices.to(device)

            # (B, T, |C_s|) and (B, T, |C_t|)
            s_cluster = student_acts[:, :, s_idx]
            t_cluster = teacher_acts[:, :, t_idx]

            if B < 2:
                continue

            # CKA at each valid token position, then average
            token_ckas = []
            lam1_tokens: List[float] = []
            key = (pair.subclass, pair.student_cluster_idx, pair.teacher_cluster_idx)
            for t in valid_positions:
                s_t = s_cluster[:, t, :]  # (B, |C_s|)
                t_t = t_cluster[:, t, :]  # (B, |C_t|)
                cka = linear_cka_efficient(s_t, t_t, eps=self.eps)
                token_ckas.append(cka)
                if compute_kc_leading_eigenvalue:
                    with torch.no_grad():
                        lam1 = leading_eigenvalue_hkh(s_t)
                    lam1_tokens.append(float(lam1.item()))

            mean_cka = torch.stack(token_ckas).mean()
            pair_losses.append(1.0 - mean_cka)
            w = pair.importance if importance_weighting else 1.0
            if cluster_size_weighting:
                n_s = float(s_idx.numel())
                w = w * (n_s / (float(d_student_total) + 1e-12))
            pair_weights.append(w)
            cka_scores[key] = mean_cka.item()
            if compute_kc_leading_eigenvalue and lam1_tokens:
                kc_lam1_scores[key] = sum(lam1_tokens) / len(lam1_tokens)

        if not pair_losses:
            return (
                torch.tensor(0.0, device=device, requires_grad=True),
                {},
                {},
            )

        losses_t = torch.stack(pair_losses)
        weights_t = torch.tensor(pair_weights, device=device, dtype=losses_t.dtype)
        weights_t = weights_t / (weights_t.sum() + 1e-12)

        total = (losses_t * weights_t).sum()
        return total, cka_scores, kc_lam1_scores


# ---------------------------------------------------------------------------
# Dataset & collate (masked KL over the full causal sequence)
# ---------------------------------------------------------------------------

class AddDataset(Dataset):
    """Tokenize at construction; each sample is prompt+answer ids + ``prompt_len``."""

    def __init__(self, data: Union[str, Dict], tokenizer):
        if isinstance(data, str):
            with open(data, "r", encoding="utf-8") as f:
                raw = json.load(f)
            data = json_to_prompt_answer_dict(raw)
        self.samples = []
        for prompt, answer in data.items():
            answer = str(answer)
            prompt_ids = tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                add_special_tokens=False,
            )["input_ids"].squeeze(0)
            answer_ids = tokenizer(
                answer + tokenizer.eos_token,
                return_tensors="pt",
                padding=False,
                add_special_tokens=False,
            )["input_ids"].squeeze(0)
            input_ids = torch.cat([prompt_ids, answer_ids])
            self.samples.append(
                {"input_ids": input_ids, "prompt_len": len(prompt_ids)}
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(examples, pad_id: int):
    """Right-pad batch sequences and build ``kl_mask`` for all valid next-token positions.

    For causal LM, ``logits[:, t, :]`` predicts ``input_ids[:, t + 1]``.  The final
    sequence token is EOS (after ``answer + eos`` in :class:`AddDataset`).  KL is
    applied at ``t = 0 .. L-2`` so we align on every available next-token target
    in the sequence, including prompt continuation, answer tokens, and EOS.
    """
    max_len = max(ex["input_ids"].size(0) for ex in examples)
    B = len(examples)

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long)
    kl_mask = torch.zeros(B, max_len, dtype=torch.float32)

    for i, ex in enumerate(examples):
        ids = ex["input_ids"]
        L = ids.size(0)
        input_ids[i, :L] = ids
        attention_mask[i, :L] = 1
        # Distill the full causal sequence: every valid next-token prediction.
        for pos in range(0, max(L - 1, 0)):
            kl_mask[i, pos] = 1.0

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "kl_mask": kl_mask,
    }


# ---------------------------------------------------------------------------
# Pre-classification utility
# ---------------------------------------------------------------------------

def preclassify_training_data(
    circuit_model,
    tokenizer,
    train_data: Dict[str, int],
    device: str = "cpu",
) -> Dict[str, int]:
    """Assign each training problem to a latent subclass.

    Args:
        circuit_model: Trained ``CircuitDiscoveryModel`` (eval mode).
        tokenizer: HuggingFace tokenizer.
        train_data: ``{prompt_str: answer_int, ...}``
        device: Device for inference.

    Returns:
        ``{prompt_str: subclass_int, ...}``
    """
    from circuit_discovery.utils import parse_equation

    circuit_model.eval()
    prompts = list(train_data.keys())
    answers = list(train_data.values())
    subclass_map: Dict[str, int] = {}

    batch_size = 256
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_answers = answers[i : i + batch_size]

        full_strs = [f"{p}{a}" for p, a in zip(batch_prompts, batch_answers)]

        op1, op2, res = parse_equation(full_strs, device=device)
        with torch.no_grad():
            logits = circuit_model.classify_problem(op1, op2, res)
            classes = torch.argmax(logits, dim=-1).tolist()

        for prompt, cls in zip(batch_prompts, classes):
            subclass_map[prompt] = cls

    return subclass_map


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _topk_next_token_probs(
    lm: nn.Module,
    tokenizer,
    prompt: str,
    device: torch.device,
    k: int = 5,
    temperature: float = 1.0,
) -> List[Tuple[str, float]]:
    """Softmax top-``k`` at the last prefill position (same scaling as KL: ``logits / T``)."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        add_special_tokens=False,
    )
    enc = {key: val.to(device) for key, val in enc.items()}
    with torch.no_grad():
        out = lm(**enc)
    logits = out.logits[0, -1].float()
    probs = torch.softmax(logits / temperature, dim=-1)
    top_p, top_i = torch.topk(probs, min(k, probs.numel()))
    rows: List[Tuple[str, float]] = []
    for prob, idx in zip(top_p.tolist(), top_i.tolist()):
        tok = tokenizer.decode([idx], skip_special_tokens=False)
        rows.append((tok, float(prob)))
    return rows


def _print_topk_lines(name: str, rows: List[Tuple[str, float]], temperature: float) -> None:
    print(f"  {name} top-{len(rows)} next-token (softmax at T={temperature:g}):")
    for rank, (tok, p) in enumerate(rows, start=1):
        print(f"    {rank}. p={p:.4f} {tok!r}")


@torch.no_grad()
def eval_accuracy(
    model,
    tokenizer,
    data: Dict[str, int],
    batch_size: int = 50,
    max_new_tokens: Optional[int] = None,
    print_samples: int = 0,
    eval_label: str = "",
    teacher_for_topk_print: Optional[nn.Module] = None,
    temperature: float = 1.0,
) -> float:
    if max_new_tokens is None:
        max_new_tokens = EVAL_MAX_NEW_TOKENS
    model.eval()
    if teacher_for_topk_print is not None:
        teacher_for_topk_print.eval()
    prompts = list(data.keys())
    answers = list(data.values())
    correct = total = 0
    printed = 0
    label_note = f" ({eval_label.strip()})" if eval_label.strip() else ""

    # Match AddDataset/collate_fn (right-pad) so RoPE positions align with training.
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_answers = answers[i : i + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for j, (pred_text, gold) in enumerate(zip(decoded, batch_answers)):
            if printed < print_samples:
                idx = i + j
                print(
                    f"\n--- eval sample {printed + 1}/{print_samples}{label_note} "
                    f"[test index {idx}] ---",
                )
                print(f"prompt:\n{batch_prompts[j]}")
                primary_name = (
                    "student"
                    if teacher_for_topk_print is not None
                    else ("teacher" if "teacher" in eval_label.lower() else "student")
                )
                s_rows = _topk_next_token_probs(
                    model,
                    tokenizer,
                    batch_prompts[j],
                    model.device,
                    k=5,
                    temperature=temperature,
                )
                _print_topk_lines(primary_name, s_rows, temperature)
                if teacher_for_topk_print is not None:
                    t_rows = _topk_next_token_probs(
                        teacher_for_topk_print,
                        tokenizer,
                        batch_prompts[j],
                        teacher_for_topk_print.device,
                        k=5,
                        temperature=temperature,
                    )
                    _print_topk_lines("teacher", t_rows, temperature)
                printed += 1
            pred = _extract_int_after_equals(pred_text)
            if pred == gold:
                correct += 1
            total += 1

    tokenizer.padding_side = original_side
    return correct / max(total, 1)


def format_flops(n: float) -> str:
    """Human-readable FLOPs string (SI prefixes)."""
    if n <= 0:
        return "0"
    if n >= 1e15:
        return f"{n / 1e15:.3f} PFLOP"
    if n >= 1e12:
        return f"{n / 1e12:.3f} TFLOP"
    if n >= 1e9:
        return f"{n / 1e9:.3f} GFLOP"
    if n >= 1e6:
        return f"{n / 1e6:.3f} MFLOP"
    return f"{n:.0f} FLOP"


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ClusterDistillationTrainer:
    """Full training loop for neuron-cluster-level circuit distillation."""

    def __init__(
        self,
        config: ClusterDistillationConfig,
        cluster_pairs: List[ClusterPairInfo],
        train_data: Dict[str, int],
        test_data: Dict[str, int],
        tokenizer=None,
        student=None,
        teacher=None,
        resume: bool = False,
    ):
        self.config = config
        self._resume = resume
        self._standard = config.distillation_mode == "standard"
        self.cluster_pairs = cluster_pairs
        self.train_data = train_data
        self.test_data = test_data
        self.device = config.device

        _seed_all(config.seed)
        self._loader_generator = torch.Generator()
        self._loader_generator.manual_seed(config.seed)

        # Tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(config.student_model)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer = patch_tokenizer_no_special_tokens(self.tokenizer)

        # Models — student in float32 on CUDA for stable KL + gradients (train loop uses
        # .float()). fp16 student + composite loss often goes
        # non-finite after the first optimizer step.
        student_dtype = torch.float32
        teacher_dtype = torch.float16 if config.device == "cuda" else torch.float32
        if student is not None:
            self.student = student
        else:
            print(f"Loading student: {config.student_model}")
            self.student = AutoModelForCausalLM.from_pretrained(
                config.student_model, dtype=student_dtype,
            ).to(config.device)

        if teacher is not None:
            self.teacher = teacher
        else:
            print(f"Loading teacher: {config.teacher_model}")
            self.teacher = AutoModelForCausalLM.from_pretrained(
                config.teacher_model, dtype=teacher_dtype,
            ).to(config.device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Activation caches
        self.student_cache = ClusterActivationCache()
        self.teacher_cache = ClusterActivationCache()

        # Losses
        self.cluster_loss_fn = ClusterAlignmentLoss()

        # Optimizer
        self.optimizer = AdamW(params=self.student.parameters(), lr=config.learning_rate)

        # Dataset / loader (padding + kl_mask over the full next-token sequence)
        self.dataset = AddDataset(train_data, self.tokenizer)
        self.loader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=self._loader_generator,
            collate_fn=partial(collate_fn, pad_id=self.tokenizer.eos_token_id),
        )

        if (
            not self._standard
            and self.config.cluster_size_weighting
            and self.cluster_pairs
        ):
            scfg = self.student.config
            tcfg = self.teacher.config
            d_s = int(scfg.num_hidden_layers) * int(scfg.intermediate_size)
            d_t = int(tcfg.num_hidden_layers) * int(tcfg.intermediate_size)
            print("\nCluster size weighting: |C|/D (fraction of flattened MLP per pair)")
            for p in self.cluster_pairs:
                n_s = int(p.student_neuron_indices.numel())
                n_t = int(p.teacher_neuron_indices.numel())
                print(
                    f"  subclass={p.subclass}  s_cl={p.student_cluster_idx}  "
                    f"t_cl={p.teacher_cluster_idx}  "
                    f"|C_s|/D_s={n_s / d_s:.6f}  |C_t|/D_t={n_t / d_t:.6f}",
                )

        # History
        self.history: Dict[str, List] = defaultdict(list)

        # Optional KL vocab restriction (token IDs must exist in both student/teacher logits)
        self._kl_restrict_token_ids: Optional[torch.Tensor] = None
        if config.kl_mask_range is not None:
            lo, hi = config.kl_mask_range
            raw = token_ids_for_integer_range(self.tokenizer, lo, hi)
            vm = min(
                int(self.student.config.vocab_size),
                int(self.teacher.config.vocab_size),
            )
            raw = raw[raw < vm]
            if raw.numel() == 0:
                raise ValueError(
                    "kl_mask_range produced no token IDs valid for both models' vocab sizes.",
                )
            self._kl_restrict_token_ids = raw.to(self.device)
            print(
                f"KL vocab mask: {raw.numel()} columns (token IDs from ints in [{lo}, {hi}]); "
                f"full softmax, KL sum masked to those columns.",
            )

    def _align_epoch_flops_with_epoch(self) -> None:
        """Pad ``epoch_flops`` so length matches ``epoch`` (resume from old JSON)."""
        n = len(self.history.get("epoch", []))
        if n == 0:
            return
        if "epoch_flops" not in self.history:
            self.history["epoch_flops"] = [None] * n
        elif len(self.history["epoch_flops"]) < n:
            self.history["epoch_flops"].extend(
                [None] * (n - len(self.history["epoch_flops"])),
            )

    def _truncate_history_to_resume_checkpoint(self, start_epoch: int) -> None:
        """If ``training_history.json`` has more rows than ``training_state.pt`` says were
        completed, truncate lists so appending does not duplicate epoch indices.

        ``start_epoch`` is ``next_epoch`` from the checkpoint (number of finished epochs;
        next loop index).
        """
        ep = self.history.get("epoch")
        if not isinstance(ep, list) or not ep:
            return
        n_hist = len(ep)
        if n_hist <= start_epoch:
            if n_hist < start_epoch:
                print(
                    f"WARN: training_history.json has {n_hist} epoch rows but checkpoint "
                    f"next_epoch={start_epoch}; optimizer is ahead of saved metrics "
                    "(epoch indices may jump until history catches up)."
                )
            return
        print(
            f"WARN: training_history.json has {n_hist} epoch rows but checkpoint "
            f"next_epoch={start_epoch}; truncating history to {start_epoch} rows to "
            "match the checkpoint (fixes duplicate epoch entries on resume)."
        )
        for k, v in list(self.history.items()):
            if isinstance(v, list) and len(v) > start_epoch:
                self.history[k] = v[:start_epoch]

    # ------------------------------------------------------------------

    @staticmethod
    def _masked_hard_ce_loss(
        student_logits: torch.Tensor,
        input_ids: torch.Tensor,
        kl_mask: torch.Tensor,
    ) -> torch.Tensor:
        """CE between logits and integer next-token targets at positions where ``kl_mask`` is 1."""
        B, L, V = student_logits.shape
        if L < 2:
            return student_logits.sum() * 0.0
        logits_pred = student_logits[:, :-1, :]
        targets = input_ids[:, 1:]
        ce_mask = kl_mask[:, :-1]
        ce_per = F.cross_entropy(
            logits_pred.reshape(-1, V),
            targets.reshape(-1),
            reduction="none",
        ).view(B, L - 1)
        denom = ce_mask.sum().clamp_min(1.0)
        return (ce_per * ce_mask).sum() / denom

    def _forward_and_loss(
        self, batch: Dict,
    ) -> Tuple[torch.Tensor, Dict, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """One forward pass through student + teacher.

        Returns:
            ``(total, metrics, kl_loss, cluster_loss, hard_ce_loss)``. ``cluster_loss`` is
            ``None`` in standard mode. ``hard_ce_loss`` is ``None`` when ``hard_ce_weight`` is 0.
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        kl_mask = batch["kl_mask"].to(self.device)
        T = self.config.temperature

        try:
            if self._standard:
                with torch.no_grad():
                    teacher_logits = self.teacher(
                        input_ids=input_ids, attention_mask=attention_mask,
                    ).logits
                student_logits = self.student(
                    input_ids=input_ids, attention_mask=attention_mask,
                ).logits
                kl_loss = _masked_kl_loss_restricted(
                    student_logits,
                    teacher_logits,
                    kl_mask,
                    T,
                    self._kl_restrict_token_ids,
                )
                w_h = self.config.hard_ce_weight
                hard_ce = (
                    self._masked_hard_ce_loss(student_logits, input_ids, kl_mask)
                    if w_h > 0
                    else None
                )
                total = kl_loss
                if hard_ce is not None:
                    total = total + w_h * hard_ce
                metrics = {
                    "kl_loss": kl_loss.item(),
                    "cluster_loss": 0.0,
                    "total_loss": total.item(),
                    "mean_cka": 0.0,
                    "kc_lam1": {},
                    "hard_ce_loss": hard_ce.item() if hard_ce is not None else 0.0,
                }
                return total, metrics, kl_loss, None, hard_ce

            self.teacher_cache.register_hooks(self.teacher, detach=True)
            self.student_cache.register_hooks(self.student, detach=False)

            with torch.no_grad():
                teacher_out = self.teacher(input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_out.logits
            teacher_acts = self.teacher_cache.get_flattened_per_token()

            student_out = self.student(
                input_ids=input_ids, attention_mask=attention_mask,
            )
            student_logits = student_out.logits
            student_acts = self.student_cache.get_flattened_per_token()

            # Masked KL over the full next-token sequence (same mask as standard mode).
            kl_loss = _masked_kl_loss_restricted(
                student_logits,
                teacher_logits,
                kl_mask,
                T,
                self._kl_restrict_token_ids,
            )

            w_h = self.config.hard_ce_weight
            hard_ce = (
                self._masked_hard_ce_loss(student_logits, input_ids, kl_mask)
                if w_h > 0
                else None
            )

            cluster_loss, cka_scores, kc_lam1_scores = self.cluster_loss_fn(
                student_acts, teacher_acts, self.cluster_pairs,
                attention_mask=attention_mask,
                importance_weighting=self.config.importance_weighting,
                cluster_size_weighting=self.config.cluster_size_weighting,
                compute_kc_leading_eigenvalue=self.config.log_kl_cka_grad_norms,
            )

            total = kl_loss + self.config.lambda_cluster * cluster_loss
            if hard_ce is not None:
                total = (
                    (1.0 - w_h) * kl_loss
                    + self.config.lambda_cluster * cluster_loss
                    + w_h * hard_ce
                )

            kc_lam1 = {
                kc_lam1_metric_key(sk, sc, tc): float(v)
                for (sk, sc, tc), v in kc_lam1_scores.items()
            }
            metrics = {
                "kl_loss": kl_loss.item(),
                "cluster_loss": cluster_loss.item(),
                "total_loss": total.item(),
                "mean_cka": (
                    sum(cka_scores.values()) / len(cka_scores)
                    if cka_scores else 0.0
                ),
                "kc_lam1": kc_lam1,
                "hard_ce_loss": hard_ce.item() if hard_ce is not None else 0.0,
            }
            return total, metrics, kl_loss, cluster_loss, hard_ce

        finally:
            self.student_cache.clear()
            self.teacher_cache.clear()

    def _assign_grad_from_kl_cka(
        self,
        kl_loss: torch.Tensor,
        cluster_loss: Optional[torch.Tensor],
    ) -> Tuple[float, float, float]:
        """Set ``param.grad`` from KL and λ·CKA (circuit: CKA grads scaled by ``‖g_KL‖/‖g_{λ·CKA}‖``).

        ``kl_loss`` may be the combined distillation objective
        ``(1 - w) * kl + w * hard_ce`` (``w = hard_ce_weight``) when hard CE is enabled.

        Circuit mode: ``p.grad = g_KL + r · g_{λ·CKA}`` with
        ``r = ||g_KL|| / (||g_{λ·CKA}|| + ε)``. Standard mode only backprops ``kl_loss``; ``r`` is ``1.0``.

        Returns:
            ``(||g_KL||_2, ||g_{λ·CKA}||_2, r)``. Standard KL-only: third value is ``1.0``.
        """
        params = [p for p in self.student.parameters() if p.requires_grad]
        if self._standard:
            g_kl = torch.autograd.grad(
                kl_loss, params, retain_graph=False, allow_unused=True,
            )
            kl_gn = _grad_tuple_global_l2_norm(g_kl)
            for p, g in zip(params, g_kl):
                p.grad = g if g is not None else torch.zeros_like(p)
            return kl_gn, 0.0, 1.0

        assert cluster_loss is not None
        g_kl = torch.autograd.grad(
            kl_loss, params, retain_graph=True, allow_unused=True,
        )
        kl_gn = _grad_tuple_global_l2_norm(g_kl)
        cka_term = self.config.lambda_cluster * cluster_loss
        g_cka = torch.autograd.grad(
            cka_term, params, retain_graph=False, allow_unused=True,
        )
        cka_gn = _grad_tuple_global_l2_norm(g_cka)
        eps = 1e-12
        ratio = kl_gn / (cka_gn + eps)
        if not math.isfinite(ratio):
            ratio = 1.0
        for p, gk, gc in zip(params, g_kl, g_cka):
            gk = torch.zeros_like(p) if gk is None else gk
            gc = torch.zeros_like(p) if gc is None else gc
            p.grad = gk + float(ratio) * gc
        return kl_gn, cka_gn, float(ratio)

    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> Dict[str, Any]:
        self.student.train()
        agg = defaultdict(float)
        agg_kc_lam1: Dict[str, float] = defaultdict(float)
        n = 0
        skipped_nonfinite = 0
        epoch_flops = 0
        cfg = self.config
        count_flops_this_epoch = (
            cfg.count_flops_every > 0
            and FlopCounterMode is not None
            and (epoch % cfg.count_flops_every == 0)
        )

        for step, batch in enumerate(self.loader):
            if count_flops_this_epoch:
                assert FlopCounterMode is not None
                fcm = FlopCounterMode(display=False)
                with fcm:
                    loss, metrics, kl_loss, cluster_loss, hard_ce = self._forward_and_loss(batch)
                    if loss is None:
                        pass
                    elif torch.isfinite(loss).item():
                        self.optimizer.zero_grad()
                        if cfg.log_kl_cka_grad_norms:
                            kl_for_backward = kl_loss
                            if hard_ce is not None:
                                w_h = cfg.hard_ce_weight
                                kl_for_backward = (1.0 - w_h) * kl_loss + w_h * hard_ce
                            kl_gn, cka_gn, kl_over_cka = self._assign_grad_from_kl_cka(
                                kl_for_backward, cluster_loss,
                            )
                            metrics["kl_grad_norm"] = kl_gn
                            metrics["cka_grad_norm"] = cka_gn
                            metrics["cka_kl_grad_scale"] = kl_over_cka
                        else:
                            loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.student.parameters(), self.config.grad_clip,
                        )
                        self.optimizer.step()
                epoch_flops += int(fcm.get_total_flops())
                if loss is None:
                    continue
            else:
                loss, metrics, kl_loss, cluster_loss, hard_ce = self._forward_and_loss(batch)
                if loss is None:
                    continue
                if not torch.isfinite(loss).item():
                    skipped_nonfinite += 1
                    continue
                self.optimizer.zero_grad()
                if cfg.log_kl_cka_grad_norms:
                    kl_for_backward = kl_loss
                    if hard_ce is not None:
                        w_h = cfg.hard_ce_weight
                        kl_for_backward = (1.0 - w_h) * kl_loss + w_h * hard_ce
                    kl_gn, cka_gn, kl_over_cka = self._assign_grad_from_kl_cka(
                        kl_for_backward, cluster_loss,
                    )
                    metrics["kl_grad_norm"] = kl_gn
                    metrics["cka_grad_norm"] = cka_gn
                    metrics["cka_kl_grad_scale"] = kl_over_cka
                else:
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(), self.config.grad_clip,
                )
                self.optimizer.step()

            if not torch.isfinite(loss).item():
                skipped_nonfinite += 1
                continue

            for k, v in metrics.items():
                if k == "kc_lam1":
                    if isinstance(v, dict):
                        for pk, pv in v.items():
                            agg_kc_lam1[pk] += float(pv)
                    continue
                agg[k] += float(v)
            n += 1

            if step % max(1, self.config.step_log_interval) == 0:
                flop_s = (
                    f" | cum {format_flops(epoch_flops)}"
                    if count_flops_this_epoch
                    else ""
                )
                grad_s = ""
                if cfg.log_kl_cka_grad_norms:
                    grad_s = (
                        f" | ||g||_KL {metrics['kl_grad_norm']:.4f}"
                        f" | ||g||_λCKA {metrics['cka_grad_norm']:.4f}"
                    )
                    if not self._standard:
                        grad_s += (
                            f" | ‖g_KL‖/‖g_CKA‖ {metrics.get('cka_kl_grad_scale', 1.0):.4f}"
                        )
                        kc = metrics.get("kc_lam1") or {}
                        if kc:
                            parts = [f"{pk}={pv:.4f}" for pk, pv in sorted(kc.items())]
                            grad_s += " | λ_max(K_c) " + " ".join(parts)
                hard_s = ""
                if cfg.hard_ce_weight > 0:
                    hard_s = f" | hardCE {metrics.get('hard_ce_loss', 0.0):.4f}"
                if self._standard:
                    print(
                        f"  step {step:04d} | KL {metrics['kl_loss']:.4f}"
                        f"{hard_s}{grad_s}{flop_s}",
                    )
                else:
                    print(
                        f"  step {step:04d} | "
                        f"KL {metrics['kl_loss']:.4f} | "
                        f"Cluster {metrics['cluster_loss']:.4f} | "
                        f"CKA {metrics['mean_cka']:.4f}"
                        f"{hard_s}{grad_s}{flop_s}",
                    )

        if n == 0:
            print(
                f"  WARNING: no valid optimizer steps this epoch "
                f"({skipped_nonfinite} batch(es) had non-finite loss; "
                f"{len(self.loader)} batches total). "
                f"Epoch metrics below are undefined (not zero loss)."
            )
        out: Dict[str, Any] = {k: v / max(n, 1) for k, v in agg.items()}
        out["kc_lam1"] = {pk: v / max(n, 1) for pk, v in agg_kc_lam1.items()}
        out["epoch_flops"] = float(epoch_flops) if count_flops_this_epoch else None
        return out

    # ------------------------------------------------------------------

    def train(self) -> Dict[str, List]:
        """Run the full training loop."""
        cfg = self.config

        hist_path = os.path.join(cfg.save_dir, "training_history.json")
        state_path = training_state_path(cfg.save_dir)
        start_epoch = 0
        best_acc = 0.0

        if self._resume:
            if os.path.isfile(hist_path):
                with open(hist_path, "r") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    for k, v in loaded.items():
                        self.history[k] = v
            for key in (
                "epoch",
                "kl_loss",
                "hard_ce_loss",
                "accuracy",
                "cluster_loss",
                "mean_cka",
                "kc_lam1",
                "kl_grad_norm",
                "cka_grad_norm",
                "cka_kl_grad_scale",
            ):
                if key not in self.history:
                    self.history[key] = []

            if os.path.isfile(state_path):
                start_epoch, best_acc = load_training_state(
                    state_path, self.optimizer, self.device,
                )
                print(
                    f"Resumed optimizer state (starting at global epoch {start_epoch + 1}, "
                    f"best_acc={best_acc:.4f})"
                )
                self._truncate_history_to_resume_checkpoint(start_epoch)
            else:
                start_epoch = len(self.history.get("epoch", []))
                best_acc = (
                    max(self.history["accuracy"])
                    if self.history.get("accuracy")
                    else 0.0
                )
                print(
                    f"No training_state.pt in {cfg.save_dir} — warm-starting from student_model "
                    f"with new optimizer (continuing from epoch {start_epoch + 1})."
                )

            print("Skipping baseline eval (resumed run).")
            student_base = float(self.history.get("student_baseline", 0.0))
            teacher_base = float(self.history.get("teacher_baseline", 0.0))
        else:
            print("Evaluating baselines...")
            student_base = eval_accuracy(
                self.student, self.tokenizer, self.test_data,
                batch_size=cfg.eval_batch_size,
                max_new_tokens=cfg.eval_max_new_tokens,
                print_samples=cfg.eval_print_samples,
                eval_label="student baseline",
                teacher_for_topk_print=(
                    self.teacher if cfg.eval_print_samples > 0 else None
                ),
                temperature=cfg.temperature,
            )
            teacher_base = eval_accuracy(
                self.teacher, self.tokenizer, self.test_data,
                batch_size=cfg.eval_batch_size,
                max_new_tokens=cfg.eval_max_new_tokens,
                print_samples=0,
                eval_label="teacher baseline",
                temperature=cfg.temperature,
            )
            print(f"  Student baseline accuracy: {student_base:.4f}")
            print(f"  Teacher baseline accuracy: {teacher_base:.4f}")
            self.history["student_baseline"] = student_base
            self.history["teacher_baseline"] = teacher_base

        self._align_epoch_flops_with_epoch()

        end_epoch = start_epoch + cfg.epochs
        print("=" * 60)
        if self._standard:
            print("Standard KL Distillation (neuron-distillation entry)")
        else:
            print("Neuron-Cluster KL + CKA Distillation")
        print(f"  Run dir:          {cfg.save_dir}")
        if self._resume:
            print(
                f"  Epochs:           +{cfg.epochs} this run  "
                f"(epochs {start_epoch + 1}..{end_epoch}, {end_epoch} total)"
            )
        else:
            print(f"  Epochs:           {cfg.epochs}")
        print(f"  Batch size:       {cfg.batch_size}")
        print(f"  RNG seed:         {cfg.seed}")
        print(f"  LR:               {cfg.learning_rate}")
        print(f"  Temperature:      {cfg.temperature}")
        if cfg.kl_mask_range is not None:
            km_lo, km_hi = cfg.kl_mask_range
            print(
                f"  kl_mask_range:    [{km_lo}, {km_hi}] "
                f"(KL sum over vocab columns for those ints; full softmax)",
            )
        print(f"  eval batch size:  {cfg.eval_batch_size}")
        print(f"  eval max_new_tokens: {cfg.eval_max_new_tokens}")
        if cfg.eval_print_samples > 0:
            print(
                f"  eval print samples: {cfg.eval_print_samples} "
                f"(prompt + top-5 softmax at T={cfg.temperature:g} for student & teacher)",
            )
        if not self._standard:
            print(f"  lambda_cluster:   {cfg.lambda_cluster}")
            print(f"  Cluster pairs:    {len(self.cluster_pairs)}")
        print(f"  Save every:       {cfg.save_every}")
        print(f"  Step log interval:{cfg.step_log_interval}")
        if cfg.log_kl_cka_grad_norms:
            print(
                "  KL/CKA grad norms: on (split grads; ~2× work); circuit mode scales λ·CKA grads "
                "by ‖g_KL‖/‖g_{λ·CKA}‖",
            )
            if not self._standard:
                print(
                    "  K_c spectrum:     λ_max(H X X^T H) per cluster (student), "
                    "mean over pairs/tokens (same CKA mask)",
                )
        if cfg.count_flops_every <= 0:
            print("  FLOP counting:    off (--count-flops-every 0)")
        elif FlopCounterMode is None:
            print("  FLOP counting:    unavailable (PyTorch flop_counter missing)")
        elif cfg.count_flops_every == 1:
            print("  FLOP counting:    every epoch (FlopCounterMode; ~10% step overhead when on)")
        else:
            print(
                f"  FLOP counting:    every {cfg.count_flops_every} epochs "
                f"(0-based indices 0, {cfg.count_flops_every}, …)"
            )
        print("=" * 60)

        if start_epoch >= end_epoch:
            print(
                f"Nothing to train: resume starts at epoch index {start_epoch}, "
                f"end index {end_epoch}. Increase --epochs."
            )
            self._save_history()
            return dict(self.history)

        for epoch in range(start_epoch, end_epoch):
            epoch_metrics = self.train_epoch(epoch)

            acc = 0.0
            saved_fast_weights = False
            if (epoch + 1) % cfg.eval_every == 0:
                acc = eval_accuracy(
                    self.student, self.tokenizer, self.test_data,
                    batch_size=cfg.eval_batch_size,
                    max_new_tokens=cfg.eval_max_new_tokens,
                    print_samples=cfg.eval_print_samples,
                    eval_label=f"epoch {epoch + 1} student",
                    teacher_for_topk_print=(
                        self.teacher if cfg.eval_print_samples > 0 else None
                    ),
                    temperature=cfg.temperature,
                )
                epoch_metrics["accuracy"] = acc

                if acc > best_acc:
                    best_acc = acc
                    if cfg.save_best:
                        self._save_fast_weights_and_training_state(epoch + 1, best_acc)
                        print(
                            f"  Saved {STUDENT_WEIGHTS_FILE} + training_state.pt "
                            f"(new best accuracy)",
                        )
                        saved_fast_weights = True

            if (
                cfg.save_every > 0
                and (epoch + 1) % cfg.save_every == 0
                and not saved_fast_weights
            ):
                self._save_fast_weights_and_training_state(epoch + 1, best_acc)
                print(
                    f"  Saved {STUDENT_WEIGHTS_FILE} + training_state.pt "
                    f"(epoch {epoch + 1})",
                )

            self.history["epoch"].append(epoch + 1)
            for k, v in epoch_metrics.items():
                self.history[k].append(v)

            # Per-epoch history only (training_state.pt matches fast weights / final save)
            self._save_history()

            if epoch_metrics:
                ef = epoch_metrics.get("epoch_flops")
                flop_s = ""
                if ef is not None:
                    flop_s = f", FLOPs={format_flops(float(ef))}"
                if self._standard:
                    g_line = ""
                    if cfg.log_kl_cka_grad_norms and "kl_grad_norm" in epoch_metrics:
                        g_line = (
                            f", ||g||_KL={epoch_metrics['kl_grad_norm']:.4f}"
                        )
                    ce_s = ""
                    if cfg.hard_ce_weight > 0:
                        ce_s = (
                            f", hardCE={epoch_metrics.get('hard_ce_loss', float('nan')):.4f}"
                        )
                    print(
                        f"Epoch {epoch + 1}/{end_epoch}: "
                        f"KL={epoch_metrics.get('kl_loss', float('nan')):.4f}"
                        f"{ce_s}{g_line}, Acc={acc:.4f}{flop_s}"
                    )
                else:
                    g_line = ""
                    if cfg.log_kl_cka_grad_norms and "kl_grad_norm" in epoch_metrics:
                        kc = epoch_metrics.get("kc_lam1") or {}
                        kc_s = ""
                        if kc:
                            kc_s = (
                                ", λ_max(K_c): "
                                + ", ".join(
                                    f"{pk}={pv:.4f}"
                                    for pk, pv in sorted(kc.items())
                                )
                            )
                        g_line = (
                            f", ||g||_KL={epoch_metrics['kl_grad_norm']:.4f}, "
                            f"||g||_λCKA={epoch_metrics['cka_grad_norm']:.4f}, "
                            f"‖g_KL‖/‖g_CKA‖={epoch_metrics.get('cka_kl_grad_scale', float('nan')):.4f}"
                            f"{kc_s}"
                        )
                    ce_s = ""
                    if cfg.hard_ce_weight > 0:
                        ce_s = (
                            f", hardCE={epoch_metrics.get('hard_ce_loss', float('nan')):.4f}"
                        )
                    print(
                        f"Epoch {epoch + 1}/{end_epoch}: "
                        f"KL={epoch_metrics.get('kl_loss', float('nan')):.4f}, "
                        f"Cluster={epoch_metrics.get('cluster_loss', float('nan')):.4f}, "
                        f"CKA={epoch_metrics.get('mean_cka', float('nan')):.4f}"
                        f"{ce_s}{g_line}, Acc={acc:.4f}{flop_s}"
                    )
            else:
                tag = "KL=n/a" if self._standard else "KL/Cluster/CKA=n/a"
                print(
                    f"Epoch {epoch + 1}/{end_epoch}: "
                    f"{tag} (no valid steps), Acc={acc:.4f}"
                )

        # Write history and curves BEFORE slow checkpoint save
        self._save_history()
        self._save_curves()

        self._save_checkpoint()
        save_training_state(cfg.save_dir, self.optimizer, end_epoch, best_acc)
        print(f"  Saved {STUDENT_MODEL_DIR}/ + training_state.pt (final)")

        # Clean up fast weights file (superseded by full checkpoint)
        wt_path = os.path.join(cfg.save_dir, STUDENT_WEIGHTS_FILE)
        try:
            os.remove(wt_path)
        except FileNotFoundError:
            pass

        print(f"\nDone. Best accuracy: {best_acc:.4f}")
        print(f"Results saved to: {cfg.save_dir}")
        return dict(self.history)

    # ------------------------------------------------------------------

    def _save_curves(self) -> None:
        """Save KL-loss, cluster-loss, and accuracy curves as a PNG."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed — skipping curve plots.")
            return

        history = dict(self.history)
        epochs = history.get("epoch", [])
        if not epochs:
            return

        if self._standard:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(epochs, history.get("kl_loss", []), marker="o", markersize=3, linewidth=1.5)
            axes[0].set_title("KL Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("KL Loss")
            axes[0].grid(True, alpha=0.3)
            axes[1].plot(epochs, history.get("accuracy", []), marker="o", markersize=3,
                         linewidth=1.5, color="tab:orange")
            axes[1].set_title("Test Accuracy")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Accuracy")
            axes[1].set_ylim(0, 1)
            axes[1].grid(True, alpha=0.3)
            fig.suptitle("Standard KL Distillation", fontsize=13)
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].plot(epochs, history.get("kl_loss", []), marker="o", markersize=3, linewidth=1.5)
            axes[0].set_title("KL Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("KL Loss")
            axes[0].grid(True, alpha=0.3)
            axes[1].plot(epochs, history.get("cluster_loss", []), marker="o", markersize=3,
                         linewidth=1.5, color="tab:green")
            axes[1].set_title("Cluster CKA Loss")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Cluster Loss")
            axes[1].grid(True, alpha=0.3)
            axes[2].plot(epochs, history.get("accuracy", []), marker="o", markersize=3,
                         linewidth=1.5, color="tab:orange")
            axes[2].set_title("Test Accuracy")
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("Accuracy")
            axes[2].set_ylim(0, 1)
            axes[2].grid(True, alpha=0.3)
            fig.suptitle("Neuron-Cluster KL + CKA Distillation", fontsize=13)
        fig.tight_layout()

        os.makedirs(self.config.save_dir, exist_ok=True)
        out = os.path.join(self.config.save_dir, "training_curves.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved training curves → {out}")

    def _save_weights_fast(self) -> None:
        """Save only state_dict to a single .pt — 10-20x faster than save_pretrained."""
        path = os.path.join(self.config.save_dir, STUDENT_WEIGHTS_FILE)
        torch.save(self.student.state_dict(), path)

    def _save_fast_weights_and_training_state(
        self, completed_epochs: int, best_acc: float,
    ) -> None:
        """``student_weights.pt`` plus matching ``training_state.pt`` (resume pair)."""
        self._save_weights_fast()
        save_training_state(
            self.config.save_dir, self.optimizer, completed_epochs, best_acc,
        )

    def _save_checkpoint(self) -> None:
        """Overwrite ``save_dir/student_model/`` with current student + tokenizer.
        Used for the final save at end of training."""
        path = os.path.join(self.config.save_dir, STUDENT_MODEL_DIR)
        rm_dir_tree(path)
        os.makedirs(path, exist_ok=True)
        self.student.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def _save_history(self):
        os.makedirs(self.config.save_dir, exist_ok=True)
        path = os.path.join(self.config.save_dir, "training_history.json")
        with open(path, "w") as f:
            json.dump(dict(self.history), f, indent=2)
            f.flush()
            os.fsync(f.fileno())

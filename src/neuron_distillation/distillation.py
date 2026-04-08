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
  4. The total loss uses masked per-position KL at the first-answer
     logit, plus ``lambda * L_cluster_align`` (CKA). No ``batchmean`` over the full vocab grid.

Pipeline prerequisites (run before this module):
  - Neuron clustering  (clustering.py)
  - Cluster ablation   (ablation.py)
  - Cluster pairing    (pairing.py)
"""

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from torch.utils.flop_counter import FlopCounterMode
except ImportError:
    FlopCounterMode = None  # type: ignore[misc, assignment]

from cka_loss import linear_cka_efficient
from neuron_distillation.pairing import (
    ClusterMapping,
    _load_single_ablation_performance,
    create_cluster_mapping,
)
from utils import (
    EVAL_MAX_NEW_TOKENS,
    STUDENT_MODEL_DIR,
    STUDENT_WEIGHTS_FILE,
    json_to_prompt_answer_dict,
    load_training_state,
    rm_dir_tree,
    save_training_state,
    training_state_path,
)


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
    lambda_proj: float = 0.0
    importance_weighting: bool = True

    use_projection_heads: bool = False

    eval_every: int = 1
    # In-epoch step log every N batches.
    step_log_interval: int = 50
    save_every: int = 5
    save_best: bool = False
    eval_max_new_tokens: int = EVAL_MAX_NEW_TOKENS
    save_dir: str = "results/cluster-distillation"
    # Count FLOPs (FlopCounterMode) only on epochs where ``epoch_index % N == 0`` (0-based).
    # 1 = every epoch; 0 = never; N>1 = every Nth epoch (0, N, 2N, …). ~10% step overhead when active.
    count_flops_every: int = 1

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def checkpoint_every(self) -> int:
        """Alias for ``save_every`` (older code / notebooks referenced this name)."""
        return self.save_every


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

    Returns:
        List of :class:`ClusterPairInfo` sorted by importance (descending).
    """
    delta_s = _load_single_ablation_performance(student_ablation_path)
    delta_t = _load_single_ablation_performance(teacher_ablation_path)

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
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            student_acts: ``(B, T, D_student_total)`` per-token up_proj activations.
            teacher_acts: ``(B, T, D_teacher_total)`` same for teacher.
            cluster_pairs: List of :class:`ClusterPairInfo`.
            attention_mask: ``(B, T)`` binary mask; 1 = real token, 0 = pad.
            importance_weighting: Scale each pair's loss by its importance.

        Returns:
            ``(total_loss, info_dict)`` where info_dict has per-pair mean CKA.
        """
        device = student_acts.device
        B, T, _ = student_acts.shape

        if attention_mask is not None:
            valid_mask = attention_mask.bool()  # (B, T)
        else:
            valid_mask = torch.ones(B, T, dtype=torch.bool, device=device)

        # Find token positions where ALL examples are non-pad so CKA gets
        # the full batch at each position.
        all_valid = valid_mask.all(dim=0)  # (T,)
        valid_positions = all_valid.nonzero(as_tuple=False).squeeze(-1)

        if valid_positions.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), {}

        pair_losses = []
        pair_weights = []
        cka_scores: Dict[Tuple[int, int, int], float] = {}

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
            for t in valid_positions:
                s_t = s_cluster[:, t, :]  # (B, |C_s|)
                t_t = t_cluster[:, t, :]  # (B, |C_t|)
                cka = linear_cka_efficient(s_t, t_t, eps=self.eps)
                token_ckas.append(cka)

            mean_cka = torch.stack(token_ckas).mean()
            pair_losses.append(1.0 - mean_cka)
            pair_weights.append(pair.importance if importance_weighting else 1.0)
            cka_scores[(pair.subclass, pair.student_cluster_idx,
                         pair.teacher_cluster_idx)] = mean_cka.item()

        if not pair_losses:
            return torch.tensor(0.0, device=device, requires_grad=True), {}

        losses_t = torch.stack(pair_losses)
        weights_t = torch.tensor(pair_weights, device=device, dtype=losses_t.dtype)
        weights_t = weights_t / (weights_t.sum() + 1e-12)

        total = (losses_t * weights_t).sum()
        return total, cka_scores


# ---------------------------------------------------------------------------
# Optional: Projection-based alignment
# ---------------------------------------------------------------------------

class ProjectionHeadBank(nn.Module):
    """Lightweight learned linear projections for direct cluster alignment.

    For each cluster pair, maps student cluster activations into the
    teacher's cluster dimensionality and applies an MSE loss.
    """

    def __init__(self, cluster_pairs: List[ClusterPairInfo]):
        super().__init__()
        self.projections = nn.ModuleDict()
        self._pair_keys: List[str] = []

        for pair in cluster_pairs:
            key = f"s{pair.subclass}_c{pair.student_cluster_idx}"
            s_dim = pair.student_neuron_indices.numel()
            t_dim = pair.teacher_neuron_indices.numel()
            self.projections[key] = nn.Linear(s_dim, t_dim, bias=False)
            self._pair_keys.append(key)

    def forward(
        self,
        student_flat: torch.Tensor,
        teacher_flat: torch.Tensor,
        cluster_pairs: List[ClusterPairInfo],
    ) -> torch.Tensor:
        device = student_flat.device
        losses = []

        for pair, key in zip(cluster_pairs, self._pair_keys):
            s_idx = pair.student_neuron_indices.to(device)
            t_idx = pair.teacher_neuron_indices.to(device)

            s_act = student_flat[:, s_idx]
            t_act = teacher_flat[:, t_idx].detach()

            projected = self.projections[key](s_act)
            losses.append(F.mse_loss(projected, t_act))

        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Dataset & collate (masked KL at first answer token)
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
    """Right-pad batch sequences and build ``kl_mask`` (first-answer logit only)."""
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
        pos = ex["prompt_len"] - 1
        if 0 <= pos < L:
            kl_mask[i, pos] = 1.0

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "kl_mask": kl_mask,
    }


def masked_mean_over_tokens(
    acts: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Pool ``(B, T, D)`` → ``(B, D)`` by averaging only real tokens (not padding)."""
    m = attention_mask.to(dtype=acts.dtype).unsqueeze(-1)
    num = (acts * m).sum(dim=1)
    den = m.sum(dim=1).clamp_min(1.0)
    return num / den


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

def _extract_int_after_equals(text: str) -> Optional[int]:
    m = re.search(r"=\s*(\d+)", text)
    return int(m.group(1)) if m else None


@torch.no_grad()
def eval_accuracy(
    model,
    tokenizer,
    data: Dict[str, int],
    batch_size: int = 50,
    max_new_tokens: Optional[int] = None,
) -> float:
    if max_new_tokens is None:
        max_new_tokens = EVAL_MAX_NEW_TOKENS
    model.eval()
    prompts = list(data.keys())
    answers = list(data.values())
    correct = total = 0

    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_answers = answers[i : i + batch_size]

        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True,
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for pred_text, gold in zip(decoded, batch_answers):
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

        # Tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(config.student_model)
            self.tokenizer.pad_token = self.tokenizer.eos_token

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

        self.proj_heads: Optional[ProjectionHeadBank] = None
        if (
            not self._standard
            and config.use_projection_heads
            and cluster_pairs
        ):
            self.proj_heads = ProjectionHeadBank(cluster_pairs).to(self.device)

        # Optimizer
        params = list(self.student.parameters())
        if self.proj_heads is not None:
            params += list(self.proj_heads.parameters())
        self.optimizer = AdamW(params, lr=config.learning_rate)

        # Dataset / loader (padding + kl_mask at answer position)
        self.dataset = AddDataset(train_data, self.tokenizer)
        self.loader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=partial(collate_fn, pad_id=self.tokenizer.eos_token_id),
        )

        # History
        self.history: Dict[str, List] = defaultdict(list)

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

    # ------------------------------------------------------------------

    def _forward_and_loss(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """One forward pass through student + teacher, returns composite loss."""
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
                log_p_s = F.log_softmax(student_logits.float() / T, dim=-1)
                p_t = F.softmax(teacher_logits.float() / T, dim=-1)
                kl_per_token = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1)
                kl_loss = (
                    (kl_per_token * kl_mask).sum()
                    / kl_mask.sum().clamp_min(1.0)
                    * (T**2)
                )
                total = kl_loss
                metrics = {
                    "kl_loss": kl_loss.item(),
                    "cluster_loss": 0.0,
                    "proj_loss": 0.0,
                    "total_loss": total.item(),
                    "mean_cka": 0.0,
                }
                return total, metrics

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

            # Masked KL at first-answer logit only.
            log_p_s = F.log_softmax(student_logits.float() / T, dim=-1)
            p_t = F.softmax(teacher_logits.float() / T, dim=-1)
            kl_per_token = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1)
            kl_loss = (
                (kl_per_token * kl_mask).sum()
                / kl_mask.sum().clamp_min(1.0)
                * (T**2)
            )

            cluster_loss, cka_scores = self.cluster_loss_fn(
                student_acts, teacher_acts, self.cluster_pairs,
                attention_mask=attention_mask,
                importance_weighting=self.config.importance_weighting,
            )

            proj_loss = torch.tensor(0.0, device=self.device)
            if self.proj_heads is not None and self.config.lambda_proj > 0:
                s_pooled = masked_mean_over_tokens(student_acts, attention_mask)
                t_pooled = masked_mean_over_tokens(teacher_acts, attention_mask)
                proj_loss = self.proj_heads(
                    s_pooled, t_pooled, self.cluster_pairs,
                )

            total = (
                kl_loss
                + self.config.lambda_cluster * cluster_loss
                + self.config.lambda_proj * proj_loss
            )

            metrics = {
                "kl_loss": kl_loss.item(),
                "cluster_loss": cluster_loss.item(),
                "proj_loss": proj_loss.item(),
                "total_loss": total.item(),
                "mean_cka": (
                    sum(cka_scores.values()) / len(cka_scores)
                    if cka_scores else 0.0
                ),
            }
            return total, metrics

        finally:
            self.student_cache.clear()
            self.teacher_cache.clear()

    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> Dict[str, Any]:
        self.student.train()
        agg = defaultdict(float)
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
                    loss, metrics = self._forward_and_loss(batch)
                    if loss is None:
                        pass
                    elif torch.isfinite(loss).item():
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.student.parameters(), self.config.grad_clip,
                        )
                        self.optimizer.step()
                epoch_flops += int(fcm.get_total_flops())
                if loss is None:
                    continue
            else:
                loss, metrics = self._forward_and_loss(batch)
                if loss is None:
                    continue
                if not torch.isfinite(loss).item():
                    skipped_nonfinite += 1
                    continue
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(), self.config.grad_clip,
                )
                self.optimizer.step()

            if not torch.isfinite(loss).item():
                skipped_nonfinite += 1
                continue

            for k, v in metrics.items():
                agg[k] += v
            n += 1

            if step % max(1, self.config.step_log_interval) == 0:
                flop_s = (
                    f" | cum {format_flops(epoch_flops)}"
                    if count_flops_this_epoch
                    else ""
                )
                if self._standard:
                    print(f"  step {step:04d} | KL {metrics['kl_loss']:.4f}{flop_s}")
                else:
                    print(
                        f"  step {step:04d} | "
                        f"KL {metrics['kl_loss']:.4f} | "
                        f"Cluster {metrics['cluster_loss']:.4f} | "
                        f"CKA {metrics['mean_cka']:.4f}{flop_s}"
                    )

        if n == 0:
            print(
                f"  WARNING: no valid optimizer steps this epoch "
                f"({skipped_nonfinite} batch(es) had non-finite loss; "
                f"{len(self.loader)} batches total). "
                f"Epoch metrics below are undefined (not zero loss)."
            )
        out: Dict[str, Any] = {k: v / max(n, 1) for k, v in agg.items()}
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
            for key in ("epoch", "kl_loss", "accuracy", "cluster_loss", "mean_cka"):
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
                max_new_tokens=cfg.eval_max_new_tokens,
            )
            teacher_base = eval_accuracy(
                self.teacher, self.tokenizer, self.test_data,
                max_new_tokens=cfg.eval_max_new_tokens,
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
        print(f"  LR:               {cfg.learning_rate}")
        print(f"  Temperature:      {cfg.temperature}")
        print(f"  eval max_new_tokens: {cfg.eval_max_new_tokens}")
        if not self._standard:
            print(f"  lambda_cluster:   {cfg.lambda_cluster}")
            print(f"  lambda_proj:      {cfg.lambda_proj}")
            print(f"  Cluster pairs:    {len(self.cluster_pairs)}")
            print(f"  Projection heads: {cfg.use_projection_heads}")
        print(f"  Save every:       {cfg.save_every}")
        print(f"  Step log interval:{cfg.step_log_interval}")
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

        for epoch in range(start_epoch, end_epoch):
            epoch_metrics = self.train_epoch(epoch)

            acc = 0.0
            if (epoch + 1) % cfg.eval_every == 0:
                acc = eval_accuracy(
                    self.student, self.tokenizer, self.test_data,
                    max_new_tokens=cfg.eval_max_new_tokens,
                )
                epoch_metrics["accuracy"] = acc

                if acc > best_acc:
                    best_acc = acc
                    if cfg.save_best:
                        self._save_weights_fast()
                        print(f"  Saved {STUDENT_WEIGHTS_FILE} (new best accuracy)")

            if cfg.save_every > 0 and (epoch + 1) % cfg.save_every == 0:
                self._save_weights_fast()
                print(f"  Saved {STUDENT_WEIGHTS_FILE} (epoch {epoch + 1})")

            self.history["epoch"].append(epoch + 1)
            for k, v in epoch_metrics.items():
                self.history[k].append(v)

            # Per-epoch history save (fsynced) + training state
            self._save_history()
            save_training_state(cfg.save_dir, self.optimizer, epoch + 1, best_acc)

            if epoch_metrics:
                ef = epoch_metrics.get("epoch_flops")
                flop_s = ""
                if ef is not None:
                    flop_s = f", FLOPs={format_flops(float(ef))}"
                if self._standard:
                    print(
                        f"Epoch {epoch + 1}/{end_epoch}: "
                        f"KL={epoch_metrics.get('kl_loss', float('nan')):.4f}, "
                        f"Acc={acc:.4f}{flop_s}"
                    )
                else:
                    print(
                        f"Epoch {epoch + 1}/{end_epoch}: "
                        f"KL={epoch_metrics.get('kl_loss', float('nan')):.4f}, "
                        f"Cluster={epoch_metrics.get('cluster_loss', float('nan')):.4f}, "
                        f"CKA={epoch_metrics.get('mean_cka', float('nan')):.4f}, "
                        f"Acc={acc:.4f}{flop_s}"
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
        print(f"  Saved {STUDENT_MODEL_DIR}/ (final)")

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

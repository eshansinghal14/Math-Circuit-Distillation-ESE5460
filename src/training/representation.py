"""Representation-level distillation shared by ``bert_kd`` and ``cka_kd``.

Both are baselines for the graph loss: the KL term ``standard_kd`` uses plus a
term on the student's internal activations, weighted by ``--lambda-rep`` the way
``--lambda-graph`` weights the graph term.

* ``bert_kd`` is the TinyBERT recipe (Jiao et al., 2020): MSE between a learned
  linear projection of the student's residual stream and the teacher's at mapped
  layers, plus MSE between attention patterns at the same layers.
* ``cka_kd`` is linear centred kernel alignment (Kornblith et al., 2019) between
  the two residual streams at mapped layers, loss ``1 - CKA``. It needs no
  projection and is invariant to isotropic scaling and orthogonal rotation of
  either side, so it constrains the geometry of the representation rather than
  its coordinates.

What the two share lives here: which layers are paired, which token positions
are matched, how activations are captured, the losses, the projector and its
least-squares initialisation, and a trainer base that is ``standard_kd``'s loop
with the representation term added and the same gradient diagnostics
``graph_kd`` records under ``--track-grad-metrics``.

Activations are captured with forward hooks rather than ``output_hidden_states``
so the layer semantics do not depend on the transformers version: hidden state
``j`` (1-based) is the residual stream leaving decoder layer ``j``, before the
final norm, for every ``j`` including the last. Attention patterns are the
post-softmax weights of layer ``j``'s self-attention, which only the eager
kernel materialises, so both models are switched to eager for the training
forward when they are needed. The student's gradient checkpointing is turned
off for that forward: under HF's default reentrant checkpointing every tensor a
hook sees inside a layer is produced in a no-grad replay and would silently
carry no gradient. The sequences here are short, so the memory cost is
negligible, and a runtime check refuses to continue if a captured student
tensor has no gradient.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (
    DataLoader,
    PromptAnswerDataset,
    collate_fn,
    eval_model,
    load_data,
    load_model,
    seed_all,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_loss.freeze import without_gradient_checkpointing
from training.utils import (
    eval_batch_size_for,
    DEFAULT_SEED,
    SWEEP_PARAMS,
    ParamChangeCanary,
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
# Layer pairing and token selection
# ─────────────────────────────────────────────────────────────────────────────


def resolve_layer_map(
    spec: str,
    n_student: int,
    n_teacher: int,
    student_layers: Sequence[int] = (),
) -> List[Tuple[int, int]]:
    """``(student_layer, teacher_layer)`` pairs, 1-based decoder layer indices.

    ``"uniform"`` is TinyBERT's ``g(m) = m * N / M``: student layer ``j`` pairs
    with teacher layer ``round(j * n_teacher / n_student)``, so a 16-layer
    student on a 32-layer teacher gets ``(1, 2), (2, 4), ..., (16, 32)``.
    ``student_layers`` restricts the uniform map to a subset of student layers.
    Any other spec is an explicit ``"s:t,s:t,..."`` list.
    """
    if spec.strip().lower() == "uniform":
        layers = list(student_layers) if student_layers else list(range(1, n_student + 1))
        pairs: List[Tuple[int, int]] = []
        for s in layers:
            if not 1 <= s <= n_student:
                raise ValueError(f"student layer {s} out of range 1..{n_student}")
            t = int(round(s * n_teacher / n_student))
            pairs.append((s, max(1, min(n_teacher, t))))
        return pairs
    if student_layers:
        raise ValueError("--match-layers only applies to --layer-map uniform")
    pairs = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        s_str, t_str = item.split(":")
        s, t = int(s_str), int(t_str)
        if not 1 <= s <= n_student:
            raise ValueError(f"student layer {s} out of range 1..{n_student}")
        if not 1 <= t <= n_teacher:
            raise ValueError(f"teacher layer {t} out of range 1..{n_teacher}")
        pairs.append((s, t))
    if not pairs:
        raise ValueError(f"empty layer map {spec!r}")
    return pairs


def matched_token_mask(
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    match_positions: str,
    keep_first_position: bool,
) -> torch.Tensor:
    """Boolean ``[B, L]`` mask of the token positions the representation term uses.

    ``all`` is every non-padding token, ``response`` only the answer tokens. The
    first position is dropped by default: in Llama it is the attention sink and
    carries massive activations that would dominate an MSE or a Gram matrix.
    """
    if match_positions == "all":
        mask = attention_mask.bool()
    elif match_positions == "response":
        mask = response_mask.bool()
    else:
        raise ValueError(f"unknown match_positions {match_positions!r}")
    if not keep_first_position:
        mask = mask.clone()
        mask[:, 0] = False
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Activation capture
# ─────────────────────────────────────────────────────────────────────────────


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise TypeError(f"{type(model).__name__} has no .model.layers; only Llama-style decoders are supported")
    return layers


class ActivationCapture:
    """Residual stream after each requested decoder layer, and optionally its attention pattern.

    ``hidden[j]`` is ``[B, L, D]`` leaving layer ``j`` (1-based, before the final
    norm); ``attention[j]`` is ``[B, H, L, L]`` post-softmax weights of layer
    ``j``'s self-attention. Use through :func:`capture_activations` so the hooks
    are removed after the forward.
    """

    def __init__(self, model: nn.Module, layers: Sequence[int], want_attention: bool) -> None:
        self.hidden: Dict[int, torch.Tensor] = {}
        self.attention: Dict[int, torch.Tensor] = {}
        self._handles: list = []
        blocks = decoder_layers(model)
        for j in layers:
            block = blocks[j - 1]
            self._handles.append(block.register_forward_hook(partial(self._on_block, j)))
            if want_attention:
                self._handles.append(block.self_attn.register_forward_hook(partial(self._on_attention, j)))

    def _on_block(self, j: int, module, args, output) -> None:
        self.hidden[j] = output[0] if isinstance(output, (tuple, list)) else output

    def _on_attention(self, j: int, module, args, output) -> None:
        weights = output[1] if isinstance(output, (tuple, list)) and len(output) > 1 else None
        if weights is None:
            raise RuntimeError(
                f"layer {j} self-attention returned no attention pattern; the model must run "
                f"with the eager kernel (see eager_attention) for attention matching."
            )
        self.attention[j] = weights

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


@contextlib.contextmanager
def capture_activations(model: nn.Module, layers: Sequence[int], want_attention: bool) -> Iterator[ActivationCapture]:
    cap = ActivationCapture(model, layers, want_attention)
    try:
        yield cap
    finally:
        cap.remove()


@contextlib.contextmanager
def eager_attention(*models: nn.Module):
    """Run ``models`` with the eager attention kernel, which materialises the pattern.

    SDPA never forms the softmax output, so the attention hook above would see
    ``None``. Restores each model's previous implementation on exit so eval and
    generation keep the fast kernel.
    """
    prev: list = []
    try:
        for m in models:
            if m is None:
                continue
            prev.append((m, m.config._attn_implementation))
            m.config._attn_implementation = "eager"
        yield
    finally:
        for m, impl in prev:
            m.config._attn_implementation = impl


def _fp32_math():
    """Disable autocast so the losses below run in fp32 on fp32 copies of the activations."""
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", enabled=False)
    return contextlib.nullcontext()


# ─────────────────────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────────────────────


def hidden_state_loss(student: torch.Tensor, teacher: torch.Tensor, kind: str) -> torch.Tensor:
    """``[N, D]`` student (already projected to the teacher width) against ``[N, D]`` teacher."""
    student = student.float()
    teacher = teacher.float()
    if kind == "mse":
        return F.mse_loss(student, teacher)
    if kind == "cosine":
        return (1.0 - F.cosine_similarity(student, teacher, dim=-1, eps=1e-8)).mean()
    raise ValueError(f"unknown hidden loss {kind!r}")


def attention_pattern_loss(
    student_attn: torch.Tensor,
    teacher_attn: torch.Tensor,
    query_mask: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Squared L2 distance between attention distributions, averaged over selected query rows.

    ``mean`` compares the head-averaged pattern of each layer, which is the only
    option when the two models' heads have no correspondence (two independently
    pretrained models). ``head`` is TinyBERT's literal head-to-head matching and
    requires equal head counts. Each query row's distance is summed over keys,
    so it is a distance between two probability vectors, then averaged over
    heads (``head`` mode) and over the rows ``query_mask`` selects. Padded key
    columns are exact zeros in both patterns and contribute nothing.
    """
    s = student_attn.float()
    t = teacher_attn.float()
    if mode == "mean":
        s = s.mean(dim=1)
        t = t.mean(dim=1)
    elif mode == "head":
        if s.shape[1] != t.shape[1]:
            raise ValueError(
                f"--attn-match head needs equal head counts, got student {s.shape[1]} vs teacher {t.shape[1]}"
            )
    else:
        raise ValueError(f"unknown attention match mode {mode!r}")
    sq = (s - t).pow(2).sum(dim=-1)  # [B, L] or [B, H, L]
    if mode == "head":
        sq = sq.mean(dim=1)  # [B, L]
    selected = sq[query_mask]
    if selected.numel() == 0:
        return sq.sum() * 0.0
    return selected.mean()


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Linear CKA between ``[N, Dx]`` and ``[N, Dy]`` over the N shared samples, in [0, 1].

    ``||Y'X||_F^2 / (||X'X||_F ||Y'Y||_F)`` on column-centred inputs (Kornblith et
    al., 2019, eq. 1 with linear kernels). Evaluated in sample space through the
    ``N x N`` Gram matrices when N is below the feature widths, else in feature
    space; the two are equal by ``tr(XX'YY') = ||Y'X||_F^2``.
    """
    x = x.float()
    y = y.float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    n = x.shape[0]
    if n <= max(x.shape[1], y.shape[1]):
        k = x @ x.T
        l = y @ y.T
        hsic = (k * l).sum()
        kk = (k * k).sum()
        ll = (l * l).sum()
    else:
        c = y.T @ x
        hsic = (c * c).sum()
        kk = ((x.T @ x) ** 2).sum()
        ll = ((y.T @ y) ** 2).sum()
    return hsic / (kk.sqrt() * ll.sqrt()).clamp_min(1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Hidden-state projector (bert_kd)
# ─────────────────────────────────────────────────────────────────────────────


class HiddenProjector(nn.Module):
    """One affine map ``d_student -> d_teacher`` per matched student layer (TinyBERT's ``W_h``)."""

    def __init__(self, student_layers: Sequence[int], d_student: int, d_teacher: int) -> None:
        super().__init__()
        self.proj = nn.ModuleDict({str(j): nn.Linear(d_student, d_teacher) for j in student_layers})

    def forward(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        return self.proj[str(layer)](x.float())


class LstsqStats:
    """Sufficient statistics of a ridge regression ``y ~ x W + b``, accumulated in float64.

    Storing ``X'X`` and ``X'Y`` rather than the tokens keeps memory independent
    of how many calibration batches are used.
    """

    def __init__(self, d_x: int, d_y: int, device) -> None:
        self.n = 0
        self.sx = torch.zeros(d_x, dtype=torch.float64, device=device)
        self.sy = torch.zeros(d_y, dtype=torch.float64, device=device)
        self.sxx = torch.zeros(d_x, d_x, dtype=torch.float64, device=device)
        self.sxy = torch.zeros(d_x, d_y, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        x = x.detach().to(torch.float64)
        y = y.detach().to(torch.float64)
        self.n += int(x.shape[0])
        self.sx += x.sum(dim=0)
        self.sy += y.sum(dim=0)
        self.sxx += x.T @ x
        self.sxy += x.T @ y

    @torch.no_grad()
    def solve(self, ridge: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(W [d_x, d_y], b [d_y])`` minimising ``||Xc W - Yc||^2 + lam ||W||^2`` with intercept.

        ``lam = ridge * trace(Xc'Xc) / d_x``, i.e. relative to the mean feature
        variance, so the same ``ridge`` means the same thing at every layer.
        """
        if self.n < 2:
            raise RuntimeError("least-squares projector init needs at least two tokens")
        xm = self.sx / self.n
        ym = self.sy / self.n
        g = self.sxx - self.n * torch.outer(xm, xm)
        c = self.sxy - self.n * torch.outer(xm, ym)
        lam = ridge * g.diagonal().mean().clamp_min(1e-12)
        eye = torch.eye(g.shape[0], dtype=g.dtype, device=g.device)
        w = torch.linalg.solve(g + lam * eye, c)
        b = ym - xm @ w
        return w, b


@torch.no_grad()
def fit_projector_least_squares(projector: HiddenProjector, stats: Dict[int, LstsqStats], ridge: float) -> None:
    for j, st in stats.items():
        w, b = st.solve(ridge)
        lin = projector.proj[str(j)]
        lin.weight.copy_(w.T.to(lin.weight.dtype))
        lin.bias.copy_(b.to(lin.bias.dtype))


# ─────────────────────────────────────────────────────────────────────────────
# Config and trainer base
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RepKDConfig:
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
    save_dir: str = "results/rep_kd"
    eval_every_n_steps: int = 1
    save_every_n_steps: int = 0
    grad_accum_steps: int = 1
    eval_datasets: List[str] = field(default_factory=list)
    test_limit: Optional[int] = None
    dtype: Optional[str] = None
    bos_mode: str = "on"
    resume: bool = False
    seed: int = _SEED
    track_flops: bool = False
    # representation term
    lambda_rep: float = 0.1
    layer_map: str = "uniform"
    match_layers: List[int] = field(default_factory=list)
    match_positions: str = "all"
    keep_first_position: bool = False
    track_grad_metrics: bool = False


REP_SWEEP_PARAMS = SWEEP_PARAMS + ("lambda_rep",)
"""Sweepable names for the representation trainers: the shared set plus their own weight.

Scoped here rather than added to SWEEP_PARAMS because _sweep_signature hashes
``getattr(args, name, None)`` for every name in the tuple -- a global addition would put
``("lambda_rep", None)`` into graph_kd's and standard_kd's digests and re-key every run
already on disk.
"""


def add_rep_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("representation_args")
    group.add_argument("--lambda-rep", "--lambda_rep", type=float, nargs="+", default=[0.1],
                       dest="lambda_rep",
                       help="Weight of the representation term next to the KL term; the analogue "
                            "of graph_kd's --lambda-graph. Several values sweep; see --lr.")
    group.add_argument("--layer-map", "--layer_map", type=str, default="uniform", dest="layer_map",
                       help="'uniform' pairs student layer j with teacher layer round(j*N/M) "
                            "(TinyBERT); or an explicit 's:t,s:t' list of 1-based decoder layers.")
    group.add_argument("--match-layers", "--match_layers", type=int, nargs="*", default=[],
                       dest="match_layers", metavar="LAYER",
                       help="Student layers to match under the uniform map (default: all).")
    group.add_argument("--match-positions", "--match_positions", choices=["all", "response"],
                       default="all", dest="match_positions",
                       help="Token positions the term is computed on: every non-pad token, or "
                            "only answer tokens.")
    group.add_argument("--keep-first-position", "--keep_first_position", action="store_true",
                       dest="keep_first_position",
                       help="Include position 0 (the attention sink, with massive activations) "
                            "in the matched tokens. Off by default.")
    group.add_argument("--track-grad-metrics", "--track_grad_metrics", action="store_true",
                       dest="track_grad_metrics",
                       help="Record |g_KL|, |g_rep|, their ratio, cosine and sign-flip fraction "
                            "per step, like graph_kd. Costs one extra gradient snapshot per step.")


class RepKDTrainer:
    """``standard_kd``'s loop with a representation term. Subclasses supply the term."""

    REP_NAME = "Rep"      # label of the representation loss in the step log and curves
    RUN_NAME = "Rep-KD"   # label of the run banner

    def __init__(
        self,
        config: RepKDConfig,
        train_data: Dict[str, Any],
        test_data: Dict[str, Any],
        shared: Dict[str, Any] | None = None,
    ) -> None:
        """``shared`` is run_seeds' cross-seed dict (teacher, baselines); None loads everything."""
        self.config = config
        self.shared = shared
        seed_all(config.seed)

        # fp32 master weights with a bf16 autocast forward; see training/utils.py.
        # The teacher is inference-only and stays bf16.
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

        n_student = len(decoder_layers(self.model))
        n_teacher = len(decoder_layers(self.teacher))
        self.layer_pairs = resolve_layer_map(config.layer_map, n_student, n_teacher, config.match_layers)
        self.student_layers = [s for s, _ in self.layer_pairs]
        self.teacher_layers = [t for _, t in self.layer_pairs]

        self.optimizer = make_optimizer(self.model, config.learning_rate)
        # Subclasses may build extra trainable modules and add them to the optimizer
        # here; it must happen before resume_training_state so the optimizer state
        # dict has the same param groups it was saved with.
        self._setup_representation()

        self.history: Dict[str, List] = defaultdict(list)
        self._train_step = 0
        self._last_save_step = 0
        if config.resume:
            step, history, extra = resume_training_state(self.model, self.optimizer, config.save_dir)
            self.history = defaultdict(list, history)
            self._train_step = step
            self._last_save_step = step
            self._load_extra_state(extra)

    # ── subclass hooks ────────────────────────────────────────────────────────

    def _setup_representation(self) -> None:
        pass

    def _needs_attention(self) -> bool:
        return False

    def _extra_module(self) -> Optional[nn.Module]:
        """Trainable parameters outside the student (saved with the checkpoint)."""
        return None

    def _rep_loss(
        self,
        s_cap: ActivationCapture,
        t_cap: ActivationCapture,
        token_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """The representation term for one batch and its logged components."""
        raise NotImplementedError

    def _banner_extra(self) -> str:
        return ""

    # ── checkpoint extras ─────────────────────────────────────────────────────

    def _extra_state(self) -> Optional[Dict[str, Any]]:
        module = self._extra_module()
        return {"module": module.state_dict()} if module is not None else None

    def _load_extra_state(self, extra: Optional[Dict[str, Any]]) -> None:
        module = self._extra_module()
        if module is None:
            return
        if not extra or "module" not in extra:
            raise RuntimeError(
                "checkpoint has no state for the representation module; it was written by a "
                "trainer without one, so it cannot be resumed here."
            )
        module.load_state_dict(extra["module"])

    # ── forward ───────────────────────────────────────────────────────────────

    def _autocast(self):
        return student_autocast()

    def _forward_pair(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        want_attention: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, ActivationCapture, ActivationCapture]:
        """Student and teacher forwards with activations captured. Returns logits and captures."""
        if want_attention is None:
            want_attention = self._needs_attention()
        with contextlib.ExitStack() as stack:
            if want_attention:
                stack.enter_context(eager_attention(self.model, self.teacher))
            stack.enter_context(without_gradient_checkpointing(self.model))
            s_cap = stack.enter_context(capture_activations(self.model, self.student_layers, want_attention))
            t_cap = stack.enter_context(capture_activations(self.teacher, self.teacher_layers, want_attention))
            with self._autocast():
                s_logits = self.model(input_ids, attention_mask=attention_mask).logits
            with torch.no_grad():
                t_logits = self.teacher(input_ids, attention_mask=attention_mask).logits
        return s_logits, t_logits, s_cap, t_cap

    @staticmethod
    def _check_student_grads(s_cap: ActivationCapture) -> None:
        for j, h in s_cap.hidden.items():
            if not h.requires_grad:
                raise RuntimeError(f"student hidden state at layer {j} carries no gradient; the "
                                   f"representation term would be inert (checkpointing replay?)")
        for j, a in s_cap.attention.items():
            if not a.requires_grad:
                raise RuntimeError(f"student attention pattern at layer {j} carries no gradient")

    # ── eval ──────────────────────────────────────────────────────────────────

    def _eval_on(self, model, dataset_name: str, test_dataset: PromptAnswerDataset) -> float:
        cfg = self.config
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

    # ── training ──────────────────────────────────────────────────────────────

    def _snapshot_grads(self) -> Dict[str, torch.Tensor]:
        return {
            n: p.grad.detach().clone()
            for n, p in self.model.named_parameters() if p.grad is not None
        }

    def train_epoch(self, *, max_steps: Optional[int] = None) -> Dict[str, float]:
        self.model.train()
        cfg = self.config
        grad_accum = cfg.grad_accum_steps
        track = cfg.track_grad_metrics and cfg.lambda_rep > 0
        total_kl = 0.0
        total_rep = 0.0
        n_steps = 0
        acc: Dict[str, float] = defaultdict(float)
        accum_clip = 0.0
        accum_flops = 0
        micro_step = 0

        # Counts matmul/attention FLOPs of the forward and backward passes; a no-op
        # unless --track-flops. Entered around the compute only, as in standard_kd.
        flop_counter = step_flop_counter(cfg.track_flops)

        self.optimizer.zero_grad()
        for batch in self.loader:
            if max_steps is not None and n_steps >= max_steps:
                break
            input_ids = batch["input_ids"].to(_DEVICE)
            attention_mask = batch["attention_mask"].to(_DEVICE)
            response_mask = batch["response_mask"].to(_DEVICE)
            flop_counter.reset()

            with flop_counter:
                s_logits, t_logits, s_cap, t_cap = self._forward_pair(input_ids, attention_mask)
            self._check_student_grads(s_cap)
            self._last_tf_acc = first_answer_token_accuracy(s_logits.detach(), input_ids, response_mask)

            token_mask = matched_token_mask(
                attention_mask, response_mask, cfg.match_positions, cfg.keep_first_position,
            )
            with flop_counter:
                kl = kl_loss(
                    s_logits, t_logits,
                    kl_position_mask(attention_mask, response_mask, cfg.kl_tokens,
                                     input_ids=input_ids,
                                     bos_token_id=self.tokenizer.bos_token_id),
                    cfg.temperature, cfg.kl_token_chunk_size,
                ) / grad_accum
                with _fp32_math():
                    rep, comps = self._rep_loss(s_cap, t_cap, token_mask)
            rep = rep / grad_accum

            if not (torch.isfinite(kl) and torch.isfinite(rep)):
                print(f"  step {self._train_step + 1} | WARN: non-finite loss "
                      f"(KL={float(kl):.4g}, {self.REP_NAME}={float(rep):.4g}); skipping batch")
                micro_step += 1
                if micro_step % grad_accum == 0:
                    self.optimizer.zero_grad()
                continue

            if track:
                # Two backwards through one forward so the KL and representation
                # gradients can be compared, exactly as graph_kd does with its
                # separate graph forward. grads_start handles carried-over
                # accumulation the same way.
                grads_start = self._snapshot_grads() if grad_accum > 1 else None
                with flop_counter:
                    kl.backward(retain_graph=True)
                grads_mid = self._snapshot_grads()
                with flop_counter:
                    (cfg.lambda_rep * rep).backward()
                dot = kl_sq = rep_sq = 0.0
                flipped = total_elems = 0
                for n, p in self.model.named_parameters():
                    if p.grad is None:
                        continue
                    mid = grads_mid.get(n)
                    g_kl = mid.float() if mid is not None else torch.zeros_like(p.grad, dtype=torch.float32)
                    if grads_start is not None:
                        start = grads_start.get(n)
                        if start is not None:
                            g_kl = g_kl - start.float()
                    g_rep = (p.grad.float() - (mid.float() if mid is not None else 0.0))
                    dot += float((g_kl * g_rep).sum().item())
                    kl_sq += float((g_kl * g_kl).sum().item())
                    rep_sq += float((g_rep * g_rep).sum().item())
                    flipped += int((torch.sign(g_kl + g_rep) != torch.sign(g_kl)).sum().item())
                    total_elems += g_kl.numel()
                denom = (kl_sq ** 0.5) * (rep_sq ** 0.5)
                acc["kl_gnorm"] += kl_sq ** 0.5
                acc["rep_gnorm"] += rep_sq ** 0.5
                acc["cos"] += dot / denom if denom > 0 else 0.0
                acc["flip"] += flipped / total_elems if total_elems else 0.0
                del grads_mid, grads_start
            else:
                total = kl + cfg.lambda_rep * rep if cfg.lambda_rep > 0 else kl
                with flop_counter:
                    total.backward()
            accum_flops += flop_counter.flops

            acc["kl"] += float(kl.item())
            acc["rep"] += float(rep.item())
            for k, v in comps.items():
                acc[k] += float(v) / grad_accum
            micro_step += 1

            if micro_step % grad_accum == 0:
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), _GRAD_CLIP)
                extra = self._extra_module()
                if extra is not None:
                    torch.nn.utils.clip_grad_norm_(extra.parameters(), _GRAD_CLIP)
                accum_clip = float(total_norm)
                if torch.isfinite(total_norm):
                    canary = ParamChangeCanary(self.model) if self._train_step == 0 else None
                    self._last_lr = apply_lr_schedule(
                        self.optimizer, self._train_step + 1, cfg.steps, cfg.learning_rate,
                        cfg.warmup_steps, cfg.lr_floor)
                    self.optimizer.step()
                    if canary is not None:
                        log_first_step_canary(canary.report(self.model), self.history)
                else:
                    print(f"  step {self._train_step + 1} | WARN: non-finite grad norm "
                          f"({total_norm.item()}); skipping optimizer step")
                self.optimizer.zero_grad()

                self._train_step += 1
                self.history["train_step"].append(self._train_step)
                self.history["step_kl_loss"].append(acc["kl"])
                self.history["step_tf_acc"].append(self._last_tf_acc)
                self.history["step_lr"].append(self._last_lr)
                self.history["step_rep_loss"].append(acc["rep"])
                comp_str = ""
                for k in comps:
                    self.history[f"step_{k}"].append(acc[k])
                    comp_str += f" | {k}={acc[k]:.4f}"
                total_kl += acc["kl"]
                total_rep += acc["rep"]
                gnorm_str = ""
                if track:
                    ratio = acc["rep_gnorm"] / acc["kl_gnorm"] if acc["kl_gnorm"] else float("nan")
                    mean_cos = acc["cos"] / grad_accum
                    mean_flip = acc["flip"] / grad_accum
                    for key, val in (
                        ("step_kl_gnorm", acc["kl_gnorm"]),
                        ("step_rep_gnorm", acc["rep_gnorm"]),
                        ("step_grad_ratio", ratio),
                        ("step_grad_cosine", mean_cos),
                        ("step_grad_signflip", mean_flip),
                        ("step_clip_norm", accum_clip),
                    ):
                        self.history[key].append(val)
                    gnorm_str = (
                        f" | |g_KL|={acc['kl_gnorm']:.4f} | |g_rep|={acc['rep_gnorm']:.4f}"
                        f" | ratio={ratio:.4f} | cos={mean_cos:+.4f}"
                        f" | signflip={mean_flip:.3f} | clip={accum_clip:.3f}"
                    )
                flops_str = ""
                if cfg.track_flops:
                    self.history["step_flops"].append(accum_flops)
                    flops_str = f" | FLOPs={accum_flops:.3e}"
                print(
                    f"  step {self._train_step} | KL={acc['kl']:.4f} | "
                    f"{self.REP_NAME}={acc['rep']:.4f}{comp_str}{gnorm_str}{flops_str}"
                )
                self._last_save_step = maybe_save_periodic_checkpoint(
                    self.model, self.tokenizer, cfg.save_dir,
                    self._train_step, cfg.save_every_n_steps,
                    self._last_save_step, self.history,
                    optimizer=self.optimizer, extra_state=self._extra_state(),
                )
                acc.clear()
                accum_clip = 0.0
                accum_flops = 0
                n_steps += 1

        denom = max(n_steps, 1)
        return {"kl_loss": total_kl / denom, "rep_loss": total_rep / denom}

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
            self.history["layer_pairs"] = [list(p) for p in self.layer_pairs]
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

        pairs_str = ",".join(f"{s}:{t}" for s, t in self.layer_pairs)
        print(
            f"{self.RUN_NAME} | student={cfg.model} | teacher={cfg.teacher} | dataset={cfg.dataset}"
            f" | steps={cfg.steps} | lr={cfg.learning_rate} | temp={cfg.temperature}"
            f" | lambda_rep={cfg.lambda_rep} | layers={pairs_str}"
            f" | positions={cfg.match_positions}{'' if cfg.keep_first_position else ' (skip pos 0)'}"
            f"{self._banner_extra()}"
        )

        while self._train_step < cfg.steps:
            remaining = cfg.steps - self._train_step
            metrics = self.train_epoch(max_steps=min(cfg.eval_every_n_steps, remaining))
            if not metrics:
                break
            self.history["kl_loss"].append(metrics["kl_loss"])
            self.history["rep_loss"].append(metrics["rep_loss"])

            acc = self._eval()
            self.history["accuracy"].append(acc)
            self.history["accuracy_step"].append(self._train_step)
            extra_accs = self._eval_all_extra()
            for ds, ds_acc in extra_accs.items():
                self.history[f"accuracy_{ds}"].append(ds_acc)
            extra_str = "".join(f" | {ds}={a:.4f}" for ds, a in extra_accs.items())
            print(f"  [eval] step {self._train_step}/{cfg.steps} | Acc={acc:.4f}{extra_str}")
            refresh_curves(self.history, cfg.save_dir, step=self._train_step,
                           losses=[("step_kl_loss", "KL Loss"), ("step_rep_loss", f"{self.REP_NAME} Loss")])

        save_history(self.history, cfg.save_dir)
        save_curves(
            self.history, cfg.save_dir,
            losses=[("step_kl_loss", "KL Loss"), ("step_rep_loss", f"{self.REP_NAME} Loss")],
        )
        if cfg.save_every_n_steps != 0:
            save_checkpoint(self.model, self.tokenizer, cfg.save_dir)
        else:
            print("--save-every-n-steps is 0; not writing the trained model")
        print(f"Results saved to: {cfg.save_dir}")
        return dict(self.history)


def base_config_kwargs(args: argparse.Namespace, dir_root: str, seed: int | None = None) -> Dict[str, Any]:
    """The RepKDConfig fields every representation trainer's CLI shares; ``seed`` overrides the CLI's first."""
    return dict(
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
        save_dir=os.path.join(dir_root, args.save_dir),
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
        seed=args.seeds[0] if seed is None else seed,
        track_flops=args.track_flops,
        lambda_rep=args.lambda_rep,
        layer_map=args.layer_map,
        match_layers=args.match_layers,
        match_positions=args.match_positions,
        keep_first_position=args.keep_first_position,
        track_grad_metrics=args.track_grad_metrics,
    )

"""Training-time graph auxiliary loss helpers."""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import contextlib
import random
import time

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from utils import parse_response, tokenize_prompt_answer

import torch

from graph_loss.create_graph import create_graph
from graph_loss.graph import (
    aggregate_supernode_adjacency,
    SuperGraph,
    normalize_matrix,
)
from graph_loss.hf_adapter import HFLlamaGraphAdapter
from graph_loss.token_attribution import salient_logits, token_attribution
from graph_loss.loss import compute_graph_loss, edge_similarity
from graph_loss.utils import normalize_node_labels


@dataclass
class GraphAuxConfig:
    lambda_graph: float = 1.0
    graph_dtype: torch.dtype | None = None
    teacher_prop_neurons_per_layer: float = 0.1
    student_prop_neurons_per_layer: float = 0.1

    top_k_logits: float | None = 0.95
    temperature: float = 1.0
    teacher_graph_batch_size: int = 512
    student_graph_batch_size: int = 1
    verbose: bool = False

    student_anova_range_radius: int = 0
    student_nodes_per_label: int = 10
    anova_neuron_chunk: int | None = None
    teacher_nodes_per_label: int = 10
    student_mlp_input_cache_path: str | None = None
    mlp_input_cache: dict | None = None
    activation_write_result_cache: dict = field(default_factory=dict)
    graph_loss_type: Literal["jsd", "kld", "mse", "mse-norm", "mse-scale", "rel-mse"] = "jsd"
    # How the K x K supernode adjacency is aggregated from the attribution graph,
    # applied identically to teacher and student. See
    # graph_loss.graph.aggregate_supernode_adjacency for what each keeps.
    supergraph_aggregation: Literal["normalised", "raw-signed"] = "normalised"
    # Append the token-embedding nodes as extra source columns (raw-signed only
    # makes sense with it; harmless otherwise).
    token_source_columns: bool = False
    token_path_rows: str = "weighted"
    token_path_micro_batch: int = 4
    # Padded tokens (prompts x longest sequence) per token-path micro-batch; the
    # student's double-backward graph scales with this, not with the prompt count.
    token_path_micro_tokens: int | None = 1024
    # Score only the teacher's top-k attributed positions plus one aggregate
    # "everywhere else" column; 0 keeps the full profile over every prompt token.
    token_path_top_tokens: int = 0
    # Stop-gradient the attention pattern / RMSNorm denominator when computing
    # attribution-graph edges, matching the published direct-path linearisation.
    # Applied identically to teacher and student. See graph_loss.freeze.
    freeze_attention: bool = False
    freeze_rms_norm: bool = False
    # Replace frac_external with 1 in the supernode aggregation, making an edge a
    # plain mean over members. Must be applied to teacher and student alike or the
    # two supergraphs are not comparable.
    constant_node_weighting: bool = False
    graph_node_labels: list[str] | None = None
    teacher_mlp_input_cache: dict | None = None
    tokens_dla_nodes: bool = False
    compare_n_tokens: int | None = None
    compare_ans_token: bool = False
    dataset_name: str = "local"
    # Control arm: scramble the teacher's target so it carries no information
    # about the teacher's routing while staying the same kind of object. Each
    # row of the K x K supernode adjacency is permuted by a fixed derangement of
    # its columns, one per row index, drawn once per K from scramble_seed and
    # reused for every prompt, so the fake target is consistent and learnable
    # rather than fresh noise. See scramble_teacher_rows.
    scramble_teacher_graph: bool = False
    scramble_seed: int = 0
    scramble_permutations: dict = field(default_factory=dict)
    # Per-prompt cache of the teacher's target (see TeacherTargetCache); None
    # recomputes the teacher's graph for every prompt.
    teacher_target_cache: Any = None

    def __post_init__(self) -> None:
        # 'tokens' in graph_node_labels means token_source_columns, never an ANOVA
        # category: strip it here so the cache key, the label filter and the ANOVA
        # selection all see only real categories, and a 'tokens'-only list means
        # no ANOVA (None), i.e. the arg-token + DLA construction with token columns.
        if self.graph_node_labels is not None:
            labels, tokens_label = normalize_node_labels(self.graph_node_labels)
            self.graph_node_labels = labels or None
            self.token_source_columns = self.token_source_columns or tokens_label


def _aggregate_supergraph_adjacency(
    graph, supernodes: list[list[int]], constant_node_weighting: bool = False,
    aggregation: str = "normalised", token_source_columns: bool = False,
) -> SuperGraph:
    """Aggregate a differentiable graph adjacency using fixed supernode membership.

    Delegates to graph_loss.graph.aggregate_supernode_adjacency, the same
    function the teacher's build_super_graph uses, so the two sides cannot drift;
    it is built out-of-place so the gradient from the edge loss flows back through
    supernode_adjacency_matrix -> adjacency_matrix -> source vectors -> weights.
    """
    supernode_adj_matrix = aggregate_supernode_adjacency(
        graph, supernodes,
        aggregation=aggregation,
        constant_node_weighting=constant_node_weighting,
        token_source_columns=token_source_columns,
    )
    return SuperGraph(
        supernode_adjacency_matrix=supernode_adj_matrix,
        supernodes=supernodes,
    )


def _derangements(k: int, rng: random.Random, width: int | None = None) -> list[list[int]]:
    """``k`` random derangements of ``range(width)`` (default: ``width = k``).

    ``width`` is separate because a target can have more rows than columns --
    token-path with rows='all' is L x T with L > T -- and drawing one
    derangement per column then left too few for the rows.

    A derangement has no fixed point, so no entry of a scrambled row stays at
    its true source. ``k < 2`` has no derangement and gets the identity.
    """
    width = k if width is None else width
    perms: list[list[int]] = []
    for _ in range(k):
        if width < 2:
            perms.append(list(range(width)))
            continue
        while True:
            p = list(range(width))
            rng.shuffle(p)
            if all(p[i] != i for i in range(width)):
                perms.append(p)
                break
    return perms


_TOPK_LOG_CALLS = [0]
_TOPK_LOG_LIMIT = 3


def _top_token_view(
    W_T: torch.Tensor, W_S: torch.Tensor, k: int, epsilon: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Both profiles reduced to the teacher's top-k positions plus a 'rest' column.

    A full token-path row is a distribution over every prompt token, so on a
    450-token context ~440 near-zero entries set the scale of any distance and
    the handful of positions that carry the teacher's evidence barely move it.
    Scoring only the teacher's top-k restores the dynamic range.

    The extra column is the mass each model puts *outside* that set, and it is
    what keeps the loss proper: without it the student's attribution off the
    teacher's positions is unconstrained, so it could attend to distractors for
    free and still score perfectly. With it the loss asks two things -- does the
    student concentrate as much on the teacher's positions, and does it spread
    that mass the same way among them.

    The indices come from the teacher, which is detached, so this only masks the
    student's gradient; both the selected entries and the residual stay
    differentiable. A k at or above the row width is a no-op.
    """
    width = int(W_T.shape[1])
    if k <= 0 or k >= width:
        return W_T, W_S
    t_abs, s_abs = W_T.abs(), W_S.abs()
    idx = t_abs.topk(k, dim=1).indices
    t_sel, s_sel = torch.gather(t_abs, 1, idx), torch.gather(s_abs, 1, idx)
    t_rest = (t_abs.sum(dim=1, keepdim=True) - t_sel.sum(dim=1, keepdim=True)).clamp(min=0.0)
    s_rest = (s_abs.sum(dim=1, keepdim=True) - s_sel.sum(dim=1, keepdim=True)).clamp(min=0.0)
    if _TOPK_LOG_CALLS[0] < _TOPK_LOG_LIMIT:
        _TOPK_LOG_CALLS[0] += 1
        frac = float((t_sel.sum() / t_abs.sum().clamp(min=epsilon)).item())
        print(f"  [graph] token-path top-{k} of {width} positions: "
              f"teacher mass in the top-{k} = {frac:.3f}")
        if _TOPK_LOG_CALLS[0] == _TOPK_LOG_LIMIT:
            print("  [graph] further top-k reductions are not logged.")
    return torch.cat([t_sel, t_rest], dim=1), torch.cat([s_sel, s_rest], dim=1)


_SCRAMBLE_LOG_FULL_WIDTH = 24
_SCRAMBLE_LOG_SHAPES = 3


def _log_scramble_shape(config: GraphAuxConfig, n_rows: int, k: int, perms: list) -> None:
    """One line per new target shape, and only while the lines stay informative.

    Under a supernode aggregation the columns are supernodes: a handful of shapes
    for a whole run, each narrow enough that printing the permutation is a
    readable record of what the control actually scrambled. token-path's columns
    are token positions, so the cache key is the prompt length -- a new shape per
    distinct prompt length, each printing a list as long as the prompt. On
    HotpotQA that is hundreds of ~500-element lines and it buries the step log.

    So the full permutation is still printed at supernode widths, and anything
    wider prints one summary line for the first few shapes and nothing after.
    """
    n_shapes = len(config.scramble_permutations)
    if k <= _SCRAMBLE_LOG_FULL_WIDTH:
        print(f"  [graph] scrambled teacher target: {n_rows} rows x width {k}, "
              f"row permutations {perms}")
        return
    if n_shapes <= _SCRAMBLE_LOG_SHAPES:
        head = ", ".join(str(int(x)) for x in perms[0][:8])
        print(f"  [graph] scrambled teacher target: {n_rows} rows x width {k} "
              f"(row 0: {head}, ...)")
        if n_shapes == _SCRAMBLE_LOG_SHAPES:
            print("  [graph] further scramble shapes are not logged: token-path keys "
                  "one permutation per prompt length.")


def scramble_teacher_rows(W_T: torch.Tensor, config: GraphAuxConfig) -> torch.Tensor:
    """Permute each row of the teacher's supernode adjacency by a fixed derangement.

    Row ``t`` of ``W_T`` is the teacher's routing profile for supernode ``t``:
    how much of its inbound weight comes from each source supernode. Scrambling
    reassigns those weights to the wrong sources, so the target keeps the
    teacher's numbers and per-row sparsity but says nothing about which
    supernode routes to which. The permutations are drawn once per matrix size
    ``K`` from ``config.scramble_seed`` and cached on the config, so every prompt
    of the same shape is scrambled the same way for the whole run. Arg-token
    supernode members are not bound to their token's position (they are chosen
    by read direction), so there is no a-priori causal support to respect and
    the whole row is permuted.
    """
    if W_T.ndim != 2:
        raise ValueError(f"Expected a 2-D supernode adjacency, got {tuple(W_T.shape)}")
    # Each row is permuted over all its columns (supernode and any token columns),
    # so a rectangular raw-signed target is scrambled the same way as a square one.
    n_rows, k = int(W_T.shape[0]), int(W_T.shape[1])
    key = (n_rows, k) if n_rows != k else k
    perms = config.scramble_permutations.get(key)
    if perms is None:
        perms = _derangements(n_rows, random.Random(config.scramble_seed * 1_000_003 + k), width=k)
        config.scramble_permutations[key] = perms
        _log_scramble_shape(config, n_rows, k, perms)
    idx = torch.tensor(perms[:n_rows], device=W_T.device, dtype=torch.long)
    return torch.gather(W_T, 1, idx)


# ─────────────────────────────────────────────────────────────────────────────
# Teacher target cache
# ─────────────────────────────────────────────────────────────────────────────
#
# The teacher is frozen, so for a fixed teacher-side configuration its target for
# a prompt (the K x K supernode adjacency, the supernode labels, the logit-target
# ids and the DLA reference logits) never changes. Every graph-KD run in a sweep
# nevertheless rebuilt it per prompt: a full attribution pass plus, on the ANOVA
# path, labelling ~45k pre-selected neurons against the activation cache. The
# cache below stores each prompt's target once, in one file per configuration.
# Entries are a few kilobytes, so a whole dataset's worth lives in memory and is
# written back atomically, merging with what another process may have added
# meanwhile. The DLA logits are kept as their top-256 entries: the only consumer,
# _dla_kl_scores_for_output, restricts itself to the top-100 tokens and
# renormalises, which the reconstruction reproduces to float precision.
#
# Staleness is handled by content, not by code version. The file records the
# digest of the graph_loss source it was written under; when a run opens it
# under different code, TeacherTargetCache.validate recomputes a few of its
# prompts with the current code and compares them with the stored entries. A
# refactor that changes no number keeps the file (and its hours of teacher
# work); a change that alters targets sets the file aside as stale and starts
# afresh. Hashing the code into the key instead threw the cache away on every
# commit that touched graph_loss, which during development is every day.

TEACHER_TARGET_CACHE_VERSION = 1
_TEACHER_DLA_TOPK = 256  # >= the dla_top_k_vocab (100) select_anova_supernodes uses


def graph_code_digest() -> str:
    """sha1 of every .py under graph_loss/ plus utils.py (tokenisation), so any
    change to how targets are built invalidates cached targets."""
    root = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
    files.append(os.path.join(os.path.dirname(root), "utils.py"))
    h = hashlib.sha1()
    for path in files:
        if not os.path.isfile(path):
            continue
        h.update(os.path.relpath(path, os.path.dirname(root)).replace(os.sep, "/").encode())
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


def teacher_target_cache_key(config: GraphAuxConfig, teacher_name: str) -> str:
    """Hash of everything the teacher's target depends on."""
    cache_meta = (config.teacher_mlp_input_cache or {}).get("meta", {})
    fields = {
        "version": TEACHER_TARGET_CACHE_VERSION,
        "teacher": teacher_name,
        "dataset": config.dataset_name,
        "labels": sorted(config.graph_node_labels or []),
        "nodes_per_label": config.teacher_nodes_per_label,
        "prop_neurons_per_layer": config.teacher_prop_neurons_per_layer,
        "top_k_logits": config.top_k_logits,
        "temperature": config.temperature,
        "freeze_attention": config.freeze_attention,
        "freeze_rms_norm": config.freeze_rms_norm,
        "constant_node_weighting": config.constant_node_weighting,
        "supergraph_aggregation": config.supergraph_aggregation,
        "token_source_columns": config.token_source_columns,
        "token_path_rows": config.token_path_rows,
        "tokens_dla_nodes": config.tokens_dla_nodes,
        "mlp_cache": {
            "n_prompts": cache_meta.get("n_prompts"),
            "dataset_key": cache_meta.get("dataset_key"),
            "model_name": cache_meta.get("model_name"),
        },
    }
    return hashlib.sha1(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _load_cache_file(path: str) -> tuple[dict[str, dict], dict[str, Any]]:
    """``(entries, meta)`` from a cache file; a file from before ``__meta__`` existed is all entries."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "entries" in data and "__meta__" in data:
        return dict(data["entries"]), dict(data["__meta__"])
    return dict(data), {}


def _dla_top_ids(dla: dict | None, k: int = 100) -> list[int] | None:
    if dla is None:
        return None
    k = min(k, int(dla["vals"].numel()))
    order = torch.topk(dla["vals"].float(), k).indices
    return sorted(int(i) for i in dla["ids"][order].tolist())


def _entries_match(a: dict, b: dict, *, rtol: float = 1e-2, atol: float = 1e-4) -> str | None:
    """None if two entries describe the same target, else a one-line reason.

    Labels and logit-target ids must agree exactly; the adjacency to a
    tolerance that absorbs GPU non-determinism in the attribution; the DLA
    reference must pick the same top-100 tokens.
    """
    if a["labels"] != b["labels"]:
        return f"supernode labels differ: {a['labels']} vs {b['labels']}"
    if not torch.equal(a["logit_ids"], b["logit_ids"]):
        return f"logit-target ids differ: {a['logit_ids'].tolist()} vs {b['logit_ids'].tolist()}"
    if tuple(a["adj"].shape) != tuple(b["adj"].shape):
        return f"adjacency shapes differ: {tuple(a['adj'].shape)} vs {tuple(b['adj'].shape)}"
    if not torch.allclose(a["adj"].float(), b["adj"].float(), rtol=rtol, atol=atol):
        return f"adjacency differs (max |diff| {(a['adj'].float() - b['adj'].float()).abs().max().item():.3g})"
    if _dla_top_ids(a["dla"]) != _dla_top_ids(b["dla"]):
        return "DLA reference logits pick different top tokens"
    return None


class TeacherTargetCache:
    """Per-prompt teacher targets, persisted to one file per teacher-side configuration."""

    def __init__(self, path: str | None) -> None:
        self.path = path
        self.entries: dict[str, dict] = {}
        self.meta: dict[str, Any] = {}
        self.hits = 0
        self.misses = 0
        self._dirty = 0
        if path and os.path.isfile(path):
            self.entries, self.meta = _load_cache_file(path)

    def __len__(self) -> int:
        return len(self.entries)

    def validate(
        self,
        compute: Callable[[str, Any], dict],
        *,
        n_probe: int = 2,
        log: Callable[[str], None] = print,
    ) -> bool:
        """Confirm the stored targets are what the current code produces.

        Skipped when the file was written under the current graph_loss source.
        Otherwise ``n_probe`` cached prompts are rebuilt with ``compute(prompt,
        answer)`` and compared entry by entry. On agreement the file is kept and
        stamped with the current digest; on disagreement it is renamed
        ``<path>.stale-<digest>`` and this cache starts empty. Returns whether
        the existing entries were kept.
        """
        current = graph_code_digest()
        stored = self.meta.get("code_digest")
        if not self.entries:
            self.meta["code_digest"] = current
            return True
        if stored == current:
            return True
        prompts = sorted(self.entries)[:n_probe]
        log(f"Teacher target cache was written under different code ({stored or 'unknown'} -> {current}); "
            f"rebuilding {len(prompts)} probe prompt(s) to check it still matches...")
        problems = []
        for prompt in prompts:
            old = self.entries[prompt]
            new = compute(prompt, old.get("answer", "0"))
            reason = _entries_match(old, new)
            if reason is not None:
                problems.append(f"{prompt!r}: {reason}")
        if not problems:
            self.meta["code_digest"] = current
            self._dirty += 1  # persist the new stamp on the next flush
            log(f"  probes match; keeping {len(self.entries)} cached targets")
            return True
        stale = f"{self.path}.stale-{stored or 'unknown'}"
        log("  WARNING: cached teacher targets no longer match the current code; setting the file aside "
            f"as {stale} and starting an empty cache.\n    " + "\n    ".join(problems))
        if self.path and os.path.isfile(self.path):
            os.replace(self.path, stale)
        self.entries = {}
        self.meta = {"code_digest": current}
        self._dirty = 0
        return False

    def get(self, prompt: str) -> dict | None:
        entry = self.entries.get(prompt)
        if entry is None:
            self.misses += 1
        else:
            self.hits += 1
        return entry

    def put(self, prompt: str, entry: dict) -> None:
        self.entries[prompt] = entry
        self._dirty += 1

    def flush(self) -> bool:
        """Write new entries to ``path``, merged with whatever is there now. Returns whether it wrote."""
        if not self.path or self._dirty == 0:
            return False
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        merged: dict[str, dict] = {}
        if os.path.isfile(self.path):
            try:
                merged, _ = _load_cache_file(self.path)
            except Exception as e:  # a half-written file from a crash elsewhere
                logger.warning("could not read %s (%s); overwriting it", self.path, e)
        merged.update(self.entries)
        meta = dict(self.meta)
        meta.setdefault("code_digest", graph_code_digest())
        meta["version"] = TEACHER_TARGET_CACHE_VERSION
        tmp = f"{self.path}.tmp{os.getpid()}"
        torch.save({"__meta__": meta, "entries": merged}, tmp)
        os.replace(tmp, self.path)
        self.entries = merged
        self.meta = meta
        self._dirty = 0
        return True


def _teacher_target_entry(
    supergraph: SuperGraph,
    logit_token_ids: torch.Tensor,
    dla_logits: torch.Tensor | None,
    answer: Any = None,
) -> dict:
    entry: dict[str, Any] = {
        "answer": None if answer is None else str(answer),
        "adj": supergraph.supernode_adjacency_matrix.detach().cpu().clone(),
        "supernodes": [[int(i) for i in sn] for sn in supergraph.supernodes],
        "labels": [list(lbls) for lbls in (supergraph.supernode_labels or [])],
        "logit_ids": logit_token_ids.detach().cpu().clone(),
        "dla": None,
    }
    if dla_logits is not None:
        k = min(_TEACHER_DLA_TOPK, int(dla_logits.numel()))
        vals, ids = torch.topk(dla_logits.detach().float(), k)
        entry["dla"] = {
            "ids": ids.cpu(), "vals": vals.cpu(),
            "vocab": int(dla_logits.numel()), "dtype": str(dla_logits.dtype).removeprefix("torch."),
        }
    return entry


def _teacher_target_from_entry(
    entry: dict, device: torch.device,
) -> tuple[SuperGraph, torch.Tensor, torch.Tensor | None]:
    supergraph = SuperGraph(
        supernode_adjacency_matrix=entry["adj"].to(device),
        supernodes=[list(sn) for sn in entry["supernodes"]],
        supernode_labels=[list(lbls) for lbls in entry["labels"]],
    )
    logit_ids = entry["logit_ids"].to(device)
    dla = entry["dla"]
    dla_logits: torch.Tensor | None = None
    if dla is not None:
        full = torch.full((dla["vocab"],), float("-inf"), dtype=torch.float32)
        full[dla["ids"]] = dla["vals"]
        dla_logits = full.to(dtype=getattr(torch, dla["dtype"]), device=device)
    return supergraph, logit_ids, dla_logits


def _compute_teacher_target(
    prompt: str,
    answer: Any,
    teacher_adapter: HFLlamaGraphAdapter,
    config: GraphAuxConfig,
    device: torch.device,
) -> tuple[SuperGraph, torch.Tensor, torch.Tensor | None]:
    """The teacher's supergraph, logit-target ids and DLA reference logits for one prompt."""
    token_path = config.supergraph_aggregation == "token-path"
    if token_path:
        return _token_path_teacher_target(prompt, answer, teacher_adapter, config, device)
    with torch.enable_grad():
        teacher_result = create_graph(
            teacher_adapter,
            prompt,
            prop_neurons_per_layer=config.teacher_prop_neurons_per_layer,
            top_k_logits=config.top_k_logits,
            temperature=config.temperature,
            batch_size=config.teacher_graph_batch_size,
            verbose=config.verbose,
            node_labels=config.graph_node_labels,
            mlp_input_cache=config.teacher_mlp_input_cache,
            nodes_per_label=config.teacher_nodes_per_label,
            no_grad_supergraph=True,
            build_create_graph=False,
            detach_result=True,
            freeze_attention=config.freeze_attention,
            freeze_rms_norm=config.freeze_rms_norm,
            constant_node_weighting=config.constant_node_weighting,
            supergraph_aggregation=_inner_aggregation(config),
            token_source_columns=config.token_source_columns,
        )
    # Same ids as the KD batch row (BOS + prompt, answer + EOS): the teacher's
    # DLA reference logits at the last prompt position must come from the same
    # BOS-prefixed forward that create_graph ran on the prompt above.
    prompt_ids, answer_ids = tokenize_prompt_answer(
        teacher_adapter.tokenizer, prompt, str(answer),
    )
    full_input_ids = torch.cat([prompt_ids, answer_ids]).to(device)
    prompt_len = int(prompt_ids.numel())
    with torch.no_grad():
        full_logits = teacher_adapter.model(full_input_ids.unsqueeze(0)).logits.squeeze(0).detach().cpu()
    teacher_dla_logits: torch.Tensor | None = None
    if prompt_len > 0 and full_logits.shape[0] >= prompt_len:
        teacher_dla_logits = full_logits[prompt_len - 1].to(device)
    logit_token_ids = teacher_result.graph.logit_token_ids.to(device)
    teacher_supergraph = teacher_result.supergraph
    del teacher_result
    return teacher_supergraph, logit_token_ids, teacher_dla_logits


def _token_path_teacher_target(
    prompt: str, answer: Any, teacher_adapter: HFLlamaGraphAdapter,
    config: "GraphAuxConfig", device: torch.device,
) -> tuple[SuperGraph, torch.Tensor, torch.Tensor | None]:
    """Teacher target for token-path: gradient x input, no attribution graph.

    Attributed at every position that predicts an answer token, the same set the
    KD term is scored on, and averaged over the prompt positions. One forward and
    a backward per answer position instead of ~1.8 s of neuron-level attribution;
    nothing here touches pre-selection, supernodes, ANOVA labels or the MLP cache.
    """
    prompt_ids, answer_ids = tokenize_prompt_answer(teacher_adapter.tokenizer, prompt, str(answer))
    full_ids = torch.cat([prompt_ids, answer_ids]).to(device)
    n_p = int(prompt_ids.numel())
    gold = int(answer_ids[0])
    with torch.no_grad():
        dla_logits = teacher_adapter.model(full_ids.unsqueeze(0)).logits[0, n_p - 1].detach()
    logit_ids, probs = salient_logits(
        dla_logits, config.top_k_logits, config.temperature,
        gold_token=gold if config.token_path_rows == "gold" else None,
    )
    reads = [[n_p - 1 + k for k in range(int(answer_ids.numel()))]]
    attn = torch.ones_like(full_ids).unsqueeze(0)
    attr = _attribute_batch(
        teacher_adapter.model, full_ids.unsqueeze(0), attn, [n_p], reads,
        [logit_ids.to(device)], [probs.to(device)], [gold],
        rows=config.token_path_rows, create_graph=False,
        autocast=_graph_autocast(teacher_adapter, config))[0].detach()
    labels = ([["token-path"]] if attr.shape[0] == 1
              else [[f"token-path:{i}"] for i in range(attr.shape[0])])
    supergraph = SuperGraph(
        supernode_adjacency_matrix=attr,
        supernodes=[list(range(attr.shape[1])) for _ in labels],
        supernode_labels=labels,
    )
    return supergraph, logit_ids.to(device), dla_logits.to(device)


def _logit_weights(
    dla_logits: torch.Tensor | None, logit_token_ids: torch.Tensor, config: "GraphAuxConfig",
) -> torch.Tensor | None:
    """The teacher's probability over its own logit targets.

    Derived from teacher_dla_logits, which both sides already hold, so the student
    is weighted by the *teacher's* probabilities rather than its own -- weighting
    each model by its own beliefs would confound the comparison with the very
    difference the loss is trying to measure. Falls back to uniform if the DLA
    reference is missing.
    """
    # Needed by rows='weighted' to combine the logit rows, and by rows='all' to
    # weight the per-row distances. Only rows='gold' has no use for them.
    if config.token_path_rows == "gold":
        return None
    n = int(logit_token_ids.numel())
    if dla_logits is None:
        return torch.full((n,), 1.0 / max(n, 1))
    probs = torch.softmax(dla_logits.float() / max(config.temperature, 1e-6), dim=-1)
    return probs[logit_token_ids.to(probs.device)].detach().cpu()


def _inner_aggregation(config: "GraphAuxConfig") -> str:
    """The aggregation create_graph should run internally.

    token-path does not aggregate supernodes, so create_graph is given a real
    aggregation whose output is discarded; the target is built from the raw graph
    afterwards. Passing "token-path" through would hit the guard in
    aggregate_supernode_adjacency.
    """
    return "normalised" if config.supergraph_aggregation == "token-path" else config.supergraph_aggregation


def _token_path_loss(
    prompt: str,
    answer: Any,
    student_adapter: HFLlamaGraphAdapter,
    config: "GraphAuxConfig",
    student_graph: Any,
    teacher_supergraph: SuperGraph,
    logit_weights: torch.Tensor | None = None,
    logit_token_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Student side of the token-path loss: gradient x input, no attribution graph.

    create_graph=True on the inner backward so the loss can differentiate through
    the student's own attribution -- this target is itself a gradient, so training
    on it needs double backward, at roughly 2-3x a normal step.

    Teacher and student index the same token positions by construction, so the
    alignment is the identity and the "aligned" counters are trivially R of R.
    """
    prompt_ids, answer_ids = tokenize_prompt_answer(student_adapter.tokenizer, prompt, str(answer))
    device = student_adapter.device
    full_ids = torch.cat([prompt_ids, answer_ids]).to(device)
    n_p = int(prompt_ids.numel())
    reads = [[n_p - 1 + k for k in range(int(answer_ids.numel()))]]
    attn = torch.ones_like(full_ids).unsqueeze(0)
    lids = logit_token_ids if logit_token_ids is not None else torch.tensor([int(answer_ids[0])])
    W_S = _attribute_batch(
        student_adapter.model, full_ids.unsqueeze(0), attn, [n_p], reads,
        [lids.to(device)], [logit_weights], [int(answer_ids[0])],
        rows=config.token_path_rows, create_graph=True,
        autocast=_graph_autocast(student_adapter, config))[0]
    W_T = teacher_supergraph.supernode_adjacency_matrix.detach().to(
        device=W_S.device, dtype=W_S.dtype)
    if W_T.shape != W_S.shape:
        raise ValueError(
            f"token-path shape mismatch for prompt={prompt!r}: teacher {tuple(W_T.shape)} vs "
            f"student {tuple(W_S.shape)}; teacher and student must tokenise identically")

    # Columns are token positions, already corresponding one-to-one, so the
    # supernode alignment is skipped entirely (it would index the first K columns
    # as a supernode block, which raises as soon as there is more than one row).
    n_rows = int(W_T.shape[0])
    real_target_loss = None
    if config.scramble_teacher_graph:
        with torch.no_grad():
            real_target_loss = edge_similarity(
                W_T, W_S.detach(), config.graph_loss_type,
                row_weights=(logit_weights if (config.token_path_rows == "all" and n_rows > 1) else None))
        W_T = scramble_teacher_rows(W_T, config)

    # rows='all' is one row per logit target, most of them low-probability tokens
    # whose attribution is mostly noise. Weight the per-row distances by the
    # teacher's probability over its logit set, applied after the distance so each
    # row is still compared as a distribution over positions.
    row_w = logit_weights if (config.token_path_rows == "all" and n_rows > 1) else None
    graph_loss = edge_similarity(W_T, W_S, config.graph_loss_type, row_weights=row_w)
    loss_breakdown = {"edge_loss": float(graph_loss.item())}
    metrics = {
        "teacher_supernodes": n_rows,
        "student_supernodes": n_rows,
        "student_graph_neurons": 0,
        "aligned_teacher_supernodes": n_rows,
        **loss_breakdown,
    }
    if real_target_loss is not None:
        metrics["edge_loss_real_target"] = float(real_target_loss.item())
    return graph_loss, metrics


def compute_prompt_graph_loss(
    *,
    prompt: str | torch.Tensor,
    student_adapter: HFLlamaGraphAdapter,
    config: GraphAuxConfig,
    teacher_supergraph: SuperGraph,
    logit_token_ids: torch.Tensor | None,
    teacher_dla_logits: torch.Tensor | None,
    answer: Any = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if config.verbose:
        print(
            f"  [graph] teacher supergraph ready: "
            f"{len(teacher_supergraph.supernodes)} supernodes"
        )

    if config.supergraph_aggregation == "token-path":
        return _token_path_loss(
            prompt, answer, student_adapter, config, None, teacher_supergraph,
            logit_weights=_logit_weights(teacher_dla_logits, logit_token_ids, config),
            logit_token_ids=logit_token_ids)

    if config.verbose:
        print(f"  [graph] building student graph for prompt: {prompt!r}")

    supergraph_start = time.perf_counter()

    try:
        student_result = create_graph(
            student_adapter,
            prompt,
            attribution_targets=logit_token_ids.cpu() if logit_token_ids is not None else None,
            prop_neurons_per_layer=config.student_prop_neurons_per_layer,
            top_k_logits=config.top_k_logits,
            temperature=config.temperature,
            batch_size=config.student_graph_batch_size,
            dtype=config.graph_dtype,
            verbose=config.verbose,
            build_create_graph=False,
            detach_result=False,
            skip_logit_attribution=False,
            mlp_input_cache=config.mlp_input_cache,
            node_labels=config.graph_node_labels or [],
            anova_range_radius=config.student_anova_range_radius,
            anova_neuron_chunk=config.anova_neuron_chunk,
            nodes_per_label=config.student_nodes_per_label,
            dla_model_logits=teacher_dla_logits,
            no_grad_supergraph=True,
            freeze_attention=config.freeze_attention,
            freeze_rms_norm=config.freeze_rms_norm,
            constant_node_weighting=config.constant_node_weighting,
            supergraph_aggregation=config.supergraph_aggregation,
            token_source_columns=config.token_source_columns,
        )
    except ValueError as e:
        raise RuntimeError(
            f"Student supergraph build failed for prompt={prompt!r}: {e}"
        ) from e

    student_graph = student_result.graph
    student_supergraph_structure = student_result.supergraph

    # Filter supernodes to only the requested labels (if specified).
    # Supernodes added via explicit flags (DLA, arg-token) are always kept regardless
    # of the ANOVA label whitelist.
    if config.graph_node_labels is not None:
        label_set = set(config.graph_node_labels)
        if config.tokens_dla_nodes:
            label_set.add("dla")
        keep_indices = [
            i
            for i, labels in enumerate(student_supergraph_structure.supernode_labels or [])
            if labels and (
                labels[0] in label_set
                or (config.tokens_dla_nodes and labels[0].startswith("arg:"))
            )
        ]
        student_supergraph_structure = student_supergraph_structure._replace(
            supernodes=[student_supergraph_structure.supernodes[i] for i in keep_indices],
            supernode_labels=[student_supergraph_structure.supernode_labels[i] for i in keep_indices],
        )

    for i, members in enumerate(student_supergraph_structure.supernodes):
        if not members:
            label = (
                (student_supergraph_structure.supernode_labels or [])[i]
                if i < len(student_supergraph_structure.supernode_labels or [])
                else "unknown"
            )
            raise RuntimeError(
                f"Student supernode {i} (label={label!r}) has no member nodes "
                f"for prompt={prompt!r}."
            )

    student_supergraph = _aggregate_supergraph_adjacency(
        student_graph,
        student_supergraph_structure.supernodes,
        constant_node_weighting=config.constant_node_weighting,
        aggregation=config.supergraph_aggregation,
        token_source_columns=config.token_source_columns,
    )
    student_supergraph = student_supergraph._replace(
        supernode_labels=student_supergraph_structure.supernode_labels,
    )

    if config.verbose:
        print(
            "  [graph] student supergraph complete: "
            f"{len(student_supergraph.supernodes)} supernodes in "
            f"{time.perf_counter() - supergraph_start:.2f}s",
        )

    # ------------------------------------------------------------------
    # Alignment: match teacher and student supernodes by label (exact)
    # ------------------------------------------------------------------
    if config.verbose:
        print("  [graph] aligning supernodes by label")
    s_label_to_sid = {
        labels[0]: sid
        for sid, labels in enumerate(student_supergraph.supernode_labels or [])
        if labels
    }
    t_label_to_tid = {
        labels[0]: tid
        for tid, labels in enumerate(teacher_supergraph.supernode_labels or [])
        if labels
    }

    # Require an exact match between teacher and student supernode label sets.
    # Extra or missing supernodes on either side indicate a cache/flag mismatch.
    student_label_set = set(s_label_to_sid.keys())
    teacher_label_set = set(t_label_to_tid.keys())
    extra_in_teacher = teacher_label_set - student_label_set
    missing_from_teacher = student_label_set - teacher_label_set
    if extra_in_teacher or missing_from_teacher:
        parts = []
        if extra_in_teacher:
            parts.append(f"  teacher has unexpected extra supernodes: {sorted(extra_in_teacher)}")
        if missing_from_teacher:
            parts.append(f"  teacher is missing expected supernodes:  {sorted(missing_from_teacher)}")
        logger.warning(
            "Teacher/student supernode label mismatch for prompt=%r — mismatched nodes excluded.\n%s\n  Student labels: %s\n  Teacher labels: %s",
            prompt,
            "\n".join(parts),
            sorted(student_label_set),
            sorted(teacher_label_set),
        )
        # Restrict both sides to the intersection so the loss is computed
        # only over commonly-labeled supernodes.
        common_labels = student_label_set & teacher_label_set
        s_label_to_sid = {lbl: sid for lbl, sid in s_label_to_sid.items() if lbl in common_labels}
        t_label_to_tid = {lbl: tid for lbl, tid in t_label_to_tid.items() if lbl in common_labels}

    mapping = {
        tid: {s_label_to_sid[labels[0]]}
        for tid, labels in enumerate(teacher_supergraph.supernode_labels or [])
        if labels and labels[0] in s_label_to_sid
    }

    teacher_ids = list(range(len(teacher_supergraph.supernodes)))
    student_ids = list(range(len(student_supergraph.supernodes)))

    W_S = student_supergraph.supernode_adjacency_matrix
    W_T = teacher_supergraph.supernode_adjacency_matrix.detach().to(
        device=W_S.device, dtype=W_S.dtype,
    )
    real_target_loss: torch.Tensor | None = None
    if config.scramble_teacher_graph:
        # Keep measuring the loss against the real target (no gradient) so a
        # scrambled run shows whether the student drifts from the teacher's routing.
        with torch.no_grad():
            real_target_loss, _ = compute_graph_loss(
                W_T, W_S.detach(), mapping, teacher_ids, student_ids,
                similarity=config.graph_loss_type,
            )
        W_T = scramble_teacher_rows(W_T, config)

    graph_loss, loss_breakdown = compute_graph_loss(
        W_T,
        W_S,
        mapping,
        teacher_ids,
        student_ids,
        similarity=config.graph_loss_type,
    )

    metrics = {
        "teacher_supernodes": len(teacher_ids),
        "student_supernodes": len(student_ids),
        "student_graph_neurons": int(student_graph.n_neurons),
        "aligned_teacher_supernodes": sum(1 for tid in teacher_ids if mapping.get(tid)),
        **loss_breakdown,
    }
    if real_target_loss is not None:
        metrics["edge_loss_real_target"] = float(real_target_loss.item())

    return graph_loss, metrics


def _kl_per_position(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    vocab = min(teacher_logits.shape[-1], student_logits.shape[-1])
    t_probs = torch.softmax(teacher_logits[..., :vocab] / temperature, dim=-1)
    s_log_probs = torch.log_softmax(student_logits[..., :vocab] / temperature, dim=-1)
    return (t_probs * (t_probs.clamp(min=1e-10).log() - s_log_probs)).sum(dim=-1)


def _compare_tokens_loss_for_prompt(
    *,
    prompt: str,
    answer: int,
    teacher_adapter: HFLlamaGraphAdapter,
    student_adapter: HFLlamaGraphAdapter,
    config: GraphAuxConfig,
    device: torch.device,
    loss_scale: float,
    denom: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Graph loss for one prompt using KL-selected response token positions.

    Tokenizes the full sequence (prompt + answer), selects the top
    compare_n_tokens response positions by teacher-student KL divergence,
    then builds separate teacher and student supergraphs for the causal
    prefix at each selected position. Backprops immediately after each
    position to bound peak memory to one position's graphs at a time.
    Returns a detached mean loss for logging; backward is already complete.
    """
    import gc

    tokenizer = student_adapter.tokenizer
    # Same ids as the KD batch row for this prompt (BOS + prompt, answer + EOS), so
    # the DLA reference logits below come from the same forward the graph is built on.
    prompt_ids, answer_ids = tokenize_prompt_answer(tokenizer, prompt, str(answer))
    input_ids = torch.cat([prompt_ids, answer_ids]).to(device)
    response_start = int(prompt_ids.numel())
    response_end = int(input_ids.numel())

    response_positions = [
        pos for pos in range(response_start, response_end)
    ]
    if not response_positions:
        raise RuntimeError(
            f"compare_n_tokens: no non-EOS response tokens for prompt={prompt!r}"
        )

    # Teacher forward always needed: DLA logits at selected positions use t_logits[pos-1].
    with torch.no_grad():
        t_logits = teacher_adapter.model(input_ids.unsqueeze(0)).logits.squeeze(0).cpu()

    logit_positions = [pos - 1 for pos in response_positions]
    if config.compare_ans_token:
        extract_fn = lambda text: parse_response(text, config.dataset_name)
        # Include the last prompt token so "= <answer>" patterns are visible.
        context_start = max(0, response_start - 1)
        full_resp = tokenizer.decode(
            input_ids[context_start:response_end].tolist(), skip_special_tokens=True
        )
        answer_val = extract_fn(full_resp)
        selected_positions = None
        if answer_val is not None:
            response_ids = input_ids[response_start:response_end].tolist()
            cand_ids = tokenizer(str(answer_val), add_special_tokens=False)["input_ids"]
            if hasattr(cand_ids, 'tolist'):
                cand_ids = cand_ids.tolist()
            n_ans = len(cand_ids)
            for i in range(len(response_ids) - n_ans, -1, -1):
                if response_ids[i:i + n_ans] == cand_ids:
                    selected_positions = [response_positions[i + j] for j in range(n_ans)]
                    break
        if selected_positions is None:
            selected_positions = [response_positions[-1]]
        n_select = len(selected_positions)
    else:
        with torch.no_grad():
            s_logits = student_adapter.model(input_ids.unsqueeze(0)).logits.squeeze(0).detach().cpu()
        n_select = min(config.compare_n_tokens, len(response_positions))
        kl_vals = _kl_per_position(
            t_logits[logit_positions], s_logits[logit_positions], config.temperature
        )
        selected_positions = [
            response_positions[i] for i in torch.topk(kl_vals, n_select).indices.tolist()
        ]
    if config.verbose:
        print(f"      [graph] {len(input_ids)} tokens, {n_select} positions selected", flush=True)

    detached_losses: list[torch.Tensor] = []
    metric_sums: dict[str, float] = {}

    for graph_idx, pos in enumerate(selected_positions):
        prefix_ids = input_ids[:pos].cpu()

        with torch.enable_grad():
            teacher_result = create_graph(
                teacher_adapter,
                prefix_ids,
                prop_neurons_per_layer=config.teacher_prop_neurons_per_layer,
                top_k_logits=config.top_k_logits,
                temperature=config.temperature,
                batch_size=config.teacher_graph_batch_size,
                dtype=config.graph_dtype,
                nodes_per_label=config.teacher_nodes_per_label,
                no_grad_supergraph=True,
                build_create_graph=False,
                detach_result=True,
                verbose=config.verbose,
                freeze_attention=config.freeze_attention,
                freeze_rms_norm=config.freeze_rms_norm,
                constant_node_weighting=config.constant_node_weighting,
            supergraph_aggregation=config.supergraph_aggregation,
            token_source_columns=config.token_source_columns,
            )

        # Pass prefix_ids directly so the student tokenizes from the same IDs
        # as the teacher — avoids decode→re-tokenize round-trip instability.
        pos_loss, pos_metrics = compute_prompt_graph_loss(
            prompt=prefix_ids,
            student_adapter=student_adapter,
            config=config,
            teacher_supergraph=teacher_result.supergraph,
            logit_token_ids=teacher_result.graph.logit_token_ids.to(device),
            teacher_dla_logits=t_logits[pos - 1].to(device),
        )
        del teacher_result
        if config.verbose:
            token_str = tokenizer.decode([input_ids[pos].item()])
            print(f"      [graph] graph {graph_idx + 1}/{n_select} built (token {pos}: {token_str!r})", flush=True)

        scaled_pos_loss = (loss_scale / denom / n_select) * pos_loss
        if scaled_pos_loss.requires_grad:
            scaled_pos_loss.backward()
        detached_losses.append(pos_loss.detach())
        del scaled_pos_loss, pos_loss
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for key, val in pos_metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(val)

    avg_loss = torch.stack(detached_losses).mean()
    n_pos = float(len(detached_losses))
    metrics = {k: v / n_pos for k, v in metric_sums.items()}
    metrics["compare_tokens_n_selected"] = n_pos
    return avg_loss, metrics


def _token_path_batched(
    *,
    prompts: list[str],
    answers: list[Any],
    student_adapter: HFLlamaGraphAdapter,
    teacher_adapter: HFLlamaGraphAdapter,
    config: "GraphAuxConfig",
    device: torch.device,
    loss_scale: float,
    on_prompt_done: Callable[[int, int], None] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """token-path over a whole batch in one forward/backward per model.

    Attribution at batch size 1 is almost entirely kernel-launch overhead on a
    short arithmetic prompt, which the per-prompt loop paid 32 times a step. The
    batched version works because each example's target depends only on its own
    embedding row, so one backward of the *summed* targets yields
    ``d target_b / d e_b`` in row b -- no cross-talk to undo.

    Prompts are grouped by tokenised length so nothing has to be padded or
    masked; within one arithmetic dataset that is a single group. Only the
    one-row modes come here: rows='all' has a per-prompt row count, so its
    targets are not stackable and it stays on the per-prompt path.
    """
    rows = config.token_path_rows
    metric_sums: dict[str, float] = {}
    denom = float(len(prompts))
    detached: list[torch.Tensor] = []
    time_teacher = 0.0
    time_student = 0.0

    tok = student_adapter.tokenizer
    pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", 0) or 0
    seqs, prompt_lens, reads, golds = [], [], [], []
    for i, prompt in enumerate(prompts):
        p_ids, a_ids = tokenize_prompt_answer(tok, prompt, str(answers[i]))
        seqs.append(torch.cat([p_ids, a_ids]).to(device))
        n_p = int(p_ids.numel())
        prompt_lens.append(n_p)
        golds.append(int(a_ids[0]))
        # Positions whose logits predict an answer token or the EOS after it --
        # the same set kd_position_mask scores, so the two terms supervise the
        # same places rather than the graph term seeing only the first.
        reads.append([n_p - 1 + k for k in range(int(a_ids.numel()))])

    # ---- process in micro-batches: one padded forward per model per chunk, and
    # the student's loss backwarded per chunk so activations are freed before the
    # next. A 444-token batch of 32 through the 8B keeps ~36 GB of MLP
    # intermediates alive for the backward; at chunk 4 that is ~4.5 GB. Arithmetic
    # never hit this because its sequences were 9 tokens.
    #
    # Chunks are cut by padded tokens as well as by prompt count. The student's
    # double-backward graph scales with prompts x padded length, so a fixed
    # prompt count gave a peak that swung with each batch's longest prompt
    # (66-76 GB across the first nine HotpotQA steps, then over 80). Sorting
    # longest-first also means every chunk pads to a member of its own length
    # class rather than to the batch maximum. The loss is a sum over prompts
    # with one denominator, so the grouping changes nothing about it.
    chunks = _token_path_chunks(
        [int(x.numel()) for x in seqs],
        max_prompts=int(getattr(config, "token_path_micro_batch", 4) or 4),
        max_tokens=getattr(config, "token_path_micro_tokens", None))
    n_done = 0
    for idxs in chunks:
        sub = [seqs[i] for i in idxs]
        L = max(int(x.numel()) for x in sub)
        ids = torch.full((len(sub), L), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(sub), L), dtype=torch.long, device=device)
        for r, x in enumerate(sub):
            ids[r, : x.numel()] = x
            attn[r, : x.numel()] = 1
        p_lens = [prompt_lens[i] for i in idxs]
        rds = [reads[i] for i in idxs]
        gds = [golds[i] for i in idxs]

        t0 = time.perf_counter()
        with torch.no_grad():
            dla_all = teacher_adapter.model(ids, attention_mask=attn).logits.detach()
        lids_c, w_c = [], []
        for r in range(len(idxs)):
            dla = dla_all[r, p_lens[r] - 1].float()
            lids, probs = salient_logits(
                dla, config.top_k_logits, config.temperature,
                gold_token=gds[r] if rows == "gold" else None)
            lids_c.append(lids.to(device))
            w_c.append(probs.to(device))
        del dla_all
        t_profiles = _position_profiles(
            teacher_adapter.model, ids, attn, p_lens, rds, lids_c, w_c, gds, rows=rows,
            autocast=_graph_autocast(teacher_adapter, config))
        time_teacher += time.perf_counter() - t0

        t0 = time.perf_counter()
        losses, reals = _student_position_losses(
            student_adapter.model, ids, attn, p_lens, rds, lids_c, w_c, gds, t_profiles,
            rows=rows, config=config, scale=loss_scale / denom,
            autocast=_graph_autocast(student_adapter, config))
        for r in range(len(idxs)):
            detached.append(torch.tensor(losses[r]))
            metric_sums["edge_loss"] = metric_sums.get("edge_loss", 0.0) + losses[r]
            if config.scramble_teacher_graph:
                metric_sums["edge_loss_real_target"] = (
                    metric_sums.get("edge_loss_real_target", 0.0) + reals[r])
        del t_profiles, ids, attn
        time_student += time.perf_counter() - t0
        n_done += len(idxs)
        if on_prompt_done is not None:
            on_prompt_done(n_done, len(prompts))

    for key, val in (("teacher_supernodes", float(len(prompts))),
                     ("student_supernodes", float(len(prompts))),
                     ("aligned_teacher_supernodes", float(len(prompts))),
                     ("student_graph_neurons", 0.0)):
        metric_sums[key] = metric_sums.get(key, 0.0) + val

    mean_loss = torch.stack(detached).mean() if detached else torch.tensor(0.0, device=device)
    metrics = {k: v for k, v in metric_sums.items()}
    metrics.update({
        "graph_prompts": float(len(prompts)),
        "teacher_cache_hits": 0.0,
        "graph_time_teacher": time_teacher,
        "graph_time_student": time_student,
    })
    if "edge_loss_real_target" in metrics:
        metrics["edge_loss_real_target"] /= max(len(prompts), 1)
    return mean_loss, metrics


def _graph_autocast(adapter: HFLlamaGraphAdapter, config: "GraphAuxConfig"):
    """Autocast for an attribution forward, from the run's ``graph_dtype``.

    The trainer holds fp32 master weights and runs the KD forward under bf16
    autocast. The attribution forwards did not, so with fp32 weights they ran
    fp32 end to end: the student's token-path term went from 26 s to 97 s a
    step and its double-backward graph doubled, which is what took the
    HotpotQA run from 76 GB at step 1 to OOM at step 2. Under autocast the
    matmuls are bf16 and the residual stream stays fp32 -- the same regime
    the neuron-graph paths already use through ``graph_dtype``. A no-op for
    a bf16 model and off CUDA.
    """
    return adapter.autocast_context(getattr(config, "graph_dtype", None))


def _token_path_chunks(
    lengths: list[int], *, max_prompts: int, max_tokens: int | None,
) -> list[list[int]]:
    """Index groups for the token-path micro-batches, longest sequence first.

    A group holds at most ``max_prompts`` sequences and, when ``max_tokens`` is
    set, at most that many padded tokens (group size x its longest member). A
    single sequence longer than the budget still forms a group of one. Every
    index appears exactly once.
    """
    max_prompts = max(int(max_prompts or 1), 1)
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    chunks: list[list[int]] = []
    cur: list[int] = []
    for i in order:
        if cur:
            longest = lengths[cur[0]]  # longest-first, so the first member sets the pad
            full = len(cur) >= max_prompts or (
                max_tokens is not None and (len(cur) + 1) * longest > max_tokens)
            if full:
                chunks.append(cur)
                cur = []
        cur.append(i)
    if cur:
        chunks.append(cur)
    return chunks


@contextlib.contextmanager
def _no_gradient_checkpointing(model):
    """Turn gradient checkpointing off for a double-backward region.

    Checkpointing trades memory for a recomputed forward. Under
    ``create_graph=True`` that recomputed forward is *also* retained, because the
    second derivative has to differentiate through it, so it saves nothing and
    adds the recompute on top -- the OOM traceback lands in
    ``checkpoint.recompute_fn`` for exactly this reason. Restored on the way out
    so the KD term keeps it.
    """
    was_on = bool(getattr(model, "is_gradient_checkpointing", False))
    if was_on and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    try:
        yield
    finally:
        if was_on and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()


def _position_profiles(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: list[int],
    read_positions: list[list[int]],
    logit_ids: list[torch.Tensor],
    logit_weights: list[torch.Tensor | None],
    gold_tokens: list[int],
    *,
    rows: str,
    autocast: Any = None,
    epsilon: float = 1e-10,
) -> list[list[torch.Tensor]]:
    """``[b][step] -> [R, T_prompt]`` profiles, no graph retained.

    The teacher side: one forward, a backward per answer position, each freed
    immediately since no second derivative is needed.
    """
    embed = model.get_input_embeddings()
    e = embed(input_ids).detach().clone().requires_grad_(True)
    with (autocast if autocast is not None else contextlib.nullcontext()):
        out_all = model(inputs_embeds=e, attention_mask=attention_mask).logits
    B = len(prompt_lens)
    n_rows = int(logit_ids[0].numel()) if rows == "all" else 1
    profiles: list[list[torch.Tensor]] = [[] for _ in range(B)]
    max_steps = max(len(r) for r in read_positions)
    for step in range(max_steps):
        rows_here: list[list[torch.Tensor]] = [[] for _ in range(B)]
        for j in range(n_rows):
            terms, members = [], []
            for b in range(B):
                if step >= len(read_positions[b]):
                    continue
                t = _position_target(out_all[b, read_positions[b][step]].float(), rows, j,
                                     logit_ids[b], logit_weights[b], gold_tokens[b], epsilon)
                if t is None:
                    continue
                terms.append(t)
                members.append(b)
            if not terms:
                continue
            g = torch.autograd.grad(torch.stack(terms).sum(), e, retain_graph=True)[0]
            attr = (g * e).sum(dim=-1).abs()
            for b in members:
                p = attr[b, : prompt_lens[b]]
                rows_here[b].append((p / p.sum().clamp(min=epsilon)).detach())
            del g, attr, terms
        for b in range(B):
            if rows_here[b]:
                profiles[b].append(torch.stack(rows_here[b]))
    del out_all, e
    return profiles


def _position_target(
    logits: torch.Tensor, rows: str, j: int, ids: torch.Tensor,
    weights: torch.Tensor | None, gold: int, epsilon: float,
) -> torch.Tensor | None:
    """The scalar attributed at one position, for one row of the target."""
    if rows == "gold":
        return logits[int(gold)]
    ids = ids.to(logits.device).reshape(-1)
    if rows == "all":
        return logits[int(ids[j])] if j < ids.numel() else None
    w = weights
    w = (torch.full((ids.numel(),), 1.0 / max(ids.numel(), 1), device=logits.device)
         if w is None else w.to(device=logits.device, dtype=logits.dtype).reshape(-1))
    return ((w / w.sum().clamp(min=epsilon)) * logits[ids]).sum()


def _student_position_losses(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: list[int],
    read_positions: list[list[int]],
    logit_ids: list[torch.Tensor],
    logit_weights: list[torch.Tensor | None],
    gold_tokens: list[int],
    teacher_profiles: list[list[torch.Tensor]],
    *,
    rows: str,
    config: "GraphAuxConfig",
    scale: float,
    autocast: Any = None,
    epsilon: float = 1e-10,
) -> tuple[list[float], list[float]]:
    """Per-position distance for each example, backwarded position by position.

    The loss is the mean over answer positions of the per-position distance,
    which is what the KD term does with its per-position KL. Scoring the distance
    between the *averaged* profiles instead would force every position's gradient
    graph to stay alive until one final backward; with six-token answers and a
    444-token context that is what filled 80 GB. Here each position's graph is
    released as soon as its own backward runs.

    Memory: ``retain_graph=True`` on the per-position backward is required, since
    every position's attribution runs through the one forward graph. It also
    means the backward frees nothing, so the double-backward graph built for a
    position (the dense grad-of-logits and every backward intermediate, several
    GB at a 444-token context) lives until the last Python reference to it goes.
    Each position's graph is therefore built and released inside
    ``_one_position_loss`` rather than in this frame, where the previous
    position's ``g``, ``attr`` and loss would otherwise stay bound until the
    next one overwrote them and the peak held two positions instead of one.

    Returns (mean loss per example, mean real-target loss per example).
    """
    embed = model.get_input_embeddings()
    e = embed(input_ids).detach().clone().requires_grad_(True)
    B = len(prompt_lens)
    n_rows = int(logit_ids[0].numel()) if rows == "all" else 1
    losses = [0.0] * B
    reals = [0.0] * B
    counts = [0] * B
    max_steps = max(len(r) for r in read_positions)

    with _no_gradient_checkpointing(model), _double_backward_attention():
        # Autocast covers the forward only; the inner and outer backwards run in
        # the dtypes the forward recorded, exactly as the KD term's do.
        with (autocast if autocast is not None else contextlib.nullcontext()):
            out_all = model(inputs_embeds=e, attention_mask=attention_mask).logits
        for step in range(max_steps):
            _one_position_loss(
                out_all, e, step, B, n_rows, prompt_lens, read_positions, logit_ids,
                logit_weights, gold_tokens, teacher_profiles, losses, reals, counts,
                rows=rows, config=config, scale=scale, epsilon=epsilon)
    del out_all, e
    return ([losses[b] / max(counts[b], 1) for b in range(B)],
            [reals[b] / max(counts[b], 1) for b in range(B)])


def _one_position_loss(
    out_all: torch.Tensor,
    e: torch.Tensor,
    step: int,
    B: int,
    n_rows: int,
    prompt_lens: list[int],
    read_positions: list[list[int]],
    logit_ids: list[torch.Tensor],
    logit_weights: list[torch.Tensor | None],
    gold_tokens: list[int],
    teacher_profiles: list[list[torch.Tensor]],
    losses: list[float],
    reals: list[float],
    counts: list[int],
    *,
    rows: str,
    config: "GraphAuxConfig",
    scale: float,
    epsilon: float,
) -> None:
    """Attribute one answer position for the whole micro-batch and backward it.

    Everything this builds with ``create_graph=True`` is local to this frame,
    so the graph is released on return; only the forward graph behind
    ``out_all`` survives to the next position.
    """
    if rows == "all" and n_rows > 1:
        _one_position_loss_rows_all(
            out_all, e, step, B, n_rows, prompt_lens, read_positions, logit_ids,
            logit_weights, gold_tokens, teacher_profiles, losses, reals, counts,
            config=config, scale=scale, epsilon=epsilon)
        return
    rows_here: list[list[torch.Tensor]] = [[] for _ in range(B)]
    for j in range(n_rows):
        terms, members = [], []
        for b in range(B):
            if step >= len(read_positions[b]):
                continue
            t = _position_target(out_all[b, read_positions[b][step]].float(), rows, j,
                                 logit_ids[b], logit_weights[b], gold_tokens[b], epsilon)
            if t is None:
                continue
            terms.append(t)
            members.append(b)
        if not terms:
            continue
        g = torch.autograd.grad(
            torch.stack(terms).sum(), e, create_graph=True, retain_graph=True)[0]
        attr = (g * e).sum(dim=-1).abs()
        for b in members:
            pr = attr[b, : prompt_lens[b]]
            rows_here[b].append(pr / pr.sum().clamp(min=epsilon))
        del g, attr, terms
    step_loss = None
    for b in range(B):
        if not rows_here[b] or step >= len(teacher_profiles[b]):
            continue
        W_S = torch.stack(rows_here[b])
        W_T = teacher_profiles[b][step].to(device=W_S.device, dtype=W_S.dtype)
        rw = logit_weights[b] if (rows == "all" and W_T.shape[0] > 1) else None
        topk = int(getattr(config, "token_path_top_tokens", 0) or 0)
        if config.scramble_teacher_graph:
            with torch.no_grad():
                t_r, s_r = _top_token_view(W_T, W_S.detach(), topk)
                reals[b] += float(edge_similarity(
                    t_r, s_r, config.graph_loss_type, row_weights=rw).item())
            # Scrambled at full width, then reduced: the control has to move which
            # positions the teacher points at before the top-k is taken, or it
            # would only shuffle within the set the real target already chose.
            W_T = scramble_teacher_rows(W_T, config)
        W_T, W_S = _top_token_view(W_T, W_S, topk)
        d = edge_similarity(W_T, W_S, config.graph_loss_type, row_weights=rw)
        losses[b] += float(d.detach().item())
        counts[b] += 1
        contrib = d * (scale / max(len(read_positions[b]), 1))
        step_loss = contrib if step_loss is None else step_loss + contrib
        del W_S, W_T, d, contrib
    if step_loss is not None and step_loss.requires_grad:
        step_loss.backward(retain_graph=True)
    del rows_here, step_loss


def _one_position_loss_rows_all(
    out_all: torch.Tensor,
    e: torch.Tensor,
    step: int,
    B: int,
    n_rows: int,
    prompt_lens: list[int],
    read_positions: list[list[int]],
    logit_ids: list[torch.Tensor],
    logit_weights: list[torch.Tensor | None],
    gold_tokens: list[int],
    teacher_profiles: list[list[torch.Tensor]],
    losses: list[float],
    reals: list[float],
    counts: list[int],
    *,
    config: "GraphAuxConfig",
    scale: float,
    epsilon: float,
) -> None:
    """rows='all', one logit row at a time: score and backward before the next.

    The stacked path keeps every row's normalised profile in ``rows_here`` until
    the position's single backward, and each of those profiles holds its own
    ``create_graph=True`` graph alive. With one row that is the peak the
    'weighted' runs measured (55 GB at a 444-token context); with the up-to-11
    rows ``salient_logits`` returns it is that peak times the row count, which is
    the 79 GB OOM inside the first position of the first step.

    Splitting is exact rather than an approximation: the rows='all' loss is
    ``sum_j w_j * dist(W_T[j], W_S[j])`` (see ``_weighted_row_similarity``, whose
    every branch reduces over ``dim=1`` before the weighted sum), so each row's
    term can be backwarded on its own and the gradients accumulate to the same
    total. ``w`` is normalised over the full row set once, outside the loop, so
    the weights match the stacked path exactly. ``scramble_teacher_rows`` permutes
    within each row independently, so it too is taken a row at a time -- built
    from the full matrix so the cached permutation keys on the same shape.
    """
    weights: list[torch.Tensor | None] = []
    for b in range(B):
        w = logit_weights[b]
        if w is None:
            weights.append(None)
            continue
        w = w.to(device=e.device, dtype=torch.float32).reshape(-1)
        weights.append(w / w.sum().clamp(min=epsilon))
    scored = [False] * B

    for j in range(n_rows):
        terms, members = [], []
        for b in range(B):
            if step >= len(read_positions[b]) or step >= len(teacher_profiles[b]):
                continue
            t = _position_target(out_all[b, read_positions[b][step]].float(), "all", j,
                                 logit_ids[b], logit_weights[b], gold_tokens[b], epsilon)
            if t is None:
                continue
            terms.append(t)
            members.append(b)
        if not terms:
            continue
        g = torch.autograd.grad(
            torch.stack(terms).sum(), e, create_graph=True, retain_graph=True)[0]
        attr = (g * e).sum(dim=-1).abs()
        row_loss = None
        for b in members:
            W_T_all = teacher_profiles[b][step]
            if j >= int(W_T_all.shape[0]):
                continue
            pr = attr[b, : prompt_lens[b]]
            s_row = (pr / pr.sum().clamp(min=epsilon)).unsqueeze(0)
            W_T_all = W_T_all.to(device=s_row.device, dtype=s_row.dtype)
            wj = 1.0 if weights[b] is None else float(weights[b][j])
            t_row = W_T_all[j : j + 1]
            topk = int(getattr(config, "token_path_top_tokens", 0) or 0)
            if config.scramble_teacher_graph:
                with torch.no_grad():
                    t_r, s_r = _top_token_view(t_row, s_row.detach(), topk)
                    reals[b] += wj * float(edge_similarity(
                        t_r, s_r, config.graph_loss_type).item())
                t_row = scramble_teacher_rows(W_T_all, config)[j : j + 1]
            t_row, s_row = _top_token_view(t_row, s_row, topk)
            d = edge_similarity(t_row, s_row, config.graph_loss_type)
            losses[b] += wj * float(d.detach().item())
            scored[b] = True
            contrib = d * (wj * scale / max(len(read_positions[b]), 1))
            row_loss = contrib if row_loss is None else row_loss + contrib
            del s_row, t_row, W_T_all, d, contrib
        del g, attr, terms
        if row_loss is not None and row_loss.requires_grad:
            row_loss.backward(retain_graph=True)
        del row_loss

    for b in range(B):
        if scored[b]:
            counts[b] += 1


def _double_backward_attention():
    """Force the math SDPA kernel, which is the only one with a double backward.

    token-path's student side differentiates through its own attribution, so the
    inner backward runs with ``create_graph=True`` and the loss backward then asks
    for a second derivative. The fused kernels do not implement one:
    ``derivative for aten::_scaled_dot_product_efficient_attention_backward is
    not implemented``. The math backend does, at the cost of materialising the
    full B x heads x L x L attention matrix, which is why this is scoped to the
    student's attribution rather than turned on globally -- the teacher's target
    needs no second derivative and keeps the fast kernel.

    A no-op off CUDA, and tolerant of either the new or the legacy API.
    """
    if not torch.cuda.is_available():
        return contextlib.nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel(SDPBackend.MATH)
    except Exception:  # noqa: BLE001 - older torch
        pass
    try:
        return torch.backends.cuda.sdp_kernel(
            enable_math=True, enable_flash=False, enable_mem_efficient=False)
    except Exception:  # noqa: BLE001 - nothing to force; let it fail loudly instead
        return contextlib.nullcontext()


def _attribute_batch(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: list[int],
    read_positions: list[list[int]],
    logit_ids: list[torch.Tensor],
    logit_weights: list[torch.Tensor | None],
    gold_tokens: list[int],
    *,
    rows: str,
    create_graph: bool,
    autocast: Any = None,
    epsilon: float = 1e-10,
) -> list[torch.Tensor]:
    """Per-example ``[R, T_prompt]`` attribution, averaged over answer positions.

    Three things this handles that the single-position version did not.

    *Every answer position.* The KD term is scored at every position that predicts
    an answer token; attributing only at the last prompt position supervised one
    of them. ``read_positions[b]`` lists all of them for example b, and the
    profiles are averaged. That needs a backward per position index rather than
    one for the batch, because averaging the per-position profiles is not the
    same as taking one profile of the summed targets.

    *Restricted to the prompt.* Attribution at a later answer position also
    covers the answer tokens before it, so the raw profiles have different
    lengths and are not comparable. Each is sliced to the prompt positions and
    renormalised, which is also the right object: token-path is about which
    *inputs* the prediction draws on, not about the answer attending to itself.

    *Ragged batches.* Arithmetic prompts all had one length, so grouping by length
    gave a single batch. QA prompts do not, so the batch is padded and masked;
    grouping by length there would have produced batches of one and lost the
    batching entirely.
    """
    embed = model.get_input_embeddings()
    e = embed(input_ids).detach().clone().requires_grad_(True)
    # The forward and every inner backward must sit inside the same kernel choice,
    # since the second derivative is taken through this graph.
    ctx = _double_backward_attention() if create_graph else contextlib.nullcontext()
    with ctx:
        # Native dtype, cast per position below: .float() on [B, L, vocab] doubles
        # a tensor that is already gigabytes and only a few rows are ever read.
        with (autocast if autocast is not None else contextlib.nullcontext()):
            out_all = model(inputs_embeds=e, attention_mask=attention_mask).logits

        B = int(input_ids.shape[0])
        n_rows = len(logit_ids[0]) if rows == "all" else 1
        sums: list[torch.Tensor | None] = [None] * B
        counts = [0] * B
        max_steps = max(len(r) for r in read_positions)

        for step in range(max_steps):
            for j in range(n_rows if rows == "all" else 1):
                terms, members = [], []
                for b in range(B):
                    if step >= len(read_positions[b]):
                        continue
                    pos = read_positions[b][step]
                    logits_b = out_all[b, pos].float()
                    if rows == "gold":
                        terms.append(logits_b[int(gold_tokens[b])])
                    elif rows == "all":
                        ids = logit_ids[b].to(logits_b.device).reshape(-1)
                        if j >= ids.numel():
                            continue
                        terms.append(logits_b[int(ids[j])])
                    else:
                        ids = logit_ids[b].to(logits_b.device).reshape(-1)
                        w = logit_weights[b]
                        w = (torch.full((ids.numel(),), 1.0 / max(ids.numel(), 1), device=logits_b.device)
                             if w is None else w.to(device=logits_b.device, dtype=logits_b.dtype).reshape(-1))
                        terms.append(((w / w.sum().clamp(min=epsilon)) * logits_b[ids]).sum())
                    members.append(b)
                if not terms:
                    continue
                last = (step == max_steps - 1) and (j == (n_rows - 1 if rows == "all" else 0))
                g = torch.autograd.grad(
                    torch.stack(terms).sum(), e,
                    create_graph=create_graph, retain_graph=create_graph or not last)[0]
                attr = (g * e).sum(dim=-1).abs()                       # [B, T_padded]
                for b in members:
                    prof = attr[b, : prompt_lens[b]]
                    prof = prof / prof.sum().clamp(min=epsilon)
                    row = prof.unsqueeze(0) if rows != "all" else None
                    if rows == "all":
                        cur = sums[b]
                        if cur is None:
                            cur = [None] * n_rows
                            sums[b] = cur
                        cur[j] = prof if cur[j] is None else cur[j] + prof
                    else:
                        sums[b] = row if sums[b] is None else sums[b] + row
                if rows != "all":
                    for b in members:
                        counts[b] += 1
                elif j == 0:
                    for b in members:
                        counts[b] += 1

    out = []
    for b in range(B):
        n = max(counts[b], 1)
        if rows == "all":
            rowsb = [(r / n) if r is not None else torch.zeros(prompt_lens[b], device=e.device)
                     for r in (sums[b] or [])]
            m = torch.stack(rowsb) if rowsb else torch.zeros(1, prompt_lens[b], device=e.device)
        else:
            m = (sums[b] / n) if sums[b] is not None else torch.zeros(1, prompt_lens[b], device=e.device)
        out.append(m / m.sum(dim=1, keepdim=True).clamp(min=epsilon))
    return out


def backward_batch_graph_loss(
    *,
    prompts: list[str],
    student_adapter: HFLlamaGraphAdapter,
    config: GraphAuxConfig,
    device: torch.device,
    loss_scale: float,
    teacher_adapter: HFLlamaGraphAdapter,
    answers: list[int],
    on_prompt_done: Callable[[int, int], None] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute and backprop graph loss one prompt at a time.

    Processes each prompt's attribution graph immediately and backprops before
    building the next, keeping peak memory bounded to a single prompt.
    """
    if not prompts:
        return torch.tensor(0.0, device=device), {}

    if config.supergraph_aggregation == "token-path" and config.compare_n_tokens is None:
        return _token_path_batched(
            prompts=prompts, answers=answers, student_adapter=student_adapter,
            teacher_adapter=teacher_adapter, config=config, device=device,
            loss_scale=loss_scale, on_prompt_done=on_prompt_done,
        )

    import gc

    metric_sums: dict[str, float] = {}
    detached_losses = []
    denom = float(len(prompts))
    graph_backward_prompts = 0
    time_teacher = 0.0
    time_student = 0.0
    cache_hits = 0

    for i, prompt in enumerate(prompts):
        if config.compare_n_tokens is not None:
            # Backward is done per-position inside _compare_tokens_loss_for_prompt.
            prompt_loss, prompt_metrics = _compare_tokens_loss_for_prompt(
                prompt=prompt,
                answer=answers[i],
                teacher_adapter=teacher_adapter,  # type: ignore[arg-type]
                student_adapter=student_adapter,
                config=config,
                device=device,
                loss_scale=loss_scale,
                denom=denom,
            )
            detached_losses.append(prompt_loss)  # already detached
            graph_backward_prompts += 1
        else:
            t_start = time.perf_counter()
            cache = config.teacher_target_cache
            entry = cache.get(prompt) if cache is not None else None
            if entry is None:
                teacher_supergraph, logit_token_ids, teacher_dla_logits = _compute_teacher_target(
                    prompt, answers[i], teacher_adapter, config, device,
                )
                if cache is not None:
                    cache.put(prompt, _teacher_target_entry(
                        teacher_supergraph, logit_token_ids, teacher_dla_logits, answer=answers[i],
                    ))
            else:
                teacher_supergraph, logit_token_ids, teacher_dla_logits = _teacher_target_from_entry(entry, device)
                cache_hits += 1
            time_teacher += time.perf_counter() - t_start
            t_start = time.perf_counter()
            prompt_loss, prompt_metrics = compute_prompt_graph_loss(
                prompt=prompt,
                student_adapter=student_adapter,
                config=config,
                teacher_supergraph=teacher_supergraph,
                logit_token_ids=logit_token_ids,
                teacher_dla_logits=teacher_dla_logits,
                answer=answers[i],
            )
            detached_losses.append(prompt_loss.detach())
            scaled_loss = (loss_scale / denom) * prompt_loss
            if scaled_loss.requires_grad:
                scaled_loss.backward()
                graph_backward_prompts += 1
            elif config.verbose:
                print("  [graph] WARN: graph loss has no grad; skipping backward")
            del scaled_loss
            time_student += time.perf_counter() - t_start
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if on_prompt_done is not None:
            on_prompt_done(i + 1, len(prompts))
        for key, value in prompt_metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)

    loss = torch.stack(detached_losses).mean()
    metrics = {key: value / denom for key, value in metric_sums.items()}
    metrics["graph_prompts"] = float(len(prompts))
    metrics["graph_backward_prompts"] = float(graph_backward_prompts)
    # Wall-clock totals (not per-prompt means) and cache hits for this batch.
    metrics["graph_time_teacher"] = time_teacher
    metrics["graph_time_student"] = time_student
    metrics["teacher_cache_hits"] = float(cache_hits)
    return loss, metrics

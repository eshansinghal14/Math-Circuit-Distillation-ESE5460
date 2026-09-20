"""Training-time graph auxiliary loss helpers."""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
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
    salient_targets_with_gold,
    token_path_supergraph,
)
from graph_loss.hf_adapter import HFLlamaGraphAdapter
from graph_loss.loss import compute_graph_loss
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


def _derangements(k: int, rng: random.Random) -> list[list[int]]:
    """One random derangement of ``range(k)`` per row index.

    A derangement has no fixed point, so no entry of a scrambled row stays at
    its true source. ``k < 2`` has no derangement and gets the identity.
    """
    perms: list[list[int]] = []
    for _ in range(k):
        if k < 2:
            perms.append(list(range(k)))
            continue
        while True:
            p = list(range(k))
            rng.shuffle(p)
            if all(p[i] != i for i in range(k)):
                perms.append(p)
                break
    return perms


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
    k = int(W_T.shape[1])
    perms = config.scramble_permutations.get(k)
    if perms is None:
        perms = _derangements(k, random.Random(config.scramble_seed * 1_000_003 + k))
        config.scramble_permutations[k] = perms
        print(f"  [graph] scrambled teacher target: width={k}, row permutations {perms}")
    idx = torch.tensor(perms[: W_T.shape[0]], device=W_T.device, dtype=torch.long)
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
    needs_gold = token_path and config.token_path_rows == "gold"
    forced_targets = None
    if needs_gold:
        # rows="gold" attributes to the gold logit, so it has to be among the
        # attribution targets. The 95% salient set usually contains it but is not
        # guaranteed to, and a miss would raise mid-run, so force it in the way
        # inspect_graphs does. Nothing else about the graph changes.
        _p_ids, _a_ids = tokenize_prompt_answer(teacher_adapter.tokenizer, prompt, str(answer))
        _full = torch.cat([_p_ids, _a_ids]).to(device)
        with torch.no_grad():
            _dla = teacher_adapter.model(_full.unsqueeze(0)).logits[0, int(_p_ids.numel()) - 1].detach()
        forced_targets, _gold_in_salient = salient_targets_with_gold(
            _dla, int(_a_ids[0]), config.top_k_logits, config.temperature,
        )
    with torch.enable_grad():
        teacher_result = create_graph(
            teacher_adapter,
            prompt,
            attribution_targets=forced_targets,
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
    if config.supergraph_aggregation == "token-path":
        # The supernode supergraph create_graph built is discarded; the target is
        # the token-position distribution derived from the same raw graph.
        teacher_supergraph = token_path_supergraph(
            teacher_result.graph,
            int(answer_ids[0]) if config.token_path_rows == "gold" else None,
            rows=config.token_path_rows,
            logit_weights=_logit_weights(teacher_dla_logits, logit_token_ids, config),
        )
    else:
        teacher_supergraph = teacher_result.supergraph
    del teacher_result
    return teacher_supergraph, logit_token_ids, teacher_dla_logits


def _logit_weights(
    dla_logits: torch.Tensor | None, logit_token_ids: torch.Tensor, config: "GraphAuxConfig",
) -> torch.Tensor | None:
    """The teacher's probability over its own logit targets, for rows='weighted'.

    Derived from teacher_dla_logits, which both sides already hold, so the student
    is weighted by the *teacher's* probabilities rather than its own -- weighting
    each model by its own beliefs would confound the comparison with the very
    difference the loss is trying to measure. Falls back to uniform if the DLA
    reference is missing.
    """
    if config.token_path_rows != "weighted":
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
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Loss for the token-path target: one row, so no label alignment is needed.

    Teacher and student index the same token positions by construction, so the
    mapping is the identity and the "supernodes aligned" counters are trivially
    1 of 1. Everything else -- the scramble, the real-target diagnostic and every
    graph_loss_type -- works on the 1 x T matrix unchanged.
    """
    gold = None
    if config.token_path_rows == "gold":
        _, answer_ids = tokenize_prompt_answer(student_adapter.tokenizer, prompt, str(answer))
        gold = int(answer_ids[0])
    student_supergraph = token_path_supergraph(
        student_graph, gold, rows=config.token_path_rows, logit_weights=logit_weights)

    W_S = student_supergraph.supernode_adjacency_matrix
    W_T = teacher_supergraph.supernode_adjacency_matrix.detach().to(
        device=W_S.device, dtype=W_S.dtype,
    )
    if W_T.shape != W_S.shape:
        raise ValueError(
            f"token-path shape mismatch for prompt={prompt!r}: teacher {tuple(W_T.shape)} vs "
            f"student {tuple(W_S.shape)}; teacher and student must tokenise identically")

    n_rows = int(W_T.shape[0])
    mapping = {i: {i} for i in range(n_rows)}
    ids = list(range(n_rows))
    real_target_loss = None
    if config.scramble_teacher_graph:
        with torch.no_grad():
            real_target_loss, _ = compute_graph_loss(
                W_T, W_S.detach(), mapping, ids, ids, similarity=config.graph_loss_type)
        W_T = scramble_teacher_rows(W_T, config)

    graph_loss, loss_breakdown = compute_graph_loss(
        W_T, W_S, mapping, ids, ids, similarity=config.graph_loss_type)
    metrics = {
        "teacher_supernodes": n_rows,
        "student_supernodes": n_rows,
        "student_graph_neurons": int(student_graph.n_neurons),
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
    if config.supergraph_aggregation == "token-path":
        return _token_path_loss(
            prompt, answer, student_adapter, config, student_graph, teacher_supergraph,
            logit_weights=_logit_weights(teacher_dla_logits, logit_token_ids, config))
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

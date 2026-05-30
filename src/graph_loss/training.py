"""Training-time graph auxiliary loss helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from graph_loss.create_graph import create_graph
from graph_loss.graph import SuperGraph, normalize_matrix
from graph_loss.hf_adapter import HFLlamaGraphAdapter
from graph_loss.loss import compute_graph_loss
from graph_loss.teacher_data_cache import TeacherDataCache


@dataclass
class CachedTeacherPromptData:
    """Pre-computed teacher artifacts for one prompt, loaded from TeacherDataCache."""

    supergraph: SuperGraph
    logit_token_ids: torch.Tensor | None
    # Teacher logits at the last prompt-token position, used to select the student
    # DLA supernode against the teacher's output distribution rather than the
    # student's own (which is wrong early in training and shifts every step).
    teacher_dla_logits: torch.Tensor | None = None


@dataclass
class GraphAuxConfig:
    lambda_graph: float = 0.1
    graph_dtype: torch.dtype | None = None
    teacher_prop_neurons_per_layer: float = 0.1
    student_prop_neurons_per_layer: float = 0.1

    top_k_logits: float | None = 0.95
    temperature: float = 2.0
    teacher_graph_batch_size: int = 512
    student_graph_batch_size: int = 1
    verbose: bool = False

    student_anova_range_radius: int = 0
    student_anova_nodes_per_label: int = 10
    student_sum_min_specificity: float = 0.0
    student_mlp_input_cache_path: str | None = None
    mlp_input_cache: dict | None = None
    dataset: str | None = None
    student_activation_write_cache_path: str | None = None
    activation_write_result_cache: dict = field(default_factory=dict)
    graph_loss_type: Literal["jsd", "kld", "mse", "mse-norm", "mse-scale"] = "jsd"
    student_graph_labels: list[str] | None = None
    tokens_dla_nodes: bool = False
    compare_n_tokens: int | None = None
    # Teacher-specific ANOVA config.
    teacher_anova_range_radius: int = 0
    teacher_anova_nodes_per_label: int = 10
    teacher_sum_min_specificity: float = 0.0
    teacher_graph_labels: list[str] | None = None
    # Diagnostic: build teacher graph live and compare to cached version.
    debug_compare_teacher_graphs: bool = False


def _aggregate_supergraph_adjacency(graph, supernodes: list[list[int]]) -> SuperGraph:
    """Aggregate a differentiable graph adjacency using fixed supernode membership.

    Uses torch.stack (out-of-place) instead of in-place setitem so that the
    gradient from the edge loss flows back through supernode_adjacency_matrix
    → adjacency_matrix → source_vectors_t → model parameters (down_proj.weight).
    """
    adj_matrix_norm = normalize_matrix(graph.adjacency_matrix)
    num_supernodes = len(supernodes)
    if num_supernodes == 0:
        device = graph.adjacency_matrix.device
        dtype = graph.adjacency_matrix.dtype
        return SuperGraph(
            supernode_adjacency_matrix=torch.zeros((0, 0), device=device, dtype=dtype),
            supernodes=[],
        )
    rows = []
    for t in range(num_supernodes):
        total_input = torch.abs(adj_matrix_norm[:, supernodes[t]]).sum(dim=0)
        internal_input = torch.abs(adj_matrix_norm[supernodes[t]][:, supernodes[t]]).sum(dim=0)
        frac_external = (total_input - internal_input) / total_input.clamp(min=1e-10)
        row_entries = []
        for s in range(num_supernodes):
            sum_A = adj_matrix_norm[supernodes[t]][:, supernodes[s]].sum(dim=1)
            entry = (
                (frac_external * sum_A).sum(dim=0)
                / frac_external.sum(dim=0).clamp(min=1e-10)
            )
            row_entries.append(entry)
        rows.append(torch.stack(row_entries))
    supernode_adj_matrix = torch.stack(rows)
    return SuperGraph(
        supernode_adjacency_matrix=supernode_adj_matrix,
        supernodes=supernodes,
    )


def _debug_compare_teacher_graphs(
    *,
    prompt: str,
    config: GraphAuxConfig,
    cached_teacher: CachedTeacherPromptData,
    teacher_adapter: HFLlamaGraphAdapter,
    student_supergraph: "SuperGraph",
) -> None:
    """Build the teacher graph live and compare it to the cached version.

    Prints a detailed side-by-side comparison: supergraph labels, adjacency
    matrix stats, per-label row differences, and graph-loss using each teacher.
    """
    import gc

    div = "=" * 72
    print(f"\n{div}")
    print(f"[DEBUG-TEACHER-COMPARE] prompt: {prompt[:80]!r}")
    print(div)

    cached_sg = cached_teacher.supergraph
    cached_labels = [lab[0] if lab else "?" for lab in (cached_sg.supernode_labels or [])]
    cached_adj = cached_sg.supernode_adjacency_matrix.float().cpu()
    cached_n = len(cached_sg.supernodes)

    def _adj_stats(adj: torch.Tensor) -> str:
        if adj.numel() == 0:
            return "empty"
        return (
            f"min={adj.min():.4f} max={adj.max():.4f} "
            f"mean={adj.mean():.4f} frob={adj.norm():.4f}"
        )

    def _row_norms(adj: torch.Tensor) -> list[str]:
        return [f"{v:.4f}" for v in adj.abs().sum(dim=1).tolist()]

    cached_n_neurons = sum(len(sn) for sn in cached_sg.supernodes)
    print(f"\n--- CACHED TEACHER ---")
    print(f"  n_supernodes : {cached_n}")
    print(f"  n_neurons    : {cached_n_neurons}")
    print(f"  labels       : {cached_labels}")
    print(f"  adj shape    : {tuple(cached_adj.shape)}")
    print(f"  adj stats    : {_adj_stats(cached_adj)}")
    print(f"  row L1 norms : {_row_norms(cached_adj)}")
    print(f"  logit_tok_ids: {cached_teacher.logit_token_ids.tolist()[:15] if cached_teacher.logit_token_ids is not None else None}")
    if cached_teacher.logit_token_ids is not None and cached_teacher.teacher_dla_logits is not None:
        _ids = cached_teacher.logit_token_ids[:15]
        _vals = cached_teacher.teacher_dla_logits.float()[_ids]
        print(f"  logit_vals   : {[f'{v:.3f}' for v in _vals.tolist()]}")
    if cached_teacher.teacher_dla_logits is not None:
        top5 = cached_teacher.teacher_dla_logits.float().topk(5)
        print(f"  dla_logits top5 idx: {top5.indices.tolist()} vals: {[f'{v:.3f}' for v in top5.values.tolist()]}")
    else:
        print(f"  dla_logits top5 idx: None")

    # Build teacher graph live — same call as generate_teacher_data.py (string prompt → BOS added by ensure_tokenized).
    print(f"\n  [live teacher] building graph (prop_neurons={config.teacher_prop_neurons_per_layer})...")
    t0 = time.perf_counter()
    try:
        with torch.enable_grad():
            live_result = create_graph(
                teacher_adapter,
                prompt,
                attribution_targets=(
                    cached_teacher.logit_token_ids.cpu()
                    if cached_teacher.logit_token_ids is not None
                    else None
                ),
                prop_neurons_per_layer=config.teacher_prop_neurons_per_layer,
                top_k_logits=config.top_k_logits,
                temperature=config.temperature,
                batch_size=config.teacher_graph_batch_size,
                dtype=config.graph_dtype,
                verbose=False,
                build_create_graph=False,
                detach_result=True,
                skip_logit_attribution=False,
                include_dla_node=config.tokens_dla_nodes,
                dla_model_logits=None,  # Match cache generation: use model's own BOS-prefixed logits
                include_arg_nodes=config.tokens_dla_nodes,
                no_grad_supergraph=True,
                anova_nodes_per_label=config.teacher_anova_nodes_per_label,
                anova_range_radius=config.teacher_anova_range_radius,
                sum_min_specificity=config.teacher_sum_min_specificity,
                node_labels=config.teacher_graph_labels or None,
            )
    except Exception as exc:
        print(f"  ERROR building live teacher graph: {exc}")
        return
    print(f"  [live teacher] done in {time.perf_counter() - t0:.2f}s")

    live_sg = live_result.supergraph
    live_graph = live_result.graph
    live_labels = [lab[0] if lab else "?" for lab in (live_sg.supernode_labels or [])]
    live_adj = live_sg.supernode_adjacency_matrix.float().cpu()
    live_logit_ids = [t.vocab_idx for t in live_graph.logit_targets]

    # Compute DLA logits the same way as generate_teacher_data.py: no-BOS prompt-only
    # forward, last position. This matches cached teacher_dla_logits so the comparison
    # is apples-to-apples (target_position_logits uses BOS → different RoPE positions).
    _tok = teacher_adapter.tokenizer
    with torch.no_grad():
        _prompt_ids = _tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        _prompt_ids = _prompt_ids.to(teacher_adapter.cfg.device)
        _out = teacher_adapter.model(_prompt_ids)
        _live_logits_raw = (_out.logits if hasattr(_out, "logits") else _out).squeeze(0)
        live_dla_logits = _live_logits_raw[-1].detach().cpu()

    print(f"\n--- LIVE TEACHER ---")
    print(f"  n_supernodes : {len(live_sg.supernodes)}")
    print(f"  n_neurons    : {live_graph.n_neurons}")
    print(f"  labels       : {live_labels}")
    print(f"  adj shape    : {tuple(live_adj.shape)}")
    print(f"  adj stats    : {_adj_stats(live_adj)}")
    print(f"  row L1 norms : {_row_norms(live_adj)}")
    print(f"  logit_tok_ids: {live_logit_ids[:15]}")
    if live_logit_ids:
        _ids = torch.tensor(live_logit_ids[:15], dtype=torch.long)
        _vals = live_dla_logits[_ids]
        print(f"  logit_vals   : {[f'{v:.3f}' for v in _vals.tolist()]}")
    top5 = live_dla_logits.float().topk(5)
    print(f"  dla_logits top5 idx: {top5.indices.tolist()} vals: {[f'{v:.3f}' for v in top5.values.tolist()]}")
    else:
        print(f"  dla_logits top5 idx: None")

    # Label diff
    cached_label_set = set(cached_labels)
    live_label_set = set(live_labels)
    matched_labels = cached_label_set & live_label_set
    only_cached = cached_label_set - live_label_set
    only_live = live_label_set - cached_label_set

    print(f"\n--- LABEL COMPARISON ---")
    print(f"  matched ({len(matched_labels)}): {sorted(matched_labels)}")
    if only_cached:
        print(f"  only in CACHED ({len(only_cached)}): {sorted(only_cached)}")
    if only_live:
        print(f"  only in LIVE   ({len(only_live)}): {sorted(only_live)}")

    # Per-label adjacency comparison — align BOTH rows AND columns by label
    # so that element [t, s] means "supernode s → supernode t" for the same
    # (t, s) label pair in both matrices (ordering can differ between cached/live).
    if matched_labels:
        c_lab2idx = {lab: i for i, lab in enumerate(cached_labels)}
        l_lab2idx = {lab: i for i, lab in enumerate(live_labels)}
        shared_labels = sorted(matched_labels)
        print(f"\n--- PER-LABEL ADJ (rows=target, cols=source) — columns={shared_labels} ---")
        print(f"  (both rows and columns aligned by shared label order)")
        for tlab in shared_labels:
            ci = c_lab2idx[tlab]
            li = l_lab2idx[tlab]
            # Extract and reorder columns to the shared label order
            c_vals = [float(cached_adj[ci, c_lab2idx[slab]].item()) if slab in c_lab2idx else float("nan") for slab in shared_labels]
            l_vals = [float(live_adj[li, l_lab2idx[slab]].item()) if slab in l_lab2idx else float("nan") for slab in shared_labels]
            diffs = [abs(l - c) if not (c != c or l != l) else float("nan") for c, l in zip(c_vals, l_vals)]
            max_diff = max((d for d in diffs if d == d), default=float("nan"))
            mean_diff = sum(d for d in diffs if d == d) / max(sum(1 for d in diffs if d == d), 1)
            print(
                f"  row={tlab!r:25s}: "
                f"cached={[f'{v:.4f}' for v in c_vals]}  "
                f"live={[f'{v:.4f}' for v in l_vals]}  "
                f"|diff| max={max_diff:.4f} mean={mean_diff:.4f}"
            )

    # Student supergraph summary
    stu_adj = student_supergraph.supernode_adjacency_matrix.float().detach().cpu()
    stu_labels = [lab[0] if lab else "?" for lab in (student_supergraph.supernode_labels or [])]
    print(f"\n--- STUDENT ---")
    print(f"  n_supernodes : {len(student_supergraph.supernodes)}")
    print(f"  labels       : {stu_labels}")
    print(f"  adj stats    : {_adj_stats(stu_adj)}")
    print(f"  row L1 norms : {_row_norms(stu_adj)}")

    # Compute graph loss with live teacher vs cached teacher
    stu_adj_device = student_supergraph.supernode_adjacency_matrix
    s_lab2sid = {lab: sid for sid, lab in enumerate(stu_labels) if lab}

    def _compute_loss_for_teacher(teacher_sg: "SuperGraph", t_labels: list[str]) -> float | None:
        try:
            t2s = {
                tid: {s_lab2sid[lab]}
                for tid, lab in enumerate(t_labels)
                if lab in s_lab2sid
            }
            loss, _ = compute_graph_loss(
                teacher_sg.supernode_adjacency_matrix.float().to(
                    device=stu_adj_device.device, dtype=stu_adj_device.dtype
                ),
                stu_adj_device,
                t2s,
                list(range(len(teacher_sg.supernodes))),
                list(range(len(student_supergraph.supernodes))),
                similarity=config.graph_loss_type,
            )
            return float(loss.item())
        except Exception as exc:
            print(f"    (error computing loss: {exc})")
            return None

    cached_loss = _compute_loss_for_teacher(cached_sg, cached_labels)
    live_loss = _compute_loss_for_teacher(live_sg, live_labels)

    print(f"\n--- GRAPH LOSS ---")
    print(f"  cached teacher → student : {cached_loss:.4f}" if cached_loss is not None else "  cached teacher: ERROR")
    print(f"  live teacher   → student : {live_loss:.4f}" if live_loss is not None else "  live teacher: ERROR")
    if cached_loss is not None and live_loss is not None:
        print(f"  difference (live − cached): {live_loss - cached_loss:+.4f}")

    print(div + "\n")
    del live_result
    gc.collect()


def compute_prompt_graph_loss(
    *,
    prompt: str,
    student_adapter: HFLlamaGraphAdapter,
    config: GraphAuxConfig,
    cached_teacher: CachedTeacherPromptData,
    debug_teacher_adapter: HFLlamaGraphAdapter | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute graph loss for one prompt using a pre-cached teacher supergraph."""
    # ------------------------------------------------------------------
    # Teacher side: always from cache
    # ------------------------------------------------------------------
    teacher_supergraph = cached_teacher.supergraph
    logit_token_ids = cached_teacher.logit_token_ids
    if config.verbose:
        print(
            f"  [graph] loaded teacher supergraph from cache: "
            f"{len(teacher_supergraph.supernodes)} supernodes"
        )

    # ------------------------------------------------------------------
    # Student side: always live (trainable)
    # ------------------------------------------------------------------
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
            dataset=config.dataset,
            mlp_input_cache=config.mlp_input_cache,
            node_labels=config.student_graph_labels or [],
            anova_range_radius=config.student_anova_range_radius,
            anova_nodes_per_label=config.student_anova_nodes_per_label,
            sum_min_specificity=config.student_sum_min_specificity,
            include_dla_node=config.tokens_dla_nodes,
            dla_model_logits=cached_teacher.teacher_dla_logits if config.tokens_dla_nodes else None,
            include_arg_nodes=config.tokens_dla_nodes,
            no_grad_supergraph=True,
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
    if config.student_graph_labels is not None:
        label_set = set(config.student_graph_labels)
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
        raise RuntimeError(
            f"Teacher/student supernode label mismatch for prompt={prompt!r}.\n"
            + "\n".join(parts)
            + f"\n  Student labels: {sorted(student_label_set)}"
            + f"\n  Teacher labels: {sorted(teacher_label_set)}"
            + "\nRegenerate the teacher cache with flags matching the current distillation args "
            "(--include-arg-nodes / --include-dla-node)."
        )

    mapping = {
        tid: {s_label_to_sid[labels[0]]}
        for tid, labels in enumerate(teacher_supergraph.supernode_labels or [])
        if labels and labels[0] in s_label_to_sid
    }

    teacher_ids = list(range(len(teacher_supergraph.supernodes)))
    student_ids = list(range(len(student_supergraph.supernodes)))

    graph_loss, loss_breakdown = compute_graph_loss(
        teacher_supergraph.supernode_adjacency_matrix.detach().to(
            device=student_supergraph.supernode_adjacency_matrix.device,
            dtype=student_supergraph.supernode_adjacency_matrix.dtype,
        ),
        student_supergraph.supernode_adjacency_matrix,
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

    if config.debug_compare_teacher_graphs and debug_teacher_adapter is not None:
        _debug_compare_teacher_graphs(
            prompt=prompt,
            config=config,
            cached_teacher=cached_teacher,
            teacher_adapter=debug_teacher_adapter,
            student_supergraph=student_supergraph,
        )

    return graph_loss, metrics


def _load_cached_teacher(
    cache: TeacherDataCache,
    prompt: str,
    answer: int,
    device: torch.device,
) -> CachedTeacherPromptData:
    """Load one prompt's teacher artifacts from disk and reconstruct a SuperGraph."""
    from graph_loss.graph import SuperGraph  # local import to avoid circular

    try:
        sg_data = cache.load_teacher_supergraph(prompt, answer)
    except (KeyError, FileNotFoundError) as e:
        raise RuntimeError(
            "Teacher data cache is enabled but required graph data is missing for "
            f"prompt={prompt!r}, answer={answer!r}. Regenerate the cache for this "
            "dataset/tokenizer or remove --teacher-data-cache."
        ) from e

    if "supernode_labels" not in sg_data:
        raise RuntimeError(
            f"Teacher cache file for prompt={prompt!r}, answer={answer!r} is missing "
            "'supernode_labels'. Regenerate the teacher cache with the current "
            "generate_teacher_data.py."
        )

    logit_token_ids: torch.Tensor | None = sg_data.get("logit_token_ids")
    supergraph = SuperGraph(
        supernode_adjacency_matrix=sg_data["supernode_adjacency_matrix"].to(device),
        supernodes=sg_data["supernodes"],
        supernode_labels=sg_data.get("supernode_labels"),
    )

    # Load teacher logits at the last prompt-token position so that the student's
    # DLA supernode can be selected against the teacher's output distribution
    # instead of the student's own (which is wrong early in training and shifts
    # every step as the student learns).
    teacher_dla_logits: torch.Tensor | None = None
    try:
        logits_record = cache._load_logits_record(prompt, answer)
        prompt_len = int(logits_record.get("prompt_len", logits_record["input_ids"].numel()))
        full_logits: torch.Tensor = logits_record["logits"]  # [seq_len, vocab]
        if prompt_len > 0 and full_logits.shape[0] >= prompt_len:
            teacher_dla_logits = full_logits[prompt_len - 1].to(device=device)
    except Exception:
        pass  # fall back to student logits in compute_prompt_graph_loss

    return CachedTeacherPromptData(
        supergraph=supergraph,
        logit_token_ids=logit_token_ids,
        teacher_dla_logits=teacher_dla_logits,
    )


def _live_teacher_to_cached(
    result: "GraphPipelineResult",
    device: torch.device,
) -> CachedTeacherPromptData:
    """Wrap a live-built GraphPipelineResult as CachedTeacherPromptData for compute_prompt_graph_loss."""
    from graph_loss.graph import SuperGraph  # local import to avoid circular

    return CachedTeacherPromptData(
        supergraph=result.supergraph,
        logit_token_ids=result.graph.logit_token_ids.to(device),
        teacher_dla_logits=(
            result.target_position_logits.to(device)
            if result.target_position_logits is not None
            else None
        ),
    )


def _kl_per_position(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """KL(teacher || student) per token position. Returns [seq_len] float tensor."""
    t_probs = torch.softmax(teacher_logits / temperature, dim=-1)
    s_log_probs = torch.log_softmax(student_logits / temperature, dim=-1)
    return (t_probs * (t_probs.clamp(min=1e-10).log() - s_log_probs)).sum(dim=-1)


def _compare_tokens_loss_for_prompt(
    *,
    prompt: str,
    answer: int,
    teacher_adapter: HFLlamaGraphAdapter,
    student_adapter: HFLlamaGraphAdapter,
    config: GraphAuxConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Graph loss for one prompt using KL-selected response token positions.

    Tokenizes the full sequence (prompt + answer), selects the top
    compare_n_tokens response positions by teacher-student KL divergence,
    then builds separate teacher and student supergraphs for the causal
    prefix at each selected position using the unified build_teacher_supergraph path.
    """
    from graph_loss.create_graph import build_teacher_supergraph

    tokenizer = student_adapter.tokenizer
    prompt_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    )["input_ids"].squeeze(0)
    answer_ids = tokenizer(
        str(answer) + tokenizer.eos_token, return_tensors="pt", add_special_tokens=False
    )["input_ids"].squeeze(0)
    input_ids = torch.cat([prompt_ids, answer_ids]).to(device)
    response_start = int(prompt_ids.numel())
    response_end = int(input_ids.numel())

    response_positions = list(range(response_start, response_end))
    if not response_positions:
        raise RuntimeError(
            f"compare_n_tokens: no response tokens for prompt={prompt!r}"
        )

    # Forward passes (no grad) to get logits for KL-based position selection.
    with torch.no_grad():
        t_logits = teacher_adapter.model(input_ids.unsqueeze(0)).logits.squeeze(0).cpu()
        s_logits = student_adapter.model(input_ids.unsqueeze(0)).logits.squeeze(0).detach().cpu()

    logit_positions = [pos - 1 for pos in response_positions]
    n_select = min(config.compare_n_tokens, len(response_positions))
    kl_vals = _kl_per_position(
        t_logits[logit_positions], s_logits[logit_positions], config.temperature
    )
    selected_positions = [
        response_positions[i] for i in torch.topk(kl_vals, n_select).indices.tolist()
    ]

    position_losses: list[torch.Tensor] = []
    metric_sums: dict[str, float] = {}

    for pos in selected_positions:
        # Pass raw prefix token IDs so ensure_tokenized auto-prepends BOS without
        # a lossy decode/re-encode round trip. Decode only for the student call
        # (compute_prompt_graph_loss takes a string and tokenizes it the same way).
        prefix_ids = input_ids[:pos].cpu()
        prefix_str = tokenizer.decode(prefix_ids, skip_special_tokens=True)

        teacher_result = build_teacher_supergraph(
            teacher_adapter,
            prefix_ids,
            prop_neurons_per_layer=config.teacher_prop_neurons_per_layer,

            top_k_logits=config.top_k_logits,
            temperature=config.temperature,
            batch_size=config.teacher_graph_batch_size,
            dtype=config.graph_dtype,
            include_dla_node=config.tokens_dla_nodes,
            include_arg_nodes=config.tokens_dla_nodes,
            anova_nodes_per_label=config.teacher_anova_nodes_per_label,
            anova_range_radius=config.teacher_anova_range_radius,
            sum_min_specificity=config.teacher_sum_min_specificity,
            node_labels=config.teacher_graph_labels or None,
            verbose=config.verbose,
        )
        cached = _live_teacher_to_cached(teacher_result, device)
        del teacher_result

        pos_loss, pos_metrics = compute_prompt_graph_loss(
            prompt=prefix_str,
            student_adapter=student_adapter,
            config=config,
            cached_teacher=cached,
        )
        position_losses.append(pos_loss)
        for key, val in pos_metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(val)

    avg_loss = torch.stack(position_losses).mean()
    n_pos = float(len(position_losses))
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
    teacher_cache: TeacherDataCache | None = None,
    teacher_adapter: HFLlamaGraphAdapter | None = None,
    answers: list[int],
    debug_teacher_adapter: HFLlamaGraphAdapter | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute and backprop graph loss one prompt at a time.

    Accepts either ``teacher_cache`` (pre-generated, loaded from disk) or
    ``teacher_adapter`` (live, built on-the-fly via build_teacher_supergraph).
    Exactly one must be provided.

    Processes each prompt's attribution graph immediately and backprops before
    building the next, keeping peak memory bounded to a single prompt.
    """
    if teacher_cache is None and teacher_adapter is None:
        raise RuntimeError(
            "backward_batch_graph_loss requires either teacher_cache or teacher_adapter."
        )
    if config.compare_n_tokens is not None and teacher_adapter is None:
        raise RuntimeError(
            "compare_n_tokens requires a live teacher adapter (--teacher-data-cache is incompatible)."
        )
    if not prompts:
        return torch.tensor(0.0, device=device), {}

    import gc

    metric_sums: dict[str, float] = {}
    detached_losses = []
    denom = float(len(prompts))
    graph_backward_prompts = 0

    for i, prompt in enumerate(prompts):
        if config.compare_n_tokens is not None:
            prompt_loss, prompt_metrics = _compare_tokens_loss_for_prompt(
                prompt=prompt,
                answer=answers[i],
                teacher_adapter=teacher_adapter,  # type: ignore[arg-type]
                student_adapter=student_adapter,
                config=config,
                device=device,
            )
        elif teacher_cache is not None:
            cached = _load_cached_teacher(teacher_cache, prompt, answers[i], device)
            prompt_loss, prompt_metrics = compute_prompt_graph_loss(
                prompt=prompt,
                student_adapter=student_adapter,
                config=config,
                cached_teacher=cached,
                debug_teacher_adapter=debug_teacher_adapter,
            )
        else:
            from graph_loss.generate_teacher_data import generate_teacher_sample

            cached = generate_teacher_sample(
                teacher_adapter,  # type: ignore[arg-type]
                prompt,
                answers[i],
                prop_neurons_per_layer=config.teacher_prop_neurons_per_layer,
                top_k_logits=config.top_k_logits,
                temperature=config.temperature,
                batch_size=config.teacher_graph_batch_size,
                include_dla_node=config.tokens_dla_nodes,
                include_arg_nodes=config.tokens_dla_nodes,
                anova_nodes_per_label=config.teacher_anova_nodes_per_label,
                anova_range_radius=config.teacher_anova_range_radius,
                sum_min_specificity=config.teacher_sum_min_specificity,
                node_labels=config.teacher_graph_labels or None,
                verbose=config.verbose,
            )
            prompt_loss, prompt_metrics = compute_prompt_graph_loss(
                prompt=prompt,
                student_adapter=student_adapter,
                config=config,
                cached_teacher=cached,
                debug_teacher_adapter=debug_teacher_adapter,
            )
        detached_losses.append(prompt_loss.detach())
        scaled_loss = (loss_scale / denom) * prompt_loss
        if scaled_loss.requires_grad:
            scaled_loss.backward()
            graph_backward_prompts += 1
        elif config.verbose:
            print("  [graph] WARN: graph loss has no grad; skipping backward")
        del scaled_loss
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for key, value in prompt_metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)

    loss = torch.stack(detached_losses).mean()
    metrics = {key: value / denom for key, value in metric_sums.items()}
    metrics["graph_prompts"] = float(len(prompts))
    metrics["graph_backward_prompts"] = float(graph_backward_prompts)
    return loss, metrics


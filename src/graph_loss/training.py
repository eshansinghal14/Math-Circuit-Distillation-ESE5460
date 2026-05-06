"""Training-time graph auxiliary loss helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from graph_loss.align import (
    _build_full_vocab_prob_deltas,
    align_supernodes_prob_delta,
    compute_supernode_dla,
)
from graph_loss.attribution.attribute import attribute
from graph_loss.graph import (
    SuperGraph,
    build_super_graph,
    compute_neuron_logit_influence,
    extract_supernode_members,
    normalize_matrix,
    prune_graph,
)
from graph_loss.hf_adapter import HFLlamaGraphAdapter
from graph_loss.loss import compute_graph_loss, compute_logit_focus_loss
from graph_loss.teacher_data_cache import TeacherDataCache


@dataclass
class CachedTeacherPromptData:
    """Pre-computed teacher artifacts for one prompt, loaded from TeacherDataCache.

    Populating this avoids running the 8B teacher model during training — the
    supergraph, DLA vectors, and logit token IDs were generated offline by
    ``generate_teacher_data.py`` and are simply loaded from disk each step.
    """

    supergraph: SuperGraph
    dla_dict: dict[int, torch.Tensor]
    logit_token_ids: torch.Tensor | None
    n_vocab: int


@dataclass
class GraphAuxConfig:
    lambda_graph: float = 0.1
    graph_dtype: torch.dtype | None = None
    top_k_logits: int | None = 20
    prop_neurons_per_layer: float = 0.1
    teacher_graph_batch_size: int = 512
    student_graph_batch_size: int = 1
    verbose: bool = False
    graph_prune: bool = False
    graph_node_threshold: float = 0.8
    graph_edge_threshold: float = 0.98
    graph_similarity_threshold: float = 0.7
    graph_max_fan_out: int = 4
    graph_node_weight: float = 1.0  # weight of prob-delta node loss within L_graph
    graph_edge_weight: float = 0.0  # weight of edge structural loss within L_graph
    fast_teacher_graph: bool = False  # skip expensive TL/HF backward graph; linear logit block + proxy influence
    # Supergraph clustering params for the student (mirrors build_super_graph args)
    student_computation_eps: float = 0.1
    student_embedding_eps: float = 0.1
    student_activation_forward_batch_size: int = 32
    # When True, student build_graph skips populating the [logits, neurons] block
    # of B (saves ~top_k_logits backward passes, halves peak memory).  In that
    # case the alignment falls back to the legacy DLA-sum signal for the student,
    # which lives in a different functional space than the teacher's real
    # ablation prob-deltas, producing noisy / mostly-wrong matches.  Set to
    # False to populate real Jacobian-mediated logit influence rows and use them
    # as the student's "supernode_prob_delta" alignment signal — these capture
    # cross-layer + attention-mediated paths and are sign-comparable to the
    # cached teacher prob-deltas.
    # For the logit-focus loss (Phase 3), skip_logit_attribution=True is correct:
    # focus is computed via a separate cheap forward pass, not from B's logit rows.
    student_skip_logit_attribution: bool = True
    # Optional: log per-prompt cosine-matrix stats for the supernode alignment.
    align_diagnostic: bool = False
    # Phase 3 loss weight: KL divergence between teacher and student logit-focus
    # distributions (sum of |DLA_neuron| across all selected neurons, normalized
    # to a probability distribution over the top-K logit targets).  Requires no
    # cross-model alignment or supernode matching.
    graph_focus_weight: float = 0.0


def _aggregate_supernode_logit_influence(
    graph,
    supernodes: list[list[int]],
) -> torch.Tensor:
    """Sum per-neuron logit-influence rows of B over each supernode's members.

    Returns ``[n_supernodes, n_logits]`` in logit-attribution space.  This
    requires ``graph`` to have been built with ``skip_logit_attribution=False``;
    otherwise the [logits, neurons] block of B is zero and this returns a
    zero tensor (caller should detect and fall back to the DLA-based signal).

    The returned tensor is differentiable through B → ``source_vectors_t`` →
    student model parameters (down_proj.weight etc.), so when the alignment
    similarity passes through these vectors the resulting cosine gradients
    flow back into the student.
    """
    logit_inf = compute_neuron_logit_influence(graph)  # [n_neurons, n_logits]
    rows: list[torch.Tensor] = []
    zero_row = torch.zeros(
        graph.n_logits,
        dtype=logit_inf.dtype,
        device=logit_inf.device,
    )
    for members in supernodes:
        if not members:
            rows.append(zero_row)
            continue
        idx = torch.tensor(members, dtype=torch.long, device=logit_inf.device)
        rows.append(logit_inf.index_select(0, idx).sum(dim=0))
    if not rows:
        return torch.empty(
            (0, graph.n_logits),
            dtype=logit_inf.dtype,
            device=logit_inf.device,
        )
    return torch.stack(rows, dim=0)


def _log_alignment_diagnostic(
    teacher_supergraph: SuperGraph,
    student_supergraph: SuperGraph,
    teacher_graph,
    student_graph,
    *,
    n_vocab: int,
    similarity_threshold: float,
) -> None:
    """Log per-prompt teacher\u2194student supernode cosine-matrix stats.

    Cheap (one cosine matmul) and side-effect-free.  Prints:
      mean / median / max cosine over the teacher\u2192student matrix,
      fraction of teacher supernodes with at least one match >= threshold,
      and the same for >= 0.3 (a reasonable practical floor).
    """
    import torch.nn.functional as F

    t_full = _build_full_vocab_prob_deltas(teacher_supergraph, teacher_graph, n_vocab)
    s_full = _build_full_vocab_prob_deltas(student_supergraph, student_graph, n_vocab)
    if t_full.numel() == 0 or s_full.numel() == 0:
        print("  [align-diag] empty teacher or student supergraph; skipping")
        return
    if t_full.shape[1] != s_full.shape[1]:
        max_w = max(t_full.shape[1], s_full.shape[1])
        if t_full.shape[1] < max_w:
            t_full = F.pad(t_full, (0, max_w - t_full.shape[1]))
        if s_full.shape[1] < max_w:
            s_full = F.pad(s_full, (0, max_w - s_full.shape[1]))
    t_norm = F.normalize(t_full, dim=1)
    s_norm = F.normalize(s_full, dim=1)
    sim = t_norm @ s_norm.T  # [n_teacher, n_student]
    if sim.numel() == 0:
        print("  [align-diag] zero-size similarity matrix")
        return
    best_per_teacher = sim.max(dim=1).values
    n_teacher = sim.shape[0]
    n_student = sim.shape[1]
    frac_at_thr = (best_per_teacher >= similarity_threshold).float().mean().item()
    frac_at_03 = (best_per_teacher >= 0.3).float().mean().item()
    print(
        "  [align-diag] "
        f"teacher_sn={n_teacher} student_sn={n_student} "
        f"sim mean={sim.mean().item():.3f} median={sim.median().item():.3f} "
        f"max={sim.max().item():.3f} | best-per-teacher "
        f"mean={best_per_teacher.mean().item():.3f} "
        f"median={best_per_teacher.median().item():.3f} "
        f"max={best_per_teacher.max().item():.3f} | "
        f"frac>=thr({similarity_threshold:.2f})={frac_at_thr:.2f} "
        f"frac>=0.30={frac_at_03:.2f}"
    )


def _compute_teacher_logit_focus(teacher_supergraph: SuperGraph) -> torch.Tensor | None:
    """Return a [n_logits] teacher logit-focus vector from cached prob-deltas.

    Specifically: ``|supernode_prob_deltas|.sum(0)`` — for each of the top-K
    cached logit targets, the total magnitude of ablation-based causal influence
    across all teacher supernodes.  This is always computed without grad.

    Returns ``None`` if the supergraph has no cached prob-deltas.
    """
    pd = teacher_supergraph.supernode_prob_deltas  # [n_teacher_sn, n_logits] or None
    if pd is None or pd.numel() == 0:
        return None
    return pd.detach().float().abs().sum(0)  # [n_logits]


def _aggregate_supergraph_adjacency(graph, supernodes: list[list[int]]) -> SuperGraph:
    """Aggregate a differentiable graph adjacency using fixed supernode membership.

    Uses torch.stack (out-of-place) instead of in-place setitem so that the
    gradient from the edge loss flows back through supernode_adjacency_matrix
    → adjacency_matrix → source_vectors_t → model parameters (down_proj.weight).
    In-place setitem into a torch.zeros leaf tensor silently breaks the grad chain.
    """
    adj_matrix_norm = normalize_matrix(graph.adjacency_matrix)
    num_supernodes = len(supernodes)
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


def compute_prompt_graph_loss(
    *,
    prompt: str,
    teacher_graph_model: Any | None,
    student_adapter: HFLlamaGraphAdapter,
    config: GraphAuxConfig,
    cached_teacher: CachedTeacherPromptData | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    # ------------------------------------------------------------------
    # Teacher side: either live attribution or pre-computed cache
    # ------------------------------------------------------------------
    if cached_teacher is not None:
        teacher_supergraph = cached_teacher.supergraph
        teacher_dla_override = cached_teacher.dla_dict
        logit_token_ids = cached_teacher.logit_token_ids
        n_vocab = cached_teacher.n_vocab
        teacher_graph = None
        teacher_prune_result = None
        if config.verbose:
            print(
                f"  [graph] loaded teacher supergraph from cache: "
                f"{len(teacher_supergraph.supernodes)} supernodes"
            )
    else:
        if teacher_graph_model is None:
            raise RuntimeError(
                "teacher_graph_model must be provided when not using teacher cache."
            )
        if config.verbose:
            print(f"  [graph] building teacher graph for prompt: {prompt!r}")
        teacher_graph = attribute(
            prompt=prompt,
            model=teacher_graph_model,
            top_k_logits=config.top_k_logits,
            prop_neurons_per_layer=config.prop_neurons_per_layer,
            batch_size=config.teacher_graph_batch_size,
            verbose=config.verbose,
            fast=config.fast_teacher_graph,
        )
        logit_token_ids = teacher_graph.logit_token_ids
        n_vocab = teacher_graph_model.cfg.d_vocab
        teacher_prune_result = None
        if config.graph_prune:
            if config.verbose:
                print("  [graph] pruning teacher graph")
            teacher_prune_result = prune_graph(
                teacher_graph,
                node_threshold=config.graph_node_threshold,
                edge_threshold=config.graph_edge_threshold,
            )
            teacher_graph = teacher_graph.apply_prune_result(teacher_prune_result)

        if config.verbose:
            print("  [graph] building teacher supergraph")
        supergraph_start = time.perf_counter()
        with torch.no_grad():
            teacher_supergraph = build_super_graph(
                teacher_graph,
                teacher_graph_model,
                prune_result=teacher_prune_result,
                activation_forward_batch_size=config.teacher_graph_batch_size,
                cluster_method="ablation",
            )
        if config.verbose:
            print(
                "  [graph] teacher supergraph complete: "
                f"{len(teacher_supergraph.supernodes)} supernodes in "
                f"{time.perf_counter() - supergraph_start:.2f}s",
            )
        teacher_dla_override = None

    # ------------------------------------------------------------------
    # Student side: always live (trainable)
    # ------------------------------------------------------------------
    if config.verbose:
        print(f"  [graph] building student graph for prompt: {prompt!r}")
    # Force student to attribute over the same logit tokens as the teacher so that
    # prob-delta vectors occupy the same vocabulary positions in both models.
    # The student uses full Jacobian attribution (fast=False) to populate real
    # neuron-to-neuron edges, so the supernode adjacency aggregated by
    # _aggregate_supergraph_adjacency carries genuine causal structure rather than
    # the all-zero block produced by fast/DLA-only attribution.
    student_graph = student_adapter.build_graph(
        prompt,
        attribution_targets=logit_token_ids.cpu() if logit_token_ids is not None else None,
        prop_neurons_per_layer=config.prop_neurons_per_layer,
        batch_size=config.student_graph_batch_size,
        dtype=config.graph_dtype,
        verbose=config.verbose,
        create_graph=False,
        detach_result=False,
        fast=False,
        # When skip_logit_attribution=True we save ~top_k_logits backward passes
        # but the student loses real Jacobian logit-influence rows, leaving the
        # supernode alignment signal in a different functional space than the
        # cached teacher prob-deltas (sum-of-DLAs vs real ablation prob-deltas).
        # Default is False so the student supernode alignment can use real
        # logit influence aggregated per supernode (see below).
        skip_logit_attribution=config.student_skip_logit_attribution,
    )
    if config.align_diagnostic:
        print(f"  [grad-trace] student_graph.adjacency_matrix.requires_grad = {student_graph.adjacency_matrix.requires_grad}")

    student_prune_result = None
    if config.graph_prune:
        if config.verbose:
            print("  [graph] pruning student graph")
        student_prune_result = prune_graph(
            student_graph,
            node_threshold=config.graph_node_threshold,
            edge_threshold=config.graph_edge_threshold,
        )
        student_graph = student_graph.apply_prune_result(student_prune_result)
        if config.align_diagnostic:
            print(f"  [grad-trace] post-prune adjacency.requires_grad = {student_graph.adjacency_matrix.requires_grad}")

    supergraph_start = time.perf_counter()
    with torch.no_grad():
        student_supergraph_structure = build_super_graph(
            student_graph,
            student_adapter,
            prune_result=student_prune_result,
            activation_forward_batch_size=config.student_activation_forward_batch_size,
            computation_eps=config.student_computation_eps,
            embedding_eps=config.student_embedding_eps,
            cluster_method="ablation",
        )
    student_supergraph = _aggregate_supergraph_adjacency(
        student_graph,
        student_supergraph_structure.supernodes,
    )
    if config.align_diagnostic:
        print(f"  [grad-trace] student_supergraph.adjacency.requires_grad = {student_supergraph.supernode_adjacency_matrix.requires_grad}")
    # ----------------------------------------------------------------------
    # Choose the student supernode alignment signal.
    #
    # The cached teacher `supernode_prob_deltas` are real ablation prob-deltas
    # in [-1, 1] over the selected logit token ids.  For cosine similarity
    # alignment to be meaningful, the student vector must live in a
    # sign-comparable space.
    #
    # - When `skip_logit_attribution=False`, the student adjacency carries
    #   real Jacobian-mediated logit-influence rows (Anthropic's exact
    #   attribution-graph edge weight from each neuron to each selected
    #   logit).  Summing these per supernode gives a "supernode logit
    #   influence" vector that captures cross-layer + attention-mediated
    #   paths through the frozen-attention/LN replacement model — sign-
    #   comparable to the teacher's prob-delta and far better than
    #   sum-of-DLAs (which only sees the residual-direct path).
    #
    # - When `skip_logit_attribution=True`, the [logits, neurons] block of
    #   B is zero, so we fall back to the legacy DLA-based signal that the
    #   structure pass produced (kept here only for backward compat).
    if not config.student_skip_logit_attribution and student_graph.n_logits > 0:
        student_supernode_prob_deltas = _aggregate_supernode_logit_influence(
            student_graph,
            student_supergraph_structure.supernodes,
        )
    else:
        student_supernode_prob_deltas = student_supergraph_structure.supernode_prob_deltas

    student_supergraph = student_supergraph._replace(
        supernode_prob_deltas=student_supernode_prob_deltas,
        logit_token_ids=student_graph.logit_token_ids,
    )
    if config.verbose:
        print(
            "  [graph] student supergraph complete: "
            f"{len(student_supergraph.supernodes)} supernodes in "
            f"{time.perf_counter() - supergraph_start:.2f}s",
        )

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------
    if config.align_diagnostic:
        _log_alignment_diagnostic(
            teacher_supergraph,
            student_supergraph,
            teacher_graph,
            student_graph,
            n_vocab=n_vocab,
            similarity_threshold=config.graph_similarity_threshold,
        )

    if config.verbose:
        print("  [graph] aligning supernodes via prob-delta and computing graph loss")
    alignment = align_supernodes_prob_delta(
        teacher_supergraph,
        student_supergraph,
        teacher_graph,  # may be None when using cache
        student_graph,
        similarity_threshold=config.graph_similarity_threshold,
        max_fan_out=config.graph_max_fan_out,
        n_vocab=n_vocab,
    )

    teacher_ids = list(range(len(teacher_supergraph.supernodes)))
    student_ids = list(range(len(student_supergraph.supernodes)))

    # Override teacher DLA vectors with precomputed (cache) or freshly computed ones.
    if config.graph_node_weight > 0.0:
        if teacher_dla_override is not None:
            # Use cached DLA directly — no teacher model needed.
            for cid, dla_vec in teacher_dla_override.items():
                alignment.teacher_dla[cid] = dla_vec.detach()
        elif teacher_graph_model is not None and teacher_graph is not None:
            with torch.no_grad():
                teacher_sn_members = extract_supernode_members(
                    teacher_supergraph, teacher_graph, teacher_graph_model
                )
                W_U_t = teacher_graph_model.unembed.W_U
                n_vocab_t = teacher_graph_model.cfg.d_vocab
                for sn in teacher_sn_members:
                    dla = compute_supernode_dla(sn, W_U_t)[:n_vocab_t]
                    alignment.teacher_dla[sn["cluster_id"]] = dla.detach()

    if config.graph_node_weight > 0.0:
        # `align_supernodes_prob_delta` stores DETACHED student vectors in
        # alignment.student_dla (it scatters via `.detach().float().cpu()`
        # in `_build_full_vocab_prob_deltas`).  For node-loss backprop we
        # need to overwrite those entries with a differentiable equivalent.
        if config.student_skip_logit_attribution or student_graph.n_logits == 0:
            # Legacy path: differentiable DLA via the adapter (independent
            # forward pass that re-derives source vectors from down_proj).
            student_dla_with_grad_dict = student_adapter.compute_supernode_dlas_with_grad(
                prompt=prompt,
                supernodes=student_supergraph.supernodes,
                neuron_locations_t=student_graph.neuron_locations,
                n_vocab=n_vocab,
                dtype=config.graph_dtype,
            )
            for sid, dla_tensor in student_dla_with_grad_dict.items():
                alignment.student_dla[sid] = dla_tensor
        else:
            # New path: scatter the differentiable supernode logit-influence
            # vectors (already on `student_supergraph.supernode_prob_deltas`)
            # from the [n_logits] compact basis to a full-vocab basis.  This
            # preserves the autograd graph through B \u2192 source_vectors_t
            # \u2192 student parameters.
            sn_logit_inf = student_supergraph.supernode_prob_deltas
            if sn_logit_inf is not None and sn_logit_inf.numel() > 0:
                logit_token_ids_dev = student_graph.logit_token_ids.to(
                    device=sn_logit_inf.device,
                    dtype=torch.long,
                )
                actual_vocab = max(
                    int(n_vocab),
                    int(logit_token_ids_dev.max().item()) + 1,
                )
                # Use out-of-place scatter so the autograd chain through
                # sn_logit_inf survives.  Building a leaf zeros tensor and
                # then setitem'ing with [logit_token_ids_dev] silently breaks
                # the grad chain (same bug pattern already documented in
                # hf_adapter.build_graph and _aggregate_supergraph_adjacency).
                for sid in range(sn_logit_inf.shape[0]):
                    base = torch.zeros(
                        actual_vocab,
                        dtype=sn_logit_inf.dtype,
                        device=sn_logit_inf.device,
                    )
                    full_vec = base.scatter(0, logit_token_ids_dev, sn_logit_inf[sid])
                    alignment.student_dla[sid] = full_vec

    if config.align_diagnostic:
        print(f"  [grad-trace] mapping size: {sum(len(v) for v in alignment.mapping.values())} edges, teacher_supernodes={len(teacher_ids)}, student_supernodes={len(student_ids)}")
    graph_loss, loss_breakdown = compute_graph_loss(
        teacher_supergraph.supernode_adjacency_matrix.detach().to(
            device=student_supergraph.supernode_adjacency_matrix.device,
            dtype=student_supergraph.supernode_adjacency_matrix.dtype,
        ),
        student_supergraph.supernode_adjacency_matrix,
        alignment.mapping,
        teacher_ids,
        student_ids,
        teacher_dla=alignment.teacher_dla,
        student_dla=alignment.student_dla,
        edge_weight=config.graph_edge_weight,
        node_weight=config.graph_node_weight,
    )
    if config.align_diagnostic:
        print(f"  [grad-trace] graph_loss.requires_grad = {graph_loss.requires_grad}, value = {graph_loss.item():.6f}")

    # ------------------------------------------------------------------
    # Phase 3: logit-focus distribution loss (no alignment required)
    # ------------------------------------------------------------------
    focus_loss_value = 0.0
    if config.graph_focus_weight > 0.0 and logit_token_ids is not None:
        teacher_focus = _compute_teacher_logit_focus(teacher_supergraph)
        if teacher_focus is not None and teacher_focus.numel() > 0:
            student_focus = student_adapter.compute_logit_focus_vector_with_grad(
                prompt=prompt,
                neuron_locations_t=student_graph.neuron_locations,
                logit_token_ids=logit_token_ids,
                dtype=config.graph_dtype,
            )
            # Restrict teacher_focus to the same n_logits as student_focus in case
            # the cached teacher used a different top_k (or a different subset).
            n_l = student_focus.shape[0]
            t_focus = teacher_focus[:n_l].to(
                device=student_focus.device, dtype=student_focus.dtype
            )
            focus_loss = compute_logit_focus_loss(t_focus, student_focus)
            focus_loss_value = float(focus_loss.item())
            if config.align_diagnostic:
                print(
                    f"  [focus] teacher_focus_sum={float(t_focus.sum()):.4f} "
                    f"student_focus_sum={float(student_focus.sum().item()):.4f} "
                    f"focus_loss={focus_loss_value:.6f} grad={focus_loss.requires_grad}"
                )
            graph_loss = graph_loss + config.graph_focus_weight * focus_loss

    metrics = {
        "teacher_supernodes": len(teacher_ids),
        "student_supernodes": len(student_ids),
        "teacher_graph_neurons": int(teacher_graph.n_neurons) if teacher_graph is not None else 0,
        "student_graph_neurons": int(student_graph.n_neurons),
        "aligned_teacher_supernodes": sum(
            1 for teacher_id in teacher_ids if alignment.mapping.get(teacher_id)
        ),
        "mean_alignment_similarity": (
            sum(alignment.best_sim.values()) / len(alignment.best_sim)
            if alignment.best_sim
            else 0.0
        ),
        "focus_loss": focus_loss_value,
        **loss_breakdown,
    }
    return graph_loss, metrics


def _load_cached_teacher(
    cache: TeacherDataCache,
    prompt: str,
    answer: int,
    device: torch.device,
) -> CachedTeacherPromptData | None:
    """Load one prompt's teacher artifacts from disk and reconstruct a SuperGraph.

    Returns None on a cache miss so callers can fall back to KL-only loss.
    """
    from graph_loss.graph import SuperGraph  # local import to avoid circular

    try:
        sg_data = cache.load_teacher_supergraph(prompt, answer)
    except KeyError:
        return None
    logit_token_ids: torch.Tensor | None = sg_data.get("logit_token_ids")
    supergraph = SuperGraph(
        supernode_adjacency_matrix=sg_data["supernode_adjacency_matrix"].to(device),
        supernodes=sg_data["supernodes"],
        supernode_prob_deltas=sg_data.get("supernode_prob_deltas"),
        all_supernode_prob_delta_norms=sg_data.get("all_supernode_prob_delta_norms"),
        prob_delta_elbow_index=sg_data.get("prob_delta_elbow_index"),
        # Embed logit_token_ids so _build_full_vocab_prob_deltas works without graph
        logit_token_ids=logit_token_ids,
    )
    prob_deltas = sg_data.get("supernode_prob_deltas")
    if prob_deltas is not None and prob_deltas.numel() > 0:
        # Prefer ablation prob-deltas over DLA as the teacher node-loss target.
        # The alignment step (`align_supernodes_prob_delta`) already scatters
        # these 1000-dim vectors into full vocab space via logit_token_ids and
        # stores them in `alignment.teacher_dla`.  Returning an empty dla_dict
        # prevents the subsequent override loop in `compute_prompt_graph_loss`
        # from replacing those prob-delta targets with the cheaper DLA vectors.
        dla_dict: dict[int, torch.Tensor] = {}
        n_vocab = cache.teacher_vocab_size
    else:
        # Fallback for caches generated without prob-deltas (fast-mode / legacy).
        dla_data = cache.load_teacher_supernode_dla(prompt, answer)
        dla_dict = {
            int(cid): vec.to(device)
            for cid, vec in zip(dla_data["cluster_ids"], dla_data["dla"])
        }
        n_vocab = int(dla_data["dla"].shape[-1]) if dla_data["dla"].numel() > 0 else cache.teacher_vocab_size
    return CachedTeacherPromptData(
        supergraph=supergraph,
        dla_dict=dla_dict,
        logit_token_ids=logit_token_ids,
        n_vocab=n_vocab,
    )


def compute_batch_graph_loss(
    *,
    prompts: list[str],
    teacher_graph_model: Any | None,
    student_adapter: HFLlamaGraphAdapter,
    config: GraphAuxConfig,
    device: torch.device,
    teacher_cache: TeacherDataCache | None = None,
    answers: list[int] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    losses = []
    metric_sums: dict[str, float] = {}
    for i, prompt in enumerate(prompts):
        cached = (
            _load_cached_teacher(teacher_cache, prompt, answers[i], device)
            if teacher_cache is not None and answers is not None
            else None
        )
        if cached is None and teacher_graph_model is None:
            # Cache miss and no live teacher — skip graph loss for this prompt.
            continue
        prompt_loss, prompt_metrics = compute_prompt_graph_loss(
            prompt=prompt,
            teacher_graph_model=teacher_graph_model,
            student_adapter=student_adapter,
            config=config,
            cached_teacher=cached,
        )
        losses.append(prompt_loss)
        for key, value in prompt_metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)

    if not losses:
        return torch.tensor(0.0, device=device), {}

    loss = torch.stack(losses).mean()
    denom = float(len(losses))
    metrics = {key: value / denom for key, value in metric_sums.items()}
    metrics["graph_prompts"] = float(len(losses))
    return loss, metrics


def backward_batch_graph_loss(
    *,
    prompts: list[str],
    teacher_graph_model: Any | None,
    student_adapter: HFLlamaGraphAdapter,
    config: GraphAuxConfig,
    device: torch.device,
    loss_scale: float,
    teacher_cache: TeacherDataCache | None = None,
    answers: list[int] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute and backprop graph loss one prompt at a time.

    When ``teacher_cache`` is provided together with ``answers``, the 8B teacher
    is not touched at all during this call — pre-computed supergraphs and DLA
    vectors are loaded from disk instead, making each step ~15-20× faster.

    This keeps peak graph-loss memory bounded by a single problem instead of
    retaining every prompt's attribution graph until the batch backward call.
    """
    if not prompts:
        return torch.tensor(0.0, device=device), {}

    import gc

    metric_sums: dict[str, float] = {}
    detached_losses = []
    denom = float(len(prompts))
    graph_backward_prompts = 0
    for i, prompt in enumerate(prompts):
        cached = (
            _load_cached_teacher(teacher_cache, prompt, answers[i], device)
            if teacher_cache is not None and answers is not None
            else None
        )
        if cached is None and teacher_graph_model is None:
            # Cache miss and no live teacher — skip graph loss for this prompt.
            continue
        prompt_loss, prompt_metrics = compute_prompt_graph_loss(
            prompt=prompt,
            teacher_graph_model=teacher_graph_model,
            student_adapter=student_adapter,
            config=config,
            cached_teacher=cached,
        )
        detached_losses.append(prompt_loss.detach())
        scaled_prompt_loss = (loss_scale / denom) * prompt_loss
        if scaled_prompt_loss.requires_grad:
            scaled_prompt_loss.backward()
            graph_backward_prompts += 1
        else:
            print(f"  [graph] WARN: prompt_loss has no grad; skipping backward")
        for key, value in prompt_metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
        del prompt_loss, scaled_prompt_loss
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    loss = torch.stack(detached_losses).mean()
    metrics = {key: value / denom for key, value in metric_sums.items()}
    metrics["graph_prompts"] = float(len(prompts))
    metrics["graph_backward_prompts"] = float(graph_backward_prompts)
    return loss, metrics

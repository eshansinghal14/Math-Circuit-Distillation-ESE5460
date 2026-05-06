"""Training-time graph auxiliary loss helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from graph_loss.align import (
    _build_full_vocab_prob_deltas,
    align_supernodes_prob_delta,
)
from graph_loss.attribution.attribute import attribute
from graph_loss.graph import (
    SuperGraph,
    build_super_graph,
    normalize_matrix,
    prune_graph,
)
from graph_loss.hf_adapter import HFLlamaGraphAdapter
from graph_loss.loss import compute_graph_loss
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
    graph_gen_batch_size: int = 1
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
    # When True, student build_graph skips populating the [logits, neurons]
    # block of B (saves ~top_k_logits backward passes, halves peak memory).
    # The current alignment + focus signal comes from
    # compute_supernode_ablation_prob_deltas_with_grad (real hook-based ablation
    # on the student), so this block is no longer needed for the auxiliary
    # loss — keep True for memory.  Setting to False is only useful for
    # historical experiments that consume the [logits, neurons] block of B.
    student_skip_logit_attribution: bool = True
    # Optional: log per-prompt cosine-matrix stats for the supernode alignment
    # plus signed teacher/student focus diagnostics.
    align_diagnostic: bool = False
    # Logit-focus loss weight (signed MSE between teacher and student
    # supernode_prob_deltas summed across supernodes).  Sign-preserving —
    # avoids the |DLA| collapse failure mode at high lambda_graph.  Requires
    # no cross-model alignment.
    graph_focus_weight: float = 0.0
    # Gradient pathway for the per-supernode prob-delta signal:
    #
    #   - "approx": cheap differentiable DLA-approximation
    #     (compute_supernode_dla_approx_prob_deltas_with_grad).  One grad-
    #     enabled forward + per-supernode (norm + lm_head + softmax).
    #     ~95% directionally faithful, fits easily on one GPU.
    #
    #   - "true":  full hook-based ablation with grad, processed in small
    #     chunks with backward called per-chunk so activation memory stays
    #     bounded.  100% faithful to teacher's operational space (cross-
    #     layer effects propagate naturally) at the cost of ~3-5x slower
    #     training.  Required for max-fidelity research runs.
    graph_grad_mode: str = "approx"
    # In "true" mode, how many supernodes to ablate per batched forward+backward.
    # 1 = pure serial (lowest memory), higher = more parallelism per chunk.
    # 4 keeps activation memory ~4×baseline forward, well under 80 GB on A100.
    graph_true_grad_chunk_size: int = 4
    # Skip the expensive student [neurons, neurons] Jacobian backward passes
    # entirely.  Builds a minimal graph with neuron selection + DLA-only
    # adjacency, then runs ablation clustering directly on the model.  Valid
    # ONLY when graph_edge_weight == 0 (the supernode adjacency aggregated
    # without a real Jacobian is meaningless).  Roughly 2-3x faster student
    # graph construction.
    fast_student_graph: bool = False


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
    loss_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute graph loss for one prompt.

    In ``approx`` grad mode (default) returns a differentiable ``graph_loss``
    tensor; the caller is responsible for ``.backward()``.

    In ``true`` grad mode performs sequential per-supernode backward INSIDE
    this function (memory-bounded) and returns a DETACHED scalar tensor for
    logging.  The caller MUST NOT call backward on the returned tensor — its
    gradient has already been accumulated into model parameters.  Use the
    ``loss_scale`` arg to apply the same outer scaling (loss_scale / batch_denom)
    that approx mode would have applied externally.
    """
    # ------------------------------------------------------------------
    # Teacher side: either live attribution or pre-computed cache
    # ------------------------------------------------------------------
    if cached_teacher is not None:
        teacher_supergraph = cached_teacher.supergraph
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

    # ------------------------------------------------------------------
    # Student side: always live (trainable)
    # ------------------------------------------------------------------
    if config.verbose:
        print(f"  [graph] building student graph for prompt: {prompt!r}")
    # Force student to attribute over the same logit tokens as the teacher so that
    # prob-delta vectors occupy the same vocabulary positions in both models.
    #
    # ``fast_student_graph=True`` skips the [neurons, neurons] Jacobian
    # backward passes entirely (huge speedup), producing a graph with only
    # the [logits, neurons] DLA block.  This is sufficient for node-loss-only
    # mode because:
    #   - Neuron selection still happens (top-K residual-norm contributors).
    #   - Ablation clustering in build_super_graph runs forwards directly on
    #     the model, ignoring the adjacency block.
    #   - The supernode prob_deltas (used for matching + node loss) come from
    #     true ablation, also bypassing the adjacency.
    #   - Only the edge loss truly needs the Jacobian, so this path requires
    #     graph_edge_weight == 0.  We assert that here.
    if config.fast_student_graph and config.graph_edge_weight > 0.0:
        raise ValueError(
            "fast_student_graph=True is incompatible with graph_edge_weight > 0 "
            "(edge loss requires the [neurons, neurons] Jacobian block). "
            "Set --graph-edge-weight 0 or --fast-student-graph false."
        )
    student_graph = student_adapter.build_graph(
        prompt,
        attribution_targets=logit_token_ids.cpu() if logit_token_ids is not None else None,
        prop_neurons_per_layer=config.prop_neurons_per_layer,
        batch_size=config.student_graph_batch_size,
        dtype=config.graph_dtype,
        verbose=config.verbose,
        create_graph=False,
        detach_result=False,
        fast=config.fast_student_graph,
        # In fast mode, ``skip_logit_attribution`` is moot — the fast path
        # builds [logits, neurons] linearly (not via backward) anyway.  In
        # full mode, True saves the ~top_k_logits backward passes for the
        # logit rows; default True since current alignment uses real
        # ablation prob_deltas (not the Jacobian rows).
        skip_logit_attribution=config.student_skip_logit_attribution,
    )
    if config.align_diagnostic:
        print(f"  [grad-trace] student_graph.adjacency_matrix.requires_grad = {student_graph.adjacency_matrix.requires_grad}")

    student_prune_result = None
    # In fast mode the adjacency is DLA-only; prune_graph would prune almost
    # everything because neuron→neuron influence is uniformly zero.  Skip it.
    if config.graph_prune and not config.fast_student_graph:
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
    # Skip the (expensive, dense) supernode adjacency aggregation when no
    # edge term consumes it.  In fast_student_graph mode the underlying
    # neuron→neuron block is zero anyway; in any mode with edge_weight==0
    # the aggregated matrix is purely overhead.  Build a placeholder
    # SuperGraph with a 0×0 adjacency so downstream code that reads
    # ``.supernodes`` keeps working.
    if config.fast_student_graph or config.graph_edge_weight == 0.0:
        student_supergraph = SuperGraph(
            supernode_adjacency_matrix=torch.zeros(
                (
                    len(student_supergraph_structure.supernodes),
                    len(student_supergraph_structure.supernodes),
                ),
                device=student_graph.adjacency_device,
                dtype=student_graph.adjacency_matrix.dtype,
            ),
            supernodes=student_supergraph_structure.supernodes,
        )
    else:
        student_supergraph = _aggregate_supergraph_adjacency(
            student_graph,
            student_supergraph_structure.supernodes,
        )
    if config.align_diagnostic:
        print(f"  [grad-trace] student_supergraph.adjacency.requires_grad = {student_supergraph.supernode_adjacency_matrix.requires_grad}")
    # ----------------------------------------------------------------------
    # Two-signal split: principled matching + cheap differentiable gradient.
    #
    # The cached teacher ``supernode_prob_deltas`` are real hook-based
    # ablation prob-deltas in [-1, 1] (TransformerLens hooks zero out the
    # supernode's activations across all layers, let the network flow forward,
    # measure softmax probability change).  For cross-model alignment to be
    # meaningful the student vector must live in the same operational space.
    #
    # MATCHING SIGNAL (no grad): ``compute_supernode_ablation_prob_deltas``
    # runs the same TRUE hook-based ablation on the student.  No backward
    # pass is needed — this signal exists purely to compute cosine similarity
    # against the teacher's cached prob-deltas and produce a supernode
    # mapping.  Wrapped in ``torch.no_grad()`` so the ~5–6 chunked ablated
    # forwards don't accumulate activation memory.
    #
    # GRADIENT SIGNAL (with grad): ``compute_supernode_dla_approx_prob_deltas_with_grad``
    # is computed AFTER alignment, only when ``graph_node_weight > 0``.  It
    # captures the direct-residual-path component of each supernode's effect
    # via one grad-enabled forward + per-supernode (norm + lm_head + softmax),
    # which has tiny memory footprint compared to backproping through 40+
    # ablated forwards.  The faithfulness gap is the cross-layer corrections,
    # which are second-order — once supernodes are correctly matched, the
    # gradient just needs to point roughly toward the right direction.
    if student_graph.n_logits > 0 and len(student_supergraph_structure.supernodes) > 0:
        student_supernode_prob_deltas = (
            student_adapter.compute_supernode_ablation_prob_deltas(
                prompt=prompt,
                supernodes=student_supergraph_structure.supernodes,
                neuron_locations_t=student_graph.neuron_locations,
                logit_token_ids=student_graph.logit_token_ids,
                dtype=config.graph_dtype,
            )
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

    # ----------------------------------------------------------------------
    # Branch on graph_grad_mode:
    #
    #   "approx": one-shot DLA-approximated student prob-deltas → standard
    #             compute_graph_loss → return differentiable tensor for
    #             outer backward.
    #
    #   "true":   chunked sequential true-ablation backward (this function
    #             does its own backwards and returns a detached scalar).
    # ----------------------------------------------------------------------
    if config.graph_grad_mode == "true":
        if config.align_diagnostic:
            print(
                f"  [grad-trace] mapping size: {sum(len(v) for v in alignment.mapping.values())} edges, "
                f"teacher_supernodes={len(teacher_ids)}, student_supernodes={len(student_ids)} "
                f"[true-grad mode]"
            )

        # Edge loss: differentiable through student adjacency, single backward.
        edge_loss_value = 0.0
        if config.graph_edge_weight > 0.0:
            from graph_loss.loss import _compute_edge_loss
            W_T = teacher_supergraph.supernode_adjacency_matrix.detach().to(
                device=student_supergraph.supernode_adjacency_matrix.device,
                dtype=student_supergraph.supernode_adjacency_matrix.dtype,
            )
            edge_loss = _compute_edge_loss(
                W_T,
                student_supergraph.supernode_adjacency_matrix,
                alignment.mapping,
                teacher_ids,
                student_ids,
            )
            edge_term = config.graph_edge_weight * edge_loss
            edge_loss_value = float(edge_loss.detach().item())
            scaled_edge = loss_scale * edge_term
            if scaled_edge.requires_grad:
                scaled_edge.backward()
            del edge_term, scaled_edge

        # Node + focus losses: chunked sequential backward.
        node_loss_value, focus_loss_value = _run_true_grad_node_focus_pass(
            prompt=prompt,
            student_adapter=student_adapter,
            config=config,
            student_graph=student_graph,
            student_supergraph=student_supergraph,
            teacher_supergraph=teacher_supergraph,
            alignment=alignment,
            teacher_ids=teacher_ids,
            student_ids=student_ids,
            n_vocab=n_vocab,
            loss_scale=loss_scale,
        )

        graph_loss_total = (
            config.graph_edge_weight * edge_loss_value
            + config.graph_node_weight * node_loss_value
            + config.graph_focus_weight * focus_loss_value
        )

        if config.align_diagnostic:
            print(
                f"  [grad-trace] (true-grad) edge={edge_loss_value:.4f} "
                f"node={node_loss_value:.4f} focus={focus_loss_value:.6f} "
                f"total={graph_loss_total:.6f}"
            )

        loss_breakdown = {
            "edge_loss": edge_loss_value,
            "node_loss": node_loss_value,
            "graph_loss_total": graph_loss_total,
        }
        # Return a DETACHED scalar — caller checks requires_grad and skips
        # outer backward in true-grad mode.
        graph_loss = torch.tensor(
            graph_loss_total,
            device=student_supergraph.supernode_adjacency_matrix.device,
        )
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

    # ---- Node-loss target vectors (teacher side) -------------------------
    #
    # Use teacher's cached supernode_prob_deltas (real hook-based ablation
    # over the top-K logit_token_ids) scattered into full vocab.  This puts
    # teacher's node-loss target in the SAME operational space as the
    # student's DLA-approx prob_deltas computed below (both signed
    # post-softmax probability deltas).  Previously this used cached
    # ``dla_dict`` (linear DLA vectors), which lived in a different
    # functional space than the student's signal we now use, so MSE was
    # comparing apples to oranges.
    if config.graph_node_weight > 0.0:
        teacher_pd = teacher_supergraph.supernode_prob_deltas
        if teacher_pd is not None and teacher_pd.numel() > 0:
            logit_token_ids_t = (
                teacher_supergraph.logit_token_ids
                if teacher_supergraph.logit_token_ids is not None
                else logit_token_ids
            )
            if logit_token_ids_t is not None:
                lti = logit_token_ids_t.to(device=teacher_pd.device, dtype=torch.long)
                actual_vocab_t = max(
                    int(n_vocab),
                    int(lti.max().item()) + 1,
                )
                for cid in range(teacher_pd.shape[0]):
                    base = torch.zeros(
                        actual_vocab_t,
                        dtype=teacher_pd.dtype,
                        device=teacher_pd.device,
                    )
                    alignment.teacher_dla[cid] = base.scatter(
                        0, lti, teacher_pd[cid]
                    ).detach()

    # ---- Node-loss differentiable signal (student side) -----------------
    #
    # Compute student supernode prob-deltas via the cheap DLA-approximation
    # (one grad-enabled forward + per-supernode norm/lm_head/softmax).  This
    # is differentiable through gate/up/down_proj weights and all upstream
    # layers, with negligible activation memory beyond the single forward.
    #
    # Faithfulness gap: misses cross-layer corrections that the no-grad
    # true ablation (used for matching) captures.  But since matching is
    # done with the principled signal, the gradient just needs to point
    # roughly toward the matched teacher supernodes' prob-deltas.
    if config.graph_node_weight > 0.0 and len(student_supergraph.supernodes) > 0:
        student_pd_grad = (
            student_adapter.compute_supernode_dla_approx_prob_deltas_with_grad(
                prompt=prompt,
                supernodes=student_supergraph.supernodes,
                neuron_locations_t=student_graph.neuron_locations,
                logit_token_ids=student_graph.logit_token_ids,
                dtype=config.graph_dtype,
            )
        )  # [n_sn, n_logits], differentiable
        if student_pd_grad.numel() > 0:
            logit_token_ids_dev = student_graph.logit_token_ids.to(
                device=student_pd_grad.device,
                dtype=torch.long,
            )
            actual_vocab = max(
                int(n_vocab),
                int(logit_token_ids_dev.max().item()) + 1,
            )
            for sid in range(student_pd_grad.shape[0]):
                base = torch.zeros(
                    actual_vocab,
                    dtype=student_pd_grad.dtype,
                    device=student_pd_grad.device,
                )
                full_vec = base.scatter(0, logit_token_ids_dev, student_pd_grad[sid])
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
    # Logit-focus loss (no alignment required) — SIGNED version.
    #
    # Sum-over-supernodes signed prob-deltas on both sides.  Teacher comes
    # from cache.  Student needs grad → use the same DLA-approximated
    # prob-deltas computed for node loss when available, otherwise compute
    # them on demand (kept independent so focus can run with node_weight=0).
    focus_loss_value = 0.0
    if config.graph_focus_weight > 0.0 and logit_token_ids is not None:
        teacher_pd = teacher_supergraph.supernode_prob_deltas
        if teacher_pd is not None and teacher_pd.numel() > 0:
            # Reuse the differentiable per-supernode prob-deltas computed
            # for node loss if they exist; otherwise compute them now.
            if config.graph_node_weight > 0.0 and len(student_supergraph.supernodes) > 0:
                # Reconstruct from alignment.student_dla which holds full-vocab
                # differentiable vectors per supernode (set by the node-loss
                # block above).  Sum them then index back to the compact
                # logit_token_ids basis.
                lti = student_graph.logit_token_ids.to(
                    device=teacher_pd.device, dtype=torch.long
                )
                # alignment.student_dla maps sid -> full-vocab tensor (with grad)
                if alignment.student_dla:
                    full_sum = sum(alignment.student_dla.values())
                    student_focus_signed = full_sum.index_select(0, lti)
                else:
                    student_focus_signed = None
            else:
                student_pd_grad = (
                    student_adapter.compute_supernode_dla_approx_prob_deltas_with_grad(
                        prompt=prompt,
                        supernodes=student_supergraph.supernodes,
                        neuron_locations_t=student_graph.neuron_locations,
                        logit_token_ids=student_graph.logit_token_ids,
                        dtype=config.graph_dtype,
                    )
                )
                student_focus_signed = (
                    student_pd_grad.float().sum(0)
                    if student_pd_grad.numel() > 0
                    else None
                )

            if student_focus_signed is not None:
                teacher_focus_signed = teacher_pd.detach().float().sum(0)
                n_l = min(
                    teacher_focus_signed.shape[0],
                    student_focus_signed.shape[0],
                )
                t_focus = teacher_focus_signed[:n_l].to(
                    device=student_focus_signed.device,
                    dtype=student_focus_signed.dtype,
                )
                s_focus = student_focus_signed[:n_l]
                focus_loss = torch.nn.functional.mse_loss(s_focus, t_focus)
                focus_loss_value = float(focus_loss.item())
                if config.align_diagnostic:
                    print(
                        f"  [focus] teacher_signed_sum={float(t_focus.sum()):.4f} "
                        f"student_signed_sum={float(s_focus.sum().item()):.4f} "
                        f"|t|_inf={float(t_focus.abs().max()):.4f} "
                        f"|s|_inf={float(s_focus.abs().max().item()):.4f} "
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


def _run_true_grad_node_focus_pass(
    *,
    prompt: str,
    student_adapter: HFLlamaGraphAdapter,
    config: GraphAuxConfig,
    student_graph,
    student_supergraph: SuperGraph,
    teacher_supergraph: SuperGraph,
    alignment,
    teacher_ids: list[int],
    student_ids: list[int],
    n_vocab: int,
    loss_scale: float,
) -> tuple[float, float]:
    """Sequential per-chunk TRUE-ablation backward for node + focus losses.

    Mirrors the math of ``_compute_node_loss`` (best-matching student per
    teacher, normalized L2) and the signed focus loss, but processes student
    supernodes in chunks of ``config.graph_true_grad_chunk_size`` so that
    activation memory of grad-enabled ablation forwards stays bounded.

    Returns the (detached) scalar values of node_loss and focus_loss for
    logging.  Backward has already been applied to model parameters via
    chunked ``loss_term.backward()`` calls — caller must NOT re-backward.
    """
    import torch.nn.functional as F

    device = student_supergraph.supernode_adjacency_matrix.device
    n_logits = int(student_graph.logit_token_ids.numel())

    # ---- baseline (detached) -----------------------------------------
    baseline_probs = student_adapter.compute_baseline_probs(
        prompt, student_graph.logit_token_ids, dtype=config.graph_dtype,
    )

    # ---- teacher targets (compact basis, [n_logits]) ----------------
    teacher_pd = teacher_supergraph.supernode_prob_deltas
    teacher_logit_ids = (
        teacher_supergraph.logit_token_ids
        if teacher_supergraph.logit_token_ids is not None
        else student_graph.logit_token_ids
    )
    if (
        teacher_pd is None
        or teacher_pd.numel() == 0
        or teacher_logit_ids is None
    ):
        return 0.0, 0.0

    # Align teacher prob-deltas onto the student's logit_token_ids basis.
    # Build a mapping student_logit_idx -> teacher_logit_idx for shared tokens.
    s_lti = student_graph.logit_token_ids.to(device=device, dtype=torch.long)
    t_lti = teacher_logit_ids.to(device=device, dtype=torch.long)
    # For each student logit, find its position in teacher's basis (if any).
    # teacher_idx_for_student[i] = j s.t. t_lti[j] == s_lti[i], else -1.
    t_lookup = {int(tok.item()): j for j, tok in enumerate(t_lti)}
    teacher_idx = torch.tensor(
        [t_lookup.get(int(tok.item()), -1) for tok in s_lti],
        device=device, dtype=torch.long,
    )
    valid_mask = teacher_idx >= 0  # [n_logits]
    safe_teacher_idx = teacher_idx.clamp(min=0)
    # teacher_pd_compact[c, i] = teacher_pd[c, safe_teacher_idx[i]] if valid else 0
    teacher_pd_compact = teacher_pd.to(device=device, dtype=torch.float32).index_select(
        1, safe_teacher_idx
    )
    teacher_pd_compact = teacher_pd_compact * valid_mask.float().unsqueeze(0)

    # ---- pick winning student per teacher (no-grad approximation) ---
    # For each teacher supernode t, find the student supernode in mapping[t]
    # that minimises normalised L2 distance.  Uses no-grad student prob_deltas
    # already stored in student_supergraph.
    student_pd_no_grad = (
        student_supergraph.supernode_prob_deltas.to(device=device, dtype=torch.float32)
        if student_supergraph.supernode_prob_deltas is not None
        else None
    )

    node_loss_terms: dict[int, torch.Tensor] = {}  # sid -> teacher target [n_logits]
    if (
        config.graph_node_weight > 0.0
        and student_pd_no_grad is not None
        and student_pd_no_grad.numel() > 0
    ):
        # Per-teacher: pick best student via no-grad cosine distance.
        for t_id, s_ids in alignment.mapping.items():
            if not s_ids or t_id >= teacher_pd_compact.shape[0]:
                continue
            t_vec = F.normalize(teacher_pd_compact[t_id], dim=0)
            best_sid = None
            best_dist = None
            for s_id in s_ids:
                if s_id >= student_pd_no_grad.shape[0]:
                    continue
                s_vec = F.normalize(student_pd_no_grad[s_id], dim=0)
                d = (t_vec - s_vec).pow(2).sum()
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_sid = s_id
            if best_sid is not None:
                # Accumulate target for this winning sid (a sid can win for
                # multiple teachers — we average those teacher targets).
                if best_sid in node_loss_terms:
                    node_loss_terms[best_sid] = (
                        node_loss_terms[best_sid] + teacher_pd_compact[t_id]
                    ) / 1.0
                else:
                    node_loss_terms[best_sid] = teacher_pd_compact[t_id].clone()

    # ---- focus loss precomputation (no-grad) -----------------------
    focus_residual_no_grad: torch.Tensor | None = None
    if (
        config.graph_focus_weight > 0.0
        and student_pd_no_grad is not None
        and student_pd_no_grad.numel() > 0
    ):
        s_focus_no_grad = student_pd_no_grad.sum(0)  # [n_logits]
        t_focus = teacher_pd_compact.sum(0)
        # mse_loss reduces by mean over n_logits, so per-element gradient
        # multiplier is (2 / n_logits) * residual.
        focus_residual_no_grad = (
            (2.0 / max(n_logits, 1)) * (s_focus_no_grad - t_focus)
        ).detach()

    # ---- chunked sequential backward over student supernodes -------
    chunk_size = max(1, int(config.graph_true_grad_chunk_size))
    if focus_residual_no_grad is not None:
        # Focus loss is a sum over ALL supernodes — every member-bearing
        # supernode must be ablated so its gradient contributes.
        sids_to_process = [
            sid for sid, members in enumerate(student_supergraph.supernodes)
            if members
        ]
    else:
        # Node-only: ablate just winning students.  Saves compute when
        # only a fraction of students are matched targets.
        sids_to_process = sorted(node_loss_terms.keys())

    total_node_loss = 0.0

    for start in range(0, len(sids_to_process), chunk_size):
        chunk_sids = sids_to_process[start : start + chunk_size]
        chunk_supernodes = [student_supergraph.supernodes[sid] for sid in chunk_sids]

        student_chunk_pd = (
            student_adapter.compute_supernode_ablation_prob_deltas_chunk_with_grad(
                prompt=prompt,
                supernodes_chunk=chunk_supernodes,
                neuron_locations_t=student_graph.neuron_locations,
                logit_token_ids=student_graph.logit_token_ids,
                baseline_probs_detached=baseline_probs,
                dtype=config.graph_dtype,
            )
        )  # [B, n_logits], differentiable

        chunk_loss_terms: list[torch.Tensor] = []

        if config.graph_node_weight > 0.0:
            # Per-sid normalized-L2 loss against teacher target.
            n_terms = 0
            chunk_node_sum = torch.zeros((), device=device, dtype=torch.float32)
            for b, sid in enumerate(chunk_sids):
                if sid not in node_loss_terms:
                    continue
                t_vec = F.normalize(node_loss_terms[sid].detach(), dim=0)
                s_vec = F.normalize(student_chunk_pd[b].float(), dim=0)
                chunk_node_sum = chunk_node_sum + (t_vec - s_vec).pow(2).sum()
                n_terms += 1
            if n_terms > 0:
                # Match _compute_node_loss normalisation (sum / n_terms).  But
                # n_terms here is local to the chunk — to get the same total
                # gradient magnitude as the all-at-once formulation, we should
                # divide by len(node_loss_terms) (the global #winning students).
                global_n = max(len(node_loss_terms), 1)
                node_term = (chunk_node_sum / global_n) * config.graph_node_weight
                chunk_loss_terms.append(node_term)
                total_node_loss += float(node_term.detach().item()) / max(
                    config.graph_node_weight, 1e-12
                )

        if config.graph_focus_weight > 0.0 and focus_residual_no_grad is not None:
            # Linearised focus loss contribution: residual.detach() · Σ_chunk s_i.
            # d(mse(Σ s, t))/d(s_i) = (2/n_logits) * (Σ s - t).  Total grad
            # decomposes as Σ_chunk grad_chunk, so each chunk contributes
            # ``residual.detach() · d(Σ_{i ∈ chunk} s_i)/d(theta)`` to the
            # parameter gradient.  The actual loss VALUE is reported below
            # from the no-grad sum (cheaper than tracking partial sums here).
            chunk_s_sum = student_chunk_pd.float().sum(0)  # [n_logits]
            focus_contrib = (focus_residual_no_grad * chunk_s_sum).sum()
            chunk_loss_terms.append(focus_contrib * config.graph_focus_weight)

        if not chunk_loss_terms:
            del student_chunk_pd
            continue

        chunk_total = torch.stack(chunk_loss_terms).sum()
        scaled_chunk = loss_scale * chunk_total
        if scaled_chunk.requires_grad:
            scaled_chunk.backward()

        # Free this chunk's autograd graph before processing the next.
        del student_chunk_pd, chunk_loss_terms, chunk_total, scaled_chunk
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- exact focus loss for logging (recomputed from no-grad) ----
    focus_loss_value = 0.0
    if focus_residual_no_grad is not None and student_pd_no_grad is not None:
        s_focus_no_grad = student_pd_no_grad.sum(0)
        t_focus = teacher_pd_compact.sum(0)
        focus_loss_value = float(
            torch.nn.functional.mse_loss(s_focus_no_grad, t_focus).item()
        )

    return total_node_loss, focus_loss_value


def _load_cached_teacher(
    cache: TeacherDataCache,
    prompt: str,
    answer: int,
    device: torch.device,
) -> CachedTeacherPromptData:
    """Load one prompt's teacher artifacts from disk and reconstruct a SuperGraph.

    Cache-backed distillation should be all-or-nothing: a missing prompt or
    artifact means the pregenerated cache does not match this training run.
    """
    from graph_loss.graph import SuperGraph  # local import to avoid circular

    try:
        sg_data = cache.load_teacher_supergraph(prompt, answer)
    except (KeyError, FileNotFoundError) as e:
        raise RuntimeError(
            "Teacher data cache is enabled but required graph data is missing for "
            f"prompt={prompt!r}, answer={answer!r}. Regenerate the cache for this "
            "dataset/tokenizer or remove --teacher-data-cache."
        ) from e
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
        try:
            dla_data = cache.load_teacher_supernode_dla(prompt, answer)
        except (KeyError, FileNotFoundError) as e:
            raise RuntimeError(
                "Teacher data cache is enabled but required teacher DLA data is missing for "
                f"prompt={prompt!r}, answer={answer!r}. Regenerate the cache for this "
                "dataset/tokenizer or remove --teacher-data-cache."
            ) from e
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
    if teacher_cache is not None and answers is None:
        raise RuntimeError(
            "Teacher data cache is enabled but batch answers were not provided. "
            "Cannot safely load cached teacher graph data."
        )
    for i, prompt in enumerate(prompts):
        cached = (
            _load_cached_teacher(teacher_cache, prompt, answers[i], device)
            if teacher_cache is not None and answers is not None
            else None
        )
        if cached is None and teacher_graph_model is None:
            # No cache configured and no live teacher; skip graph loss for this prompt.
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

    if teacher_cache is not None and answers is None:
        raise RuntimeError(
            "Teacher data cache is enabled but batch answers were not provided. "
            "Cannot safely load cached teacher graph data."
        )

    metric_sums: dict[str, float] = {}
    detached_losses = []
    denom = float(len(prompts))
    graph_backward_prompts = 0
    gen_batch_size = max(1, int(config.graph_gen_batch_size))
    pending_losses: list[torch.Tensor] = []

    def flush_pending_losses() -> None:
        nonlocal graph_backward_prompts, pending_losses
        if not pending_losses:
            return
        scaled_loss = (loss_scale / denom) * torch.stack(pending_losses).sum()
        if scaled_loss.requires_grad:
            scaled_loss.backward()
            graph_backward_prompts += len(pending_losses)
        else:
            print("  [graph] WARN: generated graph losses have no grad; skipping backward")
        del scaled_loss
        pending_losses = []
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for i, prompt in enumerate(prompts):
        cached = (
            _load_cached_teacher(teacher_cache, prompt, answers[i], device)
            if teacher_cache is not None and answers is not None
            else None
        )
        if cached is None and teacher_graph_model is None:
            # No cache configured and no live teacher; skip graph loss for this prompt.
            continue
        # In true-grad mode, compute_prompt_graph_loss does its own backwards
        # internally (chunked, memory-bounded) and returns a detached scalar.
        # Pass the same per-prompt loss scaling it would have applied externally.
        prompt_loss, prompt_metrics = compute_prompt_graph_loss(
            prompt=prompt,
            teacher_graph_model=teacher_graph_model,
            student_adapter=student_adapter,
            config=config,
            cached_teacher=cached,
            loss_scale=loss_scale / denom,
        )
        detached_losses.append(prompt_loss.detach())
        if config.graph_grad_mode == "true":
            # Backward already done inside compute_prompt_graph_loss.
            graph_backward_prompts += 1
            del prompt_loss
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            pending_losses.append(prompt_loss)
            if len(pending_losses) >= gen_batch_size:
                flush_pending_losses()
        for key, value in prompt_metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)

    flush_pending_losses()

    if not detached_losses:
        return torch.tensor(0.0, device=device), {}

    loss = torch.stack(detached_losses).mean()
    metrics = {key: value / denom for key, value in metric_sums.items()}
    metrics["graph_prompts"] = float(len(prompts))
    metrics["graph_backward_prompts"] = float(graph_backward_prompts)
    return loss, metrics

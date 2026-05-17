"""Training-time graph auxiliary loss helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

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
    supergraph and logit token IDs were generated offline by
    ``generate_teacher_data.py`` and are simply loaded from disk each step.
    """

    supergraph: SuperGraph
    logit_token_ids: torch.Tensor | None


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
    graph_edge_weight: float = 0.0  # weight of edge structural loss within L_graph
    fast_teacher_graph: bool = False  # skip expensive TL/HF backward graph; linear logit block + proxy influence
    # Supergraph clustering params for the student (mirrors build_super_graph args)
    student_activation_forward_batch_size: int = 32
    # When True, student build_graph skips populating the [logits, neurons]
    # block of B (saves ~top_k_logits backward passes, halves peak memory).
    student_skip_logit_attribution: bool = True
    # Skip the expensive student [neurons, neurons] Jacobian backward passes
    # entirely.  Builds a minimal graph with neuron selection + DLA-only
    # adjacency.  Valid ONLY when graph_edge_weight == 0.  Roughly 2-3x faster
    # student graph construction.
    fast_student_graph: bool = False
    # Dataset path for student full_search clustering.
    student_dataset: str | None = None
    # Local path for caching student activation-write grids between steps.
    # Defaults to a temp dir; set to a persistent path to reuse across restarts.
    student_activation_write_cache_path: str | None = None
    # Radius around the prompt's target arg/sum values when building the
    # ANOVA basis masks for student full_search clustering.  Must match the
    # teacher-cache radius for label alignment to be meaningful.  0 = exact
    # match (old, fragile default); 5 is a sensible decade-width choice.
    student_anova_range_radius: int = 0
    # Cap on members per ANOVA-labelled student supernode.
    student_anova_nodes_per_label: int = 10
    student_sum_min_specificity: float = 0.0
    # Supergraph clustering method for the student. ``live_anova`` performs
    # per-prompt ANOVA over only the attribution-selected kept neurons and
    # assigns each to the argmax category (winner-take-all) — produces at
    # most ~8 disjoint supernodes that match the teacher cache.  ``full_search``
    # is the legacy per-category top-K labelling path.
    student_cluster_method: str = "live_anova"
    # Optional MLP-input cache (built via ``precompute_mlp_inputs``) for the
    # student model.  When set, ``build_super_graph`` forwards the loaded
    # cache to ``build_neuron_activation_write_result``, which recomputes
    # SwiGLU activations directly from the cached residual-stream input
    # instead of running a forward pass per prompt — the dominant cost
    # of student supergraph clustering once attribution is fast.
    student_mlp_input_cache_path: str | None = None


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
    """Compute graph loss for one prompt."""
    # ------------------------------------------------------------------
    # Teacher side: either live attribution or pre-computed cache
    # ------------------------------------------------------------------
    if cached_teacher is not None:
        teacher_supergraph = cached_teacher.supergraph
        logit_token_ids = cached_teacher.logit_token_ids
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

    supergraph_start = time.perf_counter()

    student_mlp_input_cache = None
    if (
        config.student_cluster_method in {"full_search", "live_anova"}
        and config.student_mlp_input_cache_path
        and config.student_dataset
    ):
        from graph_loss.neuron_activation_heatmap import _resolve_dataset_path
        from graph_loss.precompute_mlp_inputs import (
            load_mlp_input_cache,
            mlp_input_cache_exists,
        )

        dataset_path = _resolve_dataset_path(config.student_dataset)
        student_model_name = getattr(
            getattr(student_adapter.model, "config", None),
            "_name_or_path",
            "unknown_model",
        )
        if mlp_input_cache_exists(
            config.student_mlp_input_cache_path, student_model_name, dataset_path
        ):
            student_mlp_input_cache = load_mlp_input_cache(
                config.student_mlp_input_cache_path,
                student_model_name,
                dataset_path,
            )

    with torch.no_grad():
        student_supergraph_structure = build_super_graph(
            student_graph,
            student_adapter,
            prune_result=student_prune_result,
            activation_forward_batch_size=config.student_activation_forward_batch_size,
            dataset=config.student_dataset,
            activation_write_cache_path=config.student_activation_write_cache_path,
            mlp_input_cache=student_mlp_input_cache,
            cluster_method=config.student_cluster_method,
            anova_range_radius=config.student_anova_range_radius,
            anova_nodes_per_label=config.student_anova_nodes_per_label,
            sum_min_specificity=config.student_sum_min_specificity,
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
    # Alignment: match teacher and student supernodes by ANOVA label
    # ------------------------------------------------------------------
    if config.verbose:
        print("  [graph] aligning supernodes by ANOVA label")
    s_label_to_sid = {
        labels[0]: sid
        for sid, labels in enumerate(student_supergraph.supernode_labels or [])
        if labels
    }
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
        edge_weight=config.graph_edge_weight,
        node_weight=0.0,
    )

    metrics = {
        "teacher_supernodes": len(teacher_ids),
        "student_supernodes": len(student_ids),
        "teacher_graph_neurons": int(teacher_graph.n_neurons) if teacher_graph is not None else 0,
        "student_graph_neurons": int(student_graph.n_neurons),
        "aligned_teacher_supernodes": sum(1 for tid in teacher_ids if mapping.get(tid)),
        **loss_breakdown,
    }
    return graph_loss, metrics


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
        supernode_labels=sg_data.get("supernode_labels"),
    )
    return CachedTeacherPromptData(
        supergraph=supergraph,
        logit_token_ids=logit_token_ids,
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
        elif config.verbose:
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

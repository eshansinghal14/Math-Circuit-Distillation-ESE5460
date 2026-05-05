"""Training-time graph auxiliary loss helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from graph_loss.align import align_supernodes_prob_delta, compute_supernode_dla
from graph_loss.attribution.attribute import attribute
from graph_loss.graph import (
    SuperGraph,
    build_super_graph,
    extract_supernode_members,
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


def _aggregate_supergraph_adjacency(graph, supernodes: list[list[int]]) -> SuperGraph:
    """Aggregate a differentiable graph adjacency using fixed supernode membership."""
    adj_matrix_norm = normalize_matrix(graph.adjacency_matrix)
    num_supernodes = len(supernodes)
    supernode_adj_matrix = torch.zeros(
        num_supernodes,
        num_supernodes,
        dtype=adj_matrix_norm.dtype,
        device=adj_matrix_norm.device,
    )
    for t in range(num_supernodes):
        total_input = torch.abs(adj_matrix_norm[:, supernodes[t]]).sum(dim=0)
        internal_input = torch.abs(adj_matrix_norm[supernodes[t]][:, supernodes[t]]).sum(dim=0)
        frac_external = (total_input - internal_input) / total_input.clamp(min=1e-10)
        for s in range(num_supernodes):
            sum_A = adj_matrix_norm[supernodes[t]][:, supernodes[s]].sum(dim=1)
            supernode_adj_matrix[t, s] = (
                (frac_external * sum_A).sum(dim=0)
                / frac_external.sum(dim=0).clamp(min=1e-10)
            )
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
    # Always use fast=True for the student: the HF adapter doesn't support Jacobian
    # attribution, and fast mode (DLA-based) enables the pre-population shortcut in
    # build_super_graph so clustering never calls the TransformerLens-specific
    # compute_ablation_prob_deltas.
    student_graph = student_adapter.build_graph(
        prompt,
        attribution_targets=logit_token_ids.cpu() if logit_token_ids is not None else None,
        prop_neurons_per_layer=config.prop_neurons_per_layer,
        batch_size=config.student_graph_batch_size,
        dtype=config.graph_dtype,
        verbose=config.verbose,
        create_graph=False,
        detach_result=False,
        fast=True,
    )

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
    # Preserve prob_deltas so alignment gets non-zero cosine similarities.
    student_supergraph = student_supergraph._replace(
        supernode_prob_deltas=student_supergraph_structure.supernode_prob_deltas
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
        student_dla_with_grad_dict = student_adapter.compute_supernode_dlas_with_grad(
            prompt=prompt,
            supernodes=student_supergraph.supernodes,
            neuron_locations_t=student_graph.neuron_locations,
            n_vocab=n_vocab,
            dtype=config.graph_dtype,
        )
        for sid, dla_tensor in student_dla_with_grad_dict.items():
            alignment.student_dla[sid] = dla_tensor

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

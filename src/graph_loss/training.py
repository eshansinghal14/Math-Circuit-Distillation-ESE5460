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


@dataclass
class GraphAuxConfig:
    lambda_graph: float = 0.1
    graph_dtype: torch.dtype | None = None
    prop_neurons_per_layer: float = 0.1
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


def _aggregate_supergraph_adjacency(graph, supernodes: list[list[int]]) -> SuperGraph:
    """Aggregate a differentiable graph adjacency using fixed supernode membership.

    Uses torch.stack (out-of-place) instead of in-place setitem so that the
    gradient from the edge loss flows back through supernode_adjacency_matrix
    → adjacency_matrix → source_vectors_t → model parameters (down_proj.weight).
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
    student_adapter: HFLlamaGraphAdapter,
    config: GraphAuxConfig,
    cached_teacher: CachedTeacherPromptData,
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
            prop_neurons_per_layer=config.prop_neurons_per_layer,
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
            anova_range_radius=config.student_anova_range_radius,
            anova_nodes_per_label=config.student_anova_nodes_per_label,
            sum_min_specificity=config.student_sum_min_specificity,
            no_grad_supergraph=True,
        )
    except ValueError as e:
        raise RuntimeError(
            f"Student supergraph build failed for prompt={prompt!r}: {e}"
        ) from e

    student_graph = student_result.graph
    student_supergraph_structure = student_result.supergraph

    # Filter supernodes to only the requested labels (if specified).
    if config.student_graph_labels is not None:
        label_set = set(config.student_graph_labels)
        keep_indices = [
            i
            for i, labels in enumerate(student_supergraph_structure.supernode_labels or [])
            if labels and labels[0] in label_set
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
    # Alignment: match teacher and student supernodes by ANOVA label
    # ------------------------------------------------------------------
    if config.verbose:
        print("  [graph] aligning supernodes by ANOVA label")
    s_label_to_sid = {
        labels[0]: sid
        for sid, labels in enumerate(student_supergraph.supernode_labels or [])
        if labels
    }
    if config.student_graph_labels:
        missing = [lbl for lbl in config.student_graph_labels if lbl not in s_label_to_sid]
        if missing:
            raise RuntimeError(
                f"Student graph is missing supernodes for expected label(s): {missing}. "
                f"Student supernode labels present: {sorted(s_label_to_sid.keys())}."
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
    return CachedTeacherPromptData(
        supergraph=supergraph,
        logit_token_ids=logit_token_ids,
    )


def backward_batch_graph_loss(
    *,
    prompts: list[str],
    student_adapter: HFLlamaGraphAdapter,
    config: GraphAuxConfig,
    device: torch.device,
    loss_scale: float,
    teacher_cache: TeacherDataCache,
    answers: list[int],
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

    for i, prompt in enumerate(prompts):
        cached = _load_cached_teacher(teacher_cache, prompt, answers[i], device)
        prompt_loss, prompt_metrics = compute_prompt_graph_loss(
            prompt=prompt,
            student_adapter=student_adapter,
            config=config,
            cached_teacher=cached,
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

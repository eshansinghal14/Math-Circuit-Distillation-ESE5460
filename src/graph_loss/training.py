"""Training-time graph auxiliary loss helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from graph_loss.create_graph import build_shared_context, create_graph, create_graph_at_position
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
    tokens_dla_nodes: bool = False
    compare_n_tokens: int | None = None
    compare_token_selection: Literal["kl", "teacher_entropy"] = "kl"


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


# ---------------------------------------------------------------------------
# Multi-token graph loss (--compare-n-tokens path)
# ---------------------------------------------------------------------------

def _kl_per_position(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """KL(teacher || student) per token position. Returns [seq_len] float tensor."""
    t_probs = torch.softmax(teacher_logits / temperature, dim=-1)
    s_log_probs = torch.log_softmax(student_logits / temperature, dim=-1)
    return (t_probs * (t_probs.clamp(min=1e-10).log() - s_log_probs)).sum(dim=-1)


def _entropy_per_position(
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Shannon entropy of the (temperature-scaled) teacher distribution per
    token position. Returns [seq_len] float tensor. Lower = teacher more
    confident (a more decisive computation step)."""
    t_log_probs = torch.log_softmax(teacher_logits / temperature, dim=-1)
    t_probs = t_log_probs.exp()
    return -(t_probs * t_log_probs).sum(dim=-1)


def compute_prompt_graph_loss_compare_tokens(
    *,
    input_ids: torch.Tensor,
    response_start_idx: int,
    teacher_adapter: HFLlamaGraphAdapter,
    student_adapter: HFLlamaGraphAdapter,
    config: GraphAuxConfig,
    n_tokens: int,
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Graph loss for one example using the compare-n-tokens strategy.

    Selects the top-``n_tokens`` response positions by KL divergence, builds
    shared token supernodes once, then computes DLA + attribution + graph loss
    at each selected position and averages the result.

    Args:
        input_ids: [seq_len] tokenized full sequence (prompt + response).
        response_start_idx: Index of the first response token in input_ids.
        teacher_adapter: Adapter wrapping the (frozen) teacher model.
        student_adapter: Adapter wrapping the (trainable) student model.
        config: GraphAuxConfig with tokens_dla_nodes=True and compare_n_tokens set.
        n_tokens: Number of top-KL positions to compute graph loss on.
        teacher_logits: [seq_len, vocab] teacher logits (detached, pre-computed).
        student_logits: [seq_len, vocab] student logits (detached, pre-computed).

    Returns:
        (averaged graph loss tensor, metrics dict)
    """
    seq_len = input_ids.numel()
    # Determine valid response positions (response tokens only, no padding).
    # A position is valid if input_ids has a non-pad token at position+1 up to seq_len.
    response_end_idx = seq_len  # KL computed up to but not including the padding region.
    # Trim trailing padding: treat token id 0 (or pad_token_id) as padding if any.
    pad_id = getattr(teacher_adapter.tokenizer, "pad_token_id", None)
    if pad_id is not None:
        non_pad = (input_ids != pad_id).nonzero(as_tuple=False)
        if non_pad.numel() > 0:
            response_end_idx = int(non_pad[-1].item()) + 1

    # Response positions: [response_start_idx, response_end_idx).
    response_positions = list(range(response_start_idx, response_end_idx))
    if not response_positions:
        device = student_adapter.device
        return torch.tensor(0.0, device=device, requires_grad=False), {"compare_tokens_skipped": 1.0}

    # Select N response positions to compute graph loss on.
    n_select = min(n_tokens, len(response_positions))
    if config.compare_token_selection == "teacher_entropy":
        # Lowest teacher entropy = most confident / decisive computation steps.
        # Student-independent (no moving target across training steps).
        entropy_vals = _entropy_per_position(
            teacher_logits[response_positions],
            temperature=config.temperature,
        )
        top_local = torch.topk(entropy_vals, n_select, largest=False).indices.tolist()
        selection_metric_key = "compare_tokens_selection_entropy"
    else:
        # Default: top-N positions by HIGHEST teacher-student KL divergence.
        kl_vals = _kl_per_position(
            teacher_logits[response_positions],
            student_logits[response_positions],
            temperature=config.temperature,
        )
        top_local = torch.topk(kl_vals, n_select).indices.tolist()
        selection_metric_key = "compare_tokens_selection_kl"
    selected_positions = [response_positions[i] for i in top_local]

    # Build shared contexts (one forward pass + token supernodes each).
    with torch.no_grad():
        teacher_shared = build_shared_context(
            teacher_adapter,
            input_ids,
            prop_neurons_per_layer=config.prop_neurons_per_layer,
            dtype=config.graph_dtype,
            dataset=config.dataset,
            mlp_input_cache=None,
            anova_nodes_per_label=config.student_anova_nodes_per_label,
            anova_range_radius=config.student_anova_range_radius,
            sum_min_specificity=config.student_sum_min_specificity,
            node_labels=config.student_graph_labels or None,
            include_arg_nodes=True,
            batch_size=config.teacher_graph_batch_size,
        )

    student_shared = build_shared_context(
        student_adapter,
        input_ids,
        prop_neurons_per_layer=config.prop_neurons_per_layer,
        dtype=config.graph_dtype,
        dataset=config.dataset,
        mlp_input_cache=config.mlp_input_cache,
        anova_nodes_per_label=config.student_anova_nodes_per_label,
        anova_range_radius=config.student_anova_range_radius,
        sum_min_specificity=config.student_sum_min_specificity,
        node_labels=config.student_graph_labels or None,
        include_arg_nodes=True,
        batch_size=config.student_graph_batch_size,
    )

    # For each selected position, compute graphs and graph loss.
    position_losses: list[torch.Tensor] = []
    metric_sums: dict[str, float] = {}

    for pos in selected_positions:
        pos_teacher_logits = teacher_logits[pos].to(device=teacher_adapter.device)

        with torch.no_grad():
            teacher_result = create_graph_at_position(
                teacher_shared,
                target_position=pos,
                include_dla_node=True,
                dla_model_logits=pos_teacher_logits,
                top_k_logits=config.top_k_logits,
                temperature=config.temperature,
                batch_size=config.teacher_graph_batch_size,
                build_create_graph=False,
                detach_result=True,
                skip_logit_attribution=False,
                no_grad_supergraph=True,
                verbose=config.verbose,
            )

        student_result = create_graph_at_position(
            student_shared,
            target_position=pos,
            include_dla_node=True,
            dla_model_logits=pos_teacher_logits,
            top_k_logits=config.top_k_logits,
            temperature=config.temperature,
            batch_size=config.student_graph_batch_size,
            build_create_graph=False,
            detach_result=False,
            skip_logit_attribution=False,
            no_grad_supergraph=True,
            verbose=config.verbose,
        )

        teacher_supergraph = teacher_result.supergraph
        student_graph = student_result.graph
        student_supergraph_structure = student_result.supergraph

        # Re-aggregate student supergraph with gradient flow.
        student_supergraph = _aggregate_supergraph_adjacency(
            student_graph,
            student_supergraph_structure.supernodes,
        )
        student_supergraph = student_supergraph._replace(
            supernode_labels=student_supergraph_structure.supernode_labels,
        )

        # Align by label and compute loss.
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
        student_label_set = set(s_label_to_sid.keys())
        teacher_label_set = set(t_label_to_tid.keys())
        extra_in_teacher = teacher_label_set - student_label_set
        missing_from_teacher = student_label_set - teacher_label_set
        if extra_in_teacher or missing_from_teacher:
            parts = []
            if extra_in_teacher:
                parts.append(f"  teacher has extra supernodes: {sorted(extra_in_teacher)}")
            if missing_from_teacher:
                parts.append(f"  teacher missing supernodes:   {sorted(missing_from_teacher)}")
            raise RuntimeError(
                f"Teacher/student supernode label mismatch at position {pos}.\n"
                + "\n".join(parts)
                + f"\n  Student: {sorted(student_label_set)}"
                + f"\n  Teacher: {sorted(teacher_label_set)}"
            )

        mapping = {
            tid: {s_label_to_sid[labels[0]]}
            for tid, labels in enumerate(teacher_supergraph.supernode_labels or [])
            if labels and labels[0] in s_label_to_sid
        }
        teacher_ids = list(range(len(teacher_supergraph.supernodes)))
        student_ids = list(range(len(student_supergraph.supernodes)))

        pos_loss, pos_breakdown = compute_graph_loss(
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
        position_losses.append(pos_loss)
        for key, value in pos_breakdown.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)

    if not position_losses:
        device = student_adapter.device
        return torch.tensor(0.0, device=device, requires_grad=False), {}

    avg_loss = torch.stack(position_losses).mean()
    n_pos = float(len(position_losses))
    metrics = {key: value / n_pos for key, value in metric_sums.items()}
    metrics["compare_tokens_n_selected"] = n_pos
    metrics[selection_metric_key] = 1.0
    return avg_loss, metrics


def backward_batch_graph_loss_compare_tokens(
    *,
    prompts: list[str],
    input_ids_batch: list[torch.Tensor],
    response_start_indices: list[int],
    teacher_adapter: HFLlamaGraphAdapter,
    student_adapter: HFLlamaGraphAdapter,
    config: GraphAuxConfig,
    device: torch.device,
    loss_scale: float,
    teacher_logits_batch: torch.Tensor,
    student_logits_batch: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute and backprop compare-n-tokens graph loss one example at a time.

    Like backward_batch_graph_loss() but uses the live teacher/student adapters
    and per-position KL divergence to select which tokens to compute graphs on.
    """
    if not prompts:
        return torch.tensor(0.0, device=device), {}

    import gc

    n_tokens = config.compare_n_tokens
    assert n_tokens is not None

    metric_sums: dict[str, float] = {}
    detached_losses: list[torch.Tensor] = []
    denom = float(len(prompts))
    graph_backward_prompts = 0

    for i, (input_ids, response_start_idx) in enumerate(zip(input_ids_batch, response_start_indices)):
        t_logits = teacher_logits_batch[i]
        s_logits = student_logits_batch[i]

        prompt_loss, prompt_metrics = compute_prompt_graph_loss_compare_tokens(
            input_ids=input_ids.to(device),
            response_start_idx=response_start_idx,
            teacher_adapter=teacher_adapter,
            student_adapter=student_adapter,
            config=config,
            n_tokens=n_tokens,
            teacher_logits=t_logits.to(device),
            student_logits=s_logits.to(device),
        )
        detached_losses.append(prompt_loss.detach())
        scaled_loss = (loss_scale / denom) * prompt_loss
        if scaled_loss.requires_grad:
            scaled_loss.backward()
            graph_backward_prompts += 1
        del scaled_loss
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for key, value in prompt_metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)

    loss = torch.stack(detached_losses).mean() if detached_losses else torch.tensor(0.0, device=device)
    metrics = {key: value / denom for key, value in metric_sums.items()}
    metrics["graph_prompts"] = float(len(prompts))
    metrics["graph_backward_prompts"] = float(graph_backward_prompts)
    return loss, metrics

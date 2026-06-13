"""Training-time graph auxiliary loss helpers."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch

from graph_loss.create_graph import GraphPipelineResult, create_graph
from graph_loss.graph import SuperGraph, normalize_matrix
from graph_loss.hf_adapter import HFLlamaGraphAdapter
from graph_loss.loss import compute_graph_loss


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
    student_nodes_per_label: int = 10
    teacher_nodes_per_label: int = 10
    mlp_input_cache: dict | None = None
    teacher_mlp_input_cache: dict | None = None
    activation_write_result_cache: dict = field(default_factory=dict)
    graph_loss_type: Literal["jsd", "kld", "mse", "mse-norm", "mse-scale"] = "jsd"
    graph_labels: list[str] | None = None
    compare_n_tokens: int | None = None
    compare_ans_token: bool = False
    extract_fn: Callable | None = None


def generate_teacher_sample(
    adapter: HFLlamaGraphAdapter,
    prompt: str,
    *,
    top_k_logits: float | None = 0.95,
    temperature: float = 2.0,
    prop_neurons_per_layer: float = 0.1,
    batch_size: int = 256,
    verbose: bool = False,
    mlp_input_cache: dict | None = None,
    model_name: str | None = None,
    nodes_per_label: int = 10,
    anova_range_radius: int = 0,
    node_labels: list[str] | None = None,
    logger: logging.Logger | None = None,
) -> GraphPipelineResult:
    """Generate teacher graph for one sample in-memory."""
    _logger = logger or logging.getLogger(__name__)

    with torch.enable_grad():
        result: GraphPipelineResult = create_graph(
            adapter,
            prompt,
            top_k_logits=top_k_logits,
            temperature=temperature,
            prop_neurons_per_layer=prop_neurons_per_layer,
            batch_size=batch_size,
            verbose=verbose,
            mlp_input_cache=mlp_input_cache,
            model_name=model_name,
            nodes_per_label=nodes_per_label,
            anova_range_radius=anova_range_radius,
            node_labels=node_labels,
            no_grad_supergraph=True,
            build_create_graph=False,
            detach_result=True,
            logger=_logger,
        )

    return result


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
    prompt: str | torch.Tensor,
    student_adapter: HFLlamaGraphAdapter,
    config: GraphAuxConfig,
    teacher_result: GraphPipelineResult,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute graph loss for one prompt using a pre-built teacher graph."""
    # ------------------------------------------------------------------
    # Teacher side: always from the pre-built result
    # ------------------------------------------------------------------
    teacher_supergraph = teacher_result.supergraph
    logit_token_ids = teacher_result.graph.logit_token_ids
    if logit_token_ids is not None:
        logit_token_ids = logit_token_ids.to(student_adapter.cfg.device)
    if config.verbose:
        print(
            f"  [graph] teacher supergraph ready: "
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
            mlp_input_cache=config.mlp_input_cache,
            node_labels=config.graph_labels or [],
            anova_range_radius=config.student_anova_range_radius,
            nodes_per_label=config.student_nodes_per_label,
            dla_model_logits=teacher_result.target_position_logits,
            no_grad_supergraph=True,
        )
    except ValueError as e:
        raise RuntimeError(
            f"Student supergraph build failed for prompt={prompt!r}: {e}"
        ) from e

    student_graph = student_result.graph
    student_supergraph_structure = student_result.supergraph

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
            + "\nEnsure teacher and student are built with the same supernode flags."
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


def _kl_per_position(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """KL(teacher || student) per token position. Returns [seq_len] float tensor."""
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
    from graph_loss.create_graph import create_graph

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

    response_positions = [
        pos for pos in range(response_start, response_end)
    ]
    if not response_positions:
        raise RuntimeError(
            f"compare_n_tokens: no non-EOS response tokens for prompt={prompt!r}"
        )

    logit_positions = [pos - 1 for pos in response_positions]
    if config.compare_ans_token:
        extract_fn = config.extract_fn
        if extract_fn is None:
            from utils import parse_response
            extract_fn = lambda text: parse_response(text, "")
        # Include the last prompt token so "= <answer>" patterns are visible to extract_fn.
        context_start = max(0, response_start - 1)
        full_resp = tokenizer.decode(
            input_ids[context_start:response_end].tolist(), skip_special_tokens=True
        )
        answer_str = extract_fn(full_resp)
        selected_positions = None
        if answer_str is not None:
            response_ids = input_ids[response_start:response_end].tolist()
            cand_ids = tokenizer(answer_str, add_special_tokens=False)["input_ids"]
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
            t_logits = teacher_adapter.model(input_ids.unsqueeze(0)).logits.squeeze(0).cpu()
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
                node_labels=config.graph_labels or [],
                mlp_input_cache=config.teacher_mlp_input_cache,
                no_grad_supergraph=True,
                build_create_graph=False,
                detach_result=True,
                verbose=config.verbose,
            )

        # Pass prefix_ids directly so the student tokenizes from the same IDs
        # as the teacher — avoids decode→re-tokenize round-trip instability.
        pos_loss, pos_metrics = compute_prompt_graph_loss(
            prompt=prefix_ids,
            student_adapter=student_adapter,
            config=config,
            teacher_result=teacher_result,
        )
        if config.verbose:
            token_str = tokenizer.decode([input_ids[pos].item()])
            print(f"      [graph] graph {graph_idx + 1}/{n_select} built (token {pos}: {token_str!r})", flush=True)

        scaled_pos_loss = (loss_scale / denom / n_select) * pos_loss
        if scaled_pos_loss.requires_grad:
            scaled_pos_loss.backward()
        detached_losses.append(pos_loss.detach())
        del scaled_pos_loss, teacher_result, pos_loss
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
    """Compute and backprop graph loss one prompt at a time using a live teacher adapter.

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
        if config.compare_n_tokens is not None:
            # Backward is done per-position inside _compare_tokens_loss_for_prompt.
            prompt_loss, prompt_metrics = _compare_tokens_loss_for_prompt(
                prompt=prompt,
                answer=answers[i],
                teacher_adapter=teacher_adapter,
                student_adapter=student_adapter,
                config=config,
                device=device,
                loss_scale=loss_scale,
                denom=denom,
            )
            detached_losses.append(prompt_loss)  # already detached
            graph_backward_prompts += 1
        else:
            teacher_result = generate_teacher_sample(
                teacher_adapter,
                prompt,
                prop_neurons_per_layer=config.teacher_prop_neurons_per_layer,
                top_k_logits=config.top_k_logits,
                temperature=config.temperature,
                batch_size=config.teacher_graph_batch_size,
                verbose=config.verbose,
                node_labels=config.graph_labels,
                mlp_input_cache=config.teacher_mlp_input_cache,
                nodes_per_label=config.teacher_nodes_per_label,
            )
            prompt_loss, prompt_metrics = compute_prompt_graph_loss(
                prompt=prompt,
                student_adapter=student_adapter,
                config=config,
                teacher_result=teacher_result,
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
        if on_prompt_done is not None:
            on_prompt_done(i + 1, len(prompts))
        for key, value in prompt_metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)

    loss = torch.stack(detached_losses).mean()
    metrics = {key: value / denom for key, value in metric_sums.items()}
    metrics["graph_prompts"] = float(len(prompts))
    metrics["graph_backward_prompts"] = float(graph_backward_prompts)
    return loss, metrics

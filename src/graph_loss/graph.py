"""Graph container and influence utilities for LLaMA neuron attribution."""

from __future__ import annotations

import logging
import math
import os
from typing import NamedTuple

import torch
import torch.nn.functional as F

from graph_loss.anova_node_labels import (
    ANOVA_LABEL_CATEGORIES,
    build_anova_basis_rules,
    label_activation_heatmaps,
    parse_numeric_args,
)
from graph_loss.attribution.targets import LogitTarget
from graph_loss.neuron_activation_heatmap import (
    build_neuron_activation_write_result,
    save_supernode_activation_heatmap_pdf,
)
from graph_loss.utils import (
    ActivationWriteResult,
    UnifiedConfig,
    activation_write_cache_file,
    convert_nnsight_config_to_transformerlens,
    load_activation_write_cache,
    safe_cache_segment,
    save_activation_write_cache,
)


class Graph:
    input_string: str
    input_tokens: torch.Tensor
    logit_targets: list[LogitTarget]
    neuron_locations: torch.Tensor
    adjacency_matrix: torch.Tensor
    neuron_activations: torch.Tensor
    logit_probabilities: torch.Tensor
    vocab_size: int
    cfg: UnifiedConfig
    n_pos: int
    attribution_mode: str
    neuron_write_vectors: torch.Tensor | None

    def __init__(
        self,
        input_string: str,
        input_tokens: torch.Tensor,
        neuron_locations: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        cfg,
        neuron_activations: torch.Tensor,
        logit_targets: list[LogitTarget],
        logit_probabilities: torch.Tensor,
        vocab_size: int | None = None,
        *,
        attribution_mode: str = "full",
        neuron_write_vectors: torch.Tensor | None = None,
    ):
        """Container for neuron/token/logit attribution graphs.

        Nodes are stored in the order:
        ``[neurons, token_embeddings, logits]``.
        Rows represent target nodes and columns represent source nodes.
        """

        self.logit_targets = logit_targets
        self.logit_probabilities = logit_probabilities
        self.vocab_size = vocab_size if vocab_size is not None else cfg.d_vocab

        self.input_string = input_string
        self.adjacency_matrix = adjacency_matrix
        self.cfg = convert_nnsight_config_to_transformerlens(cfg)
        self.n_pos = len(input_tokens)
        self.neuron_locations = neuron_locations
        self.input_tokens = input_tokens
        self.neuron_activations = neuron_activations
        self.attribution_mode = attribution_mode
        self.neuron_write_vectors = neuron_write_vectors

    @property
    def n_neurons(self) -> int:
        return len(self.neuron_locations)

    @property
    def n_tokens(self) -> int:
        return len(self.input_tokens)

    @property
    def n_logits(self) -> int:
        return len(self.logit_targets)

    @property
    def n_nodes(self) -> int:
        return self.adjacency_matrix.shape[0]

    @property
    def adjacency_shape(self) -> tuple[int, int]:
        return tuple(self.adjacency_matrix.shape)

    @property
    def adjacency_device(self):
        return self.adjacency_matrix.device

    def to(self, device):
        self.adjacency_matrix = self.adjacency_matrix.to(device)
        self.neuron_locations = self.neuron_locations.to(device)
        self.neuron_activations = self.neuron_activations.to(device)
        self.logit_probabilities = self.logit_probabilities.to(device)
        if self.neuron_write_vectors is not None:
            self.neuron_write_vectors = self.neuron_write_vectors.to(device)

    @property
    def logit_token_ids(self) -> torch.Tensor:
        return torch.tensor(
            [target.vocab_idx for target in self.logit_targets],
            dtype=torch.long,
            device=self.logit_probabilities.device,
        )

    def to_pt(self, path: str):
        data = {
            "input_string": self.input_string,
            "cfg": self.cfg,
            "neuron_locations": self.neuron_locations,
            "logit_targets": self.logit_targets,
            "logit_probabilities": self.logit_probabilities,
            "vocab_size": self.vocab_size,
            "input_tokens": self.input_tokens,
            "neuron_activations": self.neuron_activations,
            "adjacency_layout": "dense",
            "adjacency_matrix": self.adjacency_matrix,
        }
        if self.attribution_mode != "full":
            data["attribution_mode"] = self.attribution_mode
        if self.neuron_write_vectors is not None:
            data["neuron_write_vectors"] = self.neuron_write_vectors
        torch.save(data, path)

    @staticmethod
    def from_pt(path: str, map_location="cpu") -> "Graph":
        data = torch.load(path, weights_only=False, map_location=map_location)
        return Graph(
            input_string=data["input_string"],
            input_tokens=data["input_tokens"],
            neuron_locations=data["neuron_locations"],
            adjacency_matrix=data["adjacency_matrix"],
            cfg=data["cfg"],
            neuron_activations=data["neuron_activations"],
            logit_targets=data["logit_targets"],
            logit_probabilities=data["logit_probabilities"],
            vocab_size=data.get("vocab_size"),
            attribution_mode=data.get("attribution_mode", "full"),
            neuron_write_vectors=data.get("neuron_write_vectors"),
        )

    def apply_prune_result(self, prune_result: "PruneResult") -> "Graph":
        """Returns a new Graph with edges and nodes zeroed out according to the PruneResult masks."""
        adjacency_matrix = self.adjacency_matrix.clone()
        
        effective_edge_mask = (
            prune_result.edge_mask 
            & prune_result.node_mask[:, None] 
            & prune_result.node_mask[None, :]
        )
        
        adjacency_matrix[~effective_edge_mask] = 0.0
        
        return Graph(
            input_string=self.input_string,
            input_tokens=self.input_tokens,
            neuron_locations=self.neuron_locations,
            adjacency_matrix=adjacency_matrix,
            cfg=self.cfg, # type: ignore
            neuron_activations=self.neuron_activations,
            logit_targets=self.logit_targets,
            logit_probabilities=self.logit_probabilities,
            vocab_size=self.vocab_size,
            attribution_mode=self.attribution_mode,
            neuron_write_vectors=self.neuron_write_vectors,
        )

def normalize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    math_dtype = (
        torch.float32
        if matrix.dtype in (torch.float16, torch.bfloat16)
        else matrix.dtype
    )
    normalized = matrix.to(dtype=math_dtype).abs()
    return normalized / normalized.sum(dim=1, keepdim=True).clamp(min=1e-10)


def normalize_signed_matrices(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    math_dtype = (
        torch.float32
        if matrix.dtype in (torch.float16, torch.bfloat16)
        else matrix.dtype
    )
    signed = matrix.to(dtype=math_dtype)
    positive = signed.clamp(min=0)
    negative = signed.clamp(max=0).abs()
    positive = positive / positive.sum(dim=1, keepdim=True).clamp(min=1e-10)
    negative = negative / negative.sum(dim=1, keepdim=True).clamp(min=1e-10)
    return positive, negative


def compute_influence(
    A: torch.Tensor,
    logit_weights: torch.Tensor,
    max_iter: int = 1000,
    atol: float | None = None,
):
    if atol is None:
        atol = 1e-6 if A.dtype in (torch.float16, torch.bfloat16) else 0.0
    logit_weights = logit_weights.to(device=A.device, dtype=A.dtype)
    current_influence = logit_weights @ A
    influence = current_influence
    iterations = 0
    while current_influence.abs().amax().item() > atol:
        if iterations >= max_iter:
            raise RuntimeError(
                f"Influence computation failed to converge after {iterations} iterations "
                f"(max residual={current_influence.abs().amax().item():.6g}, atol={atol:.6g})"
            )
        current_influence = current_influence @ A
        influence += current_influence
        iterations += 1
    return influence


def compute_node_influence(adjacency_matrix: torch.Tensor, logit_weights: torch.Tensor):
    positive, negative = normalize_signed_matrices(adjacency_matrix)
    positive_influence = compute_influence(positive, logit_weights)
    negative_influence = compute_influence(negative, logit_weights)
    return positive_influence - negative_influence


def compute_fast_proxy_node_influence(graph: Graph) -> torch.Tensor:
    """Per-node scalar influence when ``attribution_mode == 'fast'`` (no full Jacobian graph).

    Neurons ranked by residual write-vector norm; tokens and logits get uniform / prob weights
    so pruning keeps them, matching ``prune_graph`` expectations.
    """
    device = graph.adjacency_device
    dtype = graph.adjacency_matrix.dtype
    n_neurons = graph.n_neurons
    n_tokens = graph.n_tokens
    n_logits = graph.n_logits
    n_nodes = graph.n_nodes
    out = torch.zeros(n_nodes, device=device, dtype=dtype)
    nw = graph.neuron_write_vectors
    if nw is not None and n_neurons > 0:
        out[:n_neurons] = nw.to(device=device, dtype=dtype).norm(dim=-1)
    token_start = n_neurons
    logit_start = n_neurons + n_tokens
    if n_tokens > 0:
        out[token_start:logit_start] = torch.ones(n_tokens, device=device, dtype=dtype)
    if n_logits > 0:
        out[logit_start : logit_start + n_logits] = graph.logit_probabilities.to(
            device=device, dtype=dtype,
        ).clamp(min=torch.finfo(dtype).eps)
    return out


def compute_neuron_logit_influence(graph: Graph) -> torch.Tensor:
    """Return direct graph attribution from each neuron to each selected logit."""
    logit_start = graph.n_neurons + graph.n_tokens
    logit_rows = slice(logit_start, logit_start + graph.n_logits)
    return graph.adjacency_matrix[logit_rows, : graph.n_neurons].transpose(0, 1)


def compute_edge_influence(pruned_matrix: torch.Tensor, logit_weights: torch.Tensor):
    positive, negative = normalize_signed_matrices(pruned_matrix)
    normalized_pruned = positive + negative
    pruned_influence = compute_influence(positive, logit_weights) - compute_influence(
        negative,
        logit_weights,
    )
    pruned_influence += logit_weights
    edge_scores = normalized_pruned * pruned_influence.abs()[:, None]
    return edge_scores


def find_threshold(scores: torch.Tensor, threshold: float):
    sorted_scores = torch.sort(scores, descending=True).values
    cumulative_score = torch.cumsum(sorted_scores, dim=0) / torch.sum(sorted_scores)
    threshold_index = int(torch.searchsorted(cumulative_score, threshold).item())
    threshold_index = min(threshold_index, len(cumulative_score) - 1)
    return sorted_scores[threshold_index]


def find_elbow(values: torch.Tensor | list[float]) -> int:
    """Return the 0-based elbow index using max distance from the endpoint line."""
    values_t = torch.as_tensor(values, dtype=torch.float32).flatten()
    if values_t.numel() <= 2:
        return int(max(values_t.numel() - 1, 0))

    x = torch.linspace(0.0, 1.0, steps=int(values_t.numel()))
    y_min = values_t.min()
    y_range = (values_t.max() - y_min).clamp(min=1e-12)
    y = (values_t - y_min) / y_range
    points = torch.stack([x, y], dim=1)
    start = points[0]
    end = points[-1]
    line = end - start
    line_norm = line.norm().clamp(min=1e-12)
    distances = torch.abs(
        line[0] * (start[1] - points[:, 1]) - (start[0] - points[:, 0]) * line[1]
    ) / line_norm
    return int(torch.argmax(distances).item())


class PruneResult(NamedTuple):
    node_mask: torch.Tensor
    edge_mask: torch.Tensor
    cumulative_scores: torch.Tensor


def prune_graph(
    graph: Graph,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
) -> PruneResult:
    """Prune low-influence neurons while always retaining token and logit nodes."""

    if node_threshold > 1.0 or node_threshold < 0.0:
        raise ValueError("node_threshold must be between 0.0 and 1.0")
    if edge_threshold > 1.0 or edge_threshold < 0.0:
        raise ValueError("edge_threshold must be between 0.0 and 1.0")

    n_logits = graph.n_logits
    n_tokens = graph.n_tokens
    n_neurons = graph.n_neurons
    token_start = n_neurons

    adjacency_matrix = graph.adjacency_matrix
    logit_weights = torch.zeros(
        adjacency_matrix.shape[0],
        dtype=adjacency_matrix.dtype,
        device=adjacency_matrix.device,
    )
    logit_weights[-n_logits:] = graph.logit_probabilities.to(
        device=adjacency_matrix.device,
        dtype=adjacency_matrix.dtype,
    )

    if getattr(graph, "attribution_mode", "full") == "fast":
        node_influence = compute_fast_proxy_node_influence(graph)
    else:
        node_influence = compute_node_influence(adjacency_matrix, logit_weights)
    node_scores = node_influence.abs()
    node_mask = node_scores >= find_threshold(node_scores, node_threshold)
    node_mask[token_start:] = True

    pruned_matrix = adjacency_matrix.clone()
    pruned_matrix[~node_mask] = 0
    pruned_matrix[:, ~node_mask] = 0

    if getattr(graph, "attribution_mode", "full") == "fast":
        edge_scores = pruned_matrix.abs()
    else:
        edge_scores = compute_edge_influence(pruned_matrix, logit_weights)
    edge_mask = edge_scores >= find_threshold(edge_scores.flatten(), edge_threshold)

    old_node_mask = node_mask.clone()
    node_mask[:n_neurons] &= edge_mask[:, :n_neurons].any(0)
    node_mask[:n_neurons] &= edge_mask[:n_neurons].any(1)

    while not torch.all(node_mask == old_node_mask):
        old_node_mask[:] = node_mask
        edge_mask[~node_mask] = False
        edge_mask[:, ~node_mask] = False

        node_mask[:n_neurons] &= edge_mask[:, :n_neurons].any(0)
        node_mask[:n_neurons] &= edge_mask[:n_neurons].any(1)

    sorted_scores, sorted_indices = torch.sort(node_scores, descending=True)
    cumulative_scores = torch.cumsum(sorted_scores, dim=0) / torch.sum(sorted_scores)
    final_scores = torch.zeros_like(node_influence)
    final_scores[sorted_indices] = cumulative_scores

    return PruneResult(node_mask, edge_mask, final_scores)


class SuperGraph(NamedTuple):
    supernode_adjacency_matrix: torch.Tensor
    supernodes: list[list[int]]       # old node ids inside each new node
    node_labels: dict[int, list[str]] | None = None
    supernode_labels: list[list[str]] | None = None
    supernode_heatmap_pdf_paths: list[str] | None = None


def build_super_graph(
    graph: Graph,
    model,
    prune_result: PruneResult | None = None,
    dataset: str | None = None,
    activation_forward_batch_size: int = 32,
    mlp_input_cache: dict | None = None,
    model_name: str | None = None,
    supernode_heatmap_output_dir: str | None = None,
    anova_nodes_per_label: int = 10,
    anova_range_radius: int = 0,
    sum_min_specificity: float = 0.0,
    strict: bool = True,
) -> SuperGraph:
    """Cluster kept neurons into supernodes via per-category top-K ANOVA labeling.

    Each kept neuron can appear in multiple supernodes (one per category it
    scores well on), producing the overlapping supernode set used for heatmap
    inspection and distillation.

    ``mlp_input_cache`` (built via ``graph_loss.precompute_mlp_inputs``) is
    forwarded to ``build_neuron_activation_write_result`` whenever a fresh
    activation grid is built.  Cached prompts skip the per-step forward
    pass; for the kept-neuron grid the SwiGLU activation is recomputed
    directly from the cached residual-stream input via ``silu(x @ W_gate.T)
    * (x @ W_up.T)`` using the *current* model weights (so weights that
    have moved during distillation are still respected).
    """

    if prune_result is not None:
        graph = graph.apply_prune_result(prune_result)

    if anova_nodes_per_label <= 0:
        raise ValueError("anova_nodes_per_label must be positive")
    if anova_range_radius < 0:
        raise ValueError("anova_range_radius must be non-negative")

    n_neurons = graph.n_neurons
    logger = logging.getLogger(__name__)
    kept_neuron_mask = torch.ones(n_neurons, dtype=torch.bool, device=graph.adjacency_device)
    if prune_result is not None:
        kept_neuron_mask = prune_result.node_mask[:n_neurons].to(
            device=graph.adjacency_device,
            dtype=torch.bool,
        )

    adjacency_matrix = graph.adjacency_matrix
    logit_weights = torch.zeros(
        adjacency_matrix.shape[0],
        dtype=adjacency_matrix.dtype,
        device=adjacency_matrix.device,
    )
    if graph.n_logits:
        logit_weights[-graph.n_logits:] = graph.logit_probabilities.to(
            device=adjacency_matrix.device,
            dtype=adjacency_matrix.dtype,
        )
    if getattr(graph, "attribution_mode", "full") == "fast":
        node_influence = compute_fast_proxy_node_influence(graph).detach().float().cpu()
    else:
        node_influence = compute_node_influence(adjacency_matrix, logit_weights).detach().float().cpu()

    def format_member_locations(members: list[int]) -> str:
        locations = graph.neuron_locations.detach().cpu()
        formatted = []
        for graph_neuron_idx in members:
            layer = int(locations[graph_neuron_idx, 0].item())
            token_pos = int(locations[graph_neuron_idx, 1].item())
            neuron_id = int(locations[graph_neuron_idx, 2].item())
            formatted.append(
                f"{graph_neuron_idx}:(layer={layer}, token={token_pos}, neuron={neuron_id})"
            )
        return ", ".join(formatted)

    def supernode_influence(members: list[int]) -> float:
        if not members:
            return 0.0
        return float(node_influence[torch.tensor(members, dtype=torch.long)].sum().item())

    def number_token_unembed_values(
        members: list[int],
        *,
        min_value: int = 0,
        max_value: int = 200,
    ) -> dict[int, tuple[list[int], torch.Tensor]]:
        number_values = list(range(int(min_value), int(max_value) + 1))
        token_ids: list[int | None] = []
        for value in number_values:
            encoded = model.tokenizer(
                str(value),
                add_special_tokens=False,
                return_tensors=None,
            )
            input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
            if input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            token_ids.append(int(input_ids[0]) if len(input_ids) == 1 else None)

        valid_positions = [idx for idx, token_id in enumerate(token_ids) if token_id is not None]
        if not valid_positions:
            return {
                int(member): (
                    number_values,
                    torch.full((len(number_values),), float("nan"), dtype=torch.float32),
                )
                for member in members
            }

        valid_token_ids = torch.tensor(
            [int(token_ids[idx]) for idx in valid_positions],
            dtype=torch.long,
            device=model.cfg.device,
        )
        W_U = model.unembed.W_U.detach().to(device=model.cfg.device)
        W_U_numbers = W_U[:, valid_token_ids]
        locations = graph.neuron_locations.detach().cpu().to(dtype=torch.long)
        out: dict[int, tuple[list[int], torch.Tensor]] = {}
        w_out_cache: dict[int, torch.Tensor] = {}
        for member in members:
            layer = int(locations[member, 0].item())
            neuron_id = int(locations[member, 2].item())
            if layer not in w_out_cache:
                old_mlp = model.blocks[layer].mlp.old_mlp
                w_out_cache[layer] = model._row_oriented_weight(
                    old_mlp.W_out.to(device=model.cfg.device, dtype=W_U.dtype)
                )
            values = torch.full((len(number_values),), float("nan"), dtype=torch.float32)
            projected = (w_out_cache[layer][neuron_id] @ W_U_numbers).detach().float().cpu()
            values[torch.tensor(valid_positions, dtype=torch.long)] = projected
            out[int(member)] = (number_values, values)
        return out

    def number_token_dla_cosine_scores(
        members: list[int],
        *,
        target_value: int,
        units: bool,
        min_value: int = 0,
        max_value: int = 200,
    ) -> dict[int, float]:
        number_values = list(range(int(min_value), int(max_value) + 1))
        token_ids: list[int | None] = []
        for value in number_values:
            encoded = model.tokenizer(
                str(value),
                add_special_tokens=False,
                return_tensors=None,
            )
            input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
            if input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            token_ids.append(int(input_ids[0]) if len(input_ids) == 1 else None)

        valid_positions = [idx for idx, token_id in enumerate(token_ids) if token_id is not None]
        if not valid_positions:
            return {int(member): 0.0 for member in members}

        basis = torch.zeros(len(valid_positions), dtype=torch.float32)
        target_unit = int(target_value) % 10
        for basis_idx, number_idx in enumerate(valid_positions):
            number_value = number_values[number_idx]
            if (number_value % 10 == target_unit) if units else (number_value == int(target_value)):
                basis[basis_idx] = 1.0
        if float(basis.norm().item()) == 0.0:
            return {int(member): 0.0 for member in members}

        valid_token_ids = torch.tensor(
            [int(token_ids[idx]) for idx in valid_positions],
            dtype=torch.long,
            device=model.cfg.device,
        )
        W_U = model.unembed.W_U.detach().to(device=model.cfg.device)
        W_U_numbers = W_U[:, valid_token_ids]
        basis = basis.to(device=model.cfg.device, dtype=W_U.dtype)
        locations = graph.neuron_locations.detach().cpu().to(dtype=torch.long)
        activations = graph.neuron_activations.detach().to(device=model.cfg.device, dtype=W_U.dtype)
        scores: dict[int, float] = {}
        w_out_cache: dict[int, torch.Tensor] = {}
        for member in members:
            layer = int(locations[member, 0].item())
            neuron_id = int(locations[member, 2].item())
            if layer not in w_out_cache:
                old_mlp = model.blocks[layer].mlp.old_mlp
                w_out_cache[layer] = model._row_oriented_weight(
                    old_mlp.W_out.to(device=model.cfg.device, dtype=W_U.dtype)
                )
            neuron_activation = activations[member]
            unembed_projection = w_out_cache[layer][neuron_id] @ W_U_numbers
            dla = neuron_activation * unembed_projection
            score = F.cosine_similarity(dla.unsqueeze(0), basis.unsqueeze(0), dim=1).item()
            scores[int(member)] = float(score)
        return scores

    kept_neuron_indices_device = torch.where(kept_neuron_mask)[0]
    kept_neuron_indices = kept_neuron_indices_device.detach().cpu()
    logger.info("  Kept neurons for supergraph: %d", int(kept_neuron_indices.numel()))
    activation_write_result_for_kept: ActivationWriteResult | None = None
    node_labels: dict[int, list[str]] = {}
    supernode_labels: list[list[str]] | None = None
    supernode_heatmap_pdf_paths: list[str] | None = None
    supernodes: list[list[int]] = []

    def get_activation_write_result_for_kept() -> ActivationWriteResult:
        nonlocal activation_write_result_for_kept
        if activation_write_result_for_kept is not None:
            return activation_write_result_for_kept

        kept_neuron_locations = graph.neuron_locations[kept_neuron_indices_device].detach().cpu()

        resolved_model_name = model_name or getattr(model.cfg, "model_name", "model")
        if dataset is None:
            raise ValueError(
                "A dataset is required to build activation-write matrices. "
                "Pass --dataset or --activation-write-cache-path."
            )
        logger.info(
            "  Building activation-write matrices for %d kept graph neurons from dataset: %s",
            int(kept_neuron_locations.shape[0]),
            dataset,
        )
        activation_write_result_for_kept = build_neuron_activation_write_result(
            model,
            dataset,
            kept_neuron_locations,
            forward_batch_size=activation_forward_batch_size,
            mlp_input_cache=mlp_input_cache,
        )

        return activation_write_result_for_kept

    if kept_neuron_indices.numel():
        activation_write_result = get_activation_write_result_for_kept()
        target_args = parse_numeric_args(graph.input_string)
        label_results = label_activation_heatmaps(
            activation_write_result.activations,
            activation_write_result.arg_values,
            target_args=target_args,
            anova_range_radius=anova_range_radius,
        )
        selected_member_ids: set[int] = set()
        supernodes = []
        supernode_labels = []
        supernode_heatmaps = []
        target_sum = int(target_args[0] + target_args[1]) if len(target_args) >= 2 else 0
        kept_member_list = [int(member) for member in kept_neuron_indices.tolist()]

        for category in ANOVA_LABEL_CATEGORIES:
            if category in {"sum range", "sum units"}:
                sum_cosine_scores = number_token_dla_cosine_scores(
                    kept_member_list,
                    target_value=target_sum,
                    units=category == "sum units",
                )
                spec_threshold = sum_min_specificity if category == "sum range" else 0.0
                all_scored_rows = [
                    (
                        row_idx,
                        sum_cosine_scores[int(kept_neuron_indices[row_idx].item())],
                    )
                    for row_idx, label_result in enumerate(label_results)
                    if category in label_result.category_scores
                    and label_result.category_scores[category] > 0.0
                    and label_result.category_specificity.get(category, float("-inf"))
                    > spec_threshold
                ]
            else:
                all_scored_rows = [
                    (row_idx, label_result.category_specificity[category])
                    for row_idx, label_result in enumerate(label_results)
                    if category in label_result.category_specificity
                    and label_result.category_scores.get(category, 0.0) > 0.0
                ]
            sorted_all_rows = sorted(all_scored_rows, key=lambda item: item[1], reverse=True)
            scored_rows = sorted_all_rows
            if not scored_rows:
                if strict:
                    if category in {"sum range", "sum units"}:
                        pre_filter = [
                            (row_idx, sum_cosine_scores[int(kept_neuron_indices[row_idx].item())],
                             label_result.category_specificity.get(category, float("-inf")))
                            for row_idx, label_result in enumerate(label_results)
                            if category in label_result.category_scores
                            and label_result.category_scores[category] > 0.0
                        ]
                        top_spec = sorted(pre_filter, key=lambda x: x[2], reverse=True)[:5]
                        if category == "sum range":
                            raise ValueError(
                                f"Student ANOVA category {category!r} has no nodes "
                                f"(sum_min_specificity={sum_min_specificity}). "
                                f"Candidates before specificity filter: {len(pre_filter)}. "
                                f"Top specificities: {[(round(s,6), round(c,4)) for _,c,s in top_spec]}"
                            )
                        else:
                            raise ValueError(
                                f"Student ANOVA category {category!r} has no nodes with positive specificity. "
                                f"Candidates before specificity filter: {len(pre_filter)}. "
                                f"Top specificities: {[(round(s,6), round(c,4)) for _,c,s in top_spec]}"
                            )
                    raise ValueError(
                        f"Student ANOVA category {category!r} has no positive-variance nodes."
                    )
                if category == "sum range":
                    logger.info(
                        "  ANOVA label %s: no positive-variance nodes above sum_min_specificity=%.6g",
                        category,
                        sum_min_specificity,
                    )
                elif category == "sum units":
                    logger.info("  ANOVA label %s: no positive-variance nodes with positive specificity", category)
                else:
                    logger.info("  ANOVA label %s: no positive-variance nodes", category)
                continue
            keep_count = min(int(anova_nodes_per_label), len(scored_rows))
            top_rows = scored_rows[:keep_count]
            row_indices = torch.tensor([row_idx for row_idx, _score in top_rows], dtype=torch.long)
            members = [int(kept_neuron_indices[row_idx].item()) for row_idx, _score in top_rows]
            cluster_heatmaps = activation_write_result.activations[row_indices].detach().float()
            member_plot_labels: dict[int, list[str]] = {}
            member_specificity = {
                int(kept_neuron_indices[row_idx].item()): float(score)
                for row_idx, score in sorted_all_rows
            }
            for member in members:
                row_idx = int(torch.where(kept_neuron_indices == member)[0][0].item())
                label = label_results[row_idx].categories[category]
                ranking_score = member_specificity[member]
                if category in {"sum range", "sum units"}:
                    specificity_score = label_results[row_idx].category_specificity.get(category, 0.0)
                    member_plot_labels[member] = [f"{label} (spec={specificity_score:.3f}, cos={ranking_score:.3f})"]
                else:
                    variance_score = label_results[row_idx].category_scores[category]
                    member_plot_labels[member] = [f"{label} (var={variance_score:.3f}, spec={ranking_score:.3f})"]
                node_labels.setdefault(member, [])
                if label not in node_labels[member]:
                    node_labels[member].append(label)
                selected_member_ids.add(member)
            member_number_unembed = (
                number_token_unembed_values(members)
                if category in {"sum range", "sum units"}
                else None
            )

            supernodes.append(members)
            supernode_labels.append([category])
            supernode_heatmaps.append(
                (
                    cluster_heatmaps,
                    activation_write_result.arg_values,
                    members,
                    category,
                    member_plot_labels,
                    member_number_unembed,
                    member_specificity,
                )
            )
            logger.info(
                "  ANOVA label %s: selected_nodes=%d/%d cap=%d best_%s=%.6g",
                category,
                len(members),
                len(scored_rows),
                anova_nodes_per_label,
                "cos" if category in {"sum range", "sum units"} else "specificity",
                float(top_rows[0][1]),
            )

        logger.info(
            "  ANOVA labeling: selected_unique_neurons=%d/%d",
            len(selected_member_ids),
            int(kept_neuron_indices.numel()),
        )

        for supernode_idx, members in enumerate(supernodes):
            labels = (
                ", ".join(supernode_labels[supernode_idx])
                if supernode_labels is not None and supernode_idx < len(supernode_labels)
                else "none"
            )
            logger.info(
                "  supernode %d labels=%s influence=%.6g neuron locations: %s",
                supernode_idx,
                labels,
                supernode_influence(members),
                format_member_locations(members),
            )

        if supernode_heatmap_output_dir is not None:
            supernode_heatmap_pdf_paths = []
            for supernode_idx, (
                activation_grid,
                heatmap_arg_values,
                members,
                title,
                member_plot_labels,
                member_number_unembed,
                member_specificity,
            ) in enumerate(supernode_heatmaps):
                sn_title = f"supernode {supernode_idx}: {title}"
                saved_path = save_supernode_activation_heatmap_pdf(
                    activation_grid,
                    heatmap_arg_values,
                    members,
                    graph.neuron_locations.detach().cpu(),
                    output_path=os.path.join(
                        supernode_heatmap_output_dir,
                        f"supernode_{supernode_idx}.pdf",
                    ),
                    title=sn_title,
                    member_labels=member_plot_labels,
                    member_number_unembed=member_number_unembed,
                    member_specificity=member_specificity,
                )
                logger.info("  Saved supernode heatmap PDF: %s", saved_path)
                supernode_heatmap_pdf_paths.append(saved_path)

    logger.info("  Aggregating supergraph")
    adj_matrix_norm = normalize_matrix(adjacency_matrix)
    num_supernodes = len(supernodes)
    supernode_adj_matrix = torch.zeros(
        num_supernodes,
        num_supernodes,
        dtype=adj_matrix_norm.dtype,
        device=adjacency_matrix.device,
    )
    for t in range(num_supernodes):
        target_members = supernodes[t]
        total_input = torch.abs(adj_matrix_norm[:, target_members]).sum(dim=0)
        internal_input = torch.abs(adj_matrix_norm[target_members][:, target_members]).sum(dim=0)
        frac_external = (total_input - internal_input) / total_input.clamp(min=1e-10)
        
        for s in range(num_supernodes):
            source_members = supernodes[s]
            sum_A = adj_matrix_norm[target_members][:, source_members].sum(dim=1)
            supernode_adj_matrix[t, s] = (frac_external * sum_A).sum(dim=0) / frac_external.sum(dim=0).clamp(min=1e-10)

    return SuperGraph(
        supernode_adjacency_matrix=supernode_adj_matrix,
        supernodes=supernodes,
        node_labels=node_labels,
        supernode_labels=supernode_labels,
        supernode_heatmap_pdf_paths=supernode_heatmap_pdf_paths,
    )



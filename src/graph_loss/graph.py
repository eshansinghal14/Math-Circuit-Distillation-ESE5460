"""Graph container and influence utilities for LLaMA neuron attribution."""

from __future__ import annotations

import hashlib
import importlib
import logging
import math
import os
from typing import NamedTuple

import torch
import torch.nn.functional as F

from graph_loss.attribution.targets import LogitTarget
from graph_loss.neuron_activation_heatmap import (
    build_neuron_activation_write_result,
    save_supernode_activation_heatmap_pdf,
)
from graph_loss.utils import (
    ActivationWriteResult,
    UnifiedConfig,
    convert_nnsight_config_to_transformerlens,
    safe_cache_segment,
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
    supernode_prob_deltas: torch.Tensor | None = None
    all_supernode_prob_delta_norms: torch.Tensor | None = None
    prob_delta_elbow_index: int | None = None
    logit_token_ids: torch.Tensor | None = None


def build_super_graph(
    graph: Graph,
    model,
    prune_result: PruneResult | None = None,
    cossim_eps: float = 0.1,
    embedding_sigma: float = 1.5,
    embedding_eps: float | None = None,
    computation_sigma: float = 1.5,
    computation_eps: float | None = None,
    dataset: str | None = None,
    activation_forward_batch_size: int = 32,
    activation_write_cache_path: str | None = None,
    model_name: str | None = None,
    cluster_method: str = "full_search",
    supernode_heatmap_output_dir: str | None = None,
) -> SuperGraph:
    """Cluster kept neurons into numeric-token embedding and final-token computation supernodes."""

    if prune_result is not None:
        graph = graph.apply_prune_result(prune_result)

    if embedding_eps is None:
        embedding_eps = cossim_eps
    if computation_eps is None:
        computation_eps = cossim_eps
    if embedding_eps < 0.0 or computation_eps < 0.0:
        raise ValueError("embedding_eps and computation_eps must be non-negative")
    if embedding_sigma < 0.0 or computation_sigma < 0.0:
        raise ValueError("embedding_sigma and computation_sigma must be non-negative")
    if cluster_method not in {"full_search", "ablation"}:
        raise ValueError("cluster_method must be one of: full_search, ablation")

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

    def decoded_prompt_tokens() -> list[str]:
        token_ids = graph.input_tokens.detach().cpu().flatten().tolist()
        return [model.tokenizer.decode([int(token_id)]) for token_id in token_ids]

    def spatial_activation_similarity(activation_maps: torch.Tensor, sigma: float) -> torch.Tensor:
        if len(activation_maps) == 0:
            raise ValueError("spatial_activation_similarity requires at least one activation map")
        if len(activation_maps) == 1:
            return torch.ones((1, 1), dtype=torch.float32)

        try:
            gaussian_filter = importlib.import_module("scipy.ndimage").gaussian_filter
        except ImportError as exc:
            raise ImportError(
                "scipy is required for Gaussian-smoothed supergraph clustering"
            ) from exc

        activation_maps = torch.nan_to_num(activation_maps.detach().float().cpu())
        smoothed = torch.empty_like(activation_maps)
        activation_arrays = activation_maps.numpy()
        for row_idx in range(activation_maps.shape[0]):
            smoothed[row_idx] = torch.from_numpy(
                gaussian_filter(activation_arrays[row_idx], sigma=float(sigma))
            )

        flat_maps = smoothed.flatten(start_dim=1)
        flat_maps = F.normalize(flat_maps, p=2, dim=1, eps=1e-12)
        return flat_maps @ flat_maps.T

    def unembedding_write_similarity(w_down_vectors: torch.Tensor) -> torch.Tensor:
        if len(w_down_vectors) == 0:
            raise ValueError("unembedding_write_similarity requires at least one vector")
        if len(w_down_vectors) == 1:
            return torch.ones((1, 1), dtype=torch.float32)

        W_U = model.unembed.W_U.detach()
        projected_logits = (
            w_down_vectors.to(device=W_U.device, dtype=W_U.dtype)
            @ W_U
        ).detach().float().cpu()
        projected_logits = F.normalize(projected_logits, p=2, dim=1, eps=1e-12)
        return projected_logits @ projected_logits.T

    def angular_distance_connected_components(distance_matrix: torch.Tensor, eps: float) -> torch.Tensor:
        if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError(
                f"Expected a square distance matrix, got {tuple(distance_matrix.shape)}"
            )
        if distance_matrix.shape[0] == 0:
            raise ValueError("angular_distance_connected_components requires at least one point")

        connected = distance_matrix <= float(eps)
        connected.fill_diagonal_(True)
        labels = torch.full((distance_matrix.shape[0],), -1, dtype=torch.long)
        cluster_id = 0
        for start_idx in range(distance_matrix.shape[0]):
            if labels[start_idx] >= 0:
                continue
            stack = [start_idx]
            labels[start_idx] = cluster_id
            while stack:
                current_idx = stack.pop()
                neighbors = torch.where(connected[current_idx] & (labels < 0))[0].tolist()
                for neighbor_idx in neighbors:
                    labels[neighbor_idx] = cluster_id
                    stack.append(int(neighbor_idx))
            cluster_id += 1
        return labels

    def cluster_phase(
        *,
        phase_name: str,
        phase_rows: torch.Tensor,
        activation_maps: torch.Tensor,
        w_down_vectors: torch.Tensor,
        sigma: float,
        eps: float,
    ) -> torch.Tensor:
        if phase_rows.numel() == 0:
            logger.info("  %s clustering: no neurons", phase_name)
            return torch.empty(0, dtype=torch.long)

        activation_similarity = spatial_activation_similarity(activation_maps, sigma)
        write_similarity = unembedding_write_similarity(w_down_vectors)
        combined_similarity = (activation_similarity * write_similarity).clamp(min=-1.0, max=1.0)
        distance_matrix = torch.arccos(combined_similarity) / math.pi
        assignments = angular_distance_connected_components(distance_matrix, eps)
        unique_assignments, cluster_sizes = torch.unique(assignments, return_counts=True)
        logger.info(
            "  %s clustering: sigma=%.6g eps=%.6g neurons=%d clusters=%d min_size=%d max_size=%d",
            phase_name,
            float(sigma),
            float(eps),
            int(phase_rows.numel()),
            int(unique_assignments.numel()),
            int(cluster_sizes.min().item()),
            int(cluster_sizes.max().item()),
        )
        return assignments

    @torch.no_grad()
    def compute_ablation_prob_deltas(neuron_indices: torch.Tensor) -> torch.Tensor:
        if graph.n_logits == 0:
            return torch.empty((neuron_indices.numel(), 0), dtype=torch.float32)
        if activation_forward_batch_size <= 0:
            raise ValueError("activation_forward_batch_size must be positive")

        input_ids = graph.input_tokens.to(model.cfg.device)
        target_token_ids = graph.logit_token_ids.to(device=input_ids.device)
        if target_token_ids.numel() and int(target_token_ids.max().item()) >= model.cfg.d_vocab:
            raise ValueError("Ablation clustering only supports real vocabulary logit targets")

        baseline_logits = model(input_ids)
        baseline_probs = torch.softmax(baseline_logits[0, -1], dim=-1)[target_token_ids]

        device = input_ids.device
        dtype = baseline_logits.dtype
        selected_locations = graph.neuron_locations[neuron_indices].to(device=device)
        selected_activations = graph.neuron_activations[neuron_indices].to(
            device=device,
            dtype=dtype,
        )
        source_vectors = torch.empty(
            neuron_indices.numel(),
            model.cfg.d_model,
            device=device,
            dtype=dtype,
        )
        w_out_cache = {}
        for local_idx in range(neuron_indices.numel()):
            layer = int(selected_locations[local_idx, 0].item())
            neuron_id = int(selected_locations[local_idx, 2].item())
            if layer not in w_out_cache:
                old_mlp = model.blocks[layer].mlp.old_mlp
                w_out_cache[layer] = model._row_oriented_weight(
                    old_mlp.W_out.to(device=device, dtype=dtype)
                )
            source_vectors[local_idx] = selected_activations[local_idx] * w_out_cache[layer][neuron_id]

        deltas = torch.empty(
            neuron_indices.numel(),
            graph.n_logits,
            dtype=torch.float32,
            device="cpu",
        )
        total_batches = math.ceil(neuron_indices.numel() / activation_forward_batch_size)

        def make_ablation_hook(
            batch_indices: torch.Tensor,
            positions: torch.Tensor,
            vectors: torch.Tensor,
        ):
            def hook_fn(acts: torch.Tensor, hook):
                acts_out = acts.clone()
                acts_out[batch_indices, positions] -= vectors.to(device=acts.device, dtype=acts.dtype)
                return acts_out

            return hook_fn

        for batch_idx, start in enumerate(
            range(0, neuron_indices.numel(), activation_forward_batch_size),
            start=1,
        ):
            end = min(start + activation_forward_batch_size, neuron_indices.numel())
            batch_locations = selected_locations[start:end]
            batch_vectors = source_vectors[start:end]
            hooks = []
            for layer in batch_locations[:, 0].unique().tolist():
                layer_idx = int(layer)
                layer_mask = batch_locations[:, 0] == layer_idx
                batch_indices = torch.where(layer_mask)[0].to(device=device)
                hooks.append(
                    (
                        f"blocks.{layer_idx}.{model.feature_output_hook}",
                        make_ablation_hook(
                            batch_indices,
                            batch_locations[layer_mask, 1].to(device=device),
                            batch_vectors[layer_mask],
                        ),
                    )
                )

            ablated_logits = model.run_with_hooks(
                input_ids.expand(end - start, -1),
                fwd_hooks=hooks,
            )
            ablated_probs = torch.softmax(ablated_logits[:, -1], dim=-1)[:, target_token_ids]
            deltas[start:end] = (baseline_probs.unsqueeze(0) - ablated_probs).detach().float().cpu()

            logger.info(
                "  ablation clustering batch %d/%d: neurons %d-%d",
                batch_idx,
                total_batches,
                int(neuron_indices[start].item()),
                int(neuron_indices[end - 1].item()),
            )

        return deltas

    @torch.no_grad()
    def compute_supernode_ablation_prob_deltas(supernode_members: list[list[int]]) -> torch.Tensor:
        if graph.n_logits == 0:
            return torch.empty((len(supernode_members), 0), dtype=torch.float32)
        if activation_forward_batch_size <= 0:
            raise ValueError("activation_forward_batch_size must be positive")
        if not supernode_members:
            return torch.empty((0, graph.n_logits), dtype=torch.float32)

        model_device = getattr(model, "device", None)
        if model_device is None and hasattr(model, "cfg"):
            model_device = model.cfg.device
        model_d_vocab = getattr(model, "d_vocab", None)
        if model_d_vocab is None and hasattr(model, "cfg"):
            model_d_vocab = model.cfg.d_vocab

        input_ids = graph.input_tokens.to(model_device)
        target_token_ids = graph.logit_token_ids.to(device=input_ids.device)
        if target_token_ids.numel() and int(target_token_ids.max().item()) >= model_d_vocab:
            raise ValueError("Supernode probability-delta ranking only supports real vocabulary logit targets")

        if hasattr(model, "run_with_hooks"):
            baseline_logits = model(input_ids)
        else:
            baseline_logits = model.model(input_ids.unsqueeze(0)).logits
            
        baseline_probs = torch.softmax(baseline_logits[0, -1], dim=-1)[target_token_ids]
        device = input_ids.device
        dtype = baseline_logits.dtype

        w_out_cache = {}

        def source_vector_for_member(member: int) -> tuple[int, torch.Tensor]:
            layer = int(graph.neuron_locations[member, 0].item())
            neuron_id = int(graph.neuron_locations[member, 2].item())
            if layer not in w_out_cache:
                if hasattr(model, "blocks"):
                    old_mlp = model.blocks[layer].mlp.old_mlp
                    w_out_cache[layer] = model._row_oriented_weight(
                        old_mlp.W_out.to(device=device, dtype=dtype)
                    )
                else:
                    _, _, out_rows, _, _ = model._layer_weights(layer, device=device, dtype=dtype)
                    w_out_cache[layer] = out_rows
            activation = graph.neuron_activations[member].to(device=device, dtype=dtype)
            return layer, activation * w_out_cache[layer][neuron_id]

        deltas = torch.empty(
            len(supernode_members),
            graph.n_logits,
            dtype=torch.float32,
            device="cpu",
        )

        has_hooks = hasattr(model, "run_with_hooks")

        if not has_hooks:
            # Fast DLA approximation for HFLlamaGraphAdapter
            if hasattr(model, "model"):
                out = model.model(input_ids.unsqueeze(0), output_hidden_states=True, return_dict=True)
                final_resid = out.hidden_states[-1][0, -1]  # shape (d_model,)
                # For LlamaForCausalLM, the norm is in .model.norm
                norm_fn = getattr(model.model, "model", model.model).norm
                head_fn = model.lm_head
            else:
                raise RuntimeError("Cannot approximate ablation prob deltas without hidden states")

            for i, members in enumerate(supernode_members):
                total_source = torch.zeros_like(final_resid)
                for member in members:
                    _, source_vector = source_vector_for_member(member)
                    total_source += source_vector
                
                ablated_resid = final_resid - total_source
                ablated_logits = head_fn(norm_fn(ablated_resid.unsqueeze(0).unsqueeze(0)))[0, 0]
                ablated_probs = torch.softmax(ablated_logits, dim=-1)[target_token_ids]
                deltas[i] = (baseline_probs - ablated_probs).detach().float().cpu()

        else:
            # Full network ablation using hooks for HookedTransformer
            total_batches = math.ceil(len(supernode_members) / activation_forward_batch_size)

            def make_supernode_ablation_hook(entries: list[tuple[int, int, torch.Tensor]]):
                def hook_fn(acts: torch.Tensor, hook):
                    acts_out = acts.clone()
                    for batch_idx, token_pos, vector in entries:
                        acts_out[batch_idx, token_pos] -= vector.to(device=acts.device, dtype=acts.dtype)
                    return acts_out
                return hook_fn

            for batch_idx, start in enumerate(
                range(0, len(supernode_members), activation_forward_batch_size),
                start=1,
            ):
                batch_supernodes = supernode_members[start:start + activation_forward_batch_size]
                entries_by_layer: dict[int, list[tuple[int, int, torch.Tensor]]] = {}
                for local_supernode_idx, members in enumerate(batch_supernodes):
                    for member in members:
                        layer, source_vector = source_vector_for_member(member)
                        token_pos = int(graph.neuron_locations[member, 1].item())
                        entries_by_layer.setdefault(layer, []).append(
                            (local_supernode_idx, token_pos, source_vector)
                        )

                hooks = [
                    (
                        f"blocks.{layer_idx}.{model.feature_output_hook}",
                        make_supernode_ablation_hook(entries),
                    )
                    for layer_idx, entries in entries_by_layer.items()
                ]
                ablated_logits = model.run_with_hooks(
                    input_ids.expand(len(batch_supernodes), -1),
                    fwd_hooks=hooks,
                )
                ablated_probs = torch.softmax(ablated_logits[:, -1], dim=-1)[:, target_token_ids]
                deltas[start:start + len(batch_supernodes)] = (
                    baseline_probs.unsqueeze(0) - ablated_probs
                ).detach().float().cpu()
                logger.info(
                    "  supernode probability-delta ranking batch %d/%d: supernodes %d-%d",
                    batch_idx,
                    total_batches,
                    start,
                    start + len(batch_supernodes) - 1,
                )

        return deltas

    def cluster_by_graph_node_deltas(neuron_indices: torch.Tensor) -> list[list[int]]:
        if neuron_indices.numel() == 0:
            return []
        sim_device = graph.adjacency_device
        neuron_indices_device = neuron_indices.to(device=sim_device, dtype=torch.long)
        deltas = graph.adjacency_matrix[:, neuron_indices_device].transpose(0, 1).detach()
        normalized_deltas = F.normalize(deltas.to(sim_device), p=2, dim=1, eps=1e-12)
        similarity = (normalized_deltas @ normalized_deltas.T).clamp(min=-1.0, max=1.0)
        # Connected-components uses Python loops → must be on CPU
        distance_matrix = (torch.arccos(similarity) / math.pi).cpu()
        assignments = angular_distance_connected_components(distance_matrix, computation_eps)
        unique_assignments, cluster_sizes = torch.unique(assignments, return_counts=True)
        logger.info(
            "  graph-node delta clustering: eps=%.6g neurons=%d graph_nodes=%d clusters=%d min_size=%d max_size=%d",
            float(computation_eps),
            int(neuron_indices.numel()),
            int(graph.n_nodes),
            int(unique_assignments.numel()),
            int(cluster_sizes.min().item()),
            int(cluster_sizes.max().item()),
        )

        clustered: list[list[int]] = []
        for cluster_id in unique_assignments.tolist():
            cluster_rows = torch.where(assignments == int(cluster_id))[0]
            members = [int(member) for member in neuron_indices[cluster_rows].tolist()]
            members.sort(key=lambda member: abs(float(node_influence[member].item())), reverse=True)
            clustered.append(members)
        return clustered

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

    prob_delta_norm_by_neuron: dict[int, float] = {}
    prob_delta_by_neuron: dict[int, torch.Tensor] = {}

    def ensure_prob_deltas(neuron_indices: torch.Tensor) -> None:
        missing = [
            int(member)
            for member in neuron_indices.detach().cpu().tolist()
            if int(member) not in prob_delta_by_neuron
        ]
        if not missing:
            return
        missing_indices = torch.tensor(missing, dtype=torch.long)
        deltas = compute_ablation_prob_deltas(missing_indices)
        norms = deltas.norm(dim=1)
        for member, delta, norm in zip(missing, deltas, norms, strict=True):
            prob_delta_by_neuron[int(member)] = delta.detach().float().cpu()
            prob_delta_norm_by_neuron[int(member)] = float(norm.item())

    def supernode_prob_delta_norm(members: list[int]) -> float:
        if not members:
            return 0.0
        ensure_prob_deltas(torch.tensor(members, dtype=torch.long))
        if graph.n_logits == 0:
            return 0.0
        total_delta = torch.stack([prob_delta_by_neuron[member] for member in members]).sum(dim=0)
        return float(total_delta.norm().item())

    def sort_supernodes_by_output_prob_delta(
        supernodes_to_sort: list[list[int]],
    ) -> tuple[list[list[int]], torch.Tensor, list[float]]:
        if not supernodes_to_sort:
            empty_deltas = torch.empty((0, graph.n_logits), dtype=torch.float32)
            return supernodes_to_sort, empty_deltas, []
        deltas = compute_supernode_ablation_prob_deltas(supernodes_to_sort)
        scores = deltas.norm(dim=1).tolist()
        order = sorted(range(len(supernodes_to_sort)), key=lambda idx: scores[idx], reverse=True)
        order_tensor = torch.tensor(order, dtype=torch.long)
        return (
            [supernodes_to_sort[idx] for idx in order],
            deltas[order_tensor],
            [float(scores[idx]) for idx in order],
        )

    def elbow_keep_count(norms: list[float]) -> tuple[int, int | None]:
        if not norms:
            return 0, None
        elbow_idx = find_elbow(norms)
        keep_count = max(1, int(elbow_idx))
        keep_count = min(keep_count, len(norms))
        return keep_count, elbow_idx

    def sort_cluster_by_abs_influence(
        members: list[int],
        activation_maps: torch.Tensor,
    ) -> tuple[list[int], torch.Tensor]:
        if not members:
            return members, activation_maps
        order = sorted(
            range(len(members)),
            key=lambda idx: abs(float(node_influence[members[idx]].item())),
            reverse=True,
        )
        order_tensor = torch.tensor(order, dtype=torch.long)
        return [int(members[idx]) for idx in order], activation_maps[order_tensor]

    def problem_activation_write_cache_file(neuron_locations: torch.Tensor) -> str:
        if activation_write_cache_path is None:
            raise ValueError("activation_write_cache_path is not configured")

        resolved_model_name = model_name or getattr(model.cfg, "model_name", "model")
        model_cache_dir = os.path.join(
            activation_write_cache_path,
            safe_cache_segment(str(resolved_model_name)),
        )
        os.makedirs(model_cache_dir, exist_ok=True)

        locations = neuron_locations.detach().cpu().to(dtype=torch.long).contiguous()
        key = hashlib.sha256()
        key.update(b"problem-local-activation-write-cache-v1")
        key.update(str(dataset).encode("utf-8"))
        key.update(graph.input_string.encode("utf-8"))
        key.update(
            (
                f"n_pos={int(graph.n_pos)};"
                f"n_locations={int(locations.shape[0])};"
                f"d_model={int(model.cfg.d_model)}"
            ).encode("utf-8"),
        )
        key.update(locations.numpy().tobytes())
        return os.path.join(
            model_cache_dir,
            f"activation_write_problem_{key.hexdigest()[:16]}.pt",
        )

    def w_down_vectors_for_locations(neuron_locations: torch.Tensor) -> torch.Tensor:
        locations = neuron_locations.detach().cpu().to(dtype=torch.long)
        d_model = int(model.cfg.d_model)
        w_down_vectors = torch.empty((locations.shape[0], d_model), dtype=torch.float32)
        w_out_cache: dict[int, torch.Tensor] = {}
        for location_idx, (layer_t, _token_pos_t, neuron_id_t) in enumerate(locations):
            layer = int(layer_t.item())
            neuron_id = int(neuron_id_t.item())
            if layer not in w_out_cache:
                old_mlp = model.blocks[layer].mlp.old_mlp
                w_out_cache[layer] = model._row_oriented_weight(
                    old_mlp.W_out.to(device=model.cfg.device)
                )
            w_down_vectors[location_idx] = w_out_cache[layer][neuron_id].detach().float().cpu()
        return w_down_vectors

    kept_neuron_indices_device = torch.where(kept_neuron_mask)[0]
    kept_neuron_indices = kept_neuron_indices_device.detach().cpu()
    logger.info("  Kept neurons for supergraph: %d", int(kept_neuron_indices.numel()))
    all_supernode_prob_delta_norms = torch.empty(0, dtype=torch.float32)
    prob_delta_elbow_index = None
    activation_write_result_for_kept: ActivationWriteResult | None = None

    def get_activation_write_result_for_kept() -> ActivationWriteResult:
        nonlocal activation_write_result_for_kept
        if activation_write_result_for_kept is not None:
            return activation_write_result_for_kept
        if dataset is None:
            raise ValueError("A dataset is required to build supernode activation heatmaps")

        kept_neuron_locations = graph.neuron_locations[kept_neuron_indices_device].detach().cpu()
        if activation_write_cache_path:
            cache_file = problem_activation_write_cache_file(kept_neuron_locations)
            if os.path.isfile(cache_file):
                logger.info(
                    "  Loading problem-local activation-write cache: %s",
                    cache_file,
                )
                data = torch.load(cache_file, weights_only=False, map_location="cpu")
                cached_locations = data.get("neuron_locations")
                if (
                    cached_locations is not None
                    and torch.equal(
                        cached_locations.detach().cpu().to(dtype=torch.long),
                        kept_neuron_locations.to(dtype=torch.long),
                    )
                ):
                    activation_write_result_for_kept = ActivationWriteResult(
                        activations=data["activations"].detach().float().cpu(),
                        w_down_vectors=w_down_vectors_for_locations(kept_neuron_locations),
                        arg_values=data["arg_values"],
                    )
                    return activation_write_result_for_kept
                logger.info(
                    "  Ignoring stale activation-write cache with mismatched neuron locations: %s",
                    cache_file,
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
        )
        if activation_write_cache_path:
            logger.info("  Saving problem-local activation-write cache: %s", cache_file)
            torch.save(
                {
                    "activations": activation_write_result_for_kept.activations.detach()
                    .to(dtype=torch.float16)
                    .cpu(),
                    "arg_values": activation_write_result_for_kept.arg_values,
                    "neuron_locations": kept_neuron_locations.to(dtype=torch.long),
                },
                cache_file,
            )
        return activation_write_result_for_kept

    # For fast-mode graphs, pre-populate prob_delta_by_neuron with DLA vectors
    # (activation × W_out × W_U_logits) so supernode probability-delta ranking
    # can skip extra ablation forward passes.
    _fast_dla_populated = False
    if (
        cluster_method == "ablation"
        and graph.neuron_write_vectors is not None
        and kept_neuron_indices.numel() > 0
        and graph.n_logits > 0
    ):
        _wU = model.unembed.W_U if hasattr(model, "unembed") else model.W_U
        _logit_ids = graph.logit_token_ids.to(_wU.device)
        with torch.no_grad():
            _W_U_logits = _wU.detach()[:, _logit_ids].float()  # [d_model, n_logits]
            _kept_wv = (
                graph.neuron_write_vectors[kept_neuron_indices_device]
                .float()
                .to(_W_U_logits.device)
            )
            _neuron_dla = _kept_wv @ _W_U_logits  # [K, n_logits], kept on GPU
        for _k, _member in enumerate(kept_neuron_indices.tolist()):
            _dla = _neuron_dla[_k]
            prob_delta_by_neuron[_member] = _dla  # GPU tensor; clustering matmul stays on GPU
            prob_delta_norm_by_neuron[_member] = float(_dla.norm().item())
        del _wU, _W_U_logits, _kept_wv, _neuron_dla
        _fast_dla_populated = True
        logger.info("  Fast DLA pre-population: %d neurons", int(kept_neuron_indices.numel()))
    if cluster_method == "ablation" and kept_neuron_indices.numel():
        logger.info("  Building supernodes with graph-node delta clustering")
        supernodes = cluster_by_graph_node_deltas(kept_neuron_indices)

        if _fast_dla_populated:
            # Fast path: compute supernode DLAs by summing pre-computed member DLAs.
            # No extra forward passes.
            _sn_dla_list = []
            for _sn in supernodes:
                _sn_dla = torch.stack([prob_delta_by_neuron[_m] for _m in _sn]).sum(dim=0) if _sn else torch.zeros(graph.n_logits, dtype=torch.float32)
                _sn_dla_list.append(_sn_dla)
            supernode_prob_deltas = torch.stack(_sn_dla_list) if _sn_dla_list else torch.empty((0, graph.n_logits), dtype=torch.float32)
            supernode_prob_delta_norms = supernode_prob_deltas.norm(dim=1).tolist()
            _order = sorted(range(len(supernodes)), key=lambda _i: supernode_prob_delta_norms[_i], reverse=True)
            supernodes = [supernodes[_i] for _i in _order]
            if _order:
                supernode_prob_deltas = supernode_prob_deltas[torch.tensor(_order, dtype=torch.long)]
            supernode_prob_delta_norms = [supernode_prob_delta_norms[_i] for _i in _order]
        else:
            supernodes, supernode_prob_deltas, supernode_prob_delta_norms = (
                sort_supernodes_by_output_prob_delta(supernodes)
            )

        all_supernode_prob_delta_norms = torch.tensor(
            supernode_prob_delta_norms,
            dtype=torch.float32,
        )
        keep_count, prob_delta_elbow_index = elbow_keep_count(supernode_prob_delta_norms)
        logger.info(
            "  Probability-delta elbow: index=%s keeping_supernodes=%d/%d",
            prob_delta_elbow_index,
            keep_count,
            len(supernodes),
        )
        supernodes = supernodes[:keep_count]
        supernode_prob_deltas = supernode_prob_deltas[:keep_count]
        supernode_prob_delta_norms = supernode_prob_delta_norms[:keep_count]
        for supernode_idx, (members, prob_delta_norm) in enumerate(
            zip(supernodes, supernode_prob_delta_norms, strict=True)
        ):
            logger.info(
                "  supernode %d prob_delta_norm=%.6g influence=%.6g neuron locations: %s",
                supernode_idx,
                prob_delta_norm,
                supernode_influence(members),
                format_member_locations(members),
            )
        if supernode_heatmap_output_dir is not None:
            if dataset is None:
                logger.info(
                    "  Skipping supernode heatmap PDFs: no dataset was provided",
                )
            else:
                activation_write_result = get_activation_write_result_for_kept()
                kept_row_by_member = {
                    int(member): row_idx
                    for row_idx, member in enumerate(kept_neuron_indices.tolist())
                }
                for supernode_idx, members in enumerate(supernodes):
                    row_indices = torch.tensor(
                        [kept_row_by_member[int(member)] for member in members],
                        dtype=torch.long,
                    )
                    saved_path = save_supernode_activation_heatmap_pdf(
                        activation_write_result.activations[row_indices],
                        activation_write_result.arg_values,
                        members,
                        graph.neuron_locations.detach().cpu(),
                        output_path=os.path.join(
                            supernode_heatmap_output_dir,
                            f"supernode_{supernode_idx}.pdf",
                        ),
                        title=f"supernode {supernode_idx}: ablation clustering",
                    )
                    logger.info("  Saved supernode heatmap PDF: %s", saved_path)
    elif dataset and kept_neuron_indices.numel():
        kept_neuron_locations = graph.neuron_locations[kept_neuron_indices_device].detach().cpu()
        activation_write_result = get_activation_write_result_for_kept()
        prompt_tokens = decoded_prompt_tokens()
        kept_token_positions = kept_neuron_locations[:, 1]
        supernodes: list[list[int]] = []
        supernode_heatmaps: list[tuple[torch.Tensor, list[list[int]], list[int], str]] = []

        for token_pos in sorted(int(pos.item()) for pos in torch.unique(kept_token_positions)):
            phase_rows = torch.where(kept_token_positions == int(token_pos))[0]
            if phase_rows.numel() == 0:
                logger.info(
                    "  Full-grid clustering for token_pos=%d: no neurons",
                    int(token_pos),
                )
                continue

            phase_maps = activation_write_result.activations[phase_rows].detach().float()
            assignments = cluster_phase(
                phase_name=f"token_pos={int(token_pos)}",
                phase_rows=phase_rows,
                activation_maps=phase_maps,
                w_down_vectors=activation_write_result.w_down_vectors[phase_rows],
                sigma=computation_sigma,
                eps=computation_eps,
            )
            for cluster_id in torch.unique(assignments).tolist():
                cluster_rows = torch.where(assignments == int(cluster_id))[0]
                result_rows = phase_rows[cluster_rows]
                members = kept_neuron_indices[result_rows].tolist()
                cluster_heatmaps = activation_write_result.activations[
                    result_rows
                ].detach().float()
                members, cluster_heatmaps = sort_cluster_by_abs_influence(
                    [int(member) for member in members],
                    cluster_heatmaps,
                )
                supernodes.append(members)
                token_text = (
                    prompt_tokens[int(token_pos)]
                    if int(token_pos) < len(prompt_tokens)
                    else "<out-of-range>"
                ).replace("\n", "\\n").replace("\r", "\\r")
                supernode_heatmaps.append((
                    cluster_heatmaps,
                    activation_write_result.arg_values,
                    members,
                    f"token {int(token_pos)} {token_text!r}",
                ))

        supernode_deltas = compute_supernode_ablation_prob_deltas(supernodes)
        supernode_prob_delta_norms = supernode_deltas.norm(dim=1).tolist()
        supernode_order = sorted(
            range(len(supernodes)),
            key=lambda idx: supernode_prob_delta_norms[idx],
            reverse=True,
        )
        supernode_order_tensor = torch.tensor(supernode_order, dtype=torch.long)
        supernodes = [supernodes[idx] for idx in supernode_order]
        supernode_heatmaps = [supernode_heatmaps[idx] for idx in supernode_order]
        supernode_prob_deltas = supernode_deltas[supernode_order_tensor]
        supernode_prob_delta_norms = [float(supernode_prob_delta_norms[idx]) for idx in supernode_order]
        all_supernode_prob_delta_norms = torch.tensor(
            supernode_prob_delta_norms,
            dtype=torch.float32,
        )
        keep_count, prob_delta_elbow_index = elbow_keep_count(supernode_prob_delta_norms)
        logger.info(
            "  Probability-delta elbow: index=%s keeping_supernodes=%d/%d",
            prob_delta_elbow_index,
            keep_count,
            len(supernodes),
        )
        supernodes = supernodes[:keep_count]
        supernode_heatmaps = supernode_heatmaps[:keep_count]
        supernode_prob_deltas = supernode_prob_deltas[:keep_count]
        supernode_prob_delta_norms = supernode_prob_delta_norms[:keep_count]

        for supernode_idx, (members, prob_delta_norm) in enumerate(
            zip(supernodes, supernode_prob_delta_norms, strict=True)
        ):
            logger.info(
                "  supernode %d prob_delta_norm=%.6g influence=%.6g neuron locations: %s",
                supernode_idx,
                prob_delta_norm,
                supernode_influence(members),
                format_member_locations(members),
            )

        if supernode_heatmap_output_dir is not None:
            for supernode_idx, (
                activation_grid,
                heatmap_arg_values,
                members,
                title,
            ) in enumerate(supernode_heatmaps):
                saved_path = save_supernode_activation_heatmap_pdf(
                    activation_grid,
                    heatmap_arg_values,
                    members,
                    graph.neuron_locations.detach().cpu(),
                    output_path=os.path.join(
                        supernode_heatmap_output_dir,
                        f"supernode_{supernode_idx}.pdf",
                    ),
                    title=f"supernode {supernode_idx}: {title}",
                )
                logger.info("  Saved supernode heatmap PDF: %s", saved_path)
    else:
        if dataset:
            logger.info("  No kept neurons for dataset activation clustering")
        else:
            logger.info("  No dataset provided; using all kept neurons as one supernode")
        fallback_members = [int(member) for member in kept_neuron_indices.tolist()]
        fallback_members.sort(key=lambda member: abs(float(node_influence[member].item())), reverse=True)
        supernodes = [fallback_members] if fallback_members else []
        supernode_prob_deltas = compute_supernode_ablation_prob_deltas(supernodes)
        all_supernode_prob_delta_norms = supernode_prob_deltas.norm(dim=1).detach().float().cpu()
        keep_count, prob_delta_elbow_index = elbow_keep_count(all_supernode_prob_delta_norms.tolist())
        supernodes = supernodes[:keep_count]
        supernode_prob_deltas = supernode_prob_deltas[:keep_count]

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
        supernode_prob_deltas=supernode_prob_deltas,
        all_supernode_prob_delta_norms=all_supernode_prob_delta_norms,
        prob_delta_elbow_index=prob_delta_elbow_index,
    )


def extract_supernode_members(supergraph: SuperGraph, graph: Graph, model) -> list[dict]:
    """Build per-supernode activation and W_out data for DLA alignment.

    Returns a list of dicts, one per supernode, with:
        cluster_id: int (index into supergraph.supernodes)
        activations: list[float] (a_i for each member neuron)
        w_out_rows: list[Tensor] (W_out[neuron_id, :] ∈ R^{d_model} per member)
    """
    result = []
    
    # Cache the W_out matrices per layer to avoid transferring and transposing 
    # massive 235MB tensors 4,500 times in the inner loop!
    w_out_cache = {}
    
    for i, members in enumerate(supergraph.supernodes):
        acts = []
        w_outs = []
        for nid in members:
            layer = int(graph.neuron_locations[nid, 0].item())
            neuron_id = int(graph.neuron_locations[nid, 2].item())
            
            if layer not in w_out_cache:
                old_mlp = model.blocks[layer].mlp.old_mlp
                W_out = model._row_oriented_weight(
                    old_mlp.W_out.to(device=graph.adjacency_device)
                )
                w_out_cache[layer] = W_out
                
            acts.append(graph.neuron_activations[nid].unsqueeze(0))
            w_outs.append(w_out_cache[layer][neuron_id].detach().clone())
        result.append({
            "cluster_id": i,
            "activations": acts,
            "w_out_rows": w_outs,
            "size": len(members),
        })
    return result

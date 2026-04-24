"""Graph container and influence utilities for LLaMA neuron attribution."""

from __future__ import annotations

from typing import NamedTuple

import torch

from graph_loss.attribution.targets import LogitTarget
from graph_loss.utils import UnifiedConfig, convert_nnsight_config_to_transformerlens


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
    ):
        """Container for dense neuron/token/logit attribution graphs.

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

    @property
    def n_neurons(self) -> int:
        return len(self.neuron_locations)

    @property
    def n_tokens(self) -> int:
        return len(self.input_tokens)

    @property
    def n_logits(self) -> int:
        return len(self.logit_targets)

    def to(self, device):
        self.adjacency_matrix = self.adjacency_matrix.to(device)
        self.neuron_locations = self.neuron_locations.to(device)
        self.neuron_activations = self.neuron_activations.to(device)
        self.logit_probabilities = self.logit_probabilities.to(device)

    @property
    def logit_token_ids(self) -> torch.Tensor:
        return torch.tensor(
            [target.vocab_idx for target in self.logit_targets],
            dtype=torch.long,
            device=self.logit_probabilities.device,
        )

    def to_pt(self, path: str):
        torch.save(
            {
                "input_string": self.input_string,
                "adjacency_matrix": self.adjacency_matrix,
                "cfg": self.cfg,
                "neuron_locations": self.neuron_locations,
                "logit_targets": self.logit_targets,
                "logit_probabilities": self.logit_probabilities,
                "vocab_size": self.vocab_size,
                "input_tokens": self.input_tokens,
                "neuron_activations": self.neuron_activations,
            },
            path,
        )

    @staticmethod
    def from_pt(path: str, map_location="cpu") -> "Graph":
        return Graph(**torch.load(path, weights_only=False, map_location=map_location))


def normalize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    normalized = matrix.abs()
    return normalized / normalized.sum(dim=1, keepdim=True).clamp(min=1e-10)


def compute_influence(A: torch.Tensor, logit_weights: torch.Tensor, max_iter: int = 1000):
    current_influence = logit_weights @ A
    influence = current_influence
    iterations = 0
    while current_influence.any():
        if iterations >= max_iter:
            raise RuntimeError(
                f"Influence computation failed to converge after {iterations} iterations"
            )
        current_influence = current_influence @ A
        influence += current_influence
        iterations += 1
    return influence


def compute_node_influence(adjacency_matrix: torch.Tensor, logit_weights: torch.Tensor):
    return compute_influence(normalize_matrix(adjacency_matrix), logit_weights)


def compute_edge_influence(pruned_matrix: torch.Tensor, logit_weights: torch.Tensor):
    normalized_pruned = normalize_matrix(pruned_matrix)
    pruned_influence = compute_influence(normalized_pruned, logit_weights)
    pruned_influence += logit_weights
    edge_scores = normalized_pruned * pruned_influence[:, None]
    return edge_scores


def find_threshold(scores: torch.Tensor, threshold: float):
    sorted_scores = torch.sort(scores, descending=True).values
    cumulative_score = torch.cumsum(sorted_scores, dim=0) / torch.sum(sorted_scores)
    threshold_index = int(torch.searchsorted(cumulative_score, threshold).item())
    threshold_index = min(threshold_index, len(cumulative_score) - 1)
    return sorted_scores[threshold_index]


class PruneResult(NamedTuple):
    node_mask: torch.Tensor
    edge_mask: torch.Tensor
    cumulative_scores: torch.Tensor


def prune_graph(graph: Graph, node_threshold: float = 0.8, edge_threshold: float = 0.98) -> PruneResult:
    """Prune low-influence neurons while always retaining token and logit nodes."""

    if node_threshold > 1.0 or node_threshold < 0.0:
        raise ValueError("node_threshold must be between 0.0 and 1.0")
    if edge_threshold > 1.0 or edge_threshold < 0.0:
        raise ValueError("edge_threshold must be between 0.0 and 1.0")

    n_logits = graph.n_logits
    n_tokens = graph.n_tokens
    n_neurons = graph.n_neurons
    token_start = n_neurons

    logit_weights = torch.zeros(
        graph.adjacency_matrix.shape[0], device=graph.adjacency_matrix.device
    )
    logit_weights[-n_logits:] = graph.logit_probabilities

    node_influence = compute_node_influence(graph.adjacency_matrix, logit_weights)
    node_mask = node_influence >= find_threshold(node_influence, node_threshold)
    node_mask[token_start:] = True

    pruned_matrix = graph.adjacency_matrix.clone()
    pruned_matrix[~node_mask] = 0
    pruned_matrix[:, ~node_mask] = 0

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

    sorted_scores, sorted_indices = torch.sort(node_influence, descending=True)
    cumulative_scores = torch.cumsum(sorted_scores, dim=0) / torch.sum(sorted_scores)
    final_scores = torch.zeros_like(node_influence)
    final_scores[sorted_indices] = cumulative_scores

    return PruneResult(node_mask, edge_mask, final_scores)


class SuperGraph(NamedTuple):
    supernode_adjacency_matrix: torch.Tensor
    supernodes: list[list[int]]       # old node ids inside each new node


def build_super_graph(graph: Graph, epsilon: float = 1e-3, min_cum_logit_influence: float = 0.9) -> SuperGraph:
    """Create supernodes as clusters of nodes that have distance < epsilon and cumulative logit influence > min_cum_logit_influence."""

    if epsilon > 1.0 or epsilon < 0.0:
        raise ValueError("epsilon must be between 0.0 and 1.0")
    if min_cum_logit_influence < 0.0:
        raise ValueError("min_cum_logit_influence must be non-negative")

    n_logits = graph.n_logits
    n_tokens = graph.n_tokens
    n_neurons = graph.n_neurons
    token_start = n_neurons
    logit_start = token_start + n_tokens

    logit_basis = torch.zeros(n_logits, graph.adjacency_matrix.shape[0], 
        device=graph.adjacency_matrix.device)
    logit_basis[
        torch.arange(n_logits, device=graph.adjacency_matrix.device), 
        logit_start + torch.arange(n_logits, device=graph.adjacency_matrix.device)
    ] = 1

    logit_influence = compute_node_influence(graph.adjacency_matrix, logit_basis)
    node_influence_vectors = (
        logit_influence.T[:n_neurons] * (1 - graph.logit_probabilities) * graph.logit_probabilities
    )
    node_influence_normalized = node_influence_vectors / node_influence_vectors.norm(
        dim=-1, keepdim=True
    ).clamp(min=1e-12)

    num_supernodes = 0
    node_clusters = [0] * n_neurons
    for n in range(n_neurons):
        if node_clusters[n] != 0:
            continue

        neighbors = range_query(node_influence_normalized, n, epsilon)
        if node_influence_vectors[neighbors].sum(dim=0).norm(dim=-1) < min_cum_logit_influence:
            node_clusters[n] = -1
            continue
        
        num_supernodes += 1
        node_clusters[n] = num_supernodes
        neighbors = [neighbor for neighbor in neighbors if neighbor != n]
        idx = 0

        while idx < len(neighbors):
            s = neighbors[idx]
            idx += 1
            if node_clusters[s] == -1:
                node_clusters[s] = num_supernodes
            if node_clusters[s] != 0:
                continue

            node_clusters[s] = num_supernodes
            new_neighbors = range_query(node_influence_normalized, s, epsilon)

            if node_influence_vectors[new_neighbors].sum(dim=0).norm(dim=-1) >= min_cum_logit_influence:
                neighbors = list(set(neighbors) | set(new_neighbors))

    adj_matrix_norm = normalize_matrix(graph.adjacency_matrix)
    supernode_adj_matrix = torch.zeros(num_supernodes, num_supernodes, dtype=graph.adjacency_matrix.dtype, device=graph.adjacency_matrix.device)
    supernodes = [[i for i in range(n_neurons) if node_clusters[i] == n] for n in range(1, num_supernodes + 1)]
    for t in range(num_supernodes):
        total_input = torch.abs(adj_matrix_norm[:, supernodes[t]]).sum(dim=0)
        internal_input = torch.abs(adj_matrix_norm[supernodes[t]][:, supernodes[t]]).sum(dim=0)
        frac_external = (total_input - internal_input) / total_input.clamp(min=1e-10)
        
        for s in range(num_supernodes):
            sum_A = adj_matrix_norm[supernodes[t]][:, supernodes[s]].sum(dim=1)
            supernode_adj_matrix[t, s] = (frac_external * sum_A).sum(dim=0) / frac_external.sum(dim=0).clamp(min=1e-10)

    return SuperGraph(supernode_adjacency_matrix=supernode_adj_matrix, supernodes=supernodes)

def range_query(node_influence_normalized: torch.Tensor, neuron_idx: int, epsilon: float) -> list[int]:
    """Find all nodes within epsilon of node_idx in the normalized node influence space."""
    return torch.where(1 - node_influence_normalized @ node_influence_normalized[neuron_idx] <= epsilon)[0].tolist()







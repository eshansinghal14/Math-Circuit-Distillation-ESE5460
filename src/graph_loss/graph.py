"""Graph container and influence utilities for LLaMA neuron attribution."""

from __future__ import annotations

import logging
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
        )

def normalize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    math_dtype = (
        torch.float32
        if matrix.dtype in (torch.float16, torch.bfloat16)
        else matrix.dtype
    )
    normalized = matrix.to(dtype=math_dtype).abs()
    return normalized / normalized.sum(dim=1, keepdim=True).clamp(min=1e-10)


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

    node_influence = compute_node_influence(adjacency_matrix, logit_weights)
    node_mask = node_influence >= find_threshold(node_influence, node_threshold)
    node_mask[token_start:] = True

    pruned_matrix = adjacency_matrix.clone()
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
    logger = logging.getLogger(__name__)

    adjacency_matrix = graph.adjacency_matrix
    all_nodes_basis = torch.zeros(
        adjacency_matrix.shape[0],
        adjacency_matrix.shape[0],
        dtype=adjacency_matrix.dtype,
        device=adjacency_matrix.device,
    )
    all_nodes_basis[
        torch.arange(adjacency_matrix.shape[0], device=adjacency_matrix.device), 
        torch.arange(adjacency_matrix.shape[0], device=adjacency_matrix.device)
    ] = 1

    logger.info("  Computing logit influence for supergraph")
    logit_influence = compute_node_influence(adjacency_matrix, all_nodes_basis)
    logit_probabilities = graph.logit_probabilities.to(
        device=adjacency_matrix.device,
        dtype=adjacency_matrix.dtype,
    )
    # node_influence_vectors = (
    #     logit_influence.T[:n_neurons][:, :logit_start] * (1 - logit_probabilities) * logit_probabilities
    # )
    node_influence_vectors = logit_influence.T[:n_neurons]

    node_influence_normalized = node_influence_vectors / node_influence_vectors.norm(
        dim=-1, keepdim=True
    ).clamp(min=1e-12)

    def build_supernodes_svd(B_weighted, coverage_threshold, min_cluster_influence):
        """
        Cluster on directions, validate on magnitudes.
        """
        def silhouette_score(X, assignments, dist_type="cossim"):
            unique_assignments = torch.unique(assignments)
            if len(unique_assignments) <= 1 or len(unique_assignments) >= len(X):
                return torch.tensor(float("nan"), device=X.device)

            if dist_type == "cossim":
                X_norm = X / X.norm(dim=1, keepdim=True).clamp(min=1e-12)
                dists = 1 - X_norm @ X_norm.T
            else:
                dists = torch.cdist(X, X)

            scores = []
            for i in range(len(X)):
                same_cluster = assignments == assignments[i]
                other_clusters = unique_assignments[unique_assignments != assignments[i]]

                same_cluster[i] = False
                a = dists[i, same_cluster].mean() if same_cluster.any() else torch.tensor(0.0, device=X.device)
                b = torch.stack([
                    dists[i, assignments == cluster].mean()
                    for cluster in other_clusters
                ]).min()
                scores.append((b - a) / torch.maximum(a, b).clamp(min=1e-12))

            return torch.stack(scores).mean()

        # magnitudes — used for noise filtering and validity
        magnitudes = B_weighted.norm(dim=1)           # [n_neurons]
        finite_magnitudes = magnitudes[torch.isfinite(magnitudes)]
        if finite_magnitudes.numel():
            logger.info(
                "  Supernode influence magnitudes: finite=%d/%d min=%.6g median=%.6g max=%.6g",
                int(finite_magnitudes.numel()),
                int(magnitudes.numel()),
                float(finite_magnitudes.min().item()),
                float(finite_magnitudes.median().item()),
                float(finite_magnitudes.max().item()),
            )
        else:
            logger.info("  Supernode influence magnitudes: no finite values")
        
        # directions — used for clustering
        directions = B_weighted / magnitudes.unsqueeze(1).clamp(min=1e-12)
        
        # filter noise before clustering
        if min_cluster_influence is None:
            min_cluster_influence = magnitudes.median() * 0.1
        valid_mask = magnitudes >= min_cluster_influence
        
        valid_directions = directions[valid_mask]      # [n_valid × n_logits]
        if len(valid_directions) == 0:
            return torch.full((n_neurons,), -1, dtype=torch.long, device=B_weighted.device), 0

        best_k = 1
        best_score = torch.tensor(float("-inf"), device=B_weighted.device)
        best_assignments_valid = torch.zeros(len(valid_directions), dtype=torch.long, device=B_weighted.device)
        for candidate_k in range(1, min(10, len(valid_directions)) + 1):
            candidate_assignments = kmeans(valid_directions, candidate_k, dist_type="cossim")
            score = silhouette_score(valid_directions, candidate_assignments, dist_type="cossim")
            print(f"k={candidate_k} silhouette_score={float(score.item()):.6g}")
            if torch.isfinite(score) and score > best_score:
                best_k = candidate_k
                best_score = score
                best_assignments_valid = candidate_assignments

        assignments_valid = best_assignments_valid
        k = best_k
        
        # map back
        assignments = torch.full((n_neurons,), -1, dtype=torch.long, device=B_weighted.device)
        assignments[valid_mask] = assignments_valid + 1
        
        return assignments, k
    
    logger.info("  Clustering supernodes")
    node_clusters, num_supernodes = build_supernodes_svd(node_influence_vectors, min_cum_logit_influence, epsilon)

    # num_supernodes = 0
    # node_clusters = [0] * n_neurons
    # for n in range(n_neurons):
    #     if node_clusters[n] != 0:
    #         continue

    #     neighbors = range_query(node_influence_normalized, n, epsilon)
    #     if node_influence_vectors[neighbors].sum(dim=0).norm(dim=-1) < min_cum_logit_influence:
    #         node_clusters[n] = -1
    #         continue
        
    #     num_supernodes += 1
    #     node_clusters[n] = num_supernodes
    #     neighbors = [neighbor for neighbor in neighbors if neighbor != n]
    #     idx = 0

    #     while idx < len(neighbors):
    #         s = neighbors[idx]
    #         idx += 1
    #         if node_clusters[s] == -1:
    #             node_clusters[s] = num_supernodes
    #         if node_clusters[s] != 0:
    #             continue

    #         node_clusters[s] = num_supernodes
    #         new_neighbors = range_query(node_influence_normalized, s, epsilon)

    #         if node_influence_vectors[new_neighbors].sum(dim=0).norm(dim=-1) >= min_cum_logit_influence:
    #             neighbors = list(set(neighbors) | set(new_neighbors))

    logger.info("  Aggregating supernode adjacency")
    adj_matrix_norm = normalize_matrix(adjacency_matrix)
    supernode_adj_matrix = torch.zeros(
        num_supernodes,
        num_supernodes,
        dtype=adj_matrix_norm.dtype,
        device=adjacency_matrix.device,
    )
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


def kmeans(X, k, n_iter=100, dist_type="p_norm", p=2):
    # X: [n × d]
    if len(X) == 0:
        raise ValueError("kmeans requires at least one point")
    if k <= 0:
        raise ValueError("kmeans requires k > 0")
    if dist_type not in ("p_norm", "cossim"):
        raise ValueError("dist_type must be either 'p_norm' or 'cossim'")
    k = min(k, len(X))

    def pairwise_dists(points, centers):
        if dist_type == "p_norm":
            return torch.cdist(points, centers, p=p)

        points_norm = points / points.norm(dim=1, keepdim=True).clamp(min=1e-12)
        centers_norm = centers / centers.norm(dim=1, keepdim=True).clamp(min=1e-12)
        return 1 - points_norm @ centers_norm.T

    # initialize centers with kmeans++
    centers = [X[torch.randint(len(X), (1,)).item()]]
    for _ in range(k - 1):
        dists = pairwise_dists(X, torch.stack(centers)).min(dim=1).values
        init_weights = dists - dists.min()
        total_weight = init_weights.sum()
        if not torch.isfinite(total_weight) or total_weight <= 0:
            centers.append(X[torch.randint(len(X), (1,)).item()])
        else:
            probs = init_weights / total_weight
            centers.append(X[torch.multinomial(probs, 1).item()])
    centers = torch.stack(centers)  # [k × d]

    for _ in range(n_iter):
        # assign
        dists = pairwise_dists(X, centers)       # [n × k]
        assignments = dists.argmin(dim=1)        # [n]
        # update
        new_centers = torch.stack([
            X[assignments == i].mean(dim=0) 
            if (assignments == i).any() 
            else centers[i]
            for i in range(k)
        ])
        if (new_centers - centers).norm() < 1e-6:
            break
        centers = new_centers

    return assignments


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

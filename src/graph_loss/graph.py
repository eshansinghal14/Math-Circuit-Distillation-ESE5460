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
        """Container for neuron/token/logit attribution graphs.

        Nodes are stored in the order:
        ``[neurons, token_embeddings, logits]``.
        Rows represent target nodes and columns represent source nodes.
        """

        self.logit_targets = logit_targets
        self.logit_probabilities = logit_probabilities
        self.vocab_size = vocab_size if vocab_size is not None else cfg.d_vocab

        if not adjacency_matrix.is_sparse:
            raise ValueError(
                "Graph adjacency must be a sparse COO tensor. Dense adjacency matrices "
                "are not supported because they can exceed available memory."
            )

        self.input_string = input_string
        self.adjacency_matrix = adjacency_matrix.coalesce()
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
        if self.adjacency_matrix.is_sparse:
            self.adjacency_matrix = self.adjacency_matrix.coalesce()
        self.neuron_locations = self.neuron_locations.to(device)
        self.neuron_activations = self.neuron_activations.to(device)
        self.logit_probabilities = self.logit_probabilities.to(device)

    def adjacency_dense(self) -> torch.Tensor:
        raise RuntimeError(
            "Dense adjacency materialization is disabled. Use sparse graph utilities instead."
        )

    def adjacency_nnz(self) -> int:
        if self.adjacency_matrix.is_sparse:
            return int(self.adjacency_matrix._nnz())
        return int(self.adjacency_matrix.count_nonzero().item())

    def adjacency_abs_sum(self) -> float:
        if self.adjacency_matrix.is_sparse:
            return float(self.adjacency_matrix.values().abs().sum().item())
        return float(self.adjacency_matrix.abs().sum().item())

    def block_nonzero_count(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
    ) -> int:
        if self.adjacency_matrix.is_sparse:
            adjacency = self.adjacency_matrix.coalesce()
            indices = adjacency.indices()
            mask = (
                (indices[0] >= row_start)
                & (indices[0] < row_end)
                & (indices[1] >= col_start)
                & (indices[1] < col_end)
            )
            return int(mask.sum().item())
        return int(
            self.adjacency_matrix[row_start:row_end, col_start:col_end].count_nonzero().item()
        )

    def block_abs_sum(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
    ) -> float:
        if self.adjacency_matrix.is_sparse:
            adjacency = self.adjacency_matrix.coalesce()
            indices = adjacency.indices()
            values = adjacency.values()
            mask = (
                (indices[0] >= row_start)
                & (indices[0] < row_end)
                & (indices[1] >= col_start)
                & (indices[1] < col_end)
            )
            return float(values[mask].abs().sum().item())
        return float(
            self.adjacency_matrix[row_start:row_end, col_start:col_end].abs().sum().item()
        )

    @property
    def logit_token_ids(self) -> torch.Tensor:
        return torch.tensor(
            [target.vocab_idx for target in self.logit_targets],
            dtype=torch.long,
            device=self.logit_probabilities.device,
        )

    def to_pt(self, path: str):
        adjacency = self.adjacency_matrix.coalesce()
        data = {
            "input_string": self.input_string,
            "cfg": self.cfg,
            "neuron_locations": self.neuron_locations,
            "logit_targets": self.logit_targets,
            "logit_probabilities": self.logit_probabilities,
            "vocab_size": self.vocab_size,
            "input_tokens": self.input_tokens,
            "neuron_activations": self.neuron_activations,
            "adjacency_layout": "sparse_coo",
            "adjacency_indices": adjacency.indices(),
            "adjacency_values": adjacency.values(),
            "adjacency_size": tuple(adjacency.shape),
        }
        torch.save(data, path)

    @staticmethod
    def from_pt(path: str, map_location="cpu") -> "Graph":
        data = torch.load(path, weights_only=False, map_location=map_location)
        if data.get("adjacency_layout") != "sparse_coo":
            raise ValueError(
                "Only sparse COO graph files are supported. Legacy dense graph files "
                "must be regenerated with the sparse graph pipeline."
            )
        adjacency_matrix = torch.sparse_coo_tensor(
            data["adjacency_indices"],
            data["adjacency_values"],
            size=tuple(data["adjacency_size"]),
        ).coalesce()
        return Graph(
            input_string=data["input_string"],
            input_tokens=data["input_tokens"],
            neuron_locations=data["neuron_locations"],
            adjacency_matrix=adjacency_matrix,
            cfg=data["cfg"],
            neuron_activations=data["neuron_activations"],
            logit_targets=data["logit_targets"],
            logit_probabilities=data["logit_probabilities"],
            vocab_size=data.get("vocab_size"),
        )

    def apply_prune_result(self, prune_result: "PruneResult") -> "Graph":
        """Returns a new Graph with edges and nodes zeroed out according to the PruneResult masks."""
        adjacency = self.adjacency_matrix.coalesce()
        edge_mask = prune_result.edge_mask.coalesce()
        indices = adjacency.indices()
        values = adjacency.values()

        node_mask = prune_result.node_mask.to(device=indices.device)
        keep = node_mask[indices[0]] & node_mask[indices[1]]
        keep &= _sparse_edge_membership(
            indices,
            edge_mask.indices().to(device=indices.device),
            self.n_nodes,
        )

        new_adjacency = torch.sparse_coo_tensor(
            indices[:, keep],
            values[keep],
            size=adjacency.shape,
            dtype=values.dtype,
            device=values.device,
        ).coalesce()

        return Graph(
            input_string=self.input_string,
            input_tokens=self.input_tokens,
            neuron_locations=self.neuron_locations,
            adjacency_matrix=new_adjacency,
            cfg=self.cfg, # type: ignore
            neuron_activations=self.neuron_activations,
            logit_targets=self.logit_targets,
            logit_probabilities=self.logit_probabilities,
            vocab_size=self.vocab_size,
        )


def _require_sparse_coo(matrix: torch.Tensor, *, name: str = "matrix") -> torch.Tensor:
    if not matrix.is_sparse:
        raise ValueError(f"{name} must be a sparse COO tensor")
    return matrix.coalesce()


def normalize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    matrix = _require_sparse_coo(matrix)
    indices = matrix.indices()
    values = matrix.values().abs()
    row_sums = torch.zeros(
        matrix.shape[0],
        dtype=values.dtype,
        device=values.device,
    )
    row_sums.scatter_add_(0, indices[0], values)
    normalized_values = values / row_sums[indices[0]].clamp(min=1e-10)
    return torch.sparse_coo_tensor(
        indices,
        normalized_values,
        size=matrix.shape,
        dtype=normalized_values.dtype,
        device=normalized_values.device,
    ).coalesce()


def _dense_right_multiply_sparse(dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
    sparse = _require_sparse_coo(sparse)
    dense_was_1d = dense.ndim == 1
    dense_2d = dense.unsqueeze(0) if dense_was_1d else dense
    result = torch.sparse.mm(sparse.transpose(0, 1), dense_2d.T).T
    return result.squeeze(0) if dense_was_1d else result


def compute_influence(A: torch.Tensor, logit_weights: torch.Tensor, max_iter: int = 1000):
    A = _require_sparse_coo(A, name="A")
    current_influence = _dense_right_multiply_sparse(logit_weights, A)
    influence = current_influence
    iterations = 0
    while current_influence.any():
        if iterations >= max_iter:
            raise RuntimeError(
                f"Influence computation failed to converge after {iterations} iterations"
            )
        current_influence = _dense_right_multiply_sparse(current_influence, A)
        influence += current_influence
        iterations += 1
    return influence


def compute_node_influence(adjacency_matrix: torch.Tensor, logit_weights: torch.Tensor):
    return compute_influence(normalize_matrix(adjacency_matrix), logit_weights)


def compute_edge_influence(pruned_matrix: torch.Tensor, logit_weights: torch.Tensor):
    normalized_pruned = normalize_matrix(pruned_matrix)
    pruned_influence = compute_influence(normalized_pruned, logit_weights)
    pruned_influence += logit_weights
    normalized_pruned = normalized_pruned.coalesce()
    indices = normalized_pruned.indices()
    values = normalized_pruned.values() * pruned_influence[indices[0]]
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=normalized_pruned.shape,
        dtype=values.dtype,
        device=values.device,
    ).coalesce()


def find_threshold(scores: torch.Tensor, threshold: float):
    if scores.numel() == 0:
        return torch.tensor(0.0, dtype=scores.dtype, device=scores.device)
    sorted_scores = torch.sort(scores, descending=True).values
    score_sum = torch.sum(sorted_scores)
    if score_sum <= 0:
        return torch.tensor(0.0, dtype=scores.dtype, device=scores.device)
    cumulative_score = torch.cumsum(sorted_scores, dim=0) / score_sum
    threshold_index = int(torch.searchsorted(cumulative_score, threshold).item())
    threshold_index = min(threshold_index, len(cumulative_score) - 1)
    return sorted_scores[threshold_index]


def _sparse_edge_keys(indices: torch.Tensor, n_nodes: int) -> torch.Tensor:
    return indices[0] * n_nodes + indices[1]


def _sparse_edge_membership(
    query_indices: torch.Tensor,
    mask_indices: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    if query_indices.shape[1] == 0 or mask_indices.shape[1] == 0:
        return torch.zeros(query_indices.shape[1], dtype=torch.bool, device=query_indices.device)
    query_keys = _sparse_edge_keys(query_indices, n_nodes)
    mask_keys = _sparse_edge_keys(mask_indices, n_nodes).to(device=query_keys.device)
    return torch.isin(query_keys, mask_keys)


def _filter_sparse_by_node_mask(matrix: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    matrix = _require_sparse_coo(matrix)
    indices = matrix.indices()
    values = matrix.values()
    node_mask = node_mask.to(device=indices.device)
    keep = node_mask[indices[0]] & node_mask[indices[1]]
    return torch.sparse_coo_tensor(
        indices[:, keep],
        values[keep],
        size=matrix.shape,
        dtype=values.dtype,
        device=values.device,
    ).coalesce()


def _sparse_bool_from_indices(
    indices: torch.Tensor,
    size: tuple[int, int] | torch.Size,
    *,
    device: torch.device,
) -> torch.Tensor:
    values = torch.ones(indices.shape[1], dtype=torch.bool, device=device)
    return torch.sparse_coo_tensor(indices, values, size=size, device=device).coalesce()


def _edge_mask_from_scores(edge_scores: torch.Tensor, threshold: float) -> torch.Tensor:
    edge_scores = _require_sparse_coo(edge_scores, name="edge_scores")
    score_values = edge_scores.values()
    if score_values.numel() == 0:
        return _sparse_bool_from_indices(
            torch.zeros((2, 0), dtype=torch.long, device=score_values.device),
            edge_scores.shape,
            device=score_values.device,
        )
    score_threshold = find_threshold(score_values, threshold)
    keep = (score_values >= score_threshold) & (score_values != 0)
    return _sparse_bool_from_indices(
        edge_scores.indices()[:, keep],
        edge_scores.shape,
        device=score_values.device,
    )


def _filter_sparse_edge_mask_by_nodes(edge_mask: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    edge_mask = _require_sparse_coo(edge_mask, name="edge_mask")
    indices = edge_mask.indices()
    node_mask = node_mask.to(device=indices.device)
    keep = node_mask[indices[0]] & node_mask[indices[1]]
    return _sparse_bool_from_indices(indices[:, keep], edge_mask.shape, device=indices.device)


def _neuron_connectivity_from_sparse_edges(
    edge_mask: torch.Tensor,
    node_mask: torch.Tensor,
    n_neurons: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_mask = _filter_sparse_edge_mask_by_nodes(edge_mask, node_mask)
    indices = edge_mask.indices()
    row_has_edge = torch.zeros(n_neurons, dtype=torch.bool, device=node_mask.device)
    col_has_edge = torch.zeros(n_neurons, dtype=torch.bool, device=node_mask.device)
    if indices.shape[1] == 0:
        return row_has_edge, col_has_edge

    rows = indices[0].to(device=node_mask.device)
    cols = indices[1].to(device=node_mask.device)
    row_neuron_edges = rows < n_neurons
    col_neuron_edges = cols < n_neurons
    if row_neuron_edges.any():
        row_has_edge[rows[row_neuron_edges]] = True
    if col_neuron_edges.any():
        col_has_edge[cols[col_neuron_edges]] = True
    return row_has_edge, col_has_edge


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

    adjacency_matrix = graph.adjacency_matrix.coalesce()
    logit_weights = torch.zeros(
        adjacency_matrix.shape[0],
        dtype=adjacency_matrix.dtype,
        device=adjacency_matrix.device,
    )
    logit_weights[-n_logits:] = graph.logit_probabilities.to(adjacency_matrix.device)

    node_influence = compute_node_influence(adjacency_matrix, logit_weights)
    node_mask = node_influence >= find_threshold(node_influence, node_threshold)
    node_mask[token_start:] = True

    pruned_matrix = _filter_sparse_by_node_mask(adjacency_matrix, node_mask)

    edge_scores = compute_edge_influence(pruned_matrix, logit_weights)
    edge_mask = _edge_mask_from_scores(edge_scores, edge_threshold)

    old_node_mask = node_mask.clone()
    row_has_edge, col_has_edge = _neuron_connectivity_from_sparse_edges(
        edge_mask,
        node_mask,
        n_neurons,
    )
    node_mask[:n_neurons] &= col_has_edge
    node_mask[:n_neurons] &= row_has_edge

    while not torch.all(node_mask == old_node_mask):
        old_node_mask[:] = node_mask
        row_has_edge, col_has_edge = _neuron_connectivity_from_sparse_edges(
            edge_mask,
            node_mask,
            n_neurons,
        )
        node_mask[:n_neurons] &= col_has_edge
        node_mask[:n_neurons] &= row_has_edge

    sorted_scores, sorted_indices = torch.sort(node_influence, descending=True)
    cumulative_scores = torch.cumsum(sorted_scores, dim=0) / torch.sum(sorted_scores)
    final_scores = torch.zeros_like(node_influence)
    final_scores[sorted_indices] = cumulative_scores

    edge_mask = _filter_sparse_edge_mask_by_nodes(edge_mask, node_mask)
    return PruneResult(node_mask, edge_mask, final_scores)


class SuperGraph(NamedTuple):
    supernode_adjacency_matrix: torch.Tensor
    supernodes: list[list[int]]       # old node ids inside each new node


def _supernode_adjacency_from_sparse(
    adjacency_matrix: torch.Tensor,
    supernodes: list[list[int]],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    adjacency_matrix = _require_sparse_coo(adjacency_matrix, name="adjacency_matrix")
    indices = adjacency_matrix.indices()
    values = adjacency_matrix.values()
    rows = indices[0]
    cols = indices[1]
    num_supernodes = len(supernodes)
    supernode_adj_matrix = torch.zeros(
        num_supernodes,
        num_supernodes,
        dtype=dtype,
        device=device,
    )

    for t, target_members in enumerate(supernodes):
        if not target_members:
            continue
        target_nodes = torch.tensor(target_members, dtype=torch.long, device=device)
        target_col_mask = torch.isin(cols, target_nodes)
        target_row_mask = torch.isin(rows, target_nodes)

        total_input = torch.zeros(len(target_members), dtype=dtype, device=device)
        if target_col_mask.any():
            col_positions = torch.searchsorted(target_nodes, cols[target_col_mask])
            total_input.scatter_add_(0, col_positions, values[target_col_mask].abs())

        internal_input = torch.zeros(len(target_members), dtype=dtype, device=device)
        internal_mask = target_col_mask & target_row_mask
        if internal_mask.any():
            internal_col_positions = torch.searchsorted(target_nodes, cols[internal_mask])
            internal_input.scatter_add_(
                0,
                internal_col_positions,
                values[internal_mask].abs(),
            )

        frac_external = (total_input - internal_input) / total_input.clamp(min=1e-10)
        frac_external_sum = frac_external.sum(dim=0).clamp(min=1e-10)

        for s, source_members in enumerate(supernodes):
            if not source_members:
                continue
            source_nodes = torch.tensor(source_members, dtype=torch.long, device=device)
            source_col_mask = torch.isin(cols, source_nodes)
            block_mask = target_row_mask & source_col_mask
            if not block_mask.any():
                continue

            sum_A = torch.zeros(len(target_members), dtype=dtype, device=device)
            row_positions = torch.searchsorted(target_nodes, rows[block_mask])
            sum_A.scatter_add_(0, row_positions, values[block_mask])
            supernode_adj_matrix[t, s] = (frac_external * sum_A).sum(dim=0) / frac_external_sum

    return supernode_adj_matrix


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

    adjacency_matrix = graph.adjacency_matrix.coalesce()
    logit_basis = torch.zeros(
        n_logits,
        adjacency_matrix.shape[0],
        dtype=adjacency_matrix.dtype,
        device=adjacency_matrix.device,
    )
    logit_basis[
        torch.arange(n_logits, device=adjacency_matrix.device), 
        logit_start + torch.arange(n_logits, device=adjacency_matrix.device)
    ] = 1

    logit_influence = compute_node_influence(adjacency_matrix, logit_basis)
    logit_probabilities = graph.logit_probabilities.to(adjacency_matrix.device)
    node_influence_vectors = (
        logit_influence.T[:n_neurons] * (1 - logit_probabilities) * logit_probabilities
    )
    node_influence_normalized = node_influence_vectors / node_influence_vectors.norm(
        dim=-1, keepdim=True
    ).clamp(min=1e-12)

    def build_supernodes_svd(B_weighted, coverage_threshold, min_cluster_influence):
        """
        Cluster on directions, validate on magnitudes.
        """
        # magnitudes — used for noise filtering and validity
        magnitudes = B_weighted.norm(dim=1)           # [n_neurons]
        
        # directions — used for clustering
        directions = B_weighted / magnitudes.unsqueeze(1).clamp(min=1e-12)
        
        # filter noise before clustering
        if min_cluster_influence is None:
            min_cluster_influence = magnitudes.median() * 0.1
        valid_mask = magnitudes >= min_cluster_influence
        
        valid_directions = directions[valid_mask]      # [n_valid × n_logits]
        valid_magnitudes = magnitudes[valid_mask]      # [n_valid]
        if len(valid_directions) == 0:
            return torch.full((n_neurons,), -1, dtype=torch.long, device=B_weighted.device), 0
        
        # determine k: how many directions explain coverage_threshold 
        # of total influence mass?
        # use magnitude-weighted PCA instead of unweighted SVD
        weighted_B = valid_directions * valid_magnitudes.unsqueeze(1)
        U, S, Vt = torch.linalg.svd(weighted_B, full_matrices=False)
        
        total = (S ** 2).sum()
        if total <= 0:
            return torch.full((n_neurons,), -1, dtype=torch.long, device=B_weighted.device), 0
        cumulative = torch.cumsum(S ** 2, dim=0) / total
        k = min((cumulative < coverage_threshold).sum().item() + 1, len(valid_directions))
        
        # project directions (not raw B) into SVD subspace for clustering
        # this gives balanced coordinates — no singular value dominance
        direction_projections = valid_directions @ Vt[:k].T   # [n_valid × k]
        
        # k-means on unit-normalized direction projections
        assignments_valid = kmeans(direction_projections, k)
        
        # map back
        assignments = torch.full((n_neurons,), -1, dtype=torch.long, device=B_weighted.device)
        assignments[valid_mask] = assignments_valid + 1
        
        return assignments, k
    
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

    adj_matrix_norm = normalize_matrix(adjacency_matrix)
    supernodes = [[i for i in range(n_neurons) if node_clusters[i] == n] for n in range(1, num_supernodes + 1)]
    supernode_adj_matrix = _supernode_adjacency_from_sparse(
        adj_matrix_norm,
        supernodes,
        dtype=adjacency_matrix.dtype,
        device=adjacency_matrix.device,
    )

    return SuperGraph(supernode_adjacency_matrix=supernode_adj_matrix, supernodes=supernodes)

def range_query(node_influence_normalized: torch.Tensor, neuron_idx: int, epsilon: float) -> list[int]:
    """Find all nodes within epsilon of node_idx in the normalized node influence space."""
    return torch.where(1 - node_influence_normalized @ node_influence_normalized[neuron_idx] <= epsilon)[0].tolist()


def kmeans(X, k, n_iter=100):
    # X: [n × d]
    if len(X) == 0:
        raise ValueError("kmeans requires at least one point")
    if k <= 0:
        raise ValueError("kmeans requires k > 0")
    k = min(k, len(X))

    # initialize centers with kmeans++
    centers = [X[torch.randint(len(X), (1,)).item()]]
    for _ in range(k - 1):
        dists = torch.stack([
            ((X - c) ** 2).sum(dim=1) for c in centers
        ]).min(dim=0).values
        if dists.sum() <= 0:
            centers.append(X[torch.randint(len(X), (1,)).item()])
        else:
            probs = dists / dists.sum()
            centers.append(X[torch.multinomial(probs, 1).item()])
    centers = torch.stack(centers)  # [k × d]

    for _ in range(n_iter):
        # assign
        dists = torch.cdist(X, centers)          # [n × k]
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

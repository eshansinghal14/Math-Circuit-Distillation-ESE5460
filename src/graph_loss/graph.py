"""Graph container and influence utilities for LLaMA neuron attribution."""

from __future__ import annotations

import logging
import importlib
from typing import NamedTuple

import matplotlib.pyplot as plt
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


def build_super_graph(
    graph: Graph,
    model,
    prune_result: PruneResult | None = None,
) -> SuperGraph:
    """Create a single supernode from the selected output-node neurons."""

    if prune_result is not None:
        graph = graph.apply_prune_result(prune_result)

    n_logits = graph.n_logits
    n_tokens = graph.n_tokens
    n_neurons = graph.n_neurons
    token_start = n_neurons
    logit_start = token_start + n_tokens
    logger = logging.getLogger(__name__)
    kept_neuron_mask = torch.ones(n_neurons, dtype=torch.bool, device=graph.adjacency_device)
    if prune_result is not None:
        kept_neuron_mask = prune_result.node_mask[:n_neurons].to(
            device=graph.adjacency_device,
            dtype=torch.bool,
        )

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

    def find_output_elbow_count(sorted_scores: torch.Tensor) -> int:
        n_scores = int(sorted_scores.numel())
        if n_scores <= 2:
            return n_scores

        score_range = sorted_scores[0] - sorted_scores[-1]
        if score_range.abs() <= 1e-12:
            return n_scores

        x = torch.linspace(0, 1, n_scores, device=sorted_scores.device, dtype=sorted_scores.dtype)
        y = (sorted_scores - sorted_scores[-1]) / (
            sorted_scores[0] - sorted_scores[-1]
        ).clamp(min=1e-12)
        distance_from_diagonal = (1 - x) - y
        return int(distance_from_diagonal.argmax().item()) + 1

    def hdbscan_cossim(X: torch.Tensor) -> torch.Tensor:
        if len(X) == 0:
            raise ValueError("hdbscan_cossim requires at least one point")
        if len(X) == 1:
            return torch.zeros(1, dtype=torch.long, device=X.device)

        X_norm = X / X.norm(dim=1, keepdim=True).clamp(min=1e-12)
        distances = (1 - X_norm @ X_norm.T).clamp(min=0).detach().float().cpu().numpy()
        HDBSCAN = importlib.import_module("sklearn.cluster").HDBSCAN
        clusterer = HDBSCAN(
            metric="precomputed",
            min_cluster_size=2,
            min_samples=1,
            allow_single_cluster=True,
        )
        raw_labels = clusterer.fit_predict(distances).tolist()
        next_cluster = max(raw_labels, default=-1) + 1
        labels = []
        for label in raw_labels:
            if label == -1:
                labels.append(next_cluster)
                next_cluster += 1
            else:
                labels.append(int(label))
        return torch.tensor(labels, dtype=torch.long, device=X.device)

    def silhouette_score_cossim(X: torch.Tensor, assignments: torch.Tensor) -> torch.Tensor:
        unique_assignments = torch.unique(assignments)
        if len(unique_assignments) <= 1 or len(unique_assignments) >= len(X):
            return torch.tensor(float("nan"), device=X.device)

        X_norm = X / X.norm(dim=1, keepdim=True).clamp(min=1e-12)
        dists = 1 - X_norm @ X_norm.T
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

    def output_dla_hdbscan_assignments(output_dla: torch.Tensor) -> torch.Tensor:
        if len(output_dla) == 0:
            logger.info("  Output-node normalized DLA HDBSCAN: no vectors")
            return torch.empty(0, dtype=torch.long, device=output_dla.device)

        assignments = hdbscan_cossim(output_dla)
        unique_assignments, cluster_sizes = torch.unique(assignments, return_counts=True)
        score = silhouette_score_cossim(output_dla, assignments)
        logger.info(
            "  Output-node normalized DLA HDBSCAN: clusters=%d min_size=%d max_size=%d silhouette_score=%.6g",
            int(unique_assignments.numel()),
            int(cluster_sizes.min().item()),
            int(cluster_sizes.max().item()),
            float(score.item()),
        )
        return assignments

    def build_output_node():
        kept_neurons = torch.where(kept_neuron_mask)[0]
        device = model.unembed.W_U.device
        W_U = model.unembed.W_U.to(device=device)
        w_out_cache = {}
        write_vectors = []
        activation_magnitudes = []

        for neuron_idx in kept_neurons.tolist():
            layer = int(graph.neuron_locations[neuron_idx, 0].item())
            neuron_id = int(graph.neuron_locations[neuron_idx, 2].item())
            if layer not in w_out_cache:
                old_mlp = model.blocks[layer].mlp.old_mlp
                w_out_cache[layer] = model._row_oriented_weight(old_mlp.W_out.to(device=device))

            activation = graph.neuron_activations[neuron_idx].to(device=device, dtype=W_U.dtype)
            write_vectors.append(activation * w_out_cache[layer][neuron_id].to(dtype=W_U.dtype))
            activation_magnitudes.append(activation.abs())

        if not write_vectors:
            return (
                kept_neurons[:0],
                torch.empty(0, device=kept_neurons.device),
                torch.empty((0, W_U.shape[1]), device=device, dtype=W_U.dtype),
                0,
                torch.empty(0),
            )

        dla_matrix = torch.stack(write_vectors) @ W_U
        target_vocab_indices = torch.tensor(
            [target.vocab_idx for target in graph.logit_targets],
            device=device,
            dtype=torch.long,
        )

        dla_normalized = dla_matrix / dla_matrix.norm(dim=1, keepdim=True).clamp(min=1e-12)
        target_probabilities = logit_probabilities.to(device=device, dtype=dla_matrix.dtype)
        activation_magnitudes_t = torch.stack(activation_magnitudes).clamp(min=1e-12)
        neuron_scores = (
            dla_matrix[:, target_vocab_indices] * target_probabilities
        ).sum(dim=-1)
        sorted_scores, sorted_positions = torch.sort(neuron_scores, descending=True)
        elbow_count = find_output_elbow_count(sorted_scores)
        output_node_positions = sorted_positions[:elbow_count].to(device=kept_neurons.device)
        output_dla = dla_matrix[sorted_positions[:elbow_count]]
        if output_dla.numel():
            log_softmax_output_dla = torch.log_softmax(output_dla.detach().float(), dim=1)
            output_entropies = -(log_softmax_output_dla.exp() * log_softmax_output_dla).sum(dim=1)
            sorted_negative_output_entropies = torch.sort(-output_entropies, descending=True).values
            plt.figure(figsize=(8, 4))
            plt.plot(sorted_negative_output_entropies.cpu().tolist(), marker="o")
            plt.xlabel("Output-node neuron rank")
            plt.ylabel("Negative softmax DLA entropy")
            plt.title("Output-node negative softmax DLA entropies, sorted high to low")
            plt.tight_layout()
            plt.savefig("output_node_entropies.png")
            plt.close()

        return (
            kept_neurons[output_node_positions],
            neuron_scores[output_node_positions.to(device=neuron_scores.device)].to(
                device=kept_neurons.device,
            ),
            output_dla,
            elbow_count,
            sorted_scores.detach().float().cpu(),
        )

    def decode_vocab_token(vocab_idx: int) -> str:
        token = model.tokenizer.decode([vocab_idx])
        return token.replace("\n", "\\n").replace("\r", "\\r")

    def format_top_dla_logit_lines(
        dla_vector: torch.Tensor,
        top_k: int = 30,
        line_size: int = 10,
    ) -> list[str]:
        if dla_vector.numel() == 0:
            return ["none"]
        k = min(top_k, int(dla_vector.numel()))
        values, indices = torch.topk(dla_vector.detach().float().cpu(), k=k)
        formatted = [
            f"{decode_vocab_token(int(idx.item()))!r}: {float(value.item()):.6g}"
            for value, idx in zip(values, indices, strict=True)
        ]
        return [
            ", ".join(formatted[start:start + line_size])
            for start in range(0, len(formatted), line_size)
        ]

    def format_top_output_logits(top_k: int = 20) -> str:
        if logit_probabilities.numel() == 0:
            return "none"
        k = min(top_k, int(logit_probabilities.numel()))
        values, indices = torch.topk(logit_probabilities.detach().float().cpu(), k=k)
        return ", ".join(
            f"{graph.logit_targets[int(idx.item())].token_str!r}:{float(value.item()):.6g}"
            for value, idx in zip(values, indices, strict=True)
        )

    def softmax_dla_entropy(dla_vector: torch.Tensor) -> float:
        log_probs = torch.log_softmax(dla_vector.detach().float(), dim=0)
        return float((-(log_probs.exp() * log_probs).sum()).item())

    def log_output_node(output_node_indices, output_node_scores, output_node_dla, elbow_count, all_sorted_scores):
        logger.info("  Output node members: %d", int(output_node_indices.numel()))
        sorted_positions = torch.argsort(output_node_scores, descending=True)
        output_node_indices = output_node_indices[sorted_positions]
        output_node_scores = output_node_scores[sorted_positions]
        output_node_dla = output_node_dla[sorted_positions.to(device=output_node_dla.device)]
        if all_sorted_scores.numel():
            plt.figure(figsize=(8, 4))
            plt.plot(all_sorted_scores.tolist(), marker="o")
            if 0 < elbow_count <= int(all_sorted_scores.numel()):
                plt.axvline(elbow_count - 1, color="red", linestyle="--")
            plt.xlabel("Neuron rank")
            plt.ylabel("Normalized DLA score")
            plt.title("Output-node normalized DLA scores, sorted high to low")
            plt.tight_layout()
            plt.savefig("output_node_scores.png")
            plt.close()

        if output_node_scores.numel():
            logger.info(
                "  Output node elbow: selected=%d elbow_score=%.6g max_score=%.6g",
                int(elbow_count),
                float(output_node_scores[-1].item()),
                float(output_node_scores[0].item()),
            )
            logger.info(
                "  top 20 logits: %s",
                format_top_output_logits(top_k=20),
            )

        cluster_assignments = output_dla_hdbscan_assignments(output_node_dla)
        unique_cluster_ids = torch.unique(cluster_assignments).tolist()
        for cluster_id in unique_cluster_ids:
            cluster_mask = cluster_assignments == int(cluster_id)
            cluster_rows = torch.where(cluster_mask)[0].tolist()
            mean_dla_norm = output_node_dla[cluster_mask].detach().float().mean(dim=0).norm()
            logger.info(
                "  cluster %d size=%d mean_dla_norm=%.6g",
                int(cluster_id),
                len(cluster_rows),
                float(mean_dla_norm.item()),
            )
            for row_idx in cluster_rows:
                neuron_idx = int(output_node_indices[row_idx].item())
                score = float(output_node_scores[row_idx].item())
                layer = int(graph.neuron_locations[neuron_idx, 0].item())
                token_pos = int(graph.neuron_locations[neuron_idx, 1].item())
                neuron_id = int(graph.neuron_locations[neuron_idx, 2].item())
                logger.info(
                    "    graph_neuron_idx=%d layer=%d token=%d neuron=%d score=%.6g entropy=%.6g",
                    neuron_idx,
                    layer,
                    token_pos,
                    neuron_id,
                    score,
                    softmax_dla_entropy(output_node_dla[row_idx]),
                )
                for line_idx, line in enumerate(
                    format_top_dla_logit_lines(output_node_dla[row_idx], top_k=30, line_size=10),
                    start=1,
                ):
                    logger.info("      top DLA logits %d: %s", line_idx, line)

        if output_node_indices.numel() == 0:
            logger.info("  Output node normalized DLA cossim matrix: []")
            return

        cossim_matrix = output_node_dla @ output_node_dla.T
        logger.info("  Output node normalized DLA cossim matrix:")
        for row in cossim_matrix.detach().float().cpu().tolist():
            logger.info("    [%s]", ", ".join(f"{value:.6g}" for value in row))

    logger.info("  Building output node")
    (
        output_node_indices,
        output_node_scores,
        output_node_dla,
        output_node_count,
        all_output_scores,
    ) = build_output_node()
    log_output_node(
        output_node_indices,
        output_node_scores,
        output_node_dla,
        output_node_count,
        all_output_scores,
    )

    logger.info("  Aggregating output-node supergraph")
    adj_matrix_norm = normalize_matrix(adjacency_matrix)
    supernodes = [output_node_indices.tolist()]
    num_supernodes = len(supernodes)
    supernode_adj_matrix = torch.zeros(
        num_supernodes,
        num_supernodes,
        dtype=adj_matrix_norm.dtype,
        device=adjacency_matrix.device,
    )
    for t in range(num_supernodes):
        total_input = torch.abs(adj_matrix_norm[:, supernodes[t]]).sum(dim=0)
        internal_input = torch.abs(adj_matrix_norm[supernodes[t]][:, supernodes[t]]).sum(dim=0)
        frac_external = (total_input - internal_input) / total_input.clamp(min=1e-10)
        
        for s in range(num_supernodes):
            sum_A = adj_matrix_norm[supernodes[t]][:, supernodes[s]].sum(dim=1)
            supernode_adj_matrix[t, s] = (frac_external * sum_A).sum(dim=0) / frac_external.sum(dim=0).clamp(min=1e-10)

    return SuperGraph(supernode_adjacency_matrix=supernode_adj_matrix, supernodes=supernodes)


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

"""Graph container and influence utilities for LLaMA neuron attribution."""

from __future__ import annotations

import logging
import os
from typing import NamedTuple

import torch
import torch.nn.functional as F

from graph_loss.anova_node_labels import ANOVA_LABEL_CATEGORIES
from graph_loss.attribution.targets import LogitTarget
from graph_loss.neuron_activation_heatmap import save_supernode_activation_heatmap_pdf
from graph_loss.utils import (
    ActivationWriteResult,
    UnifiedConfig,
    convert_nnsight_config_to_transformerlens,
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
            neuron_write_vectors=data.get("neuron_write_vectors"),
        )


def normalize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    math_dtype = (
        torch.float32
        if matrix.dtype in (torch.float16, torch.bfloat16)
        else matrix.dtype
    )
    normalized = matrix.to(dtype=math_dtype).abs()
    return normalized / normalized.sum(dim=1, keepdim=True).clamp(min=1e-10)


def compute_neuron_logit_influence(graph: Graph) -> torch.Tensor:
    """Return direct graph attribution from each neuron to each selected logit."""
    logit_start = graph.n_neurons + graph.n_tokens
    logit_rows = slice(logit_start, logit_start + graph.n_logits)
    return graph.adjacency_matrix[logit_rows, : graph.n_neurons].transpose(0, 1)


class SuperGraph(NamedTuple):
    supernode_adjacency_matrix: torch.Tensor
    supernodes: list[list[int]]
    node_labels: dict[int, list[str]] | None = None
    supernode_labels: list[list[str]] | None = None
    supernode_heatmap_pdf_paths: list[str] | None = None


def _dla_kl_scores_for_sum(
    source_vectors: torch.Tensor,
    W_U: torch.Tensor,
    tokenizer,
    target_value: int,
    units: bool,
    min_value: int = 0,
    max_value: int = 200,
) -> list[float]:
    """Compute DLA-based KL divergence scores for sum-category neurons.

    Projects source_vectors (neuron write vectors from setup_attribution) onto W_U
    to get direct logit contributions, then computes KL divergence vs the target
    distribution over number tokens.

    Returns a list of N scores where higher (less negative KL) = better match
    to the target sum or sum-units distribution.
    """
    number_values = list(range(int(min_value), int(max_value) + 1))
    token_ids: list[int | None] = []
    for value in number_values:
        encoded = tokenizer(str(value), add_special_tokens=False, return_tensors=None)
        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        token_ids.append(int(input_ids[0]) if len(input_ids) == 1 else None)

    valid_positions = [idx for idx, tid in enumerate(token_ids) if tid is not None]
    if not valid_positions:
        return [0.0] * source_vectors.shape[0]

    target_unit = int(target_value) % 10
    target_weights = torch.zeros(len(valid_positions), dtype=torch.float32)
    for pos_in_valid, orig_pos in enumerate(valid_positions):
        number = number_values[orig_pos]
        match = (number % 10 == target_unit) if units else (number == int(target_value))
        if match:
            target_weights[pos_in_valid] = 1.0
    if float(target_weights.sum().item()) == 0.0:
        return [0.0] * source_vectors.shape[0]
    Q = target_weights / target_weights.sum()

    valid_vocab_ids = torch.tensor(
        [int(token_ids[pos]) for pos in valid_positions],
        dtype=torch.long,
        device=W_U.device,
    )
    W_U_numbers = W_U[:, valid_vocab_ids]  # [d_model, n_valid]

    sv = source_vectors.to(device=W_U.device, dtype=W_U.dtype)
    dla_logits = sv @ W_U_numbers  # [N, n_valid]
    P = F.softmax(dla_logits.float(), dim=-1)  # [N, n_valid]

    Q = Q.to(device=P.device)
    nonzero = Q > 0
    Q_nz = Q[nonzero]
    P_nz = P[:, nonzero].clamp(min=1e-10)
    kl = (Q_nz * (Q_nz.log() - P_nz.log())).sum(dim=-1)  # [N]
    return (-kl).detach().cpu().tolist()


def select_anova_supernodes(
    label_results: list,
    anova_nodes_per_label: int,
    sum_min_specificity: float = 0.0,
    strict: bool = True,
    source_vectors: torch.Tensor | None = None,
    W_U: torch.Tensor | None = None,
    tokenizer=None,
    target_args: list[int] | None = None,
) -> tuple[list[int], list[list[int]], list[list[str]], dict[int, list[str]]]:
    """Select ANOVA supernodes from pre-computed label results.

    For non-sum categories ranks by ANOVA variance score.
    For sum categories uses DLA-KL scoring if source_vectors/W_U/tokenizer/target_args
    are all provided; otherwise falls back to ANOVA variance.

    Args:
        label_results: Output of label_activation_heatmaps(). Entry i corresponds
            to neuron index i passed to build_neuron_activation_write_result.
        anova_nodes_per_label: Max neurons per ANOVA label category.
        sum_min_specificity: Min ANOVA specificity for sum-range/sum-units candidates.
        strict: Raise ValueError when a non-sum category has no positive-variance nodes.
        source_vectors: [N, d_model] neuron write vectors from setup_attribution.
        W_U: [d_model, d_vocab] unembedding matrix.
        tokenizer: Model tokenizer for mapping numbers to token IDs.
        target_args: Numeric arguments from the prompt (e.g. [arg1, arg2]).

    Returns:
        selected_row_indices: sorted unique row indices into label_results that appear
            in at least one supernode.
        supernodes: list of supernodes; each is a list of row indices into label_results.
        supernode_labels: list of [category_name] per supernode.
        node_labels: dict from row_index -> list of label strings.
    """
    logger = logging.getLogger(__name__)
    use_dla = (
        source_vectors is not None
        and W_U is not None
        and tokenizer is not None
        and target_args is not None
        and len(target_args) >= 2
    )
    target_sum = int(target_args[0] + target_args[1]) if use_dla else 0

    selected_member_ids: set[int] = set()
    supernodes: list[list[int]] = []
    supernode_labels_out: list[list[str]] = []
    node_labels: dict[int, list[str]] = {}

    for category in ANOVA_LABEL_CATEGORIES:
        is_sum_category = category in {"sum range", "sum units"}

        if is_sum_category:
            candidates = [
                row_idx
                for row_idx, lr in enumerate(label_results)
                if category in lr.category_scores
                and lr.category_scores[category] > 0.0
                and lr.category_specificity.get(category, 0.0) > sum_min_specificity
            ]
            if candidates and use_dla:
                kl_all = _dla_kl_scores_for_sum(
                    source_vectors,  # type: ignore[arg-type]
                    W_U,             # type: ignore[arg-type]
                    tokenizer,
                    target_value=target_sum,
                    units=(category == "sum units"),
                )
                all_scored_rows = [(row_idx, kl_all[row_idx]) for row_idx in candidates]
            else:
                all_scored_rows = [
                    (row_idx, label_results[row_idx].category_scores[category])
                    for row_idx in candidates
                ]
        else:
            all_scored_rows = [
                (row_idx, lr.category_scores[category])
                for row_idx, lr in enumerate(label_results)
                if category in lr.category_scores
                and lr.category_scores[category] > 0.0
            ]

        sorted_rows = sorted(all_scored_rows, key=lambda item: item[1], reverse=True)

        if not sorted_rows:
            if strict and not is_sum_category:
                raise ValueError(
                    f"ANOVA category {category!r} has no positive-variance nodes."
                )
            if is_sum_category:
                logger.info(
                    "  ANOVA label %s: no candidates above sum_min_specificity=%.6g",
                    category,
                    sum_min_specificity,
                )
            else:
                logger.info("  ANOVA label %s: no positive-variance nodes", category)
            continue

        keep_count = min(anova_nodes_per_label, len(sorted_rows))
        top_rows = sorted_rows[:keep_count]

        for row_idx, _score in top_rows:
            label = label_results[row_idx].categories.get(category, category)
            node_labels.setdefault(row_idx, [])
            if label not in node_labels[row_idx]:
                node_labels[row_idx].append(label)
            selected_member_ids.add(row_idx)

        members = [row_idx for row_idx, _score in top_rows]
        supernodes.append(members)
        supernode_labels_out.append([category])
        logger.info(
            "  ANOVA label %s: selected=%d/%d cap=%d best_score=%.6g",
            category,
            len(members),
            len(sorted_rows),
            anova_nodes_per_label,
            float(top_rows[0][1]),
        )

    selected_row_indices = sorted(selected_member_ids)
    logger.info(
        "  ANOVA selection: unique_neurons=%d / total_candidates=%d",
        len(selected_row_indices),
        len(label_results),
    )
    return selected_row_indices, supernodes, supernode_labels_out, node_labels


def build_super_graph(
    graph: Graph,
    supernodes: list[list[int]],
    supernode_labels: list[list[str]],
    node_labels: dict[int, list[str]] | None = None,
    supernode_heatmap_output_dir: str | None = None,
    activation_write_result: ActivationWriteResult | None = None,
) -> SuperGraph:
    """Aggregate the attribution adjacency matrix into a supergraph.

    Supernodes are lists of neuron row indices already remapped to the filtered
    graph (i.e. members index graph.neuron_locations directly).
    """
    logger = logging.getLogger(__name__)
    adjacency_matrix = graph.adjacency_matrix
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
            supernode_adj_matrix[t, s] = (
                (frac_external * sum_A).sum(dim=0)
                / frac_external.sum(dim=0).clamp(min=1e-10)
            )

    supernode_heatmap_pdf_paths: list[str] | None = None
    if supernode_heatmap_output_dir is not None:
        if activation_write_result is None:
            logger.warning(
                "supernode_heatmap_output_dir set but activation_write_result not provided; "
                "skipping heatmap PDF generation."
            )
        else:
            supernode_heatmap_pdf_paths = []
            neuron_locs = graph.neuron_locations.detach().cpu()
            # Compute per-neuron proportion of total residual write-norm.
            neuron_norm_props: dict[int, float] | None = None
            if graph.neuron_write_vectors is not None:
                norms = graph.neuron_write_vectors.detach().float().norm(dim=-1)
                total_norm = norms.sum().item()
                if total_norm > 0:
                    neuron_norm_props = {
                        i: float(norms[i].item()) / total_norm
                        for i in range(len(norms))
                    }
            for supernode_idx, members in enumerate(supernodes):
                category = (
                    supernode_labels[supernode_idx][0]
                    if supernode_labels[supernode_idx]
                    else "none"
                )
                row_indices = torch.tensor(members, dtype=torch.long)
                cluster_heatmaps = activation_write_result.activations[row_indices].detach().float()
                member_labels = (
                    {m: node_labels.get(m, []) for m in members}
                    if node_labels
                    else None
                )
                member_norm_props = (
                    {m: neuron_norm_props[m] for m in members if m in neuron_norm_props}
                    if neuron_norm_props is not None
                    else None
                )
                saved_path = save_supernode_activation_heatmap_pdf(
                    cluster_heatmaps,
                    activation_write_result.arg_values,
                    members,
                    neuron_locs,
                    output_path=os.path.join(
                        supernode_heatmap_output_dir,
                        f"supernode_{supernode_idx}.pdf",
                    ),
                    title=f"supernode {supernode_idx}: {category}",
                    member_labels=member_labels,
                    member_norm_props=member_norm_props,
                )
                logger.info("  Saved supernode heatmap PDF: %s", saved_path)
                supernode_heatmap_pdf_paths.append(saved_path)

    logger.info("  Aggregated supergraph: %d supernodes", num_supernodes)
    return SuperGraph(
        supernode_adjacency_matrix=supernode_adj_matrix,
        supernodes=supernodes,
        node_labels=node_labels,
        supernode_labels=supernode_labels,
        supernode_heatmap_pdf_paths=supernode_heatmap_pdf_paths,
    )

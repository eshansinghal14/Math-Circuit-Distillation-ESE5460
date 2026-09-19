"""Graph container and influence utilities for LLaMA neuron attribution."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn.functional as F

from graph_loss.anova_node_labels import ANOVA_LABEL_CATEGORIES, LabelTable
from graph_loss.attribution.targets import LogitTarget
from graph_loss.neuron_activation_heatmap import (
    save_dla_heatmap_pdf,
    save_supernode_activation_heatmap_pdf,
)
from graph_loss.utils import (
    ActivationWriteResult,
    UnifiedConfig,
    convert_nnsight_config_to_transformerlens,
)

if TYPE_CHECKING:
    from graph_loss.attribution.context import HFAttributionContext


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


def _dla_kl_scores_for_output(
    source_vectors: torch.Tensor,
    W_U: torch.Tensor,
    model_logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 100,
) -> list[float]:
    """Compute DLA-KL scores vs the model's actual output distribution.

    Restricts comparison to the top-K tokens from the model output distribution
    to avoid memory issues with large vocabularies.  Q is the renormalized model
    output distribution over those tokens; P is the softmax of each neuron's DLA
    logits over the same tokens.  Returns -KL(Q || P) so that higher = better.

    Args:
        source_vectors: [N, d_model] neuron write vectors from setup_attribution.
        W_U: [d_model, d_vocab] unembedding matrix.
        model_logits: [d_vocab] raw logits from the forward pass (last token position).
        temperature: Softmax temperature applied to model_logits and DLA logits.
        top_k: Number of top model-output tokens to restrict the KL comparison to.

    Returns:
        list of N floats; higher (less negative KL) = neuron DLA closer to model output.
    """
    model_probs = torch.softmax(model_logits.float() / temperature, dim=-1)  # [d_vocab]
    k = min(top_k, model_probs.numel())
    top_probs, top_indices = torch.topk(model_probs, k)  # [k]
    Q = (top_probs / top_probs.sum()).to(device=W_U.device)  # renormalized [k]

    W_U_top = W_U[:, top_indices.to(device=W_U.device)]  # [d_model, k]
    sv = source_vectors.to(device=W_U.device, dtype=W_U.dtype)
    dla_logits = sv @ W_U_top  # [N, k]
    P = F.softmax(dla_logits.float(), dim=-1)  # [N, k]

    nonzero = Q > 0
    Q_nz = Q[nonzero]
    P_nz = P[:, nonzero].clamp(min=1e-10)
    kl = (Q_nz * (Q_nz.log() - P_nz.log())).sum(dim=-1)  # [N]
    return (-kl).detach().cpu().tolist()


def select_anova_supernodes(
    label_results: list,
    nodes_per_label: int,
    strict: bool = True,
    source_vectors: torch.Tensor | None = None,
    W_U: torch.Tensor | None = None,
    tokenizer=None,
    target_args: list[int] | None = None,
    allowed_labels: set[str] | None = None,
    include_dla_node: bool = False,
    model_logits: torch.Tensor | None = None,
    dla_temperature: float = 1.0,
    dla_top_k_vocab: int = 100,
) -> tuple[list[int], list[list[int]], list[list[str]], dict[int, list[str]], dict[str, dict[int, tuple[float, float, float]]]]:
    """Select ANOVA supernodes from pre-computed label results.

    For non-sum categories ranks by ANOVA specificity score.
    For sum categories ranks by DLA-KL (requires source_vectors/W_U/tokenizer/target_args).

    Optionally creates one additional "dla" supernode containing the top
    ``nodes_per_label`` neurons whose DLA distribution (neuron write vector
    projected through W_U, softmaxed) most closely matches the model's actual
    output distribution (lowest KL divergence).

    Args:
        label_results: Output of gpu_label_activation_heatmaps(). Entry i corresponds
            to neuron index i passed to build_neuron_activation_write_result.
        nodes_per_label: Max neurons per ANOVA label category (also used as the
            size cap for the DLA supernode).
        strict: Raise ValueError when a non-sum category has no positive-variance nodes.
        source_vectors: [N, d_model] neuron write vectors from setup_attribution.
        W_U: [d_model, d_vocab] unembedding matrix.
        tokenizer: Model tokenizer for mapping numbers to token IDs.
        target_args: Numeric arguments from the prompt (e.g. [arg1, arg2]).
        include_dla_node: If True, add a "dla" supernode ranked by KL divergence
            between each neuron's DLA distribution and the model output distribution.
            Requires source_vectors, W_U, and model_logits.
        model_logits: [d_vocab] raw logits from the forward pass (last token position).
            Required when include_dla_node is True.
        dla_temperature: Softmax temperature for both model output and DLA distributions.
        dla_top_k_vocab: Number of top model-output tokens to restrict the KL comparison
            to (avoids O(N × d_vocab) memory usage).

    Returns:
        selected_row_indices: sorted unique row indices into label_results that appear
            in at least one supernode.
        supernodes: list of supernodes; each is a list of row indices into label_results.
        supernode_labels: list of [category_name] per supernode.
        node_labels: dict from row_index -> list of label strings.
        sum_member_scores: dict from category -> {row_idx -> (var, spec, dla_cossim)}
            for sum-category supernodes only.
    """
    if isinstance(label_results, LabelTable):
        return _select_anova_supernodes_table(
            label_results, nodes_per_label=nodes_per_label, strict=strict,
            source_vectors=source_vectors, W_U=W_U, tokenizer=tokenizer, target_args=target_args,
            allowed_labels=allowed_labels, include_dla_node=include_dla_node,
            model_logits=model_logits, dla_temperature=dla_temperature, dla_top_k_vocab=dla_top_k_vocab,
        )
    logger = logging.getLogger(__name__)
    use_dla = (
        source_vectors is not None
        and W_U is not None
        and tokenizer is not None
        and target_args is not None
        and len(target_args) >= 2
    )
    target_sum = int(sum(target_args)) if use_dla else 0

    selected_member_ids: set[int] = set()
    supernodes: list[list[int]] = []
    supernode_labels_out: list[list[str]] = []
    node_labels: dict[int, list[str]] = {}
    sum_member_scores: dict[str, dict[int, tuple[float, float, float]]] = {}

    _base_categories = set(ANOVA_LABEL_CATEGORIES)
    _extra_categories = (
        [lbl for lbl in sorted(allowed_labels) if lbl not in _base_categories]
        if allowed_labels
        else []
    )
    for category in list(ANOVA_LABEL_CATEGORIES) + _extra_categories:
        if allowed_labels is not None and category not in allowed_labels:
            continue
        is_sum_category = category in {"sum range", "sum units"}

        if is_sum_category:
            candidates = [
                row_idx
                for row_idx, lr in enumerate(label_results)
                if category in lr.category_scores
                and lr.category_scores[category] > 0.0
            ]
            if use_dla:
                kl_all = _dla_kl_scores_for_sum(
                    source_vectors,  # type: ignore[arg-type]
                    W_U,             # type: ignore[arg-type]
                    tokenizer,
                    target_value=target_sum,
                    units=(category == "sum units"),
                )
                all_scored_rows = [(row_idx, kl_all[row_idx]) for row_idx in candidates]
            else:
                kl_all = None
                all_scored_rows = []
        else:
            all_scored_rows = [
                (row_idx, lr.category_specificity.get(category, 0.0))
                for row_idx, lr in enumerate(label_results)
                if category in lr.category_scores
                and lr.category_scores[category] > 0.0
            ]

        sorted_rows = sorted(all_scored_rows, key=lambda item: item[1], reverse=True)

        if not sorted_rows:
            if strict and not is_sum_category and category in _base_categories:
                raise ValueError(
                    f"ANOVA category {category!r} has no positive-variance nodes."
                )
            explicitly_requested = allowed_labels is not None and category in allowed_labels
            if is_sum_category:
                logger.info("  ANOVA label %s: no DLA-scored candidates", category)
            elif explicitly_requested and category not in _base_categories:
                logger.warning(
                    "  ANOVA label %r: no positive-variance nodes — this may not be a valid "
                    "category name. Valid dynamic categories include 'arg1 range', "
                    "'arg1 arg2 sum range', 'sum range', etc.",
                    category,
                )
            else:
                logger.info("  ANOVA label %s: no positive-variance nodes", category)
            continue

        keep_count = min(nodes_per_label, len(sorted_rows))
        top_rows = sorted_rows[:keep_count]

        for row_idx, _score in top_rows:
            label = label_results[row_idx].categories.get(category, category)
            node_labels.setdefault(row_idx, [])
            if label not in node_labels[row_idx]:
                node_labels[row_idx].append(label)
            selected_member_ids.add(row_idx)

        if is_sum_category:
            sum_member_scores[category] = {
                row_idx: (
                    float(label_results[row_idx].category_scores.get(category, 0.0)),
                    float(label_results[row_idx].category_specificity.get(category, 0.0)),
                    float(-kl_all[row_idx]) if kl_all is not None else 0.0,
                )
                for row_idx, _ in top_rows
            }

        members = [row_idx for row_idx, _score in top_rows]
        supernodes.append(members)
        supernode_labels_out.append([category])
        logger.info(
            "  ANOVA label %s: selected=%d/%d cap=%d best_score=%.6g",
            category,
            len(members),
            len(sorted_rows),
            nodes_per_label,
            float(top_rows[0][1]),
        )

    # DLA supernode: neurons whose write-vector DLA distribution best matches the
    # model's actual output distribution (lowest KL divergence).
    if include_dla_node:
        if source_vectors is None or W_U is None or model_logits is None:
            logger.warning(
                "  include_dla_node=True but source_vectors/W_U/model_logits not all "
                "provided — skipping DLA supernode."
            )
        else:
            kl_scores = _dla_kl_scores_for_output(
                source_vectors,
                W_U,
                model_logits,
                temperature=dla_temperature,
                top_k=dla_top_k_vocab,
            )
            n_candidates = len(kl_scores)
            all_dla_rows = [(i, kl_scores[i]) for i in range(n_candidates)]
            sorted_dla = sorted(all_dla_rows, key=lambda item: item[1], reverse=True)
            keep_count = min(nodes_per_label, len(sorted_dla))
            top_dla = sorted_dla[:keep_count]
            dla_members = []
            for row_idx, _score in top_dla:
                node_labels.setdefault(row_idx, [])
                if "dla" not in node_labels[row_idx]:
                    node_labels[row_idx].append("dla")
                selected_member_ids.add(row_idx)
                dla_members.append(row_idx)
            supernodes.append(dla_members)
            supernode_labels_out.append(["dla"])
            logger.info(
                "  DLA supernode: selected=%d/%d cap=%d best_score=%.6g",
                len(dla_members),
                n_candidates,
                nodes_per_label,
                float(top_dla[0][1]) if top_dla else 0.0,
            )

    selected_row_indices = sorted(selected_member_ids)
    logger.info(
        "  ANOVA selection: unique_neurons=%d / total_candidates=%d",
        len(selected_row_indices),
        len(label_results),
    )
    return selected_row_indices, supernodes, supernode_labels_out, node_labels, sum_member_scores


def _select_anova_supernodes_table(
    table: LabelTable,
    *,
    nodes_per_label: int,
    strict: bool,
    source_vectors: torch.Tensor | None,
    W_U: torch.Tensor | None,
    tokenizer,
    target_args: list[int] | None,
    allowed_labels: set[str] | None,
    include_dla_node: bool,
    model_logits: torch.Tensor | None,
    dla_temperature: float,
    dla_top_k_vocab: int,
) -> tuple[list[int], list[list[int]], list[list[str]], dict[int, list[str]], dict[str, dict[int, tuple[float, float, float]]]]:
    """select_anova_supernodes on a LabelTable: the same selection, done with tensor ops.

    Per category the candidates are the rows with a positive category score,
    ranked by specificity (DLA-KL for the sum categories) with a stable
    descending sort, so ties resolve by row order exactly as the list path's
    ``sorted(..., reverse=True)`` does. Only the selected rows ever become
    Python objects. Ranking scores that the list path handled as Python floats
    are compared in float64 so no ordering can differ.
    """
    logger = logging.getLogger(__name__)
    use_dla = (
        source_vectors is not None
        and W_U is not None
        and tokenizer is not None
        and target_args is not None
        and len(target_args) >= 2
    )
    target_sum = int(sum(target_args)) if use_dla else 0
    dev = table.cat_scores.device

    selected_member_ids: set[int] = set()
    supernodes: list[list[int]] = []
    supernode_labels_out: list[list[str]] = []
    node_labels: dict[int, list[str]] = {}
    sum_member_scores: dict[str, dict[int, tuple[float, float, float]]] = {}

    _base_categories = set(ANOVA_LABEL_CATEGORIES)
    _extra_categories = (
        [lbl for lbl in sorted(allowed_labels) if lbl not in _base_categories]
        if allowed_labels
        else []
    )
    for category in list(ANOVA_LABEL_CATEGORIES) + _extra_categories:
        if allowed_labels is not None and category not in allowed_labels:
            continue
        is_sum_category = category in {"sum range", "sum units"}

        col = table.category_scores_column(category)
        if col is None:
            cand = torch.zeros(0, dtype=torch.long, device=dev)
        else:
            cand = torch.nonzero(col > 0.0, as_tuple=False).squeeze(1)
        kl_all: torch.Tensor | None = None
        if is_sum_category:
            if use_dla and cand.numel() > 0:
                kl_all = torch.tensor(
                    _dla_kl_scores_for_sum(
                        source_vectors,  # type: ignore[arg-type]
                        W_U,             # type: ignore[arg-type]
                        tokenizer,
                        target_value=target_sum,
                        units=(category == "sum units"),
                    ),
                    dtype=torch.float64, device=dev,
                )
                scores = kl_all[cand]
            else:
                cand = cand[:0]
                scores = cand.to(torch.float64)
        else:
            scores = table.specificity_column(category)[cand]

        if cand.numel() == 0:
            if strict and not is_sum_category and category in _base_categories:
                raise ValueError(
                    f"ANOVA category {category!r} has no positive-variance nodes."
                )
            explicitly_requested = allowed_labels is not None and category in allowed_labels
            if is_sum_category:
                logger.info("  ANOVA label %s: no DLA-scored candidates", category)
            elif explicitly_requested and category not in _base_categories:
                logger.warning(
                    "  ANOVA label %r: no positive-variance nodes — this may not be a valid "
                    "category name. Valid dynamic categories include 'arg1 range', "
                    "'arg1 arg2 sum range', 'sum range', etc.",
                    category,
                )
            else:
                logger.info("  ANOVA label %s: no positive-variance nodes", category)
            continue

        order = torch.sort(scores, descending=True, stable=True).indices
        keep_count = min(nodes_per_label, int(cand.numel()))
        top_rows = cand[order[:keep_count]].tolist()
        top_scores = scores[order[:keep_count]].tolist()

        label = table.label_for(category)
        for row_idx in top_rows:
            node_labels.setdefault(row_idx, [])
            if label not in node_labels[row_idx]:
                node_labels[row_idx].append(label)
            selected_member_ids.add(row_idx)

        if is_sum_category:
            spec_col = table.specificity_column(category)
            sum_member_scores[category] = {
                row_idx: (
                    float(col[row_idx].item()),
                    float(spec_col[row_idx].item()),
                    float(-kl_all[row_idx].item()) if kl_all is not None else 0.0,
                )
                for row_idx in top_rows
            }

        supernodes.append(list(top_rows))
        supernode_labels_out.append([category])
        logger.info(
            "  ANOVA label %s: selected=%d/%d cap=%d best_score=%.6g",
            category,
            len(top_rows),
            int(cand.numel()),
            nodes_per_label,
            float(top_scores[0]),
        )

    # DLA supernode: neurons whose write-vector DLA distribution best matches the
    # model's actual output distribution (lowest KL divergence).
    if include_dla_node:
        if source_vectors is None or W_U is None or model_logits is None:
            logger.warning(
                "  include_dla_node=True but source_vectors/W_U/model_logits not all "
                "provided — skipping DLA supernode."
            )
        else:
            kl_scores = torch.tensor(
                _dla_kl_scores_for_output(
                    source_vectors,
                    W_U,
                    model_logits,
                    temperature=dla_temperature,
                    top_k=dla_top_k_vocab,
                ),
                dtype=torch.float64, device=dev,
            )
            n_candidates = int(kl_scores.numel())
            order = torch.sort(kl_scores, descending=True, stable=True).indices
            keep_count = min(nodes_per_label, n_candidates)
            top_dla = order[:keep_count].tolist()
            dla_members = []
            for row_idx in top_dla:
                node_labels.setdefault(row_idx, [])
                if "dla" not in node_labels[row_idx]:
                    node_labels[row_idx].append("dla")
                selected_member_ids.add(row_idx)
                dla_members.append(row_idx)
            supernodes.append(dla_members)
            supernode_labels_out.append(["dla"])
            logger.info(
                "  DLA supernode: selected=%d/%d cap=%d best_score=%.6g",
                len(dla_members),
                n_candidates,
                nodes_per_label,
                float(kl_scores[top_dla[0]].item()) if top_dla else 0.0,
            )

    selected_row_indices = sorted(selected_member_ids)
    logger.info(
        "  ANOVA selection: unique_neurons=%d / total_candidates=%d",
        len(selected_row_indices),
        len(table),
    )
    return selected_row_indices, supernodes, supernode_labels_out, node_labels, sum_member_scores


def select_arg_supernodes(
    ctx: "HFAttributionContext",
    tokenizer,
    input_ids: torch.Tensor,
    nodes_per_token: int = 10,
) -> tuple[list[list[int]], list[list[str]]]:
    """Select arg-token supernodes via inner-product proxy.

    For each token position ``p``, selects the ``nodes_per_token`` neurons
    whose read direction (target_encoder) is most aligned with that token's
    embedding: ``score[i, q] = |target_encoders[i] · E[q]|``.

    Args:
        ctx: Attribution context with pre-selected neurons (unfiltered).
        tokenizer: Tokenizer used to decode token-position labels.
        input_ids: 1-D ``[seq_len]`` integer token IDs for the prompt.
        nodes_per_token: Maximum neurons per token-position supernode.

    Returns:
        raw_supernodes: list of ``seq_len`` lists of neuron row indices (into ``ctx``).
        supernode_labels: list of ``seq_len`` label lists, e.g. ``["arg:43"]``.
    """
    logger = logging.getLogger(__name__)
    n_neurons = ctx.n_neurons
    n_pos = ctx.n_tokens

    logger.info(
        "  [arg-nodes] computing inner-product scores for %d neurons",
        n_neurons,
    )
    E = ctx.embed_out.squeeze(0).to(ctx.target_encoders.dtype)  # [n_tokens, d_model]
    all_norms = (ctx.target_encoders @ E.T).abs()  # [n_neurons, n_tokens]

    # ── Step 2: normalise to distribution d_f(p) per neuron ──────────────────
    total = all_norms.sum(dim=-1, keepdim=True).clamp(min=1e-10)  # [n_neurons, 1]
    d_f = all_norms / total  # [n_neurons, n_pos]

    # ── Step 3: for each token position pick top-K neurons by d_f(p) ─────────
    token_ids_list = input_ids.detach().cpu().tolist()
    raw_supernodes: list[list[int]] = []
    supernode_labels_out: list[list[str]] = []

    bos_id = getattr(tokenizer, "bos_token_id", None)
    start_pos = 1 if (bos_id is not None and token_ids_list and token_ids_list[0] == bos_id) else 0

    for pos in range(start_pos, n_pos):
        scores = d_f[:, pos]  # [n_neurons]
        k = min(nodes_per_token, n_neurons)
        top_indices = torch.topk(scores, k, dim=0).indices.tolist()
        members = [int(idx) for idx in top_indices]

        token_id = int(token_ids_list[pos])
        try:
            token_str = tokenizer.decode([token_id])
        except Exception:
            token_str = str(token_id)

        label = f"arg:{token_str}"
        raw_supernodes.append(members)
        supernode_labels_out.append([label])

        best_score = float(scores[int(top_indices[0])].item()) if members else 0.0
        logger.info(
            "  [arg-nodes] pos=%d token=%r: selected=%d cap=%d best_d_f=%.4g",
            pos,
            token_str,
            len(members),
            nodes_per_token,
            best_score,
        )

    return raw_supernodes, supernode_labels_out


SUPERGRAPH_AGGREGATIONS = ("normalised", "raw-signed")
SUPERNODE_MEMBERSHIP_TYPES = ("topk", "soft")


def soft_membership_weights(
    scored_rows: list[tuple[int, float]],
    nodes_per_label: int,
    temperature: float,
    n_rows: int,
    *,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """A soft membership indicator in [0, 1] over ``n_rows`` graph rows.

    ``scored_rows`` is (row index, score) with higher scores meaning stronger
    membership -- exactly what ``select_anova_supernodes`` sorts before taking the
    top k. Instead of cutting, every candidate gets

        u(n) = sigmoid((score(n) - score_(k)) / T)

    centred midway between the k-th and (k+1)-th ranked scores, so the top k sit
    above 0.5 and the effective mass ``sum(u)`` stays near k on every prompt. As ``T`` approaches 0
    this becomes the hard top-k indicator and the whole aggregation reproduces the
    ``topk`` path exactly, which is the regression test for this code.

    The scale of ``T`` is the score scale, so it is taken relative to the spread
    of the candidate scores: ``T_eff = temperature * (score_(k) - median)``, with
    a fallback to the absolute temperature when that spread is degenerate. Rows
    that were never scored (not candidates for this label) get weight 0.

    A sigmoid indicator is used rather than a softmax over the pool deliberately.
    A softmax spreads a unit of mass over thousands of rows, which drives every
    weight to nearly zero, collapses the internal inbound mass, saturates
    frac_external at 1 and reduces the aggregation to a plain mean -- the same
    degeneracy already observed when the RMSNorm freeze pushed frac_external to
    0.995.
    """
    u = torch.zeros(n_rows, device=device, dtype=dtype)
    if not scored_rows:
        return u
    idx = torch.tensor([r for r, _ in scored_rows], device=device, dtype=torch.long)
    s = torch.tensor([float(v) for _, v in scored_rows], device=device, dtype=dtype)
    k = max(1, min(int(nodes_per_label), s.numel()))
    ordered = torch.sort(s, descending=True).values
    # Centre the sigmoid *between* the k-th and (k+1)-th scores, not on the k-th.
    # Sitting on it would leave that candidate at sigmoid(0) = 0.5 for every
    # temperature, so the effective mass would converge to k - 0.5 and the low-T
    # limit would not reproduce the hard top-k.
    if k < ordered.numel():
        centre = 0.5 * (ordered[k - 1] + ordered[k])
    else:
        gap = (ordered[0] - ordered[-1]).abs() / max(ordered.numel() - 1, 1)
        centre = ordered[k - 1] - gap.clamp(min=1e-6)
    spread = (centre - s.median()).abs()
    t_eff = float(temperature) * float(spread)
    if not (t_eff > 0):
        t_eff = float(temperature) if temperature > 0 else 1e-8
    u[idx] = torch.sigmoid((s - centre) / t_eff)
    return u


def _aggregate_soft(
    adjacency: torch.Tensor,
    weights: list[torch.Tensor],
    *,
    aggregation: str,
    n_neurons: int,
    n_tokens: int,
    constant_node_weighting: bool,
    epsilon: float,
) -> torch.Tensor:
    """``aggregate_supernode_adjacency`` with membership as weights in [0, 1].

    Every place the hard path writes ``j in supernodes[t]`` this writes a
    ``u_t(j)``-weighted sum, in all three roles membership plays: the source sum,
    the outer average over target rows, and the internal inbound mass inside
    frac_external. With ``u`` a 0/1 indicator each expression is identical to the
    hard path's, term for term.
    """
    if aggregation == "normalised":
        E = normalize_matrix(adjacency)                 # already non-negative
        En = E[:n_neurons, :n_neurons]
        total = E[:n_neurons].sum(dim=1)                # over every source column
        source = [En @ w for w in weights]
        rows = []
        for t, u in enumerate(weights):
            if constant_node_weighting:
                frac_external = torch.ones_like(total)
            else:
                internal = En @ u
                frac_external = (total - internal) / total.clamp(min=epsilon)
            g = u * frac_external
            denom = g.sum().clamp(min=epsilon)
            entries = [(g * src).sum() / denom for src in source]
            for p in range(n_tokens):
                entries.append((g * E[:n_neurons, n_neurons + p]).sum() / denom)
            rows.append(torch.stack(entries))
        return torch.stack(rows)

    math_dtype = torch.float32 if adjacency.dtype in (torch.float16, torch.bfloat16) else adjacency.dtype
    A = adjacency.to(dtype=math_dtype)
    An = A[:n_neurons, :n_neurons]
    source = [An @ w.to(dtype=math_dtype) for w in weights]
    rows = []
    for u in weights:
        u = u.to(dtype=math_dtype)
        denom = u.sum().clamp(min=epsilon)
        entries = [(u * src).sum() / denom for src in source]
        for p in range(n_tokens):
            entries.append((u * A[:n_neurons, n_neurons + p]).sum() / denom)
        rows.append(torch.stack(entries))
    W = torch.stack(rows)
    return W / W.abs().sum().clamp(min=epsilon)



def aggregate_supernode_adjacency(
    graph: Graph,
    supernodes: list[list[int]],
    *,
    aggregation: str = "normalised",
    constant_node_weighting: bool = False,
    token_source_columns: bool = False,
    membership_weights: list[torch.Tensor] | None = None,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    """The supernode adjacency for fixed membership -- the one arithmetic both the
    teacher (build_super_graph) and the differentiable student path use.

    ``normalised`` (the original construction): each target's |inbound| edges are
    first divided by their sum over *all* sources (normalize_matrix), then entry
    (t, s) is the frac_external-weighted mean over t's members of the sum over
    s's members. Every entry is a share of a unit of inbound mass, so its level
    depends on how many sources compete for that unit -- i.e. on the pre-selected
    pool size, which differs 35x between an 8B at 10% and a 1B at 1% -- and only
    the *shape* of a row survives a cross-model comparison. Relative row strength
    and edge sign are discarded.

    ``raw-signed``: entry (t, s) is the mean over t's members of the sum over s's
    members of the raw signed direct-effect edges. Raw edges depend only on the
    two nodes, not on the pool, so the entries are pool-independent; sign and the
    relative strength of rows and edges are kept. The whole matrix is then divided
    once by its total |mass|, which removes the two models' overall activation /
    gradient scales and nothing else. frac_external is not used on this path.

    With ``token_source_columns`` the token-embedding nodes are appended as extra
    source columns on either path (they exist in both models regardless of
    pre-selection): entry (t, token p) is the same aggregate over t's members of
    that member's edge from token p -- the frac_external-weighted mean of the
    normalised |edge| on ``normalised`` (normalize_matrix already counts the token
    columns in each target's unit of inbound mass), the plain mean of the raw
    signed edge on ``raw-signed``. Without it the matrix is K x K as before.

    Built out-of-place (torch.stack) so gradient flows through the student's
    adjacency into the model.
    """
    if aggregation not in SUPERGRAPH_AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {SUPERGRAPH_AGGREGATIONS}, got {aggregation!r}")
    adjacency = graph.adjacency_matrix
    num_supernodes = len(supernodes)
    if num_supernodes == 0:
        return torch.zeros((0, 0), device=adjacency.device, dtype=adjacency.dtype)
    n_neurons = graph.n_neurons
    n_tokens = graph.n_tokens if token_source_columns else 0

    if membership_weights is not None:
        if len(membership_weights) != num_supernodes:
            raise ValueError(
                f"membership_weights has {len(membership_weights)} entries for "
                f"{num_supernodes} supernodes")
        weights = [w.to(device=adjacency.device) for w in membership_weights]
        return _aggregate_soft(
            adjacency, weights, aggregation=aggregation, n_neurons=n_neurons,
            n_tokens=n_tokens, constant_node_weighting=constant_node_weighting,
            epsilon=epsilon,
        )

    if aggregation == "normalised":
        adj_matrix_norm = normalize_matrix(adjacency)
        rows = []
        for t in range(num_supernodes):
            target_members = supernodes[t]
            # frac_external(n_t): fraction of the absolute-valued *input* weights to n_t
            # that originate outside the supernode -- a row sum over n_t's incoming edges.
            total_input = torch.abs(adj_matrix_norm[target_members]).sum(dim=1)
            internal_input = torch.abs(adj_matrix_norm[target_members][:, target_members]).sum(dim=1)
            if constant_node_weighting:
                frac_external = torch.ones_like(total_input)
            else:
                frac_external = (total_input - internal_input) / total_input.clamp(min=epsilon)
            weight_sum = frac_external.sum(dim=0).clamp(min=epsilon)
            entries = []
            for s in range(num_supernodes):
                sum_A = adj_matrix_norm[target_members][:, supernodes[s]].sum(dim=1)
                entries.append((frac_external * sum_A).sum(dim=0) / weight_sum)
            for p in range(n_tokens):
                entries.append((frac_external * adj_matrix_norm[target_members, n_neurons + p]).sum(dim=0) / weight_sum)
            rows.append(torch.stack(entries))
        return torch.stack(rows)

    math_dtype = torch.float32 if adjacency.dtype in (torch.float16, torch.bfloat16) else adjacency.dtype
    A = adjacency.to(dtype=math_dtype)
    rows = []
    for t in range(num_supernodes):
        A_t = A[supernodes[t]]  # [n_members_t, n_sources]
        entries = [A_t[:, supernodes[s]].sum(dim=1).mean() for s in range(num_supernodes)]
        for p in range(n_tokens):
            entries.append(A_t[:, n_neurons + p].mean())
        rows.append(torch.stack(entries))
    W = torch.stack(rows)
    return W / W.abs().sum().clamp(min=epsilon)


def build_super_graph(
    graph: Graph,
    supernodes: list[list[int]],
    supernode_labels: list[list[str]],
    node_labels: dict[int, list[str]] | None = None,
    constant_node_weighting: bool = False,
    supergraph_aggregation: str = "normalised",
    token_source_columns: bool = False,
    supernode_heatmap_output_dir: str | None = None,
    activation_write_result: ActivationWriteResult | None = None,
    awr_index_map: dict[int, int] | None = None,
    sum_member_scores: dict[str, dict[int, tuple[float, float, float]]] | None = None,
    filtered_label_results: dict | None = None,
    W_U: torch.Tensor | None = None,
    tokenizer=None,
) -> SuperGraph:
    """Aggregate the attribution adjacency matrix into a supergraph.

    Supernodes are lists of neuron row indices already remapped to the filtered
    graph (i.e. members index graph.neuron_locations directly).
    """
    logger = logging.getLogger(__name__)
    num_supernodes = len(supernodes)
    supernode_adj_matrix = aggregate_supernode_adjacency(
        graph, supernodes,
        aggregation=supergraph_aggregation,
        constant_node_weighting=constant_node_weighting,
        token_source_columns=token_source_columns,
    )

    supernode_heatmap_pdf_paths: list[str] | None = None
    if supernode_heatmap_output_dir is not None:
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
        # Pre-compute number-token unembed mapping once if W_U/tokenizer available.
        # Used by DLA supernodes (1-D heatmap) and sum supernodes (side panel).
        number_unembed_precomputed: tuple | None = None
        if W_U is not None and tokenizer is not None:
            number_values = list(range(0, 201))
            raw_token_ids: list[int | None] = []
            for value in number_values:
                encoded = tokenizer(str(value), add_special_tokens=False, return_tensors=None)
                input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
                if input_ids and isinstance(input_ids[0], list):
                    input_ids = input_ids[0]
                raw_token_ids.append(int(input_ids[0]) if len(input_ids) == 1 else None)
            valid_positions = [i for i, tid in enumerate(raw_token_ids) if tid is not None]
            valid_numbers = [number_values[i] for i in valid_positions]
            valid_vocab_ids = torch.tensor(
                [int(raw_token_ids[i]) for i in valid_positions], dtype=torch.long, device=W_U.device
            )
            W_U_numbers = W_U[:, valid_vocab_ids]  # [d_model, n_valid]
            number_unembed_precomputed = (valid_numbers, W_U_numbers)

        for supernode_idx, members in enumerate(supernodes):
            category = (
                supernode_labels[supernode_idx][0]
                if supernode_labels[supernode_idx]
                else "none"
            )
            is_sum_category = category in {"sum range", "sum units"}
            is_dla_category = category == "dla"
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

            if is_dla_category:
                # --- 1-D DLA heatmap: W_out @ W_U over number tokens 0–200 ---
                # Does not require activation_write_result (no arg1×arg2 grid needed).
                if number_unembed_precomputed is None or graph.neuron_write_vectors is None:
                    logger.warning(
                        "  Cannot plot DLA heatmap for supernode %d: "
                        "W_U or neuron_write_vectors not available.",
                        supernode_idx,
                    )
                    supernode_heatmap_pdf_paths.append("")
                    continue
                valid_numbers, W_U_numbers = number_unembed_precomputed
                member_number_unembed_dla: dict[int, tuple[list[int], torch.Tensor]] = {}
                for m in members:
                    sv = graph.neuron_write_vectors[m].to(
                        device=W_U_numbers.device, dtype=W_U_numbers.dtype
                    )
                    dla_vals = (sv @ W_U_numbers).detach().float().cpu()
                    member_number_unembed_dla[m] = (valid_numbers, dla_vals)
                saved_path = save_dla_heatmap_pdf(
                    members,
                    neuron_locs,
                    member_number_unembed_dla,
                    output_path=os.path.join(
                        supernode_heatmap_output_dir,
                        f"supernode_{supernode_idx}.pdf",
                    ),
                    title=f"supernode {supernode_idx}: {category} — DLA influence over 0–200",
                    member_labels=member_labels,
                    member_norm_props=member_norm_props,
                )
                logger.info("  Saved DLA heatmap PDF: %s", saved_path)
                supernode_heatmap_pdf_paths.append(saved_path)
                continue

            # --- 2-D arg1 × arg2 activation heatmap (ANOVA / arg-token supernodes) ---
            if activation_write_result is None:
                logger.warning(
                    "  Skipping heatmap for supernode %d (%s): "
                    "activation_write_result not provided (pass --dataset when using "
                    "--include-arg-nodes to enable arg1×arg2 heatmaps).",
                    supernode_idx,
                    category,
                )
                supernode_heatmap_pdf_paths.append("")
                continue

            if awr_index_map is not None:
                awr_members = [awr_index_map[m] for m in members]
            else:
                awr_members = members
            row_indices = torch.tensor(awr_members, dtype=torch.long)
            cluster_heatmaps = activation_write_result.activations[row_indices].detach().float()
            member_var_spec = None
            if filtered_label_results is not None:
                member_var_spec = {}
                for m in members:
                    nl = filtered_label_results.get(m)
                    if nl is not None:
                        member_var_spec[m] = (
                            float(nl.category_scores.get(category, 0.0)),
                            float(nl.category_specificity.get(category, 0.0)),
                        )
            member_dla_kl = None
            if is_sum_category and sum_member_scores is not None:
                cat_scores = sum_member_scores.get(category, {})
                member_dla_kl = {m: cat_scores[m][2] for m in members if m in cat_scores}
            member_number_unembed = None
            if is_sum_category and number_unembed_precomputed is not None and graph.neuron_write_vectors is not None:
                valid_numbers, W_U_numbers = number_unembed_precomputed
                member_number_unembed = {}
                for m in members:
                    sv = graph.neuron_write_vectors[m].to(device=W_U_numbers.device, dtype=W_U_numbers.dtype)
                    dla_vals = (sv @ W_U_numbers).detach().float().cpu()
                    member_number_unembed[m] = (valid_numbers, dla_vals)
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
                supernode_category=category,
                member_labels=member_labels,
                member_norm_props=member_norm_props,
                member_var_spec=member_var_spec,
                member_dla_kl=member_dla_kl,
                member_number_unembed=member_number_unembed,
            )
            logger.info("  Saved supernode heatmap PDF: %s", saved_path)
            supernode_heatmap_pdf_paths.append(saved_path)

    labels_short = [
        (supernode_labels[i][0] if supernode_labels and supernode_labels[i] else str(i))
        for i in range(num_supernodes)
    ]
    col_w = max((len(l) for l in labels_short), default=4) + 2
    header = " " * col_w + "".join(l.rjust(col_w) for l in labels_short)
    logger.info("  Supernode adjacency matrix:\n%s", header)
    mat_cpu = supernode_adj_matrix.detach().cpu()
    for i, row_label in enumerate(labels_short):
        row_str = row_label.ljust(col_w) + "".join(
            f"{mat_cpu[i, j].item():.3f}".rjust(col_w) for j in range(num_supernodes)
        )
        logger.info("    %s", row_str)

    logger.info("  Aggregated supergraph: %d supernodes", num_supernodes)
    return SuperGraph(
        supernode_adjacency_matrix=supernode_adj_matrix,
        supernodes=supernodes,
        node_labels=node_labels,
        supernode_labels=supernode_labels,
        supernode_heatmap_pdf_paths=supernode_heatmap_pdf_paths,
    )

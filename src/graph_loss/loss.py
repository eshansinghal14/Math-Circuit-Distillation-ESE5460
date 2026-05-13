import torch
import torch.nn.functional as F


def _compute_edge_loss(
    W_T: torch.Tensor,
    W_S: torch.Tensor,
    mapping: dict[int, set[int]],
    teacher_ids: list[int],
    student_ids: list[int],
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Edge-level structural loss via teacher-coarsening.

    Why coarsening rather than per-entry comparison:
    Teacher and student adjacency matrices have different shapes (n_T x n_T vs
    n_S x n_S) and, after L1 row-normalisation, different natural per-entry
    scales (each entry ~ 1/row_size).  Comparing entries directly forces the
    student to match values that its row-normalisation constraint forbids.

    Coarsening fixes this by viewing the student matrix through the teacher's
    clustering:
        W_S_coarse[T_tgt, T_src] =
            (sum_{s_tgt in map[T_tgt], s_src in map[T_src]} W_S[s_tgt, s_src])
            / |map[T_tgt]|
    Both W_T and W_S_coarse now live on the same (n_T x n_T) support; rows of
    each are distributions over teacher source supernodes; their values are
    directly comparable.  We then compute element-wise MSE.

    Notes:
    - Teacher rows with no mapped students contribute (W_T[t,:]**2).mean() with
      zero gradient — honest: the student lacks a corresponding cluster, so the
      loss surfaces that gap in value but cannot push the student to grow one.
    - A student in multiple teacher buckets (rare due to the mapping's gap
      threshold) is double-counted softly; this is treated as noise.
    - Builds W_S_coarse as a single tensor (not appended scalars) so the autograd
      graph is preserved through indexing-and-sum on W_S.
    """
    n_T = len(teacher_ids)
    s_id2idx = {cid: i for i, cid in enumerate(student_ids)}
    device = W_S.device
    dtype = W_S.dtype

    if n_T == 0 or not mapping:
        return torch.tensor(0.0, device=device, dtype=dtype)

    rows: list[torch.Tensor] = []
    row_indices: list[int] = []

    for ti_tgt, t_tgt in enumerate(teacher_ids):
        s_tgts = [s_id2idx[s] for s in mapping.get(t_tgt, set()) if s in s_id2idx]
        if not s_tgts:
            continue
        n_tgt = float(len(s_tgts))
        s_tgts_t = torch.tensor(s_tgts, device=device, dtype=torch.long)
        row_entries: list[torch.Tensor] = []
        for t_src in teacher_ids:
            s_srcs = [s_id2idx[s] for s in mapping.get(t_src, set()) if s in s_id2idx]
            if not s_srcs:
                row_entries.append(torch.zeros((), device=device, dtype=dtype))
                continue
            s_srcs_t = torch.tensor(s_srcs, device=device, dtype=torch.long)
            block_sum = W_S.index_select(0, s_tgts_t).index_select(1, s_srcs_t).sum()
            row_entries.append(block_sum / n_tgt)
        rows.append(torch.stack(row_entries))
        row_indices.append(ti_tgt)

    if not rows:
        return torch.tensor(0.0, device=device, dtype=dtype)

    W_S_coarse_partial = torch.stack(rows)
    teacher_rows = W_T[row_indices].to(device=device, dtype=dtype)
    diff = teacher_rows - W_S_coarse_partial
    return (diff ** 2).mean()


def _compute_node_loss(
    teacher_dla: dict[int, torch.Tensor],
    student_dla: dict[int, torch.Tensor],
    mapping: dict[int, set[int]],
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Node-level functional loss: cosine distance between aligned DLA vectors.

    Uses normalized L2 (= 2*(1 - cosine_similarity)) so the loss is scale-invariant
    and bounded in [0, 4], regardless of whether DLAs are in logit or probability space.

    For each teacher supernode t mapped to student supernodes {s1, s2, ...},
    penalize the angular distance to the best-matching student supernode.
    """
    device = next(iter(teacher_dla.values())).device if teacher_dla else torch.device("cpu")

    loss_terms: list[torch.Tensor] = []

    for t_id, s_ids in mapping.items():
        if not s_ids or t_id not in teacher_dla:
            continue
        # Normalize teacher vector once (detached — teacher is frozen)
        t_vec = F.normalize(teacher_dla[t_id].float().to(device), dim=0)

        # Best-matching student supernode (min normalized L2 = min cosine distance)
        min_dist_sq = None
        for s_id in s_ids:
            if s_id not in student_dla:
                continue
            s_vec = F.normalize(student_dla[s_id].float().to(device), dim=0)
            dist_sq = (t_vec - s_vec).pow(2).sum()  # in [0, 4]
            if min_dist_sq is None or dist_sq < min_dist_sq:
                min_dist_sq = dist_sq

        if min_dist_sq is not None:
            loss_terms.append(min_dist_sq)

    if loss_terms:
        loss = torch.stack(loss_terms).sum() / len(loss_terms)
    else:
        loss = torch.tensor(0.0, device=device, dtype=torch.float32)

    return loss


def compute_graph_loss(
    W_T: torch.Tensor,
    W_S: torch.Tensor,
    mapping: dict[int, set[int]],
    teacher_ids: list[int],
    student_ids: list[int],
    teacher_dla: dict[int, torch.Tensor] | None = None,
    student_dla: dict[int, torch.Tensor] | None = None,
    epsilon: float = 1e-4,
    edge_weight: float = 1.0,
    node_weight: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Unified graph loss: edge structure + node DLA matching."""
    edge_loss = _compute_edge_loss(W_T, W_S, mapping, teacher_ids, student_ids, epsilon)

    if teacher_dla is not None and student_dla is not None and node_weight > 0:
        node_loss = _compute_node_loss(teacher_dla, student_dla, mapping, epsilon)
    else:
        node_loss = torch.tensor(0.0, device=W_T.device, dtype=W_T.dtype)

    total = edge_weight * edge_loss + node_weight * node_loss

    return total, {
        "edge_loss": edge_loss.item(),
        "node_loss": node_loss.item(),
        "graph_loss_total": total.item(),
    }


# Backward-compatible alias so existing notebooks/scripts don't break.
def compute_L_graph(
    W_T: torch.Tensor,
    W_S: torch.Tensor,
    mapping: dict[int, set[int]],
    teacher_ids: list[int],
    student_ids: list[int],
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Legacy wrapper — returns edge loss only (no node loss)."""
    return _compute_edge_loss(W_T, W_S, mapping, teacher_ids, student_ids, epsilon)


def compute_logit_focus_loss(
    teacher_focus: torch.Tensor,
    student_focus: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Phase-3 loss: KL(teacher_focus || student_focus), both normalised to distributions.

    ``teacher_focus`` and ``student_focus`` are un-normalised [n_logits] vectors of
    non-negative values representing total causal influence on each of the top-K logit
    targets for a given prompt.

    * Teacher: ``|supernode_prob_deltas|.sum(0)`` — sum of ablation-delta magnitudes.
    * Student: ``Σ_neuron |DLA_neuron @ W_U[:, logit_ids]|`` — sum of DLA magnitudes.

    After L1 normalisation, ``F.kl_div`` computes KL(teacher || student), encouraging
    the student's neurons to concentrate their DLA signal on the same output tokens as
    the teacher's circuits.  No cross-model alignment is required.
    """
    t_norm = teacher_focus / teacher_focus.sum().clamp(min=eps)
    s_norm = student_focus / student_focus.sum().clamp(min=eps)
    # F.kl_div(log_input, target) = Σ target * (log_target - log_input) = KL(target || input)
    return F.kl_div(s_norm.clamp(min=eps).log(), t_norm.detach(), reduction="sum")
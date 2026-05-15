import torch
import torch.nn.functional as F


def _compute_edge_loss(
    W_T: torch.Tensor,
    W_S: torch.Tensor,
    mapping: dict[int, set[int]],
    teacher_ids: list[int],
    student_ids: list[int],
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Edge-level structural loss: row-wise Jensen-Shannon between coarsened adjacency.

    Conceptual model:  each row of the supernode adjacency matrix represents
    "where does this supernode route its causal influence to downstream
    supernodes".  After L1 normalisation of absolute values, each row is a
    probability distribution over downstream supernodes.  The loss is the
    mean over rows of JSD(teacher_row, student_row).

    Why KL on distributions instead of MSE on raw entries:
    1. Scale-invariant — the loss compares routing *structure*, not absolute
       magnitudes.  Teacher and student adjacency entries are at different
       scales (different model sizes, different attribution magnitudes); MSE
       on raw entries was O(1e-6), making the graph signal ~100x smaller than
       KL distillation (~1.0) even at lambda=10.
    2. Probabilistically interpretable — "the student should distribute its
       influence the way the teacher does", which is the intent of structural
       distillation.
    3. Naturally bounded — KL on n_T-dim distributions is in [0, log(n_T)],
       same order of magnitude as the token-level KL distillation loss.

    Coarsening step (unchanged): the student adjacency is aggregated into a
    teacher-sized matrix using the alignment mapping, so both matrices live
    on the same (n_T x n_T) support before KL is computed.

    Notes:
    - Uses |entries| then L1-normalises so the sign of attribution magnitudes
      doesn't matter (they represent strength of influence, not direction).
    - Teacher distribution is detached: it's the fixed target.
    - Rows with no matched student supernode are skipped (no gradient).
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

    t_abs = teacher_rows.float().abs()
    s_abs = W_S_coarse_partial.float().abs()
    t_dist = t_abs / t_abs.sum(dim=1, keepdim=True).clamp(min=epsilon)
    s_dist = s_abs / s_abs.sum(dim=1, keepdim=True).clamp(min=epsilon)

    # Jensen-Shannon Divergence per row (mean over rows).  JSD = 0.5*KL(t||m) +
    # 0.5*KL(s||m) where m = (t+s)/2.  Symmetric, bounded in [0, log 2], and
    # well-defined even when one distribution has zero mass at indices the
    # other has nonzero mass at.  Forward-only KL doesn't penalize student
    # mass at indices where teacher has 0 (e.g. spurious self-loops); JSD does.
    m_dist = 0.5 * (t_dist.detach() + s_dist)
    log_m = (m_dist + epsilon).log()
    log_t = (t_dist.detach() + epsilon).log()
    log_s = (s_dist + epsilon).log()
    kl_t_m = (t_dist.detach() * (log_t - log_m)).sum(dim=1)
    kl_s_m = (s_dist * (log_s - log_m)).sum(dim=1)
    jsd = 0.5 * (kl_t_m + kl_s_m)
    return jsd.mean().to(dtype)


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
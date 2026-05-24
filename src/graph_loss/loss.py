from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


def _compute_edge_loss(
    W_T: torch.Tensor,
    W_S: torch.Tensor,
    mapping: dict[int, set[int]],
    teacher_ids: list[int],
    student_ids: list[int],
    epsilon: float = 1e-8,
    similarity: Literal["jsd", "kld", "mse", "mse-norm", "mse-scale"] = "jsd",
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

    log_t = (t_dist.detach() + epsilon).log()
    log_s = (s_dist + epsilon).log()

    if similarity == "kld":
        # Forward KL(teacher || student): penalizes student for missing teacher
        # mass but produces no gradient where teacher has zero (spurious student
        # edges are not penalized).
        kld = (t_dist.detach() * (log_t - log_s)).sum(dim=1)
        return kld.mean().to(dtype)

    if similarity == "mse":
        # Raw Frobenius-style MSE on absolute coarsened adjacency values.
        # Captures both shape and magnitude differences; gradient is simple
        # 2*(student - teacher), no explosion risk.
        return F.mse_loss(s_abs, t_abs.detach()).to(dtype)

    if similarity == "mse-norm":
        # MSE on L1-row-normalised distributions (same normalisation used for
        # KLD/JSD).  Scale-invariant: only the relative routing structure is
        # penalised, not absolute magnitudes.
        return F.mse_loss(s_dist, t_dist.detach()).to(dtype)

    if similarity == "mse-scale":
        # Combined: shape penalty (mse-norm) + scale penalty on row sums so
        # the student also learns to match the teacher's magnitude per row.
        shape_loss = F.mse_loss(s_dist, t_dist.detach())
        t_row_sums = t_abs.sum(dim=1).detach()
        s_row_sums = s_abs.sum(dim=1)
        scale_loss = F.mse_loss(s_row_sums, t_row_sums)
        return (shape_loss + 0.1 * scale_loss).to(dtype)

    # JSD = 0.5*KL(t||m) + 0.5*KL(s||m) where m = (t+s)/2.  Symmetric,
    # bounded in [0, log 2].  Also penalizes spurious student edges (mass
    # where teacher has zero), unlike forward-only KL.
    m_dist = 0.5 * (t_dist.detach() + s_dist)
    log_m = (m_dist + epsilon).log()
    kl_t_m = (t_dist.detach() * (log_t - log_m)).sum(dim=1)
    kl_s_m = (s_dist * (log_s - log_m)).sum(dim=1)
    jsd = 0.5 * (kl_t_m + kl_s_m)
    return jsd.mean().to(dtype)


def compute_graph_loss(
    W_T: torch.Tensor,
    W_S: torch.Tensor,
    mapping: dict[int, set[int]],
    teacher_ids: list[int],
    student_ids: list[int],
    epsilon: float = 1e-8,
    similarity: Literal["jsd", "kld", "mse", "mse-norm", "mse-scale"] = "jsd",
) -> tuple[torch.Tensor, dict]:
    """Graph loss: edge-structure similarity between aligned supernode adjacency rows."""
    loss = _compute_edge_loss(W_T, W_S, mapping, teacher_ids, student_ids, epsilon, similarity)
    return loss, {"edge_loss": loss.item()}


# Backward-compatible alias so existing notebooks/scripts don't break.
def compute_L_graph(
    W_T: torch.Tensor,
    W_S: torch.Tensor,
    mapping: dict[int, set[int]],
    teacher_ids: list[int],
    student_ids: list[int],
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Legacy wrapper — returns edge loss only."""
    return _compute_edge_loss(W_T, W_S, mapping, teacher_ids, student_ids, epsilon)

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
    similarity: Literal["jsd", "kld", "mse", "mse-norm", "mse-scale", "rel-mse"] = "jsd",
) -> torch.Tensor:
    """Edge-level structural loss: row-wise similarity between aligned supernode adjacency.

    Matrices may carry extra source columns beyond the supernodes (token-embedding
    nodes, see aggregate_supernode_adjacency); those are aligned by position and
    must be equally many on both sides.

    Assumes a 1-to-1 mapping between teacher and student supernodes.  Each row of
    the adjacency matrix represents "where does this supernode route its causal
    influence"; after L1 normalisation of absolute values each row is a probability
    distribution and the loss is the mean row-wise similarity (JSD/KLD/MSE).

    Notes:
    - Uses |entries| then L1-normalises so attribution sign doesn't matter.
    - Teacher distribution is detached: it's the fixed target.
    - Rows with no matched student supernode are skipped (no gradient).
    """
    n_T = len(teacher_ids)
    s_id2idx = {cid: i for i, cid in enumerate(student_ids)}
    device = W_S.device
    dtype = W_S.dtype

    if n_T == 0 or not mapping:
        return torch.tensor(0.0, device=device, dtype=dtype)

    # 1-to-1: teacher index -> single aligned student index
    t_to_s: dict[int, int] = {}
    for ti, t in enumerate(teacher_ids):
        s_mapped = [s_id2idx[s] for s in mapping.get(t, set()) if s in s_id2idx]
        if s_mapped:
            t_to_s[ti] = s_mapped[0]

    if not t_to_s:
        return torch.tensor(0.0, device=device, dtype=dtype)

    valid_t = sorted(t_to_s.keys())
    t_idx = torch.tensor(valid_t, device=device, dtype=torch.long)
    s_idx = torch.tensor([t_to_s[ti] for ti in valid_t], device=device, dtype=torch.long)

    # Aligned student submatrix: rows = valid targets, cols = all n_T teacher slots
    # followed by any extra (token) source columns, which align by position.
    n_S = len(student_ids)
    extra_t, extra_s = W_T.shape[1] - n_T, W_S.shape[1] - n_S
    if extra_t != extra_s:
        raise ValueError(f"teacher has {extra_t} extra source columns but student has {extra_s}")
    W_S_aligned = torch.zeros(len(valid_t), n_T + extra_t, device=device, dtype=dtype)
    W_S_aligned[:, t_idx] = W_S[s_idx][:, s_idx]
    if extra_t:
        W_S_aligned[:, n_T:] = W_S[s_idx][:, n_S:]

    teacher_rows = W_T[t_idx].to(device=device, dtype=dtype)

    return edge_similarity(teacher_rows, W_S_aligned, similarity, epsilon)


def _weighted_row_similarity(
    teacher_rows: torch.Tensor,
    W_S_aligned: torch.Tensor,
    similarity: str,
    epsilon: float,
    row_weights: torch.Tensor,
) -> torch.Tensor:
    """Per-row distance, then a weighted sum instead of an equal-weight mean.

    Used by token-path with rows='all', where each row is one logit target and
    most of them are low-probability tokens whose attribution is mostly noise.
    Averaging rows equally would let those outvote the top logit -- the same
    complaint as an empty supernode row getting the same vote as the circuit row.
    The weights are the teacher's probability over its own logit set, applied
    *after* the distance so each row is still compared as a distribution over
    positions.

    Kept separate from the unweighted path so that stays bit-identical: rel-mse
    there divides by the whole matrix's energy rather than per row, and changing
    it would silently move every existing supernode result.
    """
    w = row_weights.to(device=W_S_aligned.device, dtype=torch.float32).reshape(-1)
    if w.numel() != teacher_rows.shape[0]:
        raise ValueError(
            f"row_weights has {w.numel()} entries for {teacher_rows.shape[0]} rows")
    w = w / w.sum().clamp(min=epsilon)
    t = teacher_rows.float().detach()
    s_ = W_S_aligned.float()

    if similarity == "rel-mse":
        per_row = (s_ - t).pow(2).sum(dim=1) / t.pow(2).sum(dim=1).clamp(min=epsilon)
        return (w * per_row).sum().to(W_S_aligned.dtype)

    t_abs, s_abs = t.abs(), s_.abs()
    t_dist = t_abs / t_abs.sum(dim=1, keepdim=True).clamp(min=epsilon)
    s_dist = s_abs / s_abs.sum(dim=1, keepdim=True).clamp(min=epsilon)
    log_t, log_s = (t_dist + epsilon).log(), (s_dist + epsilon).log()

    if similarity == "kld":
        per_row = (t_dist * (log_t - log_s)).sum(dim=1)
    elif similarity == "mse":
        per_row = (s_abs - t_abs).pow(2).mean(dim=1)
    elif similarity == "mse-norm":
        per_row = (s_dist - t_dist).pow(2).mean(dim=1)
    elif similarity == "mse-scale":
        shape = (s_dist - t_dist).pow(2).mean(dim=1)
        scale = (s_abs.sum(dim=1) - t_abs.sum(dim=1)).pow(2)
        per_row = shape + 0.1 * scale
    else:  # jsd
        m_dist = 0.5 * (t_dist + s_dist)
        log_m = (m_dist + epsilon).log()
        per_row = 0.5 * ((t_dist * (log_t - log_m)).sum(dim=1)
                         + (s_dist * (log_s - log_m)).sum(dim=1))
    return (w * per_row).sum().to(W_S_aligned.dtype)


def edge_similarity(
    teacher_rows: torch.Tensor,
    W_S_aligned: torch.Tensor,
    similarity: str = "jsd",
    epsilon: float = 1e-8,
    row_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Distance between two already-aligned matrices of identical shape.

    Split out of _compute_edge_loss so callers whose rows and columns already
    correspond can skip the supernode alignment entirely. The token-path target
    is such a caller: its columns are token positions, not supernodes, so the
    alignment's assumption that the first K columns are the supernode block does
    not hold and indexing them raises once there is more than one row.
    """
    device, dtype = W_S_aligned.device, W_S_aligned.dtype
    if row_weights is not None:
        return _weighted_row_similarity(
            teacher_rows, W_S_aligned, similarity, epsilon, row_weights)
    if similarity == "rel-mse":
        # Relative squared error on the signed, globally normalised matrices: the
        # fraction of the teacher's edge energy the student fails to reproduce.
        # Sign-aware, weights every edge by its size, no saturation, gradient
        # 2(W_S - W_T) / ||W_T||^2 everywhere. Meant for the raw-signed aggregation.
        diff = W_S_aligned.float() - teacher_rows.float().detach()
        return (diff.pow(2).sum() / teacher_rows.float().pow(2).sum().clamp(min=epsilon)).to(dtype)

    t_abs = teacher_rows.float().abs()
    s_abs = W_S_aligned.float().abs()
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
    similarity: Literal["jsd", "kld", "mse", "mse-norm", "mse-scale", "rel-mse"] = "jsd",
) -> tuple[torch.Tensor, dict]:
    """Graph loss: edge-structure similarity between aligned supernode adjacency rows."""
    loss = _compute_edge_loss(W_T, W_S, mapping, teacher_ids, student_ids, epsilon, similarity)
    return loss, {"edge_loss": loss.item()}

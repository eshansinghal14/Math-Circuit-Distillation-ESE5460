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
    """Edge-level structural loss: row-wise similarity between aligned supernode adjacency.

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
    W_S_aligned = torch.zeros(len(valid_t), n_T, device=device, dtype=dtype)
    W_S_aligned[:, t_idx] = W_S[s_idx][:, s_idx]

    teacher_rows = W_T[t_idx].to(device=device, dtype=dtype)

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
    similarity: Literal["jsd", "kld", "mse", "mse-norm", "mse-scale"] = "jsd",
) -> tuple[torch.Tensor, dict]:
    """Graph loss: edge-structure similarity between aligned supernode adjacency rows."""
    loss = _compute_edge_loss(W_T, W_S, mapping, teacher_ids, student_ids, epsilon, similarity)
    return loss, {"edge_loss": loss.item()}

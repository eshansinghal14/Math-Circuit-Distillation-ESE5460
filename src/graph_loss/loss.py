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
    """Edge-level structural loss: penalize student edges that under-cover teacher edges."""
    t_id2idx = {cid: i for i, cid in enumerate(teacher_ids)}
    s_id2idx = {cid: i for i, cid in enumerate(student_ids)}
    device = W_T.device

    loss = torch.tensor(0.0, device=device, dtype=W_T.dtype)
    total_weight = torch.tensor(0.0, device=device, dtype=W_T.dtype)

    for t_src in teacher_ids:
        for t_tgt in teacher_ids:
            w_teacher = W_T[t_id2idx[t_tgt], t_id2idx[t_src]]

            if w_teacher.abs() < epsilon:
                continue

            total_weight += w_teacher.abs()

            src_students = mapping.get(t_src, set())
            tgt_students = mapping.get(t_tgt, set())

            if src_students and tgt_students:
                coverage_list = [
                    W_S[s_id2idx[s_tgt], s_id2idx[s_src]]
                    for s_src in src_students for s_tgt in tgt_students
                    if s_src in s_id2idx and s_tgt in s_id2idx
                ]

                if coverage_list:
                    coverage_vals = torch.stack(coverage_list)
                    if w_teacher > 0:
                        coverage = coverage_vals.max()
                    else:
                        coverage = coverage_vals.min()
                else:
                    coverage = torch.tensor(0.0, device=device, dtype=W_T.dtype)
            else:
                coverage = torch.tensor(0.0, device=device, dtype=W_T.dtype)

            gap = w_teacher - coverage
            loss += F.relu(w_teacher.sign() * gap) ** 2

    if total_weight > epsilon:
        loss = loss / total_weight

    return loss


def _compute_node_loss(
    teacher_dla: dict[int, torch.Tensor],
    student_dla: dict[int, torch.Tensor],
    mapping: dict[int, set[int]],
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Node-level functional loss: L2 on DLA mismatch between aligned supernodes.

    For each teacher supernode t mapped to student supernodes {s1, s2, ...},
    penalize the distance between the teacher's DLA and the closest student DLA.
    """
    device = next(iter(teacher_dla.values())).device if teacher_dla else torch.device("cpu")
    dtype = next(iter(teacher_dla.values())).dtype if teacher_dla else torch.float32

    loss = torch.tensor(0.0, device=device, dtype=dtype)
    count = 0

    for t_id, s_ids in mapping.items():
        if not s_ids or t_id not in teacher_dla:
            continue
        t_vec = teacher_dla[t_id]

        # Best-matching student supernode (min L2 distance)
        min_dist_sq = None
        for s_id in s_ids:
            if s_id not in student_dla:
                continue
            s_vec = student_dla[s_id].to(device=device, dtype=dtype)
            dist_sq = (t_vec - s_vec).pow(2).sum()
            if min_dist_sq is None or dist_sq < min_dist_sq:
                min_dist_sq = dist_sq

        if min_dist_sq is not None:
            loss = loss + min_dist_sq
            count += 1

    if count > 0:
        loss = loss / count

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
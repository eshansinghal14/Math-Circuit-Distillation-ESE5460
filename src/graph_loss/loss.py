import torch
import torch.nn.functional as F


def compute_L_graph(
    W_T: torch.Tensor,
    W_S: torch.Tensor,
    mapping: dict[int, set[int]],
    teacher_ids: list[int],
    student_ids: list[int],
    epsilon: float = 1e-4,
) -> torch.Tensor:
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



def compute_L_total(
    L_task: torch.Tensor,
    L_graph: torch.Tensor,
    L_repr: torch.Tensor | None = None,
    beta_graph: float = 0.1,
    alpha_repr: float = 0.01,
    error_contamination: float = 0.0,
    contamination_threshold: float = 0.3,
) -> tuple[torch.Tensor, dict]:
    # L_total = L_task + \beta·L_graph + \alpha·L_repr
    if error_contamination > contamination_threshold:
        graph_scale = max(0.0, 1.0 - error_contamination)
    else:
        graph_scale = 1.0

    weighted_graph = beta_graph * graph_scale * L_graph
    weighted_repr = (
        alpha_repr * L_repr
        if L_repr is not None
        else torch.tensor(0.0, device=L_task.device)
    )

    total = L_task + weighted_graph + weighted_repr

    return total, {
        "L_task": L_task.item(),
        "L_graph_raw": L_graph.item(),
        "L_graph_weighted": weighted_graph.item(),
        "L_repr_weighted": weighted_repr.item(),
        "graph_scale": graph_scale,
        "error_contamination": error_contamination,
        "L_total": total.item(),
    }
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field


@dataclass
class AlignmentResult:
    mapping: dict[int, set[int]] = field(default_factory=dict)
    teacher_dla: dict[int, torch.Tensor] = field(default_factory=dict)
    student_dla: dict[int, torch.Tensor] = field(default_factory=dict)
    best_sim: dict[int, float] = field(default_factory=dict)


def compute_supernode_dla(
    supernode: dict,
    W_U: torch.Tensor,
) -> torch.Tensor:
    # DLA = (Σ_i a_i · W_out[i]) @ W_U  →  R^{vocab}
    d_model = W_U.shape[0]
    write_vec = torch.zeros(d_model, device=W_U.device, dtype=W_U.dtype)
    for act, w_out_row in zip(supernode["activations"], supernode["w_out_rows"]):
        write_vec += act * w_out_row.to(W_U.device)
    return write_vec @ W_U


def align_supernodes(
    teacher_supernodes: list[dict],
    student_supernodes: list[dict],
    W_U_teacher: torch.Tensor,
    W_U_student: torch.Tensor,
    similarity_threshold: float = 0.7,
    max_fan_out: int = 4,
) -> AlignmentResult:
    # one teacher can match multiple students (split concept)
    # and multiple teachers can share a student (merged concept).
    # unmatched teachers (empty set) are penalized by L_graph.
    assert W_U_teacher.shape[1] == W_U_student.shape[1], (
        f"Vocab mismatch: {W_U_teacher.shape[1]} vs {W_U_student.shape[1]}"
    )

    result = AlignmentResult()

    for sn in teacher_supernodes:
        result.teacher_dla[sn["cluster_id"]] = compute_supernode_dla(sn, W_U_teacher)
    for sn in student_supernodes:
        result.student_dla[sn["cluster_id"]] = compute_supernode_dla(sn, W_U_student)

    def _normalize(vecs):
        return {
            cid: F.normalize(v.unsqueeze(0), dim=1).squeeze(0)
            for cid, v in vecs.items()
        }

    t_norm = _normalize(result.teacher_dla)
    s_norm = _normalize(result.student_dla)

    s_ids = list(s_norm.keys())
    if not s_ids:
        return result

    s_matrix = torch.stack([s_norm[sid] for sid in s_ids])

    for tid, t_vec in t_norm.items():
        sims = s_matrix @ t_vec

        above = [(sims[i].item(), s_ids[i]) for i in range(len(s_ids))
                 if sims[i].item() >= similarity_threshold]
        above.sort(reverse=True)

        if len(above) >= 2:
            gap = above[0][0] - above[1][0]
            if gap < 0.05:
                result.mapping[tid] = {above[0][1]}
                result.best_sim[tid] = above[0][0]
                continue

        above = above[:max_fan_out]

        result.mapping[tid] = {sid for _, sid in above}
        result.best_sim[tid] = above[0][0] if above else 0.0

    return result
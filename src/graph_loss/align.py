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
    # DLA = (Σ_i a_i · W_out[i])    device = W_U.device
    device = W_U.device
    dtype = W_U.dtype
    d_model = W_U.shape[0]
    write_vec = torch.zeros(d_model, device=device, dtype=dtype)
    for act, w_out_row in zip(supernode["activations"], supernode["w_out_rows"]):
        if not isinstance(act, torch.Tensor):
            act = torch.tensor(act, device=device, dtype=dtype)
        else:
            act = act.to(device=device, dtype=dtype)
            
        w_row = w_out_row.to(device=device, dtype=dtype)
        write_vec += act * w_row
        
    return write_vec @ W_U


def _build_full_vocab_prob_deltas(
    supergraph,
    graph,
    n_vocab: int,
) -> torch.Tensor:
    """Expand per-supernode prob-deltas from logit-target indices to full vocab.

    Returns shape [n_supernodes, n_vocab] with zeros for non-target tokens.
    """
    n_supernodes = len(supergraph.supernodes)

    if supergraph.supernode_prob_deltas is None or supergraph.supernode_prob_deltas.numel() == 0:
        return torch.zeros(n_supernodes, n_vocab, dtype=torch.float32)

    logit_token_ids = graph.logit_token_ids.cpu()
    deltas = supergraph.supernode_prob_deltas.detach().float().cpu()

    # Expand n_vocab if token IDs exceed it (special tokens beyond stated vocab)
    max_id = int(logit_token_ids.max().item()) + 1 if logit_token_ids.numel() else 0
    actual_vocab = max(n_vocab, max_id)
    full = torch.zeros(n_supernodes, actual_vocab, dtype=torch.float32)

    n_rows = min(deltas.shape[0], n_supernodes)
    for i in range(n_rows):
        full[i, logit_token_ids] = deltas[i]

    return full


def align_supernodes_prob_delta(
    teacher_supergraph,
    student_supergraph,
    teacher_graph,
    student_graph,
    *,
    similarity_threshold: float = 0.3,
    max_fan_out: int = 4,
    n_vocab: int = 128000,
) -> AlignmentResult:
    """Align supernodes using ablation probability-delta cosine similarity.

    This replaces DLA-based alignment with a signal that lives in shared
    probability space, making cross-model comparison meaningful.
    """
    t_full = _build_full_vocab_prob_deltas(teacher_supergraph, teacher_graph, n_vocab)
    s_full = _build_full_vocab_prob_deltas(student_supergraph, student_graph, n_vocab)

    # Pad to same width if special tokens caused different expansion
    if t_full.shape[1] != s_full.shape[1]:
        max_w = max(t_full.shape[1], s_full.shape[1])
        if t_full.shape[1] < max_w:
            t_full = F.pad(t_full, (0, max_w - t_full.shape[1]))
        if s_full.shape[1] < max_w:
            s_full = F.pad(s_full, (0, max_w - s_full.shape[1]))

    # Normalize for cosine similarity
    t_norm = F.normalize(t_full, dim=1)
    s_norm = F.normalize(s_full, dim=1)

    # sim_matrix[i, j] = cosine similarity between teacher SN i and student SN j
    sim_matrix = t_norm @ s_norm.T

    result = AlignmentResult()

    # Store the prob-delta vectors as "DLA" for backward compat with loss.py
    for tid in range(t_full.shape[0]):
        result.teacher_dla[tid] = t_full[tid]
    for sid in range(s_full.shape[0]):
        result.student_dla[sid] = s_full[sid]

    n_teacher = sim_matrix.shape[0]
    n_student = sim_matrix.shape[1]

    for tid in range(n_teacher):
        sims = sim_matrix[tid]
        above = [
            (sims[sid].item(), sid)
            for sid in range(n_student)
            if sims[sid].item() >= similarity_threshold
        ]
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


# Legacy DLA-based alignment (kept for reference/comparison)
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
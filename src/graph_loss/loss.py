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
    """Edge-level structural loss: MSE between teacher edge and mean matched student edge.

    Uses symmetric MSE (``gap**2``) so the gradient is non-zero whether the student
    over-covers or under-covers the teacher.  The one-sided relu formulation silently
    killed the gradient whenever the student — having fewer, coarser supernodes —
    had systematically larger per-entry adjacency values after L1 row-normalisation.

    Accumulates per-edge loss terms in a list and stacks at the end so the
    returned scalar carries the autograd graph through W_S.  Initializing
    ``loss = torch.tensor(0.0)`` and using ``loss += ...`` silently drops
    gradient because the leaf zero scalar has ``requires_grad=False`` and
    in-place adds on it do not promote it to a grad-tracking tensor.
    """
    t_id2idx = {cid: i for i, cid in enumerate(teacher_ids)}
    s_id2idx = {cid: i for i, cid in enumerate(student_ids)}
    device = W_T.device

    loss_terms: list[torch.Tensor] = []
    total_weight = torch.tensor(0.0, device=device, dtype=W_T.dtype)

    for t_src in teacher_ids:
        for t_tgt in teacher_ids:
            w_teacher = W_T[t_id2idx[t_tgt], t_id2idx[t_src]]

            if w_teacher.abs() < epsilon:
                continue

            total_weight = total_weight + w_teacher.abs()

            src_students = mapping.get(t_src, set())
            tgt_students = mapping.get(t_tgt, set())

            if src_students and tgt_students:
                coverage_list = [
                    W_S[s_id2idx[s_tgt], s_id2idx[s_src]]
                    for s_src in src_students for s_tgt in tgt_students
                    if s_src in s_id2idx and s_tgt in s_id2idx
                ]

                if coverage_list:
                    # Mean over matched student edges so every mapped entry gets
                    # gradient, not just the argmax.
                    coverage = torch.stack(coverage_list).mean()
                else:
                    coverage = torch.tensor(0.0, device=device, dtype=W_T.dtype)
            else:
                coverage = torch.tensor(0.0, device=device, dtype=W_T.dtype)

            gap = w_teacher - coverage
            # Symmetric MSE: penalise both over- and under-coverage.
            loss_terms.append(gap ** 2)

    if loss_terms:
        loss = torch.stack(loss_terms).sum()
    else:
        loss = torch.tensor(0.0, device=device, dtype=W_T.dtype)

    if total_weight > epsilon:
        loss = loss / total_weight

    return loss


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
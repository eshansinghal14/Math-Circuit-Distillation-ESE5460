"""Look at teacher and student supergraphs where the student is right and where it is wrong.

Scores the student on a whole local arithmetic split under teacher forcing
(argmax and probability of the correct first answer token at the last prompt
position, plus the teacher's own probability), tags each prompt with structural
attributes (units carry, sum >= 100, single-digit operand), sorts prompts into
buckets -- right and confident, right but unsure, wrong with a carry, wrong
without one -- and for a few prompts per bucket builds the teacher and student
graphs once and derives every candidate supergraph construction from the same
two attributions:

  * ``normalised`` (the trainer's default aggregation, here always with the
    token-embedding source columns, i.e. what ``--graph-node-labels ... tokens``
    trains on): per-target |inbound| shares over the whole pre-selected pool,
    then the K x (K+T) block.
  * ``raw-signed``: signed edges summed over source members, mean over target
    members, token-embedding source columns appended, whole matrix divided by
    its |mass|.
  * ``gold-signed``: ``raw-signed`` after orienting every target neuron so that
    a positive edge means "the source pushes the target in the direction that
    raises the gold-answer logit": edge(t<-s) * sign(e(gold<-t)) * sign(a_t).
    A neuron's sign convention is arbitrary; this makes it canonical so member
    sums add coherently instead of cancelling.
  * ``gold-path``: the two-hop path attribution to the gold logit,
    edge(t<-s) * e(gold<-t) / a_t, aggregated like ``raw-signed``.
  * ``composition``: per target block, the distribution of inbound |mass| over
    three source groups -- arg-type blocks, sum/DLA-type blocks, token
    embeddings. How deep the block's inputs are, with no sign and no per-edge
    detail.

Every construction is scored with the same four distances: relative squared
error, cosine, row JSD with equal row weights (the trainer's) and row JSD with
rows weighted by the teacher's row |mass| (so near-empty rows stop out-voting
the rows that hold the circuit).

Controls, computed from the collected matrices and written to ``summary.json``
under ``controls`` (and printed):

  * same-prompt distance (student_i vs teacher_i) against the *shuffled-prompt*
    distance (student_i vs teacher_j, j != i). A construction that carries
    prompt-level information has the first well below the second; one whose
    target is a constant has them equal.
  * teacher-vs-teacher and student-vs-student across prompts: how much of the
    target is prompt-specific at all.
  * right vs wrong bucket means: whether the distance sees competence.
  * membership noise floor: the teacher against itself with the last
    ``--noise-drop-members`` members of every supernode dropped (same graph,
    same pool), per construction. A right/wrong gap must clear this.
  * optional pool noise floor (``--noise-pool-prop``): a second teacher graph
    at a different pre-selection fraction, aligned by label.

The gold token is forced into the attribution targets when the teacher's 95%
salient set does not already contain it (recorded per prompt as
``gold_in_salient``); nothing else about the graphs changes.

``--kd-checkpoint`` adds a third model -- a standard-KD-trained student -- graphed
on the same prompts with the same targets. Per prompt the summary then carries
its score (``kd_correct``, ``kd_p_gold``), its distance to the teacher
(``losses_kd``) and to the untrained student (``losses_kd_vs_student``); the
controls add ``kd_same_prompt``, ``kd_shuffled_prompt``, ``kd_minus_student``
(how much closer to the teacher KD moved the graph; negative = closer) and
``kd_vs_student``. That is the sensitivity test: a construction worth training
on moves under KD by more than the membership noise floor and towards the
teacher. Buckets stay defined by the untrained student's scores.

One heatmap figure and one node-and-edge figure per prompt. Pass
``--student-checkpoint`` to run the identical analysis on a trained student.

Usage (from the repository root, GPU):
    PYTHONPATH=src python -m experiments.inspect_graphs --n-per-bucket 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from types import SimpleNamespace
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DIR_ROOT, load_data, load_model, tokenize_prompt_answer  # noqa: E402
from training.utils import load_student, student_autocast  # noqa: E402
from graph_loss.hf_adapter import HFLlamaGraphAdapter  # noqa: E402
from graph_loss.graph import aggregate_supernode_adjacency  # noqa: E402
from graph_loss.utils import normalize_node_labels  # noqa: E402
from graph_loss.training import GraphAuxConfig  # noqa: E402
from graph_loss.create_graph import create_graph  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANOVA_LABELS = ["arg1 range", "arg1 units", "arg2 range", "arg2 units", "sum range", "sum units"]
BUCKETS = ["right_confident", "right_unsure", "wrong_carry", "wrong_nocarry"]
RIGHT_BUCKETS = {"right_confident", "right_unsure"}

# construction -> (signed?, has token columns?)
CONSTRUCTIONS: dict[str, tuple[bool, bool]] = {
    "normalised": (False, True),
    "raw-signed": (True, True),
    "gold-signed": (True, True),
    "gold-path": (True, True),
    "composition": (False, False),
}
COMPOSITION_GROUPS = ["arg blocks", "sum blocks", "tokens"]
DISTANCES = ["rel_mse", "cos", "jsd", "jsd_mass"]
EPS = 1e-8


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def parse_prompt(prompt: str) -> list[int] | None:
    """Operands of an N-ary addition prompt ('12+34=' or '12+34+56='), else None."""
    m = re.fullmatch(r"\s*(\d+(?:\s*\+\s*\d+)+)\s*=\s*", prompt)
    return [int(x) for x in re.findall(r"\d+", m.group(1))] if m else None


def attributes(prompt: str) -> dict[str, bool]:
    args = parse_prompt(prompt)
    if args is None:
        return {}
    return {
        "carry_units": sum(a % 10 for a in args) >= 10,
        "sum_ge_100": sum(args) >= 100,
        "single_digit": any(a < 10 for a in args),
    }


@torch.no_grad()
def score_prompts(model, tokenizer, prompts: list[str], answers: list, batch_size: int, autocast) -> list[dict[str, Any]]:
    """Teacher-forced first-answer-token argmax and probability per prompt."""
    out = []
    for i in range(0, len(prompts), batch_size):
        chunk = list(zip(prompts[i:i + batch_size], answers[i:i + batch_size]))
        seqs, p_lens, golds = [], [], []
        for p, a in chunk:
            p_ids, a_ids = tokenize_prompt_answer(tokenizer, p, str(a))
            seqs.append(torch.cat([p_ids, a_ids]))
            p_lens.append(int(p_ids.numel()))
            golds.append(int(a_ids[0]))
        L = max(s.numel() for s in seqs)
        ids = torch.full((len(seqs), L), tokenizer.eos_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, s in enumerate(seqs):
            ids[r, : s.numel()] = s
            mask[r, : s.numel()] = 1
        ids, mask = ids.to(DEVICE), mask.to(DEVICE)
        with autocast():
            logits = model(ids, attention_mask=mask).logits.float()
        for r, (p, a) in enumerate(chunk):
            z = logits[r, p_lens[r] - 1]
            probs = torch.softmax(z, dim=-1)
            top = int(z.argmax())
            out.append({
                "prompt": p, "gold": a, "gold_token": golds[r],
                "argmax_token": top, "argmax_str": tokenizer.decode([top]),
                "correct": top == golds[r], "p_gold": float(probs[golds[r]]), "p_top": float(probs[top]),
            })
    return out


def bucket_of(rec: dict[str, Any], tags: dict[str, bool], confident: float, unsure: float) -> str | None:
    if rec["correct"]:
        if rec["p_gold"] >= confident:
            return "right_confident"
        if rec["p_gold"] < unsure:
            return "right_unsure"
        return None
    return "wrong_carry" if tags.get("carry_units") else "wrong_nocarry"


def salient_targets_with_gold(logits: torch.Tensor, gold_token: int, top_k_logits: float,
                              temperature: float) -> tuple[torch.Tensor, bool]:
    """The teacher's salient logit set (AttributionTargets._from_salient: fewest top
    logits reaching ``top_k_logits`` cumulative mass, capped at 10) with the gold
    token appended when it is missing. Returns (token ids, gold_was_in_salient)."""
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    k = int((torch.cumsum(sorted_probs, dim=-1) < top_k_logits).sum().item()) + 1
    k = min(k, 10, probs.numel())
    top = sorted_indices[:k].cpu()
    in_salient = bool((top == gold_token).any())
    if not in_salient:
        top = torch.cat([top, torch.tensor([gold_token], dtype=top.dtype)])
    return top, in_salient


# ---------------------------------------------------------------------------
# Graphs -> constructions
# ---------------------------------------------------------------------------

def label_of(supergraph, i: int) -> str:
    labels = supergraph.supernode_labels or []
    return labels[i][0] if i < len(labels) and labels[i] else f"sn{i}"


def align(teacher_sg, student_sg):
    """Teacher-ordered shared labels -> (labels, teacher_ids, student_ids)."""
    s_index = {label_of(student_sg, j): j for j in range(len(student_sg.supernodes))}
    labels, t_ids, s_ids = [], [], []
    for i in range(len(teacher_sg.supernodes)):
        lbl = label_of(teacher_sg, i)
        if lbl in s_index:
            labels.append(lbl)
            t_ids.append(i)
            s_ids.append(s_index[lbl])
    return labels, t_ids, s_ids


def submatrix(W: torch.Tensor, ids: list[int], n_super: int) -> torch.Tensor:
    """Rows and supernode columns in ``ids`` order, extra (token) columns kept after."""
    W = W.detach().float().cpu()
    idx = torch.tensor(ids, dtype=torch.long)
    block = W[idx][:, idx]
    extra = W[idx][:, n_super:]
    return torch.cat([block, extra], dim=1) if extra.shape[1] else block


def gold_logit_index(graph, gold_token: int) -> int | None:
    for i, t in enumerate(graph.logit_targets):
        if int(t.vocab_idx) == int(gold_token):
            return i
    return None


def gold_oriented_graph(graph, gold_token: int, mode: str):
    """A view of ``graph`` whose neuron rows are re-signed / re-scaled by the
    target neuron's effect on the gold logit, for aggregate_supernode_adjacency.

    e(gold<-t) is the logit row of the adjacency (direct effect of t's write on the
    gold logit), a_t the neuron's activation, so g_t = e(gold<-t) / a_t is the
    gold-logit change per unit of t's activation.

    ``gold-signed``: row t *= sign(g_t)   (magnitudes untouched, sign canonical)
    ``gold-path``:   row t *= g_t         (two-hop path attribution s -> t -> gold)
    Rows of neurons with no gold effect become zero on both.
    """
    gi = gold_logit_index(graph, gold_token)
    if gi is None:
        raise ValueError(f"gold token {gold_token} is not among the graph's logit targets")
    A = graph.adjacency_matrix.detach().float()
    n = graph.n_neurons
    logit_row = n + graph.n_tokens + gi
    e_gold = A[logit_row, :n]                                  # [n_neurons]
    a = graph.neuron_activations.detach().float().reshape(-1)[:n]
    g = e_gold / a.abs().clamp(min=1e-6) * torch.sign(a)
    factor = torch.sign(g) if mode == "gold-signed" else g
    oriented = A.clone()
    oriented[:n] = oriented[:n] * factor.unsqueeze(1)
    return SimpleNamespace(adjacency_matrix=oriented, n_neurons=n, n_tokens=graph.n_tokens)


def is_arg_label(label: str) -> bool:
    return label.lower().startswith("arg")


def composition_matrix(W_raw: torch.Tensor, labels: list[str]) -> torch.Tensor:
    """K x 3 row distributions of inbound |mass| over {arg blocks, sum blocks, tokens}
    from an aligned raw-signed K x (K+T) matrix."""
    K = len(labels)
    a = W_raw.abs()
    arg_cols = [j for j, l in enumerate(labels) if is_arg_label(l)]
    sum_cols = [j for j in range(K) if j not in arg_cols]
    groups = torch.stack([
        a[:, arg_cols].sum(dim=1) if arg_cols else torch.zeros(K),
        a[:, sum_cols].sum(dim=1) if sum_cols else torch.zeros(K),
        a[:, K:].sum(dim=1) if a.shape[1] > K else torch.zeros(K),
    ], dim=1)
    return groups / groups.sum(dim=1, keepdim=True).clamp(min=EPS)


def build_constructions(graph, supernodes: list[list[int]], ids: list[int], labels: list[str],
                        gold_token: int) -> dict[str, torch.Tensor]:
    """Every construction, aligned to ``labels`` (rows and block columns in ``ids`` order)."""
    n_super = len(supernodes)
    out: dict[str, torch.Tensor] = {}
    out["normalised"] = submatrix(aggregate_supernode_adjacency(graph, supernodes, aggregation="normalised",
                                                                token_source_columns=True), ids, n_super)
    out["raw-signed"] = submatrix(aggregate_supernode_adjacency(graph, supernodes, aggregation="raw-signed",
                                                                token_source_columns=True), ids, n_super)
    for mode in ("gold-signed", "gold-path"):
        view = gold_oriented_graph(graph, gold_token, mode)
        out[mode] = submatrix(aggregate_supernode_adjacency(view, supernodes, aggregation="raw-signed",
                                                            token_source_columns=True), ids, n_super)
    out["composition"] = composition_matrix(out["raw-signed"], labels)
    return out


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

def _row_dists(M: torch.Tensor) -> torch.Tensor:
    a = M.abs()
    return a / a.sum(dim=1, keepdim=True).clamp(min=EPS)


def _row_jsd(T: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    t, s = _row_dists(T), _row_dists(S)
    m = 0.5 * (t + s)
    kl = lambda p, q: (p * ((p + EPS).log() - (q + EPS).log())).sum(dim=1)  # noqa: E731
    return 0.5 * (kl(t, m) + kl(s, m))


def distances(T: torch.Tensor, S: torch.Tensor) -> dict[str, float]:
    """``rel_mse``: ||S-T||^2 / ||T||^2 (signed; 0 identical, 1 empty student, 4 sign flip).
    ``cos``: cosine of the flattened matrices. ``jsd``: mean row JSD on |.| row
    distributions (the trainer's loss). ``jsd_mass``: the same rows weighted by the
    teacher's share of total |mass|."""
    T, S = T.float(), S.float()
    diff2 = (S - T).pow(2).sum()
    rows = _row_jsd(T, S)
    w = T.abs().sum(dim=1)
    w = w / w.sum().clamp(min=EPS)
    return {
        "rel_mse": float(diff2 / T.pow(2).sum().clamp(min=EPS)),
        "cos": float((S * T).sum() / (S.norm() * T.norm()).clamp(min=EPS)),
        "jsd": float(rows.mean()),
        "jsd_mass": float((w * rows).sum()),
    }


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(xs) / len(xs) if xs else None


def controls(records: list[dict[str, Any]], mats: list[dict[str, dict[str, torch.Tensor]]]) -> dict[str, Any]:
    """Cross-prompt controls per construction and distance. Cross-prompt pairs are
    formed only between prompts whose matrices have the same shape and labels.
    The ``kd_*`` fields exist only when a KD-trained student was graphed."""
    n = len(records)
    out: dict[str, Any] = {}

    def compatible(i: int, j: int, cons: str) -> bool:
        return (i != j and records[i]["labels"] == records[j]["labels"]
                and mats[i][cons]["teacher"].shape == mats[j][cons]["teacher"].shape)

    for cons in CONSTRUCTIONS:
        same = [distances(mats[i][cons]["teacher"], mats[i][cons]["student"]) for i in range(n)]
        shuffled, tt, ss, kd_shuffled = [], [], [], []
        for i in range(n):
            for j in range(n):
                if not compatible(i, j, cons):
                    continue
                shuffled.append(distances(mats[j][cons]["teacher"], mats[i][cons]["student"]))
                tt.append(distances(mats[i][cons]["teacher"], mats[j][cons]["teacher"]))
                ss.append(distances(mats[i][cons]["student"], mats[j][cons]["student"]))
                if "kd" in mats[i][cons]:
                    kd_shuffled.append(distances(mats[j][cons]["teacher"], mats[i][cons]["kd"]))
        noise_m = [distances(m[cons]["teacher"], m[cons]["teacher_fewer_members"]) for m in mats
                   if "teacher_fewer_members" in m[cons]]
        noise_p = [distances(m[cons]["teacher"], m[cons]["teacher_other_pool"]) for m in mats
                   if "teacher_other_pool" in m[cons]]
        kd_idx = [i for i in range(n) if "kd" in mats[i][cons]]
        kd_same = [distances(mats[i][cons]["teacher"], mats[i][cons]["kd"]) for i in kd_idx]
        kd_vs_s = [distances(mats[i][cons]["student"], mats[i][cons]["kd"]) for i in kd_idx]
        same_on_kd = [same[i] for i in kd_idx]
        right = [same[i] for i in range(n) if records[i]["bucket"] in RIGHT_BUCKETS]
        wrong = [same[i] for i in range(n) if records[i]["bucket"] not in RIGHT_BUCKETS]
        out[cons] = {}
        for d in DISTANCES:
            pick = lambda lst: _mean([x[d] for x in lst])  # noqa: E731
            r, w = pick(right), pick(wrong)
            entry = {
                "same_prompt": pick(same), "shuffled_prompt": pick(shuffled),
                "teacher_vs_teacher": pick(tt), "student_vs_student": pick(ss),
                "right": r, "wrong": w,
                "wrong_minus_right": (w - r) if (r is not None and w is not None) else None,
                "noise_members": pick(noise_m), "noise_pool": pick(noise_p),
                "n_pairs": len(shuffled),
            }
            if kd_idx:
                k, s_on_k = pick(kd_same), pick(same_on_kd)
                entry.update({
                    "kd_same_prompt": k, "kd_shuffled_prompt": pick(kd_shuffled),
                    "kd_minus_student": (k - s_on_k) if (k is not None and s_on_k is not None) else None,
                    "kd_vs_student": pick(kd_vs_s), "n_kd": len(kd_idx),
                })
            out[cons][d] = entry
    return out


def print_controls(ctrl: dict[str, Any]) -> None:
    cols = ["same_prompt", "shuffled_prompt", "teacher_vs_teacher", "student_vs_student", "right", "wrong",
            "wrong_minus_right", "noise_members", "noise_pool"]
    has_kd = any("kd_same_prompt" in v for per_d in ctrl.values() for v in per_d.values())
    if has_kd:
        cols += ["kd_same_prompt", "kd_shuffled_prompt", "kd_minus_student", "kd_vs_student"]
    fmt = lambda v: "   --  " if v is None else f"{v:7.3f}"  # noqa: E731
    print("\ncontrols (means over prompts / prompt pairs):")
    print(f"  {'construction':13s} {'dist':9s} " + " ".join(f"{c[:10]:>10s}" for c in cols))
    for cons, per_d in ctrl.items():
        for d, v in per_d.items():
            print(f"  {cons:13s} {d:9s} " + " ".join(f"{fmt(v.get(c)):>10s}" for c in cols))
    print("  read: a construction carries prompt-level information only if same_prompt << shuffled_prompt;\n"
          "        it sees competence only if wrong_minus_right clears noise_members (and noise_pool)"
          + (";\n        KD moved the graph towards the teacher only if kd_minus_student is negative and\n"
             "        |kd_minus_student| and kd_vs_student clear noise_members." if has_kd else "."))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _col_labels(cons: str, labels: list[str], token_strs: list[str], M: torch.Tensor) -> list[str]:
    if cons == "composition":
        return COMPOSITION_GROUPS
    return labels + ([f"tok:{t}" for t in token_strs] if M.shape[1] > len(labels) else [])


def _models_in(mats: dict[str, dict[str, torch.Tensor]]) -> list[str]:
    """Model columns present: teacher, student and, when graphed, the KD student."""
    return [m for m in ("teacher", "student", "kd") if m in next(iter(mats.values()))]


def _title(prompt: str, rec: dict[str, Any], teacher_p: float) -> str:
    t = (f"{prompt!r}  gold {rec['gold']}  |  student argmax {rec['argmax_str']!r} "
         f"p(gold)={rec['p_gold']:.2f}  |  teacher p(gold)={teacher_p:.2f}")
    if "kd_p_gold" in rec:
        t += f"  |  KD student argmax {rec['kd_argmax_str']!r} p(gold)={rec['kd_p_gold']:.2f}"
    return t


def render(out_path: str, prompt: str, rec: dict[str, Any], teacher_p: float, labels: list[str],
           token_strs: list[str], mats: dict[str, dict[str, torch.Tensor]], dist_by: dict[str, dict[str, float]],
           dist_kd: dict[str, dict[str, float]] | None = None) -> None:
    """Heatmaps: one row per construction; columns = each model, then each
    non-teacher model minus the teacher."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(CONSTRUCTIONS)
    models = _models_in(mats)
    others = [m for m in models if m != "teacher"]
    n_cols = len(models) + len(others)
    fig, axes = plt.subplots(len(names), n_cols, figsize=(4.2 * n_cols, 3.9 * len(names)), squeeze=False)
    for r, cons in enumerate(names):
        signed, _ = CONSTRUCTIONS[cons]
        T = mats[cons]["teacher"]
        vmax = float(max([mats[cons][m].abs().max() for m in models] + [torch.tensor(1e-8)]))
        panels = [(m if m != "kd" else "KD student", mats[cons][m], False) for m in models]
        panels += [(f"{m if m != 'kd' else 'KD student'} - teacher", mats[cons][m] - T, True) for m in others]
        dmax = float(max([(mats[cons][m] - T).abs().max() for m in others] + [torch.tensor(1e-8)]))
        for c, (title, M, is_diff) in enumerate(panels):
            ax = axes[r][c]
            if not signed and not is_diff:
                im = ax.imshow(M.numpy(), cmap="magma", vmin=0, vmax=vmax, aspect="auto")
            else:
                lim = dmax if is_diff else vmax
                im = ax.imshow(M.numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
            cols = _col_labels(cons, labels, token_strs, M)
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels(cols, rotation=90, fontsize=7)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels if c == 0 else [], fontsize=7)
            ax.set_title(title, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        key = "rel_mse" if signed else "jsd_mass"
        d = dist_by[cons]
        label = f"{cons}\nstudent: {key} = {d[key]:.3f}, cos = {d['cos']:.2f}"
        if dist_kd is not None and cons in dist_kd:
            label += f"\nKD: {key} = {dist_kd[cons][key]:.3f}, cos = {dist_kd[cons]['cos']:.2f}"
        axes[r][0].set_ylabel(label, fontsize=8)
    fig.suptitle(_title(prompt, rec, teacher_p), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _node_positions(labels: list[str], token_strs: list[str]) -> tuple[dict, dict]:
    """Layered layout: token nodes at the bottom, arg-type supernodes in the middle,
    sum / dla supernodes at the top. Returns (supernode positions, token positions)."""
    low = [i for i, l in enumerate(labels) if is_arg_label(l)]
    high = [i for i in range(len(labels)) if i not in low]
    pos: dict[int, tuple[float, float]] = {}
    layers = ((0.5, low), (0.9, high)) if token_strs else ((0.2, low), (0.8, high))
    for y, members in layers:
        for k, i in enumerate(members):
            pos[i] = ((k + 1) / (len(members) + 1), y)
    tok_pos = {p: ((p + 1) / (len(token_strs) + 1), 0.1) for p in range(len(token_strs))}
    return pos, tok_pos


def draw_graph(ax, labels: list[str], token_strs: list[str], W: torch.Tensor, *, title: str,
               scale: float, signed: bool, threshold: float = 0.06, annotate_top: int = 6) -> None:
    """Node-and-edge view of one supergraph. Rows of ``W`` are targets, the first
    ``len(labels)`` columns are supernode sources and any further columns are
    token-embedding sources. Edge width is |w| / ``scale`` (pass the same scale for
    teacher and student so widths are comparable); blue is positive, red negative
    (grey when ``signed`` is False); edges under ``threshold`` x scale are omitted
    and the ``annotate_top`` largest are labelled with their value. Self-loops are
    not drawn (they are on the heatmap's diagonal)."""
    from matplotlib.patches import FancyArrowPatch

    K = len(labels)
    has_tokens = W.shape[1] > K
    pos, tok_pos = _node_positions(labels, token_strs if has_tokens else [])
    edges = []
    for t in range(K):
        for s in range(W.shape[1]):
            if s == t:
                continue
            w = float(W[t, s])
            if abs(w) < threshold * scale:
                continue
            src = pos[s] if s < K else tok_pos[s - K]
            edges.append((abs(w), w, src, pos[t]))
    edges.sort(key=lambda e: e[0])
    cut = edges[-annotate_top][0] if len(edges) >= annotate_top else (edges[0][0] if edges else 0.0)
    for mag, w, src, dst in edges:
        rel = min(1.0, mag / max(scale, 1e-12))
        colour = ("#2a78d6" if w >= 0 else "#e34948") if signed else "#52514e"
        ax.add_patch(FancyArrowPatch(src, dst, arrowstyle="-|>", mutation_scale=10,
                                     lw=0.4 + 5.0 * rel, alpha=min(1.0, 0.3 + 0.7 * rel), color=colour,
                                     connectionstyle="arc3,rad=0.12", shrinkA=13, shrinkB=13, zorder=1))
        if mag >= cut:
            # 35% of the way from source to target, off the midpoint where edges cross
            mx, my = 0.65 * src[0] + 0.35 * dst[0], 0.65 * src[1] + 0.35 * dst[1]
            ax.text(mx, my + 0.02, f"{w:+.3f}" if signed else f"{w:.3f}", fontsize=5.5, ha="center",
                    color=colour, zorder=4, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    for i, (x, y) in pos.items():
        ax.scatter([x], [y], s=800, color="#e9e8e5", edgecolors="#52514e", zorder=2)
        ax.text(x, y, labels[i], ha="center", va="center", fontsize=6.5, zorder=3)
    for p, (x, y) in tok_pos.items():
        ax.scatter([x], [y], s=420, color="#fff4e0", edgecolors="#eda100", zorder=2)
        ax.text(x, y, token_strs[p], ha="center", va="center", fontsize=6.5, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)


def render_graphs(out_path: str, prompt: str, rec: dict[str, Any], teacher_p: float, labels: list[str],
                  token_strs: list[str], mats: dict[str, dict[str, torch.Tensor]],
                  dist_by: dict[str, dict[str, float]], dist_kd: dict[str, dict[str, float]] | None = None) -> None:
    """Node-and-edge diagrams: rows = edge-level constructions, columns = models.
    Within a row all panels share the edge-width scale (the largest matrix
    maximum), so a thinner edge in one panel means a weaker edge."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [c for c in CONSTRUCTIONS if c != "composition"]
    models = _models_in(mats)
    fig, axes = plt.subplots(len(names), len(models), figsize=(5.5 * len(models), 5 * len(names)), squeeze=False)
    for r, cons in enumerate(names):
        signed, _ = CONSTRUCTIONS[cons]
        scale = float(max([mats[cons][m].abs().max() for m in models] + [torch.tensor(1e-8)]))
        key = "rel_mse" if signed else "jsd_mass"
        for c, m in enumerate(models):
            if m == "teacher":
                title = f"teacher  [{cons}]"
            elif m == "student":
                title = f"student  [{cons}]  ({key} = {dist_by[cons][key]:.3f})"
            else:
                title = f"KD student  [{cons}]" + (f"  ({key} = {dist_kd[cons][key]:.3f})" if dist_kd else "")
            draw_graph(axes[r][c], labels, token_strs, mats[cons][m], title=title, scale=scale, signed=signed)
    fig.suptitle(
        _title(prompt, rec, teacher_p) + "\nedge width = |weight| on a shared scale per row; "
        "blue positive, red negative; token nodes in orange",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--student", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--student-checkpoint", default=None, help="Path to a trained student (final_checkpoint) instead of --student.")
    ap.add_argument("--kd-checkpoint", default=None,
                    help="Path to a standard-KD-trained student (final_checkpoint). Graphed as a third model on "
                         "the same prompts, with its distances to the teacher and to the untrained student.")
    ap.add_argument("--teacher", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--dataset", default="22_add",
                    help="Local dataset under datasets/ (e.g. 22_add, 222_add). Prompts are N-ary additions; "
                         "the ANOVA grid is built from its train split, so the labels must exist for its "
                         "operand count ('sum range', 'sum units', 'tokens' and 'argN ...' do).")
    ap.add_argument("--split", choices=["test", "train"], default="test")
    ap.add_argument("--n-per-bucket", type=int, default=3)
    ap.add_argument("--confident", type=float, default=0.8)
    ap.add_argument("--unsure", type=float, default=0.5)
    ap.add_argument("--graph-node-labels", nargs="*", default=ANOVA_LABELS,
                    help="ANOVA supernode labels (e.g. 'sum units' 'arg1 range'), 'all' for every "
                         "category, or 'none' / nothing for the arg-token + DLA construction the "
                         "trainer uses when no labels are given. Default: the six ANOVA labels.")
    ap.add_argument("--nodes-per-label", type=int, default=10)
    ap.add_argument("--teacher-prop-neurons", type=float, default=0.003)
    ap.add_argument("--student-prop-neurons", type=float, default=0.01)
    ap.add_argument("--noise-drop-members", type=int, default=2,
                    help="Membership noise floor: re-aggregate the teacher with this many lowest-ranked "
                         "members dropped from every supernode (same graph). 0 disables.")
    ap.add_argument("--noise-pool-prop", type=float, default=None,
                    help="Pool noise floor: also build the teacher graph at this pre-selection fraction "
                         "and report its distance to the default-pool teacher. Off by default.")
    ap.add_argument("--teacher-graph-batch-size", type=int, default=512)
    ap.add_argument("--student-graph-batch-size", type=int, default=128)
    ap.add_argument("--mlp-cache-batch-size", type=int, default=32)
    ap.add_argument("--score-batch-size", type=int, default=2048,
                    help="Prompts per teacher-forced forward when scoring the whole split.")
    ap.add_argument("--out", default=os.path.join(DIR_ROOT, "results", "inspect_graphs"))
    args = ap.parse_args()

    # Same normalisation as the trainer ('arg 1 units' -> 'arg1 units'; 'tokens' is
    # accepted and dropped, since every construction here already carries the token
    # columns); 'none' or an empty list selects the arg-token + DLA construction.
    raw_labels, _ = normalize_node_labels(args.graph_node_labels)
    node_labels = None if not raw_labels or raw_labels == ["none"] else raw_labels
    args.graph_node_labels = node_labels
    print("supernode construction:", "arg-token + DLA" if node_labels is None else node_labels,
          f"| dataset {args.dataset} ({args.split} split scored, {args.dataset} train split for the ANOVA grid)")

    train_data, test_data = load_data(args.dataset)
    data = test_data if args.split == "test" else train_data
    items = list(data.items())  # the whole split is scored; buckets draw from all of it
    prompts = [p for p, _ in items]
    answers = [a for _, a in items]

    student_name = args.student_checkpoint or args.student
    student, tokenizer = load_student(student_name)
    kd = None
    if args.kd_checkpoint:
        kd, _ = load_student(args.kd_checkpoint)
    teacher, _ = load_model(args.teacher)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    if hasattr(teacher.config, "use_cache"):
        teacher.config.use_cache = False

    # ---- score --------------------------------------------------------------
    s_scores = score_prompts(student, tokenizer, prompts, answers, args.score_batch_size, student_autocast)
    t_scores = score_prompts(teacher, tokenizer, prompts, answers, args.score_batch_size, torch.no_grad)
    n_right = sum(r["correct"] for r in s_scores)
    print(f"scored {len(prompts)} {args.split} prompts: student argmax right on {n_right} "
          f"({n_right / len(prompts):.3f}); teacher right on {sum(r['correct'] for r in t_scores)}")
    k_scores = None
    if kd is not None:
        k_scores = score_prompts(kd, tokenizer, prompts, answers, args.score_batch_size, student_autocast)
        print(f"  KD student right on {sum(r['correct'] for r in k_scores)}")

    picked: dict[str, list[int]] = {b: [] for b in BUCKETS}
    for i, rec in enumerate(s_scores):
        tags = attributes(rec["prompt"])
        b = bucket_of(rec, tags, args.confident, args.unsure)
        if b is not None and len(picked[b]) < args.n_per_bucket:
            picked[b].append(i)
    for b in BUCKETS:
        print(f"  {b}: {[prompts[i] for i in picked[b]]}")

    # ---- graphs -------------------------------------------------------------
    from graph_loss.precompute_mlp_inputs import build_mlp_input_cache

    student_adapter = HFLlamaGraphAdapter(student, tokenizer, DEVICE)
    teacher_adapter = HFLlamaGraphAdapter(teacher, tokenizer, DEVICE)
    kd_adapter = HFLlamaGraphAdapter(kd, tokenizer, DEVICE) if kd is not None else None
    student_cache = teacher_cache = kd_cache = None
    if node_labels is not None:  # the ANOVA path needs the probe-grid MLP inputs, keyed per set of weights
        student_cache = build_mlp_input_cache(student_adapter, args.dataset, student_name, data_dict=train_data,
                                              batch_size=args.mlp_cache_batch_size)
        teacher_cache = build_mlp_input_cache(teacher_adapter, args.dataset, args.teacher, data_dict=train_data,
                                              batch_size=args.mlp_cache_batch_size)
        if kd_adapter is not None:
            kd_cache = build_mlp_input_cache(kd_adapter, args.dataset, args.kd_checkpoint, data_dict=train_data,
                                             batch_size=args.mlp_cache_batch_size)
    config = GraphAuxConfig(
        graph_dtype=torch.bfloat16,
        teacher_prop_neurons_per_layer=args.teacher_prop_neurons,
        student_prop_neurons_per_layer=args.student_prop_neurons,
        top_k_logits=0.95, temperature=1.0,
        teacher_graph_batch_size=args.teacher_graph_batch_size,
        student_graph_batch_size=args.student_graph_batch_size,
        student_nodes_per_label=args.nodes_per_label, teacher_nodes_per_label=args.nodes_per_label,
        graph_node_labels=node_labels,
        mlp_input_cache=student_cache, teacher_mlp_input_cache=teacher_cache,
        supergraph_aggregation="raw-signed", token_source_columns=True,
        dataset_name=args.dataset,
    )

    def teacher_graph(prompt: str, targets: torch.Tensor, prop: float):
        return create_graph(
            teacher_adapter, prompt, attribution_targets=targets,
            prop_neurons_per_layer=prop,
            top_k_logits=config.top_k_logits, temperature=config.temperature,
            batch_size=config.teacher_graph_batch_size, node_labels=config.graph_node_labels,
            mlp_input_cache=config.teacher_mlp_input_cache, nodes_per_label=config.teacher_nodes_per_label,
            no_grad_supergraph=True, build_create_graph=False, detach_result=True,
            supergraph_aggregation="raw-signed", token_source_columns=True,
        )

    def student_graph(adapter, cache, prompt: str, logit_token_ids: torch.Tensor, teacher_dla_logits: torch.Tensor):
        """The student-side graph the trainer builds: teacher's logit targets, teacher's
        logits for the DLA reference, the model's own MLP-input cache for ANOVA labels."""
        return create_graph(
            adapter, prompt,
            attribution_targets=logit_token_ids.cpu(),
            prop_neurons_per_layer=config.student_prop_neurons_per_layer,
            top_k_logits=config.top_k_logits, temperature=config.temperature,
            batch_size=config.student_graph_batch_size, dtype=config.graph_dtype,
            build_create_graph=False, detach_result=True, skip_logit_attribution=False,
            mlp_input_cache=cache, node_labels=config.graph_node_labels or [],
            nodes_per_label=config.student_nodes_per_label, dla_model_logits=teacher_dla_logits,
            no_grad_supergraph=True, supergraph_aggregation="raw-signed", token_source_columns=True,
        )

    def ids_for(supergraph, labels: list[str]) -> list[int] | None:
        """Supernode indices of ``supergraph`` in ``labels`` order, or None if one is missing."""
        index = {label_of(supergraph, j): j for j in range(len(supergraph.supernodes))}
        return [index[l] for l in labels] if all(l in index for l in labels) else None

    records: list[dict[str, Any]] = []
    all_mats: list[dict[str, dict[str, torch.Tensor]]] = []
    for b in BUCKETS:
        for i in picked[b]:
            prompt, answer = prompts[i], answers[i]
            gold_token = int(s_scores[i]["gold_token"])
            print(f"[{b}] building graphs for {prompt!r} ...")
            # Mirrors _compute_teacher_target, but keeps the teacher's Graph so every
            # construction can be aggregated from the same attribution, and forces the
            # gold token into the logit targets so the gold-oriented constructions exist.
            p_ids, a_ids = tokenize_prompt_answer(tokenizer, prompt, str(answer))
            full_ids = torch.cat([p_ids, a_ids]).to(DEVICE)
            with torch.no_grad():
                teacher_dla_logits = teacher(full_ids.unsqueeze(0)).logits[0, p_ids.numel() - 1].detach()
            targets, gold_in_salient = salient_targets_with_gold(teacher_dla_logits, gold_token,
                                                                 config.top_k_logits, config.temperature)
            with torch.enable_grad():
                t_res = teacher_graph(prompt, targets, config.teacher_prop_neurons_per_layer)
                logit_token_ids = t_res.graph.logit_token_ids.to(DEVICE)
                s_res = student_graph(student_adapter, student_cache, prompt, logit_token_ids, teacher_dla_logits)
                k_res = (student_graph(kd_adapter, kd_cache, prompt, logit_token_ids, teacher_dla_logits)
                         if kd_adapter is not None else None)
                t_pool = teacher_graph(prompt, targets, args.noise_pool_prop) if args.noise_pool_prop else None
            labels, t_ids, s_ids = align(t_res.supergraph, s_res.supergraph)
            if not labels:
                print("   no shared supernode labels; skipping")
                continue
            with torch.no_grad():
                T_by = build_constructions(t_res.graph, t_res.supergraph.supernodes, t_ids, labels, gold_token)
                S_by = build_constructions(s_res.graph, s_res.supergraph.supernodes, s_ids, labels, gold_token)
                mats = {cons: {"teacher": T_by[cons], "student": S_by[cons]} for cons in CONSTRUCTIONS}
                if k_res is not None:
                    k_ids = ids_for(k_res.supergraph, labels)
                    if k_ids is None:
                        print("   KD student lacks one of the shared supernode labels; no KD graph for this prompt")
                    else:
                        K_by = build_constructions(k_res.graph, k_res.supergraph.supernodes, k_ids, labels, gold_token)
                        for cons in CONSTRUCTIONS:
                            mats[cons]["kd"] = K_by[cons]
                drop = args.noise_drop_members
                if drop > 0 and all(len(m) > drop for m in t_res.supergraph.supernodes):
                    fewer = [m[:-drop] for m in t_res.supergraph.supernodes]
                    F_by = build_constructions(t_res.graph, fewer, t_ids, labels, gold_token)
                    for cons in CONSTRUCTIONS:
                        mats[cons]["teacher_fewer_members"] = F_by[cons]
                if t_pool is not None:
                    p_labels, p_t_ids, p_s_ids = align(t_res.supergraph, t_pool.supergraph)
                    if p_labels == labels:
                        P_by = build_constructions(t_pool.graph, t_pool.supergraph.supernodes, p_s_ids, labels, gold_token)
                        for cons in CONSTRUCTIONS:
                            mats[cons]["teacher_other_pool"] = P_by[cons]
                    else:
                        print(f"   pool-noise teacher shares only {p_labels}; skipping pool noise for this prompt")
            dist_by = {cons: distances(mats[cons]["teacher"], mats[cons]["student"]) for cons in CONSTRUCTIONS}
            has_kd = all("kd" in mats[cons] for cons in CONSTRUCTIONS)
            dist_kd = {cons: distances(mats[cons]["teacher"], mats[cons]["kd"]) for cons in CONSTRUCTIONS} if has_kd else None
            token_strs = [tokenizer.decode([int(t)]) for t in t_res.graph.input_tokens.tolist()]
            rec = dict(s_scores[i])
            rec.update(bucket=b, tags=attributes(prompt), teacher_p_gold=t_scores[i]["p_gold"],
                       gold_in_salient=gold_in_salient, labels=labels, token_strs=token_strs,
                       losses=dist_by,
                       matrices={cons: {k: v.tolist() for k, v in m.items()} for cons, m in mats.items()})
            if k_scores is not None:
                rec.update(kd_correct=k_scores[i]["correct"], kd_p_gold=k_scores[i]["p_gold"],
                           kd_argmax_str=k_scores[i]["argmax_str"])
            if has_kd:
                rec.update(losses_kd=dist_kd,
                           losses_kd_vs_student={cons: distances(mats[cons]["student"], mats[cons]["kd"])
                                                 for cons in CONSTRUCTIONS})
            records.append(rec)
            all_mats.append(mats)
            safe = re.sub(r"[^0-9a-zA-Z]+", "_", prompt).strip("_")
            render(os.path.join(args.out, b, f"{safe}_adj.png"), prompt, rec, t_scores[i]["p_gold"], labels,
                   token_strs, mats, dist_by, dist_kd)
            render_graphs(os.path.join(args.out, b, f"{safe}_graph.png"), prompt, rec, t_scores[i]["p_gold"],
                          labels, token_strs, mats, dist_by, dist_kd)
            for name, dd in (("student", dist_by), ("KD", dist_kd)):
                if dd is None:
                    continue
                print(f"   {name:8s}" + " | ".join(
                    f"{cons} {('rel_mse' if CONSTRUCTIONS[cons][0] else 'jsd_mass')}="
                    f"{dd[cons]['rel_mse' if CONSTRUCTIONS[cons][0] else 'jsd_mass']:.3f} cos={dd[cons]['cos']:.2f}"
                    for cons in CONSTRUCTIONS))
            del t_res, s_res, k_res, t_pool
            torch.cuda.empty_cache()

    # ---- summary ------------------------------------------------------------
    summary: dict[str, Any] = {"args": vars(args), "n_scanned": len(prompts), "student_right": n_right,
                               "constructions": list(CONSTRUCTIONS), "distances": DISTANCES,
                               "buckets": {}, "controls": {}, "prompts": records}
    def bucket_means(rs: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]] | None:
        rs = [r for r in rs if key in r]
        if not rs:
            return None
        return {cons: {d: sum(r[key][cons][d] for r in rs) / len(rs) for d in DISTANCES} for cons in CONSTRUCTIONS}

    for b in BUCKETS:
        rs = [r for r in records if r["bucket"] == b]
        if rs:
            entry = {
                "n": len(rs),
                "student_p_gold": sum(r["p_gold"] for r in rs) / len(rs),
                "losses": bucket_means(rs, "losses"),
            }
            if k_scores is not None:
                entry["kd_p_gold"] = sum(r["kd_p_gold"] for r in rs) / len(rs)
                entry["kd_right"] = sum(bool(r["kd_correct"]) for r in rs)
                entry["losses_kd"] = bucket_means(rs, "losses_kd")
                entry["losses_kd_vs_student"] = bucket_means(rs, "losses_kd_vs_student")
            summary["buckets"][b] = entry
    if records:
        summary["controls"] = controls(records, all_mats)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nper-bucket means:")
    for b, v in summary["buckets"].items():
        line = f"  {b:16s} n={v['n']}  student p(gold)={v['student_p_gold']:.2f}"
        if "kd_p_gold" in v:
            line += f"  KD p(gold)={v['kd_p_gold']:.2f} right {v['kd_right']}/{v['n']}"
        print(line)
        for cons in CONSTRUCTIONS:
            print(f"      {cons:13s} student  " + "  ".join(f"{d}={v['losses'][cons][d]:.3f}" for d in DISTANCES))
            if v.get("losses_kd"):
                print(f"      {'':13s} KD       " + "  ".join(f"{d}={v['losses_kd'][cons][d]:.3f}" for d in DISTANCES))
    if summary["controls"]:
        print_controls(summary["controls"])
    print("wrote", os.path.join(args.out, "summary.json"))


if __name__ == "__main__":
    main()

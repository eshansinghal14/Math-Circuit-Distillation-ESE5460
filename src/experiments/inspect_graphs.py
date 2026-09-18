"""Look at teacher and student supergraphs where the student is right and where it is wrong.

Scores the student on a slice of a local arithmetic split under teacher forcing
(argmax and probability of the correct first answer token at the last prompt
position, plus the teacher's own probability), tags each prompt with structural
attributes (units carry, sum >= 100, single-digit operand), sorts prompts into
buckets -- right and confident, right but unsure, wrong with a carry, wrong
without one -- and for a few prompts per bucket builds the teacher and student
supergraphs once and renders both constructions from the same graphs:

  * ``normalised`` (the original): per-target |inbound| shares, row-normalised,
    compared with the row-wise JSD the training loss uses;
  * ``raw-signed`` with token-embedding source columns: signed edges, whole
    matrix normalised once, compared with the relative squared error.

One figure per prompt (two rows, one per construction; teacher / student /
difference) and a ``summary.json`` with every prompt's scores, tags, bucket and
both losses, plus per-bucket means -- so the same run says whether either loss
separates prompts the student gets right from ones it gets wrong. Pass
``--student-checkpoint`` to run the identical analysis on a trained student.

Usage (from the repository root, GPU):
    PYTHONPATH=src python -m experiments.inspect_graphs --n-scan 400 --n-per-bucket 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DIR_ROOT, load_data, load_model, tokenize_prompt_answer  # noqa: E402
from training.utils import load_student, student_autocast  # noqa: E402
from graph_loss.hf_adapter import HFLlamaGraphAdapter  # noqa: E402
from graph_loss.graph import aggregate_supernode_adjacency  # noqa: E402
from graph_loss.loss import _compute_edge_loss  # noqa: E402
from graph_loss.training import GraphAuxConfig  # noqa: E402
from graph_loss.create_graph import create_graph  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANOVA_LABELS = ["arg1 range", "arg1 units", "arg2 range", "arg2 units", "sum range", "sum units"]
BUCKETS = ["right_confident", "right_unsure", "wrong_carry", "wrong_nocarry"]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def parse_prompt(prompt: str) -> tuple[int, int] | None:
    m = re.fullmatch(r"\s*(\d+)\s*\+\s*(\d+)\s*=\s*", prompt)
    return (int(m.group(1)), int(m.group(2))) if m else None


def attributes(prompt: str) -> dict[str, bool]:
    ab = parse_prompt(prompt)
    if ab is None:
        return {}
    a, b = ab
    return {
        "carry_units": (a % 10 + b % 10) >= 10,
        "sum_ge_100": a + b >= 100,
        "single_digit": a < 10 or b < 10,
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


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------

def label_of(supergraph, i: int) -> str:
    labels = supergraph.supernode_labels or []
    return labels[i][0] if i < len(labels) and labels[i] else f"sn{i}"


def align(teacher_sg, student_sg):
    """Teacher-ordered shared labels -> (labels, teacher_ids, student_ids, mapping)."""
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


def both_constructions(graph, supergraph) -> dict[str, torch.Tensor]:
    return {
        "normalised": aggregate_supernode_adjacency(graph, supergraph.supernodes, aggregation="normalised"),
        "raw-signed": aggregate_supernode_adjacency(graph, supergraph.supernodes, aggregation="raw-signed",
                                                   token_source_columns=True),
    }


def losses(WT: torch.Tensor, WS: torch.Tensor, n: int) -> dict[str, float]:
    ids = list(range(n))
    mapping = {i: {i} for i in ids}
    return {
        "jsd": float(_compute_edge_loss(WT, WS, mapping, ids, ids, similarity="jsd")),
        "rel_mse": float(_compute_edge_loss(WT, WS, mapping, ids, ids, similarity="rel-mse")),
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def render(out_path: str, prompt: str, rec: dict[str, Any], teacher_p: float, labels: list[str],
           token_strs: list[str], mats: dict[str, dict[str, torch.Tensor]], loss_by: dict[str, dict[str, float]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.2))
    rows = [("normalised", "magma", "jsd", "row JSD"), ("raw-signed", "RdBu_r", "rel_mse", "rel. sq. error")]
    for r, (cons, cmap, key, name) in enumerate(rows):
        T, S = mats[cons]["teacher"], mats[cons]["student"]
        D = S - T
        vmax = float(max(T.abs().max(), S.abs().max(), 1e-8))
        panels = [("teacher", T), ("student", S), ("student - teacher", D)]
        for c, (title, M) in enumerate(panels):
            ax = axes[r][c]
            if cons == "normalised" and c < 2:
                im = ax.imshow(M.numpy(), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
            else:
                lim = vmax if c < 2 else float(D.abs().max().clamp(min=1e-8))
                im = ax.imshow(M.numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
            cols = labels + ([f"tok:{t}" for t in token_strs] if M.shape[1] > len(labels) else [])
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels(cols, rotation=90, fontsize=7)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels if c == 0 else [], fontsize=7)
            ax.set_title(title, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        axes[r][0].set_ylabel(f"{cons}\n({name} = {loss_by[cons][key]:.4f})", fontsize=9)
    fig.suptitle(
        f"{prompt!r}  gold {rec['gold']}  |  student argmax {rec['argmax_str']!r} "
        f"p(gold)={rec['p_gold']:.2f}  |  teacher p(gold)={teacher_p:.2f}",
        fontsize=10,
    )
    fig.tight_layout()
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
    ap.add_argument("--teacher", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--dataset", default="22_add")
    ap.add_argument("--split", choices=["test", "train"], default="test")
    ap.add_argument("--n-scan", type=int, default=400, help="Prompts scored before bucketing.")
    ap.add_argument("--n-per-bucket", type=int, default=3)
    ap.add_argument("--confident", type=float, default=0.8)
    ap.add_argument("--unsure", type=float, default=0.5)
    ap.add_argument("--graph-node-labels", nargs="+", default=ANOVA_LABELS)
    ap.add_argument("--nodes-per-label", type=int, default=10)
    ap.add_argument("--teacher-prop-neurons", type=float, default=0.003)
    ap.add_argument("--student-prop-neurons", type=float, default=0.01)
    ap.add_argument("--teacher-graph-batch-size", type=int, default=512)
    ap.add_argument("--student-graph-batch-size", type=int, default=128)
    ap.add_argument("--mlp-cache-batch-size", type=int, default=32)
    ap.add_argument("--score-batch-size", type=int, default=64)
    ap.add_argument("--out", default=os.path.join(DIR_ROOT, "results", "inspect_graphs"))
    args = ap.parse_args()

    train_data, test_data = load_data(args.dataset)
    data = test_data if args.split == "test" else train_data
    items = list(data.items())[: args.n_scan]
    prompts = [p for p, _ in items]
    answers = [a for _, a in items]

    student, tokenizer = load_student(args.student_checkpoint or args.student)
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
    student_cache = build_mlp_input_cache(student_adapter, args.dataset, args.student, data_dict=train_data,
                                          batch_size=args.mlp_cache_batch_size)
    teacher_cache = build_mlp_input_cache(teacher_adapter, args.dataset, args.teacher, data_dict=train_data,
                                          batch_size=args.mlp_cache_batch_size)
    config = GraphAuxConfig(
        graph_dtype=torch.bfloat16,
        teacher_prop_neurons_per_layer=args.teacher_prop_neurons,
        student_prop_neurons_per_layer=args.student_prop_neurons,
        top_k_logits=0.95, temperature=1.0,
        teacher_graph_batch_size=args.teacher_graph_batch_size,
        student_graph_batch_size=args.student_graph_batch_size,
        student_nodes_per_label=args.nodes_per_label, teacher_nodes_per_label=args.nodes_per_label,
        graph_node_labels=list(args.graph_node_labels),
        mlp_input_cache=student_cache, teacher_mlp_input_cache=teacher_cache,
        supergraph_aggregation="raw-signed", token_source_columns=True,
        dataset_name=args.dataset,
    )

    records: list[dict[str, Any]] = []
    for b in BUCKETS:
        for i in picked[b]:
            prompt, answer = prompts[i], answers[i]
            print(f"[{b}] building graphs for {prompt!r} ...")
            # Mirrors _compute_teacher_target, but keeps the teacher's Graph so both
            # constructions can be aggregated from the same attribution.
            p_ids, a_ids = tokenize_prompt_answer(tokenizer, prompt, str(answer))
            full_ids = torch.cat([p_ids, a_ids]).to(DEVICE)
            with torch.no_grad():
                teacher_dla_logits = teacher(full_ids.unsqueeze(0)).logits[0, p_ids.numel() - 1].detach()
            with torch.enable_grad():
                t_res = create_graph(
                    teacher_adapter, prompt,
                    prop_neurons_per_layer=config.teacher_prop_neurons_per_layer,
                    top_k_logits=config.top_k_logits, temperature=config.temperature,
                    batch_size=config.teacher_graph_batch_size, node_labels=config.graph_node_labels,
                    mlp_input_cache=config.teacher_mlp_input_cache, nodes_per_label=config.teacher_nodes_per_label,
                    no_grad_supergraph=True, build_create_graph=False, detach_result=True,
                    supergraph_aggregation="raw-signed", token_source_columns=True,
                )
                logit_token_ids = t_res.graph.logit_token_ids.to(DEVICE)
                s_res = create_graph(
                    student_adapter, prompt,
                    attribution_targets=logit_token_ids.cpu() if logit_token_ids is not None else None,
                    prop_neurons_per_layer=config.student_prop_neurons_per_layer,
                    top_k_logits=config.top_k_logits, temperature=config.temperature,
                    batch_size=config.student_graph_batch_size, dtype=config.graph_dtype,
                    build_create_graph=False, detach_result=True, skip_logit_attribution=False,
                    mlp_input_cache=config.mlp_input_cache, node_labels=config.graph_node_labels or [],
                    nodes_per_label=config.student_nodes_per_label, dla_model_logits=teacher_dla_logits,
                    no_grad_supergraph=True, supergraph_aggregation="raw-signed", token_source_columns=True,
                )
            labels, t_ids, s_ids = align(t_res.supergraph, s_res.supergraph)
            if not labels:
                print("   no shared supernode labels; skipping")
                continue
            nT, nS = len(t_res.supergraph.supernodes), len(s_res.supergraph.supernodes)
            with torch.no_grad():
                T_by = both_constructions(t_res.graph, t_res.supergraph)
                S_by = both_constructions(s_res.graph, s_res.supergraph)
            mats = {cons: {"teacher": submatrix(T_by[cons], t_ids, nT), "student": submatrix(S_by[cons], s_ids, nS)}
                    for cons in T_by}
            loss_by = {cons: losses(mats[cons]["teacher"], mats[cons]["student"], len(labels)) for cons in mats}
            token_strs = [tokenizer.decode([int(t)]) for t in t_res.graph.input_tokens.tolist()]
            rec = dict(s_scores[i])
            rec.update(bucket=b, tags=attributes(prompt), teacher_p_gold=t_scores[i]["p_gold"],
                       labels=labels, losses=loss_by,
                       matrices={cons: {k: v.tolist() for k, v in m.items()} for cons, m in mats.items()})
            records.append(rec)
            safe = re.sub(r"[^0-9a-zA-Z]+", "_", prompt).strip("_")
            render(os.path.join(args.out, b, f"{safe}.png"), prompt, rec, t_scores[i]["p_gold"], labels,
                   token_strs, mats, loss_by)
            print(f"   normalised JSD {loss_by['normalised']['jsd']:.4f} | raw-signed rel-mse {loss_by['raw-signed']['rel_mse']:.4f}")
            del t_res, s_res
            torch.cuda.empty_cache()

    # ---- summary ------------------------------------------------------------
    summary: dict[str, Any] = {"args": vars(args), "n_scanned": len(prompts), "student_right": n_right,
                               "buckets": {}, "prompts": records}
    for b in BUCKETS:
        rs = [r for r in records if r["bucket"] == b]
        if rs:
            summary["buckets"][b] = {
                "n": len(rs),
                "jsd_normalised": sum(r["losses"]["normalised"]["jsd"] for r in rs) / len(rs),
                "rel_mse_raw_signed": sum(r["losses"]["raw-signed"]["rel_mse"] for r in rs) / len(rs),
                "student_p_gold": sum(r["p_gold"] for r in rs) / len(rs),
            }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nper-bucket means:")
    for b, v in summary["buckets"].items():
        print(f"  {b:16s} n={v['n']}  JSD(normalised)={v['jsd_normalised']:.4f}  rel-mse(raw-signed)={v['rel_mse_raw_signed']:.4f}  p(gold)={v['student_p_gold']:.2f}")
    print("wrote", os.path.join(args.out, "summary.json"))


if __name__ == "__main__":
    main()

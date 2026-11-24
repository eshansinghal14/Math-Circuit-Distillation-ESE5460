"""The attribution graph's answer row: does matching it point the student at the fix?

The supergraph the trainer scores holds rows for neuron groups only: with
--graph-rows last, the inputs of the last-position node and of the DLA node.
The edges INTO THE ANSWER, each node's attribution to the answer logit, are
computed for every graph (Graph.adjacency_matrix's logit rows) and then left
out. They are the edges whose sign tracks what knocking a node out really does
(experiments/neuron_selection_eval.py --by-position: rank correlation 0.6-0.7
at 10-100 neurons per node, against 0.07-0.15 for their absolute values).

This builds that row for teacher and student, with no neuron rows at all (one
backward pass per answer token, whatever the node size):

    row[node]  = sum over the node's neurons of their attribution to the answer,
                 the teacher's salient next tokens weighted by its probabilities
    row[token] = the token embedding's direct attribution, one entry per position

in logit units, signed, and compares the two models' rows two ways:

    signed   relative squared error between the signed rows (the size of each
             contribution counts, so a student that does not produce the
             teacher's answer is far from it)
    shares   Jensen-Shannon distance between the |.|-normalised rows (what the
             trainer does to the rows it scores today)

For each, the gradient on the student's weights is compared with the KD
gradient and the gold cross-entropy gradient on the same prompts, exactly as
experiments/graph_grad_coherence.py does for the trainer's graph loss.

    cd src && python -u -m experiments.answer_row_check --teacher <t> --student <s> \\
        --dataset svamp_eq --n 16 --nodes-per-label 100 --teacher-prop 0.03 --student-prop 0.03
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DIR_ROOT, PromptAnswerDataset, collate_fn, load_data, load_model, set_bos_mode  # noqa: E402
from training.utils import kl_loss, kl_position_mask, load_student, student_autocast  # noqa: E402
from graph_loss.create_graph import create_graph  # noqa: E402
from graph_loss.hf_adapter import HFLlamaGraphAdapter  # noqa: E402
from graph_loss.loss import edge_similarity  # noqa: E402
from experiments.graph_grad_coherence import Sketcher, coherence, cross, gram  # noqa: E402

LOSSES = ("signed", "shares")
REFS = ("kl", "kl1", "ce", "ce1")


def answer_row(result, weights: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """({node label: signed attribution to the weighted answer logit}, the same per token embedding)."""
    g = result.graph
    n, t = g.n_neurons, g.n_tokens
    A = g.adjacency_matrix[n + t:, : n + t].float()             # [answer tokens, neurons + tokens]
    w = weights.to(A.device).float()
    row = ((w / w.sum())[:, None] * A).sum(0)
    nodes = {}
    for members, labels in zip(result.supergraph.supernodes, result.supergraph.supernode_labels or []):
        if labels and labels[0].startswith("pos:") and members:
            nodes[labels[0]] = row[torch.tensor(members, device=A.device, dtype=torch.long)].sum()
    return nodes, row[n: n + t]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--dataset", default="svamp_eq")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--nodes-per-label", type=int, default=100)
    ap.add_argument("--teacher-prop", type=float, default=0.03)
    ap.add_argument("--student-prop", type=float, default=0.03)
    ap.add_argument("--neuron-selection", choices=["norm", "influence"], default="influence")
    ap.add_argument("--no-tokens", action="store_true", help="Leave the token-embedding entries out of the row.")
    ap.add_argument("--sketch-chunk", type=int, default=4096)
    ap.add_argument("--tag", default="default")
    args = ap.parse_args()

    set_bos_mode("on")
    dev = torch.device("cuda")
    train, _ = load_data(args.dataset)
    every = sorted(train.items())
    items = every[::max(1, len(every) // args.n)][: args.n]

    student, tok = load_student(args.student, "bfloat16")
    student.train()
    s_ad = HFLlamaGraphAdapter(student, tok, dev)
    teacher, _ = load_model(args.teacher)
    teacher.eval()
    for p_ in teacher.parameters():
        p_.requires_grad_(False)
    t_ad = HFLlamaGraphAdapter(teacher, tok, dev)
    sk = Sketcher(student, args.sketch_chunk)

    common = dict(top_k_logits=0.95, temperature=1.0, dtype=torch.bfloat16, node_labels=[],
                  nodes_per_label=args.nodes_per_label, no_grad_supergraph=True, build_create_graph=False,
                  neuron_selection=args.neuron_selection, supernode_type="position", target_rows="none")
    names = LOSSES + REFS
    Z = {k: [] for k in names}
    N = {k: [] for k in names}
    recs = []
    t0 = time.perf_counter()
    for i, (prompt, answer) in enumerate(items):
        with torch.enable_grad():
            t_res = create_graph(t_ad, prompt, prop_neurons_per_layer=args.teacher_prop, batch_size=8,
                                 detach_result=True, **common)
        ids_t = t_res.graph.logit_token_ids.to(dev)
        probs = t_res.graph.logit_probabilities.detach().float().to(dev)
        t_logits_end = t_res.target_position_logits.to(dev)
        t_nodes, t_tok = answer_row(t_res, probs)
        t_nodes = {k: v.detach() for k, v in t_nodes.items()}
        t_tok = t_tok.detach()
        del t_res

        student.zero_grad(set_to_none=True)
        s_res = create_graph(s_ad, prompt, attribution_targets=ids_t.cpu(), dla_model_logits=t_logits_end,
                             prop_neurons_per_layer=args.student_prop, batch_size=8, detach_result=False,
                             skip_logit_attribution=False, **common)
        s_nodes, s_tok = answer_row(s_res, probs)
        labels = [k for k in t_nodes if k in s_nodes]
        t_row = torch.stack([t_nodes[k] for k in labels])
        s_row = torch.stack([s_nodes[k] for k in labels])
        if not args.no_tokens:
            t_row, s_row = torch.cat([t_row, t_tok]), torch.cat([s_row, s_tok])
        losses = {"signed": edge_similarity(t_row[None], s_row[None], "rel-mse"),
                  "shares": edge_similarity(t_row[None], s_row[None], "jsd")}
        got = {}
        ok = all(v.requires_grad and torch.isfinite(v) for v in losses.values())
        if ok:
            for j, name in enumerate(LOSSES):
                student.zero_grad(set_to_none=True)
                losses[name].backward(retain_graph=j + 1 < len(LOSSES))
                got[name] = sk.grab()[:2]
        vals = {k: float(v.detach()) for k, v in losses.items()}
        size_t, size_s = float(t_row.abs().sum()), float(s_row.detach().abs().sum())
        del losses, s_res, s_nodes, s_tok, s_row
        gc.collect()
        torch.cuda.empty_cache()
        if not ok:
            print(f"  [{i}] answer row unusable; skipped", flush=True)
            continue

        ds = PromptAnswerDataset(args.dataset, {prompt: answer}, tok)
        batch = collate_fn([ds[0]], pad_id=tok.eos_token_id)
        ids, am, rm = (batch[k].to(dev) for k in ("input_ids", "attention_mask", "response_mask"))
        with torch.no_grad():
            t_logits = teacher(ids, attention_mask=am).logits
        resp = kl_position_mask(am, rm, "resp", input_ids=ids, bos_token_id=tok.bos_token_id)
        pos = resp[0].nonzero().squeeze(1)
        first = int(pos[0])
        only_first = torch.zeros_like(resp)
        only_first[0, first] = True
        for name in REFS:
            student.zero_grad(set_to_none=True)
            with student_autocast():
                s_logits = student(ids, attention_mask=am).logits
            if name == "kl":
                obj = kl_loss(s_logits, t_logits, resp, 1.0)
            elif name == "kl1":
                obj = kl_loss(s_logits, t_logits, only_first, 1.0)
            elif name == "ce":
                obj = F.cross_entropy(s_logits[0, pos].float(), ids[0, pos + 1])
            else:
                obj = F.cross_entropy(s_logits[0, first: first + 1].float(), ids[0, first + 1: first + 2])
            obj.backward()
            got[name] = sk.grab()[:2]
            del s_logits, obj
        del t_logits
        if not all(torch.isfinite(z).all() and nrm > 0 for z, nrm in got.values()):
            print(f"  [{i}] non-finite or zero gradient; skipped", flush=True)
            continue
        for k, (z, nrm) in got.items():
            Z[k].append(z)
            N[k].append(nrm)
        recs.append({"i": i, **vals, "nodes": len(labels), "row_size_teacher": size_t, "row_size_student": size_s})
        print(f"  [{i}] signed {vals['signed']:.4f} |g| {got['signed'][1]:.3f} | shares {vals['shares']:.4f} "
              f"|g| {got['shares'][1]:.3f} | {len(labels)} nodes | row size T {size_t:.2f} S {size_s:.2f} "
              f"| {time.perf_counter() - t0:.0f}s", flush=True)
    student.zero_grad(set_to_none=True)

    n = len(recs)
    Zt = {k: torch.stack(v) for k, v in Z.items()}
    Nt = {k: torch.tensor(v, dtype=torch.float64) for k, v in N.items()}
    summary = {"args": vars(args), "n_prompts": n, "records": recs}
    print(f"\n{n} prompts | {args.nodes_per_label} neurons per node | signed loss "
          f"{sum(r['signed'] for r in recs) / n:.4f} | shares loss {sum(r['shares'] for r in recs) / n:.4f} | "
          f"row size teacher {sum(r['row_size_teacher'] for r in recs) / n:.2f} "
          f"student {sum(r['row_size_student'] for r in recs) / n:.2f}")
    for name in LOSSES:
        summary[name] = coherence(f"answer row, {name}", Zt[name], Nt[name])
    summary["kl"] = coherence("KD, response positions", Zt["kl"], Nt["kl"])
    mean_norm = lambda k: float(gram(Zt[k], Nt[k]).sum().clamp(min=0).sqrt() / n)  # noqa: E731
    print("\n== answer row against the directions that fix the student (cosine of the mean gradients; same-prompt mean) ==")
    print(f"{'':10s} " + " ".join(f"{r:>22s}" for r in ("KD, response", "KD, first token", "gold, response", "gold, first token")))
    summary["alignment"] = {}
    for name in LOSSES:
        cells, row = [], {}
        for ref in REFS:
            c = float((Zt[name].mean(0) @ Zt[ref].mean(0)) / (mean_norm(name) * mean_norm(ref)))
            same = torch.diagonal(Zt[name] @ Zt[ref].T) / (Nt[name] * Nt[ref])
            row[ref] = {"cos_of_means": c, "same_prompt_mean": float(same.mean()),
                        "same_prompt_se": float(same.std() / n ** 0.5)}
            cells.append(f"{c:+8.3f} ({float(same.mean()):+.3f})")
        summary["alignment"][name] = row
        print(f"{name:10s} " + " ".join(f"{c:>22s}" for c in cells))
    print("\n== the two comparisons against each other ==")
    summary["signed_vs_shares"] = cross("signed vs shares", Zt["signed"], Nt["signed"], Zt["shares"], Nt["shares"])

    path = os.path.join(DIR_ROOT, f"FINAL_RESULTS/{args.dataset}/answer_row_check_{args.tag}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

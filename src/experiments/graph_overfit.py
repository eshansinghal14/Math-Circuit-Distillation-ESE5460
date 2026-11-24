"""Does matching the teacher's graph make the student give the teacher's answer? (diagnostic, single problems)

For problems where the student's prediction at the graph position differs from
the teacher's, starting each time from the same student weights, take S Adam
steps on ONE problem with one of three objectives and track, after every step,
the objective, the graph loss, and the student's probability of the teacher's
token at that position:

    graph      the trainer's graph loss against this problem's teacher graph
    shuffled   the same loss against ANOTHER problem's teacher graph (control:
               what moving the graph anywhere does to the answer)
    kd         KD on that one token (reference: what fixing the answer looks like)

If "graph" drives its loss down but moves the teacher-token probability no more
than "shuffled" does, a matched graph does not imply a matched answer: the
graph underdetermines the computation, and no amount of graph training can
transmit the teacher's answer through it.

Graph-only steps here are a probe of what the loss constrains, not a training
method.

    cd src && python -u -m experiments.graph_overfit --teacher <8B> --student <kd 1B> --n 12 --steps 15
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DIR_ROOT, PromptAnswerDataset, collate_fn, load_model, load_split, set_bos_mode  # noqa: E402
from training.utils import load_student, student_autocast  # noqa: E402
import graph_loss.training as GT  # noqa: E402
from graph_loss.hf_adapter import HFLlamaGraphAdapter  # noqa: E402
from experiments.graph_gate_two_phase import graph_cut  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--dataset", default="svamp_eq")
    ap.add_argument("--split", default="test")
    ap.add_argument("--pool", type=int, default=150, help="Problems scanned for student != teacher.")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--graph-position", choices=["prompt-end", "operator", "result"], default="prompt-end")
    ap.add_argument("--nodes-per-label", type=int, default=100)
    ap.add_argument("--tag", default="default")
    args = ap.parse_args()

    set_bos_mode("on")
    dev = torch.device("cuda")
    K = args.nodes_per_label
    config = GT.GraphAuxConfig(
        graph_dtype=torch.bfloat16, teacher_prop_neurons_per_layer=0.01, student_prop_neurons_per_layer=0.03,
        top_k_logits=0.95, temperature=1.0, teacher_graph_batch_size=16, student_graph_batch_size=32,
        student_nodes_per_label=K, teacher_nodes_per_label=K, graph_loss_type="jsd", graph_node_labels=None,
        token_path_rows="weighted", dataset_name=args.dataset, graph_rows="last", graph_edge_topk=0,
        neuron_selection="influence", supernode_type="position", position_bands=1, restrict_target_rows=True)
    every = sorted(load_split(args.dataset, args.split).items())
    pool = every[::max(1, len(every) // args.pool)][: args.pool]

    def seq(tok, prompt, answer):
        b = collate_fn([PromptAnswerDataset(args.dataset, {prompt: answer}, tok)[0]], pad_id=tok.eos_token_id)
        return b["input_ids"].to(dev), b["attention_mask"].to(dev)

    # ---- phase 1: teacher predictions on the pool; graphs for the chosen problems
    teacher, tok = load_model(args.teacher)
    teacher.eval()
    for p_ in teacher.parameters():
        p_.requires_grad_(False)
    cands = []
    for prompt, answer in pool:
        cut = graph_cut(prompt, answer, tok, args.dataset, args.graph_position)
        if cut is None:
            continue
        ids, am = seq(tok, prompt, answer)
        with torch.no_grad():
            t_top = int(teacher(ids, attention_mask=am).logits[0, cut[2] - 1].argmax())
        cands.append((prompt, answer, cut, t_top))
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    student, _ = load_student(args.student, "bfloat16")
    chosen = []
    for prompt, answer, cut, t_top in cands:
        ids, am = seq(tok, prompt, answer)
        with torch.no_grad(), student_autocast():
            s_top = int(student(ids, attention_mask=am).logits[0, cut[2] - 1].argmax())
        if s_top != t_top:
            chosen.append((prompt, answer, cut, t_top))
    chosen = chosen[: args.n]
    print(f"{len(chosen)} problems where the student's token at {args.graph_position} differs from the teacher's "
          f"(of {len(cands)} scanned)", flush=True)
    del student
    gc.collect()
    torch.cuda.empty_cache()

    teacher, _ = load_model(args.teacher)
    teacher.eval()
    for p_ in teacher.parameters():
        p_.requires_grad_(False)
    t_ad = HFLlamaGraphAdapter(teacher, tok, dev)
    entries = []
    for prompt, answer, (g_prompt, g_next, tgt), t_top in chosen:
        t_sg, logit_ids, dla = GT._compute_teacher_target(g_prompt, g_next, t_ad, config, dev)
        entries.append(GT._teacher_target_entry(t_sg, logit_ids, dla, answer=g_next))
    del t_ad, teacher
    gc.collect()
    torch.cuda.empty_cache()

    # ---- phase 2: single-problem optimisation from the same student weights each time
    student, _ = load_student(args.student, "bfloat16")
    student.train()
    s_ad = HFLlamaGraphAdapter(student, tok, dev)
    start = {k: v.detach().clone() for k, v in student.state_dict().items()}
    results = []
    for i, (prompt, answer, (g_prompt, g_next, tgt), t_top) in enumerate(chosen):
        ids, am = seq(tok, prompt, answer)
        other = entries[(i + 1) % len(entries)]
        rec = {"i": i, "teacher_token": tok.decode([t_top]), "gold_token": g_next}
        for mode in ("graph", "shuffled", "kd"):
            student.load_state_dict(start)
            opt = torch.optim.AdamW(student.parameters(), lr=args.lr, foreach=False)
            entry = entries[i] if mode != "shuffled" else other
            trace = []
            for step in range(args.steps + 1):
                # measure: graph loss against THIS problem's teacher graph, and P(teacher token)
                opt.zero_grad(set_to_none=True)
                t_sg, logit_ids, dla = GT._teacher_target_from_entry(entries[i], dev)
                # not under no_grad: building the attribution graph itself takes gradients
                g_true, _ = GT.compute_prompt_graph_loss(
                    prompt=g_prompt, student_adapter=s_ad, config=config, teacher_supergraph=t_sg,
                    logit_token_ids=logit_ids, teacher_dla_logits=dla, answer=g_next)
                g_true = g_true.detach()
                with student_autocast():
                    logits = student(ids, attention_mask=am).logits[0, tgt - 1].float()
                logp = F.log_softmax(logits, -1)
                trace.append({"step": step, "graph_loss": float(g_true), "p_teacher": float(logp[t_top].exp()),
                              "top_is_teacher": int(logits.argmax()) == t_top})
                if step == args.steps:
                    break
                # update
                opt.zero_grad(set_to_none=True)
                if mode == "kd":
                    obj = -logp[t_top]
                else:
                    t_sg, logit_ids, dla = GT._teacher_target_from_entry(entry, dev)
                    obj, _ = GT.compute_prompt_graph_loss(
                        prompt=g_prompt, student_adapter=s_ad, config=config, teacher_supergraph=t_sg,
                        logit_token_ids=logit_ids, teacher_dla_logits=dla, answer=g_next)
                obj.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()
                del obj, logits, logp
                gc.collect()
                torch.cuda.empty_cache()
            rec[mode] = trace
            a, b = trace[0], trace[-1]
            print(f"  [{i}] {mode:8s} graph loss {a['graph_loss']:.4f} -> {b['graph_loss']:.4f} | P(teacher token "
                  f"{rec['teacher_token']!r}) {a['p_teacher']:.3f} -> {b['p_teacher']:.3f} | flipped "
                  f"{int(b['top_is_teacher'])}", flush=True)
        results.append(rec)
    student.load_state_dict(start)

    print(f"\n== {len(results)} problems, {args.steps} steps each, lr {args.lr}, graph at {args.graph_position} ==")
    print(f"{'objective':10s} {'graph loss drop':>16s} {'P(teacher) change':>18s} {'answers flipped':>16s}")
    summary = {}
    for mode in ("graph", "shuffled", "kd"):
        dg = [r[mode][0]["graph_loss"] - r[mode][-1]["graph_loss"] for r in results]
        dp = [r[mode][-1]["p_teacher"] - r[mode][0]["p_teacher"] for r in results]
        fl = sum(r[mode][-1]["top_is_teacher"] for r in results)
        summary[mode] = {"graph_loss_drop": sum(dg) / len(dg), "p_teacher_change": sum(dp) / len(dp), "flipped": fl}
        print(f"{mode:10s} {sum(dg) / len(dg):16.4f} {sum(dp) / len(dp):+18.3f} {fl:>11d} of {len(results)}")
    path = os.path.join(DIR_ROOT, f"FINAL_RESULTS/{args.dataset}/graph_overfit_{args.tag}.json")
    with open(path, "w") as f:
        json.dump({"args": vars(args), "summary": summary, "results": results}, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

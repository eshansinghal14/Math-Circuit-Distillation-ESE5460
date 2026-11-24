"""What the swap graph looks like for a teacher and for students, before any training on it.

For N training prompts, one number per prompt is swapped for a replacement at
every depth in turn, through each of the two nodes (training/swap_graph.py: the
number's own position, and every later position). Reported per depth:

  every model   effect of each node: how far its swap moves the first response
                token (KL clean || swapped, nats), next to the effect of the
                number-changed prompt itself; follow: of the prompts whose
                first response token was the swapped number, the share where
                the number-node swap turns it into the replacement.
  each student  the graph loss of each node at that depth
                (KL teacher-swapped || student-swapped over the response
                positions), next to the KD loss on the unswapped prompts.

A student that KD has already brought to the teacher's outputs but whose graph
loss stays high at the deeper nodes has structure left to learn from the graph;
if the graph loss falls with the KD loss, the graph is redundant with KD again.

Self-checks, printed first for the teacher and the first student:
  own state      swapping in the model's own state changes nothing;
  depth 0        a depth-0 number swap equals running the number-changed prompt;
  completeness   at every depth, swapping both nodes equals running the
                 number-changed prompt: the two nodes carry the number's whole
                 influence, with no part of the model left out.

    cd src && python -u -m experiments.swap_graph_inspect --teacher <teacher> \\
        --student raw=meta-llama/Llama-3.2-1B-Instruct --student kd=<checkpoint> --dataset svamp_eq --n 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DIR_ROOT, PromptAnswerDataset, collate_fn, load_data, load_model, set_bos_mode  # noqa: E402
from training.utils import kl_loss, kl_position_mask, load_student, student_autocast  # noqa: E402
from training.swap_graph import NumberTokens, block_index, pick_edges, prompt_end_effect, swap_forward  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", action="append", required=True, help="label=path; repeatable")
    ap.add_argument("--dataset", default="svamp_eq")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--depths", default="0,0.125,0.25,0.375,0.5,0.625,0.75,0.875")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="default")
    args = ap.parse_args()

    set_bos_mode("on")
    dev = torch.device("cuda")
    depths = [float(x) for x in args.depths.split(",")]
    students = {}
    tok = None
    for spec in args.student:
        label, path = spec.split("=", 1)
        model, tok = load_student(path, "bfloat16")
        model.eval()
        students[label] = model
    teacher, _ = load_model(args.teacher)
    teacher.eval()
    numbers = NumberTokens(tok)
    gen = torch.Generator().manual_seed(args.seed)

    train, _ = load_data(args.dataset)
    every = sorted(train.items())
    items = every[::max(1, len(every) // args.n)][: args.n]
    data = PromptAnswerDataset(args.dataset, dict(items), tok)
    batches = [collate_fn([data[i] for i in range(s, min(s + args.batch_size, len(data)))], pad_id=tok.eos_token_id)
               for s in range(0, len(data), args.batch_size)]

    names = ["teacher"] + list(students)
    models = {"teacher": teacher, **students}

    def run(m, ids, am, edges=None):
        with torch.no_grad():
            if m == "teacher":
                return (teacher(ids, attention_mask=am).logits if edges is None
                        else swap_forward(teacher, ids, am, edges, depths))
            with student_autocast():
                return (models[m](ids, attention_mask=am).logits if edges is None
                        else swap_forward(models[m], ids, am, edges, depths))

    D = len(depths)
    keys = ("effect_number", "effect_rest", "follow", "follow_n", "graph_number", "graph_rest", "top_same")
    S = {m: {k: [0.0] * D for k in keys} for m in names}
    full = {m: 0.0 for m in names}
    kd = {m: 0.0 for m in students}
    n_edges = 0
    for bi, batch in enumerate(batches):
        ids, am, rm = (batch[k].to(dev) for k in ("input_ids", "attention_mask", "response_mask"))
        mask = kl_position_mask(am, rm, "resp", input_ids=ids, bos_token_id=tok.bos_token_id)
        first = rm.argmax(dim=1)
        base = pick_edges(ids, first.cpu(), numbers, [0.0], gen).to(dev)
        rows = torch.arange(ids.size(0), device=dev)
        has = base.has
        n_has = int(has.sum())
        n_edges += n_has
        scored = mask & has[:, None]
        valid = am.bool() & has[:, None]
        old_tok = ids[rows, base.pos]
        new_tok = base.twin_ids[rows, base.pos]
        if bi == 0:
            for i in range(min(3, ids.size(0))):
                nodes = numbers.standalone(ids[i].cpu(), int(first[i]))
                print(f"  prompt {i}: numbers at {nodes} = {[tok.decode(ids[i, p]) for p in nodes]}; "
                      f"swapping {tok.decode(old_tok[i])!r} -> {tok.decode(new_tok[i])!r}", flush=True)

        clean = {m: run(m, ids, am) for m in names}
        twin = {m: run(m, base.twin_ids, am) for m in names}
        for m in names:
            full[m] += float(prompt_end_effect(clean[m], twin[m], first)[has].sum())
        for m in students:
            kd[m] += float(kl_loss(clean[m], clean["teacher"], scored, 1.0)) * n_has

        if bi == 0:
            for m in names[:2]:
                own = base.at(D // 2)
                own.twin_ids = ids.clone()
                d_own = float((run(m, ids, am, own).float() - clean[m].float()).abs().max())
                d_zero = float((run(m, ids, am, base.at(0)).float() - twin[m].float())[valid].abs().max())
                d_both = max(float((run(m, ids, am, base.at(d, True, True)).float() - twin[m].float())[valid].abs().max())
                             for d in range(D))
                print(f"  self-check {m}: own-state swap moves logits by {d_own:.2e} | depth-0 swap vs "
                      f"number-changed prompt {d_zero:.2e} | both nodes swapped vs number-changed prompt, "
                      f"worst depth {d_both:.2e} | (mean |logit| {float(clean[m].float().abs().mean()):.2f})",
                      flush=True)

        for d in range(D):
            for kind in ("number", "rest"):
                if kind == "rest" and depths[d] == 0:
                    continue  # the later tokens' embeddings do not depend on the number
                edges = base.at(d, kind == "number", kind == "rest")
                swapped = {m: run(m, ids, am, edges) for m in names}
                t_top = swapped["teacher"][rows, first - 1].argmax(-1)
                for m in names:
                    eff = prompt_end_effect(clean[m], swapped[m], first)
                    S[m][f"effect_{kind}"][d] += float(eff[has].sum())
                    top_swap = swapped[m][rows, first - 1].argmax(-1)
                    if kind == "number":
                        copied = has & (clean[m][rows, first - 1].argmax(-1) == old_tok)
                        S[m]["follow"][d] += float((top_swap[copied] == new_tok[copied]).sum())
                        S[m]["follow_n"][d] += float(copied.sum())
                        if m != "teacher":
                            S[m]["top_same"][d] += float((top_swap[has] == t_top[has]).sum())
                    if m != "teacher":
                        S[m][f"graph_{kind}"][d] += float(kl_loss(swapped[m], swapped["teacher"], scored, 1.0)) * n_has
                del swapped
        del clean, twin
        print(f"  batch {bi + 1}/{len(batches)} done", flush=True)

    n = max(n_edges, 1)
    out = {"teacher": args.teacher, "students": dict(s.split("=", 1) for s in args.student), "dataset": args.dataset,
           "n_prompts": len(data), "n_edges": n_edges, "depths": depths, "models": {}}
    print(f"\n{n_edges} prompts with a number, of {len(data)} | teacher blocks "
          f"{[block_index(teacher, d) for d in depths]} | student blocks "
          f"{[block_index(next(iter(students.values())), d) for d in depths]}")
    for m in names:
        rec = {"effect_number_changed_prompt": full[m] / n,
               "effect_number": [v / n for v in S[m]["effect_number"]],
               "effect_rest": [v / n for v in S[m]["effect_rest"]],
               "follow": [a / b if b else float("nan") for a, b in zip(S[m]["follow"], S[m]["follow_n"])],
               "follow_n": S[m]["follow_n"]}
        print(f"\n== {m} == effect of the number-changed prompt itself: {rec['effect_number_changed_prompt']:.3f} nats")
        head = f"{'depth':>7s} {'number node':>12s} {'rest node':>10s} {'follow':>7s} {'(n)':>5s}"
        if m != "teacher":
            rec["kd_loss_unswapped"] = kd[m] / n
            rec["graph_number"] = [v / n for v in S[m]["graph_number"]]
            rec["graph_rest"] = [v / n for v in S[m]["graph_rest"]]
            rec["first_token_agrees_with_teacher"] = [v / n for v in S[m]["top_same"]]
            print(f"KD loss on the unswapped prompts: {rec['kd_loss_unswapped']:.4f}")
            head += f" | {'graph loss: number':>19s} {'rest':>8s} {'1st tok = teacher':>18s}"
        print(head)
        for d, depth in enumerate(depths):
            line = (f"{depth:7.3f} {rec['effect_number'][d]:12.3f} {rec['effect_rest'][d]:10.3f} "
                    f"{rec['follow'][d]:7.3f} {int(S[m]['follow_n'][d]):5d}")
            if m != "teacher":
                line += (f" | {rec['graph_number'][d]:19.4f} {rec['graph_rest'][d]:8.4f} "
                         f"{rec['first_token_agrees_with_teacher'][d]:18.3f}")
            print(line)
        out["models"][m] = rec

    path = os.path.join(DIR_ROOT, f"FINAL_RESULTS/{args.dataset}/swap_graph_inspect_{args.tag}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

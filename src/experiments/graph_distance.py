"""How far is each model's attribution graph from the teacher's, and does that distance track correctness?

The teacher's graph target is built once per test problem. Each listed model is
then scored with the trainer's graph loss against it (no training, no
gradients kept), and its greedy answer token is checked against the gold one.
Reported per model: test accuracy on the scored problems, mean graph loss,
mean graph loss on the problems it gets right and wrong, and the AUC of graph
loss for "wrong" (0.5 = the graph distance says nothing about correctness).
The KD loss at the answer position, KL(teacher || student), is recorded per
problem next to it (and its AUC), so the graph's signal can be compared with
the one KD already has.

--answer-tokens (Qwen, Gemma: one digit per token): the graph loss is the mean
over one graph per answer token (the trainer's answer-tokens cuts), the KD loss
the mean over the answer tokens, and "right" is the greedy answer's leading
integer equal to the gold one (otherwise "right" is the first answer token).

--node-labels (e.g. the mix3 factor categories "x1 tens" ... "result units"
"carry1"): ANOVA supernodes instead of position supernodes, labelled from MLP
input caches over the first --anova-prompts training problems (one per model).

Questions it answers (mix3, 2026-11-18):
  - Does KD alone already bring the student's graph to the teacher's? Score
    KD students at several points of training next to a KD + graph student.
  - Does the graph encode what makes the teacher right? Score the same 8B
    before its fine-tune (0.46 accuracy) against the fine-tuned one (1.00):
    same architecture, neurons correspond one to one.
  - Floor: the teacher scored against itself.

    cd src && python -u -m experiments.graph_distance --teacher <ft 8B> --dataset mix3 --n 200 \\
        --model self=<ft 8B>:0.01 --model base8b=meta-llama/Meta-Llama-3-8B-Instruct:0.01 \\
        --model raw1b=meta-llama/Llama-3.2-1B-Instruct:0.03 --model kd150=<ckpt>:0.03
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DIR_ROOT, PromptAnswerDataset, collate_fn, load_data, load_model, set_bos_mode  # noqa: E402
from training.utils import load_student, student_autocast  # noqa: E402
import graph_loss.training as GT  # noqa: E402
from training.graph_kd import answer_token_cuts  # noqa: E402
from graph_loss.hf_adapter import HFLlamaGraphAdapter  # noqa: E402
from graph_loss.precompute_mlp_inputs import build_mlp_input_cache  # noqa: E402


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--dataset", default="mix3")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--nodes-per-label", type=int, default=100)
    ap.add_argument("--teacher-prop", type=float, default=0.01)
    ap.add_argument("--model", action="append", required=True, help="label=path:student_prop; repeatable")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--answer-tokens", action="store_true",
                    help="one graph per answer token, KD loss over all answer tokens, correctness by greedy decoding")
    ap.add_argument("--node-labels", nargs="+", default=None, help="ANOVA / factor supernode categories")
    ap.add_argument("--graph-rows", default=None, help="default: last (position nodes), all (labelled nodes)")
    ap.add_argument("--anova-prompts", type=int, default=3000, help="training problems in each MLP-input cache")
    args = ap.parse_args()

    set_bos_mode("on")
    dev = torch.device("cuda")
    train, test = load_data(args.dataset)
    every = sorted(test.items())
    labelled = bool(args.node_labels)
    rows = args.graph_rows or ("all" if labelled else "last")
    cache_data = dict(list(train.items())[: args.anova_prompts])
    cache_key = f"{args.dataset}_gd{args.anova_prompts}"
    items = every[::max(1, len(every) // args.n)][: args.n]
    K = args.nodes_per_label
    base = GT.GraphAuxConfig(
        graph_dtype=torch.bfloat16, teacher_prop_neurons_per_layer=args.teacher_prop,
        student_prop_neurons_per_layer=0.03, top_k_logits=0.95, temperature=1.0,
        teacher_graph_batch_size=16, student_graph_batch_size=32, student_nodes_per_label=K,
        teacher_nodes_per_label=K, graph_loss_type="jsd", graph_node_labels=args.node_labels,
        token_path_rows="weighted", dataset_name=args.dataset, graph_rows=rows, graph_edge_topk=0,
        neuron_selection="influence", supernode_type="position", position_bands=1,
        restrict_target_rows=rows in ("last", "answer"))

    teacher, tok = load_model(args.teacher)
    teacher.eval()
    for p_ in teacher.parameters():
        p_.requires_grad_(False)
    t_ad = HFLlamaGraphAdapter(teacher, tok, dev)
    if labelled:
        base = dataclasses.replace(base, teacher_mlp_input_cache=build_mlp_input_cache(
            t_ad, cache_key, args.teacher, data_dict=cache_data, batch_size=64))
    def answer_input(prompt, answer):
        """Token ids of prompt + gold answer, the first answer position, and the number of answer tokens scored."""
        b = collate_fn([PromptAnswerDataset(args.dataset, {prompt: answer}, tok)[0]], pad_id=tok.eos_token_id)
        n = len(tok(str(answer), add_special_tokens=False)["input_ids"]) if args.answer_tokens else 1
        return b["input_ids"].to(dev), int(b["response_mask"][0].nonzero()[0]), n

    cuts = [answer_token_cuts(tok, [p], [a]) if args.answer_tokens else ([p], [a]) for p, a in items]
    entries, t_logp, t0 = [], [], time.perf_counter()
    for (prompt, answer), (cut_p, cut_a) in zip(items, cuts):
        ent = []
        for cp, ca in zip(cut_p, cut_a):
            t_sg, logit_ids, dla = GT._compute_teacher_target(cp, ca, t_ad, base, dev)
            ent.append(GT._teacher_target_entry(t_sg, logit_ids, dla, answer=ca))
        entries.append(ent)
        ids, first, n = answer_input(prompt, answer)
        with torch.no_grad():
            t_logp.append(teacher(ids).logits[0, first - 1:first - 1 + n].float().log_softmax(-1).cpu())
    print(f"teacher targets for {len(entries)} problems ({sum(len(e) for e in entries)} graphs) "
          f"in {time.perf_counter() - t0:.0f}s", flush=True)
    del t_ad, teacher
    base = dataclasses.replace(base, teacher_mlp_input_cache=None)
    gc.collect()
    torch.cuda.empty_cache()

    out = {"teacher": args.teacher, "dataset": args.dataset, "n": len(items), "answer_tokens": args.answer_tokens,
           "node_labels": args.node_labels, "graph_rows": rows, "models": {}}
    for spec in args.model:
        label, rest = spec.split("=", 1)
        path, prop = rest.rsplit(":", 1)
        cfg = dataclasses.replace(base, student_prop_neurons_per_layer=float(prop))
        model, _ = load_student(path, "bfloat16")
        model.train()
        ad = HFLlamaGraphAdapter(model, tok, dev)
        if labelled:
            cfg = dataclasses.replace(cfg, mlp_input_cache=build_mlp_input_cache(
                ad, cache_key, path, data_dict=cache_data, batch_size=64))
        losses, hits, preds, kls = [], [], [], []
        t0 = time.perf_counter()
        for (prompt, answer), ent, (cut_p, cut_a), tl in zip(items, entries, cuts, t_logp):
            cut_losses = []
            for entry, cp, ca in zip(ent, cut_p, cut_a):
                t_sg, logit_ids, dla = GT._teacher_target_from_entry(entry, dev)
                loss, _ = GT.compute_prompt_graph_loss(
                    prompt=cp, student_adapter=ad, config=cfg, teacher_supergraph=t_sg,
                    logit_token_ids=logit_ids, teacher_dla_logits=dla, answer=ca)
                cut_losses.append(float(loss.detach()))
                model.zero_grad(set_to_none=True)
                del loss, t_sg
            losses.append(sum(cut_losses) / len(cut_losses))
            ids, first, n = answer_input(prompt, answer)
            with torch.no_grad(), student_autocast():
                logits = model(ids).logits[0, first - 1:first - 1 + n].float()
            tl = tl.to(dev)
            kls.append(float((tl.exp() * (tl - logits.log_softmax(-1))).sum(-1).mean()))
            if args.answer_tokens:
                # eval mode: in train mode generate() keeps no KV cache under gradient checkpointing and
                # decodes every token after the first from that token alone (Qwen: "50" became "5、")
                model.eval()
                with torch.no_grad(), student_autocast():
                    gen = model.generate(ids[:, :first], max_new_tokens=n + 3, do_sample=False,
                                         pad_token_id=tok.eos_token_id)
                model.train()
                text = tok.decode(gen[0, first:], skip_special_tokens=True)
                m_ = re.match(r"\s*(-?\d+)", text)
                preds.append(m_.group(1) if m_ else text)
                hits.append(preds[-1] == str(answer).strip())
            else:
                top = int(logits[0].argmax())
                hits.append(top == int(ids[0, first]))
                preds.append(tok.decode([top]))
        right = [l for l, h in zip(losses, hits) if h]
        wrong = [l for l, h in zip(losses, hits) if not h]
        mean = lambda v: sum(v) / len(v) if v else float("nan")  # noqa: E731
        rec = {"path": path, "student_prop": float(prop), "accuracy": mean([float(h) for h in hits]),
               "graph_loss": mean(losses), "graph_loss_right": mean(right), "graph_loss_wrong": mean(wrong),
               "auc_wrong": auc(wrong, right),
               "auc_wrong_kl": auc([k for k, h in zip(kls, hits) if not h], [k for k, h in zip(kls, hits) if h]),
               "kl": kls, "losses": losses, "hits": hits, "preds": preds,
               "prompts": [p for p, _ in items], "answers": [str(a) for _, a in items]}
        out["models"][label] = rec
        print(f"{label:10s} acc {rec['accuracy']:.3f} | graph loss {rec['graph_loss']:.4f} | right {rec['graph_loss_right']:.4f} "
              f"| wrong {rec['graph_loss_wrong']:.4f} | AUC wrong {rec['auc_wrong']:.2f} (KD loss: {rec['auc_wrong_kl']:.2f}) "
              f"| {time.perf_counter() - t0:.0f}s",
              flush=True)
        del ad, model
        gc.collect()
        torch.cuda.empty_cache()

    path = os.path.join(DIR_ROOT, f"FINAL_RESULTS/{args.dataset}/graph_distance_{args.tag}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

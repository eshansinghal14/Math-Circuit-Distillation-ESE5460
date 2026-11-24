"""Does the student's graph mismatch predict its errors? (svamp_eq / svamp_cot, arg|position + DLA graph)

A graph target is only worth training on if the student's distance from it says
something about where the student is wrong. On training prompts, with the SFT
student and SFT teacher, per prompt:

* the row-wise JSD between the teacher's and the student's supergraph at
  ``--graph-position``, by row type: the final position's row (``LAST``), the DLA
  row, the question-number rows, the question-word rows, all rows; each on every
  edge and on the teacher's top-k edges. Recomputed here from the two aligned
  matrices and checked against the trainer's own loss;
* what the ``LAST`` and DLA rows draw on, by source type (the question number the
  equation starts with, other numbers, question words, instruction, template,
  the final position itself, DLA), for teacher and student, and whether the
  most-attributed question number is the right one;
* whether the student's own greedy output gets the right final answer, and,
  teacher-forced at the graph position, whether its argmax is the gold next
  token (the prediction the graph is built for), with that token's NLL;
* prompt length in tokens, the obvious confound.

``--graph-position prompt-end`` puts the graph on the prediction of the first
response token: on svamp_eq that is the first operand (or an opening bracket),
the step the SFT student gets wrong most often. ``final-answer`` is the number
after "=" (the arithmetic) on svamp_eq and a copy step on svamp_cot.

Teacher targets come from the per-prompt cache when present and are built (and
cached) for these prompts only when not, so nothing here needs a full
precompute.

    cd src && python -m experiments.graph_error_signal --teacher <sft teacher> \\
        --student <sft student> --dataset svamp_eq --graph-position prompt-end \\
        --neuron-selection influence --supernode-type position --n 160
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (DIR_ROOT, dataset_answer_type, extract_cot_answer, extract_eq_answer,  # noqa: E402
                   load_data, load_model, set_bos_mode, split_final_answer)
from training.utils import load_student  # noqa: E402
import graph_loss.training as GT  # noqa: E402
from graph_loss.graph import occurrence_labels  # noqa: E402
from graph_loss.hf_adapter import HFLlamaGraphAdapter  # noqa: E402

NUM = re.compile(r"\d+(?:\.\d+)?")
SRC = ("this-number", "other-number", "question-word", "instruction", "template", "response", "LAST", "dla")


def norm_num(s: str) -> str:
    return str(int(float(s))) if float(s).is_integer() else str(float(s))


def auc(scores, labels):
    """P(score of a positive > score of a negative); ties count half."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    w = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return w / (len(pos) * len(neg))


def auc_se(a, n_pos, n_neg):
    """Hanley-McNeil standard error of an AUC."""
    if not (a == a) or not n_pos or not n_neg:
        return float("nan")
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return max((a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg), 0.0) ** 0.5


def spearman(a, b):
    def rank(x):
        o = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for k, i in enumerate(o):
            r[i] = float(k)
        return r
    ra, rb = torch.tensor(rank(a)), torch.tensor(rank(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp(min=1e-12))


@torch.no_grad()
def greedy_correct(model, tok, prompts, golds, grade, bs=32, max_new=64):
    tok.padding_side = "left"
    out = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        out += [grade(t) == g for t, g in zip(texts, golds[i:i + bs])]
    tok.padding_side = "right"
    return out


@torch.no_grad()
def forced_next(model, tok, gp, continuation):
    """NLL of the gold next token after ``gp`` and whether the argmax is that token."""
    pre = tok(gp, add_special_tokens=False)["input_ids"]
    full = tok(gp + continuation, add_special_tokens=False)["input_ids"]
    n = 0
    while n < min(len(pre), len(full)) and pre[n] == full[n]:
        n += 1
    ids = torch.tensor([full[: n + 1]], device="cuda")
    logp = model(ids).logits[0, n - 1].float().log_softmax(-1)
    return float(-logp[full[n]]), bool(logp.argmax() == full[n]), n


def position_types(tok, prompt: str, gp: str, this_number: str | None) -> tuple[dict[int, str], list[int]]:
    """Source type of every token position of the graph input ``gp`` (= prompt [+ response prefix])."""
    enc = tok(gp, add_special_tokens=False, return_offsets_mapping=True)
    q_start = prompt.index("<|end_header_id|>") + len("<|end_header_id|>\n\n")
    instr = prompt.index("\nWrite the equation")
    asst = prompt.index("<|eot_id|>")
    spans = [(m.start(), m.end(), norm_num(m.group())) for m in NUM.finditer(gp)]
    types = {}
    for p, (s, e) in enumerate(enc["offset_mapping"]):
        piece = gp[s:e]
        if s >= len(prompt):
            t = "response"
        elif s < q_start or s >= asst:
            t = "template"
        elif s >= instr:
            t = "instruction"
        elif any(c.isdigit() for c in piece):
            val = next((v for a, b, v in spans if s < b and e > a), None)
            t = "this-number" if (this_number is not None and val == this_number) else "other-number"
        else:
            t = "question-word"
        types[p] = t
    types[len(enc["input_ids"]) - 1] = "LAST"
    return types, enc["input_ids"]


def row_jsd(t_rows: torch.Tensor, s_rows: torch.Tensor, topk: int = 0) -> torch.Tensor:
    """Per-row JSD of the |.|-normalised rows, the trainer's edge_similarity('jsd') before its mean."""
    eps = 1e-8
    t, s = t_rows.float().abs(), s_rows.float().abs()
    if topk and topk < t.shape[1]:
        cols = t.topk(topk, dim=1).indices
        t, s = t.gather(1, cols), s.gather(1, cols)
    td = t / t.sum(1, keepdim=True).clamp(min=eps)
    sd = s / s.sum(1, keepdim=True).clamp(min=eps)
    m = 0.5 * (td + sd)
    lt, ls, lm = (td + eps).log(), (sd + eps).log(), (m + eps).log()
    return 0.5 * ((td * (lt - lm)).sum(1) + (sd * (ls - lm)).sum(1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--dataset", default="svamp_eq")
    ap.add_argument("--graph-position", choices=["prompt-end", "final-answer"], default="prompt-end")
    ap.add_argument("--neuron-selection", choices=["norm", "influence"], default="norm")
    ap.add_argument("--supernode-type", choices=["arg", "position"], default="arg")
    ap.add_argument("--position-bands", type=int, default=1)
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--teacher-prop", type=float, default=0.01)
    ap.add_argument("--student-prop", type=float, default=0.003)
    ap.add_argument("--nodes-per-label", type=int, default=10)
    ap.add_argument("--edge-topk", type=int, default=10)
    ap.add_argument("--teacher-graph-batch-size", type=int, default=32)
    ap.add_argument("--student-graph-batch-size", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    set_bos_mode("on")
    # a node present on one side only is reported per prompt with both label lists
    logging.getLogger("graph_loss.training").setLevel(logging.ERROR)
    dev = torch.device("cuda")
    tag = f"{args.graph_position}_{args.neuron_selection}_{args.supernode_type}" + (
        f"-b{args.position_bands}" if args.position_bands > 1 else "") + (
        f"_k{args.nodes_per_label}" if args.nodes_per_label != 10 else "")
    out = os.path.join(DIR_ROOT, args.out or f"FINAL_RESULTS/{args.dataset}/graph_error_signal_{tag}.json")

    train, _ = load_data(args.dataset)
    items = sorted(train.items())[: args.n]
    grade = extract_cot_answer if dataset_answer_type(args.dataset) == "cot" else extract_eq_answer

    student, tok = load_student(args.student, "bfloat16")
    student.eval()
    s_ad = HFLlamaGraphAdapter(student, tok, dev)

    config = GT.GraphAuxConfig(
        graph_dtype=torch.bfloat16, teacher_prop_neurons_per_layer=args.teacher_prop,
        student_prop_neurons_per_layer=args.student_prop, top_k_logits=0.95, temperature=1.0,
        teacher_graph_batch_size=args.teacher_graph_batch_size,
        student_graph_batch_size=args.student_graph_batch_size,
        student_nodes_per_label=args.nodes_per_label, teacher_nodes_per_label=args.nodes_per_label,
        graph_loss_type="jsd", graph_node_labels=None, token_path_rows="weighted",
        dataset_name=args.dataset, graph_rows="all", graph_edge_topk=0,
        neuron_selection=args.neuron_selection, supernode_type=args.supernode_type,
        position_bands=args.position_bands)
    key = GT.teacher_target_cache_key(config, args.teacher)
    cache = GT.TeacherTargetCache(os.path.join(DIR_ROOT, "cache/teacher_targets", f"{key}.pt"))
    print(f"teacher cache {key}: {len(cache)} prompts | {tag}", flush=True)

    prompts = [p for p, _ in items]
    golds = [grade(str(r)) for _, r in items]
    gen_ok = greedy_correct(student, tok, prompts, golds, grade)
    print(f"student greedy accuracy on these {len(items)} train prompts: {sum(gen_ok) / len(gen_ok):.3f}", flush=True)

    if args.graph_position == "final-answer":
        cut = [split_final_answer(str(r)) for _, r in items]
        graph_inputs = [(p + pre, num) for p, (pre, num) in zip(prompts, cut)]
    else:
        graph_inputs = [(p, str(r)) for p, r in items]

    teacher = t_ad = None
    if any(gp not in cache.entries for gp, _ in graph_inputs):
        teacher, _ = load_model(args.teacher)
        teacher.eval()
        for p_ in teacher.parameters():
            p_.requires_grad_(False)
        t_ad = HFLlamaGraphAdapter(teacher, tok, dev)

    grab = {}
    orig = GT.compute_graph_loss

    def spy(W_T, W_S, mapping, teacher_ids, student_ids, **kw):
        grab.update(W_T=W_T.detach().float().cpu(), W_S=W_S.detach().float().cpu(), mapping=mapping)
        return orig(W_T, W_S, mapping, teacher_ids, student_ids, **kw)
    GT.compute_graph_loss = spy

    prefix = "pos" if args.supernode_type == "position" else "arg"
    recs = []
    t_teacher = t_student = 0.0
    worst_check = 0.0
    for i, ((gp, cont), ok) in enumerate(zip(graph_inputs, gen_ok)):
        t0 = time.perf_counter()
        entry = cache.get(gp)
        if entry is None:
            t_sg, logit_ids, dla = GT._compute_teacher_target(gp, cont, t_ad, config, dev)
            cache.put(gp, GT._teacher_target_entry(t_sg, logit_ids, dla, answer=cont))
            if (i + 1) % 20 == 0:
                cache.flush()
        else:
            t_sg, logit_ids, dla = GT._teacher_target_from_entry(entry, dev)
        t1 = time.perf_counter()
        loss, _ = GT.compute_prompt_graph_loss(
            prompt=gp, student_adapter=s_ad, config=config, teacher_supergraph=t_sg,
            logit_token_ids=logit_ids, teacher_dla_logits=dla, answer=cont)
        loss = float(loss.detach())
        t_teacher += t1 - t0
        t_student += time.perf_counter() - t1

        # ---- aligned matrices in the teacher's index space, as _compute_edge_loss builds them
        W_T, W_S, mapping = grab["W_T"], grab["W_S"], grab["mapping"]
        t_labels = [l[0] if l else "" for l in t_sg.supernode_labels]
        n_T = len(t_labels)
        t2s = {t: next(iter(s)) for t, s in mapping.items() if s}
        valid = sorted(t2s)
        tv = torch.tensor(valid, dtype=torch.long)
        sv = torch.tensor([t2s[t] for t in valid], dtype=torch.long)
        S = torch.zeros(len(valid), n_T)
        S[:, tv] = W_S[sv][:, sv]
        T = W_T[tv]
        jsd_full, jsd_top = row_jsd(T, S), row_jsd(T, S, args.edge_topk)
        worst_check = max(worst_check, abs(float(jsd_full.mean()) - loss))

        # ---- node types
        first = NUM.search(str(items[i][1]))
        this_number = norm_num(first.group()) if (first and args.graph_position == "prompt-end") else None
        ptypes, ids = position_types(tok, prompts[i], gp, this_number)
        lab2pos = {l: p for p, l in enumerate(occurrence_labels(tok, ids, prefix)) if l is not None}

        def node_type(label: str) -> str:
            if label == "dla":
                return "dla"
            base = label.split("/b")[0] if args.position_bands > 1 else label
            return ptypes.get(lab2pos.get(base, -1), "template")
        col_type = [node_type(l) for l in t_labels]
        row_type = [col_type[t] for t in valid]

        nll, hit, n_tok = forced_next(student, tok, gp, cont)
        r = {"i": i, "gen_correct": bool(ok), "forced_hit": hit, "gold_nll": nll, "tokens": n_tok,
             "n_nodes_teacher": n_T, "n_nodes_aligned": len(valid), "loss": loss,
             "first_token_paren": str(items[i][1]).lstrip().startswith("("),
             "jsd_all": float(jsd_full.mean()), "jsd_all_top": float(jsd_top.mean())}
        for rt in ("LAST", "dla", "this-number", "other-number", "question-word"):
            sel = [k for k, x in enumerate(row_type) if x == rt]
            r[f"jsd_{rt}"] = float(jsd_full[sel].mean()) if sel else None
            r[f"jsd_{rt}_top"] = float(jsd_top[sel].mean()) if sel else None
        # ---- the LAST and DLA rows on a subset of source columns (renormalised over it):
        # the question numbers only, and the question (numbers + words) only
        col_sets = {"numcols": [c for c, ct in enumerate(col_type) if ct in ("this-number", "other-number")],
                    "qcols": [c for c, ct in enumerate(col_type) if ct in ("this-number", "other-number", "question-word")]}
        for rt in ("LAST", "dla"):
            sel = [k for k, x in enumerate(row_type) if x == rt]
            for cname, cols in col_sets.items():
                r[f"jsd_{rt}_{cname}"] = (float(row_jsd(T[sel][:, cols], S[sel][:, cols]).mean())
                                          if sel and len(cols) >= 2 else None)
        # ---- what the LAST and DLA rows draw on
        for rt in ("LAST", "dla"):
            sel = [k for k, x in enumerate(row_type) if x == rt]
            if not sel:
                continue
            for name, M in (("teacher", T), ("student", S)):
                d = M[sel].abs().sum(0)
                d = d / d.sum().clamp(min=1e-12)
                mass = defaultdict(float)
                for w, ct in zip(d.tolist(), col_type):
                    mass[ct] += w
                r[f"{name}_{rt}_mass"] = {c: mass.get(c, 0.0) for c in SRC}
                per_num = defaultdict(float)
                for w, ct, lab in zip(d.tolist(), col_type, t_labels):
                    if ct in ("this-number", "other-number"):
                        per_num[(ct, lab.split("#")[0].split("/b")[0])] += w
                if any(k[0] == "this-number" for k in per_num) and len(per_num) > 1:
                    r[f"{name}_{rt}_top_number_right"] = max(per_num, key=per_num.get)[0] == "this-number"
            r[f"chance_this_{rt}"] = sum(c == "this-number" for c in col_type) / max(1, n_T)
        recs.append(r)
        print(f"  [{i}] ok={int(ok)} hit={int(hit)} nll={nll:.3f} nodes={len(valid)}/{n_T} all={r['jsd_all']:.4f} "
              f"LAST={r['jsd_LAST'] if r['jsd_LAST'] is not None else float('nan'):.4f} "
              f"dla={r['jsd_dla'] if r['jsd_dla'] is not None else float('nan'):.4f} | teacher {t1 - t0:.1f}s", flush=True)
        torch.cuda.empty_cache()
    GT.compute_graph_loss = orig
    cache.flush()

    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    n_wrong = sum(not r["gen_correct"] for r in recs)
    n_miss = sum(not r["forced_hit"] for r in recs)
    summary = {"n": len(recs), "gen_acc": 1 - n_wrong / len(recs), "forced_acc": 1 - n_miss / len(recs),
               "teacher_seconds": t_teacher, "student_seconds": t_student, "loss_recompute_max_abs_diff": worst_check,
               "nodes_teacher": mean([r["n_nodes_teacher"] for r in recs]),
               "nodes_aligned": mean([r["n_nodes_aligned"] for r in recs])}
    print(f"\n{tag}: n={len(recs)}  greedy acc {summary['gen_acc']:.3f}  forced next-token acc {summary['forced_acc']:.3f}  "
          f"| nodes {summary['nodes_aligned']:.1f} aligned of {summary['nodes_teacher']:.1f} | teacher {t_teacher:.0f}s "
          f"student {t_student:.0f}s | recomputed all-rows JSD vs trainer loss: max |diff| {worst_check:.4f}")

    print("\n== does the teacher-student row distance predict the student's errors? ==")
    print(f"AUC = P(measure on an error > on a non-error); 0.5 = none. greedy: final answer wrong ({n_wrong}); "
          f"forced: argmax misses the gold token at the graph position ({n_miss})")
    print(f"{'rows':22s} {'n':>4s} {'AUC greedy':>10s} {'se':>6s} {'AUC forced':>10s} {'se':>6s} {'rho nll':>8s} "
          f"{'rho len':>8s} {'mean ok':>8s} {'mean err':>8s}")
    measures = ["jsd_all", "jsd_all_top"]
    for rt in ("LAST", "dla", "this-number", "other-number", "question-word"):
        measures += [f"jsd_{rt}", f"jsd_{rt}_top"]
    measures += [f"jsd_{rt}_{c}" for rt in ("LAST", "dla") for c in ("numcols", "qcols")]
    measures.append("tokens")
    for k in measures:
        sub = [r for r in recs if r.get(k) is not None]
        if len(sub) < 10:
            continue
        v = [float(r[k]) for r in sub]
        wrong = [not r["gen_correct"] for r in sub]
        miss = [not r["forced_hit"] for r in sub]
        s = {"n": len(sub), "auc_greedy": auc(v, wrong), "auc_forced": auc(v, miss),
             "rho_nll": spearman(v, [r["gold_nll"] for r in sub]), "rho_len": spearman(v, [r["tokens"] for r in sub]),
             "mean_ok": mean([x for x, w in zip(v, wrong) if not w]), "mean_err": mean([x for x, w in zip(v, wrong) if w])}
        s["se_greedy"] = auc_se(s["auc_greedy"], wrong.count(True), wrong.count(False))
        s["se_forced"] = auc_se(s["auc_forced"], miss.count(True), miss.count(False))
        summary[k] = s
        print(f"{k:22s} {s['n']:4d} {s['auc_greedy']:10.3f} {s['se_greedy']:6.3f} {s['auc_forced']:10.3f} "
              f"{s['se_forced']:6.3f} {s['rho_nll']:8.3f} {s['rho_len']:8.3f} {s['mean_ok']:8.4f} {s['mean_err']:8.4f}")

    for rt in ("LAST", "dla"):
        sub = [r for r in recs if f"teacher_{rt}_mass" in r]
        if not sub:
            continue
        print(f"\n== what the {rt} row draws on (mean share of the row's edge mass) ==")
        print(f"{'':22s} " + " ".join(f"{c[:13]:>13s}" for c in SRC))
        groups = {"teacher": ("teacher", sub), "student": ("student", sub),
                  "student | forced ok": ("student", [r for r in sub if r["forced_hit"]]),
                  "student | forced miss": ("student", [r for r in sub if not r["forced_hit"]]),
                  "teacher | S forced miss": ("teacher", [r for r in sub if not r["forced_hit"]])}
        summary[f"{rt}_mass"] = {}
        for gname, (name, rs) in groups.items():
            row = {c: mean([r[f"{name}_{rt}_mass"][c] for r in rs]) for c in SRC}
            summary[f"{rt}_mass"][gname] = row
            print(f"{gname:22s} " + " ".join(f"{row[c]:13.3f}" for c in SRC))
        has = [r for r in sub if r.get(f"teacher_{rt}_top_number_right") is not None]
        if has:
            miss = [not r["forced_hit"] for r in has]
            wrong = [not r["gen_correct"] for r in has]
            tr = [r[f"teacher_{rt}_top_number_right"] for r in has]
            sr = [r[f"student_{rt}_top_number_right"] for r in has]
            s_this = [r[f"student_{rt}_mass"]["this-number"] for r in has]
            t_this = [r[f"teacher_{rt}_mass"]["this-number"] for r in has]
            s = {"n": len(has), "chance_this": mean([r[f"chance_this_{rt}"] for r in has]),
                 "teacher_this": mean(t_this), "student_this": mean(s_this),
                 "teacher_top1": mean(tr), "student_top1": mean(sr),
                 "student_top1_forced_ok": mean([x for x, m in zip(sr, miss) if not m]),
                 "student_top1_forced_miss": mean([x for x, m in zip(sr, miss) if m]),
                 "auc_low_student_this_vs_forced": auc([-x for x in s_this], miss),
                 "auc_low_student_this_vs_greedy": auc([-x for x in s_this], wrong),
                 "auc_T_minus_S_this_vs_forced": auc([a - b for a, b in zip(t_this, s_this)], miss),
                 "auc_T_minus_S_this_vs_greedy": auc([a - b for a, b in zip(t_this, s_this)], wrong)}
            summary[f"{rt}_numbers"] = s
            print(f"right question number (n={s['n']}, chance share {s['chance_this']:.3f}): mass T {s['teacher_this']:.3f} "
                  f"S {s['student_this']:.3f} | top-attributed number is right: T {s['teacher_top1']:.3f} S {s['student_top1']:.3f} "
                  f"(S when forced ok {s['student_top1_forced_ok']:.3f}, when forced miss {s['student_top1_forced_miss']:.3f})")
            print(f"AUC low student mass on the right number: vs forced miss {s['auc_low_student_this_vs_forced']:.3f}, "
                  f"vs greedy wrong {s['auc_low_student_this_vs_greedy']:.3f} | teacher-minus-student mass: "
                  f"{s['auc_T_minus_S_this_vs_forced']:.3f}, {s['auc_T_minus_S_this_vs_greedy']:.3f}")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"args": vars(args), "summary": summary, "records": recs}, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

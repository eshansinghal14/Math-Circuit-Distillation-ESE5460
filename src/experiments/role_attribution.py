"""Where does each model look when it writes each token of the equation? (svamp_eq / svamp_cot)

The graph term was built at the final answer, a step the student never gets wrong.
The computation is in the equation: which question number goes first, which
operation, which number goes second, then the arithmetic. This measures, with no
neuron graph at all, whether a target built at those positions would carry signal.

Per training prompt, teacher-forced on prompt + gold equation, for teacher and
student, at every response token j of the equation (operands, operators, "=",
result): the input attribution of the gold token's log-probability,
``|d logp / d e_p . e_p|`` over input positions p < j (BOS excluded), normalised
to a distribution. Input positions are grouped by role:

* ``this-number``    the question number being written (operand positions only)
* ``other-operand``  question numbers the equation uses elsewhere
* ``distractor``     question numbers the equation never uses
* ``question-word``, ``instruction``, ``template``
* ``resp-number``, ``resp-other``  earlier tokens of the equation

Questions answered:

1. Does the teacher's attribution name the computation: at an operand position,
   how much of its mass is on the right question number (vs the chance share by
   token count), and is the top-attributed question number the right one?
2. Is the student different from the teacher there, and at which position type?
3. Does the difference predict the student's errors: AUC of the per-prompt
   teacher-student divergence (role-level and token-level JSD) and of the
   student's own mass on the right number, against (a) the student's greedy
   answer being wrong and (b) its teacher-forced argmax missing the gold token at
   that position. Prompt length is reported as the confound.

Also classifies the student's greedy errors (wrong numbers / wrong operation or
order / wrong arithmetic), i.e. where in the equation it actually goes wrong.

    cd src && python -m experiments.role_attribution --dataset svamp_eq \\
        --teacher <sft teacher> --student <sft student> --n 160
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (DIR_ROOT, dataset_answer_type, extract_cot_answer, extract_eq_answer,  # noqa: E402
                   load_data, load_model, set_bos_mode)

NUM = re.compile(r"\d+(?:\.\d+)?")
ROLES = ("this-number", "other-operand", "distractor", "question-word", "instruction", "template",
         "resp-number", "resp-other")
POS_TYPES = ("operand1", "operand2+", "operator", "equals", "result")


def norm_num(s: str) -> str:
    return str(int(float(s))) if float(s).is_integer() else str(float(s))


def auc(scores, labels):
    """P(score of a positive > score of a negative); ties count half."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def jsd(p: torch.Tensor, q: torch.Tensor) -> float:
    p, q = p.double().clamp(min=0), q.double().clamp(min=0)
    p, q = p / p.sum().clamp(min=1e-12), q / q.sum().clamp(min=1e-12)
    m = 0.5 * (p + q)

    def kl(a, b):
        nz = a > 0
        return float((a[nz] * (a[nz] / b[nz]).log()).sum())
    return 0.5 * (kl(p, m) + kl(q, m))


def layout(tok, prompt: str, equation: str):
    """Token ids of prompt + equation with, per token, a base role and number value.

    Returns (ids, info, targets): info[p] = (base_role, number_or_None) for every
    input position; targets = [(j, pos_type, number_or_None)] for the equation's
    response tokens (j indexes ids; the token at j is predicted from position j-1).
    """
    text = prompt + equation
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    q_start = prompt.index("<|end_header_id|>") + len("<|end_header_id|>\n\n")
    instr = prompt.index("\nWrite the equation")
    asst = prompt.index("<|eot_id|>")
    P = len(prompt)

    spans = [(m.start(), m.end(), norm_num(m.group())) for m in NUM.finditer(text)]

    def number_at(s, e):
        for a, b, v in spans:
            if s < b and e > a:
                return a, v
        return None, None

    eq_nums = [v for a, b, v in spans if a >= P]
    result_start = max(a for a, b, v in spans if a >= P)
    eq_operands = {v for a, b, v in spans if P <= a < result_start}
    info, targets, seen_starts = [], [], set()
    n_operand = 0
    for j, ((s, e), tid) in enumerate(zip(offs, ids)):
        piece = text[s:e]
        start, val = number_at(s, e) if any(c.isdigit() for c in piece) else (None, None)
        if s >= P:
            info.append(("resp-number" if val is not None else "resp-other", val))
            if val is not None:
                first = start not in seen_starts
                seen_starts.add(start)
                if start == result_start:
                    kind = "result"
                else:
                    if first:
                        n_operand += 1
                    kind = "operand1" if n_operand == 1 else "operand2+"
                # only the first token of a number is a decision; later pieces are continuation
                if first:
                    targets.append((j, kind, val))
            elif "=" in piece:
                targets.append((j, "equals", None))
            elif any(c in piece for c in "+-*/"):
                targets.append((j, "operator", None))
        elif tid == tok.bos_token_id or s < q_start or s >= asst:
            info.append(("template", None))
        elif s >= instr:
            info.append(("instruction", None))
        elif val is not None:
            info.append(("other-operand" if val in eq_operands else "distractor", val))
        else:
            info.append(("question-word", None))
    return ids, info, targets, eq_nums


def attribute(model, ids: list[int], target_js: list[int]):
    """[len(target_js), T] |grad x input| of the gold token's log-prob, plus forced hit / nll."""
    dev = next(model.parameters()).device
    R = len(target_js)
    x = torch.tensor([ids], device=dev).expand(R, -1)
    emb = model.get_input_embeddings()(x).detach().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(inputs_embeds=emb, use_cache=False).logits
    rows = torch.arange(R, device=dev)
    js = torch.tensor(target_js, device=dev)
    lp = logits[rows, js - 1].float().log_softmax(-1)
    gold = torch.tensor([ids[j] for j in target_js], device=dev)
    picked = lp[rows, gold]
    (g,) = torch.autograd.grad(picked.sum(), emb)
    attr = (g.float() * emb.detach().float()).sum(-1).abs()  # [R, T]
    hit = (lp.argmax(-1) == gold).tolist()
    return attr.cpu(), hit, (-picked.detach()).tolist()


@torch.no_grad()
def greedy(model, tok, prompts, bs=32, max_new=32):
    tok.padding_side = "left"
    out = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    tok.padding_side = "right"
    return out


def error_kind(gen: str, gold_eq: str) -> str:
    """Why a wrong greedy answer is wrong, from its first line."""
    line = (gen.strip().splitlines() or [""])[0]
    if "=" not in line:
        return "no-equation"
    g_lhs, w_lhs = gold_eq.split("=")[0], line.split("=")[0]
    g_nums, w_nums = sorted(norm_num(x) for x in NUM.findall(g_lhs)), sorted(norm_num(x) for x in NUM.findall(w_lhs))
    if g_nums != w_nums:
        return "wrong-numbers"
    ops = lambda s: re.findall(r"[+\-*/]", s)
    if ops(g_lhs) != ops(w_lhs) or [norm_num(x) for x in NUM.findall(g_lhs)] != [norm_num(x) for x in NUM.findall(w_lhs)]:
        return "wrong-operation-or-order"
    return "wrong-arithmetic"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--dataset", default="svamp_eq")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    set_bos_mode("on")
    out_path = os.path.join(DIR_ROOT, args.out or f"FINAL_RESULTS/{args.dataset}/role_attribution_{args.split}.json")

    train, test = load_data(args.dataset)
    data = train if args.split == "train" else test
    items = sorted(data.items()) if isinstance(data, dict) else sorted(
        (s["formatted_prompt"], s["answer"]) for s in data.samples)
    items = items[: args.n]
    grade = extract_cot_answer if dataset_answer_type(args.dataset) == "cot" else extract_eq_answer

    student, tok = load_model(args.student)
    teacher, _ = load_model(args.teacher)
    for m in (student, teacher):
        m.eval()
        for p_ in m.parameters():
            p_.requires_grad_(False)

    prompts = [p for p, _ in items]
    gold_eqs = [str(r).splitlines()[0] for _, r in items]
    golds = [grade(str(r)) for _, r in items]
    gens = {"student": greedy(student, tok, prompts), "teacher": greedy(teacher, tok, prompts)}
    ok = {k: [grade(t) == g for t, g in zip(v, golds)] for k, v in gens.items()}
    for k in ok:
        print(f"{k} greedy accuracy on {len(items)} {args.split} prompts: {sum(ok[k]) / len(items):.3f}")
    kinds = Counter(error_kind(t, e) for t, e, o in zip(gens["student"], gold_eqs, ok["student"]) if not o)
    print("student greedy errors by kind:", dict(kinds), flush=True)

    recs = []
    for i, (prompt, eq) in enumerate(zip(prompts, gold_eqs)):
        ids, info, targets, _ = layout(tok, prompt, eq)
        if not targets:
            continue
        js = [j for j, _, _ in targets]
        a_t, hit_t, nll_t = attribute(teacher, ids, js)
        a_s, hit_s, nll_s = attribute(student, ids, js)
        for r, (j, kind, val) in enumerate(targets):
            src = list(range(1, j))  # BOS excluded, strictly earlier tokens
            roles = []
            for p in src:
                base, v = info[p]
                roles.append("this-number" if (val is not None and kind != "result" and base in ("other-operand", "distractor")
                                               and v == val) else base)
            rec = {"i": i, "j": j, "kind": kind, "n_src": len(src), "student_wrong": not ok["student"][i],
                   "teacher_wrong": not ok["teacher"][i], "student_hit": bool(hit_s[r]), "teacher_hit": bool(hit_t[r]),
                   "student_nll": nll_s[r], "teacher_nll": nll_t[r]}
            dists = {}
            for name, a in (("teacher", a_t), ("student", a_s)):
                d = a[r, src]
                d = d / d.sum().clamp(min=1e-12)
                dists[name] = d
                mass = defaultdict(float)
                for w, ro in zip(d.tolist(), roles):
                    mass[ro] += w
                rec[f"{name}_mass"] = {ro: mass.get(ro, 0.0) for ro in ROLES}
                # per question-number totals: is the most-attributed number the right one?
                per_num = defaultdict(float)
                for w, p in zip(d.tolist(), src):
                    base, v = info[p]
                    if base in ("other-operand", "distractor"):
                        per_num[v] += w
                if kind.startswith("operand") and val in per_num and len(per_num) > 1:
                    rec[f"{name}_top_number_right"] = max(per_num, key=per_num.get) == val
            n_this = sum(ro == "this-number" for ro in roles)
            rec["chance_this"] = n_this / len(src) if n_this else None
            rec["n_distractor_tok"] = sum(ro == "distractor" for ro in roles)
            rec["jsd_token"] = jsd(dists["teacher"], dists["student"])
            rec["jsd_role"] = jsd(torch.tensor([rec["teacher_mass"][ro] for ro in ROLES]),
                                  torch.tensor([rec["student_mass"][ro] for ro in ROLES]))
            recs.append(rec)
        if i % 20 == 0:
            print(f"  [{i}] {eq!r}: {len(targets)} positions", flush=True)
        del a_t, a_s
        torch.cuda.empty_cache()

    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    summary = {"n_prompts": len(items), "greedy_acc": {k: sum(v) / len(v) for k, v in ok.items()},
               "student_error_kinds": dict(kinds), "by_kind": {}}
    print("\n== mean attribution mass by role (T = teacher, S = student) ==")
    print(f"{'position':10s} {'n':>4s} " + " ".join(f"{ro[:13]:>15s}" for ro in ROLES))
    for kind in POS_TYPES:
        rs = [r for r in recs if r["kind"] == kind]
        if not rs:
            continue
        row = {"n": len(rs)}
        for name in ("teacher", "student"):
            row[f"{name}_mass"] = {ro: mean([r[f"{name}_mass"][ro] for r in rs]) for ro in ROLES}
            print(f"{kind:10s} {len(rs):4d} " + " ".join(f"{name[0].upper()} {row[f'{name}_mass'][ro]:13.3f}" for ro in ROLES))
        summary["by_kind"][kind] = row

    print("\n== operand positions: is the right question number found? ==")
    print(f"{'position':10s} {'n':>4s} {'chance':>7s} {'T mass':>7s} {'S mass':>7s} {'T top1':>7s} {'S top1':>7s} "
          f"{'T forced':>8s} {'S forced':>8s}")
    for kind in ("operand1", "operand2+"):
        rs = [r for r in recs if r["kind"] == kind and r["chance_this"]]
        if not rs:
            continue
        top = [r for r in rs if "teacher_top_number_right" in r]
        row = summary["by_kind"][kind]
        row.update(chance_this=mean([r["chance_this"] for r in rs]),
                   teacher_this=mean([r["teacher_mass"]["this-number"] for r in rs]),
                   student_this=mean([r["student_mass"]["this-number"] for r in rs]),
                   teacher_top1=mean([r["teacher_top_number_right"] for r in top]),
                   student_top1=mean([r["student_top_number_right"] for r in top]), n_top=len(top))
        print(f"{kind:10s} {len(rs):4d} {row['chance_this']:7.3f} {row['teacher_this']:7.3f} {row['student_this']:7.3f} "
              f"{row['teacher_top1']:7.3f} {row['student_top1']:7.3f} "
              f"{mean([r['teacher_hit'] for r in rs]):8.3f} {mean([r['student_hit'] for r in rs]):8.3f}")

    print("\n== does the teacher-student difference predict the student's errors? ==")
    print("AUC = P(measure on an error > on a non-error); 0.5 = none. 'greedy' = final answer wrong, "
          "'forced' = argmax misses the gold token at this position")
    print(f"{'position':10s} {'measure':22s} {'n':>4s} {'forced acc':>10s} {'AUC greedy':>10s} {'AUC forced':>10s} "
          f"{'mean ok':>8s} {'mean err':>8s}")
    for kind in POS_TYPES:
        rs = [r for r in recs if r["kind"] == kind]
        if not rs:
            continue
        measures = {"jsd_token": lambda r: r["jsd_token"], "jsd_role": lambda r: r["jsd_role"],
                    "n_src (length)": lambda r: float(r["n_src"]),
                    "n_distractor_tok": lambda r: float(r["n_distractor_tok"])}
        if kind.startswith("operand"):
            measures["-student this-number"] = lambda r: -r["student_mass"]["this-number"]
            measures["T-S this-number"] = lambda r: r["teacher_mass"]["this-number"] - r["student_mass"]["this-number"]
        for name, fn in measures.items():
            sub = [r for r in rs if r["chance_this"]] if "this-number" in name else rs
            v = [fn(r) for r in sub]
            wrong = [r["student_wrong"] for r in sub]
            miss = [not r["student_hit"] for r in sub]
            s = {"n": len(sub), "forced_acc": 1 - mean(miss), "auc_greedy": auc(v, wrong), "auc_forced": auc(v, miss),
                 "mean_ok": mean([x for x, w in zip(v, wrong) if not w]), "mean_err": mean([x for x, w in zip(v, wrong) if w])}
            summary["by_kind"][kind][name] = s
            print(f"{kind:10s} {name:22s} {s['n']:4d} {s['forced_acc']:10.3f} {s['auc_greedy']:10.3f} "
                  f"{s['auc_forced']:10.3f} {s['mean_ok']:8.4f} {s['mean_err']:8.4f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "summary": summary, "records": recs,
                   "student_generations": gens["student"], "gold_equations": gold_eqs}, f, indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

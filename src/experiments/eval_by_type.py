"""Accuracy by problem type on svamp_eq, for the teacher and the student(s).

The gold response is ``<expression> = N``, so the problem type is read off the gold
expression: one-step problems by their operator (add / sub / mul / div), multi-step
problems (two or more operators) split into add/sub-only and those with a mul or
div. Decoding and grading are eval_model's (greedy, EQ_EVAL_TOKENS, the number after
the first "="), so the test-split total must reproduce the baselines the trainer
prints.

Why: to see whether a split that trains on some problem types and tests on others
is workable here. That needs a teacher that is strong on the held-out types and a
student that is not.

Also reported, from the generated text: whether the expression itself evaluates to
the gold answer (reasoning right, whatever result was written after "="), and on
multi-step problems how often the model writes fewer operations than the gold
expression has.

    cd src && python -u -m experiments.eval_by_type \\
        --model teacher=<sft teacher> --model student=<sft student> \\
        --model "kd+graph=<trained student>" --test-only kd+graph
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DIR_ROOT, default_eval_tokens, extract_eq_answer, load_model, load_split, set_bos_mode  # noqa: E402

OPS = re.compile(r"[+\-*/]")
SAFE = re.compile(r"^[\d\s.+\-*/()]+$")
ONE = {"+": "one-step add", "-": "one-step sub", "*": "one-step mul", "/": "one-step div"}
ORDER = ["one-step add", "one-step sub", "one-step mul", "one-step div", "multi-step add/sub",
         "multi-step with mul/div", "ALL one-step", "ALL multi-step", "ALL"]


def expr_of(text: str) -> str:
    return str(text).split("=")[0].strip()


def category(expr: str) -> str:
    ops = OPS.findall(expr)
    if len(ops) == 1:
        return ONE[ops[0]]
    if not ops:
        return "no operator"
    return "multi-step with mul/div" if any(o in "*/" for o in ops) else "multi-step add/sub"


def value_of(expr: str):
    if not expr or not SAFE.match(expr):
        return None
    try:
        return eval(expr, {"__builtins__": {}}, {})  # digits, + - * / ( ) only
    except Exception:
        return None


@torch.no_grad()
def generate(model, tok, prompts: list[str], bs: int, max_new: int) -> list[str]:
    model.eval()
    side = tok.padding_side
    tok.padding_side = "left"
    bos = tok.bos_token or ""
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
    out = [""] * len(prompts)
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        batch = [prompts[i] for i in idx]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  add_special_tokens=not (bos and batch[0].startswith(bos))).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, t in zip(idx, tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            out[i] = t
    tok.padding_side = side
    return out


def acc(rows: list[dict], key: str) -> tuple[float, float, int]:
    n = len(rows)
    if not n:
        return float("nan"), float("nan"), 0
    p = sum(bool(r[key]) for r in rows) / n
    return p, (p * (1 - p) / n) ** 0.5, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", action="append", required=True, help="label=path; repeat for each model")
    ap.add_argument("--test-only", nargs="*", default=[], help="labels trained on the train split: scored on test only")
    ap.add_argument("--dataset", default="svamp_eq")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    set_bos_mode("on")
    max_new = default_eval_tokens(args.dataset)
    out_path = os.path.join(DIR_ROOT, args.out or f"FINAL_RESULTS/{args.dataset}/eval_by_type.json")

    items = []
    for split in ("train", "test"):
        for prompt, gold in load_split(args.dataset, split).items():
            expr = expr_of(gold)
            items.append({"split": split, "prompt": prompt, "gold": str(gold), "gold_answer": extract_eq_answer(str(gold)),
                          "category": category(expr), "n_ops": len(OPS.findall(expr)), "paren": expr.startswith("(")})
    print(f"{args.dataset}: {sum(i['split'] == 'train' for i in items)} train + {sum(i['split'] == 'test' for i in items)} test "
          f"problems, {max_new} new tokens", flush=True)

    labels = []
    for spec in args.model:
        label, path = spec.split("=", 1)
        labels.append(label)
        use = [i for i in items if i["split"] == "test"] if label in args.test_only else items
        model, tok = load_model(path)
        texts = generate(model, tok, [i["prompt"] for i in use], args.batch_size, max_new)
        for it, text in zip(use, texts):
            pred = extract_eq_answer(text)
            pexpr = expr_of(text)
            val = value_of(pexpr)
            it[label] = {"text": text, "pred": pred,
                         "correct": pred is not None and pred == it["gold_answer"],
                         "expr_correct": val is not None and abs(val - it["gold_answer"]) < 1e-6,
                         "n_ops": len(OPS.findall(pexpr)), "wrote_equation": "=" in text}
        test = [i for i in use if i["split"] == "test"]
        print(f"{label}: test accuracy {sum(i[label]['correct'] for i in test) / len(test):.4f} "
              f"({len(test)} problems) <- must match the trainer's baseline", flush=True)
        del model
        torch.cuda.empty_cache()

    def table(title: str, rows: list[dict], cols: list[str], key: str) -> dict:
        print(f"\n== {title} ==")
        print(f"{'':26s} {'n':>4s} " + " ".join(f"{c:>16s}" for c in cols))
        res = {}
        groups = {c: [r for r in rows if r["category"] == c] for c in ORDER[:6]}
        groups["ALL one-step"] = [r for r in rows if r["n_ops"] == 1]
        groups["ALL multi-step"] = [r for r in rows if r["n_ops"] >= 2]
        groups["ALL"] = rows
        for g in ORDER:
            rs = groups[g]
            if not rs:
                continue
            cells = []
            res[g] = {"n": len(rs)}
            for c in cols:
                p, se, _ = acc([{key: r[c][key]} for r in rs], key)
                res[g][c] = p
                cells.append(f"{p:8.3f} +-{se:5.3f}")
            print(f"{g:26s} {len(rs):4d} " + " ".join(f"{x:>16s}" for x in cells))
        return res

    summary = {}
    both = [l for l in labels if l not in args.test_only]
    if both:
        summary["train+test"] = table("answer accuracy, train + test (neither split was seen in the format SFT)", items, both, "correct")
        summary["train+test expression"] = table("the written expression evaluates to the gold answer, train + test", items, both, "expr_correct")
    test_items = [i for i in items if i["split"] == "test"]
    summary["test"] = table("answer accuracy, test split only", test_items, labels, "correct")

    print("\n== multi-step problems: how the response goes wrong ==")
    for label in labels:
        rs = [i for i in items if i["n_ops"] >= 2 and label in i]
        if not rs:
            continue
        fewer = sum(i[label]["n_ops"] < i["n_ops"] for i in rs) / len(rs)
        wrong = [i for i in rs if not i[label]["correct"]]
        fewer_wrong = sum(i[label]["n_ops"] < i["n_ops"] for i in wrong) / max(len(wrong), 1)
        summary[f"multi_step_{label}"] = {"n": len(rs), "fewer_ops": fewer, "n_wrong": len(wrong), "fewer_ops_among_wrong": fewer_wrong}
        print(f"  {label:12s} n={len(rs):3d}: writes fewer operations than the gold expression {fewer:.3f} "
              f"| among its {len(wrong)} wrong answers {fewer_wrong:.3f}")
    counts = {}
    for i in items:
        counts[i["category"]] = counts.get(i["category"], 0) + 1
    summary["counts"] = counts
    summary["paren_start"] = sum(i["paren"] for i in items)
    print(f"\ncounts by type (train + test): {counts} | gold starts with '(': {summary['paren_start']}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "summary": summary, "records": items}, f, indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

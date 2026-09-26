"""Behavioural case studies: what each model outputs on each eval set, and where it breaks.

Loads any subset of teacher / base student / SFT / standard-KD / graph-KD one at a
time and decodes every eval set through ``utils.eval_model`` itself:
``model.generate`` is wrapped only to record each batch, rows are re-aligned by
eval_model's own longest-first order (checked against the decoded inputs), and the
per-prompt grading must reproduce eval_model's accuracy exactly, so the numbers are
the training logs' when ``--bos-mode`` / ``--max-eval-tokens`` / ``--test-limit``
match the run (the recorded runs: legacy, 5, none). A second, teacher-forced
forward gives the next-token distribution at the answer position with the same
BOS rule as eval; ``p_gold`` is the probability of the gold answer's first token.

Outputs under ``--out-dir``:

  * ``error_analysis.md``: per set, accuracy split by carry count (columns that
    carry out, 0 / 1 / 2+), whether the answer gains a digit over the widest
    operand (sum >= 100 on 22_add), and single-digit operands; then the wrong
    answers by ERRORS, first match wins: no leading integer (``format``), an
    operand echoed, off by exactly 10^k (k >= 1) below / above gold (a dropped /
    extra carry), the gold digits permuted, units wrong, units right but tens
    wrong, units and tens right but a higher digit wrong.
  * ``pairs.md`` / ``pairs.json``: FAMILIES of counterfactual minimal pairs -- one
    digit changed so a carry appears, or the same sum restated in a held-out
    format -- with each model's answer and p_gold on both sides, how often it
    gets the base right and the variant wrong (``breaks``), and how often its
    right/wrong pattern on the pair equals the teacher's.
  * ``examples.md`` / ``examples.json``: per held-out set, prompts the teacher and
    graph KD get right while every loaded SFT / standard-KD student is wrong
    (``--n-examples``), and prompts graph KD gets wrong while one of them is right
    (``--n-reverse``), spread over carry counts, most confident first, with every
    model's output string and top-5 next tokens.
  * ``summary.json``: every number in the markdown. ``predictions/<role>.json``:
    the per-prompt records. With ``--reuse`` a role whose file already covers the
    requested prompts under the same settings is not reloaded, so the models can
    run in separate sessions and the tables be rebuilt with no GPU.

Usage (from the repository root, GPU):
    PYTHONPATH=src python -m experiments.case_studies --teacher meta-llama/Meta-Llama-3-8B-Instruct \\
        --base meta-llama/Llama-3.2-1B-Instruct --sft <dir> --kd <dir> --graph <dir>
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import random
import re
import sys
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (BOS_MODES, DIR_ROOT, PromptAnswerDataset, bos_in_eval, eval_model,  # noqa: E402
                   extract_leading_int, load_data, load_model, set_bos_mode)
from training.utils import load_student, student_autocast  # noqa: E402
from experiments.inspect_graphs import attributes, parse_prompt  # noqa: E402
from graph_loss.static_figures import true_answer  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROLES = ["teacher", "base", "sft", "kd", "graph"]
LABELS = {"teacher": "Teacher 8B", "base": "Base 1B", "sft": "SFT", "kd": "Standard KD", "graph": "Graph KD"}
DATASETS = ["22_add", "222_add", "2222_add", "33_add", "21_mult"]
PAIR_SET = "pairs"
ERRORS = ["format", "copy_operand", "carry_dropped", "carry_extra", "transposition",
          "units_wrong", "tens_wrong", "high_wrong"]
STRATA = [
    ("all", lambda s: True),
    ("0 carries", lambda s: s.get("carries") == 0),
    ("1 carry", lambda s: s.get("carries") == 1),
    ("2+ carries", lambda s: (s.get("carries") or 0) >= 2),
    ("new digit", lambda s: s.get("new_digit") is True),
    ("no new digit", lambda s: s.get("new_digit") is False),
    ("1-digit operand", lambda s: s.get("single_digit") is True),
]


# ---------------------------------------------------------------------------
# Prompt structure and error taxonomy
# ---------------------------------------------------------------------------

def operands(prompt: str) -> tuple[str, list[int]] | None:
    """('+', args) for an N-ary addition (inspect_graphs.parse_prompt), ('*', [a, b]) for 'a*b='."""
    args = parse_prompt(prompt)
    if args is not None:
        return "+", args
    m = re.fullmatch(r"\s*(\d+)\s*\*\s*(\d+)\s*=\s*", prompt)
    return ("*", [int(m.group(1)), int(m.group(2))]) if m else None


def carries(op: str, args: list[int]) -> int | None:
    """Columns that carry out, units first; the last column's carry is the one that adds a digit.

    Multiplication is counted for a single-digit multiplier only (the 21_mult
    format), as the carries of the partial products d * a_k; None otherwise.
    """
    if op == "*":
        if min(args) >= 10:
            return None
        totals = [int(c) * min(args) for c in reversed(str(max(args)))]
    else:
        totals = [sum(a // 10 ** k % 10 for a in args) for k in range(max(len(str(a)) for a in args))]
    n = c = 0
    for t in totals:
        c = (t + c) // 10
        n += c > 0
    return n


def structure(prompt: str, gold: int) -> dict[str, Any]:
    parsed = operands(prompt)
    if parsed is None:
        return {}
    op, args = parsed
    out = {"op": op, "operands": args, "carries": carries(op, args),
           "new_digit": len(str(gold)) > max(len(str(a)) for a in args)}
    out.update(attributes(prompt))
    return out


def error_type(pred: int | None, gold: int, args: list[int]) -> str | None:
    """None when right, else the first ERRORS category that fits."""
    if pred == gold:
        return None
    if pred is None:
        return "format"
    if pred in args:
        return "copy_operand"
    diff = pred - gold
    if re.fullmatch(r"10+", str(abs(diff))):
        return "carry_dropped" if diff < 0 else "carry_extra"
    p, g = str(abs(pred)), str(gold)
    if sorted(p) == sorted(g):
        return "transposition"
    if p[-1] != g[-1]:
        return "units_wrong"
    return "tens_wrong" if p[-2:-1] != g[-2:-1] else "high_wrong"


# ---------------------------------------------------------------------------
# Counterfactual minimal pairs
# ---------------------------------------------------------------------------

def _units_carry(rng, tens_total):
    a, b = rng.randint(11, 99), rng.randint(10, 99)
    ua, ub, ta, tb = a % 10, b % 10, a // 10, b // 10
    if ua == 0 or ua + ub >= 10 or not tens_total(ta + tb):
        return None
    return f"{a}+{b}=", f"{a}+{10 * tb + rng.randint(10 - ua, 9)}="


def _tens_carry(rng):
    a, b = rng.randint(10, 99), rng.randint(10, 99)
    if a % 10 + b % 10 >= 10 or a // 10 + b // 10 >= 10:
        return None
    return f"{a}+{b}=", f"{a}+{10 * rng.randint(10 - a // 10, 9) + b % 10}="


def _mult_carry(rng):
    d = rng.randint(2, 9)
    ta = rng.randint(1, 9 // d)  # the base product carries nowhere
    return f"{10 * ta + rng.randint(0, 9 // d)}*{d}=", f"{10 * ta + rng.randint(-(-10 // d), 9)}*{d}="


# name -> (description, sampler(rng) -> (base, variant) or None to resample)
FAMILIES = {
    "units_carry": ("change one units digit so the units column carries; both sums < 100 (36+53 vs 36+59)",
                    lambda r: _units_carry(r, lambda t: t <= 8)),
    "carry_to_100": ("change one units digit so a units carry ripples into a new digit (45+54 vs 45+55)",
                     lambda r: _units_carry(r, lambda t: t == 9)),
    "tens_carry": ("change one tens digit so the tens column carries into a new digit (42+35 vs 42+75)",
                   _tens_carry),
    "extra_operand": ("add a third two-digit operand, 22_add -> 222_add (36+59 vs 36+59+11)",
                      lambda r: (lambda a, b, c: (f"{a}+{b}=", f"{a}+{b}+{c}="))(
                          r.randint(10, 99), r.randint(10, 99), r.randint(10, 99))),
    "widen_operands": ("prefix a hundreds digit to both operands, 22_add -> 33_add (36+59 vs 136+259)",
                       lambda r: (lambda a, b, x, y: (f"{a}+{b}=", f"{100 * x + a}+{100 * y + b}="))(
                           r.randint(10, 99), r.randint(10, 99), r.randint(1, 9), r.randint(1, 9))),
    "double_to_mult": ("the same sum as a product, 22_add -> 21_mult (36+36 vs 36*2)",
                       lambda r: (lambda a: (f"{a}+{a}=", f"{a}*2="))(r.randint(10, 99))),
    "triple_to_mult": ("the same sum as a product, 222_add -> 21_mult (36+36+36 vs 36*3)",
                       lambda r: (lambda a: (f"{a}+{a}+{a}=", f"{a}*3="))(r.randint(10, 99))),
    "mult_carry": ("change the units digit of the two-digit factor so a carry-free product carries (31*3 vs 34*3)",
                   _mult_carry),
}


def make_pairs(rng: random.Random, n: int, exclude: set[str] = frozenset()) -> dict[str, list[dict[str, Any]]]:
    """Up to ``n`` distinct pairs per family, neither prompt in ``exclude`` (the training split).

    double_to_mult has only 90 bases, so against a half-size training split it stops near 45.
    """
    out = {}
    for fam, (_, sample) in FAMILIES.items():
        seen, pairs = set(), []
        for _ in range(1000 * n):
            if len(pairs) == n:
                break
            pair = sample(rng)
            if pair is None or pair in seen or any(p in exclude for p in pair):
                continue
            seen.add(pair)
            pairs.append({"base": pair[0], "variant": pair[1],
                          "gold_base": int(true_answer(pair[0])), "gold_variant": int(true_answer(pair[1]))})
        out[fam] = pairs
    return out


# ---------------------------------------------------------------------------
# Running a model
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def recording_generate(model):
    """Record the (input_ids, output) of every generate call eval_model makes."""
    calls = []
    orig = model.generate

    def generate(*a, **kw):
        out = orig(*a, **kw)
        calls.append((kw["input_ids"], out))
        return out

    model.generate = generate
    try:
        yield calls
    finally:
        del model.generate


@torch.no_grad()
def greedy_rows(model, tokenizer, name: str, data: dict, batch_size: int, max_tokens: int):
    ds = PromptAnswerDataset(name, data, tokenizer)
    with recording_generate(model) as calls, student_autocast():
        acc = eval_model(model, tokenizer, ds, name, batch_size, max_tokens)
    order = sorted(ds.samples, key=lambda s: len(s["formatted_prompt"]), reverse=True)
    texts, fed = [], []
    for ids, out in calls:
        texts += tokenizer.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)
        fed += tokenizer.batch_decode(ids, skip_special_tokens=True)
    if len(texts) != len(order):
        raise RuntimeError(f"{name}: recorded {len(texts)} generations for {len(order)} prompts")
    rows = {}
    for s, text, seen in zip(order, texts, fed):
        if seen != s["formatted_prompt"]:
            raise RuntimeError(f"{name}: generation for {s['prompt']!r} was fed {seen!r}")
        pred = extract_leading_int(text)
        rows[s["prompt"]] = {"gold": s["answer"], "text": text, "pred": pred,
                             "correct": pred is not None and pred == s["answer"]}
    mine = sum(r["correct"] for r in rows.values()) / max(len(rows), 1)
    if abs(mine - acc) > 1e-9:
        raise RuntimeError(f"{name}: per-prompt grading gives {mine:.4f}, eval_model {acc:.4f}")
    return acc, rows


@torch.no_grad()
def next_token_stats(model, tokenizer, rows: dict, batch_size: int, k: int = 5) -> None:
    """Add p_gold (first gold token), first_right and the top-k next tokens to every row, in place."""
    model.eval()
    tokenizer.padding_side = "right"
    bos = tokenizer.bos_token or ""
    prompts = list(rows)
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        add_special_tokens=bos_in_eval() and not (bos and chunk[0].startswith(bos))).to(DEVICE)
        with student_autocast():
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        last = enc["attention_mask"].sum(1) - 1
        probs = torch.softmax(logits[torch.arange(len(chunk), device=DEVICE), last].float(), -1)
        top_p, top_i = probs.topk(k, -1)
        for r, p in enumerate(chunk):
            g = tokenizer(str(rows[p]["gold"]), add_special_tokens=False)["input_ids"][0]
            rows[p].update(p_gold=round(float(probs[r, g]), 5), first_right=int(top_i[r, 0]) == g,
                           top=[[tokenizer.decode([int(t)]), round(float(q), 4)] for t, q in zip(top_i[r], top_p[r])])


def run_model(role: str, path: str, sets: dict[str, dict], args) -> dict[str, Any]:
    print(f"\n=== {LABELS[role]}: {path}")
    # Students in the precision they were trained in (--student-dtype), bf16 autocast; teacher bf16.
    model, tokenizer = load_model(path) if role == "teacher" else load_student(path, args.student_dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    rec = {"role": role, "path": path, "settings": settings(args), "accuracy": {}, "rows": {}}
    for name, data in sets.items():
        acc, rows = greedy_rows(model, tokenizer, name, data, args.eval_batch_size, args.max_eval_tokens)
        next_token_stats(model, tokenizer, rows, args.score_batch_size)
        rec["accuracy"][name], rec["rows"][name] = acc, rows
        print(f"  {name:10s} acc {acc:.4f}  (n={len(rows)})")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec


def settings(args) -> dict[str, Any]:
    return {"bos_mode": args.bos_mode, "max_eval_tokens": args.max_eval_tokens, "test_limit": args.test_limit}


def covers(saved: dict, sets: dict[str, dict], args, path: str | None) -> bool:
    return (saved.get("settings") == settings(args) and (path is None or saved.get("path") == path)
            and all(set(data) <= set(saved["rows"].get(name, {})) for name, data in sets.items()))


# ---------------------------------------------------------------------------
# Analysis (pure python on the prediction records)
# ---------------------------------------------------------------------------

def md_table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def code(s: str) -> str:
    s = s.replace("\n", "\\n").replace("|", "\\|")
    return f"`{s}`" if s.strip() else "(empty)"


def tok(s: str) -> str:
    return code(s.replace(" ", "\u2423"))


def mark(row: dict) -> str:
    return "\u2713" if row["correct"] else "\u2717"


def error_analysis(preds: dict, sets: dict[str, dict], structs: dict) -> tuple[dict, str]:
    roles = list(preds)
    summary, md = {"strata": {}, "errors": {}}, ["# Error analysis\n"]
    for name, data in sets.items():
        if name == PAIR_SET:
            continue
        rows = {r: preds[r]["rows"][name] for r in roles}
        strata, table = {}, []
        for label, keep in STRATA:
            ps = [p for p in data if keep(structs[p])]
            if not ps:
                continue
            strata[label] = {"n": len(ps), **{r: sum(rows[r][p]["correct"] for p in ps) / len(ps) for r in roles}}
            table.append([label, len(ps)] + [f"{strata[label][r]:.3f}" for r in roles])
        errors = {r: {e: 0 for e in ERRORS} for r in roles}
        for r in roles:
            for p in data:
                e = error_type(rows[r][p]["pred"], data[p], structs[p].get("operands", []))
                if e is not None:
                    errors[r][e] += 1
        wrong = {r: sum(errors[r].values()) for r in roles}
        err_table = [[e] + [f"{errors[r][e]} ({errors[r][e] / wrong[r]:.0%})" if wrong[r] else "0" for r in roles]
                     for e in ERRORS] + [["**wrong total**"] + [wrong[r] for r in roles]]
        summary["strata"][name], summary["errors"][name] = strata, errors
        md += [f"## {name}\n", "Accuracy by structure:\n", md_table(["stratum", "n"] + [LABELS[r] for r in roles], table),
               "\nWrong answers by type (count, share of that model's errors):\n",
               md_table(["error"] + [LABELS[r] for r in roles], err_table), ""]
    return summary, "\n".join(md)


def pair_analysis(preds: dict, pairs: dict, n_show: int = 3) -> tuple[dict, list, str]:
    roles = list(preds)
    rows = {r: preds[r]["rows"][PAIR_SET] for r in roles}
    summary, records, md = {}, [], ["# Counterfactual minimal pairs\n"]
    for fam, plist in pairs.items():
        if not plist:
            continue
        stats, table = {}, []
        for r in roles:
            b = [rows[r][q["base"]] for q in plist]
            v = [rows[r][q["variant"]] for q in plist]
            s = {"base_acc": sum(x["correct"] for x in b) / len(b),
                 "variant_acc": sum(x["correct"] for x in v) / len(v),
                 "both": sum(x["correct"] and y["correct"] for x, y in zip(b, v)) / len(b),
                 "breaks": sum(x["correct"] and not y["correct"] for x, y in zip(b, v)) / len(b),
                 "p_gold_base": sum(x["p_gold"] for x in b) / len(b),
                 "p_gold_variant": sum(x["p_gold"] for x in v) / len(v)}
            if "teacher" in roles:
                s["same_as_teacher"] = sum(
                    (rows[r][q["base"]]["correct"], rows[r][q["variant"]]["correct"])
                    == (rows["teacher"][q["base"]]["correct"], rows["teacher"][q["variant"]]["correct"])
                    for q in plist) / len(plist)
            stats[r] = s
            table.append([LABELS[r]] + [f"{s[k]:.2f}" for k in ("base_acc", "variant_acc", "both", "breaks")]
                         + [f"{s['p_gold_base']:.2f} \u2192 {s['p_gold_variant']:.2f}"]
                         + ([f"{s['same_as_teacher']:.2f}"] if "teacher" in roles else []))
        summary[fam] = stats
        fam_records = [dict(q, family=fam, models={r: {side: {k: rows[r][q[side]][k] for k in ("pred", "correct", "p_gold", "text")}
                                                       for side in ("base", "variant")} for r in roles})
                       for q in plist]
        records += fam_records
        # disagreements first: most distinct (base, variant) answer patterns across models
        shown = sorted(fam_records, key=lambda q: -len({(m["base"]["pred"], m["variant"]["pred"])
                                                         for m in q["models"].values()}))[:n_show]
        md += [f"## {fam}\n", f"{FAMILIES[fam][0]}; {len(plist)} pairs.\n",
               md_table(["model", "base acc", "variant acc", "both right", "breaks", "p(gold) base \u2192 variant"]
                        + (["same pattern as teacher"] if "teacher" in roles else []), table),
               "\nIllustrative pairs (most model disagreement first; the table above is over all pairs):\n",
               md_table(["base", "variant"] + [LABELS[r] for r in roles],
                        [[f"{code(q['base'])} {q['gold_base']}", f"{code(q['variant'])} {q['gold_variant']}"]
                         + [f"{q['models'][r]['base']['pred']}{mark(q['models'][r]['base'])} / "
                            f"{q['models'][r]['variant']['pred']}{mark(q['models'][r]['variant'])}" for r in roles]
                         for q in shown]), ""]
    return summary, records, "\n".join(md)


def spread(prompts: list[str], n: int, score, structs: dict) -> list[str]:
    """Up to ``n`` prompts, round-robin over carry counts, highest ``score`` first within each."""
    groups: dict[Any, list[str]] = {}
    for p in sorted(prompts, key=lambda p: (-score(p), p)):
        groups.setdefault(structs[p].get("carries"), []).append(p)
    keys = sorted(groups, key=lambda k: (k is None, k or 0))
    out = []
    while len(out) < n and any(groups.values()):
        for k in keys:
            if groups[k] and len(out) < n:
                out.append(groups[k].pop(0))
    return out


def pick_examples(preds: dict, sets: dict[str, dict], structs: dict, train_dataset: str,
                  n: int, n_reverse: int) -> tuple[list, str]:
    roles = list(preds)
    others = [r for r in ("kd", "sft") if r in roles]
    if "graph" not in roles or not others:
        return [], "# Examples\n\nNeeds --graph and at least one of --kd / --sft.\n"
    other_names = " and ".join(LABELS[r] for r in others)
    records, md = [], ["# Examples\n"]
    for name, data in sets.items():
        if name in (PAIR_SET, train_dataset):
            continue
        rows = {r: preds[r]["rows"][name] for r in roles}
        wins, losses = [], []
        for p in data:
            if "teacher" in roles and not rows["teacher"][p]["correct"]:
                continue
            o = [rows[r][p]["correct"] for r in others]
            if rows["graph"][p]["correct"] and not any(o):
                wins.append(p)
            elif not rows["graph"][p]["correct"] and any(o):
                losses.append(p)
        kinds = [("graph_right", f"Graph KD right, {other_names} wrong", wins,
                  spread(wins, n, lambda p: rows["graph"][p]["p_gold"], structs)),
                 ("graph_wrong", f"Graph KD wrong, {other_names.replace(' and ', ' or ')} right", losses,
                  spread(losses, n_reverse, lambda p: max(rows[r][p]["p_gold"] for r in others), structs))]
        md.append(f"## {name}\n")
        for kind, title, pool, chosen in kinds:
            md.append(f"### {title} ({len(pool)} of {len(data)} prompts"
                      + ("; teacher right on all" if "teacher" in roles else "") + ")\n")
            for p in chosen:
                s = structs[p]
                models = {r: dict(rows[r][p], error=error_type(rows[r][p]["pred"], data[p], s.get("operands", [])))
                          for r in roles}
                records.append({"set": name, "kind": kind, "prompt": p, "gold": data[p], "structure": s, "models": models})
                c = s.get("carries")
                md += [f"{code(p)} = {data[p]}"
                       + (f" ({c} carr{'y' if c == 1 else 'ies'}{', new digit' if s.get('new_digit') else ''})"
                          if c is not None else "") + "\n",
                       md_table(["model", "output", "error", "p(gold first token)", "top-5 next tokens"],
                                [[LABELS[r], f"{code(m['text'])} {mark(m)}", m["error"] or "", f"{m['p_gold']:.3f}",
                                  ", ".join(f"{tok(t)} {q:.2f}" for t, q in m["top"])]
                                 for r, m in models.items()]), ""]
    return records, "\n".join(md)


def analyse(preds: dict, sets: dict[str, dict], pairs: dict, train_dataset: str,
            n_examples: int, n_reverse: int) -> tuple[dict, dict[str, str], dict]:
    """(summary, {markdown file: text}, {json file: object}) from the prediction records."""
    structs = {p: structure(p, gold) for data in sets.values() for p, gold in data.items()}
    summary = {"accuracy": {r: preds[r]["accuracy"] for r in preds},
               "models": {r: preds[r]["path"] for r in preds}}
    err, err_md = error_analysis(preds, sets, structs)
    summary.update(err)
    summary["pairs"], pair_records, pair_md = pair_analysis(preds, pairs)
    examples, ex_md = pick_examples(preds, sets, structs, train_dataset, n_examples, n_reverse)
    summary["examples"] = {}
    for e in examples:
        summary["examples"].setdefault(f"{e['set']}/{e['kind']}", []).append(e["prompt"])
    return (summary, {"error_analysis.md": err_md, "pairs.md": pair_md, "examples.md": ex_md},
            {"pairs.json": pair_records, "examples.json": examples})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for role in ROLES:
        ap.add_argument(f"--{role}", default=None, help=f"{LABELS[role]}: HF id or checkpoint dir (final_checkpoint).")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--train-dataset", default="22_add",
                    help="Excluded from the examples; its train split is excluded from the pair prompts.")
    ap.add_argument("--bos-mode", default="legacy", choices=BOS_MODES,
                    help="Must match the runs being examined; every recorded result used legacy.")
    ap.add_argument("--max-eval-tokens", type=int, default=5, help="The recorded runs used 5.")
    ap.add_argument("--eval-batch-size", type=int, default=1024)
    ap.add_argument("--score-batch-size", type=int, default=256,
                    help="Prompts per teacher-forced forward (full-vocab logits at every position).")
    ap.add_argument("--test-limit", type=int, default=None)
    ap.add_argument("--student-dtype", default="bfloat16", choices=["bfloat16", "float32"],
                    help="Match the students' --dtype at training time.")
    ap.add_argument("--n-pairs", type=int, default=50, help="Pairs per family.")
    ap.add_argument("--n-examples", type=int, default=5)
    ap.add_argument("--n-reverse", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse", action="store_true",
                    help="Load predictions/<role>.json instead of the model when it covers every prompt.")
    ap.add_argument("--out-dir", default=os.path.join(DIR_ROOT, "results", "case_studies"))
    args = ap.parse_args()
    set_bos_mode(args.bos_mode)

    sets = {name: load_data(name, test_limit=args.test_limit)[1] for name in args.datasets}
    pairs = make_pairs(random.Random(args.seed), args.n_pairs, set(load_data(args.train_dataset)[0]))
    sets[PAIR_SET] = {q[side]: q[f"gold_{side}"] for plist in pairs.values() for q in plist for side in ("base", "variant")}

    pred_dir = os.path.join(args.out_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    preds = {}
    for role in ROLES:
        path, f = getattr(args, role), os.path.join(pred_dir, f"{role}.json")
        if args.reuse and os.path.isfile(f):
            with open(f, encoding="utf-8") as fh:
                saved = json.load(fh)
            if covers(saved, sets, args, path):
                print(f"{LABELS[role]}: reusing {f}")
                preds[role] = saved
                continue
        if path:
            preds[role] = run_model(role, path, sets, args)
            with open(f, "w", encoding="utf-8") as fh:
                json.dump(preds[role], fh)
    if not preds:
        ap.error("no models: pass at least one of " + ", ".join(f"--{r}" for r in ROLES))

    summary, mds, jsons = analyse(preds, sets, pairs, args.train_dataset, args.n_examples, args.n_reverse)
    summary["args"] = vars(args)
    for name, obj in {**jsons, "summary.json": summary}.items():
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2)
    for name, text in mds.items():
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as fh:
            fh.write(text)

    print("\naccuracy (eval_model):")
    print(f"  {'':12s}" + "".join(f"{n:>10s}" for n in sets))
    for r in preds:
        print(f"  {LABELS[r]:12s}" + "".join(f"{preds[r]['accuracy'].get(n, float('nan')):10.4f}" for n in sets))
    print("wrote", ", ".join(sorted([*mds, *jsons, "summary.json"])), "to", args.out_dir)


if __name__ == "__main__":
    main()

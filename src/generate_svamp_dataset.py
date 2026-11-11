"""SVAMP as a local dataset, in the format the arithmetic ones use.

Why not the existing ``svamp`` name. ``svamp`` is in ``utils._HF_DATASETS``: eval
generates 256 tokens and grades the *last* ``= N`` anywhere in the decoded text,
prompt echo included. Writing it under another name sends it down the local path,
graded by ``answer_type`` in ``meta.json``.

Two formats:

* ``--format cot`` (default, -> ``svamp_cot``): the response is a short gold
  chain of thought built from SVAMP's annotated equation, then the final answer::

      290 / 2 = 145
      The answer is 145.<|eot_id|>

  ``answer_type: cot`` makes eval grade the number in "The answer is N", and
  ``--graph-position final-answer`` builds the graph term at the position that
  predicts that number. Every equation is evaluated and must reproduce the gold
  answer, or the problem is dropped.
* ``--format direct`` (-> ``svamp_direct``): the bare number, graded by the
  leading integer, graph at the last prompt position.

Splits. ``--n-sft`` problems are held out of train as ``sft.json`` for the
format-fixing SFT of teacher and student (``sft --use-sft-split``), so the
distillation runs train on problems neither model was fine-tuned on.

Prompt. The Llama-3 instruct chat format written out literally (not
``apply_chat_template``: Llama-3.2's template injects today's date into a system
block, which would make the prompt depend on the day it was built and differ
between the 3 and 3.2 tokenizers). The text carries ``<|begin_of_text|>`` itself;
tokenize_prompt_answer, eval_model and the attribution adapter all detect that and
add no second BOS.

Answers. Integer answers only (the graders read integers). ``--max-answer``
(default 999) keeps answers Llama-3 writes as one token, so the graph at the
final-answer position is the graph for the whole final answer.

Usage (from src/, on Colab where DIR_ROOT is the Drive folder):
    python generate_svamp_dataset.py                       # -> datasets/svamp_cot/
    python generate_svamp_dataset.py --format direct       # -> datasets/svamp_direct/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import DIR_ROOT, extract_cot_answer, split_final_answer  # noqa: E402

TEMPLATES = {
    "direct": (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "{question}\nAnswer with only the final number.<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    "cot": (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "{question}\nWrite the equation, then the final answer as \"The answer is N.\"<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
}
_SAFE = re.compile(r"^[\d\s.+\-*/()]+$")


def clean_equation(eq: str) -> str:
    """'( ( 7.0 + 2.0 ) * 3.0 )' -> '(7 + 2) * 3': drop '.0', tidy spaces, outer parens."""
    s = re.sub(r"(\d+)\.0\b", r"\1", eq.strip())
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\s+", " ", s).strip()
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        for i, c in enumerate(s):
            depth += c == "("
            depth -= c == ")"
            if depth == 0 and i < len(s) - 1:
                break
        else:
            s = s[1:-1].strip()
            continue
        break
    return s


def _rows(split, fmt: str, max_answer: int, stats: dict) -> list[dict]:
    rows = []
    for ex in split:
        body = str(ex.get("Body") or "").strip()
        question = str(ex.get("Question") or "").strip()
        text = f"{body} {question}".strip() if (body or question) else str(ex.get("question_concat") or "").strip()
        try:
            ans = float(ex["Answer"])
        except (KeyError, TypeError, ValueError):
            stats["unparsable"] += 1
            continue
        if not ans.is_integer():
            stats["non_integer"] += 1
            continue
        ans = int(ans)
        if ans < 0 or ans > max_answer:
            stats["out_of_range"] += 1
            continue
        if fmt == "cot":
            eq = clean_equation(str(ex.get("Equation") or ""))
            if not eq or not _SAFE.match(eq):
                stats["bad_equation"] += 1
                continue
            try:
                val = eval(eq, {"__builtins__": {}}, {})  # digits, + - * / ( ) only
            except Exception:
                stats["bad_equation"] += 1
                continue
            if abs(val - ans) > 1e-6:
                stats["equation_mismatch"] += 1
                continue
            response = f"{eq} = {ans}\nThe answer is {ans}."
            assert extract_cot_answer(response) == ans and split_final_answer(response)[1] == str(ans)
        else:
            response = str(ans)
        rows.append({"q_str": TEMPLATES[fmt].format(question=text), "a_str": response})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--format", choices=sorted(TEMPLATES), default="cot")
    p.add_argument("--dataset-name", default=None, help="Default: svamp_<format>")
    p.add_argument("--hub-id", default="ChilleD/SVAMP")
    p.add_argument("--max-answer", type=int, default=999)
    p.add_argument("--n-sft", type=int, default=150,
                   help="Train problems held out as sft.json for the format-fixing SFT (0: none).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    name = args.dataset_name or f"svamp_{args.format}"
    if name in ("svamp", "gsm8k"):
        raise SystemExit(f"{name!r} is an HF chain-of-thought name in utils._HF_DATASETS; pick another")

    from datasets import load_dataset
    ds = load_dataset(args.hub_id)
    stats = {k: 0 for k in ("unparsable", "non_integer", "out_of_range", "bad_equation", "equation_mismatch")}
    train = _rows(ds["train"], args.format, args.max_answer, stats)
    test = _rows(ds["test"], args.format, args.max_answer, stats)
    train_q = {r["q_str"] for r in train}
    n_overlap = sum(r["q_str"] in train_q for r in test)
    test = [r for r in test if r["q_str"] not in train_q]
    random.Random(args.seed).shuffle(train)
    sft, train = train[: args.n_sft], train[args.n_sft:]

    out_dir = os.path.join(DIR_ROOT, "datasets", name)
    os.makedirs(out_dir, exist_ok=True)
    files = {"train.json": train, "test.json": test, "all.json": train + test + sft}
    if sft:
        files["sft.json"] = sft
    for fname, data in files.items():
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"answer_type": "cot" if args.format == "cot" else "int", "source": args.hub_id,
                   "format": args.format}, f, indent=4)
    print(f"wrote {out_dir}: train {len(train)} | sft {len(sft)} | test {len(test)} | dropped {stats} | "
          f"train/test overlap removed {n_overlap}")
    print("example prompt:\n" + train[0]["q_str"] + "\nresponse: " + repr(train[0]["a_str"]))


if __name__ == "__main__":
    main()

"""SVAMP as a direct-answer local dataset, in the format the arithmetic ones use.

Why not the existing ``svamp`` name. ``svamp`` is in ``utils._HF_DATASETS``: eval
generates 256 tokens and grades the *last* ``= N`` in the decoded text, a
chain-of-thought convention. Every trainer here scores a bare answer + EOS, so a
student distilled on that would emit "15" with no "=" and grade 0. Writing it
under another name (default ``svamp_direct``) sends it down the local arithmetic
path instead: one answer token predicted at the last prompt position, greedy
decode of a few tokens, graded by the leading integer -- the path every 22_add
result went through, and the position the graph term attributes from.

Prompt. The Llama-3 instruct chat format written out literally (not
``apply_chat_template``: Llama-3.2's template injects today's date into a system
block, which would make the prompt depend on the day it was built, and differ
between the 3 and 3.2 tokenizers). The text carries ``<|begin_of_text|>`` itself;
tokenize_prompt_answer, eval_model and the attribution adapter all detect that and
add no second BOS. The answer is the number followed by the tokenizer's EOS, which
for the instruct models is ``<|eot_id|>``, so the target ends the assistant turn
the way the teacher would.

Answers. Integer answers only (SVAMP has a few decimals, which the leading-int
grader cannot read). ``--max-answer`` (default 999) keeps answers that Llama-3
writes as a single token, so "the graph at the answer position" is exactly the
graph for the final response token; pass a larger value to keep them all.

Usage (from src/, on Colab where DIR_ROOT is the Drive folder):
    python generate_svamp_dataset.py                 # -> datasets/svamp_direct/
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import DIR_ROOT  # noqa: E402

TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "{question}\nAnswer with only the final number.<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)


def _rows(split, max_answer: int, stats: dict) -> list[dict]:
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
        rows.append({"q_str": TEMPLATE.format(question=text), "a_str": str(ans)})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset-name", default="svamp_direct")
    p.add_argument("--hub-id", default="ChilleD/SVAMP")
    p.add_argument("--max-answer", type=int, default=999)
    args = p.parse_args()
    if args.dataset_name in ("svamp", "gsm8k"):
        raise SystemExit(f"{args.dataset_name!r} is an HF chain-of-thought name in utils._HF_DATASETS; pick another")

    from datasets import load_dataset
    ds = load_dataset(args.hub_id)
    stats = {"unparsable": 0, "non_integer": 0, "out_of_range": 0}
    train = _rows(ds["train"], args.max_answer, stats)
    test = _rows(ds["test"], args.max_answer, stats)
    overlap = len({r["q_str"] for r in train} & {r["q_str"] for r in test})
    if overlap:
        test = [r for r in test if r["q_str"] not in {x["q_str"] for x in train}]

    out_dir = os.path.join(DIR_ROOT, "datasets", args.dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    for name, data in (("train.json", train), ("test.json", test), ("all.json", train + test)):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    print(f"wrote {out_dir}: train {len(train)} | test {len(test)} | dropped {stats} | "
          f"train/test overlap removed {overlap}")
    print("example prompt:\n" + train[0]["q_str"] + "\nanswer: " + train[0]["a_str"])


if __name__ == "__main__":
    main()

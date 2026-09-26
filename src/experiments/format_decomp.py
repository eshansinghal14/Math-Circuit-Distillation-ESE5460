"""Split each model's held-out accuracy into "answers with a number" x "the number is right".

Under ``--bos-mode legacy`` the KD and SFT terms train on sequences with no BOS
while eval_model (and the graph term's attribution adapter) put BOS first, so a
student can know the answer and still open its eval continuation with ``?`` or a
newline. This script separates the two, for every model, with BOS at eval (the
recorded convention) and without it (the format KD and SFT actually trained on):

  * ``acc``: eval_model's greedy exact match, through case_studies.greedy_rows so
    it reproduces the training logs' numbers under legacy;
  * ``commit``: the continuation starts with an integer (extract_leading_int);
  * ``acc|commit``: exact match among those;
  * ``p_num``: next-token probability mass on number tokens at the answer position,
    teacher-forced, same BOS rule as the greedy pass; ``p_q``: mass on tokens that
    start with ``?``;
  * ``num_argmax``: the most likely *number* token equals the gold answer's first
    token, i.e. accuracy with the format decision taken away. The answers of every
    set but 33_add fit in one Llama-3 token, so there it is the whole answer.

Usage (from the repository root, GPU):
    PYTHONPATH=src python -m experiments.format_decomp --test-limit 2000 \\
        --models teacher=meta-llama/Meta-Llama-3-8B-Instruct base=meta-llama/Llama-3.2-1B-Instruct \\
        kd1=<dir> graph1=<dir> ...
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DIR_ROOT, bos_in_eval, load_data, load_model, set_bos_mode  # noqa: E402
from training.utils import load_student, student_autocast  # noqa: E402
from experiments.case_studies import DATASETS, DEVICE, greedy_rows  # noqa: E402

MODES = {"bos": "legacy", "nobos": "off"}


def token_classes(tokenizer) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Masks over the vocabulary: tokens that open an integer, tokens that open with '?'."""
    text = [tokenizer.decode([i]) for i in range(len(tokenizer))]
    num = torch.tensor([bool(re.fullmatch(r"\s?\d+", t)) for t in text], device=DEVICE)
    q = torch.tensor([t.lstrip().startswith("?") for t in text], device=DEVICE)
    return num, q, text


@torch.no_grad()
def answer_position(model, tokenizer, rows: dict, masks, batch_size: int) -> None:
    """Add p_num, p_q, top1_num and num_argmax to every row, in place (BOS as bos_in_eval)."""
    num, q, text = masks
    tokenizer.padding_side = "right"
    prompts = list(rows)
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        add_special_tokens=bos_in_eval()).to(DEVICE)
        with student_autocast():
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        last = enc["attention_mask"].sum(1) - 1
        probs = torch.softmax(logits[torch.arange(len(chunk), device=DEVICE), last].float(), -1)
        p_num, p_q = probs[:, num].sum(-1), probs[:, q].sum(-1)
        top1 = probs.argmax(-1)
        best_num = probs.masked_fill(~num, -1.0).argmax(-1)
        for r, p in enumerate(chunk):
            gold_first = tokenizer.decode(tokenizer(str(rows[p]["gold"]), add_special_tokens=False)["input_ids"][0])
            rows[p].update(p_num=float(p_num[r]), p_q=float(p_q[r]), top1_num=bool(num[top1[r]]),
                           num_argmax=text[int(best_num[r])].strip() == gold_first)


def summarise(rows: dict) -> dict:
    n = len(rows)
    committed = [r for r in rows.values() if r["pred"] is not None]
    mean = lambda k: sum(float(r[k]) for r in rows.values()) / n  # noqa: E731
    return {"n": n, "acc": mean("correct"), "commit": len(committed) / n,
            "acc_given_commit": sum(r["correct"] for r in committed) / max(len(committed), 1),
            "p_num": mean("p_num"), "p_q": mean("p_q"), "top1_num": mean("top1_num"),
            "num_argmax": mean("num_argmax")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", required=True, help="name=path; a name starting 'teacher' loads bf16 via load_model.")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--test-limit", type=int, default=2000, help="The recorded runs evaluated 2000 prompts.")
    ap.add_argument("--max-eval-tokens", type=int, default=5)
    ap.add_argument("--eval-batch-size", type=int, default=1024)
    ap.add_argument("--score-batch-size", type=int, default=256)
    ap.add_argument("--student-dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--out", default=os.path.join(DIR_ROOT, "results", "format_decomp", "summary.json"))
    args = ap.parse_args()

    sets = {name: load_data(name, test_limit=args.test_limit)[1] for name in args.datasets}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = {}
    if os.path.isfile(args.out):
        with open(args.out, encoding="utf-8") as fh:
            out = json.load(fh)
    for spec in args.models:
        name, path = spec.split("=", 1)
        print(f"\n=== {name}: {path}")
        model, tokenizer = load_model(path) if name.startswith("teacher") else load_student(path, args.student_dtype)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        masks = token_classes(tokenizer)
        out[name] = {"path": path, "results": {}}
        for mode in args.modes:
            set_bos_mode(MODES[mode])
            for ds, data in sets.items():
                _, rows = greedy_rows(model, tokenizer, ds, data, args.eval_batch_size, args.max_eval_tokens)
                answer_position(model, tokenizer, rows, masks, args.score_batch_size)
                s = summarise(rows)
                out[name]["results"].setdefault(mode, {})[ds] = s
                print(f"  {mode:5s} {ds:9s} acc {s['acc']:.3f} commit {s['commit']:.3f} acc|commit {s['acc_given_commit']:.3f} "
                      f"p_num {s['p_num']:.3f} p_q {s['p_q']:.3f} num_argmax {s['num_argmax']:.3f}")
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("wrote", args.out)


if __name__ == "__main__":
    main()

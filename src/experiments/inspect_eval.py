"""Print what a model actually generates at eval time, next to how it is graded.

Written because the 8B teacher scored 0.066 on SQuAD while the 1B student scored
0.363. A teacher that far below a student it is supposed to teach is a grading or
prompt-format problem, not a capability one, and the only way to tell which is to
look at the raw continuation beside the parsed answer.

Shows, per prompt: the gold answer, the raw continuation, what the grader
extracted from it, and whether that counted as correct. Also reports how often the
gold string appears *somewhere* in the continuation -- if that rate is far above
the exact-match rate, the model knows the answer and is losing points to verbosity
rather than to ignorance.

Usage (from src/):
    python -m experiments.inspect_eval --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --dataset squad --n 20 --max-eval-tokens 16
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (  # noqa: E402
    dataset_answer_type,
    extract_leading_int,
    extract_text_answer,
    load_data,
    load_model,
    normalize_answer_text,
    tokenize_prompt_answer,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--max-eval-tokens", type=int, default=16, dest="max_eval_tokens")
    p.add_argument("--split", default="test", choices=["test", "train"])
    p.add_argument("--show-prompt-chars", type=int, default=0, dest="show_prompt_chars",
                   help="Print this many characters of each prompt (0 = just the question line).")
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, test = load_data(args.dataset)
    data = list((train if args.split == "train" else test).items())[: args.n]
    is_text = dataset_answer_type(args.dataset) == "text"
    model, tokenizer = load_model(args.model)
    model.eval()

    exact = contains = 0
    for prompt, gold in data:
        p_ids, _ = tokenize_prompt_answer(tokenizer, prompt, str(gold))
        with torch.no_grad():
            out = model.generate(
                p_ids.unsqueeze(0).to(device),
                max_new_tokens=args.max_eval_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        cont = tokenizer.decode(out[0, p_ids.numel():], skip_special_tokens=True)
        if is_text:
            pred = extract_text_answer(cont)
            gold_parsed = normalize_answer_text(str(gold)) or None
            in_cont = normalize_answer_text(str(gold)) in normalize_answer_text(cont)
        else:
            pred = extract_leading_int(cont)
            gold_parsed = gold if isinstance(gold, int) else extract_leading_int(str(gold))
            in_cont = str(gold) in cont
        ok = pred is not None and gold_parsed is not None and pred == gold_parsed
        exact += bool(ok)
        contains += bool(in_cont)
        head = prompt[: args.show_prompt_chars] if args.show_prompt_chars else prompt.split("\n")[-2:]
        print(f"[{'OK ' if ok else 'BAD'}] gold={gold!r}")
        print(f"       prompt tail : {head}")
        print(f"       continuation: {cont!r}")
        print(f"       graded as   : {pred!r}")
    n = max(len(data), 1)
    print()
    print(f"exact match          : {exact}/{n} = {exact / n:.3f}")
    print(f"gold appears in cont : {contains}/{n} = {contains / n:.3f}")
    if contains > exact * 1.5 and contains > 0:
        print("  The gold answer is in the continuation far more often than it is graded")
        print("  correct, so the model knows the answer and is losing points to format.")
        print("  Fix the prompt or the metric, not the model.")


if __name__ == "__main__":
    main()

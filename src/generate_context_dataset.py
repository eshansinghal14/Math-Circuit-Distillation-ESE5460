"""Build a context-grounded QA dataset in the same format the arithmetic ones use.

Why this exists. The token-path loss is a distribution over *input positions*, so
it only carries signal when the answer-relevant positions move from example to
example. On ``22_add`` they do not -- "the operands matter, the + and = do not"
is the same story for every prompt -- and the same-vs-shuffled control came back
at 0.94, meaning a wrong prompt's profile fit the student slightly better than
the right one. In extractive QA the supporting span sits somewhere different in
every context, which is the precondition that arithmetic cannot satisfy.

It also gives distillation something to lose at. Plain fine-tuning closes 114% of
the teacher gap on ``22_add``, leaving no room for any distillation method to show
value. The canonical small-model failure in context-grounded QA is answering from
memorised priors instead of reading the passage, which fine-tuning on answers does
not fix and can worsen.

Writes ``datasets/<name>/{train,test,all}.json`` as ``{"q_str", "a_str"}`` rows,
exactly like generate_math_dataset.py, plus a ``meta.json`` marking
``answer_type: text`` so eval_model grades by normalised exact match instead of
parsing a leading integer.

Usage (from src/):
    python generate_context_dataset.py --dataset-name squad --source squad \\
        --train 5000 --test 5000
    python generate_context_dataset.py --dataset-name hotpot --source hotpotqa \\
        --train 5000 --test 5000        # multi-hop; use as the OOD eval set

Train on one and evaluate on the other to get the transfer axis: SQuAD is
single-passage single-hop, HotpotQA needs two supporting passages, so a student
that learned to read rather than to memorise should carry over.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re

from utils import DIR_ROOT

PROMPT = "{context}\nQ: {question}\nA:"


def _clean(text: str) -> str:
    """Collapse whitespace so a passage stays one block and the prompt's own
    newlines remain the only line breaks -- extract_text_answer cuts the model's
    continuation at the first newline."""
    return re.sub(r"\s+", " ", str(text)).strip()


def _squad_rows(split: str, limit: int, max_context_words: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("squad", split=split)
    rows, seen = [], set()
    for ex in ds:
        answers = ex["answers"]["text"]
        if not answers:
            continue
        context = _clean(ex["context"])
        if len(context.split()) > max_context_words:
            continue
        answer = _clean(answers[0])
        if not answer or answer.lower() not in context.lower():
            continue  # keep it extractive: the span must be in the passage
        q = PROMPT.format(context=context, question=_clean(ex["question"]))
        if q in seen:
            continue
        seen.add(q)
        rows.append({"q_str": q, "a_str": answer})
        if len(rows) >= limit:
            break
    return rows


def _hotpot_rows(split: str, limit: int, max_context_words: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", "distractor", split=split)
    rows, seen = [], set()
    for ex in ds:
        answer = _clean(ex["answer"])
        if not answer or answer.lower() in ("yes", "no"):
            continue  # yes/no needs no span, so it carries no positional signal
        titles = ex["context"]["title"]
        sentences = ex["context"]["sentences"]
        context = _clean(" ".join(" ".join(s) for s in sentences))
        if len(context.split()) > max_context_words:
            continue
        if answer.lower() not in context.lower():
            continue
        q = PROMPT.format(context=context, question=_clean(ex["question"]))
        if q in seen:
            continue
        seen.add(q)
        rows.append({"q_str": q, "a_str": answer})
        if len(rows) >= limit:
            break
    _ = titles  # titles are unused; the passages are concatenated in order
    return rows


SOURCES = {
    "squad": (_squad_rows, "train", "validation"),
    "hotpotqa": (_hotpot_rows, "train", "validation"),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset-name", required=True, help="Folder name under datasets/")
    p.add_argument("--source", required=True, choices=sorted(SOURCES))
    p.add_argument("--train", type=int, default=5000, dest="n_train")
    p.add_argument("--test", type=int, default=5000, dest="n_test")
    p.add_argument("--max-context-words", type=int, default=180, dest="max_context_words",
                   help="Skip longer passages. The token-path backward runs over the whole "
                        "sequence, so this bounds memory and step time.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-shuffle", action="store_false", dest="shuffle")
    return p


def main() -> None:
    args = build_parser().parse_args()
    fn, train_split, test_split = SOURCES[args.source]
    rng = random.Random(args.seed)

    # Train and test come from different upstream splits, so they cannot overlap.
    train = fn(train_split, args.n_train, args.max_context_words)
    test = fn(test_split, args.n_test, args.max_context_words)
    if args.shuffle:
        rng.shuffle(train)
        rng.shuffle(test)

    train_q = {r["q_str"] for r in train}
    overlap = sum(1 for r in test if r["q_str"] in train_q)
    if overlap:
        raise SystemExit(f"{overlap} test prompts also appear in train; refusing to write")

    out_dir = os.path.join(DIR_ROOT, "datasets", args.dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    def _write(name: str, data) -> None:
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    _write("train.json", train)
    _write("test.json", test)
    _write("all.json", train + test)
    _write("meta.json", {
        "answer_type": "text",
        "source": args.source,
        "max_context_words": args.max_context_words,
        "n_train": len(train),
        "n_test": len(test),
    })
    words = [len(r["q_str"].split()) for r in train]
    ans = [len(r["a_str"].split()) for r in train]
    print(f"Wrote {len(train)} train + {len(test)} test rows to {out_dir}/")
    print(f"  prompt words: median {sorted(words)[len(words) // 2]}, max {max(words)}")
    print(f"  answer words: median {sorted(ans)[len(ans) // 2]}, max {max(ans)}")
    print(f"  example prompt:\n{train[0]['q_str'][:300]}\n  answer: {train[0]['a_str']!r}")


if __name__ == "__main__":
    main()

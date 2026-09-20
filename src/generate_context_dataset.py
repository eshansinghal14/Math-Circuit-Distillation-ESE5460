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


def _load_split(candidates: list[str], split: str, config: str | None = None):
    """First of ``candidates`` that resolves, so a hub rename does not break this.

    The hub now requires a namespaced repo id, so a bare "squad" raises
    HfUriError ("Repository id must be 'namespace/name'"). The namespaced id is
    tried first and the legacy bare name kept as a fallback for older hub
    versions; every failure is reported so a genuine network or auth error is not
    mistaken for a rename.
    """
    from datasets import load_dataset

    errors = []
    for repo in candidates:
        try:
            return load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
        except Exception as e:  # noqa: BLE001 - report them all and move on
            errors.append(f"  {repo}: {type(e).__name__}: {e}")
    raise SystemExit(
        "could not load any of " + ", ".join(candidates) + ":\n" + "\n".join(errors))


def _clean(text: str) -> str:
    """Collapse whitespace so a passage stays one block and the prompt's own
    newlines remain the only line breaks -- extract_text_answer cuts the model's
    continuation at the first newline."""
    return re.sub(r"\s+", " ", str(text)).strip()


def _squad_rows(split: str, limit: int, max_context_words: int,
                max_answer_words: int, dropped: dict) -> list[dict]:
    ds = _load_split(["rajpurkar/squad", "squad"], split)
    rows, seen = [], set()
    for ex in ds:
        answers = ex["answers"]["text"]
        if not answers:
            continue
        context = _clean(ex["context"])
        if len(context.split()) > max_context_words:
            dropped["too_long_context"] = dropped.get("too_long_context", 0) + 1
            continue
        answer = _clean(answers[0])
        if not answer or answer.lower() not in context.lower():
            dropped["answer_not_in_context"] = dropped.get("answer_not_in_context", 0) + 1
            continue  # keep it extractive: the span must be in the passage
        dropped.setdefault("lengths", []).append(len(answer.split()))
        if len(answer.split()) > max_answer_words:
            dropped["long"] = dropped.get("long", 0) + 1
            continue
        q = PROMPT.format(context=context, question=_clean(ex["question"]))
        if q in seen:
            continue
        seen.add(q)
        rows.append({"q_str": q, "a_str": answer})
        if len(rows) >= limit:
            break
    return rows


def _hotpot_rows(split: str, limit: int, max_context_words: int,
                 max_answer_words: int, dropped: dict, n_paragraphs: int = 4,
                 seed: int = 0) -> list[dict]:
    """HotpotQA distractor, trimmed to a fixed number of paragraphs.

    The raw distractor setting concatenates ten paragraphs, typically 800-1200
    words, so a context cap sized for SQuAD rejects essentially every example.
    Keeping the supporting paragraphs plus a few distractors preserves what makes
    the task useful here -- the answer needs two passages, and the gold ones sit
    at a position that moves between examples -- at a length the token-path
    backward can afford.

    The kept paragraphs are shuffled so the gold ones are not always first, which
    would otherwise make position predictable and hand the loss a constant.
    """
    ds = _load_split(["hotpotqa/hotpot_qa", "hotpot_qa"], split, "distractor")
    rng = random.Random(seed)
    rows, seen = [], set()
    for ex in ds:
        answer = _clean(ex["answer"])
        if not answer:
            dropped["no_answer"] = dropped.get("no_answer", 0) + 1
            continue
        if answer.lower() in ("yes", "no"):
            dropped["yes_no"] = dropped.get("yes_no", 0) + 1
            continue  # yes/no needs no span, so it carries no positional signal

        titles = list(ex["context"]["title"])
        sentences = ex["context"]["sentences"]
        gold_titles = set(ex["supporting_facts"]["title"])
        gold = [i for i, t in enumerate(titles) if t in gold_titles]
        distract = [i for i in range(len(titles)) if i not in set(gold)]
        rng.shuffle(distract)
        keep = gold + distract[: max(n_paragraphs - len(gold), 0)]
        rng.shuffle(keep)                       # gold position must vary
        context = _clean(" ".join(" ".join(sentences[i]) for i in keep))

        if len(context.split()) > max_context_words:
            dropped["too_long_context"] = dropped.get("too_long_context", 0) + 1
            continue
        if answer.lower() not in context.lower():
            dropped["answer_not_in_context"] = dropped.get("answer_not_in_context", 0) + 1
            continue
        dropped.setdefault("lengths", []).append(len(answer.split()))
        if len(answer.split()) > max_answer_words:
            dropped["long"] = dropped.get("long", 0) + 1
            continue
        q = PROMPT.format(context=context, question=_clean(ex["question"]))
        if q in seen:
            continue
        seen.add(q)
        rows.append({"q_str": q, "a_str": answer})
        if len(rows) >= limit:
            break
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
    p.add_argument("--sft", type=int, default=0, dest="n_sft",
                   help="Also write sft.json with this many rows, taken from the upstream train "
                        "split *after* the ones used for train.json, so it is disjoint from both "
                        "train and test. Fine-tune the answer format on it (sft.py --use-sft-split) "
                        "and every distillation run still sees fresh prompts.")
    p.add_argument("--max-context-words", type=int, default=180, dest="max_context_words",
                   help="Skip longer passages. The token-path backward runs over the whole "
                        "sequence, so this bounds memory and step time.")
    p.add_argument("--max-answer-words", type=int, default=6, dest="max_answer_words",
                   help="Drop examples whose answer is longer. A long tail of many-word spans "
                        "forces a large --max-eval-tokens on every prompt, which costs generation "
                        "time on all of them; capping lets the eval budget match the median. "
                        "Applied to train and test alike so the eval distribution matches training.")
    p.add_argument("--hotpot-paragraphs", type=int, default=4, dest="n_paragraphs",
                   help="HotpotQA only: keep this many paragraphs, the supporting ones plus "
                        "distractors, shuffled so the gold position varies. The raw distractor "
                        "setting has 10 (800-1200 words), which a SQuAD-sized context cap rejects "
                        "almost entirely.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-shuffle", action="store_false", dest="shuffle")
    return p


def main() -> None:
    args = build_parser().parse_args()
    fn, train_split, test_split = SOURCES[args.source]
    rng = random.Random(args.seed)

    # Train and test come from different upstream splits, so they cannot overlap.
    stats: dict = {}
    # One pass for train + sft so the sft rows are the ones straight after train's,
    # then a slice: taking two independent passes would return the same rows twice.
    extra = {"n_paragraphs": args.n_paragraphs, "seed": args.seed} if args.source == "hotpotqa" else {}
    pool = fn(train_split, args.n_train + args.n_sft, args.max_context_words,
              args.max_answer_words, stats, **extra)
    train, sft = pool[: args.n_train], pool[args.n_train:]
    test = fn(test_split, args.n_test, args.max_context_words, args.max_answer_words, stats, **extra)
    if args.shuffle:
        rng.shuffle(train)
        rng.shuffle(test)

    if args.shuffle:
        rng.shuffle(sft)
    train_q = {r["q_str"] for r in train}
    overlap = sum(1 for r in test if r["q_str"] in train_q)
    overlap += sum(1 for r in sft if r["q_str"] in train_q)
    overlap += sum(1 for r in test if r["q_str"] in {x["q_str"] for x in sft})
    if overlap:
        raise SystemExit(f"{overlap} prompts are shared between splits; refusing to write")

    reasons = {k: v for k, v in stats.items() if k != "lengths"}
    if not train or not test:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())) or "no rows reached the filters"
        raise SystemExit(
            f"produced {len(train)} train and {len(test)} test rows. Dropped by filter: {detail}.\n"
            "  too_long_context dominating usually means --max-context-words is sized for one source "
            "and you are building another: HotpotQA's raw distractor setting is 800-1200 words, so "
            "either raise the cap or lower --hotpot-paragraphs.")
    reasons = {k: v for k, v in stats.items() if k != "lengths"}
    if reasons:
        print("dropped by filter: " + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())))
    out_dir = os.path.join(DIR_ROOT, "datasets", args.dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    def _write(name: str, data) -> None:
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    _write("train.json", train)
    _write("test.json", test)
    _write("all.json", train + test + sft)
    if sft:
        _write("sft.json", sft)
    _write("meta.json", {
        "answer_type": "text",
        "source": args.source,
        "max_context_words": args.max_context_words,
        "n_train": len(train),
        "n_test": len(test),
        "n_sft": len(sft),
    })
    seen_lengths = sorted(stats.get("lengths", []))
    if seen_lengths:
        def pct(q: float) -> int:
            return seen_lengths[min(int(q * len(seen_lengths)), len(seen_lengths) - 1)]
        n_long = stats.get("long", 0)
        print(f"answer length before the cap ({len(seen_lengths)} candidates): "
              f"median {pct(0.5)}, p90 {pct(0.90)}, p95 {pct(0.95)}, p99 {pct(0.99)}, "
              f"max {seen_lengths[-1]} words")
        print(f"  --max-answer-words {args.max_answer_words} dropped {n_long} "
              f"({100 * n_long / len(seen_lengths):.1f}%)")
    words = [len(r["q_str"].split()) for r in train]
    ans = [len(r["a_str"].split()) for r in train]
    print(f"Wrote {len(train)} train + {len(test)} test"
          + (f" + {len(sft)} sft" if sft else "") + f" rows to {out_dir}/")
    print(f"  prompt words: median {sorted(words)[len(words) // 2]}, max {max(words)}")
    print(f"  answer words: median {sorted(ans)[len(ans) // 2]}, max {max(ans)}")
    print(f"  example prompt:\n{train[0]['q_str'][:300]}\n  answer: {train[0]['a_str']!r}")


if __name__ == "__main__":
    main()

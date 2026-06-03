"""Fetch GSM8K and write local ``gsm8k_{train,test}.json`` for CoT distillation.

Downloads ``openai/gsm8k`` (``main`` config) and writes chain-of-thought
prompt/answer JSON in the flat ``{prompt: answer}`` schema consumed by
:func:`utils.load_prompt_answer_json`.  Each ``answer`` is the full GSM8K
solution **string** (the reasoning trace plus the trailing ``#### <number>``),
stored verbatim so the distillation path can train on CoT traces.

Usage (run from the ``src/`` directory so the ``utils`` package is importable)::

    python -m utils.make_gsm8k_dataset --num-train 100 --num-test 100

The output files land in the project's datasets dir (see
:func:`utils.default_datasets_dir`), i.e. ``datasets/gsm8k_train.json`` and
``datasets/gsm8k_test.json``, which the trainer loads via ``--dataset gsm8k``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from typing import Dict

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from utils import default_datasets_dir

DATASET_PREFIX = "gsm8k"


def _quiet_hf_logging() -> None:
    """Silence verbose HuggingFace ``datasets``/``transformers`` logging."""
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    try:
        from datasets.utils.logging import set_verbosity_error

        set_verbosity_error()
    except Exception:
        pass


def build_prompt_answer_dict(split, limit: int) -> Dict[str, str]:
    """Take the first ``limit`` rows as ``{instruction + question: answer_str}``."""
    out: Dict[str, str] = {}
    n = min(limit, len(split)) if limit is not None else len(split)
    for i in range(n):
        row = split[i]
        prompt = f"Solve the math problem step by step. {str(row['question'])}"
        answer = str(row["answer"])  # full CoT trace incl. trailing "#### <number>"
        # Strip GSM8K inline calculator annotations like ``<<16-3-4=9>>``.  The
        # result number sits OUTSIDE the brackets (``= <<16-3-4=9>>9`` -> ``= 9``),
        # so removal keeps the natural trace and the trailing ``#### <number>``
        # intact.  The instruct teacher never emits ``<<``, so these would only
        # pollute the teacher-forced distillation target.
        answer = re.sub(r"<<[^>]*>>", "", answer)
        answer = re.sub(r"  +", " ", answer)  # collapse double spaces; keep newlines
        out[prompt] = answer
    return out


def write_json(path: str, data: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch GSM8K into local gsm8k_train.json / gsm8k_test.json (CoT string answers).",
    )
    parser.add_argument("--num-train", type=int, default=100, dest="num_train",
                        help="Number of training examples to write (default 100).")
    parser.add_argument("--num-test", type=int, default=100, dest="num_test",
                        help="Number of test examples to write (default 100).")
    parser.add_argument("--datasets-dir", type=str, default=None, dest="datasets_dir",
                        help="Output dir (default: project datasets dir).")
    parser.add_argument("--seed", type=int, default=None,
                        help="If set, shuffle each split with this seed before slicing "
                             "(default: deterministic first-N, no shuffle).")
    args = parser.parse_args()

    if args.num_train < 0 or args.num_test < 0:
        raise SystemExit("--num-train and --num-test must be >= 0")

    _quiet_hf_logging()

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "GSM8K requires the `datasets` package. Install with: pip install datasets",
        ) from e

    datasets_dir = os.path.abspath(args.datasets_dir) if args.datasets_dir else default_datasets_dir()

    print("Loading openai/gsm8k (main)...")
    ds = load_dataset("openai/gsm8k", "main")
    train_split = ds["train"]
    test_split = ds["test"]

    if args.seed is not None:
        train_split = train_split.shuffle(seed=args.seed)
        test_split = test_split.shuffle(seed=args.seed)

    train_data = build_prompt_answer_dict(train_split, args.num_train)
    test_data = build_prompt_answer_dict(test_split, args.num_test)

    train_path = os.path.join(datasets_dir, f"{DATASET_PREFIX}_train.json")
    test_path = os.path.join(datasets_dir, f"{DATASET_PREFIX}_test.json")
    write_json(train_path, train_data)
    write_json(test_path, test_data)

    print(f"Wrote {len(train_data)} train examples -> {train_path}")
    print(f"Wrote {len(test_data)} test examples  -> {test_path}")
    print(f"Train with: --dataset {DATASET_PREFIX}")


if __name__ == "__main__":
    main()

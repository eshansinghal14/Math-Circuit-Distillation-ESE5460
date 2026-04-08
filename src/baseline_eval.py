"""Baseline accuracy on the 2-digit addition test set (print only; no files written).

Same protocol as training logs: left padding, greedy decode, integer after ``=``.

From ``src/``::

  # Default: Llama-3.2-1B and Meta-Llama-3-8B (HF); --dataset is required
  python baseline_eval.py --dataset 2d_add

  # Local save_pretrained folder
  python baseline_eval.py --dataset 2d_add /path/to/student_best

  # Any HF id and/or paths (each arg is one model)
  python baseline_eval.py --dataset 2d_add meta-llama/Llama-3.2-1B
  python baseline_eval.py --dataset 2d_add ./student_best meta-llama/Meta-Llama-3-8B
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from standard_distillation import evaluate  # noqa: E402
from utils import EVAL_MAX_NEW_TOKENS, load_model, resolve_test_path  # noqa: E402

_DEFAULT_MODELS = (
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Meta-Llama-3-8B",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline eval on addition test JSON (prints only)",
    )
    parser.add_argument(
        "models",
        nargs="*",
        metavar="PATH_OR_HF_ID",
        help="Local save_pretrained directory and/or Hugging Face model id "
        "(omit for default 1B + 8B)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        metavar="PREFIX",
        help="e.g. 2d_add -> <PREFIX>_test_20.json (required)",
    )
    parser.add_argument(
        "--datasets-dir",
        default=None,
        help="Directory containing *_test_20.json (default: repo datasets/)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=EVAL_MAX_NEW_TOKENS,
        help=f"Greedy decode cap (default: {EVAL_MAX_NEW_TOKENS})",
    )
    args = parser.parse_args()

    try:
        test_path, _ = resolve_test_path(
            dataset=args.dataset,
            datasets_dir=args.datasets_dir,
        )
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e

    specs = [os.path.expanduser(s) for s in args.models] if args.models else list(_DEFAULT_MODELS)

    if not os.path.isfile(test_path):
        raise SystemExit(f"Test file not found: {test_path}")

    with open(test_path, "r") as f:
        raw_test = json.load(f)
    n_items = len(raw_test) if isinstance(raw_test, dict) else len(raw_test)

    print("=" * 60)
    print("Baseline evaluation")
    print("=" * 60)
    print(f"  Test:            {test_path}")
    print(f"  Examples:        {n_items}")
    print(f"  max_new_tokens:  {args.max_new_tokens}")
    print(f"  batch_size:      {args.batch_size}")
    print("=" * 60)

    for spec in specs:
        print(f"\nLoading: {spec}")
        model, tokenizer = load_model(spec)
        model.eval()

        acc = evaluate(
            model,
            tokenizer,
            test_path,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"  Accuracy: {acc:.4f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()

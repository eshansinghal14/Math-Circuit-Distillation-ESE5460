"""Load a saved student checkpoint and score the math test set.

Uses :func:`utils.evaluate` (left padding, greedy decode; default ``max_new_tokens`` from utils).

Examples:
  cd src
  python eval_trained_student.py --dataset 2d_add --checkpoint ../results/student_model

  # Colab / Drive:
  python eval_trained_student.py \\
    --dataset 2d_add \\
    --checkpoint "/path/to/results/student_model"
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow `python eval_trained_student.py` from repo root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from utils import EVAL_MAX_NEW_TOKENS, evaluate, resolve_test_path  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Eval a HuggingFace-saved student on 2d_add test JSON")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Folder from save_pretrained (e.g. .../your-save-dir/student_model)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        metavar="PREFIX",
        help="e.g. 2d_add -> datasets/<PREFIX>_test_20.json (required)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=EVAL_MAX_NEW_TOKENS,
        help="Greedy decode cap after prompt (default: utils.EVAL_MAX_NEW_TOKENS)",
    )
    args = parser.parse_args()
    try:
        test_path, _ = resolve_test_path(dataset=args.dataset)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e

    ckpt = os.path.expanduser(args.checkpoint)
    if not os.path.isdir(ckpt):
        raise SystemExit(f"Not a directory: {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=dtype)
    model.to(device)
    model.eval()

    acc = evaluate(
        model,
        tokenizer,
        test_path,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Checkpoint: {ckpt}")
    print(f"Test file:  {test_path}")
    print(f"Accuracy:   {acc:.4f}")


if __name__ == "__main__":
    main()

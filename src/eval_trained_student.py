"""Load a saved standard-KL student checkpoint and score the addition test set.

Uses the same decoding/eval as standard_distillation.py (left padding, greedy, max_new_tokens=5).

Examples:
  cd src
  python eval_trained_student.py --checkpoint ../results/standard-kl/student_best

  # Colab / Drive:
  python eval_trained_student.py \\
    --checkpoint "/content/drive/MyDrive/Math Circuit Distillation (ESE 5460)/results/standard-kl/student_best"
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

from standard_distillation import evaluate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Eval a HuggingFace-saved student on 2d_add test JSON")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Folder from save_pretrained (e.g. .../standard-kl/student_best or student_final)",
    )
    parser.add_argument(
        "--test-path",
        default=os.path.join(_SCRIPT_DIR, "..", "datasets", "2d_add_test_20.json"),
        help="Test JSON (dict prompt->int or list of q_str/a_str)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    ckpt = os.path.expanduser(args.checkpoint)
    if not os.path.isdir(ckpt):
        raise SystemExit(f"Not a directory: {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=dtype)
    model.to(device)
    model.eval()

    test_path = os.path.abspath(os.path.expanduser(args.test_path))
    acc = evaluate(model, tokenizer, test_path, batch_size=args.batch_size)
    print(f"Checkpoint: {ckpt}")
    print(f"Test file:  {test_path}")
    print(f"Accuracy:   {acc:.4f}")


if __name__ == "__main__":
    main()

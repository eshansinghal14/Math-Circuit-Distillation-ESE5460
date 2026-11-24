"""Test accuracy of models on datasets, with the trainers' eval. Usage:
    python -u -m experiments.baseline_acc --models teacher=<path> student=<path> --datasets mix3 mix4 --limit 500"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import PromptAnswerDataset, eval_model, load_data, load_model, set_bos_mode  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()
    set_bos_mode("on")
    for spec in args.models:
        label, path = spec.split("=", 1)
        model, tok = load_model(path)
        model.eval()
        for ds in args.datasets:
            _, test = load_data(ds, test_limit=args.limit)
            acc = eval_model(model, tok, PromptAnswerDataset(ds, test, tok), ds, 256, None)
            print(f"{label:10s} {ds:8s} {acc:.3f}", flush=True)
        del model


if __name__ == "__main__":
    main()

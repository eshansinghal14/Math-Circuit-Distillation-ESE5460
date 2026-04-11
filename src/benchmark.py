import argparse
import os
from typing import List, Optional

from utils import (
    EVAL_MAX_NEW_TOKENS,
    default_datasets_dir,
    load_model,
    parse_answer,
    test_model,
)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a model on a dataset JSON and print accuracy (saves generations under results/model_outputs/<model>/).",
    )
    p.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="HuggingFace model identifier (e.g. meta-llama/Llama-3.2-1B)",
    )
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        metavar="FILE",
        help="Dataset JSON filename in datasets/ (e.g. 3d_add_test.json)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for generation (default: 50)",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=EVAL_MAX_NEW_TOKENS,
        help=f"Greedy decode length after prompt (default: {EVAL_MAX_NEW_TOKENS}, from utils)",
    )
    return p.parse_args(argv)


def _safe_model_dir(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    raw = (args.dataset or "").strip()
    if not raw:
        raise SystemExit("ERROR: --dataset must be a non-empty JSON filename")

    if "/" in raw or "\\" in raw:
        raise SystemExit(
            "ERROR: --dataset must be a filename only, not a path "
            "(file must live under datasets/, e.g. 3d_add_test.json)",
        )

    dataset_file = raw
    if not dataset_file.endswith(".json"):
        raise SystemExit("ERROR: --dataset must be a .json filename (e.g. 3d_add_test.json)")

    datasets_dir = default_datasets_dir()
    dataset_path = os.path.join(datasets_dir, dataset_file)
    if not os.path.isfile(dataset_path):
        raise SystemExit(f"Dataset not found: {dataset_path}")

    model_name = args.model_name
    model, tokenizer = load_model(model_name)

    repo_root = os.path.dirname(datasets_dir)
    results_dir = os.path.join(
        repo_root, "results", "model_outputs", _safe_model_dir(model_name),
    )
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, dataset_file)

    results = test_model(
        model,
        tokenizer,
        dataset_path,
        results_path,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    correct = sum(
        1 for r in results if parse_answer(r["response"]) == int(r["answer"])
    )
    n = len(results)
    acc = correct / n if n else 0.0

    print(f"Model:   {model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Results: {results_path}")
    print(f"Accuracy: {acc:.4f} ({correct}/{n})")


if __name__ == "__main__":
    main()

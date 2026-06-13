import argparse
import sys
from typing import List, Optional

from utils import PromptAnswerDataset, eval_model, load_data, load_model


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a model on a local math dataset.")
    p.add_argument("--model-name", required=True, help="HuggingFace model id or local path")
    p.add_argument("--dataset", required=True, help="Dataset name under datasets/ (e.g. 22_add)")
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--test-limit", type=int, default=None, help="Limit number of test examples")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    model, tokenizer = load_model(args.model_name)
    _, test_data = load_data(args.dataset, test_limit=args.test_limit)
    test_dataset = PromptAnswerDataset(args.dataset, test_data, tokenizer)
    acc = eval_model(model, tokenizer, test_dataset, args.dataset, args.batch_size, args.max_new_tokens)
    n = len(test_dataset)
    correct = round(acc * n)
    print(f"Model:    {args.model_name}")
    print(f"Dataset:  {args.dataset}")
    print(f"Accuracy: {acc:.4f} ({correct}/{n})")


if __name__ == "__main__":
    main()

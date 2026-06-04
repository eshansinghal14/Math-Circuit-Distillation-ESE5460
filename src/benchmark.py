import argparse
import os
import sys
from typing import List, Optional

from utils import (
    EVAL_MAX_NEW_TOKENS,
    FEWSHOT_POOL_SIZE,
    LLAMA_1B_MODEL_NAME,
    TRAIN_SPLIT_SIZE,
    default_datasets_dir,
    load_model,
    load_prompt_answer_json,
    parse_answer,
    resolve_train_test_paths,
    run_hf_benchmark,
    test_model,
)

_HF_DATASET_ALIASES = frozenset({"gsm8k", "svamp"})


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run a model on a local math JSON (datasets/) or on GSM8K / SVAMP via load_dataset. "
            "Saves generations under results/model_outputs/<model>/."
        ),
    )
    p.add_argument(
        "--model-name",
        type=str,
        required=True,
        help=f"HuggingFace model identifier (e.g. {LLAMA_1B_MODEL_NAME})",
    )
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        metavar="FILE_OR_NAME",
        help=(
            "Local JSON filename in datasets/ (e.g. 3d_add_test.json), or gsm8k / svamp "
            "(Hugging Face; no local JSON required)."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for generation (default: 50; use smaller for long HF generations)",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        metavar="K",
        help=(
            "Greedy decode length after prompt. Omitted: 256 for gsm8k/svamp, "
            f"{EVAL_MAX_NEW_TOKENS} for local JSON (short '='-style answers)."
        ),
    )
    p.add_argument(
        "--max-problems",
        type=int,
        default=None,
        metavar="N",
        help="HF only: evaluate at most N examples (default: full split)",
    )
    p.add_argument(
        "--num-fewshot",
        type=int,
        default=0,
        metavar="N",
        dest="num_fewshot",
        help="Number of few-shot examples prepended to each prompt (drawn from last 473 gsm8k train examples).",
    )
    return p.parse_args(argv)


def _safe_model_dir(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    raw = (args.dataset or "").strip()
    if not raw:
        raise SystemExit("ERROR: --dataset must be non-empty")

    datasets_dir = default_datasets_dir()
    repo_root = os.path.dirname(datasets_dir)
    results_dir = os.path.join(
        repo_root, "results", "model_outputs", _safe_model_dir(args.model_name),
    )
    os.makedirs(results_dir, exist_ok=True)

    model_name = args.model_name
    model, tokenizer = load_model(model_name)

    key = raw.lower()
    if key in _HF_DATASET_ALIASES:
        max_new = args.max_new_tokens if args.max_new_tokens is not None else 256
        if max_new <= max(8, EVAL_MAX_NEW_TOKENS):
            print(
                "WARN: gsm8k/svamp usually need --max-new-tokens 128–512 "
                f"(got {max_new}).",
                file=sys.stderr,
            )

        fewshot_pool = None
        if args.num_fewshot > 0:
            train_path, _, _ = resolve_train_test_paths(dataset="gsm8k", datasets_dir=None)
            all_train = load_prompt_answer_json(train_path)
            all_train_items = list(all_train.items())
            fewshot_pool = dict(all_train_items[TRAIN_SPLIT_SIZE:TRAIN_SPLIT_SIZE + FEWSHOT_POOL_SIZE])
            print(f"Using {args.num_fewshot}-shot prompts (random per example) from fewshot pool ({len(fewshot_pool)} examples).")

        results_name = f"{key}_results.json"
        results_path = os.path.join(results_dir, results_name)
        results, acc = run_hf_benchmark(
            model,
            tokenizer,
            key,
            results_path,
            batch_size=args.batch_size,
            max_new_tokens=max_new,
            limit=args.max_problems,
            log=True,
            fewshot_pool=fewshot_pool,
            num_fewshot=args.num_fewshot,
        )
        n = len(results)
        correct = int(round(acc * n)) if n else 0
        print(f"Model:   {model_name}")
        print(f"Dataset: Hugging Face ({key}; load_dataset, no local JSON)")
        print(f"Results: {results_path}")
        print(f"Accuracy: {acc:.4f} ({correct}/{n})")
        return

    if "/" in raw or "\\" in raw:
        raise SystemExit(
            "ERROR: --dataset must be a filename only, not a path "
            "(file must live under datasets/, e.g. 3d_add_test.json)",
        )

    dataset_file = raw
    if not dataset_file.endswith(".json"):
        raise SystemExit(
            "ERROR: --dataset must be a .json filename (e.g. 3d_add_test.json) "
            "or gsm8k / svamp",
        )

    dataset_path = os.path.join(datasets_dir, dataset_file)
    if not os.path.isfile(dataset_path):
        raise SystemExit(f"Dataset not found: {dataset_path}")

    max_new = args.max_new_tokens if args.max_new_tokens is not None else EVAL_MAX_NEW_TOKENS
    results_path = os.path.join(results_dir, dataset_file)
    results = test_model(
        model,
        tokenizer,
        dataset_path,
        results_path,
        batch_size=args.batch_size,
        max_new_tokens=max_new,
    )

    correct = sum(
        1
        for r in results
        if (p := parse_answer(r["response"])) is not None and p == int(r["answer"])
    )
    n = len(results)
    acc = correct / n if n else 0.0

    print(f"Model:   {model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Results: {results_path}")
    print(f"Accuracy: {acc:.4f} ({correct}/{n})")


if __name__ == "__main__":
    main()

import argparse
from typing import Tuple

from transformers import AutoTokenizer

from utils import HF_READ_TOKEN, dataset_all_json_path, generate_math_dataset


def _parse_digits(raw: str) -> Tuple[int, int]:
    text = (raw or "").strip()
    if "," in text:
        parts = [p.strip() for p in text.split(",")]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("--digits must be INT or LEFT,RIGHT")
        try:
            a, b = int(parts[0]), int(parts[1])
        except ValueError as e:
            raise argparse.ArgumentTypeError("--digits must contain integers") from e
        return (a, b)
    try:
        d = int(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError("--digits must be INT or LEFT,RIGHT") from e
    return (d, d)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate dataset JSONs under datasets/ with optional train/test split.",
    )
    p.add_argument(
        "--name",
        type=str,
        required=True,
        help="Dataset prefix (writes datasets/<name>_all.json and optional split files).",
    )
    p.add_argument(
        "--dataset-type",
        choices=["arithmetic", "mod", "greater_than", "linear_eq"],
        default="arithmetic",
    )
    p.add_argument(
        "--digits",
        type=_parse_digits,
        default=(2, 2),
        help="Operand digits as INT or LEFT,RIGHT (e.g. 2 or 3,2).",
    )
    p.add_argument(
        "--operation",
        type=str,
        default="+",
        help="Operator for arithmetic/mod: +, -, *, ×, /, // (mod supports +,-,*,×).",
    )
    p.add_argument(
        "--pair-mode",
        choices=["grid", "2d1d_mult"],
        default="grid",
        help="Only for arithmetic datasets.",
    )
    p.add_argument(
        "--modulus-digits",
        type=int,
        default=2,
        help="For mod datasets: z sampled from [1, 10^d - 1].",
    )
    p.add_argument(
        "--variable-name",
        type=str,
        default="x",
        help="For linear_eq datasets: variable symbol in prompt.",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Optional random subset size from generated full set.",
    )
    p.add_argument(
        "--split-test-frac",
        type=float,
        default=0.2,
        help="If set, writes *_train_<pct>.json and *_test_<pct>.json.",
    )
    p.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffle before split/subsample.",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Tokenizer model id used to build ids field.",
    )
    p.add_argument(
        "--datasets-dir",
        type=str,
        default=None,
        help="Optional custom datasets directory.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, token=HF_READ_TOKEN or None)
    out_path = dataset_all_json_path(args.name, datasets_dir=args.datasets_dir)
    generate_math_dataset(
        out_path,
        tokenizer,
        dataset_type=args.dataset_type,
        operand_digits=args.digits,
        operation=args.operation,
        pair_mode=args.pair_mode,
        modulus_digits=args.modulus_digits,
        variable_name=args.variable_name,
        shuffle=not args.no_shuffle,
        samples=args.samples,
        split_test_frac=args.split_test_frac,
        datasets_dir=args.datasets_dir,
    )
    print(f"Wrote dataset(s) for prefix '{args.name}' at: {out_path}")


if __name__ == "__main__":
    main()

import argparse
from typing import List

from transformers import AutoTokenizer

from utils import HF_READ_TOKEN, dataset_all_json_path, generate_math_dataset, normalize_op_patterns


def _parse_digits(raw: str) -> List[int]:
    text = (raw or "").strip()
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(
            "--digits must list at least two widths (e.g. 2,2 or 2,2,3)",
        )
    try:
        return [int(p) for p in parts]
    except ValueError as e:
        raise argparse.ArgumentTypeError("--digits must contain integers") from e


def _parse_ops_arg(raw: str, num_gaps: int) -> List[List[str]]:
    """``;`` separates rows (allowed orderings); ``,`` separates operators within a row."""
    text = (raw or "").strip()
    if not text:
        patterns: List[List[str]] = [["+"]]
    else:
        rows = [r.strip() for r in text.split(";") if r.strip()]
        patterns = [[p.strip() for p in r.split(",") if p.strip()] for r in rows]
    return normalize_op_patterns(patterns, num_gaps)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate chained math dataset JSONs under datasets/ with optional train/test split.",
    )
    p.add_argument(
        "--name",
        type=str,
        required=True,
        help="Dataset prefix (writes datasets/<name>_all.json and optional split files).",
    )
    p.add_argument(
        "--digits",
        type=_parse_digits,
        default="2,2",
        help="Comma-separated decimal widths per operand (at least two), e.g. 2,2 or 2,2,3.",
    )
    p.add_argument(
        "--ops",
        type=_parse_ops,
        default=["+"],
        help="Comma-separated pool of + and/or * (or ×); PEMDAS. One op drawn at random between operands.",
    )
    p.add_argument(
        "--mod-n",
        type=int,
        default=None,
        metavar="N",
        help="Append 'mod N' before '= '; answer is (PEMDAS value) %% N.",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of random unique problems to generate (no full Cartesian product). "
        "Omit to enumerate the entire grid (only feasible for small digit widths).",
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
    num_gaps = len(args.digits) - 1
    op_patterns = _parse_ops_arg(args.ops, num_gaps)
    generate_math_dataset(
        out_path,
        tokenizer,
        digits=args.digits,
        operations=op_patterns,
        mod_n=args.mod_n,
        shuffle=not args.no_shuffle,
        samples=args.samples,
        split_test_frac=args.split_test_frac,
        datasets_dir=args.datasets_dir,
    )
    print(f"Wrote dataset(s) for prefix '{args.name}' at: {out_path}")


if __name__ == "__main__":
    main()

import argparse
import itertools
import json
import os
import random
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from new_utils import DIR_ROOT


def generate_math_dataset(
    dataset_name: str,
    digits: List[int],
    operations: List[str],
    *,
    spaces: bool = False,
    shuffle: bool = True,
    samples: Optional[int] = None,
    test_split: Optional[float] = None,
    prompt: Optional[str] = None,
) -> None:
    if len(digits) < 2:
        raise ValueError("digits must have at least 2 elements.")
    if not operations:
        raise ValueError("operations must be non-empty.")

    prefix = prompt or ""

    _OP_FN = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}

    def build_pair(nums: List[int], op: str):
        if spaces:
            parts = [str(nums[0])] + [f" {op} {n}" for n in nums[1:]]
            expr = "".join(parts)
            q_str = prefix + expr + " = "
        else:
            parts = [str(nums[0])] + [f"{op}{n}" for n in nums[1:]]
            expr = "".join(parts)
            q_str = prefix + expr + "="
        fn = _OP_FN.get(op)
        if fn is not None:
            result = nums[0]
            for n in nums[1:]:
                result = fn(result, n)
            a_str = str(result)
        else:
            a_str = str(eval(expr.replace(" ", "")))
        return q_str, a_str

    pairs: List[tuple] = []

    all_pairs: List[tuple] = []
    for nums in itertools.product(*[range(10**d) for d in digits]):
        for op in operations:
            all_pairs.append(build_pair(list(nums), op))

    if samples is None:
        pairs = all_pairs
        if shuffle:
            random.shuffle(pairs)
    else:
        if samples > len(all_pairs):
            raise ValueError(
                f"Requested {samples} samples but only {len(all_pairs)} unique problems exist."
            )
        pairs = random.sample(all_pairs, samples)

    rows = [{"q_str": q, "a_str": a} for q, a in pairs]

    out_dir = os.path.join(DIR_ROOT, "datasets", dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    def _write(path: str, data: list) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    if test_split is not None:
        if not (0.0 < test_split <= 1.0):
            raise ValueError("test_split must be in (0, 1].")
        split_i = int(len(rows) * (1.0 - test_split))
        _write(os.path.join(out_dir, "train.json"), rows[:split_i])
        _write(os.path.join(out_dir, "test.json"), rows[split_i:])
        print(f"Wrote {split_i} train + {len(rows) - split_i} test rows to {out_dir}/")
    else:
        path = os.path.join(out_dir, "data.json")
        _write(path, rows)
        print(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a math equation dataset.")
    parser.add_argument("--dataset-name", required=True, help="Folder name under datasets/")
    parser.add_argument("--digits", nargs="+", type=int, required=True,
                        help="Digit width per operand, e.g. --digits 2 1")
    parser.add_argument("--operations", nargs="+", required=True,
                        help="Operators to use, e.g. --operations + '*'")
    parser.add_argument("--spaces", action="store_true", default=False,
                        help="Add spaces around operators (default: no spaces)")
    parser.add_argument("--no-shuffle", action="store_false", dest="shuffle",
                        help="Disable shuffling")
    parser.add_argument("--samples", type=int, default=None,
                        help="Random sample count (default: enumerate all)")
    parser.add_argument("--test-split", type=float, default=None,
                        help="Fraction for test set, e.g. 0.5")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Constant string prepended to every equation")
    args = parser.parse_args()

    generate_math_dataset(
        dataset_name=args.dataset_name,
        digits=args.digits,
        operations=args.operations,
        spaces=args.spaces,
        shuffle=args.shuffle,
        samples=args.samples,
        test_split=args.test_split,
        prompt=args.prompt,
    )

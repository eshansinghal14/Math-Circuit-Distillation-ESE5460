"""Multi-step arithmetic answered as one number, for testing graph distillation.

Every prompt is a short expression ending in "=", e.g. "47+38-25=", and the
answer is its value, a single Llama-3 token (0-999). The whole computation then
happens before one predicted token, so one attribution graph covers it.

Families (operands drawn uniformly, problems kept only when every intermediate
and the result lie in 0..999):

    mix3     a o b o c        two-digit operands, each o in {+, -}
    mix4     a o b o c o d    two-digit operands, each o in {+, -}
    muladd   a*b o c          a two-digit, b one-digit (2-9), c two-digit, o in {+, -}

    python generate_multistep_dataset.py --family mix3 --train 5000 --test 1000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import DIR_ROOT  # noqa: E402


def sample(family: str, rng: random.Random):
    if family in ("mix3", "mix4"):
        k = 3 if family == "mix3" else 4
        nums = [rng.randint(10, 99) for _ in range(k)]
        ops = [rng.choice("+-") for _ in range(k - 1)]
        val, expr = nums[0], str(nums[0])
        for o, n in zip(ops, nums[1:]):
            val = val + n if o == "+" else val - n
            if not 0 <= val <= 999:
                return None
            expr += o + str(n)
        return expr + "=", str(val)
    if family == "muladd":
        a, b, c = rng.randint(10, 99), rng.randint(2, 9), rng.randint(10, 99)
        o = rng.choice("+-")
        val = a * b + c if o == "+" else a * b - c
        if not 0 <= val <= 999:
            return None
        return f"{a}*{b}{o}{c}=", str(val)
    raise ValueError(family)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--family", required=True, choices=["mix3", "mix4", "muladd"])
    ap.add_argument("--train", type=int, default=5000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    seen, rows = set(), []
    while len(rows) < args.train + args.test:
        s = sample(args.family, rng)
        if s and s[0] not in seen:
            seen.add(s[0])
            rows.append({"q_str": s[0], "a_str": s[1]})
    out = os.path.join(DIR_ROOT, "datasets", args.family)
    os.makedirs(out, exist_ok=True)
    for name, part in (("train", rows[: args.train]), ("test", rows[args.train:]), ("all", rows)):
        with open(os.path.join(out, f"{name}.json"), "w") as f:
            json.dump(part, f)
    print(f"{args.family}: {args.train} train, {args.test} test -> {out}; e.g. {rows[:3]}")


if __name__ == "__main__":
    main()

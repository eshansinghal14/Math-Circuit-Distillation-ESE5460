"""Multi-step arithmetic answered as one number, for testing graph distillation.

Every prompt is a short expression ending in "=", e.g. "47+38-25=", and the
answer is its value, a single Llama-3 token (0-999). The whole computation then
happens before one predicted token, so one attribution graph covers it.

Families (operands drawn uniformly, problems kept only when every intermediate
and the result lie in 0..999):

    mix3     a o b o c        two-digit operands, each o in {+, -}
    mix4     a o b o c o d    two-digit operands, each o in {+, -}
    muladd   a*b o c          a two-digit, b one-digit (2-9), c two-digit, o in {+, -}
    bind     "q=47, f=12, x=85, b=30, m=66. f+x-q="   five two-digit variables under random
             distinct letters in random order; the expression is mix3 over three of
             them, so two are distractors. Which values feed the answer has to be
             looked up (routing), not just read off the expression.
    chain    "k=p, w=31, p=t, d=w, t=58, g=d. k="   three variable chains of equal length
             (2 or 3 hops) ending in distinct two-digit numbers, definitions shuffled;
             the answer is the number the queried chain ends in. No arithmetic: a
             wrong answer that is another chain's number is a routing error by
             construction.

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
    if family == "bind":
        names = rng.sample("abcdefghijklmnopqrstuvwxyz", 5)
        vals = {v: rng.randint(10, 99) for v in names}
        used = rng.sample(names, 3)
        ops = [rng.choice("+-") for _ in range(2)]
        val, expr = vals[used[0]], used[0]
        for o, v in zip(ops, used[1:]):
            val = val + vals[v] if o == "+" else val - vals[v]
            if not 0 <= val <= 999:
                return None
            expr += o + v
        return ", ".join(f"{v}={vals[v]}" for v in names) + f". {expr}=", str(val)
    if family == "chain":
        hops = rng.choice((2, 3))
        names = rng.sample("abcdefghijklmnopqrstuvwxyz", 3 * hops)
        chains = [names[i * hops:(i + 1) * hops] for i in range(3)]
        nums = rng.sample(range(10, 100), 3)
        defs = [f"{a}={b}" for c in chains for a, b in zip(c, c[1:])]
        defs += [f"{c[-1]}={n}" for c, n in zip(chains, nums)]
        rng.shuffle(defs)
        return ", ".join(defs) + f". {chains[0][0]}=", str(nums[0])
    raise ValueError(family)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--family", required=True, choices=["mix3", "mix4", "muladd", "bind", "chain"])
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

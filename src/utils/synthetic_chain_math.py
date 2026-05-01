import itertools
import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

from .config import DATASET_TEST_SUFFIX, DATASET_TRAIN_SUFFIX
from .dataset_paths import _resolve_dataset_output_path


def normalize_op_patterns(
    operations: Sequence[Sequence[str]],
    num_gaps: int,
) -> List[List[str]]:
    """Validate 2D ``operations``: each row is one full operator sequence between operands.

    Each row must have length ``num_gaps`` (``len(digits) - 1``), or length 1 to broadcast that
    operator to every gap. Only ``+``, ``*``, and ``x`` are allowed.
    """
    if num_gaps < 1:
        raise ValueError("num_gaps must be >= 1.")
    if not operations:
        raise ValueError("operations must be a non-empty list of operator rows.")
    allowed = {"+", "*", "x", "×"}
    out: List[List[str]] = []
    for row in operations:
        ops = [o.strip() for o in row if o and str(o).strip()]
        if len(ops) == 1 and num_gaps > 1:
            ops = [ops[0]] * num_gaps
        elif len(ops) != num_gaps:
            raise ValueError(
                f"each operator row must have length {num_gaps} "
                f"(or length 1 to broadcast); got {len(ops)} in {list(row)!r}",
            )
        for o in ops:
            if o not in allowed:
                raise ValueError(
                    f"operations must use only '+' or '*' (or 'x'/'×'); got {o!r}.",
                )
        out.append(ops)
    return out


def _eval_pemdas_plus_mul(numbers: Sequence[int], op_syms: Sequence[str]) -> int:
    """Evaluate with multiplication before addition (PEMDAS for ``+`` and ``*`` only)."""
    values = list(numbers)
    ops: List[str] = []
    for o in op_syms:
        s = o.strip()
        if s in {"×", "x"}:
            s = "*"
        ops.append(s)

    i = 0
    while i < len(ops):
        if ops[i] == "*":
            values[i] = values[i] * values[i + 1]
            del values[i + 1]
            del ops[i]
        else:
            i += 1
    return sum(values)


def _format_chain_prompt(
    nums: Sequence[int],
    op_syms: Sequence[str],
    mod_n: Optional[int],
    *,
    spaces: bool = True,
) -> str:
    parts: List[str] = [str(nums[0])]
    for i, sym in enumerate(op_syms):
        if spaces:
            parts.append(f" {sym} {nums[i + 1]}")
        else:
            parts.append(f"{sym}{nums[i + 1]}")
    body = "".join(parts)
    if mod_n is not None:
        if spaces:
            return f"{body} mod {mod_n} = "
        return f"{body}mod{mod_n}="
    if spaces:
        return f"{body} = "
    return f"{body}="


def _iter_chain_pairs(
    digits: Sequence[int],
    operations: Sequence[Sequence[str]],
    mod_n: Optional[int],
    *,
    spaces: bool = True,
) -> List[Tuple[str, int]]:
    """Cartesian product over operand ranges; each row picks one operator pattern at random.

    **PEMDAS:** ``*`` / ``x`` / ``×`` before ``+``. ``operations`` is a 2D list: each inner list
    is one allowed ordering of ``len(digits)-1`` operators; one ordering is chosen uniformly at
    random per problem. If ``mod_n`` is set, the gold answer is ``(value) % mod_n`` and the
    prompt ends with ``... mod n = ``.
    """
    if len(digits) < 2:
        raise ValueError("digits must list at least two operand widths.")
    if any(d < 1 for d in digits):
        raise ValueError("each entry in digits must be >= 1.")
    num_gaps = len(digits) - 1
    op_patterns = normalize_op_patterns(operations, num_gaps)
    if mod_n is not None and mod_n < 1:
        raise ValueError("mod_n must be >= 1 when set.")

    ranges = [range(10**int(d)) for d in digits]
    pairs: List[Tuple[str, int]] = []
    for nums in itertools.product(*ranges):
        nlist = list(nums)
        op_syms = list(random.choice(op_patterns))
        inner = _eval_pemdas_plus_mul(nlist, op_syms)
        ans = inner % mod_n if mod_n is not None else inner
        pairs.append((_format_chain_prompt(nlist, op_syms, mod_n, spaces=spaces), ans))
    return pairs


def _sample_chain_pairs(
    digits: Sequence[int],
    operations: Sequence[Sequence[str]],
    mod_n: Optional[int],
    samples: int,
    shuffle: bool,
    *,
    spaces: bool = True,
) -> List[Tuple[str, int]]:
    """Draw ``samples`` unique random problems without enumerating the full Cartesian product."""
    if samples < 1:
        raise ValueError("samples must be >= 1.")
    if len(digits) < 2:
        raise ValueError("digits must list at least two operand widths.")
    if any(d < 1 for d in digits):
        raise ValueError("each entry in digits must be >= 1.")
    num_gaps = len(digits) - 1
    op_patterns = normalize_op_patterns(operations, num_gaps)
    if mod_n is not None and mod_n < 1:
        raise ValueError("mod_n must be >= 1 when set.")

    selected: List[Tuple[str, int]] = []
    seen = set()
    cap = max(1_000_000, samples * 10_000)
    for _ in range(cap):
        if len(selected) >= samples:
            break
        nlist = [random.randrange(10**int(d)) for d in digits]
        op_syms = list(random.choice(op_patterns))
        inner = _eval_pemdas_plus_mul(nlist, op_syms)
        ans = inner % mod_n if mod_n is not None else inner
        prompt = _format_chain_prompt(nlist, op_syms, mod_n, spaces=spaces)
        key = (prompt, ans)
        if key in seen:
            continue
        seen.add(key)
        selected.append((prompt, ans))
    if len(selected) < samples:
        raise ValueError(
            f"Could only draw {len(selected)} unique problems (requested {samples}); "
            "the search space may be too small for that count.",
        )
    if shuffle:
        random.shuffle(selected)
    return selected


def generate_math_dataset(
    dataset_fname: str,
    tokenizer,
    *,
    digits: List[int],
    operations: List[List[str]],
    mod_n: Optional[int] = None,
    shuffle: bool = True,
    samples: Optional[int] = None,
    split_test_frac: Optional[float] = None,
    datasets_dir: Optional[str] = None,
    spaces: bool = True,
) -> None:
    """Build a math JSON dataset ``{{q_str, a_str, ids}}`` compatible with the rest of the repo.

    Chained prompts: operand ``i`` uses ``digits[i]`` decimal digits (values ``0 .. 10**digits[i]-1``).
    ``operations`` is a list of rows; each row is one allowed operator ordering (length
    ``len(digits)-1``). For each problem, one row is chosen uniformly at random.
    Evaluation uses **PEMDAS** for ``+`` and ``*`` (or ``x`` / ``×``): multiplication before addition.
    If ``mod_n`` is set, the prompt ends with ``... mod n = `` and the answer is
    ``(PEMDAS value) % mod_n``.
    """
    dataset_fname = _resolve_dataset_output_path(dataset_fname, datasets_dir)
    if samples is not None:
        selected = _sample_chain_pairs(
            digits, operations, mod_n, samples, shuffle, spaces=spaces,
        )
    else:
        selected = _iter_chain_pairs(digits, operations, mod_n, spaces=spaces)
        if shuffle:
            random.shuffle(selected)

    rows: List[Dict] = []
    for prompt, answer in selected:
        q_str = prompt
        a_str = str(answer)
        ids = tokenizer.encode(q_str + a_str, add_special_tokens=False)
        rows.append({"q_str": q_str, "a_str": a_str, "ids": ids})

    out_dir = os.path.dirname(os.path.abspath(dataset_fname))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    def _write(path: str, data: List[Dict]) -> None:
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    _write(dataset_fname, rows)

    if split_test_frac is not None:
        if not (0.0 <= split_test_frac < 1.0):
            raise ValueError("split_test_frac must be in [0, 1).")
        if not dataset_fname.endswith("_all.json"):
            raise ValueError(
                "split_test_frac requires dataset_fname to end with '_all.json' "
                "(e.g. datasets/2d_add_all.json).",
            )
        n = len(rows)
        split_i = int(n * (1.0 - split_test_frac))
        train_path = dataset_fname.replace("_all.json", f"{DATASET_TRAIN_SUFFIX}.json")
        test_path = dataset_fname.replace("_all.json", f"{DATASET_TEST_SUFFIX}.json")
        _write(train_path, rows[:split_i])
        _write(test_path, rows[split_i:])

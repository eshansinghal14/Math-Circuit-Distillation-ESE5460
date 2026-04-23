import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import DATASET_TEST_SUFFIX, DATASET_TRAIN_SUFFIX
from .dataset_paths import _resolve_dataset_output_path


def _rand_expr_number(allow_decimals: bool, max_int: int) -> str:
    if allow_decimals and random.random() < 0.45:
        return f"{random.randint(0, max_int)}.{random.randint(0, 9)}"
    return str(random.randint(0, max_int))


def _safe_eval_bin(a: float, op: str, b: float) -> Optional[float]:
    try:
        if op == "+":
            out = a + b
        elif op == "-":
            out = a - b
        elif op == "*":
            out = a * b
        elif op == "/":
            if abs(b) < 1e-12:
                return None
            out = a / b
        else:
            return None
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _fmt_expr_answer(v: float) -> str:
    if abs(v - round(v)) < 1e-10:
        return str(int(round(v)))
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


def generate_expression_json_dataset(
    dataset_fname: str,
    tokenizer,
    *,
    n_examples: int = 1000,
    max_depth: int = 2,
    allow_decimals: bool = False,
    max_int: int = 999,
    seed: Optional[int] = None,
    ensure_diversity: bool = True,
    allowed_ops: str = "+-*/",
    max_parentheses_pairs: int = 1,
    require_integer_answers: bool = True,
    require_positive_answers: bool = True,
    max_answer_abs: int = 10000,
    min_parentheses_frac: float = 0.30,
    split_test_frac: Optional[float] = None,
    datasets_dir: Optional[str] = None,
) -> None:
    """Generate expression dataset rows: ``{q_str, a_str, ids}``.

    ``q_str`` is compact and ends with ``=`` (no spaces, no words), e.g. ``(19+4)*15=``.
    ``a_str`` stores the computed result as a compact numeric string.
    """
    if n_examples < 1:
        raise ValueError("n_examples must be >= 1")
    if max_depth < 1:
        raise ValueError("max_depth must be >= 1")
    if max_int < 1:
        raise ValueError("max_int must be >= 1")
    if max_parentheses_pairs < 0 or max_parentheses_pairs > 1:
        raise ValueError("max_parentheses_pairs must be 0 or 1")
    if max_answer_abs < 1:
        raise ValueError("max_answer_abs must be >= 1")
    if not (0.0 <= min_parentheses_frac <= 1.0):
        raise ValueError("min_parentheses_frac must be in [0, 1]")
    ops = [op for op in allowed_ops if op in {"+", "-", "*", "/"}]
    if not ops:
        raise ValueError("allowed_ops must include at least one of '+', '-', '*', '/'")

    dataset_fname = _resolve_dataset_output_path(dataset_fname, datasets_dir)
    old_state = random.getstate()
    if seed is not None:
        random.seed(seed)

    seen_q = set()
    rows: List[Dict] = []
    template_cycle = ["two_term", "three_term", "paren_left", "paren_right"]
    paren_count = 0

    def _n() -> str:
        # Mix easy + medium + harder magnitudes (not only 3-digit operands).
        r = random.random()
        if r < 0.50:
            lim = min(max_int, 20)
        elif r < 0.85:
            lim = min(max_int, 99)
        else:
            lim = max_int
        return _rand_expr_number(allow_decimals, lim)

    def _sample_expression(template: str) -> Tuple[str, Optional[float]]:
        # single-step: a+b
        if template == "two_term":
            a, b = _n(), _n()
            op = random.choice(ops)
            expr = f"{a}{op}{b}"
            return expr, _safe_eval_bin(float(a), op, float(b))

        # multi-step, no parentheses: a+b*c
        if template == "three_term":
            a, b, c = _n(), _n(), _n()
            op1, op2 = random.choice(ops), random.choice(ops)
            expr = f"{a}{op1}{b}{op2}{c}"
            try:
                return expr, float(eval(expr))
            except Exception:
                return expr, None

        if max_parentheses_pairs == 0:
            return _sample_expression("three_term")

        # one parenthesis pair max: (a+b)*c
        if template == "paren_left":
            a, b, c = _n(), _n(), _n()
            op1, op2 = random.choice(ops), random.choice(ops)
            expr = f"({a}{op1}{b}){op2}{c}"
            try:
                return expr, float(eval(expr))
            except Exception:
                return expr, None

        # one parenthesis pair max: a*(b+c)
        a, b, c = _n(), _n(), _n()
        op1, op2 = random.choice(ops), random.choice(ops)
        expr = f"{a}{op1}({b}{op2}{c})"
        try:
            return expr, float(eval(expr))
        except Exception:
            return expr, None

    max_attempts = max(20_000, n_examples * 120)
    attempts = 0
    while len(rows) < n_examples and attempts < max_attempts:
        attempts += 1
        if max_depth <= 1:
            tpl = "two_term"
        else:
            candidates = ["two_term", "three_term"]
            if max_parentheses_pairs > 0:
                candidates += ["paren_left", "paren_right"]
            tpl = random.choice(candidates)
            if ensure_diversity and len(rows) < len(template_cycle):
                tpl = template_cycle[len(rows)]
            # Ensure we actually get parenthesized forms.
            target_paren = int(min_parentheses_frac * max(1, n_examples))
            if max_parentheses_pairs > 0 and paren_count < target_paren and random.random() < 0.6:
                tpl = random.choice(["paren_left", "paren_right"])
        e, val = _sample_expression(tpl)
        if val is None:
            continue
        q = f"{e}="
        if q in seen_q:
            continue
        if require_integer_answers and abs(val - round(val)) > 1e-10:
            continue
        if require_positive_answers and val <= 0:
            continue
        if abs(val) > max_answer_abs:
            continue
        a = _fmt_expr_answer(val)
        ids = tokenizer.encode(q + a, add_special_tokens=False)
        rows.append({"q_str": q, "a_str": a, "ids": ids})
        seen_q.add(q)
        if "(" in q:
            paren_count += 1

    if len(rows) < n_examples:
        raise ValueError(f"Could only generate {len(rows)} unique expressions; requested {n_examples}.")

    def _write(path: str, data: List[Dict]) -> None:
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    _write(dataset_fname, rows)

    if split_test_frac is not None:
        if not (0.0 < split_test_frac < 1.0):
            raise ValueError("split_test_frac must be in (0, 1).")
        if not dataset_fname.endswith("_all.json"):
            raise ValueError(
                "split_test_frac requires dataset_fname to end with '_all.json' "
                "(e.g. datasets/svamp_style_expr_all.json).",
            )
        split_i = int(len(rows) * (1.0 - split_test_frac))
        train_path = dataset_fname.replace("_all.json", f"{DATASET_TRAIN_SUFFIX}.json")
        test_path = dataset_fname.replace("_all.json", f"{DATASET_TEST_SUFFIX}.json")
        _write(train_path, rows[:split_i])
        _write(test_path, rows[split_i:])

    if seed is not None:
        random.setstate(old_state)

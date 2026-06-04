import math
import re
from typing import Optional


def _extract_int_after_equals(text: str) -> Optional[int]:
    m = re.search(r"=\s*(\d+)", text)
    return int(m.group(1)) if m else None


def _normalize_numeric_str(s: str) -> str:
    """Canonical string for comparing numeric answers (e.g. ``3.0`` -> ``3``)."""
    s = str(s).strip()
    if not s:
        return s
    try:
        x = float(s)
        if math.isfinite(x) and x == int(x):
            return str(int(x))
        return str(x)
    except ValueError:
        return s


# Matches an integer/decimal that may carry thousands separators (``1,200``)
# but excludes a trailing period (``18.`` -> ``18``). Caller strips commas.
_NUM = r"-?\d[\d,]*(?:\.\d+)?"


def _clean_num(s: str) -> str:
    """Normalize a captured number string (drop thousands commas, then canonicalize)."""
    return _normalize_numeric_str(s.replace(",", ""))


def extract_numeric_answer_from_text(text: str) -> Optional[str]:
    """Return the last numeric value in text, normalized for comparison."""
    text = (text or "").strip()
    if not text:
        return None
    matches = re.findall(_NUM, text)
    if matches:
        return _clean_num(matches[-1])
    return None


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract the number after the GSM8K #### delimiter."""
    text = (text or "").strip()
    m = re.search(r"####\s*(" + _NUM + r")", text)
    if m:
        return _clean_num(m.group(1))
    return None


def parse_answer(resp):
    return _extract_int_after_equals(resp)

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


def extract_numeric_answer_from_text(text: str) -> Optional[str]:
    """Parse a numeric answer from model output (GSM8K/SVAMP-style).

    Prefers phrases like ``The answer is``, ``Answer:``, ``is``; otherwise last number in text.
    Returns normalized string (see :func:`_normalize_numeric_str`) or ``None``.
    """
    text = (text or "").strip()
    if not text:
        return None
    m = re.search(
        r"(?:(?:The\s+answer\s+is)|(?:^|\s)Answer\s*:?|(?:^|\s)is\s*:?)\s*(-?\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    if m:
        return _normalize_numeric_str(m.group(1))
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if matches:
        return _normalize_numeric_str(matches[-1])
    return None


def _gsm8k_gold_answer_str(answer_field: str) -> Optional[str]:
    m = re.search(r"####\s*(\d+)", answer_field)
    return _normalize_numeric_str(m.group(1)) if m else None


def _slice_text_after_reasoning(full_decoded: str) -> str:
    """Use the segment after ``Reasoning:`` for extraction (model repeats prompt)."""
    idx = full_decoded.find("Reasoning:")
    if idx != -1:
        return full_decoded[idx + len("Reasoning:") :].strip()
    return full_decoded.strip()


def parse_answer(resp):
    return _extract_int_after_equals(resp)

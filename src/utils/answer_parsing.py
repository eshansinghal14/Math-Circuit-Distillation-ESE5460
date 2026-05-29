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
    """Parse a numeric answer from model output (GSM8K/SVAMP-style).

    Extraction priority (robust for free-form CoT generations):

    1. ``#### N`` final-answer marker (last occurrence) — what the GSM8K CoT
       target uses, so a well-distilled student emits it.
    2. ``The answer is N`` / ``Answer: N`` phrasing (last occurrence).
    3. Fallback: the last number in the text.

    Commas (``1,200``), a leading ``$``, and a trailing period (``18.``) are
    handled.  Returns a normalized string (see :func:`_normalize_numeric_str`)
    or ``None``.
    """
    text = (text or "").strip()
    if not text:
        return None
    # 1. Explicit GSM8K final-answer marker; take the LAST one.
    hash_matches = re.findall(rf"####\s*({_NUM})", text)
    if hash_matches:
        return _clean_num(hash_matches[-1])
    # 2. "The answer is N" / "Answer: N"; take the LAST one (final answer).
    phrase_matches = re.findall(
        rf"(?:the\s+answer\s+is|answer\s*:?)\s*\$?\s*({_NUM})",
        text,
        re.IGNORECASE,
    )
    if phrase_matches:
        return _clean_num(phrase_matches[-1])
    # 3. Fallback: last number anywhere in the text.
    matches = re.findall(_NUM, text)
    if matches:
        return _clean_num(matches[-1])
    return None


def _gsm8k_gold_answer_str(answer_field: str) -> Optional[str]:
    m = re.search(rf"####\s*({_NUM})", answer_field)
    return _clean_num(m.group(1)) if m else None


def _slice_text_after_reasoning(full_decoded: str) -> str:
    """Use the segment after ``Reasoning:`` for extraction (model repeats prompt)."""
    idx = full_decoded.find("Reasoning:")
    if idx != -1:
        return full_decoded[idx + len("Reasoning:") :].strip()
    return full_decoded.strip()


def parse_answer(resp):
    return _extract_int_after_equals(resp)

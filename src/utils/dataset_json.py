import json
from typing import Dict, List


def json_to_prompt_answer_dict(raw: object) -> Dict[str, int]:
    """Normalize math-dataset JSON to ``{prompt: int answer}``.

    Supports:

    - **Flat dict** ``{"12+34=": 46, ...}`` (string or int values).
    - **List of records** ``[{"q_str": "...", "a_str": "..."}, ...]`` (current repo format).
    """
    if isinstance(raw, dict):
        return {str(k): int(v) for k, v in raw.items()}
    if isinstance(raw, list):
        out: Dict[str, int] = {}
        for i, row in enumerate(raw):
            if not isinstance(row, dict):
                raise TypeError(f"Dataset row {i} must be a dict, got {type(row)}")
            if "q_str" not in row or "a_str" not in row:
                raise ValueError(
                    "List-format rows must include q_str and a_str; "
                    f"row {i} has keys: {list(row.keys())}",
                )
            out[str(row["q_str"])] = int(row["a_str"])
        return out
    raise TypeError(f"Dataset JSON must be a dict or list, got {type(raw)!r}")


def load_prompt_answer_json(path: str) -> Dict[str, int]:
    """Load train/test JSON from disk into ``{prompt: int answer}``."""
    with open(path, "r", encoding="utf-8") as f:
        return json_to_prompt_answer_dict(json.load(f))


def json_to_cot_prompt_answer_dict(raw: object) -> Dict[str, str]:
    """Normalize CoT/GSM8K dataset JSON to ``{prompt: str answer}``.

    Expects a list of ``{"question": str, "answer": str}`` records.
    """
    if not isinstance(raw, list):
        raise TypeError(
            f"CoT dataset JSON must be a list of {{question, answer}} records, "
            f"got {type(raw)!r}"
        )
    out: Dict[str, str] = {}
    for i, row in enumerate(raw):
        if not isinstance(row, dict):
            raise TypeError(f"CoT dataset row {i} must be a dict, got {type(row)}")
        if "question" not in row or "answer" not in row:
            raise ValueError(
                "CoT list-format rows must include 'question' and 'answer'; "
                f"row {i} has keys: {list(row.keys())}"
            )
        out[str(row["question"])] = str(row["answer"])
    return out


def load_cot_json(path: str) -> Dict[str, str]:
    """Load a CoT/GSM8K dataset from disk into ``{prompt: str answer}``."""
    with open(path, "r", encoding="utf-8") as f:
        return json_to_cot_prompt_answer_dict(json.load(f))


def load_gsm8k_json(path: str) -> List[dict]:
    """Load a GSM8K-format JSON file and return it as-is (list of dicts).

    Each element is expected to have at least ``"question"`` and ``"answer"`` keys.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(
            f"GSM8K JSON must be a list of records, got {type(data)!r}"
        )
    return data


import json
from typing import Dict, Union


def _coerce_answer(value: object) -> Union[int, str]:
    """Keep arithmetic answers as ``int``; preserve non-numeric strings as ``str``.

    Arithmetic datasets store integer answers (possibly as numeric strings) and
    rely on ``int`` typing downstream.  Chain-of-thought datasets (e.g. GSM8K)
    store the full solution trace as a string, which is not int-castable; those
    are preserved verbatim.  ``AddDataset`` already handles both types.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)


def json_to_prompt_answer_dict(raw: object) -> Dict[str, Union[int, str]]:
    """Normalize dataset JSON to ``{prompt: answer}``.

    Supports:

    - **Flat dict** ``{"12+34=": 46, ...}`` (string or int values).
    - **List of records** ``[{"q_str": "...", "a_str": "..."}, ...]`` (current repo format).

    Numeric answers are returned as ``int``; non-numeric answers (CoT/GSM8K
    solution strings) are preserved as ``str``.
    """
    if isinstance(raw, dict):
        return {str(k): _coerce_answer(v) for k, v in raw.items()}
    if isinstance(raw, list):
        out: Dict[str, Union[int, str]] = {}
        for i, row in enumerate(raw):
            if not isinstance(row, dict):
                raise TypeError(f"Dataset row {i} must be a dict, got {type(row)}")
            if "q_str" not in row or "a_str" not in row:
                raise ValueError(
                    "List-format rows must include q_str and a_str; "
                    f"row {i} has keys: {list(row.keys())}",
                )
            out[str(row["q_str"])] = _coerce_answer(row["a_str"])
        return out
    raise TypeError(f"Dataset JSON must be a dict or list, got {type(raw)!r}")


def load_prompt_answer_json(path: str) -> Dict[str, Union[int, str]]:
    """Load train/test JSON from disk into ``{prompt: answer}``.

    Numeric answers become ``int``; CoT/GSM8K solution strings stay ``str``.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json_to_prompt_answer_dict(json.load(f))


import json
import os
import random
from typing import Dict, List, Optional, Sequence

from .dataset_paths import (
    dataset_all_json_path,
    dataset_test_json_path,
    dataset_train_json_path,
    default_datasets_dir,
)


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


def _load_dataset_rows(path: str) -> List[Dict]:
    """Load a dataset JSON as a list of row dicts for mix/export helpers."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise TypeError(
            f"Expected list-format dataset at {path!r} for mixing, got {type(raw)}."
        )
    rows: List[Dict] = []
    for i, row in enumerate(raw):
        if not isinstance(row, dict):
            raise TypeError(f"Dataset row {i} in {path!r} must be a dict, got {type(row)}")
        if "q_str" not in row or "a_str" not in row:
            raise ValueError(
                f"Dataset row {i} in {path!r} must include q_str and a_str; "
                f"got keys {list(row.keys())}",
            )
        rows.append(dict(row))
    return rows


def mix_datasets(
    dataset_stems: Sequence[str],
    output_stem: str,
    *,
    datasets_dir: Optional[str] = None,
    shuffle: bool = True,
) -> Dict[str, str]:
    """Mix existing dataset families into combined ``_all``, ``_train``, and ``_test`` JSONs.

    Args:
        dataset_stems: Source dataset prefixes, e.g. ``["22_add", "222_add_tight"]``.
        output_stem: Prefix for the mixed outputs, e.g. ``"mixed_add"``.
        datasets_dir: Optional datasets root; defaults to :func:`default_datasets_dir`.
        shuffle: If True (default), shuffle each combined split after concatenation.

    Returns:
        Mapping ``{"all": ..., "train": ..., "test": ...}`` with output file paths.
    """
    stems = [str(stem).strip() for stem in dataset_stems if str(stem).strip()]
    if not stems:
        raise ValueError("dataset_stems must contain at least one non-empty dataset prefix.")
    out_stem = str(output_stem).strip()
    if not out_stem:
        raise ValueError("output_stem must be a non-empty dataset prefix.")

    d = os.path.abspath(datasets_dir) if datasets_dir else default_datasets_dir()
    mixed = {"all": [], "train": [], "test": []}
    split_paths = {
        "all": dataset_all_json_path,
        "train": dataset_train_json_path,
        "test": dataset_test_json_path,
    }

    for stem in stems:
        for split, path_fn in split_paths.items():
            path = path_fn(stem, d)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Dataset {split} file not found for prefix {stem!r}: {path}",
                )
            mixed[split].extend(_load_dataset_rows(path))

    if shuffle:
        for rows in mixed.values():
            random.shuffle(rows)

    out_paths = {
        "all": dataset_all_json_path(out_stem, d),
        "train": dataset_train_json_path(out_stem, d),
        "test": dataset_test_json_path(out_stem, d),
    }

    for split, path in out_paths.items():
        out_dir = os.path.dirname(os.path.abspath(path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mixed[split], f, indent=4)

    return out_paths

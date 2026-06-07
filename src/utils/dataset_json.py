import json
from typing import Any, Dict, List, Union

import torch
from torch.utils.data import Dataset


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


_HF_DATASETS = ("gsm8k", "svamp")


class PromptAnswerDataset(Dataset):
    """Tokenized prompt-answer dataset for SFT.

    Uses a chat template for HF datasets (gsm8k, svamp); plain text for local ones.
    Each sample stores ``input_ids`` (prompt + answer), ``prompt_len``, ``prompt``, ``answer``.
    """

    def __init__(self, dataset: str, data: Dict[str, Union[int, str]], tokenizer) -> None:
        use_chat = dataset in _HF_DATASETS
        self.samples: List[Dict[str, Any]] = []
        for prompt, answer in data.items():
            answer_text = str(answer) if isinstance(answer, int) else answer
            if use_chat and getattr(tokenizer, "chat_template", None):
                try:
                    formatted_prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    formatted_prompt = prompt + "\n\nA:"
            else:
                formatted_prompt = prompt
            prompt_ids = tokenizer(
                formatted_prompt, return_tensors="pt", padding=False, add_special_tokens=False,
            )["input_ids"].squeeze(0)
            answer_ids = tokenizer(
                answer_text + tokenizer.eos_token,
                return_tensors="pt", padding=False, add_special_tokens=False,
            )["input_ids"].squeeze(0)
            self.samples.append({
                "input_ids": torch.cat([prompt_ids, answer_ids]),
                "prompt_len": int(prompt_ids.size(0)),
                "prompt": prompt,
                "formatted_prompt": formatted_prompt,
                "answer": answer,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def collate_fn(examples: List[Dict[str, Any]], pad_id: int) -> Dict[str, Any]:
    max_len = max(ex["input_ids"].size(0) for ex in examples)
    input_ids = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(len(examples), max_len, dtype=torch.long)
    response_mask = torch.zeros(len(examples), max_len, dtype=torch.long)
    for row, ex in enumerate(examples):
        ids = ex["input_ids"]
        prompt_len = ex["prompt_len"]
        input_ids[row, : ids.size(0)] = ids
        attention_mask[row, : ids.size(0)] = 1
        response_mask[row, prompt_len : ids.size(0)] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
        "prompts": [str(ex["prompt"]) for ex in examples],
        "answers": [ex["answer"] for ex in examples],
    }


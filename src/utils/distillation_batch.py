import json
from typing import Any, Dict, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .dataset_json import json_to_prompt_answer_dict


def masked_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    kl_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """KL from teacher to student over masked next-token positions."""
    t = temperature
    vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
    student_logits = student_logits[..., :vocab]
    teacher_logits = teacher_logits[..., :vocab]

    log_p_t = F.log_softmax(teacher_logits.float() / t, dim=-1)
    p_t = log_p_t.exp()
    log_q_s = F.log_softmax(student_logits.float() / t, dim=-1)
    kl_per_vocab = p_t * (log_p_t - log_q_s)
    kl_per_token = kl_per_vocab.sum(dim=-1)
    return (kl_per_token * kl_mask).sum() / kl_mask.sum().clamp_min(1.0) * (t**2)


class AddDataset(Dataset):
    """Tokenize prompt-answer rows once for masked causal KL."""

    def __init__(self, data: Union[str, Dict[str, int]], tokenizer):
        if isinstance(data, str):
            with open(data, "r", encoding="utf-8") as f:
                raw = json.load(f)
            data = json_to_prompt_answer_dict(raw)
        self.samples = []
        for prompt, answer in data.items():
            answer_text = str(answer)
            prompt_ids = tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                add_special_tokens=False,
            )["input_ids"].squeeze(0)
            answer_ids = tokenizer(
                answer_text + tokenizer.eos_token,
                return_tensors="pt",
                padding=False,
                add_special_tokens=False,
            )["input_ids"].squeeze(0)
            self.samples.append(
                {
                    "input_ids": torch.cat([prompt_ids, answer_ids]),
                    "prompt": str(prompt),
                    "answer": int(answer),
                },
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def collate_fn(examples, pad_id: int) -> Dict[str, Any]:
    """Right-pad sequences and mark valid causal next-token positions for KL."""
    max_len = max(ex["input_ids"].size(0) for ex in examples)
    batch_size = len(examples)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    kl_mask = torch.zeros(batch_size, max_len, dtype=torch.float32)

    for row, ex in enumerate(examples):
        ids = ex["input_ids"]
        length = ids.size(0)
        input_ids[row, :length] = ids
        attention_mask[row, :length] = 1
        kl_mask[row, : max(length - 1, 0)] = 1.0

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "kl_mask": kl_mask,
        "prompts": [str(ex["prompt"]) for ex in examples],
        "answers": [int(ex["answer"]) for ex in examples],
    }

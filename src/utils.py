import json
import math
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HF_READ_TOKEN = os.environ.get("HF_READ_TOKEN", "") or os.environ.get("HF_TOKEN", "")

DIR_ROOT = "/content/drive/My Drive/Math Circuit Distillation (ESE 5460)"

_logged_in = False

_NUM = r"-?\d[\d,]*(?:\.\d+)?"


def _normalize_numeric_str(s: str) -> str:
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


def _clean_num(s: str) -> str:
    return _normalize_numeric_str(s.replace(",", ""))


def extract_local_dataset_answer(text: str) -> Optional[int]:
    matches = re.findall(r"=\s*(-?\d+)", text)
    if matches:
        try:
            return int(matches[-1])
        except ValueError:
            return None
    return None


def extract_gsm8k_answer(text: str) -> Optional[str]:
    text = (text or "").strip()
    m = re.search(r"####\s*(" + _NUM + r")", text)
    if m:
        return _clean_num(m.group(1))
    return None


def extract_svamp_answer(text: str) -> Optional[str]:
    text = (text or "").strip()
    matches = re.findall(r"=\s*(" + _NUM + r")", text)
    if matches:
        return _clean_num(matches[-1])
    return None


def parse_response(text: str, dataset: str) -> Optional:
    if dataset == "gsm8k":
        return extract_gsm8k_answer(text)
    if dataset == "svamp":
        return extract_svamp_answer(text)
    return extract_local_dataset_answer(text)


def patch_tokenizer_no_special_tokens(tokenizer):
    if getattr(tokenizer, "_math_circuit_no_special_tokens_patched", False):
        return tokenizer
    orig_call = tokenizer.__call__
    orig_encode = tokenizer.encode

    def _call(*args, **kwargs):
        kwargs.setdefault("add_special_tokens", False)
        return orig_call(*args, **kwargs)

    def _encode(*args, **kwargs):
        kwargs.setdefault("add_special_tokens", False)
        return orig_encode(*args, **kwargs)

    tokenizer.__call__ = _call
    tokenizer.encode = _encode
    tokenizer._math_circuit_no_special_tokens_patched = True
    return tokenizer


def load_model(model_name):
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    try:
        from huggingface_hub.utils import disable_progress_bars as _hfhub_disable
        _hfhub_disable()
    except Exception:
        pass

    from huggingface_hub import login
    from transformers import AutoModelForCausalLM, AutoTokenizer

    global _logged_in
    if not _logged_in and HF_READ_TOKEN:
        login(HF_READ_TOKEN)
        _logged_in = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve model path: explicit local path → HF hub → DIR_ROOT/<model_name>
    _explicit_local = model_name.startswith("/") or model_name.startswith(".") or os.path.isdir(model_name)
    if _explicit_local:
        load_path = model_name
        local_kwargs = {"local_files_only": True}
    else:
        dir_root_path = os.path.join(DIR_ROOT, model_name)
        if os.path.isdir(dir_root_path):
            load_path = dir_root_path
            local_kwargs = {"local_files_only": True}
        else:
            load_path = model_name
            local_kwargs = {}

    _prev_tqdm = os.environ.get("TQDM_DISABLE")
    os.environ["TQDM_DISABLE"] = "1"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=torch.bfloat16,
            **local_kwargs,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(load_path, **local_kwargs)
    finally:
        if _prev_tqdm is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = _prev_tqdm
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, patch_tokenizer_no_special_tokens(tokenizer)


@torch.no_grad()
def eval_model(model, tokenizer, test_dataset, dataset_name: str, batch_size: int, max_eval_tokens: int) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    correct = total = 0
    samples = test_dataset.samples
    try:
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            prompts = [s["formatted_prompt"] for s in batch]
            golds = [s["answer"] for s in batch]
            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True,
            ).to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_eval_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            # decode full output (prompt echo + gen) so "= 46" is present for arithmetic extraction
            full_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for full_text, gold in zip(full_texts, golds):
                pred = parse_response(full_text, dataset_name)
                if isinstance(gold, int):
                    if pred is not None and pred == gold:
                        correct += 1
                else:
                    gold_parsed = parse_response(str(gold), dataset_name)
                    if pred is not None and gold_parsed is not None and pred == gold_parsed:
                        correct += 1
                total += 1
    finally:
        tokenizer.padding_side = original_side
    model.train()
    return correct / max(total, 1)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_HF_DATASETS = ("gsm8k", "svamp")


class PromptAnswerDataset(Dataset):
    """Tokenized prompt-answer dataset for SFT."""

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


_TRAIN_SPLIT_SIZE = 7000


def load_split(dataset: str, split: str) -> Dict:
    """Return a single split (e.g. 'all', 'train', 'test') from DIR_ROOT/datasets/{dataset}/{split}.json."""
    path = os.path.join(DIR_ROOT, "datasets", dataset, f"{split}.json")
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    out: Dict[str, Union[int, str]] = {}
    for row in rows:
        v = row["a_str"]
        out[str(row["q_str"])] = int(v) if isinstance(v, (int, bool)) or (isinstance(v, str) and v.lstrip("-").isdigit()) else str(v)
    return out


def load_data(dataset: str, test_limit: Optional[int] = None) -> Tuple[Dict, Dict]:
    """Return (train_data, test_data) dicts for a local dataset under DIR_ROOT/datasets/."""
    datasets_base = os.path.join(DIR_ROOT, "datasets")
    train_path = os.path.join(datasets_base, dataset, "train.json")
    test_path = os.path.join(datasets_base, dataset, "test.json")

    def _load(path: str) -> Dict:
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        out: Dict[str, Union[int, str]] = {}
        for row in rows:
            v = row["a_str"]
            out[str(row["q_str"])] = int(v) if isinstance(v, (int, bool)) or (isinstance(v, str) and v.lstrip("-").isdigit()) else str(v)
        return out

    train_data = _load(train_path)
    test_data = _load(test_path)
    if dataset == "gsm8k":
        train_data = dict(list(train_data.items())[:_TRAIN_SPLIT_SIZE])
    if test_limit is not None:
        test_data = dict(list(test_data.items())[:test_limit])
    return train_data, test_data

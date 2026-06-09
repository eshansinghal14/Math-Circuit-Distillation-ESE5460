import math
import os
import re
from typing import Optional

import torch
from torch.utils.data import DataLoader

HF_READ_TOKEN = os.environ.get("HF_READ_TOKEN", "") or os.environ.get("HF_TOKEN", "")

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


def _extract_int_after_equals(text: str) -> Optional[int]:
    m = re.search(r"=\s*(\d+)", text)
    return int(m.group(1)) if m else None


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
    return _extract_int_after_equals(text)


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
    _is_local = model_name.startswith("/") or model_name.startswith(".") or os.path.isdir(model_name)
    local_kwargs = {"local_files_only": True} if _is_local else {}
    _prev_tqdm = os.environ.get("TQDM_DISABLE")
    os.environ["TQDM_DISABLE"] = "1"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            **local_kwargs,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, **local_kwargs)
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
                prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False,
            ).to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_eval_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            gen_texts = tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            )
            for gen, gold in zip(gen_texts, golds):
                pred = parse_response(gen, dataset_name)
                gold_parsed = parse_response(str(gold), dataset_name)
                # normalize both sides to str for comparison; also handle arithmetic
                # datasets where the model generates just the number (no "=" in gen)
                if pred is None and isinstance(gold, int):
                    m = re.search(r"-?\d+", gen.strip())
                    pred = int(m.group()) if m else None
                if pred is not None and gold_parsed is not None and str(pred) == str(gold_parsed):
                    correct += 1
                elif pred is not None and isinstance(gold, int) and pred == gold:
                    correct += 1
                total += 1
    finally:
        tokenizer.padding_side = original_side
    model.train()
    return correct / max(total, 1)

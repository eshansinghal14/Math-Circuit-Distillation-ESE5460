import functools
import json
import random
from typing import Dict, List, Optional, Tuple

import torch

from .answer_parsing import (
    _extract_int_after_equals,
    _normalize_numeric_str,
    extract_gsm8k_answer,
)
from .config import EVAL_MAX_NEW_TOKENS
from .dataset_json import json_to_prompt_answer_dict

TRAIN_SPLIT_SIZE = 7000
FEWSHOT_POOL_SIZE = 473


def build_fewshot_prefix(pool: Dict[str, str], n: int) -> str:
    """Return a few-shot prefix built from n randomly sampled examples in pool."""
    examples = list(pool.items())
    selected = random.sample(examples, min(n, len(examples)))
    parts = [f"{prompt}\n{answer}" for prompt, answer in selected]
    return "\n\n".join(parts) + "\n\n"


@functools.lru_cache(maxsize=None)
def _load_hf_benchmark_rows_uncached(name: str) -> List[Tuple[str, str]]:
    """Load the full dataset once; result is cached for the process lifetime."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "gsm8k/svamp require `datasets`. Install with: pip install datasets",
        ) from e

    key = name.strip().lower()
    rows: List[Tuple[str, str]] = []

    if key == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        for ex in ds:
            q = ex["question"].strip()
            gold = extract_gsm8k_answer(ex["answer"])
            if gold is None:
                continue
            rows.append((f"Q: {q}", gold))
    elif key == "svamp":
        ddict = load_dataset("ChilleD/SVAMP")
        split = "test" if "test" in ddict else "train"
        for ex in ddict[split]:
            if ex.get("question_concat"):
                q = str(ex["question_concat"]).strip()
            else:
                q = f"{str(ex.get('Body', '')).strip()}\n{str(ex.get('Question', '')).strip()}"
            raw_a = ex.get("Answer", ex.get("answer"))
            if raw_a is None:
                continue
            try:
                gold = _normalize_numeric_str(str(raw_a).strip())
            except Exception:
                continue
            rows.append((f"Q: {q}", gold))
    else:
        raise ValueError(f"Unknown HF benchmark dataset: {name!r}")

    return rows


def load_hf_benchmark_rows(
    name: str,
    *,
    limit: Optional[int] = None,
) -> List[Tuple[str, str]]:
    rows = _load_hf_benchmark_rows_uncached(name.strip().lower())
    return rows[:limit] if limit is not None else rows


def _format_instruct_prompts(tokenizer, prompts: List[str]) -> Tuple[List[str], bool]:
    """Wrap prompts in the chat template; fall back to 'Q: ...\n\nA:' style."""
    if getattr(tokenizer, "chat_template", None):
        try:
            rendered = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in prompts
            ]
            return rendered, False
        except Exception:
            pass
    return [p + "\n\nA:" for p in prompts], False


@torch.no_grad()
def run_hf_benchmark(
    model,
    tokenizer,
    name: str,
    results_fname: Optional[str] = None,
    *,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    limit: Optional[int] = None,
    log: bool = True,
    fewshot_pool: Optional[Dict[str, str]] = None,
    num_fewshot: int = 0,
) -> Tuple[List[Dict], float]:
    """GSM8K / SVAMP via ``load_dataset``; greedy generate.

    Returns ``(results, accuracy)`` where each result has ``response``, ``answer`` (gold str),
    and ``parsed`` (predicted numeric str or ``null``).
    """
    rows = load_hf_benchmark_rows(name, limit=limit)
    n = len(rows)
    results: List[Dict] = []
    correct = 0
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    try:
        for i in range(0, n, batch_size):
            batch = rows[i : i + batch_size]
            prompts = [p for p, _ in batch]
            if fewshot_pool and num_fewshot > 0:
                prompts = [build_fewshot_prefix(fewshot_pool, num_fewshot) + p for p in prompts]
            golds = [g for _, g in batch]
            model_prompts, add_special_tokens = _format_instruct_prompts(
                tokenizer,
                prompts,
            )
            inputs = tokenizer(
                model_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=add_special_tokens,
            ).to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                stop_strings=["\n\n"],
                tokenizer=tokenizer,
            )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prompt_len = inputs["input_ids"].shape[1]
            gen_only = tokenizer.batch_decode(
                outputs[:, prompt_len:], skip_special_tokens=True
            )
            for j, full_text in enumerate(decoded):
                pred = extract_gsm8k_answer(gen_only[j])
                gold = golds[j]
                if pred is not None and pred == gold:
                    correct += 1
                results.append(
                    {
                        "response": full_text,
                        "answer": gold,
                        "parsed": pred,
                    },
                )
            if log:
                end = min(i + batch_size, n)
                print(f"processing {end}/{n}")
    finally:
        tokenizer.padding_side = original_side

    acc = correct / n if n else 0.0
    if results_fname is not None:
        with open(results_fname, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

    return results, acc


@torch.no_grad()
def evaluate_prompt_answer_dict(
    model,
    tokenizer,
    data: Dict[str, int],
    batch_size: int = 32,
    max_new_tokens: Optional[int] = None,
    debug_decode: int = 0,
    debug_tag: Optional[str] = None,
) -> float:
    """Greedy generation accuracy on a prompt-answer mapping using right padding."""
    if max_new_tokens is None:
        max_new_tokens = EVAL_MAX_NEW_TOKENS
    prompts = list(data.keys())
    answers = list(data.values())

    model.eval()
    correct = total = 0
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    for i in range(0, len(prompts), batch_size):
        batch_p = prompts[i : i + batch_size]
        batch_a = answers[i : i + batch_size]
        inputs = tokenizer(
            batch_p,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        ).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            stop_strings=["\n\n"],
            tokenizer=tokenizer,
        )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # Generated-only continuation (echoed prompt stripped). For CoT/GSM8K
        # scoring we must NOT scan the prompt, otherwise the extractor grabs a
        # number out of the question text. The prompt occupies the first
        # ``inputs["input_ids"].shape[1]`` columns (pad columns included under
        # right padding), so everything after that is purely model-generated.
        prompt_len = inputs["input_ids"].shape[1]
        gen_only = tokenizer.batch_decode(
            outputs[:, prompt_len:], skip_special_tokens=True
        )
        if debug_decode > 0 and i == 0:
            n = min(debug_decode, len(decoded))
            tag = f" [{debug_tag}]" if debug_tag else ""
            print(f"--- decode debug{tag} (first batch, max_new_tokens={max_new_tokens}) ---")
            for j in range(n):
                text = decoded[j]
                gold = batch_a[j]
                pred = _extract_int_after_equals(text)
                print(f"  gold={gold}  pred={pred}  decoded={text!r}")
            print("---")

        for text, gen, gold in zip(decoded, gen_only, batch_a):
            if isinstance(gold, int):
                # Arithmetic / single-token integer path (unchanged): the gold is
                # an int and the prompt ends in ``=``; compare the first ``= <int>``
                # over the FULL decoded text (prompt echo carries the ``=``).
                pred = _extract_int_after_equals(text)
                if pred == gold:
                    correct += 1
            else:
                # CoT / GSM8K path: require the #### delimiter.
                gold_num = extract_gsm8k_answer(str(gold))
                pred_num = extract_gsm8k_answer(gen)
                if (
                    gold_num is not None
                    and pred_num is not None
                    and pred_num == gold_num
                ):
                    correct += 1
            total += 1

    tokenizer.padding_side = original_side
    return correct / max(total, 1)


def test_model(
    model,
    tokenizer,
    dataset_fname,
    results_fname,
    batch_size=50,
    max_new_tokens=EVAL_MAX_NEW_TOKENS,
    log=True,
):
    """Greedy eval on a math JSON file.

    ``dataset_fname`` uses the same formats as :func:`json_to_prompt_answer_dict`:
    flat ``{prompt: answer}`` or a list of ``{q_str, a_str}`` rows (extra keys allowed).
    """
    model.eval()
    with open(dataset_fname, encoding="utf-8") as f:
        raw = json.load(f)
    data = json_to_prompt_answer_dict(raw)
    prompts = list(data.keys())
    answers = [int(v) for v in data.values()]
    n = len(prompts)
    results = []
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for i in range(0, n, batch_size):
            with torch.no_grad():
                if log:
                    print(f"processing {i}/{n}")
                end = min(i + batch_size, n)
                batched_prompts = prompts[i:end]
                batched_answers = answers[i:end]
                input_ids = tokenizer(
                    batched_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    add_special_tokens=True,
                ).to(model.device)
                outputs = model.generate(
                    **input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                for k, resp in enumerate(responses):
                    results.append({"response": resp, "answer": str(batched_answers[k])})
    finally:
        tokenizer.padding_side = original_side

    with open(results_fname, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    return results


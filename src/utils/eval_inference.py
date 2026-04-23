import json
from typing import Dict, List, Optional, Tuple

import torch

from .answer_parsing import (
    _extract_int_after_equals,
    _gsm8k_gold_answer_str,
    _normalize_numeric_str,
    _slice_text_after_reasoning,
    extract_numeric_answer_from_text,
    parse_answer,
)
from .config import EVAL_MAX_NEW_TOKENS
from .dataset_json import json_to_prompt_answer_dict


def load_hf_benchmark_rows(
    name: str,
    *,
    limit: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """``load_dataset`` for ``gsm8k`` or ``svamp``; return ``(prompt, gold_str)`` rows.

    Prompt format: ``Question: ...\\nReasoning:``. Gold is a normalized numeric string.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "gsm8k/svamp require `datasets`. Install with: pip install datasets",
        ) from e

    key = name.strip().lower()
    rows: List[Tuple[str, str]] = []

    if key == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        for ex in ds:
            q = ex["question"].strip()
            gold = _gsm8k_gold_answer_str(ex["answer"])
            if gold is None:
                continue
            prompt = f"Question: {q}\nReasoning:"
            rows.append((prompt, gold))
            if limit is not None and len(rows) >= limit:
                break
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
            prompt = f"Question: {q}\nReasoning:"
            rows.append((prompt, gold))
            if limit is not None and len(rows) >= limit:
                break
    else:
        raise ValueError(f"Unknown HF benchmark dataset: {name!r}")

    return rows


@torch.no_grad()
def run_hf_benchmark(
    model,
    tokenizer,
    name: str,
    results_fname: str,
    *,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    limit: Optional[int] = None,
    log: bool = True,
) -> Tuple[List[Dict], float]:
    """GSM8K / SVAMP via ``load_dataset``; greedy generate; save JSON like :func:`test_model`.

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
            golds = [g for _, g in batch]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True,
            ).to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for j, full_text in enumerate(decoded):
                tail = _slice_text_after_reasoning(full_text)
                pred = extract_numeric_answer_from_text(tail)
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
    with open(results_fname, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    return results, acc


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    test_path: str,
    batch_size: int = 32,
    max_new_tokens: Optional[int] = None,
    debug_decode: int = 0,
    debug_tag: Optional[str] = None,
) -> float:
    """Greedy generation accuracy on a math test JSON (right padding; int after ``=``).

    Uses the same padding side as :class:`neuron_distillation.distillation.AddDataset` /
    ``collate_fn`` so RoPE positions match distillation training.
    """
    if max_new_tokens is None:
        max_new_tokens = EVAL_MAX_NEW_TOKENS
    with open(test_path, "r", encoding="utf-8") as f:
        data = json_to_prompt_answer_dict(json.load(f))
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
        )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
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

        for text, gold in zip(decoded, batch_a):
            pred = _extract_int_after_equals(text)
            if pred == gold:
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


def eval_model(results_fname, log: bool = True):
    with open(results_fname, "r", encoding="utf-8") as f:
        results = json.load(f)
    if not results:
        return 0.0

    correct = 0
    for res in results:
        if parse_answer(res["response"]) == int(res["answer"]):
            correct += 1

    acc = correct / len(results)
    if log:
        print("Accuracy: ", acc)
    return acc

import json
from typing import Dict, List, Optional, Tuple

import torch

from .answer_parsing import (
    _extract_int_after_equals,
    _normalize_numeric_str,
    extract_numeric_answer_from_text,
)
from .config import EVAL_MAX_NEW_TOKENS
from .dataset_json import json_to_prompt_answer_dict


def load_hf_benchmark_rows(
    name: str,
    *,
    limit: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """``load_dataset`` for ``gsm8k`` or ``svamp``; return ``(prompt, gold_str)`` rows.

    Prompt format: ``Solve the math problem step by step. ...``.
    Gold is a normalized numeric string.
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
        ds = load_dataset("openai/gsm8k", "main", split="test")
        for ex in ds:
            q = ex["question"].strip()
            gold = extract_numeric_answer_from_text(ex["answer"])
            if gold is None:
                continue
            prompt = f"Solve the math problem step by step. {q}"
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
            prompt = f"Solve the math problem step by step. {q}"
            rows.append((prompt, gold))
            if limit is not None and len(rows) >= limit:
                break
    else:
        raise ValueError(f"Unknown HF benchmark dataset: {name!r}")

    return rows


def _format_instruct_prompts(tokenizer, prompts: List[str]) -> Tuple[List[str], bool]:
    """Wrap user prompts in the model's chat template when available."""
    if not getattr(tokenizer, "chat_template", None):
        return prompts, True
    try:
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
    except Exception:
        return prompts, True
    return rendered, False


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
            )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prompt_len = inputs["input_ids"].shape[1]
            gen_only = tokenizer.batch_decode(
                outputs[:, prompt_len:], skip_special_tokens=True
            )
            for j, full_text in enumerate(decoded):
                pred = extract_numeric_answer_from_text(gen_only[j])
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
                # CoT / GSM8K path: compare the last number in the gold solution
                # against the last number in the student's generated continuation.
                gold_num = extract_numeric_answer_from_text(str(gold))
                pred_num = extract_numeric_answer_from_text(gen)
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


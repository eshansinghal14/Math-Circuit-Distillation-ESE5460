"""Quick baseline eval + sample printing for debugging.

Usage (from src/):
    python eval_baseline.py 22_add_tight 222_add_tight --samples 10
    python eval_baseline.py 22_add_tight --samples 5 --max-new-tokens 3
    python eval_baseline.py 2d_add --models student  # student only

    # Pass raw file paths with --file:
    python eval_baseline.py --file /path/to/2d_add_test_20.json --samples 10
    python eval_baseline.py 22_add_tight --file /path/to/2d_add_test_20.json --samples 5
"""

import argparse
import os
import re
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    LLAMA_1B_MODEL_NAME,
    LLAMA_8B_MODEL_NAME,
    load_prompt_answer_json,
    resolve_test_path,
    patch_tokenizer_no_special_tokens,
)


def _extract_int_after_equals(text):
    m = re.search(r"=\s*(\d+)", text)
    return int(m.group(1)) if m else None


def eval_and_print(model, tokenizer, data, max_new_tokens, n_samples, label):
    model.eval()
    prompts = list(data.keys())
    answers = list(data.values())
    correct = total = 0

    original_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    printed = 0
    batch_size = min(100, len(prompts))

    for i in range(0, len(prompts), batch_size):
        bp = prompts[i : i + batch_size]
        ba = answers[i : i + batch_size]

        inputs = tokenizer(
            bp, return_tensors="pt", padding=True, add_special_tokens=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for j, (pred_text, gold) in enumerate(zip(decoded, ba)):
            pred = _extract_int_after_equals(pred_text)
            is_correct = pred == gold
            if is_correct:
                correct += 1
            total += 1

            if printed < n_samples:
                prompt = bp[j]
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
                gen_ids = outputs[j].tolist()
                new_ids = gen_ids[len(prompt_ids):]
                new_tokens = [tokenizer.decode([t]) for t in new_ids]
                mark = "✓" if is_correct else "✗"
                print(f"  {mark} prompt={prompt!r}  gold={gold}  pred={pred}  "
                      f"decoded={pred_text!r}  new_token_ids={new_ids}  "
                      f"new_tokens={new_tokens}")
                printed += 1

    tokenizer.padding_side = original_side
    acc = correct / max(total, 1)
    print(f"\n  {label} accuracy: {acc:.4f} ({correct}/{total})\n")
    return acc


def main():
    parser = argparse.ArgumentParser(description="Baseline eval + sample debug")
    parser.add_argument("datasets", nargs="*", default=[], help="Dataset prefixes (e.g. 22_add_tight)")
    parser.add_argument("--file", "-f", action="append", default=[], help="Raw JSON file paths (e.g. /path/to/2d_add_test_20.json). Can repeat.")
    parser.add_argument("--samples", "-n", type=int, default=10, help="Number of samples to print per dataset")
    parser.add_argument("--max-new-tokens", type=int, default=2, help="Max tokens to generate")
    parser.add_argument("--models", choices=["both", "student", "teacher"], default="both")
    parser.add_argument("--datasets-dir", type=str, default=None)
    args = parser.parse_args()

    if not args.datasets and not args.file:
        parser.error("Provide at least one dataset prefix or --file path")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_1B_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer = patch_tokenizer_no_special_tokens(tokenizer)

    # Show tokenizer info
    print(f"\nTokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Tokenizer test: '24+80=' -> {tokenizer.encode('24+80=', add_special_tokens=False)}")
    print(f"Tokenizer test: '24 + 80 = ' -> {tokenizer.encode('24 + 80 = ', add_special_tokens=False)}")
    for a in ["34", "104", "297"]:
        ids = tokenizer.encode(a, add_special_tokens=False)
        print(f"  Answer '{a}' -> {ids} ({len(ids)} token(s))")

    models_to_eval = []
    if args.models in ("both", "student"):
        print(f"\nLoading student: {LLAMA_1B_MODEL_NAME}")
        student = AutoModelForCausalLM.from_pretrained(
            LLAMA_1B_MODEL_NAME,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        student.eval()
        models_to_eval.append(("Student (1B)", student))

    if args.models in ("both", "teacher"):
        print(f"Loading teacher: {LLAMA_8B_MODEL_NAME}")
        teacher = AutoModelForCausalLM.from_pretrained(
            LLAMA_8B_MODEL_NAME,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        teacher.eval()
        models_to_eval.append(("Teacher (8B)", teacher))

    # Build list of (label, path) to evaluate
    eval_sets = []
    for ds_prefix in args.datasets:
        try:
            test_path, prefix = resolve_test_path(
                dataset=ds_prefix, datasets_dir=args.datasets_dir,
            )
            eval_sets.append((ds_prefix, test_path))
        except (FileNotFoundError, SystemExit) as e:
            print(f"\n⚠ Skipping {ds_prefix}: {e}")

    for fpath in args.file:
        fpath = os.path.abspath(fpath)
        if not os.path.isfile(fpath):
            print(f"\n⚠ Skipping --file {fpath}: not found")
            continue
        label = os.path.splitext(os.path.basename(fpath))[0]
        eval_sets.append((label, fpath))

    for ds_label, test_path in eval_sets:
        data = load_prompt_answer_json(test_path)
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_label}  ({len(data)} test examples)")
        print(f"  Path: {test_path}")
        items = list(data.items())[:3]
        for p, a in items:
            print(f"  Example: prompt={p!r}  answer={a}")
        print(f"{'='*60}")

        for model_label, model in models_to_eval:
            print(f"\n--- {model_label} on {ds_label} (max_new_tokens={args.max_new_tokens}) ---")
            eval_and_print(
                model, tokenizer, data,
                max_new_tokens=args.max_new_tokens,
                n_samples=args.samples,
                label=f"{model_label} [{ds_label}]",
            )


if __name__ == "__main__":
    main()

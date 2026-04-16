"""Generate Alpaca responses with Llama 1B and keep the top-N hardest prompts.

Run from ``src/``::

  python -m experiments.alpaca_top_nll.run --top-n 100

The script loads ``tatsu-lab/alpaca``, formats each example as an Alpaca-style
instruction prompt ending in ``### Response:``, generates a greedy continuation
with the shared 1B Llama model id from ``utils``, scores that generated continuation under the
same model by summed token negative log likelihood, and saves only the prompt
strings for the top-N highest-NLL examples to JSON.
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import sys
from typing import Iterable, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError as e:  # pragma: no cover - runtime environment dependent
    raise SystemExit(
        "This experiment requires the `datasets` package. "
        "Install it first, then rerun this script."
    ) from e

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils import LLAMA_1B_MODEL_NAME, load_model  # noqa: E402


MODEL_NAME = LLAMA_1B_MODEL_NAME
ALPACA_DATASET = "tatsu-lab/alpaca"
BATCH_SIZE = 8
MAX_NEW_TOKENS = 64


def _format_alpaca_prompt(example: dict) -> str:
    instruction = str(example.get("instruction", "")).strip()
    inp = str(example.get("input", "")).strip()
    if inp:
        return (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            "### Response:\n"
        )
    return (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )


def _iter_batches(rows: List[dict], batch_size: int) -> Iterable[List[dict]]:
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def _generated_token_mask(
    generated_tokens: torch.Tensor,
    eos_token_id: int,
) -> torch.Tensor:
    """Mask real generated tokens, keeping the first EOS if one is produced."""
    batch, gen_len = generated_tokens.shape
    mask = torch.zeros((batch, gen_len), dtype=torch.long, device=generated_tokens.device)
    for i in range(batch):
        row = generated_tokens[i]
        if gen_len == 0:
            continue
        cutoff = gen_len
        for j, tok in enumerate(row.tolist()):
            cutoff = j + 1
            if tok == eos_token_id:
                break
        mask[i, :cutoff] = 1
    return mask


def _score_generated_sequences(
    model,
    tokenizer,
    prompts: List[str],
) -> Tuple[List[float], List[str]]:
    device = next(model.parameters()).device
    prompt_enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    input_ids = prompt_enc["input_ids"].to(device)
    prompt_attention = prompt_enc["attention_mask"].to(device)
    prompt_width = input_ids.shape[1]

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=prompt_attention,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    generated = outputs[:, prompt_width:]
    gen_mask = _generated_token_mask(generated, tokenizer.eos_token_id)
    full_attention = torch.cat([prompt_attention, gen_mask], dim=1)
    token_target_mask = torch.cat(
        [torch.zeros_like(prompt_attention), gen_mask],
        dim=1,
    )

    with torch.no_grad():
        logits = model(
            input_ids=outputs,
            attention_mask=full_attention,
        ).logits

    logits_pred = logits[:, :-1, :].float()
    targets = outputs[:, 1:]
    ce_per = F.cross_entropy(
        logits_pred.reshape(-1, logits_pred.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view(outputs.size(0), outputs.size(1) - 1)
    ce_mask = token_target_mask[:, 1:].float()
    nll = (ce_per * ce_mask).sum(dim=1)

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return nll.tolist(), decoded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save the top-N Alpaca prompts with highest generated-sequence NLL.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        required=True,
        metavar="N",
        help="Number of prompts to save.",
    )
    args = parser.parse_args()

    if args.top_n < 1:
        raise SystemExit("--top-n must be >= 1")

    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = load_model(MODEL_NAME)
    tokenizer.padding_side = "left"

    print(f"Loading dataset: {ALPACA_DATASET}")
    dataset = load_dataset(ALPACA_DATASET, split="train")
    rows = list(dataset)
    print(f"Loaded {len(rows)} Alpaca examples")

    top_heap: List[Tuple[float, int, str]] = []
    seen = 0

    for batch_rows in tqdm(
        _iter_batches(rows, BATCH_SIZE),
        total=(len(rows) + BATCH_SIZE - 1) // BATCH_SIZE,
        desc="Scoring Alpaca prompts",
    ):
        prompts = [_format_alpaca_prompt(row) for row in batch_rows]
        nlls, _ = _score_generated_sequences(model, tokenizer, prompts)
        for prompt, nll in zip(prompts, nlls):
            item = (float(nll), seen, prompt)
            seen += 1
            if len(top_heap) < args.top_n:
                heapq.heappush(top_heap, item)
            elif item[0] > top_heap[0][0]:
                heapq.heapreplace(top_heap, item)

    top_items = sorted(top_heap, key=lambda x: x[0], reverse=True)
    top_prompts = [prompt for _, _, prompt in top_items]

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"top_{args.top_n}_prompts.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(top_prompts, f, indent=2)

    print(f"Saved {len(top_prompts)} prompts to {out_path}")


if __name__ == "__main__":
    main()

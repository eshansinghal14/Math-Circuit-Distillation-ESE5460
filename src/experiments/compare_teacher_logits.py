"""Compare cached teacher logits against a live forward pass using the teacher adapter.

Loads a sample from the teacher data cache, runs the same prompt through the teacher
model live, and prints both sets of logits side-by-side for comparison.

Usage (from src/):
    python -m experiments.compare_teacher_logits \
        --cache-dir /path/to/teacher_cache \
        --sample-index 0 \
        --top-k 10 \
        --device cpu --dtype float32
"""

from __future__ import annotations

import argparse
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from graph_loss.hf_adapter import HFLlamaGraphAdapter
from graph_loss.teacher_data_cache import TeacherDataCache
from utils.hf_models import patch_tokenizer_no_special_tokens

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if dtype_str not in mapping:
        raise ValueError(f"Unknown dtype {dtype_str!r}, choose from {list(mapping)}")
    return mapping[dtype_str]


@torch.no_grad()
def _live_forward(adapter: HFLlamaGraphAdapter, input_ids: torch.Tensor) -> torch.Tensor:
    ids = input_ids.to(adapter.cfg.device)
    output = adapter.model(ids.unsqueeze(0))
    logits = output.logits if hasattr(output, "logits") else output
    return logits.squeeze(0).detach().cpu()


def _print_top_tokens(logits: torch.Tensor, tokenizer, top_k: int, label: str, pos: int) -> None:
    pos_logits = logits[pos]
    topk = torch.topk(pos_logits, k=min(top_k, pos_logits.shape[-1]))
    print(f"\n  {label} — position {pos} top-{top_k} tokens:")
    for rank, (tok_id, val) in enumerate(zip(topk.indices.tolist(), topk.values.tolist())):
        tok_str = repr(tokenizer.decode([tok_id]))
        print(f"    #{rank+1:2d}  id={tok_id:6d}  logit={val:9.4f}  token={tok_str}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare cached teacher logits vs live forward pass logits."
    )
    parser.add_argument("--cache-dir", required=True, help="Path to teacher data cache directory")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index in the cache manifest")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top tokens to print per position")
    parser.add_argument(
        "--positions",
        type=str,
        default=None,
        help="Comma-separated token positions to compare (default: last position only)",
    )
    parser.add_argument("--device", default=None, help="Device override (cpu / cuda / mps)")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    dtype = _resolve_dtype(args.dtype)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading teacher cache from %s", args.cache_dir)
    cache = TeacherDataCache(args.cache_dir)

    samples = cache.manifest.get("samples", [])
    if args.sample_index >= len(samples):
        raise IndexError(f"sample-index {args.sample_index} out of range (cache has {len(samples)} samples)")

    sample_meta = samples[args.sample_index]
    prompt: str = sample_meta["prompt"]
    answer: int = int(sample_meta["answer"])
    logger.info("Sample %d  prompt=%r  answer=%r", args.sample_index, prompt, answer)

    import os, torch as _torch
    logits_path = os.path.join(args.cache_dir, sample_meta["teacher_logits"])
    cached_record = _torch.load(logits_path, map_location="cpu", weights_only=False)
    cached_logits: torch.Tensor = cached_record["logits"]        # [seq_len, vocab_size]
    cached_input_ids: torch.Tensor = cached_record["input_ids"]  # [seq_len]
    logger.info("Cached logits shape: %s  input_ids shape: %s", cached_logits.shape, cached_input_ids.shape)

    teacher_model_id: str = cache.teacher_model
    logger.info("Loading teacher model %s ...", teacher_model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(teacher_model_id, torch_dtype=dtype)
    hf_model = hf_model.to(device)
    hf_model.eval()
    tokenizer = patch_tokenizer_no_special_tokens(AutoTokenizer.from_pretrained(teacher_model_id))
    adapter = HFLlamaGraphAdapter(hf_model, tokenizer, device)

    logger.info("Running live forward pass ...")
    live_logits = _live_forward(adapter, cached_input_ids)  # [seq_len, vocab_size]
    logger.info("Live logits shape: %s", live_logits.shape)

    seq_len = cached_input_ids.shape[0]
    if args.positions:
        positions = [int(p.strip()) for p in args.positions.split(",")]
    else:
        positions = [seq_len - 1]

    print("\n" + "=" * 70)
    print(f"  Prompt : {prompt!r}")
    print(f"  Answer : {answer}")
    print(f"  Tokens : {tokenizer.convert_ids_to_tokens(cached_input_ids.tolist())}")
    print("=" * 70)

    for pos in positions:
        if pos < 0 or pos >= seq_len:
            logger.warning("Position %d out of range [0, %d), skipping", pos, seq_len)
            continue

        _print_top_tokens(cached_logits, tokenizer, args.top_k, "CACHED ", pos)
        _print_top_tokens(live_logits, tokenizer, args.top_k, "LIVE   ", pos)

        diff = (cached_logits[pos] - live_logits[pos]).abs()
        print(f"\n  Max absolute diff at position {pos}: {diff.max().item():.6f}")
        print(f"  Mean absolute diff at position {pos}: {diff.mean().item():.6f}")

    overall_diff = (cached_logits - live_logits).abs()
    row_l1 = overall_diff.sum(dim=-1)  # [seq_len] — L1 norm per token position
    row_l1_strs = [f"{v:.4f}" for v in row_l1.tolist()]
    print("\n" + "=" * 70)
    print(f"  Row L1 norms (cached - live, one per token): {row_l1_strs}")
    print(f"  Overall max  |cached - live|: {overall_diff.max().item():.6f}")
    print(f"  Overall mean |cached - live|: {overall_diff.mean().item():.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""Compare cached teacher data against a live generate_teacher_sample run.

Loads a sample from the teacher data cache, regenerates it live using
generate_teacher_sample, and compares the cached vs live supergraphs
(adjacency matrix, supernodes, logit_token_ids, teacher_dla_logits).

Usage (from src/):
    python -m experiments.compare_teacher_logits \
        --cache-dir /path/to/teacher_cache \
        --sample-index 0 \
        --device cpu --dtype float32
"""

from __future__ import annotations

import argparse
import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from graph_loss.generate_teacher_data import generate_teacher_sample
from graph_loss.hf_adapter import HFLlamaGraphAdapter
from graph_loss.teacher_data_cache import TeacherDataCache
from graph_loss.training import CachedTeacherPromptData
from utils.hf_models import patch_tokenizer_no_special_tokens

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    return mapping[dtype_str]


def _load_cached_sample(cache: TeacherDataCache, sample_meta: dict) -> CachedTeacherPromptData:
    from graph_loss.graph import SuperGraph

    prompt: str = sample_meta["prompt"]
    answer: int = int(sample_meta["answer"])

    supergraph_dict = cache.load_teacher_supergraph(prompt, answer)
    cached_supergraph = SuperGraph(
        supernode_adjacency_matrix=supergraph_dict["supernode_adjacency_matrix"],
        supernodes=supergraph_dict["supernodes"],
        node_labels=supergraph_dict.get("node_labels"),
        supernode_labels=supergraph_dict.get("supernode_labels"),
        supernode_heatmap_pdf_paths=supergraph_dict.get("supernode_heatmap_pdf_paths"),
    )

    logits_record = torch.load(
        os.path.join(cache.cache_dir, sample_meta["teacher_logits"]),
        map_location="cpu",
        weights_only=False,
    )
    prompt_len: int = int(logits_record["prompt_len"])
    full_logits: torch.Tensor = logits_record["logits"]  # [seq_len, vocab_size]
    teacher_dla_logits = full_logits[prompt_len - 1] if prompt_len > 0 else None

    return CachedTeacherPromptData(
        supergraph=cached_supergraph,
        logit_token_ids=supergraph_dict.get("logit_token_ids"),
        teacher_dla_logits=teacher_dla_logits,
    )


def _compare_supergraphs(cached: CachedTeacherPromptData, live: CachedTeacherPromptData) -> None:
    print("\n" + "=" * 70)
    print("  SUPERGRAPH COMPARISON")
    print("=" * 70)

    # --- adjacency matrix ---
    ca = cached.supergraph.supernode_adjacency_matrix.float().cpu()
    la = live.supergraph.supernode_adjacency_matrix.float().cpu()
    print(f"\n  Cached adjacency shape : {tuple(ca.shape)}")
    print(f"  Live   adjacency shape : {tuple(la.shape)}")

    if ca.shape == la.shape:
        diff = (ca - la).abs()
        row_l1 = diff.sum(dim=-1)
        row_l1_strs = [f"{v:.4f}" for v in row_l1.tolist()]
        print(f"\n  Row L1 norms (|cached - live|, one per supernode): {row_l1_strs}")
        print(f"  Max  |cached - live|: {diff.max().item():.6f}")
        print(f"  Mean |cached - live|: {diff.mean().item():.6f}")
        print(f"  Frobenius norm diff : {diff.norm().item():.6f}")
    else:
        print("  [shapes differ — skipping element-wise diff]")

    # --- supernodes ---
    cn = len(cached.supergraph.supernodes)
    ln = len(live.supergraph.supernodes)
    print(f"\n  Cached supernodes count : {cn}")
    print(f"  Live   supernodes count : {ln}")

    # --- logit_token_ids ---
    c_ids = cached.logit_token_ids
    l_ids = live.logit_token_ids
    if c_ids is not None and l_ids is not None:
        c_set = set(c_ids.cpu().tolist())
        l_set = set(l_ids.cpu().tolist())
        overlap = len(c_set & l_set)
        print(f"\n  logit_token_ids  cached len={len(c_set)}  live len={len(l_set)}"
              f"  overlap={overlap}")
        only_cached = sorted(c_set - l_set)
        only_live = sorted(l_set - c_set)
        if only_cached:
            print(f"  Only in cached : {only_cached[:20]}")
        if only_live:
            print(f"  Only in live   : {only_live[:20]}")
    else:
        print(f"\n  logit_token_ids  cached={'present' if c_ids is not None else 'None'}"
              f"  live={'present' if l_ids is not None else 'None'}")

    # --- teacher_dla_logits ---
    c_dla = cached.teacher_dla_logits
    l_dla = live.teacher_dla_logits
    if c_dla is not None and l_dla is not None:
        c_dla_cpu = c_dla.float().cpu()
        l_dla_cpu = l_dla.float().cpu()
        dla_diff = (c_dla_cpu - l_dla_cpu).abs()
        print(f"\n  teacher_dla_logits  shape={tuple(c_dla_cpu.shape)}")
        print(f"  Max  |cached - live|: {dla_diff.max().item():.6f}")
        print(f"  Mean |cached - live|: {dla_diff.mean().item():.6f}")
    else:
        print(f"\n  teacher_dla_logits  cached={'present' if c_dla is not None else 'None'}"
              f"  live={'present' if l_dla is not None else 'None'}")

    print("\n" + "=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare cached teacher data vs live generate_teacher_sample output."
    )
    parser.add_argument("--cache-dir", required=True, help="Path to teacher data cache directory")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index in the manifest")
    parser.add_argument("--device", default=None, help="Device override (cpu / cuda / mps)")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    dtype = _resolve_dtype(args.dtype)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    logger.info("Loading teacher cache from %s", args.cache_dir)
    cache = TeacherDataCache(args.cache_dir)
    samples = cache.manifest.get("samples", [])
    if args.sample_index >= len(samples):
        raise IndexError(f"sample-index {args.sample_index} out of range ({len(samples)} samples)")

    sample_meta = samples[args.sample_index]
    prompt: str = sample_meta["prompt"]
    answer: int = int(sample_meta["answer"])
    logger.info("Sample %d  prompt=%r  answer=%r", args.sample_index, prompt, answer)

    logger.info("Loading cached sample ...")
    cached_data = _load_cached_sample(cache, sample_meta)

    hp: dict = cache.manifest.get("hyperparameters", {})
    teacher_model_id: str = cache.teacher_model
    logger.info("Loading teacher model %s ...", teacher_model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(teacher_model_id, torch_dtype=dtype)
    hf_model = hf_model.to(device)
    hf_model.eval()
    tokenizer = patch_tokenizer_no_special_tokens(AutoTokenizer.from_pretrained(teacher_model_id))
    adapter = HFLlamaGraphAdapter(hf_model, tokenizer, device)

    logger.info("Running live generate_teacher_sample ...")
    live_data: CachedTeacherPromptData = generate_teacher_sample(
        adapter,
        prompt,
        answer,
        top_k_logits=hp.get("top_k_logits", 0.95),
        temperature=hp.get("temperature", 2.0),
        prop_neurons_per_layer=hp.get("prop_neurons_per_layer", 0.1),
        batch_size=hp.get("attribution_batch_size", 256),
        dataset=hp.get("dataset"),
        anova_nodes_per_label=hp.get("anova_nodes_per_label", 10),
        anova_range_radius=hp.get("anova_range_radius", 0),
        sum_min_specificity=hp.get("sum_min_specificity", 0.0),
        node_labels=hp.get("graph_node_labels"),
        include_dla_node=hp.get("include_dla_node", False),
        include_arg_nodes=hp.get("include_arg_nodes", False),
        logger=logger,
    )

    print(f"\n  Prompt : {prompt!r}")
    print(f"  Answer : {answer}")
    _compare_supergraphs(cached_data, live_data)


if __name__ == "__main__":
    main()

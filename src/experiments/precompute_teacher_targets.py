"""Build the teacher's graph target for every training prompt once, ahead of training.

The teacher is frozen, so its per-prompt target (supernode adjacency, labels,
logit-target ids, DLA reference logits) is a pure function of the teacher-side
configuration. graph_kd caches those targets (``--teacher-target-cache``), but a
150-step run never repeats a prompt, so the first run over a prompt set still
pays the full teacher cost per prompt: ~9 s each on an 8B teacher with ANOVA
labels, which was 300 s of a 320 s step. This script fills the cache for the
whole training set in one pass, with only the teacher in memory, so every
subsequent run hits it for every prompt and pays the student's cost alone.

Pass exactly the teacher-side arguments the training runs will use; the cache
file is keyed on them, and both this script and graph_kd print the file they
use, so a mismatch is visible. Code changes do not invalidate the file: when the
graph_loss source has changed since it was written, a few cached prompts are
rebuilt and compared, and the file is kept if they still match. Shards let several
sessions share the work: ``--shard 0 3``, ``--shard 1 3``, ``--shard 2 3`` each
take every third prompt, and their flushes merge into the same file.

Usage (from src/):
    python -m experiments.precompute_teacher_targets \\
        --teacher meta-llama/Meta-Llama-3-8B-Instruct --dataset 22_add \\
        --graph-node-labels "sum units" "arg1 range" ... --nodes-per-label 3 \\
        --teacher-prop-neurons-per-layer 0.1 --temperature 1.0
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from typing import Callable

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DIR_ROOT, load_data, load_model  # noqa: E402
from graph_loss.utils import normalize_node_labels  # noqa: E402

from graph_loss.training import (  # noqa: E402
    GraphAuxConfig,
    TeacherTargetCache,
    _compute_teacher_target,
    _teacher_target_entry,
    teacher_target_cache_key,
)


def precompute(
    items: list[tuple[str, object]],
    cache: TeacherTargetCache,
    compute: Callable[[str, object], dict],
    *,
    flush_every: int = 100,
    log: Callable[[str], None] = print,
) -> int:
    """Fill ``cache`` with ``compute(prompt, answer)`` for every uncached item; returns how many were built."""
    todo = [(p, a) for p, a in items if p not in cache.entries]
    log(f"{len(items)} prompts, {len(items) - len(todo)} already cached, {len(todo)} to build")
    built = 0
    t0 = time.perf_counter()
    for i, (prompt, answer) in enumerate(todo, 1):
        cache.put(prompt, compute(prompt, answer))
        built += 1
        if built % flush_every == 0 or i == len(todo):
            cache.flush()
            elapsed = time.perf_counter() - t0
            rate = elapsed / built
            log(f"  {i}/{len(todo)} built | {rate:.2f} s/prompt | ETA {rate * (len(todo) - i) / 60:.1f} min"
                f" | {len(cache)} in cache")
        if built % 8 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return built


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--teacher", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--graph-node-labels", "--graph_node_labels", nargs="+", default=[],
                   dest="graph_node_labels", metavar="LABEL",
                   help="ANOVA supernode labels, exactly as passed to graph_kd (omit for arg-token + DLA; "
                        "'tokens' appends the token-embedding source columns).")
    p.add_argument("--supergraph-aggregation", type=str, default="normalised", dest="supergraph_aggregation",
                   choices=["normalised", "raw-signed", "token-path"], help="Must match graph_kd's flag.")
    p.add_argument("--token-path-rows", type=str, default="weighted", dest="token_path_rows",
                   choices=["weighted", "gold", "all"], help="Must match graph_kd's flag.")
    p.add_argument("--token-source-columns", action="store_true", dest="token_source_columns",
                   help="Must match graph_kd's flag (or pass 'tokens' as a label).")
    p.add_argument("--nodes-per-label", type=int, default=10, dest="nodes_per_label")
    p.add_argument("--teacher-prop-neurons-per-layer", type=float, default=0.1,
                   dest="teacher_prop_neurons_per_layer")
    p.add_argument("--top-k-logits", "--top_k_logits", type=float, default=0.95, dest="top_k_logits")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Must match graph_kd's --temperature (its default is 1.0).")
    p.add_argument("--teacher-graph-batch-size", type=int, default=512, dest="teacher_graph_batch_size")
    p.add_argument("--freeze-attention", action="store_true", dest="freeze_attention")
    p.add_argument("--freeze-rms-norm", action="store_true", dest="freeze_rms_norm")
    p.add_argument("--constant-node-weighting", "--constant_node_weighting", action="store_true",
                   dest="constant_node_weighting")
    p.add_argument("--cache-batch-size", type=int, default=32, dest="cache_batch_size",
                   help="Prompt batch size when building the teacher's MLP-input cache.")
    p.add_argument("--anova-cache-device", choices=["cpu", "pinned", "cuda"], default="cuda",
                   dest="anova_cache_device",
                   help="Where the teacher's MLP-input cache lives; with only the teacher loaded, "
                        "cuda (default) fits and makes the per-prompt copies free.")
    p.add_argument("--teacher-target-cache", type=str, default="cache/teacher_targets",
                   dest="teacher_target_cache_dir",
                   help="Same directory graph_kd uses (relative to the data root unless absolute).")
    p.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("INDEX", "COUNT"),
                   help="Take every COUNT-th prompt starting at INDEX, so sessions can split the work.")
    p.add_argument("--limit", type=int, default=None, help="Stop after this many prompts (smoke test).")
    p.add_argument("--flush-every", type=int, default=100, dest="flush_every")
    return p


def main() -> None:
    args = build_parser().parse_args()
    labels, tokens_label = normalize_node_labels(args.graph_node_labels)
    token_source_columns = args.token_source_columns or tokens_label
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, _ = load_data(args.dataset)
    items = sorted(train_data.items())
    shard_idx, shard_n = args.shard
    items = items[shard_idx::shard_n]
    if args.limit is not None:
        items = items[: args.limit]

    teacher, tokenizer = load_model(args.teacher)
    teacher.eval()
    for prm in teacher.parameters():
        prm.requires_grad_(False)
    if hasattr(teacher.config, "use_cache"):
        teacher.config.use_cache = False
    from graph_loss.hf_adapter import HFLlamaGraphAdapter
    adapter = HFLlamaGraphAdapter(teacher, tokenizer, device)

    teacher_mlp_cache = None
    if labels:
        from graph_loss.precompute_mlp_inputs import build_mlp_input_cache
        teacher_mlp_cache = build_mlp_input_cache(
            adapter, args.dataset, args.teacher,
            data_dict=train_data, batch_size=args.cache_batch_size,
            cache_device=args.anova_cache_device,
        )

    config = GraphAuxConfig(
        teacher_prop_neurons_per_layer=args.teacher_prop_neurons_per_layer,
        top_k_logits=args.top_k_logits,
        temperature=args.temperature,
        teacher_graph_batch_size=args.teacher_graph_batch_size,
        teacher_nodes_per_label=args.nodes_per_label,
        graph_node_labels=labels if labels else None,
        teacher_mlp_input_cache=teacher_mlp_cache,
        freeze_attention=args.freeze_attention,
        freeze_rms_norm=args.freeze_rms_norm,
        constant_node_weighting=args.constant_node_weighting,
        supergraph_aggregation=args.supergraph_aggregation,
        token_source_columns=token_source_columns,
        token_path_rows=args.token_path_rows,
        dataset_name=args.dataset,
    )
    cache_dir = args.teacher_target_cache_dir
    if not os.path.isabs(cache_dir):
        cache_dir = os.path.join(DIR_ROOT, cache_dir)
    key = teacher_target_cache_key(config, args.teacher)
    cache = TeacherTargetCache(os.path.join(cache_dir, f"{key}.pt"))
    print(f"Teacher target cache: {cache.path} ({len(cache)} prompts cached)")

    def compute(prompt: str, answer: object) -> dict:
        supergraph, logit_ids, dla_logits = _compute_teacher_target(prompt, answer, adapter, config, device)
        return _teacher_target_entry(supergraph, logit_ids, dla_logits, answer=answer)

    cache.validate(compute)
    built = precompute(items, cache, compute, flush_every=args.flush_every)
    print(f"Done: built {built} targets; {len(cache)} prompts in {cache.path}")


if __name__ == "__main__":
    main()

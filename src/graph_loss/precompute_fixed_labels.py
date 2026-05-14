"""Pre-compute a frozen neuron→label mapping for use with cluster_method='fixed_labels'.

Runs ANOVA full_search clustering ONCE on the initial student model across N
representative training prompts, aggregates which neurons were selected and
what label they received, and saves a JSON file keyed by "layer:neuron_id".

During distillation training, each step just looks up the label for each
selected neuron from this file — no ANOVA, no activation-write-cache reads.
Step time drops from ~20 min (full_search) to ~80s (same as ablation).

Usage (run once before training):
    python -m graph_loss.precompute_fixed_labels \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --dataset 22_add_tight_all \\
        --train-json "/path/to/22_add_tight_5000_train.json" \\
        --n-prompts 32 \\
        --mlp-input-cache "/path/to/mlp-input-cache" \\
        --activation-write-cache "/path/to/activation-write-cache-1b" \\
        --output "/path/to/fixed_labels_1b.json" \\
        --prop-neurons-per-layer 0.0005 \\
        --top-k-logits 200 \\
        --anova-nodes-per-label 3

The output JSON maps "layer:neuron_id" → label string, e.g.:
    {"12:7700": "carry", "15:5845": "sum units", ...}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_prompts(train_json: str, n_prompts: int) -> list[str]:
    with open(train_json, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        prompts = [entry["q_str"] for entry in data]
    elif isinstance(data, list):
        prompts = [
            entry["q_str"] if isinstance(entry, dict) and "q_str" in entry else str(entry)
            for entry in data
        ]
    else:
        raise ValueError(f"Unsupported train_json format: {type(data)}")
    return prompts[:n_prompts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HuggingFace model ID for student")
    parser.add_argument("--dataset", required=True, help="Dataset name for activation cache (e.g. 22_add_tight_all)")
    parser.add_argument("--train-json", required=True, help="Path to training JSON file (to pick prompts from)")
    parser.add_argument("--n-prompts", type=int, default=32, help="Number of training prompts to run ANOVA over")
    parser.add_argument("--mlp-input-cache", required=True, help="Path to pre-computed MLP input cache directory")
    parser.add_argument("--activation-write-cache", required=True, help="Path to activation-write-cache directory")
    parser.add_argument("--output", required=True, help="Output JSON path for the fixed label mapping")
    parser.add_argument("--prop-neurons-per-layer", type=float, default=0.0005)
    parser.add_argument("--top-k-logits", type=int, default=200)
    parser.add_argument("--anova-nodes-per-label", type=int, default=3)
    parser.add_argument("--anova-range-radius", type=int, default=0)
    parser.add_argument("--sum-min-specificity", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # --- Load model ---
    logger.info("Loading student model: %s", args.model)
    import sys, os as _os
    # Ensure the utils package (load_model) is importable when running as -m
    _src_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)

    from utils.hf_models import load_student_model_for_distillation
    from graph_loss.hf_adapter import HFLlamaGraphAdapter

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    student_model, tokenizer = load_student_model_for_distillation(
        student_source=None,
        student_model_id=args.model,
        device=device,
    )
    student_model = student_model.to(dtype=dtype)
    adapter = HFLlamaGraphAdapter(student_model, tokenizer, device)

    # --- Load prompts ---
    prompts = _load_prompts(args.train_json, args.n_prompts)
    logger.info("Using %d prompts for fixed-label pre-computation", len(prompts))

    # --- Load MLP input cache ---
    from graph_loss.precompute_mlp_inputs import load_mlp_input_cache, mlp_input_cache_exists
    from graph_loss.neuron_activation_heatmap import _resolve_dataset_path
    dataset_path = _resolve_dataset_path(args.dataset)
    model_name = getattr(adapter.model.config, "_name_or_path", args.model)
    mlp_input_cache = None
    if mlp_input_cache_exists(args.mlp_input_cache, model_name, dataset_path):
        mlp_input_cache = load_mlp_input_cache(args.mlp_input_cache, model_name, dataset_path)
        logger.info("Loaded MLP input cache for %s", model_name)
    else:
        logger.warning("MLP input cache not found — activation write cache will be built from scratch (slow)")

    # --- Run attribution + full_search for each prompt ---
    from graph_loss.graph import build_super_graph
    label_votes: Counter = Counter()
    neuron_labels: dict[str, list[str]] = {}

    for i, prompt in enumerate(prompts):
        logger.info("[%d/%d] Attributing: %r", i + 1, len(prompts), prompt)
        try:
            graph = adapter.build_graph(
                prompt,
                prop_neurons_per_layer=args.prop_neurons_per_layer,
                batch_size=1,
                dtype=dtype,
                verbose=False,
                create_graph=False,
                detach_result=True,
                fast=False,
                skip_logit_attribution=True,
            )
        except Exception as exc:
            logger.warning("  Attribution failed for %r: %s", prompt, exc)
            continue

        try:
            with torch.no_grad():
                sg = build_super_graph(
                    graph,
                    adapter,
                    activation_forward_batch_size=500,
                    computation_eps=0.05,
                    embedding_eps=0.1,
                    cluster_method="full_search",
                    dataset=args.dataset,
                    activation_write_cache_path=args.activation_write_cache,
                    mlp_input_cache=mlp_input_cache,
                    model_name=model_name,
                    anova_nodes_per_label=args.anova_nodes_per_label,
                    anova_range_radius=args.anova_range_radius,
                    sum_min_specificity=args.sum_min_specificity,
                )
        except Exception as exc:
            logger.warning("  Supergraph build failed for %r: %s", prompt, exc)
            continue

        if sg.supernode_labels is None:
            continue

        locations = graph.neuron_locations.detach().cpu().to(dtype=torch.long)
        for sn_idx, (members, labels) in enumerate(zip(sg.supernodes, sg.supernode_labels)):
            if not labels:
                continue
            label = labels[0]
            for member in members:
                layer = int(locations[member, 0].item())
                neuron_id = int(locations[member, 2].item())
                key = f"{layer}:{neuron_id}"
                neuron_labels.setdefault(key, []).append(label)
                label_votes[f"{key}:{label}"] += 1

        logger.info("  → %d labeled supernodes, running total: %d unique labeled neurons",
                    len(sg.supernodes), len(neuron_labels))

    # --- Resolve ties: pick majority label per neuron ---
    fixed_labels: dict[str, str] = {}
    for key, seen_labels in neuron_labels.items():
        majority = Counter(seen_labels).most_common(1)[0][0]
        fixed_labels[key] = majority

    logger.info(
        "Fixed label map: %d neurons labeled across %d prompts",
        len(fixed_labels), len(prompts),
    )
    label_counts: Counter = Counter(fixed_labels.values())
    for lbl, cnt in sorted(label_counts.items()):
        logger.info("  %-35s: %d neurons", lbl, cnt)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(fixed_labels, f, indent=2)
    logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()

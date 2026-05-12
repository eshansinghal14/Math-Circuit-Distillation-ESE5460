import argparse
import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

from tqdm import tqdm

import torch
from huggingface_hub import login
from transformers import AutoTokenizer

from graph_loss.align import compute_supernode_dla
from graph_loss.attribution.attribute import attribute
from graph_loss.precompute_mlp_inputs import load_mlp_input_cache, mlp_input_cache_exists
from graph_loss.__main__ import (
    _log_graph_summary,
    _log_pipeline_comparison,
    _log_prune_summary,
    _log_supergraph_summary,
)
from graph_loss.graph import (
    Graph,
    PruneResult,
    SuperGraph,
    build_super_graph,
    extract_supernode_members,
    prune_graph,
)
from graph_loss.replacement_model import TransformerLensReplacementModel
from graph_loss.utils import (
    add_graph_build_args,
    add_graph_prune_args,
    resolve_torch_dtype,
)
from utils import HF_READ_TOKEN, default_datasets_dir, load_prompt_answer_json, patch_tokenizer_no_special_tokens


MANIFEST_NAME = "manifest.json"
SHARD_MANIFEST_PATTERN = "manifest_shard_"


@dataclass
class TeacherDataConfig:
    store_path: str
    dataset_file: str
    teacher_model: str
    student_model: str
    dtype: str = "float32"
    top_k_logits: int | None = 20
    prop_neurons_per_layer: float = 0.1
    attribution_batch_size: int = 256
    verbose: bool = False
    prune: bool = False
    node_threshold: float = 0.8
    edge_threshold: float = 0.98
    limit: int | None = None
    start_index: int = 0
    overwrite: bool = False
    fast: bool = True
    cluster_method: str = "ablation"
    skip_graph_save: bool = True
    cossim_eps: float = 0.1
    embedding_sigma: float = 1.5
    embedding_eps: float = 0.1
    computation_sigma: float = 1.5
    computation_eps: float = 0.1
    dataset: str | None = None
    activation_forward_batch_size: int = 32
    activation_write_cache_path: str | None = None
    mlp_input_cache_path: str | None = None
    anova_nodes_per_label: int = 10
    anova_range_radius: int = 0
    sum_min_specificity: float = 0.0


def _resolve_dataset_file(dataset_file: str) -> str:
    expanded = os.path.expanduser(dataset_file)
    if os.path.dirname(expanded):
        return os.path.abspath(expanded)
    return os.path.join(default_datasets_dir(), os.path.basename(expanded))


def _safe_prompt_folder(prompt: str, index: int) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", prompt.strip())[:48].strip("._-")
    if not slug:
        slug = "prompt"
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
    return f"{index:06d}_{slug}_{digest}"


def _write_json(path: str, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _relative(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def _save_prune_result(path: str, prune_result: PruneResult) -> None:
    torch.save(
        {
            "node_mask": prune_result.node_mask,
            "edge_mask": prune_result.edge_mask,
            "cumulative_scores": prune_result.cumulative_scores,
        },
        path,
    )


def _save_supergraph(
    path: str, supergraph: SuperGraph, logit_token_ids: torch.Tensor | None = None
) -> None:
    data: dict[str, Any] = {
        "supernode_adjacency_matrix": supergraph.supernode_adjacency_matrix,
        "supernodes": supergraph.supernodes,
        "supernode_prob_deltas": supergraph.supernode_prob_deltas,
        "all_supernode_prob_delta_norms": supergraph.all_supernode_prob_delta_norms,
        "prob_delta_elbow_index": supergraph.prob_delta_elbow_index,
        "delta_node_indices": supergraph.delta_node_indices,
        "supernode_labels": supergraph.supernode_labels,
    }
    if logit_token_ids is not None:
        data["logit_token_ids"] = logit_token_ids.cpu()
    torch.save(data, path)


def _build_distillation_tensors(prompt: str, answer: int, tokenizer) -> dict[str, Any]:
    answer_text = str(answer)
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        add_special_tokens=False,
    )["input_ids"].squeeze(0)
    answer_ids = tokenizer(
        answer_text + tokenizer.eos_token,
        return_tensors="pt",
        padding=False,
        add_special_tokens=False,
    )["input_ids"].squeeze(0)
    input_ids = torch.cat([prompt_ids, answer_ids])
    attention_mask = torch.ones_like(input_ids)
    kl_mask = torch.zeros(input_ids.shape[0], dtype=torch.float32)
    kl_mask[: max(input_ids.shape[0] - 1, 0)] = 1.0
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "kl_mask": kl_mask,
        "prompt_len": int(prompt_ids.numel()),
        "answer_len": int(answer_ids.numel()),
    }


@torch.no_grad()
def _compute_teacher_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    input_ids = input_ids.to(model.cfg.device)
    if int(input_ids.max().item()) >= int(model.cfg.d_vocab):
        raise ValueError(
            "Distillation input ids exceed the teacher vocabulary. The teacher-data cache "
            "requires compatible student and teacher tokenizers."
        )
    output = model(input_ids.unsqueeze(0))
    logits = output.logits if hasattr(output, "logits") else output
    return logits.squeeze(0).detach().cpu()


@torch.no_grad()
def _save_teacher_supernode_dla(
    path: str,
    *,
    supergraph: SuperGraph,
    graph: Graph,
    model: TransformerLensReplacementModel,
) -> None:
    members = extract_supernode_members(supergraph, graph, model)
    W_U = model.unembed.W_U
    cluster_ids = [int(member["cluster_id"]) for member in members]
    dla_vectors = [
        compute_supernode_dla(member, W_U).detach().cpu()
        for member in members
    ]
    dla = (
        torch.stack(dla_vectors, dim=0)
        if dla_vectors
        else torch.empty((0, W_U.shape[1]), dtype=W_U.dtype)
    )
    torch.save(
        {
            "cluster_ids": cluster_ids,
            "teacher_ids": cluster_ids,
            "dla": dla,
        },
        path,
    )


def generate_teacher_data(config: TeacherDataConfig) -> dict[str, Any]:
    logger = logging.getLogger(__name__)
    store_path = os.path.abspath(config.store_path)
    dataset_path = _resolve_dataset_file(config.dataset_file)
    os.makedirs(store_path, exist_ok=True)

    data = load_prompt_answer_json(dataset_path)
    samples = list(data.items())
    if config.start_index:
        samples = samples[config.start_index:]
    if config.limit is not None:
        samples = samples[: config.limit]

    is_shard = config.start_index != 0 or config.limit is not None
    shard_end = config.start_index + len(samples) - 1 if samples else config.start_index
    if is_shard:
        manifest_filename = f"{SHARD_MANIFEST_PATTERN}{config.start_index:06d}_{shard_end:06d}.json"
    else:
        manifest_filename = MANIFEST_NAME

    if HF_READ_TOKEN:
        logger.info("Authenticating with Hugging Face token")
        login(HF_READ_TOKEN)

    dtype = resolve_torch_dtype(config.dtype)
    logger.info("Loading teacher model: %s", config.teacher_model)
    model = TransformerLensReplacementModel.from_pretrained(
        config.teacher_model,
        dtype=dtype,
    )
    model.eval()

    tokenizer = patch_tokenizer_no_special_tokens(
        AutoTokenizer.from_pretrained(config.student_model)
    )
    tokenizer.pad_token = tokenizer.eos_token

    mlp_input_cache: dict | None = None
    if config.mlp_input_cache_path:
        # The MLP input cache is keyed on the activation dataset (--dataset), not the
        # training dataset (--dataset-file), since it's built from the activation dataset.
        activation_dataset = config.dataset or config.dataset_file
        dataset_path_for_cache = _resolve_dataset_file(activation_dataset)
        if mlp_input_cache_exists(config.mlp_input_cache_path, config.teacher_model, dataset_path_for_cache):
            logger.info("Loading pre-computed MLP input cache from %s", config.mlp_input_cache_path)
            mlp_input_cache = load_mlp_input_cache(
                config.mlp_input_cache_path, config.teacher_model, dataset_path_for_cache
            )
            logger.info("  Loaded: %d prompts, %d layers", mlp_input_cache["meta"]["n_prompts"], mlp_input_cache["meta"]["n_layers"])
        else:
            logger.warning(
                "MLP input cache not found at %s for model=%s dataset=%s — falling back to forward passes",
                config.mlp_input_cache_path,
                config.teacher_model,
                dataset_path_for_cache,
            )

    manifest: dict[str, Any] = {
        "version": 1,
        "dataset_file": os.path.abspath(dataset_path),
        "store_path": store_path,
        "teacher_model": config.teacher_model,
        "student_model": config.student_model,
        "teacher_vocab_size": int(model.cfg.d_vocab),
        "hyperparameters": asdict(config),
        "samples": [],
    }

    for local_idx, (prompt, answer) in enumerate(tqdm(samples, desc="Generating teacher data", unit="sample")):
        sample_idx = config.start_index + local_idx
        folder_name = _safe_prompt_folder(prompt, sample_idx)
        sample_dir = os.path.join(store_path, folder_name)
        if os.path.exists(sample_dir) and not config.overwrite:
            raise FileExistsError(
                f"Sample output already exists: {sample_dir}. Use --overwrite to replace."
            )
        os.makedirs(sample_dir, exist_ok=True)

        logger.info("Generating teacher data for sample %d: %r", sample_idx, prompt)
        logger.info("Running attribution graph build (fast=%s)", config.fast)
        graph = attribute(
            prompt=prompt,
            model=model,
            top_k_logits=config.top_k_logits,
            prop_neurons_per_layer=config.prop_neurons_per_layer,
            batch_size=config.attribution_batch_size,
            verbose=config.verbose,
            fast=config.fast,
        )
        _log_graph_summary(graph, logger=logger, stage="Built")

        if not config.skip_graph_save:
            graph_path = os.path.join(sample_dir, "graph.pt")
            logger.info("Saving graph to %s", graph_path)
            graph.to_pt(graph_path)
        else:
            logger.info("Skipping graph.pt save (--skip-graph-save)")

        prune_result = None
        graph_for_supergraph = graph
        prune_path = None
        if config.prune:
            logger.info("Running prune_graph")
            prune_result = prune_graph(
                graph,
                node_threshold=config.node_threshold,
                edge_threshold=config.edge_threshold,
            )
            _log_prune_summary(
                graph,
                prune_result,
                node_threshold=config.node_threshold,
                edge_threshold=config.edge_threshold,
                logger=logger,
            )
            prune_path = os.path.join(sample_dir, "prune_result.pt")
            logger.info("Saving prune result to %s", prune_path)
            _save_prune_result(prune_path, prune_result)
            logger.info("Applying prune masks to graph")
            graph_for_supergraph = graph.apply_prune_result(prune_result)

        activation_dataset = config.dataset or config.dataset_file
        logger.info(
            "Running build_super_graph (cluster_method=%s, activation_dataset=%s)",
            config.cluster_method,
            activation_dataset,
        )
        with torch.no_grad():
            supergraph = build_super_graph(
                graph_for_supergraph,
                model,
                prune_result=prune_result,
                cossim_eps=config.cossim_eps,
                embedding_sigma=config.embedding_sigma,
                embedding_eps=config.embedding_eps,
                computation_sigma=config.computation_sigma,
                computation_eps=config.computation_eps,
                dataset=activation_dataset,
                activation_forward_batch_size=config.activation_forward_batch_size,
                activation_write_cache_path=config.activation_write_cache_path,
                mlp_input_cache=mlp_input_cache,
                model_name=config.teacher_model,
                cluster_method=config.cluster_method,
                anova_nodes_per_label=config.anova_nodes_per_label,
                anova_range_radius=config.anova_range_radius,
                sum_min_specificity=config.sum_min_specificity,
            )
        _log_supergraph_summary(
            graph_for_supergraph,
            supergraph,
            logger=logger,
        )
        supergraph_path = os.path.join(sample_dir, "supergraph.pt")
        logger.info("Saving supergraph to %s", supergraph_path)
        _save_supergraph(supergraph_path, supergraph, logit_token_ids=graph.logit_token_ids)
        _log_pipeline_comparison(
            graph_for_supergraph,
            supergraph,
            logger=logger,
            prune_result=prune_result,
        )

        dla_path = os.path.join(sample_dir, "teacher_supernode_dla.pt")
        logger.info("Saving teacher supernode DLA to %s", dla_path)
        with torch.no_grad():
            _save_teacher_supernode_dla(
                dla_path,
                supergraph=supergraph,
                graph=graph_for_supergraph,
                model=model,
            )

        logger.info("Computing teacher logits for distillation cache")
        distill_tensors = _build_distillation_tensors(prompt, answer, tokenizer)
        logits = _compute_teacher_logits(model, distill_tensors["input_ids"])
        logits_path = os.path.join(sample_dir, "teacher_logits.pt")
        logger.info("Saving teacher logits to %s", logits_path)
        torch.save(
            {
                "prompt": prompt,
                "answer": int(answer),
                "input_ids": distill_tensors["input_ids"].cpu(),
                "attention_mask": distill_tensors["attention_mask"].cpu(),
                "kl_mask": distill_tensors["kl_mask"].cpu(),
                "prompt_len": distill_tensors["prompt_len"],
                "answer_len": distill_tensors["answer_len"],
                "logits": logits,
            },
            logits_path,
        )

        metadata = {
            "sample_index": sample_idx,
            "prompt": prompt,
            "answer": int(answer),
            "folder": folder_name,
            "prompt_len": distill_tensors["prompt_len"],
            "answer_len": distill_tensors["answer_len"],
            "sequence_len": int(distill_tensors["input_ids"].numel()),
            "artifacts": {
                "graph": _relative(graph_path, sample_dir) if not config.skip_graph_save else None,
                "prune_result": _relative(prune_path, sample_dir) if prune_path else None,
                "supergraph": _relative(supergraph_path, sample_dir),
                "teacher_supernode_dla": _relative(dla_path, sample_dir),
                "teacher_logits": _relative(logits_path, sample_dir),
            },
        }
        metadata_path = os.path.join(sample_dir, "metadata.json")
        logger.info("Saving metadata to %s", metadata_path)
        _write_json(metadata_path, metadata)

        manifest["samples"].append(
            {
                "sample_index": sample_idx,
                "prompt": prompt,
                "answer": int(answer),
                "folder": folder_name,
                "metadata": _relative(metadata_path, store_path),
                "teacher_logits": _relative(logits_path, store_path),
                "supergraph": _relative(supergraph_path, store_path),
                "teacher_supernode_dla": _relative(dla_path, store_path),
                "graph": _relative(graph_path, store_path) if not config.skip_graph_save else None,
                "prune_result": _relative(prune_path, store_path) if prune_path else None,
            }
        )
        manifest_path = os.path.join(store_path, manifest_filename)
        logger.info("Updating manifest at %s", manifest_path)
        _write_json(manifest_path, manifest)
        logger.info("Completed teacher data for sample %d", sample_idx)

    return manifest


def merge_shard_manifests(store_path: str) -> dict[str, Any]:
    """Merge all manifest_shard_*.json files in store_path into manifest.json.

    Safe to call after all parallel shard jobs finish. Idempotent.
    """
    logger = logging.getLogger(__name__)
    store_path = os.path.abspath(store_path)
    shard_files = sorted(
        f for f in os.listdir(store_path) if f.startswith(SHARD_MANIFEST_PATTERN) and f.endswith(".json")
    )
    if not shard_files:
        raise FileNotFoundError(
            f"No shard manifests found in {store_path!r}. "
            f"Run generate_teacher_data.py with --start-index / --limit first."
        )
    logger.info("Found %d shard manifest(s): %s", len(shard_files), shard_files)

    merged: dict[str, Any] = {}
    all_samples: list[dict[str, Any]] = []
    for fname in shard_files:
        path = os.path.join(store_path, fname)
        with open(path, "r", encoding="utf-8") as f:
            shard = json.load(f)
        if not merged:
            merged = {k: v for k, v in shard.items() if k != "samples"}
        all_samples.extend(shard.get("samples", []))
        logger.info("  %s: %d samples", fname, len(shard.get("samples", [])))

    all_samples.sort(key=lambda s: s["sample_index"])
    merged["samples"] = all_samples

    manifest_path = os.path.join(store_path, MANIFEST_NAME)
    _write_json(manifest_path, merged)
    logger.info(
        "Merged %d total samples into %s", len(all_samples), manifest_path
    )
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate cached teacher graph/logit data for distillation.",
    )
    parser.add_argument("--store-path", required=True, help="Directory to write teacher cache")
    parser.add_argument(
        "--dataset-file",
        required=True,
        help="Dataset JSON filename under datasets/ or an explicit path",
    )
    parser.add_argument("--teacher-model", required=True, help="Teacher HuggingFace model name")
    parser.add_argument(
        "--student-model",
        required=True,
        help="Student model/tokenizer name used to build distillation input ids",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional number of samples")
    parser.add_argument("--start-index", type=int, default=0, help="Dataset index to start at")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing sample folders")
    parser.add_argument(
        "--fast",
        action="store_true",
        default=True,
        help="Use fast attribution (skip Jacobian; linear logit readout only). Default: True.",
    )
    parser.add_argument(
        "--no-fast",
        action="store_false",
        dest="fast",
        help="Use full Jacobian attribution instead of fast mode.",
    )
    parser.add_argument(
        "--cluster-method",
        "--cluster_method",
        dest="cluster_method",
        choices=("full_search", "ablation"),
        default="ablation",
        help="Supergraph clustering method (default: ablation)",
    )
    parser.add_argument(
        "--cossim_eps",
        type=float,
        default=0.1,
        help="Deprecated fallback clustering epsilon; use --embedding_eps and --computation_eps",
    )
    parser.add_argument(
        "--embedding_sigma",
        type=float,
        default=1.5,
        help="Gaussian smoothing sigma for numeric-token embedding supernode clustering",
    )
    parser.add_argument(
        "--embedding_eps",
        type=float,
        default=0.1,
        help="Angular distance epsilon for numeric-token embedding supernode clustering",
    )
    parser.add_argument(
        "--computation_sigma",
        type=float,
        default=1.5,
        help="Gaussian smoothing sigma for final-token computation supernode clustering",
    )
    parser.add_argument(
        "--computation_eps",
        type=float,
        default=0.1,
        help="Angular distance epsilon for final-token computation supernode clustering",
    )
    parser.add_argument(
        "--dataset",
        help=(
            "Optional dataset prefix, filename, or path for activation-write "
            "output-neuron clustering and per-cluster PDF heatmaps"
        ),
    )
    parser.add_argument(
        "--activation_forward_batch_size",
        type=int,
        default=32,
        help="Batch size for dataset activation-write forward passes",
    )
    parser.add_argument(
        "--activation-write-cache-path",
        "--activation_write_cache_path",
        dest="activation_write_cache_path",
        help=(
            "Optional cache root for dataset activation-write results. "
            "A model-name folder is created inside this path."
        ),
    )
    parser.add_argument(
        "--mlp-input-cache",
        "--mlp_input_cache_path",
        dest="mlp_input_cache_path",
        default=None,
        help=(
            "Path to a pre-computed MLP residual-stream input cache (generated by "
            "python -m graph_loss.precompute_mlp_inputs). When provided, "
            "build_neuron_activation_write_result uses cached hidden states instead "
            "of running fresh forward passes — typically 10-100x faster."
        ),
    )
    parser.add_argument(
        "--skip-graph-save",
        action="store_true",
        default=True,
        help="Skip saving the full graph.pt file (~21MB/sample, not needed for training). Default: True.",
    )
    parser.add_argument(
        "--no-skip-graph-save",
        action="store_false",
        dest="skip_graph_save",
        help="Save the full graph.pt file for each sample.",
    )
    parser.add_argument(
        "--anova-nodes-per-label",
        "--anova_nodes_per_label",
        dest="anova_nodes_per_label",
        type=int,
        default=10,
        help="Maximum positive-variance ANOVA nodes to include per label supernode",
    )
    parser.add_argument(
        "--anova-range-radius",
        "--anova_range_radius",
        dest="anova_range_radius",
        type=int,
        default=0,
        help=(
            "Radius around the target arg1/arg2 values for ANOVA range basis masks. "
            "Use 0 for exact target-value masks."
        ),
    )
    parser.add_argument(
        "--sum-min-specificity",
        "--sum_min_specificity",
        dest="sum_min_specificity",
        type=float,
        default=0.0,
        help="Minimum ANOVA specificity required for sum-label supernodes ranked by DLA cosine",
    )
    add_graph_build_args(parser)
    add_graph_prune_args(parser)
    parser.add_argument(
        "--merge-shards",
        action="store_true",
        help=(
            "Instead of generating data, merge all manifest_shard_*.json files in "
            "--store-path into a single manifest.json. Run this after all parallel "
            "shard jobs complete."
        ),
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()

    if args.merge_shards:
        if not args.store_path:
            raise SystemExit("--merge-shards requires --store-path")
        merge_shard_manifests(args.store_path)
        return

    config = TeacherDataConfig(
        store_path=args.store_path,
        dataset_file=args.dataset_file,
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        dtype=args.dtype,
        top_k_logits=args.top_k_logits,
        prop_neurons_per_layer=args.prop_neurons_per_layer,
        attribution_batch_size=args.attribution_batch_size,
        verbose=args.verbose,
        prune=args.prune,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        limit=args.limit,
        start_index=args.start_index,
        overwrite=args.overwrite,
        fast=args.fast,
        cluster_method=args.cluster_method,
        skip_graph_save=args.skip_graph_save,
        cossim_eps=args.cossim_eps,
        embedding_sigma=args.embedding_sigma,
        embedding_eps=args.embedding_eps,
        computation_sigma=args.computation_sigma,
        computation_eps=args.computation_eps,
        dataset=args.dataset,
        activation_forward_batch_size=args.activation_forward_batch_size,
        activation_write_cache_path=args.activation_write_cache_path,
        mlp_input_cache_path=args.mlp_input_cache_path,
        anova_nodes_per_label=args.anova_nodes_per_label,
        anova_range_radius=args.anova_range_radius,
        sum_min_specificity=args.sum_min_specificity,
    )
    generate_teacher_data(config)


if __name__ == "__main__":
    main()

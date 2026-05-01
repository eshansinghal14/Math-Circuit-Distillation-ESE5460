import argparse
import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

import torch
from huggingface_hub import login
from transformers import AutoTokenizer

from graph_loss.align import compute_supernode_dla
from graph_loss.attribution.attribute import attribute
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


@dataclass
class TeacherDataConfig:
    store_path: str
    dataset_file: str
    teacher_model: str
    student_model: str
    dtype: str = "float32"
    top_k_logits: int | None = 20
    prop_neurons_per_layer: float = 0.1
    batch_size: int = 512
    verbose: bool = False
    prune: bool = False
    node_threshold: float = 0.8
    edge_threshold: float = 0.98
    limit: int | None = None
    start_index: int = 0
    overwrite: bool = False


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


def _save_supergraph(path: str, supergraph: SuperGraph) -> None:
    torch.save(
        {
            "supernode_adjacency_matrix": supergraph.supernode_adjacency_matrix,
            "supernodes": supergraph.supernodes,
            "supernode_prob_deltas": supergraph.supernode_prob_deltas,
            "all_supernode_prob_delta_norms": supergraph.all_supernode_prob_delta_norms,
            "prob_delta_elbow_index": supergraph.prob_delta_elbow_index,
        },
        path,
    )


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

    for local_idx, (prompt, answer) in enumerate(samples):
        sample_idx = config.start_index + local_idx
        folder_name = _safe_prompt_folder(prompt, sample_idx)
        sample_dir = os.path.join(store_path, folder_name)
        if os.path.exists(sample_dir) and not config.overwrite:
            raise FileExistsError(
                f"Sample output already exists: {sample_dir}. Use --overwrite to replace."
            )
        os.makedirs(sample_dir, exist_ok=True)

        logger.info("Generating teacher data for sample %d: %r", sample_idx, prompt)
        logger.info("Running attribution graph build")
        graph = attribute(
            prompt=prompt,
            model=model,
            top_k_logits=config.top_k_logits,
            prop_neurons_per_layer=config.prop_neurons_per_layer,
            batch_size=config.batch_size,
            verbose=config.verbose,
        )
        _log_graph_summary(graph, logger=logger, stage="Built")

        graph_path = os.path.join(sample_dir, "graph.pt")
        logger.info("Saving graph to %s", graph_path)
        graph.to_pt(graph_path)

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

        logger.info("Running build_super_graph")
        supergraph = build_super_graph(
            graph_for_supergraph,
            model,
            prune_result=prune_result,
        )
        _log_supergraph_summary(
            graph_for_supergraph,
            supergraph,
            logger=logger,
        )
        supergraph_path = os.path.join(sample_dir, "supergraph.pt")
        logger.info("Saving supergraph to %s", supergraph_path)
        _save_supergraph(supergraph_path, supergraph)
        _log_pipeline_comparison(
            graph_for_supergraph,
            supergraph,
            logger=logger,
            prune_result=prune_result,
        )

        dla_path = os.path.join(sample_dir, "teacher_supernode_dla.pt")
        logger.info("Saving teacher supernode DLA to %s", dla_path)
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
                "graph": _relative(graph_path, sample_dir),
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
                "graph": _relative(graph_path, store_path),
                "prune_result": _relative(prune_path, store_path) if prune_path else None,
            }
        )
        manifest_path = os.path.join(store_path, MANIFEST_NAME)
        logger.info("Updating manifest at %s", manifest_path)
        _write_json(manifest_path, manifest)
        logger.info("Completed teacher data for sample %d", sample_idx)

    return manifest


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
    add_graph_build_args(parser)
    add_graph_prune_args(parser)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    config = TeacherDataConfig(
        store_path=args.store_path,
        dataset_file=args.dataset_file,
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        dtype=args.dtype,
        top_k_logits=args.top_k_logits,
        prop_neurons_per_layer=args.prop_neurons_per_layer,
        batch_size=args.batch_size,
        verbose=args.verbose,
        prune=args.prune,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        limit=args.limit,
        start_index=args.start_index,
        overwrite=args.overwrite,
    )
    generate_teacher_data(config)


if __name__ == "__main__":
    main()

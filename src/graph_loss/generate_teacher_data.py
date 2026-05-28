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

from transformers import AutoModelForCausalLM
from graph_loss.create_graph import (
    GraphPipelineResult,
    create_graph,
    save_supergraph,
)

from graph_loss.hf_adapter import HFLlamaGraphAdapter
from graph_loss.utils import (
    add_graph_build_args,
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
    dtype: str = "float32"
    top_k_logits: float | None = 0.95
    temperature: float = 2.0
    prop_neurons_per_layer: float = 0.1
    attribution_batch_size: int = 256
    verbose: bool = False
    limit: int | None = None
    start_index: int = 0
    overwrite: bool = False
    dataset: str | None = None

    anova_nodes_per_label: int = 10
    anova_range_radius: int = 0
    sum_min_specificity: float = 0.0
    graph_node_labels: list[str] | None = None
    include_dla_node: bool = False
    include_arg_nodes: bool = False

    cot_k_positions: int = 0
    cot_max_new_tokens: int = 256
    test_gsm8k: bool = False
    test_amount: int = 50


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


def _build_distillation_tensors(prompt: str, answer: int | str, tokenizer) -> dict[str, Any]:
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
def _compute_teacher_logits(adapter: HFLlamaGraphAdapter, input_ids: torch.Tensor) -> torch.Tensor:
    input_ids = input_ids.to(adapter.cfg.device)
    if int(input_ids.max().item()) >= int(adapter.cfg.d_vocab):
        raise ValueError(
            "Distillation input ids exceed the teacher vocabulary."
        )
    output = adapter.model(input_ids.unsqueeze(0))
    logits = output.logits if hasattr(output, "logits") else output
    return logits.squeeze(0).detach().cpu()


@torch.no_grad()
def _generate_cot_sample(
    *,
    adapter: "HFLlamaGraphAdapter",
    tokenizer,
    prompt: str,
    answer: str,
    sample_dir: str,
    config: "TeacherDataConfig",
    activation_dataset: str | None,
    mlp_input_cache: dict | None,
    logger: logging.Logger,
) -> None:
    """Generate K CoT supergraphs for one prompt and save them to sample_dir."""
    import torch.nn.functional as F

    device = adapter.cfg.device
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        add_special_tokens=False,
    )["input_ids"].to(device)

    output = adapter.model.generate(
        prompt_ids,
        max_new_tokens=config.cot_max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_logits=True,
    )
    generated_ids = output.sequences[0, prompt_ids.shape[1]:]
    # output.logits: tuple of T tensors, each [1, vocab_size] (one per generated token)
    gen_len = len(generated_ids)

    # Compute per-token entropy, excluding the very last generated position (EOS guard)
    entropies: list[float] = []
    for t in range(gen_len - 1):
        logits_t = output.logits[t][0]  # [vocab_size]
        log_prob = F.log_softmax(logits_t.float(), dim=-1)
        prob = log_prob.exp()
        H_t = float(-(prob * log_prob).sum().item())
        entropies.append(H_t)

    K = min(config.cot_k_positions, len(entropies))
    if K == 0:
        logger.warning("CoT trace too short to select any positions; skipping CoT supergraphs")
        _write_json(
            os.path.join(sample_dir, "cot_metadata.json"),
            {"k_positions": 0, "trace_token_positions": [], "trace_text": ""},
        )
        return

    # Select K positions with LOWEST entropy (most confident)
    sorted_by_entropy = sorted(range(len(entropies)), key=lambda i: entropies[i])
    selected_positions = sorted(sorted_by_entropy[:K])  # keep in trace order

    trace_token_positions: list[int] = []
    prompt_ids_cpu = prompt_ids[0].cpu()

    for k, pos in enumerate(selected_positions):
        t_k = pos + 1  # 1-indexed into generated trace tokens
        extended_ids = torch.cat([prompt_ids_cpu, generated_ids[:t_k].cpu()])
        extended_prefix = tokenizer.decode(extended_ids.tolist(), skip_special_tokens=True)
        dla_logits = output.logits[pos][0].detach().cpu()  # logits at position t_k

        logger.info("  CoT k=%d: t_k=%d, prefix length=%d tokens", k, t_k, len(extended_ids))
        result: GraphPipelineResult = create_graph(
            adapter,
            extended_prefix,
            top_k_logits=config.top_k_logits,
            temperature=config.temperature,
            prop_neurons_per_layer=config.prop_neurons_per_layer,
            batch_size=config.attribution_batch_size,
            verbose=config.verbose,
            dataset=activation_dataset,
            mlp_input_cache=mlp_input_cache,
            model_name=config.teacher_model,
            anova_nodes_per_label=config.anova_nodes_per_label,
            anova_range_radius=config.anova_range_radius,
            sum_min_specificity=config.sum_min_specificity,
            node_labels=config.graph_node_labels,
            include_dla_node=config.include_dla_node,
            include_arg_nodes=config.include_arg_nodes,
            no_grad_supergraph=True,
            logger=logger,
        )

        sg = result.supergraph
        sg_data: dict[str, Any] = {
            "supernode_adjacency_matrix": sg.supernode_adjacency_matrix,
            "supernodes": sg.supernodes,
            "node_labels": sg.node_labels,
            "supernode_labels": sg.supernode_labels,
            "supernode_heatmap_pdf_paths": sg.supernode_heatmap_pdf_paths,
            "prefix": extended_prefix,
            "dla_logits": dla_logits,
        }
        logit_token_ids = result.graph.logit_token_ids
        if logit_token_ids is not None:
            sg_data["logit_token_ids"] = logit_token_ids.cpu()

        supergraph_path = os.path.join(sample_dir, f"supergraph_k{k}.pt")
        torch.save(sg_data, supergraph_path)
        logger.info("  Saved CoT supergraph to %s", supergraph_path)
        trace_token_positions.append(t_k)

    trace_text = tokenizer.decode(generated_ids.cpu().tolist(), skip_special_tokens=True)
    _write_json(
        os.path.join(sample_dir, "cot_metadata.json"),
        {
            "k_positions": K,
            "trace_token_positions": trace_token_positions,
            "trace_text": trace_text,
            "answer": answer,
        },
    )


def generate_teacher_data(config: TeacherDataConfig) -> dict[str, Any]:
    logger = logging.getLogger(__name__)
    store_path = os.path.abspath(config.store_path)
    os.makedirs(store_path, exist_ok=True)

    if config.test_gsm8k:
        if config.cot_k_positions == 0:
            raise ValueError(
                "--test-gsm8k requires --cot-k-positions > 0 (GSM8K data is CoT format)."
            )
        from datasets import load_dataset as _load_hf_dataset
        _ds = _load_hf_dataset("openai/gsm8k", "main", split="train")
        samples: list = [
            (row["question"], row["answer"])
            for row in _ds.select(range(config.test_amount))
        ]
        dataset_path = None
    else:
        dataset_path = _resolve_dataset_file(config.dataset_file)
        if config.cot_k_positions > 0:
            from utils.dataset_json import load_cot_json
            data: dict[str, Any] = load_cot_json(dataset_path)
        else:
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading teacher model: %s", config.teacher_model)
    hf_model = AutoModelForCausalLM.from_pretrained(config.teacher_model, torch_dtype=dtype)
    hf_model = hf_model.to(device)
    hf_model.eval()

    teacher_tokenizer = patch_tokenizer_no_special_tokens(
        AutoTokenizer.from_pretrained(config.teacher_model)
    )
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    adapter = HFLlamaGraphAdapter(hf_model, teacher_tokenizer, device)

    manifest: dict[str, Any] = {
        "version": 1,
        "dataset_file": os.path.abspath(dataset_path) if dataset_path else "gsm8k",
        "store_path": store_path,
        "teacher_model": config.teacher_model,
        "teacher_vocab_size": int(adapter.d_vocab),
        "hyperparameters": asdict(config),
        "samples": [],
    }

    # Build MLP input cache once for the entire run — same across all prompts.
    # Only needed for ANOVA neuron labeling, which is skipped when graph_node_labels
    # is None, so skip the (expensive) cache build in that case.
    activation_dataset = config.dataset or config.dataset_file
    if activation_dataset and config.graph_node_labels is not None:
        from graph_loss.precompute_mlp_inputs import build_mlp_input_cache
        from graph_loss.neuron_activation_heatmap import _resolve_dataset_path
        activation_dataset_path = _resolve_dataset_path(activation_dataset)
        logger.info("Building MLP input cache once for dataset: %s", activation_dataset)
        mlp_input_cache = build_mlp_input_cache(
            adapter,
            activation_dataset_path,
            config.teacher_model,
            batch_size=32,
        )
        n_prompts = int(mlp_input_cache.get("meta", {}).get("n_prompts", 0))
        logger.info("  MLP cache: %d prompts, ready for reuse across all training prompts", n_prompts)
    else:
        if activation_dataset and config.graph_node_labels is None:
            logger.info("Skipping MLP input cache (no --graph-node-labels, ANOVA not needed)")
        mlp_input_cache = None

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

        if config.cot_k_positions > 0:
            _generate_cot_sample(
                adapter=adapter,
                tokenizer=teacher_tokenizer,
                prompt=prompt,
                answer=answer,
                sample_dir=sample_dir,
                config=config,
                activation_dataset=activation_dataset,
                mlp_input_cache=mlp_input_cache,
                logger=logger,
            )
            manifest["samples"].append(
                {
                    "sample_index": sample_idx,
                    "prompt": prompt,
                    "answer": answer,
                    "folder": folder_name,
                }
            )
        else:
            result: GraphPipelineResult = create_graph(
                adapter,
                prompt,
                top_k_logits=config.top_k_logits,
                temperature=config.temperature,
                prop_neurons_per_layer=config.prop_neurons_per_layer,
                batch_size=config.attribution_batch_size,
                verbose=config.verbose,
                dataset=activation_dataset,
                mlp_input_cache=mlp_input_cache,
                model_name=config.teacher_model,
                anova_nodes_per_label=config.anova_nodes_per_label,
                anova_range_radius=config.anova_range_radius,
                sum_min_specificity=config.sum_min_specificity,
                node_labels=config.graph_node_labels,
                include_dla_node=config.include_dla_node,
                include_arg_nodes=config.include_arg_nodes,
                no_grad_supergraph=True,
                logger=logger,
            )

            supergraph_path = os.path.join(sample_dir, "supergraph.pt")
            logger.info("Saving supergraph to %s", supergraph_path)
            save_supergraph(
                supergraph_path,
                result.supergraph,
                logit_token_ids=result.graph.logit_token_ids,
            )

            logger.info("Computing teacher logits for distillation cache")
            distill_tensors = _build_distillation_tensors(prompt, answer, teacher_tokenizer)
            logits = _compute_teacher_logits(adapter, distill_tensors["input_ids"])
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
                    "supergraph": _relative(supergraph_path, sample_dir),
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
                }
            )
            del result, distill_tensors, logits

        manifest_path = os.path.join(store_path, manifest_filename)
        logger.info("Updating manifest at %s", manifest_path)
        _write_json(manifest_path, manifest)
        logger.info("Completed teacher data for sample %d", sample_idx)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        default=None,
        help="Dataset JSON filename under datasets/ or an explicit path",
    )
    parser.add_argument(
        "--test-gsm8k",
        action="store_true",
        dest="test_gsm8k",
        help="Load GSM8K train split from HuggingFace instead of --dataset-file.",
    )
    parser.add_argument(
        "--test-amount",
        type=int,
        default=50,
        dest="test_amount",
        help="Number of GSM8K examples to use when --test-gsm8k is set (default 50).",
    )
    parser.add_argument("--teacher-model", required=True, help="Teacher HuggingFace model name")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of samples")
    parser.add_argument("--start-index", type=int, default=0, help="Dataset index to start at")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing sample folders")
    parser.add_argument(
        "--dataset",
        help=(
            "Optional dataset prefix, filename, or path for activation-write "
            "output-neuron clustering and per-cluster PDF heatmaps"
        ),
    )

    parser.add_argument(
        "--cot-k-positions",
        type=int,
        default=0,
        dest="cot_k_positions",
        help=(
            "Number of low-entropy positions to select from the CoT trace. "
            "0 (default) disables CoT mode and behaves exactly as before."
        ),
    )
    parser.add_argument(
        "--cot-max-new-tokens",
        type=int,
        default=256,
        dest="cot_max_new_tokens",
        help="Max new tokens to generate for the CoT trace (default 256).",
    )
    add_graph_build_args(parser)
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

    if not args.test_gsm8k and not args.dataset_file:
        raise SystemExit("Either --dataset-file or --test-gsm8k must be provided.")

    config = TeacherDataConfig(
        store_path=args.store_path,
        dataset_file=args.dataset_file,
        teacher_model=args.teacher_model,
        dtype=args.dtype,
        top_k_logits=args.top_k_logits,
        temperature=args.temperature,
        prop_neurons_per_layer=args.prop_neurons_per_layer,
        attribution_batch_size=args.attribution_batch_size,
        verbose=args.verbose,
        limit=args.limit,
        start_index=args.start_index,
        overwrite=args.overwrite,
        dataset=args.dataset,

        anova_nodes_per_label=args.anova_nodes_per_label,
        anova_range_radius=args.anova_range_radius,
        sum_min_specificity=args.sum_min_specificity,
        graph_node_labels=args.graph_node_labels,
        include_dla_node=args.include_dla_node,
        include_arg_nodes=args.include_arg_nodes,

        cot_k_positions=args.cot_k_positions,
        cot_max_new_tokens=args.cot_max_new_tokens,
        test_gsm8k=args.test_gsm8k,
        test_amount=args.test_amount,
    )
    if config.test_gsm8k:
        config.store_path = os.path.join(config.store_path, f"gsm8k_{config.test_amount}")
    generate_teacher_data(config)


if __name__ == "__main__":
    main()

"""Neuron-cluster ablation study.

For each subclass discovered by the circuit model, loads k-means neuron
clusters, ablates each cluster independently, and records the resulting
accuracy drop.  The output ``ablation_performance.json`` is consumed by
``cluster_pairing.py`` to pair student/teacher clusters by importance.

Can be used as a library (``classify_problems``, ``ablation``,
``apply_ablation``) or as a CLI script.
"""

import json
import os
import sys
import argparse
from typing import Dict, List, Optional, Tuple

import torch


from utils import (
    load_model_checkpoint,
    load_model,
    test_model,
    eval_model,
)
from circuit_discovery.utils import parse_equation

DEFAULT_CLASSIFIED_PROBLEMS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "circuit-discovery", "classified_problems.json"
)


def classify_problems(
    circuit_model,
    tokenizer,
    dataset_path: str = None,
    output_path: str = None,
    batch_size: int = 256,
) -> Dict[str, List]:
    """Classify every problem in the dataset into latent subclasses.

    Args:
        circuit_model: A trained ``CircuitDiscoveryModel`` (will be used in eval mode).
        tokenizer: HuggingFace tokenizer for decoding token ids.
        dataset_path: Path to ``2d_add_all.json``.  Defaults to ``../datasets/2d_add_all.json``.
        output_path: Where to save the JSON mapping.  ``None`` skips saving.
        batch_size: Inference batch size.

    Returns:
        ``{subclass_str: [(problem_str, token_ids), ...]}``
    """
    device = next(circuit_model.parameters()).device

    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), "..", "datasets", "2d_add_all.json")

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    ids = [record["ids"] for record in dataset]
    ids = torch.tensor(ids).to(device)
    prompts = tokenizer.batch_decode(ids, skip_special_tokens=True)

    class_to_problems: Dict[str, List] = {}

    circuit_model.eval()
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]

        op1, op2, res = parse_equation(batch_prompts, device=device)
        with torch.no_grad():
            logits = circuit_model.classify_problem(op1, op2, res)
            subclass = torch.argmax(logits, dim=-1)

        for prob, cls, tid in zip(batch_prompts, subclass.tolist(), batch_ids.tolist()):
            key = str(cls)
            class_to_problems.setdefault(key, []).append((prob, tid))

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(class_to_problems, f, indent=2)
        print(f"Saved classified problems to {output_path}")

    for key in sorted(class_to_problems, key=int):
        print(f"  Subclass {key}: {len(class_to_problems[key])} problems")

    return class_to_problems


def ablation(
    model_name: str,
    tokenizer,
    class_to_problems: Dict[str, List],
    class_clusters: Optional[List[int]] = None,
    results_base_dir: str = None,
    clusters_base_dir: str = None,
    batch_size: int = 50,
    max_new_tokens: int = 1,
) -> Dict:
    """Run per-cluster ablation study for a single model.

    For every subclass, loads the k-means cluster file, ablates each cluster
    in turn, and records the accuracy relative to the un-ablated baseline.

    Args:
        model_name: HuggingFace model identifier (e.g. ``meta-llama/Llama-3.2-1B``).
        tokenizer: Tokenizer matching ``model_name``.
        class_to_problems: Output of ``classify_problems``.
        class_clusters: List of *k* values per subclass.  Defaults to ``[6]*8``.
        results_base_dir: Root for ablation output files.
        clusters_base_dir: Root where neuron-clustering results live.
        batch_size: Evaluation batch size.
        max_new_tokens: Tokens to generate during eval.

    Returns:
        Nested dict saved as ``ablation_performance.json``::

            {subclass: {"baseline": float, "clusters": {cluster_id: accuracy}}}
    """
    if class_clusters is None:
        class_clusters = [6] * 8

    if results_base_dir is None:
        results_base_dir = os.path.join(os.path.dirname(__file__), "..", "results", "circuit-discovery", model_name)
    os.makedirs(results_base_dir, exist_ok=True)

    if clusters_base_dir is None:
        clusters_base_dir = os.path.join(os.path.dirname(__file__), "..", "results", "neuron-clustering", model_name)

    out_path = os.path.join(results_base_dir, "ablation_performance.json")
    buffer_results_path = os.path.join(results_base_dir, "ablation_results_buffer.json")

    ablation_results: Dict = {}

    if batch_size <= 0:
        batch_size = 50

    model, _ = load_model(model_name)
    model.eval()

    try:
        for subclass in range(len(class_clusters)):
            print(f"Processing subclass {subclass}")
            subclass_str = str(subclass)
            problems = class_to_problems.get(subclass_str, [])
            if not problems:
                continue

            subclass_dataset_path = os.path.join(results_base_dir, f"class_{subclass_str}_dataset.json")

            dataset = []
            for problem_str, _ in problems:
                if "=" not in problem_str:
                    continue
                lhs, rhs = problem_str.split("=", 1)
                ans_str = rhs.strip()
                if not ans_str.isdigit():
                    continue
                q_str = lhs + "="
                a_str = ans_str
                tok_ids = tokenizer.encode(q_str + a_str, add_special_tokens=False)
                dataset.append({"q_str": q_str, "a_str": a_str, "ids": tok_ids})

            with open(subclass_dataset_path, "w") as f:
                json.dump(dataset, f, indent=2)

            # Baseline (no ablation)
            _ = test_model(
                model,
                tokenizer,
                subclass_dataset_path,
                buffer_results_path,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                log=False,
            )
            baseline_acc = eval_model(buffer_results_path)
            print(f"  Subclass {subclass}: baseline accuracy = {baseline_acc:.4f}")

            k = class_clusters[subclass]
            clusters_path = os.path.join(clusters_base_dir, f"subclass_{subclass}_clusters/k{k}.pt")

            if not os.path.exists(clusters_path):
                print(f"  Skipping subclass {subclass}: missing cluster file {clusters_path}")
                continue

            ckpt = torch.load(clusters_path, map_location="cpu")
            cluster_to_indices = ckpt["cluster_to_indices"]

            subclass_result = {"baseline": baseline_acc, "clusters": {}}

            for cluster_id, neuron_indices in cluster_to_indices.items():
                handles = apply_activation_ablation_hooks(model, neuron_indices)
                try:
                    _ = test_model(
                        model,
                        tokenizer,
                        subclass_dataset_path,
                        buffer_results_path,
                        batch_size=batch_size,
                        max_new_tokens=max_new_tokens,
                        log=False,
                    )
                finally:
                    remove_ablation_hooks(handles)
                acc = eval_model(buffer_results_path)
                print(f"    Cluster {cluster_id}: accuracy = {acc:.4f}")

                subclass_result["clusters"][str(cluster_id)] = acc

            ablation_results[subclass_str] = subclass_result

        with open(out_path, "w") as f:
            json.dump(ablation_results, f, indent=2)

        print(f"Saved ablation performance to {out_path}")
        return ablation_results
    finally:
        try:
            del model
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _layer_neuron_map_from_flat_indices(model, neuron_indices) -> Dict[int, List[int]]:
    if not hasattr(model, "config"):
        return {}

    cfg = model.config
    if not hasattr(cfg, "intermediate_size") or not hasattr(cfg, "num_hidden_layers"):
        return {}

    intermediate_size = int(cfg.intermediate_size)
    num_layers = int(cfg.num_hidden_layers)

    if isinstance(neuron_indices, torch.Tensor):
        idx_list = neuron_indices.view(-1).tolist()
    else:
        idx_list = list(neuron_indices)

    layer_to_neurons: Dict[int, List[int]] = {}
    for idx in idx_list:
        if not isinstance(idx, int):
            try:
                idx = int(idx)
            except Exception:
                continue
        if idx < 0:
            continue
        layer_id = idx // intermediate_size
        neuron_id = idx % intermediate_size
        if layer_id < 0 or layer_id >= num_layers:
            continue
        layer_to_neurons.setdefault(layer_id, []).append(neuron_id)

    for lid in list(layer_to_neurons.keys()):
        layer_to_neurons[lid] = sorted(set(layer_to_neurons[lid]))

    return layer_to_neurons


def apply_activation_ablation_hooks(model, neuron_indices):
    """Temporarily ablate neurons by masking the MLP down-projection input.

    This avoids in-place weight mutation and allows reusing a single loaded model.
    Neurons are specified as flattened indices across all MLP layers, matching
    the neuron-clustering output format.

    Returns a list of hook handles; call ``remove_ablation_hooks`` after eval.
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        return []

    layer_to_neurons = _layer_neuron_map_from_flat_indices(model, neuron_indices)
    if not layer_to_neurons:
        return []

    layers = model.model.layers
    handles = []

    for layer_id, neuron_ids in layer_to_neurons.items():
        if layer_id < 0 or layer_id >= len(layers):
            continue

        block = layers[layer_id]
        if not hasattr(block, "mlp"):
            continue

        mlp = block.mlp
        down = getattr(mlp, "down_proj", None)
        if down is None:
            continue

        def _make_pre_hook(_neuron_ids: List[int]):
            def _pre_hook(module, inputs):
                if not inputs:
                    return inputs
                x = inputs[0]
                if not torch.is_tensor(x):
                    return inputs
                if x.dim() < 2:
                    return inputs

                with torch.no_grad():
                    x2 = x.clone()
                    x2[..., _neuron_ids] = 0
                return (x2,) + tuple(inputs[1:])

            return _pre_hook

        handle = down.register_forward_pre_hook(_make_pre_hook(neuron_ids))
        handles.append(handle)

    return handles


def remove_ablation_hooks(handles) -> None:
    for h in handles or []:
        try:
            h.remove()
        except Exception:
            pass


def apply_ablation(model, neuron_indices):
    """Zero-out specific neurons (by flattened index across all MLP layers)."""
    if not hasattr(model, "config"):
        return model

    cfg = model.config
    if not hasattr(cfg, "intermediate_size") or not hasattr(cfg, "num_hidden_layers"):
        return model

    intermediate_size = cfg.intermediate_size
    num_layers = cfg.num_hidden_layers

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        return model

    layers = model.model.layers

    if isinstance(neuron_indices, torch.Tensor):
        idx_list = neuron_indices.view(-1).tolist()
    else:
        idx_list = list(neuron_indices)

    with torch.no_grad():
        for idx in idx_list:
            if not isinstance(idx, int):
                try:
                    idx = int(idx)
                except Exception:
                    continue

            if idx < 0:
                continue

            layer_id = idx // intermediate_size
            neuron_id = idx % intermediate_size

            if layer_id < 0 or layer_id >= num_layers:
                continue

            block = layers[layer_id]
            if not hasattr(block, "mlp"):
                continue

            mlp = block.mlp
            gate = getattr(mlp, "gate_proj", None)
            up = getattr(mlp, "up_proj", None)
            down = getattr(mlp, "down_proj", None)

            if gate is not None and hasattr(gate, "weight"):
                if 0 <= neuron_id < gate.weight.shape[0]:
                    gate.weight[neuron_id].zero_()
                    if getattr(gate, "bias", None) is not None and neuron_id < gate.bias.shape[0]:
                        gate.bias[neuron_id].zero_()

            if up is not None and hasattr(up, "weight"):
                if 0 <= neuron_id < up.weight.shape[0]:
                    up.weight[neuron_id].zero_()
                    if getattr(up, "bias", None) is not None and neuron_id < up.bias.shape[0]:
                        up.bias[neuron_id].zero_()

            if down is not None and hasattr(down, "weight"):
                if 0 <= neuron_id < down.weight.shape[1]:
                    down.weight[:, neuron_id].zero_()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neuron cluster ablation study")
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="HuggingFace model identifier (e.g. meta-llama/Llama-3.2-1B)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Path to circuit-discovery model checkpoint (.pt file), or 'latest'",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=7,
        help="Number of clusters (k) to use per subclass",
    )
    parser.add_argument(
        "--clusters-dir",
        type=str,
        default=None,
        help="Root directory containing subclass_N_clusters/ folders",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory to write ablation results into",
    )
    parser.add_argument(
        "--k-classes",
        type=int,
        default=8,
        help="Number of latent subclasses in the circuit-discovery model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for evaluation",
    )
    args = parser.parse_args()

    _model_name = args.model_name
    _, _tokenizer = load_model(_model_name)

    _circuit_model, _, _, _ = load_model_checkpoint(
        args.checkpoint, k_classes=args.k_classes, lr=1e-3
    )
    _circuit_model.eval()

    _classified_path = DEFAULT_CLASSIFIED_PROBLEMS_PATH

    if os.path.exists(_classified_path):
        with open(_classified_path, "r") as f:
            _class_to_problems = json.load(f)
        print(f"Loaded classified problems from {_classified_path}")
    else:
        _class_to_problems = classify_problems(
            _circuit_model, _tokenizer, output_path=_classified_path
        )

    del _circuit_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _class_clusters = [args.k] * args.k_classes

    ablation(
        _model_name,
        _tokenizer,
        _class_to_problems,
        class_clusters=_class_clusters,
        results_base_dir=args.results_dir,
        clusters_base_dir=args.clusters_dir,
        batch_size=args.batch_size
    )

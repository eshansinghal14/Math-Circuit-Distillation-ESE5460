import json
import torch
import os
import sys
import copy

import torch.nn.functional as F
from transformers import AutoTokenizer

from utils import (
    load_model_checkpoint,
    load_model,
    get_model_name,
    test_model,
    eval_model,
)
from circuit_discovery.utils import parse_equation

model_name = get_model_name(sys.argv)
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model, tokenizer = load_model(sys.argv)
checkpoint_name = "model_2000"
circuit_model, _, _, _ = load_model_checkpoint(checkpoint_name, k_classes=8, lr=1e-3)
circuit_model.eval()

if model_name == "meta-llama/Llama-3.2-1B":
    class_clusters = [3] * 8
else:
    class_clusters = [3] * 8


def classify_problems(batch_size=256):
    device = next(circuit_model.parameters()).device

    dataset_path = os.path.join("..", "datasets", "2d_add_all.json")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    ids = []
    for record in dataset:
        ids.append(record['ids'])
    ids = torch.tensor(ids).to(device)
    prompts = tokenizer.batch_decode(ids, skip_special_tokens=True)

    class_to_problems = {}

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]

        op1, op2, res, term_encoding = parse_equation(batch_prompts, device=device)
        with torch.no_grad():
            logits = circuit_model.classify_problem(op1, op2, res)
            subclass = torch.argmax(logits, dim=-1)  # [batch]

        for prob, cls, id in zip(batch_prompts, subclass.tolist(), batch_ids.tolist()):
            key = str(cls)
            if key not in class_to_problems:
                class_to_problems[key] = []
            class_to_problems[key].append((prob, id))

    results_dir = os.path.join("..", "results", "circuit-discovery")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "classified_problems.json")
    with open(out_path, "w") as f:
        json.dump(class_to_problems, f, indent=2)

    print(f"Saved classified problems to {out_path}")
    for key in class_to_problems:
        print(key, len(class_to_problems[key]))
    
    return class_to_problems


def ablation(class_to_problems):
    results_dir = os.path.join("..", "results", "circuit-discovery")
    os.makedirs(results_dir, exist_ok=True)

    out_path = os.path.join(results_dir, "ablation_performance.json")
    buffer_results_path = os.path.join(results_dir, "ablation_results_buffer.json")

    ablation_results = {}

    for subclass in range(len(class_clusters)):
        print(f"Processing subclass {subclass}")
        subclass_str = str(subclass)
        problems = class_to_problems.get(subclass_str, [])
        if not problems:
            continue

        # Build a subclass-specific dataset JSON based on problem strings
        subclass_dataset_path = os.path.join(results_dir, f"class_{subclass_str}_dataset.json")

        dataset = {}
        for problem_str, _ in problems:
            if "=" not in problem_str:
                continue
            lhs, rhs = problem_str.split("=", 1)
            ans_str = rhs.strip()
            if not ans_str.isdigit():
                continue
            prompt = lhs + "="
            dataset[prompt] = int(ans_str)

        with open(subclass_dataset_path, "w") as f:
            json.dump(dataset, f, indent=2)

        # Baseline accuracy using the original test_model/eval_model pipeline
        _ = test_model(base_model, tokenizer, subclass_dataset_path, buffer_results_path, log=False)
        baseline_acc = eval_model(buffer_results_path)
        print(f"Subclass {subclass}: baseline accuracy = {baseline_acc:.4f}")

        k = class_clusters[subclass]
        clusters_dir = os.path.join("..", "results", "neuron-clustering", model_name)
        clusters_path = os.path.join(clusters_dir, f"subclass_{subclass}_clusters/k{k}.pt")

        if not os.path.exists(clusters_path):
            print(f"  Skipping subclass {subclass}: missing cluster file {clusters_path}")
            continue

        ckpt = torch.load(clusters_path, map_location="cpu")
        cluster_to_indices = ckpt["cluster_to_indices"]

        subclass_result = {
            "baseline": baseline_acc,
            "clusters": {},
        }

        for cluster_id, neuron_indices in cluster_to_indices.items():
            ablated_model = apply_ablation(base_model, neuron_indices)

            with torch.no_grad():
                base_gate = base_model.model.layers[0].mlp.gate_proj.weight
                abl_gate = ablated_model.model.layers[0].mlp.gate_proj.weight

            # Accuracy of ablated model on the same subclass dataset
            _ = test_model(ablated_model, tokenizer, subclass_dataset_path, buffer_results_path, log=False)
            acc = eval_model(buffer_results_path)
            print(f"  Cluster {cluster_id}: accuracy = {acc:.4f}")

            subclass_result["clusters"][str(cluster_id)] = acc

        ablation_results[subclass_str] = subclass_result

    with open(out_path, "w") as f:
        json.dump(ablation_results, f, indent=2)

    print(f"Saved ablation performance to {out_path}")
    return ablation_results


def apply_ablation(model, neuron_indices):
    """Return a copy of `model` with the specified MLP neurons ablated.

    `neuron_indices` is a 1D tensor of global indices over all MLP neurons,
    assuming a flattened ordering of [layer * intermediate_size + neuron].
    This implementation is tailored to LLaMA-style models used in this repo.
    """

    if not hasattr(model, "config"):
        # Fallback: no ablation if model structure is unexpected
        return model

    cfg = model.config
    if not hasattr(cfg, "intermediate_size") or not hasattr(cfg, "num_hidden_layers"):
        return model

    intermediate_size = cfg.intermediate_size
    num_layers = cfg.num_hidden_layers

    # Create a fresh copy so we don't mutate the shared base_model
    ablated_model = copy.deepcopy(model)

    # For LLaMA-like models, transformer blocks are in ablated_model.model.layers
    if not hasattr(ablated_model, "model") or not hasattr(ablated_model.model, "layers"):
        return ablated_model

    layers = ablated_model.model.layers

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

            # Zero out the selected neuron in gate/up, and its contribution in down
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

    return ablated_model


if __name__ == "__main__":
    class_to_problems = classify_problems()
    ablation(class_to_problems)

"""Full MLP layer ablation study.

For each MLP layer in a model, zeroes out the entire layer output and measures
the resulting accuracy drop.  The output ``layer_ablation_performance.json``
maps layer indices to accuracy, used by ``layer_pairing.py`` to match student
and teacher layers by functional importance.

Usage (from src/):
  python -m ffn_distillation.layer_ablation \
      --model-name meta-llama/Llama-3.2-1B \
      --results-dir /path/to/results
"""

import argparse
import json
import os
from typing import Dict, List

import torch

from utils import load_model, test_model, eval_model


def _zero_mlp_hook(module, inputs, output):
    """Forward hook that zeroes out the entire MLP layer output."""
    if isinstance(output, tuple):
        return tuple(torch.zeros_like(o) for o in output)
    return torch.zeros_like(output)


def layer_ablation(
    model_name: str,
    dataset_path: str,
    results_dir: str,
    batch_size: int = 50,
    max_new_tokens: int = 1,
) -> Dict:
    """Ablate each MLP layer one at a time and record accuracy.

    Args:
        model_name: HuggingFace model identifier.
        dataset_path: Path to a JSON evaluation dataset (``2d_add_all.json`` format).
        results_dir: Where to write results.
        batch_size: Evaluation batch size.
        max_new_tokens: Tokens to generate per example.

    Returns:
        ``{"baseline": float, "layers": {"0": float, "1": float, ...}}``
    """
    os.makedirs(results_dir, exist_ok=True)
    buffer_path = os.path.join(results_dir, "_layer_abl_buffer.json")
    out_path = os.path.join(results_dir, "layer_ablation_performance.json")

    model, tokenizer = load_model(model_name)
    model.eval()
    num_layers = model.config.num_hidden_layers

    # Baseline accuracy (no ablation)
    test_model(model, tokenizer, dataset_path, buffer_path,
               batch_size=batch_size, max_new_tokens=max_new_tokens, log=False)
    baseline = eval_model(buffer_path)
    print(f"Baseline accuracy: {baseline:.4f}")

    results: Dict = {"baseline": baseline, "layers": {}}

    for layer_idx in range(num_layers):
        mlp = model.model.layers[layer_idx].mlp
        handle = mlp.register_forward_hook(_zero_mlp_hook)

        try:
            test_model(model, tokenizer, dataset_path, buffer_path,
                       batch_size=batch_size, max_new_tokens=max_new_tokens, log=False)
            acc = eval_model(buffer_path)
        finally:
            handle.remove()

        drop = baseline - acc
        print(f"  Layer {layer_idx:2d}: acc={acc:.4f}  drop={drop:.4f}")
        results["layers"][str(layer_idx)] = acc

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved layer ablation to {out_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Full MLP layer ablation study")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset", type=str,
                        default=os.path.join(script_dir, "..", "..", "datasets", "2d_add_all.json"))
    parser.add_argument("--results-dir", type=str,
                        default=os.path.join(script_dir, "..", "..", "results", "ffn-layer-ablation"))
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    layer_ablation(
        model_name=args.model_name,
        dataset_path=args.dataset,
        results_dir=os.path.join(args.results_dir, args.model_name),
        batch_size=args.batch_size,
    )

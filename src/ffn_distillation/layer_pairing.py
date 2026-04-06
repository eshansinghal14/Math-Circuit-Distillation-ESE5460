"""MLP layer pairing between student and teacher.

Pairs student MLP layers to teacher MLP layers by matching their ablation-based
importance (accuracy drop when each layer is zeroed out).

Primary strategy: importance matching (closest |delta_s - delta_t|).
Fallback (if no ablation data): proportional mapping
    student layer i  ->  teacher layer round(i * T / S)

Usage (from src/):
  python -m ffn_distillation.layer_pairing \
      --student-ablation results/ffn-layer-ablation/meta-llama/Llama-3.2-1B/layer_ablation_performance.json \
      --teacher-ablation results/ffn-layer-ablation/meta-llama/Meta-Llama-3-8B/layer_ablation_performance.json
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class LayerPairInfo:
    """A single student-teacher MLP layer pair."""
    student_layer: int
    teacher_layer: int
    student_importance: float
    teacher_importance: float
    distance: float


def load_layer_importance(path: str) -> Tuple[float, Dict[int, float]]:
    """Load layer ablation results and compute importance (accuracy drop).

    Returns:
        (baseline_accuracy, {layer_idx: importance})
    """
    with open(path, "r") as f:
        data = json.load(f)

    baseline = float(data["baseline"])
    importance: Dict[int, float] = {}
    for layer_str, acc in data["layers"].items():
        drop = baseline - float(acc)
        importance[int(layer_str)] = max(drop, 0.0)

    return baseline, importance


def create_layer_mapping_by_importance(
    student_importance: Dict[int, float],
    teacher_importance: Dict[int, float],
    normalize: bool = True,
) -> List[LayerPairInfo]:
    """Match each student layer to the teacher layer with closest importance.

    For each student layer, finds the teacher layer minimizing
    |delta_s - delta_t|.  Greedy 1-to-1 matching (no teacher reuse).
    """
    s_imp = dict(student_importance)
    t_imp = dict(teacher_importance)

    if normalize:
        s_max = max(s_imp.values()) if s_imp else 1.0
        t_max = max(t_imp.values()) if t_imp else 1.0
        if s_max > 0:
            s_imp = {k: v / s_max for k, v in s_imp.items()}
        if t_max > 0:
            t_imp = {k: v / t_max for k, v in t_imp.items()}

    # Sort student layers by importance descending so that the most
    # important student layers get first pick of teacher layers.
    student_sorted = sorted(s_imp.items(), key=lambda x: x[1], reverse=True)
    available_teacher = set(t_imp.keys())

    pairs: List[LayerPairInfo] = []

    for s_layer, s_score in student_sorted:
        if not available_teacher:
            break

        best_t = None
        best_dist = float("inf")
        best_t_score = 0.0

        for t_layer in available_teacher:
            d = abs(s_score - t_imp[t_layer])
            if d < best_dist:
                best_dist = d
                best_t = t_layer
                best_t_score = t_imp[t_layer]

        if best_t is not None:
            pairs.append(LayerPairInfo(
                student_layer=s_layer,
                teacher_layer=best_t,
                student_importance=s_score,
                teacher_importance=best_t_score,
                distance=best_dist,
            ))
            available_teacher.discard(best_t)

    pairs.sort(key=lambda p: p.student_layer)
    return pairs


def create_proportional_mapping(
    student_layers: int,
    teacher_layers: int,
) -> List[LayerPairInfo]:
    """Fallback: map student layer i to teacher layer round(i * T / S)."""
    pairs = []
    for s in range(student_layers):
        t = round(s * teacher_layers / student_layers)
        t = min(t, teacher_layers - 1)
        pairs.append(LayerPairInfo(
            student_layer=s,
            teacher_layer=t,
            student_importance=0.0,
            teacher_importance=0.0,
            distance=0.0,
        ))
    return pairs


def get_layer_pairs(
    student_ablation_path: Optional[str],
    teacher_ablation_path: Optional[str],
    student_num_layers: int = 16,
    teacher_num_layers: int = 32,
) -> List[LayerPairInfo]:
    """Build layer pairs, preferring importance-based matching."""
    if (student_ablation_path and os.path.exists(student_ablation_path)
            and teacher_ablation_path and os.path.exists(teacher_ablation_path)):
        _, s_imp = load_layer_importance(student_ablation_path)
        _, t_imp = load_layer_importance(teacher_ablation_path)
        return create_layer_mapping_by_importance(s_imp, t_imp)

    print("  Using proportional layer mapping (no ablation data)")
    return create_proportional_mapping(student_num_layers, teacher_num_layers)


def save_layer_pairs(pairs: List[LayerPairInfo], path: str):
    data = [
        {
            "student_layer": p.student_layer,
            "teacher_layer": p.teacher_layer,
            "student_importance": p.student_importance,
            "teacher_importance": p.teacher_importance,
            "distance": p.distance,
        }
        for p in pairs
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_layer_pairs(path: str) -> List[LayerPairInfo]:
    with open(path, "r") as f:
        data = json.load(f)
    return [LayerPairInfo(**d) for d in data]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLP layer pairing")
    parser.add_argument("--student-ablation", type=str, default=None)
    parser.add_argument("--teacher-ablation", type=str, default=None)
    parser.add_argument("--student-layers", type=int, default=16)
    parser.add_argument("--teacher-layers", type=int, default=32)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    pairs = get_layer_pairs(
        args.student_ablation,
        args.teacher_ablation,
        args.student_layers,
        args.teacher_layers,
    )

    print(f"\nLayer pairs ({len(pairs)}):")
    for p in pairs:
        print(f"  Student {p.student_layer:2d} -> Teacher {p.teacher_layer:2d}  "
              f"(s_imp={p.student_importance:.3f}, t_imp={p.teacher_importance:.3f}, "
              f"dist={p.distance:.3f})")

    if args.output:
        save_layer_pairs(pairs, args.output)
        print(f"\nSaved to {args.output}")

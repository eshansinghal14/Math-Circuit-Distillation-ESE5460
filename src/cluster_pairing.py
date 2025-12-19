"""Cluster Pairing Module for Circuit Distillation

Pairs neuron clusters between two models (e.g., student and teacher)
using ablation-based cluster importance derived from
``results/circuit-discovery/<model-name>/ablation_performance.json``.

Each ablation_performance.json has the structure:
{
  "<subclass>": {
    "baseline": <float>,          # baseline accuracy for this subclass
    "clusters": {
      "<cluster_id>": <float>,    # accuracy when this cluster is ablated
      ...
    }
  },
  ...
}

We convert these into importance scores per subclass and cluster via:
    Δ(subclass, cluster) = baseline - accuracy_with_cluster_ablated
and then match clusters between models by minimizing |Δ_s - Δ_t| within
each subclass.
"""

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class ClusterMapping:
    """Represents a mapping between student and teacher neuron clusters."""

    subclass: int
    student_cluster_idx: int
    teacher_cluster_idx: int
    student_importance: float
    teacher_importance: float
    distance: float  # |delta_s - delta_t|


def _load_single_ablation_performance(path: str) -> Dict[int, Dict[int, float]]:
    """Load ablation_performance.json and return importance per subclass/cluster.

    Returns:
        Dict[subclass][cluster_id] -> importance (performance drop).
    """

    with open(path, "r") as f:
        data = json.load(f)

    result: Dict[int, Dict[int, float]] = {}

    for subclass_str, entry in data.items():
        baseline = float(entry["baseline"])
        clusters = entry["clusters"]
        subclass = int(subclass_str)

        # Importance is drop in performance when the cluster is ablated.
        # Clamp at 0 in case of numerical noise where ablated accuracy > baseline.
        inner: Dict[int, float] = {}
        for c_str, acc in clusters.items():
            c = int(c_str)
            drop = baseline - float(acc)
            if drop < 0:
                drop = 0.0
            inner[c] = drop

        result[subclass] = inner

    return result


def _normalize_nested_scores(
    scores: Dict[int, Dict[int, float]],
    baseline_per_subclass: Optional[Dict[int, float]] = None,
) -> Dict[int, Dict[int, float]]:
    """Normalize nested scores per subclass.

    If baseline_per_subclass is not provided, uses the max score within each
    subclass as reference. If a subclass has all-zero scores, it is left
    unchanged.
    """

    out: Dict[int, Dict[int, float]] = {}

    for subclass, cluster_scores in scores.items():
        if not cluster_scores:
            out[subclass] = {}
            continue

        if baseline_per_subclass is not None and subclass in baseline_per_subclass:
            baseline = baseline_per_subclass[subclass]
        else:
            baseline = max(cluster_scores.values()) if cluster_scores else 1.0

        if baseline == 0:
            out[subclass] = dict(cluster_scores)
            continue

        out[subclass] = {cid: v / baseline for cid, v in cluster_scores.items()}

    return out


def create_cluster_mapping(
    delta_s: Dict[int, Dict[int, float]],
    delta_t: Dict[int, Dict[int, float]],
    normalize: bool = True,
    top_k_student: Optional[int] = None,
    top_k_teacher: Optional[int] = None,
) -> List[ClusterMapping]:
    """Create mappings from student clusters to teacher clusters.

    For each subclass that exists in both models, and for each student cluster
    within that subclass, finds the teacher cluster that minimizes

        d_abl = |Δ_s(subclass, c_s) - Δ_t(subclass, c_t)|.

    Args:
        delta_s: Dict[subclass][cluster_id] -> performance drop for student.
        delta_t: Dict[subclass][cluster_id] -> performance drop for teacher.
        normalize: Whether to normalize scores within each subclass.
        top_k_student: If set, only map the top-k most important student
            clusters per subclass.
        top_k_teacher: If set, only consider the top-k teacher clusters per
            subclass as candidates.

    Returns:
        List of ClusterMapping objects sorted by (subclass, student_importance
        descending within subclass).
    """

    # Optionally normalize scores per subclass
    if normalize:
        delta_s = _normalize_nested_scores(delta_s)
        delta_t = _normalize_nested_scores(delta_t)

    mappings: List[ClusterMapping] = []

    # Only consider subclasses present in both models
    common_subclasses = sorted(set(delta_s.keys()) & set(delta_t.keys()))

    for subclass in common_subclasses:
        s_scores = dict(delta_s[subclass])
        t_scores = dict(delta_t[subclass])

        if not s_scores or not t_scores:
            continue

        # Filter to top-k if specified (per subclass)
        if top_k_student is not None:
            sorted_s = sorted(s_scores.items(), key=lambda x: x[1], reverse=True)
            s_scores = dict(sorted_s[:top_k_student])

        if top_k_teacher is not None:
            sorted_t = sorted(t_scores.items(), key=lambda x: x[1], reverse=True)
            t_scores = dict(sorted_t[:top_k_teacher])

        for s_idx, s_score in s_scores.items():
            best_t_idx: Optional[int] = None
            best_distance = float("inf")
            best_t_score = 0.0

            for t_idx, t_score in t_scores.items():
                distance = abs(s_score - t_score)
                if distance < best_distance:
                    best_distance = distance
                    best_t_idx = t_idx
                    best_t_score = t_score

            if best_t_idx is not None:
                mappings.append(
                    ClusterMapping(
                        subclass=subclass,
                        student_cluster_idx=s_idx,
                        teacher_cluster_idx=best_t_idx,
                        student_importance=s_score,
                        teacher_importance=best_t_score,
                        distance=best_distance,
                    )
                )

    # Sort first by subclass, then by student importance within each subclass
    mappings.sort(key=lambda m: (m.subclass, -m.student_importance))

    return mappings


def get_paired_indices(
    student_ablation_path: str,
    teacher_ablation_path: str,
    top_k_per_subclass: Optional[int] = None,
) -> Dict[int, Dict[int, int]]:
    """Convenience: get subclass-wise student->teacher cluster index mapping.

    Args:
        student_ablation_path: Path to student's ablation_performance.json.
        teacher_ablation_path: Path to teacher's ablation_performance.json.
        top_k_per_subclass: If set, only map the top-k most important student
            clusters per subclass.

    Returns:
        Dict[subclass][student_cluster_idx] = teacher_cluster_idx.
    """

    delta_s = _load_single_ablation_performance(student_ablation_path)
    delta_t = _load_single_ablation_performance(teacher_ablation_path)

    mappings = create_cluster_mapping(
        delta_s,
        delta_t,
        top_k_student=top_k_per_subclass,
    )

    out: Dict[int, Dict[int, int]] = {}
    for m in mappings:
        if m.subclass not in out:
            out[m.subclass] = {}
        out[m.subclass][m.student_cluster_idx] = m.teacher_cluster_idx

    return out


def analyze_mapping(mappings: List[ClusterMapping]) -> Dict:
    """Analyze the quality and structure of cluster mappings."""

    if not mappings:
        return {}

    distances = [m.distance for m in mappings]
    s_importance = [m.student_importance for m in mappings]
    t_importance = [m.teacher_importance for m in mappings]

    # Track teacher cluster reuse per (subclass, cluster)
    teacher_usage: Dict[Tuple[int, int], int] = {}
    for m in mappings:
        key = (m.subclass, m.teacher_cluster_idx)
        teacher_usage[key] = teacher_usage.get(key, 0) + 1

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "num_pairs": len(mappings),
        "mean_distance": mean(distances),
        "max_distance": max(distances),
        "min_distance": min(distances),
        "mean_student_importance": mean(s_importance),
        "mean_teacher_importance": mean(t_importance),
        "unique_teacher_clusters": len(teacher_usage),
        "teacher_cluster_reuse": {str(k): v for k, v in teacher_usage.items() if v > 1},
    }


def save_mapping(mappings: List[ClusterMapping], output_path: str) -> None:
    """Save mappings to JSON file."""

    data = [
        {
            "subclass": m.subclass,
            "student_cluster_idx": m.student_cluster_idx,
            "teacher_cluster_idx": m.teacher_cluster_idx,
            "student_importance": m.student_importance,
            "teacher_importance": m.teacher_importance,
            "distance": m.distance,
        }
        for m in mappings
    ]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_mapping(input_path: str) -> List[ClusterMapping]:
    """Load mappings from JSON file."""

    with open(input_path, "r") as f:
        data = json.load(f)

    return [ClusterMapping(**item) for item in data]


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Usage:
    #   python cluster_pairing.py <student_model_name> <teacher_model_name> [top_k_per_subclass]
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print(
            "Usage: python cluster_pairing.py <student_model_name> <teacher_model_name> [top_k_per_subclass]",
        )
        sys.exit(1)

    student_model_name = sys.argv[1]
    teacher_model_name = sys.argv[2]

    top_k_per_subclass: Optional[int]
    if len(sys.argv) == 4:
        try:
            top_k_per_subclass = int(sys.argv[3])
        except ValueError:
            print("top_k_per_subclass must be an integer if provided")
            sys.exit(1)
    else:
        top_k_per_subclass = None

    base_results_dir = os.path.join(script_dir, "..", "results", "circuit-discovery")

    student_ablation_path = os.path.join(
        base_results_dir,
        student_model_name,
        "ablation_performance.json",
    )
    teacher_ablation_path = os.path.join(
        base_results_dir,
        teacher_model_name,
        "ablation_performance.json",
    )

    if not os.path.exists(student_ablation_path):
        print(f"Student ablation file not found: {student_ablation_path}")
        sys.exit(1)

    if not os.path.exists(teacher_ablation_path):
        print(f"Teacher ablation file not found: {teacher_ablation_path}")
        sys.exit(1)

    print("Loading ablation performance...")
    delta_s = _load_single_ablation_performance(student_ablation_path)
    delta_t = _load_single_ablation_performance(teacher_ablation_path)

    print("Creating cluster mappings...")
    mappings = create_cluster_mapping(
        delta_s,
        delta_t,
        top_k_student=top_k_per_subclass,
    )

    stats = analyze_mapping(mappings)

    print("\nCluster mapping statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Save mappings to a JSON file named after the models.
    os.makedirs(base_results_dir, exist_ok=True)
    out_name = f"cluster_mapping_{student_model_name.replace('/', '_')}_to_{teacher_model_name.replace('/', '_')}.json"
    output_path = os.path.join(base_results_dir, out_name)

    save_mapping(mappings, output_path)
    print(f"\nSaved cluster mappings to {output_path}")

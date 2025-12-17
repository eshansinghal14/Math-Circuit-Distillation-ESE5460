"""
Head Pairing Module for Circuit Distillation

Implements ablation-based head mapping strategy from Section 2.2 of 
"Circuit Distillation" (arxiv:2509.25002).

The idea: pair student and teacher heads based on similar functional importance,
measured by performance drop when each head is ablated.
"""

import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class HeadMapping:
    """Represents a mapping between student and teacher components."""
    student_idx: int
    teacher_idx: int
    student_importance: float
    teacher_importance: float
    distance: float  # |delta_s - delta_t|


def load_ablation_scores(cache_path: str) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Load ablation scores from cache file.
    
    Args:
        cache_path: Path to ablation_cache.json
        
    Returns:
        Tuple of (student_scores, teacher_scores) dictionaries
        mapping component index -> ablation impact (performance drop)
    """
    with open(cache_path, 'r') as f:
        data = json.load(f)
    
    # Convert string keys to int
    delta_s = {int(k): v for k, v in data['delta_s'].items()}
    delta_t = {int(k): v for k, v in data['delta_t'].items()}
    
    return delta_s, delta_t


def normalize_ablation_scores(
    scores: Dict[int, float],
    baseline_performance: Optional[float] = None
) -> Dict[int, float]:
    """
    Normalize ablation scores to be relative to baseline.
    
    If baseline is not provided, uses max score as reference.
    This follows the paper's approach of computing impact relative
    to the student's operational capabilities.
    """
    if baseline_performance is None:
        baseline_performance = max(scores.values()) if scores else 1.0
    
    if baseline_performance == 0:
        return scores
    
    return {k: v / baseline_performance for k, v in scores.items()}


def create_head_mapping(
    delta_s: Dict[int, float],
    delta_t: Dict[int, float],
    normalize: bool = True,
    top_k_student: Optional[int] = None,
    top_k_teacher: Optional[int] = None
) -> List[HeadMapping]:
    """
    Create mapping from student heads to teacher heads based on ablation similarity.
    
    For each student head h_s, finds the teacher head h_t that minimizes:
        d_abl(h_s, h_t) = |Δ_s(h_s) - Δ_t(h_t)|
    
    Args:
        delta_s: Student ablation scores (head_idx -> performance drop)
        delta_t: Teacher ablation scores (head_idx -> performance drop)
        normalize: Whether to normalize scores before matching
        top_k_student: If set, only map the top-k most important student heads
        top_k_teacher: If set, only consider top-k teacher heads as candidates
        
    Returns:
        List of HeadMapping objects sorted by student importance (descending)
    """
    # Optionally normalize scores
    if normalize:
        delta_s = normalize_ablation_scores(delta_s)
        delta_t = normalize_ablation_scores(delta_t)
    
    # Filter to top-k if specified
    if top_k_student is not None:
        sorted_student = sorted(delta_s.items(), key=lambda x: x[1], reverse=True)
        delta_s = dict(sorted_student[:top_k_student])
    
    if top_k_teacher is not None:
        sorted_teacher = sorted(delta_t.items(), key=lambda x: x[1], reverse=True)
        delta_t = dict(sorted_teacher[:top_k_teacher])
    
    mappings = []
    
    for s_idx, s_score in delta_s.items():
        # Find teacher head with minimum distance
        best_t_idx = None
        best_distance = float('inf')
        best_t_score = 0.0
        
        for t_idx, t_score in delta_t.items():
            distance = abs(s_score - t_score)
            if distance < best_distance:
                best_distance = distance
                best_t_idx = t_idx
                best_t_score = t_score
        
        if best_t_idx is not None:
            mappings.append(HeadMapping(
                student_idx=s_idx,
                teacher_idx=best_t_idx,
                student_importance=s_score,
                teacher_importance=best_t_score,
                distance=best_distance
            ))
    
    # Sort by student importance (most important first)
    mappings.sort(key=lambda m: m.student_importance, reverse=True)
    
    return mappings


def get_paired_indices(
    cache_path: str,
    top_k: Optional[int] = None
) -> Dict[int, int]:
    """
    Convenience function to get simple student->teacher index mapping.
    
    Args:
        cache_path: Path to ablation_cache.json
        top_k: Only return top-k most important pairs
        
    Returns:
        Dictionary mapping student_idx -> teacher_idx
    """
    delta_s, delta_t = load_ablation_scores(cache_path)
    mappings = create_head_mapping(delta_s, delta_t, top_k_student=top_k)
    
    return {m.student_idx: m.teacher_idx for m in mappings}


def analyze_mapping(mappings: List[HeadMapping]) -> Dict:
    """
    Analyze the quality of head mappings.
    
    Returns statistics useful for understanding the pairing.
    """
    if not mappings:
        return {}
    
    distances = [m.distance for m in mappings]
    s_importance = [m.student_importance for m in mappings]
    t_importance = [m.teacher_importance for m in mappings]
    
    # Check for teacher heads used multiple times
    teacher_usage = {}
    for m in mappings:
        teacher_usage[m.teacher_idx] = teacher_usage.get(m.teacher_idx, 0) + 1
    
    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0
    
    return {
        'num_pairs': len(mappings),
        'mean_distance': mean(distances),
        'max_distance': max(distances),
        'min_distance': min(distances),
        'mean_student_importance': mean(s_importance),
        'mean_teacher_importance': mean(t_importance),
        'unique_teacher_heads': len(teacher_usage),
        'teacher_head_reuse': {k: v for k, v in teacher_usage.items() if v > 1}
    }


def save_mapping(mappings: List[HeadMapping], output_path: str):
    """Save mappings to JSON file."""
    data = [
        {
            'student_idx': m.student_idx,
            'teacher_idx': m.teacher_idx,
            'student_importance': m.student_importance,
            'teacher_importance': m.teacher_importance,
            'distance': m.distance
        }
        for m in mappings
    ]
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_mapping(input_path: str) -> List[HeadMapping]:
    """Load mappings from JSON file."""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    return [HeadMapping(**item) for item in data]


if __name__ == '__main__':
    import os
    
    # Get path to ablation cache
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(script_dir, '..', 'results', 'ablation_cache.json')
    
    # Load and create mapping
    delta_s, delta_t = load_ablation_scores(cache_path)
    
    print("Student ablation scores (delta_s):")
    for idx in sorted(delta_s.keys()):
        print(f"  MLP block {idx}: {delta_s[idx]:.4f}")
    
    print("\nTeacher ablation scores (delta_t):")
    for idx in sorted(delta_t.keys()):
        print(f"  MLP block {idx}: {delta_t[idx]:.4f}")
    
    # Create mapping
    mappings = create_head_mapping(delta_s, delta_t)
    
    print("\nHead mappings (student -> teacher):")
    for m in mappings:
        print(f"  S[{m.student_idx}] -> T[{m.teacher_idx}] "
              f"(s_imp={m.student_importance:.4f}, "
              f"t_imp={m.teacher_importance:.4f}, "
              f"dist={m.distance:.4f})")
    
    # Analyze
    stats = analyze_mapping(mappings)
    print(f"\nMapping statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Save mapping
    output_path = os.path.join(script_dir, '..', 'results', 'head_mapping.json')
    save_mapping(mappings, output_path)
    print(f"\nSaved mapping to {output_path}")

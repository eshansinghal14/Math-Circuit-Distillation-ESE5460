"""Swap k-means cluster neuron indices for i.i.d. random subsets of the flattened MLP space."""

from __future__ import annotations

import json
import os
from typing import List

import torch

from neuron_distillation.distillation import ClusterPairInfo
from utils import mlp_flatten_dim_from_pretrained_id as mlp_flatten_dim


def replace_pairs_with_random_neurons(
    pairs: List[ClusterPairInfo],
    *,
    D_student: int,
    D_teacher: int,
    fraction: float,
    seed: int,
    keep_importance_weights: bool = True,
) -> List[ClusterPairInfo]:
    """For each pair, ignore stored indices; sample ``fraction * D`` neurons (rounded, ≥1).

    Pairing topology (subclass, student/teacher cluster ids) and optional importance
    weights are unchanged so the rest of distillation matches the neuron pipeline.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1].")
    n_s = max(1, min(D_student, int(round(fraction * D_student))))
    n_t = max(1, min(D_teacher, int(round(fraction * D_teacher))))

    out: List[ClusterPairInfo] = []
    for i, p in enumerate(pairs):
        # Different RNG per pair so subsets are not identical across pairs.
        g_s = torch.Generator().manual_seed(seed + i * 10_007 + p.subclass * 131 + p.student_cluster_idx)
        g_t = torch.Generator().manual_seed(seed + i * 10_007 + p.subclass * 131 + p.teacher_cluster_idx + 17_389)
        perm_s = torch.randperm(D_student, generator=g_s)[:n_s].long()
        perm_t = torch.randperm(D_teacher, generator=g_t)[:n_t].long()
        imp = float(p.importance) if keep_importance_weights else 1.0
        out.append(
            ClusterPairInfo(
                subclass=p.subclass,
                student_cluster_idx=p.student_cluster_idx,
                teacher_cluster_idx=p.teacher_cluster_idx,
                student_neuron_indices=perm_s,
                teacher_neuron_indices=perm_t,
                importance=imp,
            )
        )
    return out


def save_manifest(
    path: str,
    *,
    fraction: float,
    seed: int,
    D_student: int,
    D_teacher: int,
    n_student_sampled: int,
    n_teacher_sampled: int,
    student_model: str,
    teacher_model: str,
    n_pairs: int,
) -> None:
    payload = {
        "experiment": "random_neuron_align",
        "fraction": fraction,
        "seed": seed,
        "D_student": D_student,
        "D_teacher": D_teacher,
        "n_student_sampled": n_student_sampled,
        "n_teacher_sampled": n_teacher_sampled,
        "student_model": student_model,
        "teacher_model": teacher_model,
        "n_pairs": n_pairs,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

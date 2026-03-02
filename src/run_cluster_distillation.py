"""CLI entry-point for the full neuron-cluster distillation pipeline.

Steps executed:
  1. Load ablation results for student and teacher.
  2. Create cluster pairings (via cluster_pairing).
  3. Load neuron indices from k-means cluster files.
  4. Run distillation training (ClusterDistillationTrainer).

Usage:
  python run_cluster_distillation.py \
      --student-model "meta-llama/Llama-3.2-1B" \
      --teacher-model "meta-llama/Meta-Llama-3-8B" \
      --student-ablation  ../results/circuit-discovery/meta-llama/Llama-3.2-1B/ablation_performance.json \
      --teacher-ablation  ../results/circuit-discovery/meta-llama/Meta-Llama-3-8B/ablation_performance.json \
      --student-clusters  ../results/neuron-clustering/meta-llama/Llama-3.2-1B/clusters \
      --teacher-clusters  ../results/neuron-clustering/meta-llama/Meta-Llama-3-8B/clusters \
      --dataset           ../datasets/2d_add_all.json \
      --k 7 --epochs 50 --batch-size 8 --lr 1e-4 --lambda-cluster 0.01
"""

import argparse
import json
import os
import random
import sys

import torch

from cluster_pairing import (
    _load_single_ablation_performance,
    create_cluster_mapping,
    analyze_mapping,
    save_mapping,
)
from cluster_distillation import (
    ClusterDistillationConfig,
    ClusterDistillationTrainer,
    ClusterPairInfo,
    eval_accuracy,
)


def build_train_test_split(dataset_path: str, test_frac: float = 0.1, seed: int = 42):
    """Load 2d_add_all.json and split into {prompt: answer} dicts."""
    with open(dataset_path, "r") as f:
        raw = json.load(f)

    pairs = [(r["q_str"], int(r["a_str"])) for r in raw]
    random.seed(seed)
    random.shuffle(pairs)

    split = int(len(pairs) * (1 - test_frac))
    train = dict(pairs[:split])
    test = dict(pairs[split:])
    return train, test


def build_cluster_pairs(
    student_ablation_path: str,
    teacher_ablation_path: str,
    student_clusters_dir: str,
    teacher_clusters_dir: str,
    k: int,
    k_classes: int = 8,
    top_k_per_subclass: int = 5,
):
    """Load ablation results, pair clusters, and attach neuron indices."""
    delta_s = _load_single_ablation_performance(student_ablation_path)
    delta_t = _load_single_ablation_performance(teacher_ablation_path)

    mappings = create_cluster_mapping(
        delta_s, delta_t, top_k_student=top_k_per_subclass,
    )

    stats = analyze_mapping(mappings)
    print("\nCluster mapping statistics:")
    for key, val in stats.items():
        print(f"  {key}: {val}")

    pairs = []
    for m in mappings:
        sc = m.subclass
        s_path = os.path.join(
            student_clusters_dir, f"subclass_{sc}_clusters/k{k}.pt"
        )
        t_path = os.path.join(
            teacher_clusters_dir, f"subclass_{sc}_clusters/k{k}.pt"
        )

        if not os.path.exists(s_path):
            print(f"  [skip] student cluster file missing: {s_path}")
            continue
        if not os.path.exists(t_path):
            print(f"  [skip] teacher cluster file missing: {t_path}")
            continue

        s_ckpt = torch.load(s_path, map_location="cpu")
        t_ckpt = torch.load(t_path, map_location="cpu")

        s_c2i = s_ckpt["cluster_to_indices"]
        t_c2i = t_ckpt["cluster_to_indices"]

        s_key = m.student_cluster_idx
        t_key = m.teacher_cluster_idx

        if s_key not in s_c2i or t_key not in t_c2i:
            print(f"  [skip] subclass {sc}: student cluster {s_key} or "
                  f"teacher cluster {t_key} not in file")
            continue

        s_idx = s_c2i[s_key]
        t_idx = t_c2i[t_key]
        if not isinstance(s_idx, torch.Tensor):
            s_idx = torch.tensor(s_idx, dtype=torch.long)
        if not isinstance(t_idx, torch.Tensor):
            t_idx = torch.tensor(t_idx, dtype=torch.long)

        if s_idx.numel() == 0 or t_idx.numel() == 0:
            continue

        pairs.append(ClusterPairInfo(
            subclass=sc,
            student_cluster_idx=m.student_cluster_idx,
            teacher_cluster_idx=m.teacher_cluster_idx,
            student_neuron_indices=s_idx,
            teacher_neuron_indices=t_idx,
            importance=m.student_importance,
        ))

    pairs.sort(key=lambda p: p.importance, reverse=True)
    print(f"\nBuilt {len(pairs)} cluster pairs across "
          f"{len(set(p.subclass for p in pairs))} subclasses")
    return pairs, mappings


def main():
    parser = argparse.ArgumentParser(
        description="Neuron-cluster circuit distillation"
    )
    parser.add_argument("--student-model", type=str,
                        default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--teacher-model", type=str,
                        default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--student-ablation", type=str, required=True,
                        help="Path to student ablation_performance.json")
    parser.add_argument("--teacher-ablation", type=str, required=True,
                        help="Path to teacher ablation_performance.json")
    parser.add_argument("--student-clusters", type=str, required=True,
                        help="Dir with student subclass_N_clusters/kK.pt files")
    parser.add_argument("--teacher-clusters", type=str, required=True,
                        help="Dir with teacher subclass_N_clusters/kK.pt files")
    parser.add_argument("--dataset", type=str,
                        default="../datasets/2d_add_all.json")
    parser.add_argument("--k", type=int, default=7,
                        help="Number of clusters per subclass")
    parser.add_argument("--k-classes", type=int, default=8,
                        help="Number of latent subclasses")
    parser.add_argument("--top-k-pairs", type=int, default=5,
                        help="Keep top-k student clusters per subclass for pairing")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-cluster", type=float, default=0.01)
    parser.add_argument("--lambda-proj", type=float, default=0.0)
    parser.add_argument("--use-projection", action="store_true")
    parser.add_argument("--save-dir", type=str,
                        default="../results/cluster-distillation")
    args = parser.parse_args()

    # ---- 1. Cluster pairing ------------------------------------------------
    print("=" * 60)
    print("Step 1: Cluster pairing")
    print("=" * 60)

    cluster_pairs, mappings = build_cluster_pairs(
        student_ablation_path=args.student_ablation,
        teacher_ablation_path=args.teacher_ablation,
        student_clusters_dir=args.student_clusters,
        teacher_clusters_dir=args.teacher_clusters,
        k=args.k,
        k_classes=args.k_classes,
        top_k_per_subclass=args.top_k_pairs,
    )

    if not cluster_pairs:
        print("No cluster pairs found. Check ablation results and cluster files.")
        sys.exit(1)

    os.makedirs(args.save_dir, exist_ok=True)
    save_mapping(mappings, os.path.join(args.save_dir, "cluster_mapping.json"))

    # ---- 2. Dataset ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Loading dataset")
    print("=" * 60)

    train_data, test_data = build_train_test_split(args.dataset)
    print(f"  Train: {len(train_data)} examples")
    print(f"  Test:  {len(test_data)} examples")

    # ---- 3. Distillation training -------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Distillation training")
    print("=" * 60)

    config = ClusterDistillationConfig(
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lambda_cluster=args.lambda_cluster,
        lambda_proj=args.lambda_proj,
        use_projection_heads=args.use_projection,
        top_k_clusters_per_subclass=args.top_k_pairs,
        save_dir=args.save_dir,
    )

    trainer = ClusterDistillationTrainer(
        config=config,
        cluster_pairs=cluster_pairs,
        train_data=train_data,
        test_data=test_data,
    )

    history = trainer.train()

    # ---- 4. Final summary ---------------------------------------------------
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    if "accuracy" in history and history["accuracy"]:
        print(f"  Best accuracy: {max(history['accuracy']):.4f}")
    print(f"  Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()

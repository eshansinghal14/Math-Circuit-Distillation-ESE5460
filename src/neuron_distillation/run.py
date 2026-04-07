"""Single entry-point for the neuron-cluster distillation pipeline.

Circuit discovery weights: place ``epoch_*.pt`` (or any ``*.pt``) in the **root**
of ``--save-dir``; auto-detection only looks there (not under subfolders).

Cluster files are expected under::

    <save-dir>/neuron-clustering/<model-id>/clusters/

Ablation outputs (``ablation_performance.json`` and related) are written under::

    <save-dir>/ablation/<model-id>/...

Distillation checkpoints and training history use ``--save-dir`` directly.

Training/eval JSON paths come from ``--dataset PREFIX`` (e.g. ``2d_add`` →
``datasets/2d_add_train_80.json`` and ``2d_add_test_20.json``). If ``--dataset``
is omitted, you are prompted (interactive TTY only).

Usage (from src/)::

  python -m neuron_distillation.run \\

      --dataset 2d_add \\

      --save-dir "/path/to/results/my-neuron-run"

  # Optional: explicit checkpoint and k_classes

  python -m neuron_distillation.run \\

      --save-dir "/path/to/results/my-neuron-run" \\

      --checkpoint /path/to/epoch_4000.pt \\

      --k-classes 2

"""

import argparse
import json
import os
import re
import sys
from typing import Dict

import torch

from neuron_distillation.ablation import classify_problems, ablation
from neuron_distillation.pairing import (
    _load_single_ablation_performance,
    create_cluster_mapping,
    analyze_mapping,
    save_mapping,
)
from neuron_distillation.distillation import (
    ClusterDistillationConfig,
    ClusterDistillationTrainer,
    ClusterPairInfo,
)
from utils import (
    EVAL_MAX_NEW_TOKENS,
    _extract_circuit_model_state_dict,
    load_model,
    load_model_checkpoint,
    resolve_train_test_paths,
)


def _load_prompt_answer_dict(path: str) -> Dict[str, int]:
    """Load ``{prompt: answer}`` JSON (same format as ``standard_distillation``)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): int(v) for k, v in data.items()}


def _ablation_dir_for_model(save_dir: str, model_name: str) -> str:
    """``save_dir/ablation/<huggingface-id-path>/`` — one folder per model."""
    return os.path.join(save_dir, "ablation", *model_name.split("/"))


def run_ablation_if_needed(
    model_name: str,
    checkpoint_path: str,
    clusters_dir: str,
    k: int,
    k_classes: int,
    results_dir: str,
    ablation_batch_size: int = 50,
):
    """Run ablation for a model, or skip if results already exist under ``results_dir``."""
    os.makedirs(results_dir, exist_ok=True)
    ablation_path = os.path.join(results_dir, "ablation_performance.json")
    if os.path.exists(ablation_path):
        print(f"  Found existing ablation results: {ablation_path}")
        return ablation_path
    print(f"  Running ablation for {model_name}...")
    _, tokenizer = load_model(model_name)
    circuit_model, _, _, _ = load_model_checkpoint(
        checkpoint_path, k_classes=k_classes, lr=1e-3,
    )
    circuit_model.eval()
    class_to_problems = classify_problems(circuit_model, tokenizer)
    del circuit_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    class_clusters = [k] * k_classes
    ablation(
        model_name,
        tokenizer,
        class_to_problems,
        class_clusters=class_clusters,
        results_base_dir=results_dir,
        clusters_base_dir=clusters_dir,
        batch_size=ablation_batch_size,
    )
    return ablation_path


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
    print("\n  Cluster mapping statistics:")
    for key, val in stats.items():
        print(f"    {key}: {val}")
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
            print(f"    [skip] student cluster file missing: {s_path}")
            continue
        if not os.path.exists(t_path):
            print(f"    [skip] teacher cluster file missing: {t_path}")
            continue
        s_ckpt = torch.load(s_path, map_location="cpu")
        t_ckpt = torch.load(t_path, map_location="cpu")
        s_c2i = s_ckpt["cluster_to_indices"]
        t_c2i = t_ckpt["cluster_to_indices"]
        s_key = m.student_cluster_idx
        t_key = m.teacher_cluster_idx
        if s_key not in s_c2i or t_key not in t_c2i:
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
    print(f"\n  Built {len(pairs)} cluster pairs across "
          f"{len(set(p.subclass for p in pairs))} subclasses")
    return pairs, mappings


def _find_circuit_checkpoint_pt(save_dir: str) -> list[str]:
    """``*.pt`` files in ``save_dir`` root only; ``epoch_N.pt`` sorted by highest N first."""
    if not save_dir or not os.path.isdir(save_dir):
        return []
    candidates = []
    for name in os.listdir(save_dir):
        path = os.path.join(save_dir, name)
        if name.endswith(".pt") and os.path.isfile(path):
            candidates.append(path)

    def sort_key(p: str) -> tuple:
        m = re.search(r"epoch_(\d+)\.pt$", os.path.basename(p), re.IGNORECASE)
        if m:
            return (0, -int(m.group(1)))
        return (1, p)

    candidates.sort(key=sort_key)
    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="Neuron-cluster circuit distillation (end-to-end)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to circuit-discovery model checkpoint (.pt). "
             "If omitted, uses a single *.pt in the root of --save-dir (prefers highest epoch_*.pt).",
    )
    parser.add_argument("--student-model", type=str,
                        default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--teacher-model", type=str,
                        default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        metavar="PREFIX",
        help="Dataset family, e.g. 2d_add → <datasets>/<PREFIX>_train_80.json and _test_20.json",
    )
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default=None,
        help="Directory containing *_train_80.json (default: repo datasets/)",
    )
    parser.add_argument("--k", type=int, default=7,
                        help="Number of clusters per subclass")
    parser.add_argument("--k-classes", type=int, default=None,
                        help="Number of latent subclasses (auto-detected from checkpoint if omitted)")
    parser.add_argument("--top-k-pairs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--step-log-interval",
        type=int,
        default=50,
        help="Print in-epoch step loss every N batches (default 50, same as standard distillation)",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=None,
        help="Greedy eval during training (default: utils.EVAL_MAX_NEW_TOKENS)",
    )
    parser.add_argument("--lambda-cluster", type=float, default=0.01)
    parser.add_argument("--lambda-proj", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save student_epoch_N every N epochs; only most recent kept (0=off)")
    parser.add_argument("--save-best", action="store_true",
                        help="Save student_best whenever eval accuracy improves (off by default)")
    parser.add_argument("--use-projection", action="store_true")
    parser.add_argument(
        "--save-dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "..", "results", "cluster-distillation"),
        help="All outputs: neuron-clustering/, ablation/, distillation checkpoints, history, cluster_mapping.json",
    )
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip ablation (expects ablation_performance.json under save-dir/ablation/...)")
    parser.add_argument(
        "--ablation-batch-size",
        type=int,
        default=50,
        help="Batch size for Step 1 neuron-cluster ablation (HF forward passes)",
    )
    args = parser.parse_args()
    try:
        train_path, test_path, dataset_prefix = resolve_train_test_paths(
            dataset=args.dataset,
            datasets_dir=args.datasets_dir,
        )
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e
    args.train_path = train_path
    args.test_path = test_path
    args.dataset_prefix = dataset_prefix
    eval_max_new_tokens = (
        args.eval_max_new_tokens
        if args.eval_max_new_tokens is not None
        else EVAL_MAX_NEW_TOKENS
    )
    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    neuron_clustering_root = os.path.join(save_dir, "neuron-clustering")
    student_clusters = os.path.join(neuron_clustering_root, args.student_model, "clusters")
    teacher_clusters = os.path.join(neuron_clustering_root, args.teacher_model, "clusters")
    student_ablation_dir = _ablation_dir_for_model(save_dir, args.student_model)
    teacher_ablation_dir = _ablation_dir_for_model(save_dir, args.teacher_model)
    # ---- Auto-detect checkpoint .pt (save-dir root only) ----------------------------
    if args.checkpoint is None:
        candidates = _find_circuit_checkpoint_pt(save_dir)
        if candidates:
            args.checkpoint = candidates[0]
            print(f"Auto-detected checkpoint: {args.checkpoint}")
        else:
            print(
                "ERROR: No --checkpoint provided and no *.pt file in --save-dir root:\n"
                f"  {save_dir}"
            )
            sys.exit(1)
    # ---- Auto-detect k_classes from checkpoint --------------------------------------
    if args.k_classes is None:
        ckpt_data = torch.load(args.checkpoint, map_location="cpu")
        state = _extract_circuit_model_state_dict(ckpt_data, args.checkpoint)
        if "classifier.classifier.4.weight" in state:
            args.k_classes = state["classifier.classifier.4.weight"].shape[0]
            print(f"Auto-detected k_classes={args.k_classes} from checkpoint")
        else:
            print("ERROR: Could not detect k_classes from checkpoint. Pass --k-classes explicitly.")
            sys.exit(1)
        del ckpt_data
    # ---- Diagnostics --------------------------------------------------------------
    print("=" * 60)
    print("Configuration")
    print("=" * 60)
    print(f"  k_classes:          {args.k_classes}")
    print(f"  k (clusters):       {args.k}")
    print(f"  save_dir:           {save_dir}")
    print(f"  neuron-clustering:  {neuron_clustering_root}")
    print(f"  checkpoint:         {args.checkpoint}")
    print(f"  student_clusters:   {student_clusters}")
    print(f"  teacher_clusters:   {teacher_clusters}")
    print(f"  student_ablation:   {student_ablation_dir}")
    print(f"  teacher_ablation:   {teacher_ablation_dir}")
    print(f"  ablation_batch:     {args.ablation_batch_size}")
    print(f"  dataset (prefix):   {args.dataset_prefix}")
    print(f"  train_path:         {args.train_path}")
    print(f"  test_path:          {args.test_path}")
    # ---- Step 1: Ablation -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 1: Neuron-cluster ablation")
    print("=" * 60)
    if args.skip_ablation:
        student_abl = os.path.join(student_ablation_dir, "ablation_performance.json")
        teacher_abl = os.path.join(teacher_ablation_dir, "ablation_performance.json")
        print(f"  Skipping ablation. Expecting:\n    {student_abl}\n    {teacher_abl}")
        if not os.path.isfile(student_abl) or not os.path.isfile(teacher_abl):
            print("ERROR: --skip-ablation but ablation JSON missing. Run without --skip-ablation first.")
            sys.exit(1)
    else:
        print(f"\n  Student: {args.student_model}")
        student_abl = run_ablation_if_needed(
            args.student_model, args.checkpoint, student_clusters,
            args.k, args.k_classes, student_ablation_dir,
            ablation_batch_size=args.ablation_batch_size,
        )
        print(f"\n  Teacher: {args.teacher_model}")
        teacher_abl = run_ablation_if_needed(
            args.teacher_model, args.checkpoint, teacher_clusters,
            args.k, args.k_classes, teacher_ablation_dir,
            ablation_batch_size=args.ablation_batch_size,
        )
    # ---- Step 2: Cluster pairing ----------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Cluster pairing")
    print("=" * 60)
    cluster_pairs, mappings = build_cluster_pairs(
        student_ablation_path=student_abl,
        teacher_ablation_path=teacher_abl,
        student_clusters_dir=student_clusters,
        teacher_clusters_dir=teacher_clusters,
        k=args.k,
        k_classes=args.k_classes,
        top_k_per_subclass=args.top_k_pairs,
    )
    if not cluster_pairs:
        print("No cluster pairs found. Check ablation results and cluster files.")
        sys.exit(1)
    save_mapping(mappings, os.path.join(save_dir, "cluster_mapping.json"))
    # ---- Step 3: Dataset ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Loading dataset")
    print("=" * 60)
    train_data = _load_prompt_answer_dict(args.train_path)
    test_data = _load_prompt_answer_dict(args.test_path)
    print(f"  Train: {len(train_data)} examples ({args.train_path})")
    print(f"  Test:  {len(test_data)} examples ({args.test_path})")
    # ---- Step 4: Distillation training -----------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Distillation training")
    print("=" * 60)
    config = ClusterDistillationConfig(
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        step_log_interval=args.step_log_interval,
        learning_rate=args.lr,
        temperature=args.temperature,
        lambda_cluster=args.lambda_cluster,
        lambda_proj=args.lambda_proj,
        use_projection_heads=args.use_projection,
        top_k_clusters_per_subclass=args.top_k_pairs,
        save_every=args.save_every,
        save_best=args.save_best,
        eval_max_new_tokens=eval_max_new_tokens,
        save_dir=save_dir,
    )
    trainer = ClusterDistillationTrainer(
        config=config,
        cluster_pairs=cluster_pairs,
        train_data=train_data,
        test_data=test_data,
    )
    history = trainer.train()
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    if "accuracy" in history and history["accuracy"]:
        print(f"  Best accuracy: {max(history['accuracy']):.4f}")
    print(f"  Results saved to: {save_dir}")


if __name__ == "__main__":
    main()

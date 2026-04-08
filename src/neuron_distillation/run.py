"""Single entry-point for KL distillation under ``<save-dir>/<run-name|timestamp>/``.

**Modes**

- ``--mode circuit`` (default): neuron-cluster ablation, pairing, KL + cluster CKA (same as before).
- ``--mode standard``: pure KL distillation (no circuit checkpoint, ablation, or CKA).

**Resume**: ``--resume`` loads weights from
``<run>/student_model/``. Use ``--checkpoint-run <datetime>`` or omit to use the most
recent run folder directly under ``--save-dir``.

Outputs per run: ``student_model/``, ``training_history.json``, ``training_state.pt``,
``training_curves.png``. Circuit runs also use ``ablation/``, ``cluster_mapping.json``,
and global ``neuron-clustering/``.

Usage (from ``src/``)::

  python -m neuron_distillation.run --dataset 2d_add --save-dir /path/to/results --k 7

  python -m neuron_distillation.run --mode standard --dataset 2d_add --save-dir /path/to/results

  python -m neuron_distillation.run --resume --dataset 2d_add --save-dir /path/to/results --epochs 20
"""

import argparse
import json
import os
import re
import sys
from typing import Optional

import torch

from neuron_distillation.ablation import classify_problems, ablation
from neuron_distillation.distillation import (
    ClusterDistillationConfig,
    ClusterDistillationTrainer,
    ClusterPairInfo,
)
from neuron_distillation.pairing import (
    _load_single_ablation_performance,
    create_cluster_mapping,
    analyze_mapping,
    save_mapping,
)
from utils import (
    EVAL_MAX_NEW_TOKENS,
    _extract_circuit_model_state_dict,
    load_model,
    load_model_checkpoint,
    load_prompt_answer_json,
    resolve_distillation_run_dir,
    resolve_train_test_paths,
)


def _ablation_dir_for_model(run_dir: str, model_name: str) -> str:
    return os.path.join(run_dir, "ablation", *model_name.split("/"))


def run_ablation_if_needed(
    model_name: str,
    checkpoint_path: str,
    clusters_dir: str,
    k: int,
    k_classes: int,
    results_dir: str,
    ablation_batch_size: int = 50,
):
    """Run ablation for a model, or skip if results already cached."""
    os.makedirs(results_dir, exist_ok=True)
    ablation_path = os.path.join(results_dir, "ablation_performance.json")
    if os.path.exists(ablation_path):
        print(f"  Found cached ablation results: {ablation_path}")
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
        model_name, tokenizer, class_to_problems,
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
    mapping_cache_path: Optional[str] = None,
):
    """Load ablation results, pair clusters, attach neuron indices.

    Uses all student clusters per subclass (no top-k truncation) for CKA.
    If ``mapping_cache_path`` exists, load pairs from that JSON instead.
    """
    if mapping_cache_path and os.path.isfile(mapping_cache_path):
        print(f"  Found cached cluster mapping: {mapping_cache_path}")
        with open(mapping_cache_path, "r") as f:
            raw = json.load(f)
        pairs = []
        for item in raw:
            sc = item["subclass"]
            s_path = os.path.join(student_clusters_dir, f"subclass_{sc}_clusters/k{k}.pt")
            t_path = os.path.join(teacher_clusters_dir, f"subclass_{sc}_clusters/k{k}.pt")
            if not os.path.exists(s_path) or not os.path.exists(t_path):
                continue
            s_ckpt = torch.load(s_path, map_location="cpu")
            t_ckpt = torch.load(t_path, map_location="cpu")
            s_c2i = s_ckpt["cluster_to_indices"]
            t_c2i = t_ckpt["cluster_to_indices"]
            s_key = item["student_cluster_idx"]
            t_key = item["teacher_cluster_idx"]
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
                student_cluster_idx=s_key,
                teacher_cluster_idx=t_key,
                student_neuron_indices=s_idx,
                teacher_neuron_indices=t_idx,
                importance=item.get("student_importance", 0.0),
            ))
        pairs.sort(key=lambda p: p.importance, reverse=True)
        print(f"  Loaded {len(pairs)} cluster pairs from cache")
        return pairs

    delta_s = _load_single_ablation_performance(student_ablation_path)
    delta_t = _load_single_ablation_performance(teacher_ablation_path)
    mappings = create_cluster_mapping(delta_s, delta_t, top_k_student=None)
    stats = analyze_mapping(mappings)
    print("\n  Cluster mapping statistics:")
    for key, val in stats.items():
        print(f"    {key}: {val}")
    pairs = []
    for m in mappings:
        sc = m.subclass
        s_path = os.path.join(student_clusters_dir, f"subclass_{sc}_clusters/k{k}.pt")
        t_path = os.path.join(teacher_clusters_dir, f"subclass_{sc}_clusters/k{k}.pt")
        if not os.path.exists(s_path) or not os.path.exists(t_path):
            print(f"    [skip] cluster file missing for subclass {sc}")
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
    if mapping_cache_path:
        save_mapping(mappings, mapping_cache_path)
    return pairs


def _find_circuit_checkpoint_pt(save_dir: str) -> list[str]:
    """*.pt files in save_dir root; epoch_N.pt sorted by highest N first."""
    if not save_dir or not os.path.isdir(save_dir):
        return []
    candidates = []
    for name in os.listdir(save_dir):
        path = os.path.join(save_dir, name)
        if name.endswith(".pt") and os.path.isfile(path):
            candidates.append(path)

    def sort_key(p: str) -> tuple:
        m = re.search(r"epoch_(\d+)\.pt$", os.path.basename(p), re.IGNORECASE)
        return (0, -int(m.group(1))) if m else (1, p)

    candidates.sort(key=sort_key)
    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="Neuron-distillation pipeline (circuit or standard KL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["circuit", "standard"],
        default="circuit",
        help="circuit: ablation + CKA + KL; standard: KL only (same run layout under --save-dir)",
    )
    parser.add_argument("--student-model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--teacher-model", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument(
        "--dataset", type=str, default=None, metavar="PREFIX",
        help="Dataset family prefix, e.g. 2d_add → datasets/2d_add_train_80.json + _test_20.json",
    )
    parser.add_argument("--datasets-dir", type=str, default=None,
                        help="Directory containing *_train_80.json (default: repo datasets/)")
    parser.add_argument(
        "--k", type=int, default=None, metavar="INT",
        help="Clusters per subclass (circuit mode only; required for new circuit runs).",
    )
    parser.add_argument("--k-classes", type=int, default=None,
                        help="Number of latent subclasses (circuit; auto-detected from checkpoint if omitted)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to circuit-discovery checkpoint .pt (circuit mode; auto-detected from --save-dir root if omitted)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-cluster", type=float, default=0.01)
    parser.add_argument("--lambda-proj", type=float, default=0.0)
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Overwrite student_model/ every N epochs (0=disable periodic save)",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="Overwrite student_model/ when eval accuracy improves (off by default)",
    )
    parser.add_argument("--use-projection", action="store_true")
    parser.add_argument(
        "--step-log-interval",
        type=int,
        default=50,
        help="Print in-epoch metrics every N batches",
    )
    parser.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=None,
        help=f"Greedy eval during training (default: {EVAL_MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--save-dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "..", "results", "cluster-distillation"),
        help="Base directory; each run is <save-dir>/<run-name|timestamp>/",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        metavar="NAME",
        help="Folder name for a new run (default: timestamp).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from <run>/student_model/; use --checkpoint-run or latest run under --save-dir",
    )
    parser.add_argument(
        "--checkpoint-run",
        default=None,
        metavar="DATETIME",
        help="Run folder name or path under --save-dir (e.g. datetime). For legacy layouts, pass neuron-cluster/<run> if needed.",
    )
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Circuit: skip ablation (expects cached ablation JSON under run dir)")
    parser.add_argument("--ablation-batch-size", type=int, default=50,
                        help="Circuit: batch size for neuron-cluster ablation")
    args = parser.parse_args()

    try:
        train_path, test_path, dataset_prefix = resolve_train_test_paths(
            dataset=args.dataset, datasets_dir=args.datasets_dir,
        )
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e

    eval_max_new_tokens = (
        args.eval_max_new_tokens
        if args.eval_max_new_tokens is not None
        else EVAL_MAX_NEW_TOKENS
    )
    args.save_dir = os.path.abspath(args.save_dir)

    run_dir, student_source = resolve_distillation_run_dir(
        args.save_dir,
        resume=args.resume,
        run_name=args.run_name,
        checkpoint_run=args.checkpoint_run,
    )
    os.makedirs(run_dir, exist_ok=True)
    is_resume = student_source is not None

    circuit = args.mode == "circuit"
    neuron_clustering_root = os.path.join(args.save_dir, "neuron-clustering")
    student_clusters = os.path.join(neuron_clustering_root, args.student_model, "clusters")
    teacher_clusters = os.path.join(neuron_clustering_root, args.teacher_model, "clusters")
    student_ablation_dir = _ablation_dir_for_model(run_dir, args.student_model)
    teacher_ablation_dir = _ablation_dir_for_model(run_dir, args.teacher_model)
    mapping_cache = os.path.join(run_dir, "cluster_mapping.json")

    print(f"Run dir: {run_dir}")
    print(f"  mode: {args.mode}")

    cluster_pairs: list = []

    if circuit:
        if is_resume and args.k_classes is None:
            pts = _find_circuit_checkpoint_pt(args.save_dir)
            if pts:
                ckpt_data = torch.load(pts[0], map_location="cpu")
                state = _extract_circuit_model_state_dict(ckpt_data, pts[0])
                if "classifier.classifier.4.weight" in state:
                    args.k_classes = state["classifier.classifier.4.weight"].shape[0]
                    print(f"Auto-detected k_classes={args.k_classes} from checkpoint")
                del ckpt_data
        if not is_resume or args.checkpoint:
            if args.checkpoint is None:
                candidates = _find_circuit_checkpoint_pt(args.save_dir)
                if candidates:
                    args.checkpoint = candidates[0]
                    print(f"Auto-detected circuit checkpoint: {args.checkpoint}")
                else:
                    print(
                        "ERROR: No --checkpoint and no *.pt in --save-dir root:\n"
                        f"  {args.save_dir}"
                    )
                    sys.exit(1)
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

        if args.k_classes is None:
            raise SystemExit(
                "ERROR: Could not determine k_classes for circuit mode. "
                "Pass --k-classes or place a circuit-discovery .pt under --save-dir.",
            )

        if args.k is None and not is_resume:
            sp = os.path.join(neuron_clustering_root, args.student_model, "plots")
            tp = os.path.join(neuron_clustering_root, args.teacher_model, "plots")
            raise SystemExit(
                "ERROR: --k INT is required for circuit mode (clusters per subclass).\n"
                f"  {sp}\n  {tp}"
            )

        if is_resume and args.k is None and os.path.isfile(mapping_cache):
            with open(mapping_cache) as f:
                raw = json.load(f)
            if raw:
                args.k = raw[0].get("k", None) or 7
                print(f"Inferred k={args.k} from cached mapping")

        if is_resume and args.k is None:
            raise SystemExit("ERROR: --k is required for circuit resume (could not infer from cache).")

        print("=" * 60)
        print("Configuration (circuit)")
        print("=" * 60)
        print(f"  k_classes:          {args.k_classes}")
        print(f"  k (clusters):       {args.k}")
        print(f"  student_clusters:   {student_clusters}")
        print(f"  teacher_clusters:   {teacher_clusters}")
        print(f"  dataset (prefix):   {dataset_prefix}")
        print(f"  train_path:         {train_path}")
        print(f"  test_path:          {test_path}")
        print("=" * 60)

        print("\n" + "=" * 60)
        print("Step 1: Neuron-cluster ablation")
        print("=" * 60)
        if args.skip_ablation:
            student_abl = os.path.join(student_ablation_dir, "ablation_performance.json")
            teacher_abl = os.path.join(teacher_ablation_dir, "ablation_performance.json")
            print(f"  --skip-ablation: expecting\n    {student_abl}\n    {teacher_abl}")
            if not os.path.isfile(student_abl) or not os.path.isfile(teacher_abl):
                print("ERROR: --skip-ablation but ablation JSON missing.")
                sys.exit(1)
        elif args.checkpoint:
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
        else:
            raise SystemExit("No circuit checkpoint provided. Cannot run ablation.")

        print("\n" + "=" * 60)
        print("Step 2: Cluster pairing")
        print("=" * 60)
        cluster_pairs = build_cluster_pairs(
            student_ablation_path=student_abl,
            teacher_ablation_path=teacher_abl,
            student_clusters_dir=student_clusters,
            teacher_clusters_dir=teacher_clusters,
            k=args.k,
            k_classes=args.k_classes,
            mapping_cache_path=mapping_cache,
        )
        if not cluster_pairs:
            print("No cluster pairs found. Check ablation results and cluster files.")
            sys.exit(1)
    else:
        print("=" * 60)
        print("Configuration (standard KL)")
        print("=" * 60)
        print(f"  dataset (prefix):   {dataset_prefix}")
        print(f"  train_path:         {train_path}")
        print(f"  test_path:          {test_path}")
        print("=" * 60)

    print("\n" + "=" * 60)
    print("Loading dataset")
    print("=" * 60)
    train_data = load_prompt_answer_json(train_path)
    test_data = load_prompt_answer_json(test_path)
    print(f"  Train: {len(train_data)} examples")
    print(f"  Test:  {len(test_data)} examples")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 60)
    print("Loading student")
    print("=" * 60)
    if student_source and student_source.endswith(".pt"):
        print(f"  Loading base model + fast weights from: {student_source!r}")
        student, tokenizer = load_model(args.student_model)
        state_dict = torch.load(student_source, map_location="cpu", weights_only=True)
        student.load_state_dict(state_dict)
        del state_dict
    elif student_source:
        print(f"  From checkpoint dir: {student_source!r}")
        student, tokenizer = load_model(student_source)
    else:
        print(f"  From Hugging Face: {args.student_model!r}")
        student, tokenizer = load_model(args.student_model)
    student = student.to("cpu").float().to(device)
    tokenizer.padding_side = "left"

    print("\n" + "=" * 60)
    print("Distillation training")
    print("=" * 60)
    config = ClusterDistillationConfig(
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        distillation_mode="standard" if not circuit else "circuit",
        epochs=args.epochs,
        batch_size=args.batch_size,
        step_log_interval=args.step_log_interval,
        learning_rate=args.lr,
        temperature=args.temperature,
        grad_clip=args.grad_clip,
        lambda_cluster=args.lambda_cluster,
        lambda_proj=args.lambda_proj,
        use_projection_heads=args.use_projection if circuit else False,
        save_every=args.save_every,
        save_best=args.save_best,
        eval_max_new_tokens=eval_max_new_tokens,
        save_dir=run_dir,
    )
    trainer = ClusterDistillationTrainer(
        config=config,
        cluster_pairs=cluster_pairs,
        train_data=train_data,
        test_data=test_data,
        tokenizer=tokenizer,
        student=student,
        resume=is_resume,
    )
    history = trainer.train()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    if "accuracy" in history and history["accuracy"]:
        print(f"  Best accuracy: {max(history['accuracy']):.4f}")
    print(f"  Results saved to: {run_dir}")


if __name__ == "__main__":
    main()

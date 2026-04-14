"""Single entry-point for KL distillation; all outputs go **directly** under ``--save-dir``.

**Modes**

- ``--mode circuit`` (default): neuron-cluster ablation, pairing, KL + cluster CKA (same as before).
- ``--mode standard``: pure KL distillation (no circuit checkpoint, ablation, or CKA).

**Resume**: ``--resume`` loads weights from ``<save-dir>/student_model/`` (or
``student_weights.pt``). Use ``--checkpoint-run <path>`` only for a legacy nested run folder;
otherwise omit if checkpoints live in ``--save-dir``.

Outputs per run: ``student_model/``, ``training_history.json`` (``epoch_flops``: floats when
measured, ``null`` when skipped by ``--count-flops-every``), ``training_state.pt`` (written
when ``student_weights.pt`` is saved or at the final ``student_model/`` save),
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
    adjust_ablation_drops_for_poly_importance,
    analyze_mapping,
    create_cluster_mapping,
    default_random_ablation_poly_json_paths,
    save_mapping,
)
from utils import (
    EVAL_MAX_NEW_TOKENS,
    _extract_circuit_model_state_dict,
    load_model,
    load_model_checkpoint,
    load_prompt_answer_json,
    load_student_model_for_distillation,
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
    importance_vs_poly: bool = True,
    student_poly_json: Optional[str] = None,
    teacher_poly_json: Optional[str] = None,
    student_model_name: Optional[str] = None,
    teacher_model_name: Optional[str] = None,
):
    """Load ablation results, pair clusters, attach neuron indices.

    Uses all student clusters per subclass (no top-k truncation) for CKA.
    If ``mapping_cache_path`` exists, load pairs from that JSON instead.

    When ``importance_vs_poly`` is True (default), pairing uses signed residual importance
    ``actual_drop − poly(|C|/D)`` with the random-ablation poly JSONs
    (see :func:`pairing.adjust_ablation_drops_for_poly_importance`). Cached
    mappings skip this step (importance values come from the JSON).
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
    if importance_vs_poly:
        if not student_model_name or not teacher_model_name:
            raise ValueError(
                "Poly-based cluster importance requires student_model_name and "
                "teacher_model_name (HF ids). Pass them into build_cluster_pairs.",
            )
        sp_default, tp_default = default_random_ablation_poly_json_paths()
        sp = student_poly_json or sp_default
        tp = teacher_poly_json or tp_default
        class_clusters_s = [k] * k_classes
        class_clusters_t = [k] * k_classes
        print(
            "  Cluster importance: residual (actual ablation drop − poly expected at |C|/D)\n"
            f"    student poly: {sp}\n"
            f"    teacher poly: {tp}",
        )
        delta_s, delta_t = adjust_ablation_drops_for_poly_importance(
            delta_s,
            delta_t,
            student_clusters_dir,
            teacher_clusters_dir,
            class_clusters_s,
            class_clusters_t,
            sp,
            tp,
            student_model_name,
            teacher_model_name,
        )
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
    parser.add_argument(
        "--datasets-dir", type=str, default=None,
        help="Directory containing *_train_80.json (default: repo datasets/)",
    )
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
    parser.add_argument(
        "--cluster-size-weighting",
        action="store_true",
        help=(
            "CKA: multiply each cluster pair's loss weight by the fraction of student MLP neurons "
            "in that cluster vs full flattened width (combines with importance weighting when both on)"
        ),
    )
    parser.add_argument(
        "--no-poly-importance",
        action="store_true",
        help=(
            "Pairing: use raw ablation drops (baseline−accuracy); do not subtract poly expected "
            "drop at |C|/D (see datasets/random_ablation_poly/random_ablation_poly_*.json)"
        ),
    )
    parser.add_argument(
        "--student-poly-json",
        type=str,
        default=None,
        metavar="PATH",
        help="Override default random_ablation_poly_1b.json for residual cluster importance",
    )
    parser.add_argument(
        "--teacher-poly-json",
        type=str,
        default=None,
        metavar="PATH",
        help="Override default random_ablation_poly_8b.json for residual cluster importance",
    )
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
    parser.add_argument(
        "--step-log-interval",
        type=int,
        default=50,
        help="Print in-epoch metrics every N batches",
    )
    parser.add_argument(
        "--log-kl-cka-grad-norms",
        action="store_true",
        help=(
            "Log ‖g_KL‖ and ‖g_{λ·CKA}‖ (autograd.grad; ~2× work). In circuit mode, scale the "
            "λ·CKA contribution to the gradient by ‖g_KL‖/‖g_{λ·CKA}‖"
        ),
    )
    parser.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=None,
        help=f"Greedy eval during training (default: {EVAL_MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--eval-print-samples",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Each time eval runs (baselines + periodic student eval), print the first N "
            "test prompts and top-5 softmax next-token predictions for student and teacher. "
            "0=off."
        ),
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=50,
        metavar="N",
        help="Batch size for greedy test accuracy during training (default: 50)",
    )
    parser.add_argument(
        "--count-flops-every",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Count training FLOPs every N epochs (0-based: epochs 0, N, 2N, …). "
            "1=all epochs; 0=never. Default 0."
        ),
    )
    parser.add_argument(
        "--save-dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "..", "results", "cluster-distillation"),
        help="Directory for ablation/, student_model/, training_history.json, training_state.pt, etc.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from save-dir/student_model/ (or student_weights.pt); --checkpoint-run for nested legacy path",
    )
    parser.add_argument(
        "--checkpoint-run",
        default=None,
        metavar="PATH",
        help="Optional path under --save-dir for an older nested run (e.g. timestamp subfolder).",
    )
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Circuit: skip ablation (expects cached ablation JSON under run dir)")
    parser.add_argument("--ablation-batch-size", type=int, default=50,
                        help="Circuit: batch size for neuron-cluster ablation")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for Python/NumPy/torch and train DataLoader shuffle (default: 42)",
    )
    args = parser.parse_args()
    if args.count_flops_every < 0:
        raise SystemExit("--count-flops-every must be >= 0 (use 0 to disable FLOP counting)")
    if args.eval_batch_size < 1:
        raise SystemExit("--eval-batch-size must be >= 1")

    try:
        train_path, test_path, dataset_prefix = resolve_train_test_paths(
            dataset=args.dataset,
            datasets_dir=args.datasets_dir,
        )
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e

    eval_max_new_tokens = (
        args.eval_max_new_tokens
        if args.eval_max_new_tokens is not None
        else EVAL_MAX_NEW_TOKENS
    )
    if args.eval_print_samples < 0:
        raise SystemExit("--eval-print-samples must be >= 0")
    args.save_dir = os.path.abspath(args.save_dir)

    run_dir, student_source = resolve_distillation_run_dir(
        args.save_dir,
        resume=args.resume,
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
            importance_vs_poly=not args.no_poly_importance,
            student_poly_json=args.student_poly_json,
            teacher_poly_json=args.teacher_poly_json,
            student_model_name=args.student_model,
            teacher_model_name=args.teacher_model,
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
    student, tokenizer = load_student_model_for_distillation(
        student_source, args.student_model, device,
    )

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
        cluster_size_weighting=args.cluster_size_weighting,
        save_every=args.save_every,
        save_best=args.save_best,
        eval_max_new_tokens=eval_max_new_tokens,
        eval_print_samples=args.eval_print_samples,
        eval_batch_size=args.eval_batch_size,
        save_dir=run_dir,
        count_flops_every=args.count_flops_every,
        log_kl_cka_grad_norms=args.log_kl_cka_grad_norms,
        seed=args.seed,
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

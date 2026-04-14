"""Circuit distillation with **random** flattened-MLP neuron sets per cluster pair.

Same steps as ``neuron_distillation.run`` in circuit mode (ablation, pairing, KL + CKA),
except after ``build_cluster_pairs`` we replace each pair's neuron indices with a random
subset of size ``round(fraction * D)`` (per model), where ``D = layers * intermediate_size``.

Usage (from ``src/``)::

  python -m experiments.random_neuron_align.run \\
    --dataset 2d_add --save-dir /path/to/results --k 7 \\
    --random-fraction 0.1 --random-seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from .random_pairs import (
    mlp_flatten_dim,
    replace_pairs_with_random_neurons,
    save_manifest,
)
from neuron_distillation.distillation import ClusterDistillationConfig, ClusterDistillationTrainer
from neuron_distillation.run import (
    _ablation_dir_for_model,
    _find_circuit_checkpoint_pt,
    build_cluster_pairs,
    run_ablation_if_needed,
)
from utils import (
    EVAL_MAX_NEW_TOKENS,
    _extract_circuit_model_state_dict,
    load_model_checkpoint,
    load_prompt_answer_json,
    load_student_model_for_distillation,
    resolve_distillation_run_dir,
    resolve_train_test_paths,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Random-neuron circuit distillation (same pipeline, random CKA subsets)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
        help="Clusters per subclass (required for new runs).",
    )
    parser.add_argument(
        "--k-classes", type=int, default=None,
        help="Latent subclasses (auto-detected from checkpoint if omitted)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Circuit-discovery .pt under --save-dir root if omitted",
    )
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
            "CKA: multiply each cluster pair's loss weight by |student cluster| / full student MLP width"
        ),
    )
    parser.add_argument(
        "--no-poly-importance",
        action="store_true",
        help="Pairing: use raw ablation drops; do not subtract poly expected drop at |C|/D",
    )
    parser.add_argument(
        "--student-poly-json",
        type=str,
        default=None,
        metavar="PATH",
        help="Override default random_ablation_poly_1b.json for residual importance",
    )
    parser.add_argument(
        "--teacher-poly-json",
        type=str,
        default=None,
        metavar="PATH",
        help="Override default random_ablation_poly_8b.json for residual importance",
    )
    parser.add_argument("--save-every", type=int, default=5,
                        help="Overwrite student_model/ every N epochs (0=disable)")
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--step-log-interval", type=int, default=50)
    parser.add_argument(
        "--log-kl-cka-grad-norms",
        action="store_true",
        help=(
            "Log ‖g_KL‖ and ‖g_{λ·CKA}‖ (autograd.grad; ~2× work); scale λ·CKA grads by "
            "‖g_KL‖/‖g_{λ·CKA}‖"
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
            "Each eval: print first N test prompts and full greedy decodes (0=off). "
            "Baselines print N for student and N for teacher."
        ),
    )
    parser.add_argument(
        "--count-flops-every",
        type=int,
        default=0,
        metavar="N",
        help="Count training FLOPs every N epochs (0=never). Default 0.",
    )
    parser.add_argument(
        "--save-dir", type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "results", "random-neuron-distillation",
        ),
        help="Run directory (ablation/, student_model/, …)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-run", default=None, metavar="PATH")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--ablation-batch-size", type=int, default=50)
    parser.add_argument(
        "--random-fraction",
        type=float,
        required=True,
        metavar="F",
        help="Fraction of flattened MLP width to sample per side (0, 1]",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for distillation (DataLoader shuffle; default: 42). "
        "Random neuron subsets use --random-seed.",
    )
    parser.add_argument(
        "--uniform-importance",
        action="store_true",
        help="Ignore mapping importances; weight every pair equally for CKA",
    )
    args = parser.parse_args()
    if args.count_flops_every < 0:
        raise SystemExit("--count-flops-every must be >= 0")

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
        checkpoint_run=args.checkpoint_run,
    )
    os.makedirs(run_dir, exist_ok=True)
    is_resume = student_source is not None

    neuron_clustering_root = os.path.join(args.save_dir, "neuron-clustering")
    student_clusters = os.path.join(neuron_clustering_root, args.student_model, "clusters")
    teacher_clusters = os.path.join(neuron_clustering_root, args.teacher_model, "clusters")
    student_ablation_dir = _ablation_dir_for_model(run_dir, args.student_model)
    teacher_ablation_dir = _ablation_dir_for_model(run_dir, args.teacher_model)
    mapping_cache = os.path.join(run_dir, "cluster_mapping.json")

    print(f"Run dir: {run_dir}")
    print("  experiment: random_neuron_align")

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
                    f"  {args.save_dir}",
                )
                sys.exit(1)
        if args.k_classes is None:
            ckpt_data = torch.load(args.checkpoint, map_location="cpu")
            state = _extract_circuit_model_state_dict(ckpt_data, args.checkpoint)
            if "classifier.classifier.4.weight" in state:
                args.k_classes = state["classifier.classifier.4.weight"].shape[0]
                print(f"Auto-detected k_classes={args.k_classes} from checkpoint")
            else:
                print("ERROR: Could not detect k_classes from checkpoint. Pass --k-classes.")
                sys.exit(1)
            del ckpt_data

    if args.k_classes is None:
        raise SystemExit(
            "ERROR: Could not determine k_classes. Pass --k-classes or place a circuit .pt under --save-dir.",
        )

    if args.k is None and not is_resume:
        sp = os.path.join(neuron_clustering_root, args.student_model, "plots")
        tp = os.path.join(neuron_clustering_root, args.teacher_model, "plots")
        raise SystemExit(
            "ERROR: --k INT is required.\n"
            f"  {sp}\n  {tp}",
        )

    if is_resume and args.k is None and os.path.isfile(mapping_cache):
        with open(mapping_cache, encoding="utf-8") as f:
            raw = json.load(f)
        if raw:
            args.k = raw[0].get("k", None) or 7
            print(f"Inferred k={args.k} from cached mapping")

    if is_resume and args.k is None:
        raise SystemExit("ERROR: --k is required for resume (could not infer from cache).")

    print("=" * 60)
    print("Configuration (random neuron align / circuit)")
    print("=" * 60)
    print(f"  k_classes:          {args.k_classes}")
    print(f"  k (clusters):       {args.k}")
    print(f"  random_fraction:    {args.random_fraction}")
    print(f"  random_seed:        {args.random_seed}")
    print(f"  uniform_importance: {args.uniform_importance}")
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
    print("Step 2: Cluster pairing (topology; indices replaced randomly next)")
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

    D_s = mlp_flatten_dim(args.student_model)
    D_t = mlp_flatten_dim(args.teacher_model)
    n_s = max(1, min(D_s, int(round(args.random_fraction * D_s))))
    n_t = max(1, min(D_t, int(round(args.random_fraction * D_t))))
    cluster_pairs = replace_pairs_with_random_neurons(
        cluster_pairs,
        D_student=D_s,
        D_teacher=D_t,
        fraction=args.random_fraction,
        seed=args.random_seed,
        keep_importance_weights=not args.uniform_importance,
    )
    manifest_path = os.path.join(run_dir, "random_neuron_manifest.json")
    save_manifest(
        manifest_path,
        fraction=args.random_fraction,
        seed=args.random_seed,
        D_student=D_s,
        D_teacher=D_t,
        n_student_sampled=n_s,
        n_teacher_sampled=n_t,
        student_model=args.student_model,
        teacher_model=args.teacher_model,
        n_pairs=len(cluster_pairs),
    )
    print(f"  Replaced neuron indices with random subsets ({n_s} student, {n_t} teacher per pair).")
    print(f"  Wrote {manifest_path}")

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
        distillation_mode="circuit",
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

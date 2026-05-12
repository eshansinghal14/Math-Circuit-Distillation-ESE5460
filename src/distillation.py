"""CLI for simplified KL + graph-loss distillation."""

from __future__ import annotations

import argparse
import os
from typing import Dict

from graph_loss.utils import DTYPE_CHOICES, resolve_torch_dtype
from utils import (
    EVAL_MAX_NEW_TOKENS,
    LLAMA_1B_MODEL_NAME,
    LLAMA_8B_MODEL_NAME,
    DistillationConfig,
    DistillationTrainer,
    get_default_device,
    load_prompt_answer_json,
    load_student_model_for_distillation,
    resolve_distillation_run_dir,
    resolve_test_path,
    resolve_train_test_paths,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simplified KL distillation with optional graph-loss auxiliary objective.",
    )
    parser.add_argument("--student-model", type=str, default=LLAMA_1B_MODEL_NAME)
    parser.add_argument("--teacher-model", type=str, default=LLAMA_8B_MODEL_NAME)
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        metavar="PREFIX",
        help="Dataset prefix, e.g. 2d_add for datasets/2d_add_train.json + _test.json.",
    )
    parser.add_argument("--datasets-dir", type=str, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-graph", type=float, default=0.1)
    parser.add_argument("--dtype", choices=DTYPE_CHOICES, default="float32")
    parser.add_argument("--graph-top-k-logits", type=int, default=20)
    parser.add_argument("--graph-prop-neurons-per-layer", type=float, default=0.1)
    parser.add_argument("--graph-batch-size", type=int, default=None)
    parser.add_argument(
        "--graph-gen-batch-size",
        type=int,
        default=1,
        help=(
            "Number of prompt graphs to generate and retain before backpropagating "
            "graph loss. Higher is faster but uses more memory."
        ),
    )
    parser.add_argument("--teacher-graph-batch-size", type=int, default=512)
    parser.add_argument("--student-graph-batch-size", type=int, default=1)
    parser.add_argument(
        "--student-computation-eps",
        type=float,
        default=0.1,
        help="Angular distance threshold for student supernode clustering.",
    )
    parser.add_argument(
        "--student-embedding-eps",
        type=float,
        default=0.1,
        help="Angular distance threshold for student embedding supernode clustering.",
    )
    parser.add_argument(
        "--student-activation-forward-batch-size",
        type=int,
        default=32,
        help="Batch size for student ablation forwards during supernode clustering.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--graph-prune", action="store_true")
    parser.add_argument("--graph-node-threshold", type=float, default=0.8)
    parser.add_argument("--graph-edge-threshold", type=float, default=0.98)
    parser.add_argument("--graph-node-weight", type=float, default=1.0)
    parser.add_argument("--graph-edge-weight", type=float, default=0.0)
    parser.add_argument(
        "--graph-focus-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for Phase-3 logit-focus distribution loss "
            "(KL(teacher_focus || student_focus), no alignment required). "
            "Set >0 to enable, e.g. --graph-focus-weight 1.0. "
            "When this is the only active graph loss term, also set "
            "--graph-node-weight 0 --graph-edge-weight 0."
        ),
    )
    parser.add_argument("--graph-similarity-threshold", type=float, default=0.7)
    parser.add_argument("--graph-max-fan-out", type=int, default=4)
    parser.add_argument("--fast-teacher-graph", action="store_true")
    parser.add_argument(
        "--graph-grad-mode",
        type=str,
        default="approx",
        choices=["approx", "true"],
        help=(
            "Gradient pathway for per-supernode prob-delta loss.  "
            "'approx' (default): cheap differentiable DLA-approximation, "
            "one forward + per-supernode (norm/lm_head/softmax).  Fast, "
            "~95%% directionally faithful.  "
            "'true': full hook-based ablation with chunked sequential "
            "backward, 100%% faithful to teacher's operational space at "
            "~3-5x training cost.  Tune chunk size with --graph-true-grad-chunk-size."
        ),
    )
    parser.add_argument(
        "--graph-true-grad-chunk-size",
        type=int,
        default=4,
        help=(
            "In --graph-grad-mode true, # of supernodes per batched "
            "ablation forward+backward.  Higher = more parallelism, "
            "more peak memory.  Default 4."
        ),
    )
    parser.add_argument(
        "--fast-student-graph",
        action="store_true",
        help=(
            "Skip the [neurons, neurons] Jacobian backward passes when "
            "building the student graph.  Selects neurons + builds a "
            "DLA-only adjacency, then clusters via real ablation.  "
            "Required when running node-loss-only (much faster).  "
            "Must be combined with --graph-edge-weight 0."
        ),
    )
    parser.add_argument(
        "--lambda-kl",
        type=float,
        default=1.0,
        help=(
            "Scale factor for the KL distillation loss.  Set to 0 to run "
            "graph-loss-only (diagnostic: tests whether graph signal alone "
            "carries useful gradient).  Default 1.0."
        ),
    )
    parser.add_argument(
        "--graph-grad-norm-scale",
        action="store_true",
        default=False,
        help=(
            "Dynamically scale the graph loss contribution so its gradient norm "
            "equals the KL gradient norm before applying lambda_graph.  Prevents "
            "graph gradients from dominating when GraphActive spikes.  "
            "Adds ~1 GB memory overhead to store the KL gradient snapshot."
        ),
    )
    parser.add_argument(
        "--ablation-batch-size",
        type=int,
        default=32,
        help=(
            "Supernodes ablated per batched forward when computing the "
            "no-grad student matching signal.  Raise on big GPUs to "
            "speed up per-prompt processing (≈linear).  Default 32."
        ),
    )
    parser.add_argument(
        "--student-skip-logit-attribution",
        action="store_true",
        help=(
            "If set, the student attribution graph keeps the legacy "
            "skip_logit_attribution=True (saves memory but the student supernode "
            "alignment signal is sum-of-DLAs, which lives in a different functional "
            "space than the teacher's cached prob-deltas).  Default is to populate "
            "real Jacobian logit-influence rows and use those for alignment."
        ),
    )
    parser.add_argument(
        "--align-diagnostic",
        action="store_true",
        help=(
            "Log per-prompt teacher/student supernode cosine-matrix stats "
            "(mean/median/max, fraction of teacher SNs with any match >= 0.3) "
            "to verify alignment quality before/after a fix."
        ),
    )
    parser.add_argument("--eval-max-new-tokens", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=50)
    parser.add_argument("--eval-datasets", nargs="*", default=None, metavar="DATASET")
    parser.add_argument(
        "--track-loss-grads",
        action="store_true",
        help="Log KL and graph gradient norms plus their cosine similarity.",
    )
    parser.add_argument("--teacher-data-cache", type=str, default=None, metavar="PATH")
    parser.add_argument(
        "--graph-align-by-label",
        action="store_true",
        help=(
            "Align teacher↔student supernodes by ANOVA label string instead of "
            "cosine similarity of prob-delta vectors.  Requires teacher cache built "
            "with full_search (labels saved) and --student-cluster-method full_search."
        ),
    )
    parser.add_argument(
        "--student-cluster-method",
        type=str,
        default="ablation",
        choices=["ablation", "full_search"],
        help="Supergraph clustering method for the student.  full_search builds "
             "activation heatmaps (requires --student-dataset).",
    )
    parser.add_argument(
        "--student-dataset",
        type=str,
        default=None,
        metavar="DATASET",
        help="Activation dataset for student full_search clustering (e.g. 22_add_tight_all.json).",
    )
    parser.add_argument(
        "--student-mlp-input-cache",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to pre-computed student MLP input cache (speeds up full_search heatmaps).",
    )
    parser.add_argument(
        "--student-activation-write-cache",
        type=str,
        default=None,
        metavar="PATH",
        help="Local path for caching student activation-write grids between steps.",
    )
    parser.add_argument(
        "--student-anova-range-radius",
        type=int,
        default=0,
        help="Radius around target arg/sum values for the student ANOVA range mask. "
             "Must match the teacher cache value for label alignment to be meaningful. "
             "5 is a sensible decade-width default; 0 (legacy) uses exact-value masks "
             "which can't distinguish range vs. units neurons.",
    )
    parser.add_argument(
        "--student-anova-nodes-per-label",
        type=int,
        default=10,
        help="Cap on neurons per labelled student supernode (default 10).",
    )
    parser.add_argument(
        "--student-sum-min-specificity",
        type=float,
        default=0.0,
        help="Minimum ANOVA specificity for sum-label student supernodes.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "results", "distillation"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-run", default=None, metavar="PATH")
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--step-log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.dataset is None:
        raise SystemExit("--dataset is required.")
    if args.steps < 0:
        raise SystemExit("--steps must be >= 0")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.eval_batch_size < 1:
        raise SystemExit("--eval-batch-size must be >= 1")
    if args.step_log_interval < 1:
        raise SystemExit("--step-log-interval must be >= 1")
    if args.lambda_graph < 0:
        raise SystemExit("--lambda-graph must be >= 0")
    if args.graph_top_k_logits is not None and args.graph_top_k_logits <= 0:
        raise SystemExit("--graph-top-k-logits must be positive")
    if not (0.0 < args.graph_prop_neurons_per_layer <= 1.0):
        raise SystemExit("--graph-prop-neurons-per-layer must be in (0, 1]")
    if args.graph_batch_size is not None and args.graph_batch_size < 1:
        raise SystemExit("--graph-batch-size must be >= 1")
    if args.graph_gen_batch_size < 1:
        raise SystemExit("--graph-gen-batch-size must be >= 1")
    if args.teacher_graph_batch_size < 1:
        raise SystemExit("--teacher-graph-batch-size must be >= 1")
    if args.student_graph_batch_size < 1:
        raise SystemExit("--student-graph-batch-size must be >= 1")
    if args.student_computation_eps < 0:
        raise SystemExit("--student-computation-eps must be >= 0")
    if args.student_embedding_eps < 0:
        raise SystemExit("--student-embedding-eps must be >= 0")
    if args.student_activation_forward_batch_size < 1:
        raise SystemExit("--student-activation-forward-batch-size must be >= 1")
    if not (0.0 <= args.graph_node_threshold <= 1.0):
        raise SystemExit("--graph-node-threshold must be in [0, 1]")
    if not (0.0 <= args.graph_edge_threshold <= 1.0):
        raise SystemExit("--graph-edge-threshold must be in [0, 1]")
    if not (0.0 <= args.graph_similarity_threshold <= 1.0):
        raise SystemExit("--graph-similarity-threshold must be in [0, 1]")
    if args.graph_max_fan_out < 1:
        raise SystemExit("--graph-max-fan-out must be >= 1")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    if args.graph_batch_size is not None:
        args.teacher_graph_batch_size = args.graph_batch_size
        args.student_graph_batch_size = args.graph_batch_size

    train_path, test_path, dataset_prefix = resolve_train_test_paths(
        dataset=args.dataset,
        datasets_dir=args.datasets_dir,
    )
    run_dir, student_source = resolve_distillation_run_dir(
        os.path.abspath(args.save_dir),
        resume=args.resume,
        checkpoint_run=args.checkpoint_run,
    )
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 60)
    print("Configuration")
    print("=" * 60)
    print(f"  dataset (prefix):   {dataset_prefix}")
    print(f"  train_path:         {train_path}")
    print(f"  test_path:          {test_path}")
    print(f"  lambda_graph:       {args.lambda_graph}")
    print(f"  graph node weight:  {args.graph_node_weight}")
    print(f"  graph edge weight:  {args.graph_edge_weight}")
    print(f"  graph focus weight: {args.graph_focus_weight}")
    print(f"  graph grad mode:    {args.graph_grad_mode}", end="")
    if args.graph_grad_mode == "true":
        print(f" (chunk_size={args.graph_true_grad_chunk_size})")
    else:
        print()
    if args.fast_student_graph:
        print(f"  fast student graph: True (node-loss-only fast path)")
    print(f"  graph dtype:        {args.dtype}")
    print(f"  teacher cache:      {args.teacher_data_cache or 'none'}")
    print(f"  save_dir:           {run_dir}")
    print("=" * 60)

    train_data = load_prompt_answer_json(train_path)
    if args.train_limit is not None:
        train_data = dict(list(train_data.items())[: args.train_limit])
    test_data = load_prompt_answer_json(test_path)
    extra_eval_data: Dict[str, Dict[str, int]] = {}
    if args.eval_datasets:
        for eval_dataset in args.eval_datasets:
            eval_test_path, eval_prefix = resolve_test_path(
                dataset=eval_dataset,
                datasets_dir=args.datasets_dir,
            )
            extra_eval_data[eval_prefix] = load_prompt_answer_json(eval_test_path)
    print(f"  Train: {len(train_data)} examples")
    print(f"  Test:  {len(test_data)} examples")

    device = get_default_device()
    student, tokenizer = load_student_model_for_distillation(
        student_source,
        args.student_model,
        device,
    )

    trainer = DistillationTrainer(
        config=DistillationConfig(
            teacher_model=args.teacher_model,
            student_model=args.student_model,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            temperature=args.temperature,
            grad_clip=args.grad_clip,
            lambda_graph=args.lambda_graph,
            graph_dtype=resolve_torch_dtype(args.dtype),
            graph_top_k_logits=args.graph_top_k_logits,
            graph_prop_neurons_per_layer=args.graph_prop_neurons_per_layer,
            graph_gen_batch_size=args.graph_gen_batch_size,
            teacher_graph_batch_size=args.teacher_graph_batch_size,
            student_graph_batch_size=args.student_graph_batch_size,
            graph_verbose=args.verbose,
            graph_prune=args.graph_prune,
            graph_node_threshold=args.graph_node_threshold,
            graph_edge_threshold=args.graph_edge_threshold,
            graph_node_weight=args.graph_node_weight,
            graph_edge_weight=args.graph_edge_weight,
            graph_similarity_threshold=args.graph_similarity_threshold,
            graph_max_fan_out=args.graph_max_fan_out,
            fast_teacher_graph=args.fast_teacher_graph,
            student_computation_eps=args.student_computation_eps,
            student_embedding_eps=args.student_embedding_eps,
            student_activation_forward_batch_size=args.student_activation_forward_batch_size,
            student_skip_logit_attribution=args.student_skip_logit_attribution,
            align_diagnostic=args.align_diagnostic,
            graph_focus_weight=args.graph_focus_weight,
            graph_grad_mode=args.graph_grad_mode,
            graph_true_grad_chunk_size=args.graph_true_grad_chunk_size,
            fast_student_graph=args.fast_student_graph,
            ablation_batch_size=args.ablation_batch_size,
            lambda_kl=args.lambda_kl,
            graph_grad_norm_scale=args.graph_grad_norm_scale,
            save_best=args.save_best,
            eval_max_new_tokens=(
                args.eval_max_new_tokens
                if args.eval_max_new_tokens is not None
                else EVAL_MAX_NEW_TOKENS
            ),
            eval_batch_size=args.eval_batch_size,
            track_loss_grads=args.track_loss_grads,
            save_dir=run_dir,
            teacher_data_cache=args.teacher_data_cache,
            align_by_label=args.graph_align_by_label,
            student_cluster_method=args.student_cluster_method,
            student_dataset=args.student_dataset,
            student_mlp_input_cache_path=args.student_mlp_input_cache,
            student_activation_write_cache_path=args.student_activation_write_cache,
            student_anova_range_radius=args.student_anova_range_radius,
            student_anova_nodes_per_label=args.student_anova_nodes_per_label,
            student_sum_min_specificity=args.student_sum_min_specificity,
            step_log_interval=args.step_log_interval,
            seed=args.seed,
            device=device,
        ),
        train_data=train_data,
        test_data=test_data,
        extra_eval_data=extra_eval_data,
        tokenizer=tokenizer,
        student=student,
        resume=student_source is not None,
    )
    history = trainer.train()
    if "accuracy" in history and history["accuracy"]:
        print(f"Best accuracy: {max(history['accuracy']):.4f}")


if __name__ == "__main__":
    main()

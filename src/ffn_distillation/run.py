"""Single entry-point for FFN layer-level CKA distillation.

Runs the full pipeline:  layer ablation -> layer pairing -> distillation training.

Usage (from src/):
  python -m ffn_distillation.run --dataset 2d_add --save-dir /path/to/results/ffn-layer

``--dataset PREFIX`` is required: it resolves train/test JSON and, unless ``--ablation-dataset`` is set,
``<PREFIX>_all.json`` for ablation (see ``utils``). ``--ablation-dataset`` overrides only the ``*_all.json`` path.
"""

import argparse
import os
import sys

from ffn_distillation.layer_ablation import layer_ablation
from ffn_distillation.layer_pairing import (
    get_layer_pairs,
    save_layer_pairs,
)
from ffn_distillation.distillation import (
    FFNDistillationConfig,
    FFNDistillationTrainer,
)
from utils import EVAL_MAX_NEW_TOKENS, resolve_ablation_all_path, resolve_train_test_paths


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="FFN layer-level CKA distillation (end-to-end)",
    )
    parser.add_argument("--student-model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--teacher-model", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument(
        "--dataset",
        default=None,
        metavar="PREFIX",
        help="e.g. 2d_add -> train/test JSON and <PREFIX>_all.json for ablation (required)",
    )
    parser.add_argument(
        "--datasets-dir",
        default=None,
        help="Directory containing dataset JSONs (default: repo datasets/)",
    )
    parser.add_argument(
        "--ablation-dataset",
        default=None,
        help="Override full JSON for layer ablation (default: <PREFIX>_all.json from --dataset)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-cka", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save student_epoch_N every N epochs; only most recent kept (0=off)")
    parser.add_argument("--save-best", action="store_true",
                        help="Save student_best whenever eval accuracy improves (off by default)")
    parser.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=None,
        help="Greedy test accuracy during training (default: utils.EVAL_MAX_NEW_TOKENS)",
    )
    parser.add_argument("--save-dir", default=os.path.join(script_dir, "..", "..", "results", "ffn-distillation"))
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip ablation (assumes layer_ablation_performance.json exists)")
    args = parser.parse_args()

    try:
        train_path, test_path, prefix = resolve_train_test_paths(
            dataset=args.dataset,
            datasets_dir=args.datasets_dir,
        )
        ablation_dataset = resolve_ablation_all_path(
            dataset=args.dataset,
            ablation_path=args.ablation_dataset,
            datasets_dir=args.datasets_dir,
            prefix=prefix,
        )
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e

    args.train_path = train_path
    args.test_path = test_path
    args.ablation_dataset = ablation_dataset

    eval_max_new_tokens = (
        args.eval_max_new_tokens
        if args.eval_max_new_tokens is not None
        else EVAL_MAX_NEW_TOKENS
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # Paths for ablation results
    student_abl_dir = os.path.join(args.save_dir, "ablation", args.student_model)
    teacher_abl_dir = os.path.join(args.save_dir, "ablation", args.teacher_model)
    student_abl_path = os.path.join(student_abl_dir, "layer_ablation_performance.json")
    teacher_abl_path = os.path.join(teacher_abl_dir, "layer_ablation_performance.json")

    # ---- Step 1: Layer ablation ----
    print("=" * 60)
    print("Step 1: MLP layer ablation")
    print("=" * 60)

    if args.skip_ablation:
        print(f"  Skipping ablation. Using:\n    {student_abl_path}\n    {teacher_abl_path}")
    else:
        if not os.path.exists(student_abl_path):
            print(f"\n  Ablating student: {args.student_model}")
            layer_ablation(
                model_name=args.student_model,
                dataset_path=args.ablation_dataset,
                results_dir=student_abl_dir,
                max_new_tokens=eval_max_new_tokens,
            )
        else:
            print(f"  Student ablation exists: {student_abl_path}")

        if not os.path.exists(teacher_abl_path):
            print(f"\n  Ablating teacher: {args.teacher_model}")
            layer_ablation(
                model_name=args.teacher_model,
                dataset_path=args.ablation_dataset,
                results_dir=teacher_abl_dir,
                max_new_tokens=eval_max_new_tokens,
            )
        else:
            print(f"  Teacher ablation exists: {teacher_abl_path}")

    # ---- Step 2: Layer pairing ----
    print("\n" + "=" * 60)
    print("Step 2: MLP layer pairing")
    print("=" * 60)

    layer_pairs = get_layer_pairs(
        student_ablation_path=student_abl_path if os.path.exists(student_abl_path) else None,
        teacher_ablation_path=teacher_abl_path if os.path.exists(teacher_abl_path) else None,
    )

    print(f"  {len(layer_pairs)} layer pairs:")
    for p in layer_pairs:
        print(f"    Student {p.student_layer:2d} -> Teacher {p.teacher_layer:2d} "
              f"(dist={p.distance:.3f})")

    save_layer_pairs(layer_pairs, os.path.join(args.save_dir, "layer_pairs.json"))

    if not layer_pairs:
        print("No layer pairs found. Exiting.")
        sys.exit(1)

    # ---- Step 3: Distillation training ----
    print("\n" + "=" * 60)
    print("Step 3: FFN CKA distillation training")
    print("=" * 60)

    config = FFNDistillationConfig(
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        temperature=args.temperature,
        lambda_cka=args.lambda_cka,
        save_every=args.save_every,
        save_best=args.save_best,
        eval_max_new_tokens=eval_max_new_tokens,
        save_dir=args.save_dir,
    )

    trainer = FFNDistillationTrainer(
        config=config,
        layer_pairs=layer_pairs,
        train_path=args.train_path,
        test_path=args.test_path,
    )

    history = trainer.train()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    if "accuracy" in history and history["accuracy"]:
        print(f"  Best accuracy: {max(history['accuracy']):.4f}")
    print(f"  Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()

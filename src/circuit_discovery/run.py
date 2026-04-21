"""Command-line entry for circuit discovery training.

Run from ``src/``::

  python -m circuit_discovery.run --k-classes 8 --dataset 2d_add

Or::

  python -m circuit_discovery --k-classes 8 --dataset 2d_add
"""

from __future__ import annotations

import argparse

from .main import train_circuit_discovery


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Circuit discovery training")
    parser.add_argument(
        "--k-classes",
        type=int,
        required=True,
        help="Number of circuit classes",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset family prefix used to load activations (e.g. 2d_add)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20000,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Circuit discovery checkpoint to resume from",
    )
    parser.add_argument(
        "--lambda-usage",
        type=float,
        default=0.15,
        help="Weight for class usage entropy (auxiliary); lambda_sim = 1 - sum(auxiliary)",
    )
    parser.add_argument(
        "--lambda-mask-cossim",
        type=float,
        default=0.25,
        help="Weight for mask orthogonality (auxiliary)",
    )
    parser.add_argument(
        "--lambda-kl",
        type=float,
        default=0.15,
        help="Weight for KL to 10% prior (auxiliary)",
    )
    parser.add_argument(
        "--lambda-sparsity",
        type=float,
        default=0.20,
        help="Weight for mask sparsity (auxiliary)",
    )
    parser.add_argument(
        "--mask-temperature",
        type=float,
        default=1.0,
        help="Constant mask sigmoid temperature: sigmoid(logits/T); lower = sharper (stored in checkpoint)",
    )
    parser.add_argument(
        "--mask-activate-threshold",
        type=float,
        default=0.99,
        help="Mask value above this counts as activated for frac_activated_* metrics (not used in loss)",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Max L2 norm for global gradient clipping (0 disables)",
    )
    parser.add_argument(
        "--class-reweight",
        action="store_true",
        help="Inverse-frequency (add-one) weights for sim and KL; per-example weights for sparsity (mask cossim unchanged)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    train_circuit_discovery(
        k_classes=args.k_classes,
        dataset_prefix=args.dataset,
        epochs=args.epochs,
        resume_model=args.checkpoint_path,
        lambda_usage=args.lambda_usage,
        lambda_mask_cossim=args.lambda_mask_cossim,
        lambda_kl=args.lambda_kl,
        lambda_sparsity=args.lambda_sparsity,
        mask_temperature=args.mask_temperature,
        mask_activate_threshold=args.mask_activate_threshold,
        grad_clip_norm=args.grad_clip_norm,
        class_reweight=args.class_reweight,
    )


if __name__ == "__main__":
    main()

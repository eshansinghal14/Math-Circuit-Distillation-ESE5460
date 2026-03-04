import argparse

from circuit_discovery import (  # noqa: F401
    ProblemEncoder,
    ProblemClassifier,
    NeuronMask,
    CircuitDiscoveryModel,
    CircuitLoss,
    parse_equation,
    _stack_layer_activations,
    train_circuit_discovery,
)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Circuit discovery training")
    parser.add_argument(
        "--k-classes",
        type=int,
        required=True,
        help="Number of circuit classes",
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
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    train_circuit_discovery(
        k_classes=args.k_classes,
        epochs=args.epochs,
        resume_model=args.checkpoint_path,
        lambda_usage=args.lambda_usage,
        lambda_mask_cossim=args.lambda_mask_cossim,
        lambda_kl=args.lambda_kl,
        lambda_sparsity=args.lambda_sparsity,
    )

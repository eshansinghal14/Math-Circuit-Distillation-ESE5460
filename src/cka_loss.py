"""Linear CKA (efficient formulation) used by neuron-cluster alignment in ``distillation``."""

import torch
from torch import Tensor


def linear_cka_efficient(X: Tensor, Y: Tensor, eps: float = 1e-8) -> Tensor:
    assert X.shape[0] == Y.shape[0], f"Batch sizes must match: {X.shape[0]} vs {Y.shape[0]}"

    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    YtX = Y.T @ X
    XtX = X.T @ X
    YtY = Y.T @ Y

    numerator = (YtX**2).sum()
    denominator = torch.sqrt((XtX**2).sum() * (YtY**2).sum() + eps)

    cka = numerator / denominator

    return torch.clamp(cka, 0.0, 1.0)


if __name__ == "__main__":
    print("Testing linear_cka_efficient...")

    X = torch.randn(32, 128)
    cka = linear_cka_efficient(X, X)
    print(f"CKA(X, X) = {cka.item():.6f} (expect ~1.0)")
    assert abs(cka.item() - 1.0) < 0.01

    Y = torch.randn(32, 64)
    cka = linear_cka_efficient(X, Y)
    print(f"CKA(X, Y) = {cka.item():.6f} (random)")
    assert 0.0 <= cka.item() <= 1.0

    Xg = torch.randn(32, 128, requires_grad=True)
    Yg = torch.randn(32, 64)
    cka = linear_cka_efficient(Xg, Yg)
    (1.0 - cka).backward()
    assert Xg.grad is not None
    print(f"Gradient norm: {Xg.grad.norm().item():.6f}")

    print("All tests passed.")

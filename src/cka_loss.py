"""Linear CKA (efficient formulation) used by neuron-cluster alignment in ``distillation``."""

import torch
from torch import Tensor


def center_columns(X: Tensor) -> Tensor:
    """Subtract column means (same first step as :func:`linear_cka_efficient`).

    ``X`` is ``(n_samples, n_features)`` — one row per sample.
    """
    return X - X.mean(dim=0, keepdim=True)


def leading_eigenvalue_hkh(X: Tensor) -> Tensor:
    """Largest eigenvalue of ``K_c = H K H`` with ``K = X X^T`` and ``H`` column-centering.

    Rows of ``X`` are samples (same layout as :func:`linear_cka_efficient`). Then
    ``K_c = X_c X_c.T`` with ``X_c =`` :func:`center_columns` ``(X)``, matching
    ``H X X^T H`` for ``H = I - (1/n) 11^T``.

    Returns:
        Scalar tensor (same device as ``X``); ``nan`` if ``X`` is empty or invalid shape.
    """
    if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] < 1:
        return torch.tensor(float("nan"), dtype=torch.float32, device=X.device)
    Xc = center_columns(X).float()
    n, d = Xc.shape
    if n >= d:
        W = Xc.T @ Xc
    else:
        W = Xc @ Xc.T
    lam = torch.linalg.eigvalsh(W)
    lam = lam.clamp(min=0.0)
    return lam[-1]


def stable_rank_centered_gram(X: Tensor, eps: float = 1e-12) -> Tensor:
    """Stable rank of ``G = X_c X_c.T`` with column-centered ``X_c``.

    Uses ``||G||_F^2 / ||G||_2^2 = sum_i λ_i^2 / λ_max^2`` for eigenvalues ``λ_i`` of ``G``.
    Squared singular values of ``X_c`` are eigenvalues of ``X_c.T @ X_c`` (or ``X_c @ X_c.T``);
    we form the smaller Gram matrix and use ``eigvalsh`` (faster than ``svdvals`` on large ``X``).

    Runs on ``X``'s device (CPU or CUDA); for large matrices, pass ``X`` on GPU for faster linear algebra.

    Args:
        X: ``(n_samples, n_features)`` — one row per sample (e.g. token positions).

    Returns:
        Scalar tensor; ``nan`` if ``X`` is empty or has zero width/height.
    """
    if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] < 1:
        return torch.tensor(float("nan"), dtype=X.dtype if X.numel() else torch.float32)
    Xc = center_columns(X).float()
    n, d = Xc.shape
    # Smaller symmetric Gram: same nonzero squared singular values as full SVD of Xc.
    if n >= d:
        W = Xc.T @ Xc
    else:
        W = Xc @ Xc.T
    lam = torch.linalg.eigvalsh(W)
    lam = lam.clamp(min=0.0)
    num = (lam**2).sum()
    den = lam[-1].clamp_min(eps) ** 2
    return num / den


def linear_cka_efficient(X: Tensor, Y: Tensor, eps: float = 1e-8) -> Tensor:
    assert X.shape[0] == Y.shape[0], f"Batch sizes must match: {X.shape[0]} vs {Y.shape[0]}"

    X = center_columns(X)
    Y = center_columns(Y)

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

    Xr = torch.randn(32, 8)
    sr = stable_rank_centered_gram(Xr)
    assert torch.isfinite(sr) and sr.item() >= 1.0 - 1e-5
    print(f"stable_rank_centered_gram(random X) = {sr.item():.6f}")

    lam1 = leading_eigenvalue_hkh(Xr)
    assert torch.isfinite(lam1) and lam1.item() >= 0.0
    print(f"leading_eigenvalue_hkh(random X) = {lam1.item():.6f}")

    if torch.cuda.is_available():
        sr_gpu = stable_rank_centered_gram(Xr.cuda())
        assert torch.isfinite(sr_gpu) and torch.allclose(
            sr.cpu(), sr_gpu.cpu(), rtol=1e-4, atol=1e-5
        )
        print("stable_rank_centered_gram CPU vs CUDA: ok")

    print("All tests passed.")

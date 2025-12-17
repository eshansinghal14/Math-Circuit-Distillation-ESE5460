"""
CKA (Centered Kernel Alignment) Loss for Circuit Distillation

Implements the representational similarity metric from Section 2.1 of
"Circuit Distillation" (arxiv:2509.25002) and Kornblith et al. (2019).

CKA measures similarity between representations, invariant to orthogonal
transformations and isotropic scaling - ideal for comparing activations
from different-sized models.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple


def centering_matrix(m: int, device: torch.device = None) -> Tensor:
    """
    Compute the centering matrix H = I_m - (1/m) * 1_m * 1_m^T
    
    Args:
        m: Size of the matrix
        device: Torch device
        
    Returns:
        Centering matrix of shape (m, m)
    """
    I = torch.eye(m, device=device)
    ones = torch.ones(m, m, device=device) / m
    return I - ones


def hsic(K: Tensor, L: Tensor, centered: bool = True) -> Tensor:
    """
    Compute the Hilbert-Schmidt Independence Criterion (HSIC).
    
    HSIC(K, L) = (1/(m-1)^2) * tr(KHLH)
    
    where H is the centering matrix.
    
    Args:
        K: First Gram matrix of shape (m, m)
        L: Second Gram matrix of shape (m, m)
        centered: If True, apply centering. If matrices are pre-centered, set False.
        
    Returns:
        HSIC value (scalar tensor)
    """
    m = K.shape[0]
    
    if m <= 1:
        return torch.tensor(0.0, device=K.device, dtype=K.dtype)
    
    if centered:
        H = centering_matrix(m, device=K.device)
        KH = K @ H
        LH = L @ H
        # tr(KHLH) = tr((KH)(LH)) 
        hsic_val = torch.trace(KH @ LH)
    else:
        # If already centered, just compute trace of product
        hsic_val = torch.trace(K @ L)
    
    # Normalize by (m-1)^2
    hsic_val = hsic_val / ((m - 1) ** 2)
    
    return hsic_val


def linear_cka(X: Tensor, Y: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Compute Linear CKA between two activation matrices.
    
    CKA(X, Y) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
    
    where K = X @ X.T and L = Y @ Y.T are linear kernel Gram matrices.
    
    Args:
        X: Activation matrix from model 1, shape (m, p1) where m is batch size
        Y: Activation matrix from model 2, shape (m, p2)
        eps: Small value for numerical stability
        
    Returns:
        CKA similarity score in [0, 1]
    """
    assert X.shape[0] == Y.shape[0], f"Batch sizes must match: {X.shape[0]} vs {Y.shape[0]}"
    
    m = X.shape[0]
    
    if m <= 1:
        return torch.tensor(1.0, device=X.device, dtype=X.dtype)
    
    # Compute Gram matrices (linear kernel)
    K = X @ X.T  # (m, m)
    L = Y @ Y.T  # (m, m)
    
    # Compute HSIC values
    hsic_kl = hsic(K, L)
    hsic_kk = hsic(K, K)
    hsic_ll = hsic(L, L)
    
    # Normalize
    denominator = torch.sqrt(hsic_kk * hsic_ll + eps)
    cka = hsic_kl / denominator
    
    # Clamp to valid range (numerical issues can cause slight overflow)
    cka = torch.clamp(cka, 0.0, 1.0)
    
    return cka


def linear_cka_efficient(X: Tensor, Y: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Efficient computation of Linear CKA avoiding explicit Gram matrices.
    
    Uses the identity: CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    
    This is more memory-efficient for large batch sizes.
    
    Args:
        X: Activation matrix, shape (m, p1)
        Y: Activation matrix, shape (m, p2)
        eps: Small value for numerical stability
        
    Returns:
        CKA similarity score in [0, 1]
    """
    assert X.shape[0] == Y.shape[0], f"Batch sizes must match: {X.shape[0]} vs {Y.shape[0]}"
    
    # Center the matrices (subtract mean across samples)
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    # Compute cross-covariance
    YtX = Y.T @ X  # (p2, p1)
    XtX = X.T @ X  # (p1, p1)
    YtY = Y.T @ Y  # (p2, p2)
    
    # Frobenius norms
    numerator = (YtX ** 2).sum()
    denominator = torch.sqrt((XtX ** 2).sum() * (YtY ** 2).sum() + eps)
    
    cka = numerator / denominator
    
    return torch.clamp(cka, 0.0, 1.0)


class CKALoss(nn.Module):
    """
    CKA-based loss for circuit distillation.
    
    L_CKA = 1 - CKA(X_student, X_teacher)
    
    Minimizing this loss maximizes CKA, pushing representations toward alignment.
    """
    
    def __init__(self, efficient: bool = True, eps: float = 1e-8):
        """
        Args:
            efficient: Use memory-efficient computation
            eps: Numerical stability constant
        """
        super().__init__()
        self.efficient = efficient
        self.eps = eps
    
    def forward(
        self, 
        student_activations: Tensor, 
        teacher_activations: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute CKA loss between student and teacher activations.
        
        Args:
            student_activations: Shape (batch, seq_len, hidden) or (batch, hidden)
            teacher_activations: Shape (batch, seq_len, hidden) or (batch, hidden)
            
        Returns:
            Tuple of (loss, cka_score) where loss = 1 - cka_score
        """
        # Flatten if 3D (batch, seq, hidden) -> (batch * seq, hidden)
        if student_activations.dim() == 3:
            b, s, h = student_activations.shape
            student_activations = student_activations.reshape(b * s, h)
        
        if teacher_activations.dim() == 3:
            b, s, h = teacher_activations.shape
            teacher_activations = teacher_activations.reshape(b * s, h)
        
        # Compute CKA
        if self.efficient:
            cka = linear_cka_efficient(
                student_activations, 
                teacher_activations, 
                eps=self.eps
            )
        else:
            cka = linear_cka(
                student_activations, 
                teacher_activations, 
                eps=self.eps
            )
        
        loss = 1.0 - cka
        
        return loss, cka


class MultiHeadCKALoss(nn.Module):
    """
    CKA loss aggregated over multiple paired circuit heads.
    
    L_total = (1/|C_paired|) * Σ L_CKA(h_s^(c), h_t^(c))
    
    where C_paired is the set of paired student-teacher heads.
    """
    
    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        """
        Args:
            reduction: 'mean' or 'sum' over paired heads
            eps: Numerical stability constant
        """
        super().__init__()
        self.cka_loss = CKALoss(efficient=True, eps=eps)
        self.reduction = reduction
    
    def forward(
        self,
        student_activations: dict,  # {head_idx: Tensor}
        teacher_activations: dict,  # {head_idx: Tensor}
        head_mapping: dict  # {student_idx: teacher_idx}
    ) -> Tuple[Tensor, dict]:
        """
        Compute aggregated CKA loss over all paired heads.
        
        Args:
            student_activations: Dict mapping student head index to activations
            teacher_activations: Dict mapping teacher head index to activations
            head_mapping: Dict mapping student head indices to teacher head indices
            
        Returns:
            Tuple of (total_loss, per_head_cka_scores)
        """
        losses = []
        cka_scores = {}
        
        for s_idx, t_idx in head_mapping.items():
            if s_idx not in student_activations:
                continue
            if t_idx not in teacher_activations:
                continue
            
            s_act = student_activations[s_idx]
            t_act = teacher_activations[t_idx]
            
            loss, cka = self.cka_loss(s_act, t_act)
            losses.append(loss)
            cka_scores[(s_idx, t_idx)] = cka.item()
        
        if not losses:
            device = next(iter(student_activations.values())).device
            return torch.tensor(0.0, device=device), {}
        
        losses = torch.stack(losses)
        
        if self.reduction == 'mean':
            total_loss = losses.mean()
        elif self.reduction == 'sum':
            total_loss = losses.sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
        
        return total_loss, cka_scores


# Tests
if __name__ == '__main__':
    print("Testing CKA implementation...")
    
    # Test 1: CKA of identical matrices should be 1
    X = torch.randn(32, 128)
    cka = linear_cka(X, X)
    print(f"CKA(X, X) = {cka.item():.6f} (should be ~1.0)")
    assert abs(cka.item() - 1.0) < 0.01, "CKA(X, X) should be 1"
    
    # Test 2: CKA should be in [0, 1]
    Y = torch.randn(32, 64)
    cka = linear_cka(X, Y)
    print(f"CKA(X, Y) = {cka.item():.6f} (random matrices)")
    assert 0.0 <= cka.item() <= 1.0, "CKA should be in [0, 1]"
    
    # Test 3: Efficient vs standard should give same result
    cka_std = linear_cka(X, Y)
    cka_eff = linear_cka_efficient(X, Y)
    print(f"Standard CKA: {cka_std.item():.6f}, Efficient CKA: {cka_eff.item():.6f}")
    
    # Test 4: CKALoss gradient flow
    loss_fn = CKALoss()
    X = torch.randn(32, 128, requires_grad=True)
    Y = torch.randn(32, 64)
    loss, cka = loss_fn(X, Y)
    loss.backward()
    print(f"Loss = {loss.item():.6f}, CKA = {cka.item():.6f}")
    print(f"Gradient norm: {X.grad.norm().item():.6f}")
    assert X.grad is not None, "Gradients should flow through CKA loss"
    
    # Test 5: 3D activations (batch, seq, hidden)
    X_3d = torch.randn(8, 16, 128)
    Y_3d = torch.randn(8, 16, 64)
    loss, cka = loss_fn(X_3d, Y_3d)
    print(f"3D input - Loss = {loss.item():.6f}, CKA = {cka.item():.6f}")
    
    print("\nAll tests passed!")

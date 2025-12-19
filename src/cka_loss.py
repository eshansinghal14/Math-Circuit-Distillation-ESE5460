import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple


def centering_matrix(m: int, device: torch.device = None) -> Tensor:
    I = torch.eye(m, device=device)
    ones = torch.ones(m, m, device=device) / m
    return I - ones


def hsic(K: Tensor, L: Tensor, centered: bool = True) -> Tensor:
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
        hsic_val = torch.trace(K @ L)
    
    hsic_val = hsic_val / ((m - 1) ** 2)
    
    return hsic_val


def linear_cka(X: Tensor, Y: Tensor, eps: float = 1e-8) -> Tensor:
    assert X.shape[0] == Y.shape[0], f"Batch sizes must match: {X.shape[0]} vs {Y.shape[0]}"
    
    m = X.shape[0]
    
    if m <= 1:
        return torch.tensor(1.0, device=X.device, dtype=X.dtype)
    
    K = X @ X.T 
    L = Y @ Y.T 
    
    hsic_kl = hsic(K, L)
    hsic_kk = hsic(K, K)
    hsic_ll = hsic(L, L)
    
    denominator = torch.sqrt(hsic_kk * hsic_ll + eps)
    cka = hsic_kl / denominator
    
    cka = torch.clamp(cka, 0.0, 1.0)
    
    return cka


def linear_cka_efficient(X: Tensor, Y: Tensor, eps: float = 1e-8) -> Tensor:
    assert X.shape[0] == Y.shape[0], f"Batch sizes must match: {X.shape[0]} vs {Y.shape[0]}"
    
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    YtX = Y.T @ X 
    XtX = X.T @ X 
    YtY = Y.T @ Y 
    
    numerator = (YtX ** 2).sum()
    denominator = torch.sqrt((XtX ** 2).sum() * (YtY ** 2).sum() + eps)
    
    cka = numerator / denominator
    
    return torch.clamp(cka, 0.0, 1.0)


class CKALoss(nn.Module):
    
    def __init__(self, efficient: bool = True, eps: float = 1e-8):
        super().__init__()
        self.efficient = efficient
        self.eps = eps
    
    def forward(
        self, 
        student_activations: Tensor, 
        teacher_activations: Tensor
    ) -> Tuple[Tensor, Tensor]:

        if student_activations.dim() == 3:
            b, s, h = student_activations.shape
            student_activations = student_activations.reshape(b * s, h)
        
        if teacher_activations.dim() == 3:
            b, s, h = teacher_activations.shape
            teacher_activations = teacher_activations.reshape(b * s, h)
        
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
    
    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()
        self.cka_loss = CKALoss(efficient=True, eps=eps)
        self.reduction = reduction
    
    def forward(
        self,
        student_activations: dict,  
        teacher_activations: dict,  
        head_mapping: dict 
    ) -> Tuple[Tensor, dict]:
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


if __name__ == '__main__':
    print("Testing CKA implementation...")
    
    X = torch.randn(32, 128)
    cka = linear_cka(X, X)
    print(f"CKA(X, X) = {cka.item():.6f} (should be ~1.0)")
    assert abs(cka.item() - 1.0) < 0.01, "CKA(X, X) should be 1"
    
    Y = torch.randn(32, 64)
    cka = linear_cka(X, Y)
    print(f"CKA(X, Y) = {cka.item():.6f} (random matrices)")
    assert 0.0 <= cka.item() <= 1.0, "CKA should be in [0, 1]"
    
    cka_std = linear_cka(X, Y)
    cka_eff = linear_cka_efficient(X, Y)
    print(f"Standard CKA: {cka_std.item():.6f}, Efficient CKA: {cka_eff.item():.6f}")
    
    loss_fn = CKALoss()
    X = torch.randn(32, 128, requires_grad=True)
    Y = torch.randn(32, 64)
    loss, cka = loss_fn(X, Y)
    loss.backward()
    print(f"Loss = {loss.item():.6f}, CKA = {cka.item():.6f}")
    print(f"Gradient norm: {X.grad.norm().item():.6f}")
    assert X.grad is not None, "Gradients should flow through CKA loss"
    
    X_3d = torch.randn(8, 16, 128)
    Y_3d = torch.randn(8, 16, 64)
    loss, cka = loss_fn(X_3d, Y_3d)
    print(f"3D input - Loss = {loss.item():.6f}, CKA = {cka.item():.6f}")
    
    print("\nAll tests passed!")

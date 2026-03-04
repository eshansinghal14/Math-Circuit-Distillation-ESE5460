"""
Automated loss weight scheduler for circuit discovery.

Schedule:
- Within-class similarity (sim) is always minimized (fixed weight).
- Phase 1 (early): Emphasize class usage entropy and mask orthogonality so classes
  are used evenly and masks become orthogonal.
- Transition: When mask orthogonality is good (low pairwise mask cosine sim) and
  class usage is roughly equal (high class_usage_entropy), switch to Phase 2.
- Phase 2: Emphasize KL (toward 10% activation) and mask sparsity to achieve
  ~10% proportion of activated neurons.

Weights in each phase sum to 1 for easier interpretation.
"""

import math
from typing import Dict, Optional

# Minimum epochs before allowing transition to phase 2
MIN_EPOCHS_PHASE1 = 10


class LossScheduler:
    """
    Updates loss weights by phase. All weights in a phase sum to 1.
    - Phase 1: High weight on class_usage_entropy and mask_orthogonality (mask_cossim).
    - Phase 2: After transition conditions are met, higher weight on KL and sparsity.
    """

    # Phase 1: (sim, usage, mask_cossim, kl, sparsity) — emphasize usage + orthogonality
    _PHASE1_WEIGHTS = (0.15, 0.30, 0.40, 0.05, 0.10)
    # Phase 2: emphasize KL + sparsity for ~10% activation
    _PHASE2_WEIGHTS = (0.20, 0.10, 0.15, 0.30, 0.25)

    def __init__(
        self,
        k_classes: int,
        orthogonality_threshold: float = 0.35,
        balance_ratio: float = 0.85,
    ):
        self.k_classes = k_classes
        self.orthogonality_threshold = orthogonality_threshold
        self.balance_ratio = balance_ratio
        self._max_class_usage_entropy = math.log(k_classes) if k_classes > 0 else 0.0
        self._phase = 0  # 0 = phase 1, 1 = phase 2

    def _transition_conditions_met(self, epoch: int, metrics: Dict[str, float]) -> bool:
        if epoch < MIN_EPOCHS_PHASE1:
            return False
        mask_cossim_1b = metrics.get("mask_cossim_1b_loss", 1.0)
        mask_cossim_8b = metrics.get("mask_cossim_8b_loss", 1.0)
        mean_mask_cossim = (mask_cossim_1b + mask_cossim_8b) / 2.0
        class_usage_entropy = metrics.get("class_usage_entropy", 0.0)
        balance_threshold = self.balance_ratio * self._max_class_usage_entropy
        orthogonality_ok = mean_mask_cossim < self.orthogonality_threshold
        usage_balanced = class_usage_entropy >= balance_threshold
        return orthogonality_ok and usage_balanced

    def get_lambdas(self, epoch: int, metrics: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Compute current loss weights (sum to 1). Phase 1 until transition conditions met.
        """
        metrics = metrics or {}
        if self._phase == 0 and self._transition_conditions_met(epoch, metrics):
            self._phase = 1

        w = self._PHASE2_WEIGHTS if self._phase == 1 else self._PHASE1_WEIGHTS
        return {
            "lambda_sim": w[0],
            "lambda_usage": w[1],
            "lambda_mask_cossim": w[2],
            "lambda_kl": w[3],
            "lambda_sparsity": w[4],
        }

    @property
    def phase(self) -> int:
        return self._phase

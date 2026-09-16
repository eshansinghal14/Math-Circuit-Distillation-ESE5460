"""KL + linear-CKA representation distillation.

The representation term is ``1 - CKA`` between the student's and the teacher's
residual streams at paired layers, averaged over the pairs, where CKA is linear
centred kernel alignment (Kornblith et al., 2019) over the matched tokens of the
batch. No projection is learned: CKA is invariant to isotropic scaling and to
any orthogonal rotation of either representation, so the term asks the student
to reproduce the teacher's similarity structure between tokens, not its
coordinates. That makes it the geometry-only counterpart to ``bert_kd``'s
coordinate matching and a second point of comparison for the graph loss.

The batch estimate of CKA is the plain (biased) one computed on the tokens in
the batch; with a few hundred matched tokens per batch it is a stable training
signal. See ``training/representation.py`` for the shared machinery.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from utils import DIR_ROOT, load_data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.representation import (
    ActivationCapture,
    RepKDConfig,
    RepKDTrainer,
    add_rep_args,
    base_config_kwargs,
    linear_cka,
)
from training.utils import add_kd_args, add_standard_args, run_seeds


@dataclass
class CKAKDConfig(RepKDConfig):
    pass


class CKAKDTrainer(RepKDTrainer):
    REP_NAME = "1-CKA"
    RUN_NAME = "CKA-KD"

    def _rep_loss(
        self,
        s_cap: ActivationCapture,
        t_cap: ActivationCapture,
        token_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        n_tokens = int(token_mask.sum().item())
        if n_tokens < 2:
            zero = next(iter(s_cap.hidden.values())).sum() * 0.0
            return zero, {"cka": float("nan")}
        values = []
        for s, t in self.layer_pairs:
            x = s_cap.hidden[s][token_mask]
            y = t_cap.hidden[t][token_mask]
            values.append(linear_cka(x, y))
        cka = torch.stack(values).mean()
        return 1.0 - cka, {"cka": float(cka.detach())}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="KL + linear-CKA representation distillation on local datasets."
    )
    add_standard_args(parser)
    add_kd_args(parser)
    add_rep_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_data, test_data = load_data(args.dataset, test_limit=args.test_limit)
    print(f"Train: {len(train_data)} | Test: {len(test_data)}")

    def build(seed: int):
        return CKAKDTrainer(
            CKAKDConfig(**base_config_kwargs(args, DIR_ROOT, seed=seed)),
            train_data,
            test_data,
        )

    run_seeds(args.seeds, args.resume, build)


if __name__ == "__main__":
    main()

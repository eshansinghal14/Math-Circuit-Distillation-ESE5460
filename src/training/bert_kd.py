"""KL + TinyBERT-style representation distillation (hidden states and attention patterns).

The representation term follows Jiao et al. (2020), "TinyBERT":

* hidden states -- MSE between a learned affine projection of the student's
  residual stream and the teacher's, at paired layers (their ``L_hidn`` with
  ``W_h``). The projection is initialised by ridge least squares on a few
  calibration batches rather than at random: at lr 1e-6 a random projection
  would never train, and the MSE against it would be noise. It then trains at
  its own learning rate (``--proj-lr``).
* attention -- MSE between attention patterns at the same layers (their
  ``L_attn``). Two departures from the paper, both forced by distilling between
  two independently pretrained models: patterns are compared after the softmax
  rather than as raw scores, since the score scale of the two models is
  unrelated, and by default the head-averaged pattern of each layer is compared
  (``--attn-match mean``) because the heads have no correspondence. TinyBERT's
  head-to-head matching is available as ``--attn-match head``.

Both parts are averaged over the paired layers, so ``--lambda-rep`` means the
same thing whether four layers or sixteen are matched. See
``training/representation.py`` for the shared machinery.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from utils import DIR_ROOT, load_data, seed_all, set_bos_mode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.representation import (
    _DEVICE,
    ActivationCapture,
    HiddenProjector,
    LstsqStats,
    RepKDConfig,
    RepKDTrainer,
    add_rep_args,
    attention_pattern_loss,
    base_config_kwargs,
    fit_projector_least_squares,
    hidden_state_loss,
    matched_token_mask,
)
from training.utils import add_kd_args, add_standard_args, run_seeds, sweep_apply


@dataclass
class BertKDConfig(RepKDConfig):
    hidden_loss: str = "mse"
    hidden_weight: float = 1.0
    attn_weight: float = 1.0
    attn_match: str = "mean"
    proj_init: str = "lstsq"
    proj_lr: float = 1e-4
    proj_init_batches: int = 8
    proj_ridge: float = 1e-2


class BertKDTrainer(RepKDTrainer):
    REP_NAME = "Rep"
    RUN_NAME = "BERT-KD"

    def _setup_representation(self) -> None:
        cfg: BertKDConfig = self.config  # type: ignore[assignment]
        if cfg.proj_init not in ("lstsq", "random"):
            raise ValueError(f"unknown --proj-init {cfg.proj_init!r}")
        d_student = int(self.model.config.hidden_size)
        d_teacher = int(self.teacher.config.hidden_size)
        self.projector = HiddenProjector(self.student_layers, d_student, d_teacher).to(_DEVICE)
        if cfg.proj_init == "lstsq" and not cfg.resume:
            self._fit_projector()
        self.optimizer.add_param_group({"params": list(self.projector.parameters()), "lr": cfg.proj_lr})
        # The projector's random init and the calibration batches consumed the
        # global RNG; re-seed so the training batch order matches the other
        # trainers, which draw their first batch straight after seed_all.
        seed_all(cfg.seed)

    def _needs_attention(self) -> bool:
        return self.config.attn_weight > 0  # type: ignore[attr-defined]

    def _extra_module(self) -> Optional[nn.Module]:
        return self.projector

    def _banner_extra(self) -> str:
        cfg: BertKDConfig = self.config  # type: ignore[assignment]
        return (
            f" | hidden={cfg.hidden_loss}x{cfg.hidden_weight} | attn={cfg.attn_match}x{cfg.attn_weight}"
            f" | proj_init={cfg.proj_init} proj_lr={cfg.proj_lr}"
        )

    @torch.no_grad()
    def _fit_projector(self) -> None:
        """Ridge least-squares fit of every projector on the first calibration batches."""
        cfg: BertKDConfig = self.config  # type: ignore[assignment]
        d_student = int(self.model.config.hidden_size)
        d_teacher = int(self.teacher.config.hidden_size)
        stats = {j: LstsqStats(d_student, d_teacher, _DEVICE) for j in self.student_layers}
        n_batches = 0
        n_tokens = 0
        last: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for batch in self.loader:
            if n_batches >= cfg.proj_init_batches:
                break
            input_ids = batch["input_ids"].to(_DEVICE)
            attention_mask = batch["attention_mask"].to(_DEVICE)
            response_mask = batch["response_mask"].to(_DEVICE)
            _, _, s_cap, t_cap = self._forward_pair(input_ids, attention_mask, want_attention=False)
            token_mask = matched_token_mask(
                attention_mask, response_mask, cfg.match_positions, cfg.keep_first_position,
            )
            for s, t in self.layer_pairs:
                x = s_cap.hidden[s][token_mask]
                y = t_cap.hidden[t][token_mask]
                stats[s].update(x, y)
                last[s] = (x.float(), y.float())
            n_tokens += int(token_mask.sum().item())
            n_batches += 1
        if n_batches == 0:
            raise RuntimeError("no calibration batches for the projector init; is the dataset empty?")
        fit_projector_least_squares(self.projector, stats, cfg.proj_ridge)

        # Fit quality on the last calibration batch: R^2 of the projected student
        # against the teacher, per layer. Near 0 means the student's residual
        # stream at that layer carries little of the teacher's; that is the
        # starting point the MSE term will pull from.
        r2 = []
        for s, _ in self.layer_pairs:
            x, y = last[s]
            resid = ((self.projector(s, x) - y) ** 2).mean()
            var = y.var(dim=0, unbiased=False).mean().clamp_min(1e-12)
            r2.append(float(1.0 - resid / var))
        r2_str = " ".join(f"{s}:{v:.3f}" for (s, _), v in zip(self.layer_pairs, r2))
        print(f"  projector lstsq init on {n_tokens} tokens from {n_batches} batches | R^2 by student layer: {r2_str}")

    def _rep_loss(
        self,
        s_cap: ActivationCapture,
        t_cap: ActivationCapture,
        token_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg: BertKDConfig = self.config  # type: ignore[assignment]
        hidden_terms = []
        attn_terms = []
        for s, t in self.layer_pairs:
            x = s_cap.hidden[s][token_mask]
            y = t_cap.hidden[t][token_mask]
            hidden_terms.append(hidden_state_loss(self.projector(s, x), y, cfg.hidden_loss))
            if cfg.attn_weight > 0:
                attn_terms.append(
                    attention_pattern_loss(s_cap.attention[s], t_cap.attention[t], token_mask, cfg.attn_match)
                )
        hidden = torch.stack(hidden_terms).mean()
        attn = torch.stack(attn_terms).mean() if attn_terms else hidden.new_zeros(())
        rep = cfg.hidden_weight * hidden + cfg.attn_weight * attn
        return rep, {"hidden_loss": float(hidden.detach()), "attn_loss": float(attn.detach())}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="KL + TinyBERT-style hidden-state and attention distillation on local datasets."
    )
    add_standard_args(parser)
    add_kd_args(parser)
    add_rep_args(parser)
    group = parser.add_argument_group("bert_args")
    group.add_argument("--hidden-loss", "--hidden_loss", choices=["mse", "cosine"], default="mse",
                       dest="hidden_loss",
                       help="MSE (TinyBERT) or 1 - cosine (DistilBERT-style) between projected "
                            "student and teacher residual streams.")
    group.add_argument("--hidden-weight", "--hidden_weight", type=float, default=1.0, dest="hidden_weight")
    group.add_argument("--attn-weight", "--attn_weight", type=float, default=1.0, dest="attn_weight",
                       help="Weight of the attention-pattern term inside the representation "
                            "term; 0 disables it and skips the eager-attention forward.")
    group.add_argument("--attn-match", "--attn_match", choices=["mean", "head"], default="mean",
                       dest="attn_match",
                       help="Compare the head-averaged pattern per layer, or head i to head i "
                            "(needs equal head counts).")
    group.add_argument("--proj-init", "--proj_init", choices=["lstsq", "random"], default="lstsq",
                       dest="proj_init",
                       help="Initialise the hidden-state projectors by ridge least squares on "
                            "calibration batches, or at random as in TinyBERT.")
    group.add_argument("--proj-lr", "--proj_lr", type=float, default=1e-4, dest="proj_lr",
                       help="Learning rate of the projectors (their own AdamW param group).")
    group.add_argument("--proj-init-batches", "--proj_init_batches", type=int, default=8,
                       dest="proj_init_batches")
    group.add_argument("--proj-ridge", "--proj_ridge", type=float, default=1e-2, dest="proj_ridge",
                       help="Ridge strength relative to the mean feature variance in the init fit.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # Before any dataset, tokenizer or adapter is built: the training sequence,
    # eval and attribution paths must all see the same setting.
    train_data, test_data = load_data(args.dataset, test_limit=args.test_limit)
    print(f"Train: {len(train_data)} | Test: {len(test_data)}")

    def build(seed: int, shared: Dict[str, Any]):
        return BertKDTrainer(
            BertKDConfig(
                **base_config_kwargs(args, DIR_ROOT, seed=seed),
                hidden_loss=args.hidden_loss,
                hidden_weight=args.hidden_weight,
                attn_weight=args.attn_weight,
                attn_match=args.attn_match,
                proj_init=args.proj_init,
                proj_lr=args.proj_lr,
                proj_init_batches=args.proj_init_batches,
                proj_ridge=args.proj_ridge,
            ),
            train_data,
            test_data,
            shared=shared,
        )

    base_dir = os.path.join(DIR_ROOT, args.save_dir)
    for point in sweep_apply(args):
        set_bos_mode(args.bos_mode)
        if point:
            print()
            print("=== sweep point: " + point + " ===")
        run_seeds(args.seeds, args.resume, build, save_dir=base_dir,
                  steps=args.steps, redo=args.redo_seeds)


if __name__ == "__main__":
    main()

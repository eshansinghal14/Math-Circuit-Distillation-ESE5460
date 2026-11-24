"""KD plus the swap graph (training/swap_graph.py) as an auxiliary term.

Each step, every sequence in the batch gets one sampled edge of the graph: one
of its prompt numbers, a replacement for it, and a relative depth. The teacher
and the student are each run with that swap, and the student is trained to
answer the swap as the teacher does (KL on the response positions), on top of
the usual KD term on the unswapped batch.

    --swap-weight W     weight of the graph term (0 = KD alone, same batches)
    --swap-depths LIST  comma-separated relative depths the edges are drawn from.
                        "0" alone is the data-augmentation baseline: KD on
                        number-changed copies of the prompts, no internal swap.

The edge sampler has its own generator, so runs with the same seed see the same
batches in the same order whatever the graph settings.

    cd src && python -u -m training.swap_graph_kd --model <student> --teacher <teacher> \\
        --dataset svamp_eq --steps 50 --batch-size 32 --lr 1e-5 --warmup-steps 10 --lr-floor 0.1 \\
        --kl-tokens resp --dtype bfloat16 --bos-mode on --seeds 1 2 3 \\
        --swap-weight 1 --swap-depths 0,0.25,0.5,0.75
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DIR_ROOT, load_data, set_bos_mode  # noqa: E402
from training.standard_kd import StandardKDConfig, StandardKDTrainer, build_parser as kd_parser  # noqa: E402
from training.swap_graph import (  # noqa: E402
    NODE_KINDS, SCORED, NumberTokens, consistent_positions, pick_edges, prompt_end_effect, swap_forward)
from training.utils import SWEEP_PARAMS, kl_loss, run_seeds, sweep_apply  # noqa: E402


@dataclass
class SwapGraphKDConfig(StandardKDConfig):
    swap_weight: float = 1.0
    swap_depths: str = "0,0.25,0.5,0.75"
    swap_nodes: str = "number+rest"
    swap_score: str = "all"


class SwapGraphKDTrainer(StandardKDTrainer):
    def __init__(self, config: SwapGraphKDConfig, train_data, test_data, shared=None) -> None:
        super().__init__(config, train_data, test_data, shared)
        self._depths = [float(x) for x in str(config.swap_depths).split(",")]
        if not self._depths or any(not 0.0 <= d < 1.0 for d in self._depths):
            raise ValueError(f"--swap-depths must be relative depths in [0, 1), got {config.swap_depths!r}")
        if config.swap_weight < 0:
            raise ValueError("--swap-weight must be >= 0")
        if config.swap_score not in SCORED:
            raise ValueError(f"--swap-score must be one of {SCORED}, got {config.swap_score!r}")
        if config.swap_nodes not in NODE_KINDS:
            raise ValueError(f"--swap-nodes must be one of {NODE_KINDS}, got {config.swap_nodes!r}")
        self._numbers = NumberTokens(self.tokenizer)
        self._edge_rng = torch.Generator().manual_seed(1_000_003 * int(config.seed) + 17)
        self._suffix = ""

    def _auxiliary_loss(self, batch, input_ids, attention_mask, kd_mask, student_logits, teacher_logits):
        cfg = self.config
        self._suffix = ""
        if cfg.swap_weight <= 0:
            return None
        first = batch["response_mask"].argmax(dim=1)
        edges = pick_edges(input_ids, first, self._numbers, self._depths, self._edge_rng,
                           cfg.swap_nodes).to(input_ids.device)
        if not bool(edges.has.any()):
            return None
        with torch.no_grad():
            t_swap = swap_forward(self.teacher, input_ids, attention_mask, edges, self._depths)
        with self._autocast():
            s_swap = swap_forward(self.model, input_ids, attention_mask, edges, self._depths)
        scored = kd_mask & edges.has[:, None]
        if cfg.swap_score == "consistent":
            scored = consistent_positions(input_ids, batch["response_mask"].to(input_ids.device), scored, edges)
        graph = kl_loss(s_swap, t_swap, scored, cfg.temperature, cfg.kl_token_chunk_size)

        # Edge weights for the log: how far each model's first response token moves under the swap.
        first = first.to(input_ids.device)
        with torch.no_grad():
            eff_t = prompt_end_effect(teacher_logits, t_swap, first)
            eff_s = prompt_end_effect(student_logits.detach(), s_swap.detach(), first)
        cells = []
        for kind, picked in (("number", edges.number), ("rest", edges.rest)):
            if kind == "rest" and cfg.swap_nodes == "number":
                continue
            for b, depth in enumerate(self._depths):
                if kind == "rest" and depth == 0:
                    continue
                sel = edges.has & picked & (edges.band == b)
                t = float(eff_t[sel].mean()) if bool(sel.any()) else float("nan")
                s = float(eff_s[sel].mean()) if bool(sel.any()) else float("nan")
                self.history[f"step_edge_teacher_{kind}_d{depth:g}"].append(t)
                self.history[f"step_edge_student_{kind}_d{depth:g}"].append(s)
                cells.append(f"{kind[0]}{depth:g}:{t:.2f}/{s:.2f}")
        value = float(graph.detach())
        self.history["step_graph_loss"].append(value)
        self._suffix = f" | graph={value:.4f} | edge T/S " + " ".join(cells)
        return cfg.swap_weight * graph

    def _step_suffix(self) -> str:
        return self._suffix


def build_parser():
    parser = kd_parser()
    parser.description = "KD plus the swap graph: edges measured by interchange, compared in output units."
    parser.add_argument("--swap-weight", type=float, nargs="+", default=[1.0], dest="swap_weight",
                        help="Weight of the swap-graph term; 0 is KD alone on the same batches.")
    parser.add_argument("--swap-depths", type=str, nargs="+", default=["0,0.25,0.5,0.75"], dest="swap_depths",
                        help="Comma-separated relative depths the edges are drawn from (0 = the token embedding, "
                             "i.e. the number-changed prompt itself). Several values sweep.")
    parser.add_argument("--swap-nodes", type=str, nargs="+", default=["number+rest"], choices=list(NODE_KINDS),
                        dest="swap_nodes",
                        help="'number': only the state at the number's own position. 'number+rest' (default): "
                             "also the states of every later position, so the two nodes at a depth carry the "
                             "number's whole influence.")
    parser.add_argument("--swap-score", type=str, nargs="+", default=["all"], choices=list(SCORED), dest="swap_score",
                        help="'all': every response position. 'consistent': only the response positions whose "
                             "teacher-forced context has not yet spelled out the original number, i.e. up to the "
                             "position that predicts its first appearance.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_data, test_data = load_data(args.dataset, test_limit=args.test_limit)
    print(f"Train: {len(train_data)} | Test: {len(test_data)}")
    save_dir = os.path.join(DIR_ROOT, args.save_dir)

    def build(seed: int, shared: Dict[str, Any]):
        return SwapGraphKDTrainer(
            SwapGraphKDConfig(
                model=args.model, teacher=args.teacher, dataset=args.dataset, steps=args.steps,
                batch_size=args.batch_size, learning_rate=args.lr, warmup_steps=args.warmup_steps,
                lr_floor=args.lr_floor, temperature=args.temperature,
                kl_token_chunk_size=args.kl_token_chunk_size, kl_tokens=args.kl_tokens, save_dir=save_dir,
                eval_every_n_steps=args.eval_every_n_steps, save_every_n_steps=args.save_every_n_steps,
                grad_accum_steps=args.grad_accum_steps, max_eval_tokens=args.max_eval_tokens,
                eval_batch_size=args.eval_batch_size, eval_datasets=args.eval_datasets,
                test_limit=args.test_limit, dtype=args.dtype, bos_mode=args.bos_mode, resume=args.resume,
                track_flops=args.track_flops, kd_target=args.kd_target, seed=seed,
                swap_weight=args.swap_weight, swap_depths=args.swap_depths, swap_nodes=args.swap_nodes,
                swap_score=args.swap_score,
            ),
            train_data, test_data, shared=shared,
        )

    params = tuple(SWEEP_PARAMS) + ("swap_weight", "swap_depths", "swap_nodes", "swap_score")
    for point in sweep_apply(args, params):
        set_bos_mode(args.bos_mode)
        if point:
            print()
            print("=== sweep point: " + point + " ===")
        print(f"swap graph | weight {args.swap_weight} | depths {args.swap_depths} | nodes {args.swap_nodes} | scored {args.swap_score}")
        run_seeds(args.seeds, args.resume, build, save_dir=save_dir, steps=args.steps, redo=args.redo_seeds)


if __name__ == "__main__":
    main()

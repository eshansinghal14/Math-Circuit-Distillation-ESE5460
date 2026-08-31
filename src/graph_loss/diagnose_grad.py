"""Compare the frozen and unfrozen graph gradients at a single fixed checkpoint.

Training curves conflate direction and magnitude: a signal that points the same
way but is half as strong looks, over 100 steps at a fixed LR, much like a signal
that points somewhere slightly different.  This separates them.

From one checkpoint and one batch it computes three gradients -- the KL term
alone, the graph term with the freeze off, and the graph term with the freeze on
-- and reports their norms and pairwise cosines, globally and per parameter
group.

How to read it:

cos(frozen, unfrozen) near 1, ||frozen|| < ||unfrozen||
    The freeze is costing you *magnitude*, not direction: the direct-path target
    is a weaker version of the same signal.  Then raising --lambda-graph for the
    frozen run should recover part of the gap and pull its accuracy peak earlier,
    and that is a cheap prediction to test.

cos(frozen, unfrozen) near 0
    The two targets carry different information, and the direct-path graph is
    genuinely missing the routing structure rather than merely attenuating it.
    Raising lambda would then make the frozen run worse, not better.

cos(graph, KL)
    How much of the graph term is already implied by the KD term.  Near zero
    while still improving accuracy is the interesting case: the graph loss is
    contributing information KD does not have, rather than acting as a
    learning-rate multiplier on it.

Per-group rows matter because the freeze zeroes different paths in different
places.  Since the scoping fix, q_proj/k_proj should receive graph gradient in
*both* modes -- a q/k row that is zero under freeze but nonzero unfrozen means
the freeze is still leaking into the parameter path somewhere.

Memory: three CPU fp32 copies of the student's gradients (~15 GB for a 1B
student).  Reduce --n-prompts before reducing anything else; it does not change
what is measured, only the batch the gradients are estimated from.

Usage:
    python -m graph_loss.diagnose_grad \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --teacher meta-llama/Meta-Llama-3-8B-Instruct \
        --dataset 22_add --batch-size 32 --n-prompts 8 --nodes-per-label 3
"""

from __future__ import annotations

import argparse
import logging
from functools import partial

import torch

from graph_loss.hf_adapter import HFLlamaGraphAdapter
from graph_loss.training import GraphAuxConfig, backward_batch_graph_loss
from training.utils import kl_loss
from utils import (
    DataLoader,
    PromptAnswerDataset,
    collate_fn,
    load_data,
    load_model,
    seed_all,
)

# Ordered longest-prefix-first so post_attention_layernorm is not eaten by "norm".
_GROUPS = [
    ("q_proj", "q_proj"),
    ("k_proj", "k_proj"),
    ("v_proj", "v_proj"),
    ("o_proj", "o_proj"),
    ("gate_proj", "gate_proj"),
    ("up_proj", "up_proj"),
    ("down_proj", "down_proj"),
    ("layernorm", "layernorm"),
    ("norm", "norm"),  # model.norm, the final RMSNorm; after layernorm so it does not swallow it
    ("embed", "embed_tokens"),
    ("lm_head", "lm_head"),
]


def _group_of(name: str) -> str:
    for label, needle in _GROUPS:
        if needle in name:
            return label
    return "other"


def _snapshot(model) -> dict[str, torch.Tensor]:
    """Detached CPU fp32 copy of the current .grad of every trainable parameter."""
    return {
        n: p.grad.detach().to("cpu", torch.float32).clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }


def _norm(g: dict[str, torch.Tensor], names=None) -> float:
    keys = g.keys() if names is None else [k for k in names if k in g]
    return float(torch.sqrt(sum((g[k] ** 2).sum() for k in keys)).item()) if keys else 0.0


def _cosine(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], names=None) -> float:
    keys = set(a) & set(b)
    if names is not None:
        keys &= set(names)
    if not keys:
        return float("nan")
    dot = sum((a[k] * b[k]).sum() for k in keys)
    na = torch.sqrt(sum((a[k] ** 2).sum() for k in keys))
    nb = torch.sqrt(sum((b[k] ** 2).sum() for k in keys))
    denom = (na * nb).item()
    return float(dot.item() / denom) if denom > 0 else float("nan")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    ap = argparse.ArgumentParser(description="Compare frozen vs unfrozen graph gradients.")
    ap.add_argument("--model", required=True, help="Student model.")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--batch-size", type=int, default=32, help="Batch for the KL term.")
    ap.add_argument("--n-prompts", type=int, default=8,
                    help="Prompts used for the graph term (training's --n-graph-prompts).")
    ap.add_argument("--nodes-per-label", type=int, default=3)
    ap.add_argument("--prop-neurons-per-layer", type=float, default=0.1)
    ap.add_argument("--top-k-logits", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--graph-loss-type", default="jsd")
    ap.add_argument("--kl-token-chunk-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)

    student, tokenizer = load_model(args.model)
    student.config.use_cache = False
    student.train()
    teacher, _ = load_model(args.teacher)
    teacher.eval()
    teacher.config.use_cache = False
    for p in teacher.parameters():
        p.requires_grad_(False)

    student_adapter = HFLlamaGraphAdapter(student, tokenizer, device)
    teacher_adapter = HFLlamaGraphAdapter(teacher, tokenizer, device)

    train_data, _ = load_data(args.dataset)
    dataset = PromptAnswerDataset(args.dataset, train_data, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_id=tokenizer.eos_token_id),
    )
    batch = next(iter(loader))
    prompts = batch["prompts"][: args.n_prompts]
    answers = batch["answers"][: args.n_prompts]

    def graph_config(freeze: bool) -> GraphAuxConfig:
        return GraphAuxConfig(
            lambda_graph=1.0,
            teacher_prop_neurons_per_layer=args.prop_neurons_per_layer,
            student_prop_neurons_per_layer=args.prop_neurons_per_layer,
            top_k_logits=args.top_k_logits,
            temperature=args.temperature,
            student_nodes_per_label=args.nodes_per_label,
            teacher_nodes_per_label=args.nodes_per_label,
            graph_loss_type=args.graph_loss_type,
            freeze_attention=freeze,
            freeze_rms_norm=freeze,
            dataset_name=args.dataset,
        )

    grads: dict[str, dict[str, torch.Tensor]] = {}

    # ---- KL term alone -----------------------------------------------------
    student.zero_grad(set_to_none=True)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    s_logits = student(input_ids, attention_mask=attention_mask).logits
    with torch.no_grad():
        t_logits = teacher(input_ids, attention_mask=attention_mask).logits
    kl = kl_loss(s_logits, t_logits, attention_mask, args.temperature, args.kl_token_chunk_size)
    kl.backward()
    grads["KL"] = _snapshot(student)
    del s_logits, t_logits
    print(f"KL = {float(kl.item()):.4f}", flush=True)

    # ---- graph term, each mode --------------------------------------------
    for label, freeze in (("unfrozen", False), ("frozen", True)):
        student.zero_grad(set_to_none=True)
        loss, _ = backward_batch_graph_loss(
            prompts=prompts,
            answers=answers,
            student_adapter=student_adapter,
            teacher_adapter=teacher_adapter,
            config=graph_config(freeze),
            device=device,
            loss_scale=1.0,
        )
        grads[label] = _snapshot(student)
        print(f"graph {label:8} = {float(loss.item()):.4f}", flush=True)

    student.zero_grad(set_to_none=True)

    # ---- report ------------------------------------------------------------
    fro, unf, klg = grads["frozen"], grads["unfrozen"], grads["KL"]

    print(f"\n{args.model}  vs  {args.teacher}  |  {args.dataset}"
          f"  |  {len(prompts)} graph prompts, batch {args.batch_size}\n")
    print(f"{'':16} {'||grad||':>12}")
    for name in ("KL", "unfrozen", "frozen"):
        print(f"{name:16} {_norm(grads[name]):12.6g}")

    print(f"\ncos(frozen, unfrozen) = {_cosine(fro, unf):+.4f}")
    print(f"cos(unfrozen, KL)    = {_cosine(unf, klg):+.4f}")
    print(f"cos(frozen,   KL)    = {_cosine(fro, klg):+.4f}")
    unf_n, fro_n = _norm(unf), _norm(fro)
    if unf_n > 0:
        print(f"||frozen|| / ||unfrozen|| = {fro_n / unf_n:.4f}")

    names_by_group: dict[str, list[str]] = {}
    for n in set(fro) | set(unf):
        names_by_group.setdefault(_group_of(n), []).append(n)

    hdr = f"\n{'group':12} {'||frozen||':>12} {'||unfrozen||':>13} {'cos':>8} {'ratio':>8}"
    print(hdr)
    print("-" * (len(hdr) - 1))
    for label, _ in _GROUPS + [("other", "")]:
        names = names_by_group.get(label)
        if not names:
            continue
        nf, nu = _norm(fro, names), _norm(unf, names)
        ratio = nf / nu if nu > 0 else float("nan")
        print(f"{label:12} {nf:12.5g} {nu:13.5g} {_cosine(fro, unf, names):+8.4f} {ratio:8.4f}")


if __name__ == "__main__":
    main()

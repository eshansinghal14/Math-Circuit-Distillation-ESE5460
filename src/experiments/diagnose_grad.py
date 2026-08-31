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

--modes attributes a direction change to one flag or the other. Every mode is
measured against the same unfrozen reference, so a run with
``--modes attn-only,rms-only,frozen`` says whether the attention freeze, the
RMSNorm freeze, or both together account for the gap -- without a training run
per hypothesis. If attn-only reproduces frozen's cosine while rms-only sits near
1.0, the attention pattern is the whole story.

Per-group rows matter because the freeze zeroes different paths in different
places.  Since the scoping fix, q_proj/k_proj should receive graph gradient in
*both* modes -- a q/k row that is zero under freeze but nonzero unfrozen means
the freeze is still leaking into the parameter path somewhere.

--graph-node-labels selects which supernodes the loss is built from, and must
match the run being diagnosed. Omitted (the default) means arg-token + DLA
supernodes; supplying labels switches to ANOVA supernodes and builds an MLP-input
cache for each model first.

Memory: three CPU fp32 copies of the student's gradients (~15 GB for a 1B
student), independent of how many --modes are requested -- only the unfrozen
reference stays resident. Reduce --n-prompts before reducing anything else; it
does not change what is measured, only the batch the gradients are estimated
from.

Usage:
    python -m experiments.diagnose_grad \
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
    ap.add_argument(
        "--graph-node-labels", "--graph_node_labels",
        nargs="+", default=[], dest="graph_node_labels", metavar="LABEL",
        help=(
            "ANOVA supernode labels, e.g. 'sum units'. Pass 'all' for every category. "
            "Omit to use the arg-token + DLA supernodes, which is what graph_kd does "
            "when its own --graph-node-labels is empty. Must match the training run "
            "being diagnosed, or the gradients describe a different objective."
        ),
    )
    ap.add_argument("--anova-range-radius", "--anova_range_radius", type=int, default=0,
                    dest="anova_range_radius")
    ap.add_argument("--anova-neuron-chunk", "--anova_neuron_chunk", type=int, default=None,
                    dest="anova_neuron_chunk")
    ap.add_argument("--cache-batch-size", "--cache_batch_size", type=int, default=32,
                    dest="cache_batch_size",
                    help="Prompt batch size when building the MLP input cache.")
    ap.add_argument("--prop-neurons-per-layer", type=float, default=0.1)
    ap.add_argument("--top-k-logits", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--graph-loss-type", default="jsd")
    ap.add_argument(
        "--modes",
        default="frozen",
        help=(
            "Comma-separated freeze modes to compare against unfrozen: any of "
            "attn-only, rms-only, frozen. Every mode is measured against the same "
            "unfrozen reference, so 'attn-only,rms-only,frozen' attributes the "
            "direction change to one flag or the other in a single run."
        ),
    )
    ap.add_argument("--kl-token-chunk-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    freeze_flags = {
        "attn-only": (True, False),
        "rms-only": (False, True),
        "frozen": (True, True),
    }
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in freeze_flags]
    if unknown:
        ap.error(f"unknown mode(s) {unknown}; choose from {sorted(freeze_flags)}")

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

    # ANOVA labelling needs an MLP-input cache per model. Built from the *train*
    # split via load_data, matching GraphKDTrainer rather than create_graph's own
    # load_split(dataset, "all") convention -- the point of this script is to
    # reproduce the training gradient, so it has to mirror the trainer.
    student_mlp_cache = teacher_mlp_cache = None
    if args.graph_node_labels:
        from graph_loss.precompute_mlp_inputs import build_mlp_input_cache

        student_mlp_cache = build_mlp_input_cache(
            student_adapter, args.dataset, args.model,
            data_dict=train_data, batch_size=args.cache_batch_size,
        )
        teacher_mlp_cache = build_mlp_input_cache(
            teacher_adapter, args.dataset, args.teacher,
            data_dict=train_data, batch_size=args.cache_batch_size,
        )

    def graph_config(freeze_attention: bool, freeze_rms_norm: bool) -> GraphAuxConfig:
        return GraphAuxConfig(
            lambda_graph=1.0,
            teacher_prop_neurons_per_layer=args.prop_neurons_per_layer,
            student_prop_neurons_per_layer=args.prop_neurons_per_layer,
            top_k_logits=args.top_k_logits,
            temperature=args.temperature,
            student_nodes_per_label=args.nodes_per_label,
            teacher_nodes_per_label=args.nodes_per_label,
            graph_loss_type=args.graph_loss_type,
            freeze_attention=freeze_attention,
            freeze_rms_norm=freeze_rms_norm,
            graph_node_labels=args.graph_node_labels or None,
            student_anova_range_radius=args.anova_range_radius,
            anova_neuron_chunk=args.anova_neuron_chunk,
            mlp_input_cache=student_mlp_cache,
            teacher_mlp_input_cache=teacher_mlp_cache,
            dataset_name=args.dataset,
        )

    # ---- KL term alone -----------------------------------------------------
    student.zero_grad(set_to_none=True)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    s_logits = student(input_ids, attention_mask=attention_mask).logits
    with torch.no_grad():
        t_logits = teacher(input_ids, attention_mask=attention_mask).logits
    kl = kl_loss(s_logits, t_logits, attention_mask, args.temperature, args.kl_token_chunk_size)
    kl.backward()
    klg = _snapshot(student)
    del s_logits, t_logits
    print(f"KL = {float(kl.item()):.4f}", flush=True)

    def graph_grad(freeze_attention: bool, freeze_rms_norm: bool, label: str):
        student.zero_grad(set_to_none=True)
        cfg = graph_config(freeze_attention, freeze_rms_norm)
        loss, _ = backward_batch_graph_loss(
            prompts=prompts,
            answers=answers,
            student_adapter=student_adapter,
            teacher_adapter=teacher_adapter,
            config=cfg,
            device=device,
            loss_scale=1.0,
        )
        print(f"graph {label:10} = {float(loss.item()):.4f}", flush=True)
        return _snapshot(student), float(loss.item())

    # Unfrozen is the reference every mode is compared against, so it alone stays
    # resident; each other mode is compared and then dropped. Peak memory is three
    # gradient copies regardless of how many modes are requested.
    unf, unf_loss = graph_grad(False, False, "unfrozen")

    names_by_group: dict[str, list[str]] = {}
    for n in unf:
        names_by_group.setdefault(_group_of(n), []).append(n)

    rows = []
    for mode in modes:
        fa, fr = freeze_flags[mode]
        g, loss_val = graph_grad(fa, fr, mode)
        per_group = {}
        for label, names in names_by_group.items():
            per_group[label] = (_norm(g, names), _norm(unf, names), _cosine(g, unf, names))
        rows.append({
            "mode": mode,
            "loss": loss_val,
            "norm": _norm(g),
            "cos_unf": _cosine(g, unf),
            "cos_kl": _cosine(g, klg),
            "groups": per_group,
        })
        del g

    student.zero_grad(set_to_none=True)

    # ---- report ------------------------------------------------------------
    unf_n = _norm(unf)
    print()
    print(f"{args.model}  vs  {args.teacher}  |  {args.dataset}"
          f"  |  {len(prompts)} graph prompts, batch {args.batch_size}")
    print()

    hdr = (f"{'':12} {'loss':>8} {'||grad||':>10} {'ratio':>8} "
           f"{'cos(.,unfrozen)':>16} {'cos(.,KL)':>11}")
    print(hdr)
    print("-" * len(hdr))
    print(f"{'KL':12} {float(kl.item()):8.4f} {_norm(klg):10.5g} {'':>8} {'':>16} {'':>11}")
    print(f"{'unfrozen':12} {unf_loss:8.4f} {unf_n:10.5g} {1.0:8.4f} "
          f"{1.0:+16.4f} {_cosine(unf, klg):+11.4f}")
    for r in rows:
        ratio = r["norm"] / unf_n if unf_n > 0 else float("nan")
        print(f"{r['mode']:12} {r['loss']:8.4f} {r['norm']:10.5g} {ratio:8.4f} "
              f"{r['cos_unf']:+16.4f} {r['cos_kl']:+11.4f}")

    for r in rows:
        print()
        print(f"{r['mode']} vs unfrozen, per group")
        ghdr = (f"{'group':12} {'||mode||':>12} {'||unfrozen||':>13} "
                f"{'cos':>8} {'ratio':>8}")
        print(ghdr)
        print("-" * len(ghdr))
        for label, _needle in _GROUPS + [("other", "")]:
            if label not in r["groups"]:
                continue
            nf, nu, c = r["groups"][label]
            ratio = nf / nu if nu > 0 else float("nan")
            print(f"{label:12} {nf:12.5g} {nu:13.5g} {c:+8.4f} {ratio:8.4f}")


if __name__ == "__main__":
    main()

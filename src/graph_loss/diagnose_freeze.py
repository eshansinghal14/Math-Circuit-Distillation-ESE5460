"""Compare frozen vs unfrozen attribution targets, without training anything.

The freeze changes only the *values* of the edge coefficients -- it is
backward-only and, since the scoping fix, applies to the edge-attribution
forwards alone.  So "why does distilling against frozen graphs work worse than
unfrozen?" is really a question about the target: is the frozen supergraph a
weaker thing to regress onto?

This builds the same supergraph the trainer builds, twice per prompt -- once with
the freeze flags on, once off -- and reports the properties that would make one a
worse regression target than the other:

target_entropy   Mean row entropy of the L1-normalised |supergraph| rows, i.e. the
                 exact distribution ``_compute_edge_loss`` regresses onto, in nats.
                 ``uniform`` beside it is log(n_supernodes), the maximum. Rows near
                 uniform carry little routing information, so the JSD gradient is
                 mostly noise no matter how well the student fits.

frac_tok/frac_bos
                 Share of each neuron row's attribution mass landing on token
                 nodes, and on position 0 specifically. Frozen attention is
                 sink-dominated in Llama, so a large frac_bos means the edges
                 mostly encode "attends to BOS" rather than argument routing.

frac_external    Mean and spread of the frac_external weights used to aggregate
                 neurons into supernodes. If the spread collapses toward zero the
                 weighting is inert and the supergraph is an unweighted mean.

labels           Distinct supernode labels vs. total supernodes. arg:<token>
                 labels collide when a prompt repeats a digit, and the training
                 alignment keeps only the last of each -- so distinct < total
                 means supernodes are silently dropped from the loss.

cross-mode JSD   How far apart the two targets are over their shared labels. Near
                 zero would mean the freeze barely matters here, and the gap you
                 measured is seed noise rather than mechanism.

Usage:
    python -m graph_loss.diagnose_freeze --model meta-llama/Meta-Llama-3-8B-Instruct \
        --dataset 22_add --n-prompts 8 --nodes-per-label 3
"""

from __future__ import annotations

import argparse
import logging
import random

import torch

from graph_loss.create_graph import create_graph
from graph_loss.graph import normalize_matrix
from graph_loss.hf_adapter import HFLlamaGraphAdapter
from utils import load_data, load_model

_EPS = 1e-8


def _row_dist(mat: torch.Tensor) -> torch.Tensor:
    """L1-normalised absolute rows -- what _compute_edge_loss actually regresses on."""
    a = mat.detach().float().abs()
    return a / a.sum(dim=1, keepdim=True).clamp(min=_EPS)


def _mean_row_entropy(mat: torch.Tensor) -> float:
    p = _row_dist(mat)
    return float((-(p * (p + 1e-12).log()).sum(dim=1)).mean().item())


def _jsd(p: torch.Tensor, q: torch.Tensor) -> float:
    m = 0.5 * (p + q)
    kl_pm = (p * ((p + _EPS).log() - (m + _EPS).log())).sum(dim=1)
    kl_qm = (q * ((q + _EPS).log() - (m + _EPS).log())).sum(dim=1)
    return float((0.5 * (kl_pm + kl_qm)).mean().item())


def _stats(result) -> dict:
    graph, sg = result.graph, result.supergraph
    adj = normalize_matrix(graph.adjacency_matrix)  # already abs + row-normalised
    n_n, n_t = graph.n_neurons, graph.n_tokens

    neuron_rows = adj[:n_n]
    tok_block = neuron_rows[:, n_n : n_n + n_t]

    fe_parts = []
    for members in sg.supernodes:
        if not members:
            continue
        total = adj[members].sum(dim=1)
        internal = adj[members][:, members].sum(dim=1)
        fe_parts.append((total - internal) / total.clamp(min=1e-10))
    fe = torch.cat(fe_parts) if fe_parts else torch.zeros(1, device=adj.device)

    labels = [lab[0] for lab in (sg.supernode_labels or []) if lab]
    n_sn = len(sg.supernodes)

    return {
        "n_supernodes": n_sn,
        "n_distinct_labels": len(set(labels)),
        "n_neurons": n_n,
        "entropy": _mean_row_entropy(sg.supernode_adjacency_matrix),
        "uniform": float(torch.tensor(float(max(n_sn, 1))).log().item()),
        "frac_tok": float(tok_block.sum(dim=1).mean().item()),
        "frac_bos": float(tok_block[:, 0].mean().item()) if n_t else 0.0,
        "fe_mean": float(fe.mean().item()),
        "fe_std": float(fe.std().item()) if fe.numel() > 1 else 0.0,
    }


def _aligned_dists(res_a, res_b):
    """Row distributions for both modes over their shared supernode labels.

    Mirrors the trainer's alignment, last-wins collision handling included, so the
    number reported is the one the loss would actually see.
    """

    def label_map(res):
        return {
            lab[0]: i
            for i, lab in enumerate(res.supergraph.supernode_labels or [])
            if lab
        }

    ma, mb = label_map(res_a), label_map(res_b)
    shared = sorted(set(ma) & set(mb))
    if len(shared) < 2:
        return None
    ia = torch.tensor([ma[k] for k in shared], dtype=torch.long)
    ib = torch.tensor([mb[k] for k in shared], dtype=torch.long)
    wa = res_a.supergraph.supernode_adjacency_matrix.detach().cpu()[ia][:, ia]
    wb = res_b.supergraph.supernode_adjacency_matrix.detach().cpu()[ib][:, ib]
    return _row_dist(wa), _row_dist(wb)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    ap = argparse.ArgumentParser(description="Compare frozen vs unfrozen graph targets.")
    ap.add_argument("--model", required=True, help="Model to profile (usually the teacher).")
    ap.add_argument("--dataset", required=True, help="Dataset under datasets/ to draw prompts from.")
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--nodes-per-label", type=int, default=3,
                    help="Must match the training run's --nodes-per-label.")
    ap.add_argument("--prop-neurons-per-layer", type=float, default=0.1)
    ap.add_argument("--attribution-batch-size", type=int, default=512)
    ap.add_argument("--top-k-logits", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.model)
    model.eval()
    adapter = HFLlamaGraphAdapter(model, tokenizer, device)

    train_data, _ = load_data(args.dataset)
    prompts = sorted(train_data.keys())
    random.Random(args.seed).shuffle(prompts)
    prompts = prompts[: args.n_prompts]

    modes = {"unfrozen": (False, False), "frozen": (True, True)}
    per_mode: dict[str, list[dict]] = {m: [] for m in modes}
    cross_jsd: list[float] = []

    for i, prompt in enumerate(prompts):
        results = {}
        for name, (fa, fr) in modes.items():
            # enable_grad, not no_grad: edge attribution itself needs autograd.
            with torch.enable_grad():
                results[name] = create_graph(
                    adapter,
                    prompt,
                    prop_neurons_per_layer=args.prop_neurons_per_layer,
                    top_k_logits=args.top_k_logits,
                    temperature=args.temperature,
                    batch_size=args.attribution_batch_size,
                    nodes_per_label=args.nodes_per_label,
                    no_grad_supergraph=True,
                    build_create_graph=False,
                    detach_result=True,
                    freeze_attention=fa,
                    freeze_rms_norm=fr,
                )
            per_mode[name].append(_stats(results[name]))

        pair = _aligned_dists(results["unfrozen"], results["frozen"])
        if pair is not None:
            cross_jsd.append(_jsd(*pair))
        print(f"  [{i + 1}/{len(prompts)}] {prompt!r}", flush=True)
        del results
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def avg(rows, key):
        return sum(r[key] for r in rows) / max(len(rows), 1)

    print(
        f"\n{args.model}  |  {args.dataset}  |  {len(prompts)} prompts"
        f"  |  nodes_per_label={args.nodes_per_label}\n"
    )
    hdr = (
        f"{'':10} {'entropy':>8} {'uniform':>8} {'frac_tok':>9} {'frac_bos':>9}"
        f" {'fe_mean':>8} {'fe_std':>7} {'labels':>10} {'neurons':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name, rows in per_mode.items():
        labels = f"{avg(rows, 'n_distinct_labels'):.1f}/{avg(rows, 'n_supernodes'):.1f}"
        print(
            f"{name:10} {avg(rows, 'entropy'):8.4f} {avg(rows, 'uniform'):8.4f}"
            f" {avg(rows, 'frac_tok'):9.4f} {avg(rows, 'frac_bos'):9.4f}"
            f" {avg(rows, 'fe_mean'):8.4f} {avg(rows, 'fe_std'):7.4f}"
            f" {labels:>10} {avg(rows, 'n_neurons'):8.0f}"
        )

    if cross_jsd:
        mean_jsd = sum(cross_jsd) / len(cross_jsd)
        print(
            f"\ncross-mode JSD(unfrozen, frozen) over shared labels: {mean_jsd:.4f}"
            f"  (0 = identical target, log 2 = 0.6931 = disjoint)"
        )


if __name__ == "__main__":
    main()

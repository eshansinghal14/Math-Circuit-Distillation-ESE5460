"""Do the graph term's per-prompt gradients agree with each other? (KD + graph, svamp_eq)

In the KD + graph run on svamp_eq (position supernodes, influence selection,
--graph-rows last, lambda 8) the graph loss stayed at 0.067-0.070 for 50 steps
while the KD loss fell by 60%, with the graph term owning 20-40% of each update.
Two readings, which call for different fixes:

(a) each prompt's graph gradient points somewhere different, so a batch's graph
    gradient is mostly per-prompt noise and nothing accumulates across steps,
    whatever the weight;
(b) the prompts agree, but the loss is flat in the weights, so it needs more
    distance (weight, steps, learning rate).

For N training prompts with a cached teacher target, at the given student, one
backward each for

    graph  the prompt's graph loss, exactly as the trainer computes it
    kl     the KD loss on that prompt's response positions (--kl-tokens resp)
    kl1    the KD loss at the first response position only, the prediction the
           graph at prompt-end is built for

Each gradient is kept as its exact norm plus a count sketch (a fixed random sign
per weight, summed over contiguous chunks). Sketch inner products are unbiased
for the full ones, so every pairwise cosine is available without holding N
gradients of 1.2e9 entries; the sketch's own norm against the exact norm is
printed as the accuracy check.

Reported per objective: per-prompt gradient norm, the norm of the mean over 16
and over N prompts against what mutually orthogonal gradients would give, the
mean pairwise cosine, the split-half cosine; then graph against KD, per prompt
and between the means; then where in the network each gradient sits.

    cd src && python -u -m experiments.graph_grad_coherence --teacher <sft teacher> \\
        --student <sft student> --dataset svamp_eq --n 48 --graph-rows last \\
        --neuron-selection influence --supernode-type position
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DIR_ROOT, PromptAnswerDataset, collate_fn, load_data, load_model, set_bos_mode  # noqa: E402
from training.utils import argmax_kd_loss, kl_loss, kl_position_mask, load_student, student_autocast  # noqa: E402
import graph_loss.training as GT  # noqa: E402
from graph_loss.hf_adapter import HFLlamaGraphAdapter  # noqa: E402

LAYER = re.compile(r"layers\.(\d+)\.")


def param_group(name: str) -> str:
    m = LAYER.search(name)
    if m is None:
        return "embed/head" if "embed" in name or "lm_head" in name else "other"
    kind = "mlp" if ".mlp." in name else ("attn" if "attn" in name else "norm")
    return f"L{int(m.group(1)):02d}.{kind}"


class Sketcher:
    """Count sketch of the model's gradient: sign[j] * g[j] summed over contiguous chunks."""

    def __init__(self, model, chunk: int, seed: int = 0) -> None:
        self.named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.chunk = chunk
        gen = torch.Generator(device="cuda").manual_seed(seed)
        self.signs = [torch.randint(0, 2, (p.numel(),), generator=gen, device="cuda", dtype=torch.int8) * 2 - 1
                      for _, p in self.named]
        self.groups = [param_group(n) for n, _ in self.named]
        self.dim = sum(-(-p.numel() // chunk) for _, p in self.named)

    def grab(self) -> tuple[torch.Tensor, float, dict[str, float]]:
        """(sketch [dim] float64 on cpu, exact gradient norm, squared norm by parameter group)."""
        parts, sq, by = [], 0.0, {}
        for (_, p), s, grp in zip(self.named, self.signs, self.groups):
            n_b = -(-p.numel() // self.chunk)
            if p.grad is None:
                parts.append(torch.zeros(n_b, dtype=torch.float64, device="cuda"))
                continue
            g = p.grad.detach().float().flatten()
            q = float(g.square().sum(dtype=torch.float64))
            sq += q
            by[grp] = by.get(grp, 0.0) + q
            v = g * s
            pad = n_b * self.chunk - v.numel()
            if pad:
                v = torch.nn.functional.pad(v, (0, pad))
            parts.append(v.view(n_b, self.chunk).sum(1, dtype=torch.float64))
        return torch.cat(parts).cpu(), sq ** 0.5, by


def gram(Z: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
    """Inner products of the full gradients: sketch estimates off the diagonal, exact on it."""
    G = Z @ Z.T
    G[range(len(norms)), range(len(norms))] = norms ** 2
    return G


def coherence(name: str, Z: torch.Tensor, norms: torch.Tensor, weights: torch.Tensor | None = None) -> dict:
    n = len(norms)
    G = gram(Z, norms)
    C = G / (norms[:, None] * norms[None, :])
    off = ~torch.eye(n, dtype=torch.bool)
    per_prompt = (C.sum(1) - 1) / (n - 1)            # each prompt's mean cosine with the others
    half = n // 2
    A, B = slice(0, half), slice(half, n)

    def mean_norm(sl) -> float:
        k = G[sl, sl].shape[0]
        return float(G[sl, sl].sum().clamp(min=0).sqrt() / k)

    shared_sq = float(G[off].mean())                 # estimate of |E g|^2
    var = float((norms ** 2).mean())
    out = {
        "n": n,
        "sketch_norm_rel_err": float(((Z.norm(dim=1) / norms) - 1).abs().mean()),
        "norm_per_prompt": float(norms.mean()),
        "norm_mean_of_16": mean_norm(slice(0, min(16, n))),
        "norm_mean_of_16_if_orthogonal": float((norms[:16] ** 2).sum().sqrt() / min(16, n)),
        "norm_mean_of_all": mean_norm(slice(0, n)),
        "norm_mean_of_all_if_orthogonal": float((norms ** 2).sum().sqrt() / n),
        "pairwise_cos_mean": float(C[off].mean()),
        "pairwise_cos_se": float(per_prompt.std() / n ** 0.5),
        "pairwise_cos_q10_q50_q90": [float(x) for x in torch.quantile(C[off], torch.tensor([0.1, 0.5, 0.9], dtype=C.dtype))],
        "split_half_cos": float(G[A, B].mean() / (mean_norm(A) * mean_norm(B))),
        "shared_norm": max(shared_sq, 0.0) ** 0.5,
        # share of a 16-prompt batch gradient's squared norm that is the shared direction
        "shared_share_of_batch16": max(shared_sq, 0.0) / (max(shared_sq, 0.0) + max(var - shared_sq, 0.0) / 16),
    }
    print(f"\n== {name}: {n} prompts ==")
    print(f"  sketch norm vs exact norm, mean |rel err|: {out['sketch_norm_rel_err']:.4f}")
    print(f"  per-prompt gradient norm (mean)          : {out['norm_per_prompt']:.4f}")
    print(f"  norm of the mean over 16 prompts         : {out['norm_mean_of_16']:.4f}   "
          f"(if the 16 were orthogonal: {out['norm_mean_of_16_if_orthogonal']:.4f})")
    print(f"  norm of the mean over all {n:3d} prompts    : {out['norm_mean_of_all']:.4f}   "
          f"(if orthogonal: {out['norm_mean_of_all_if_orthogonal']:.4f})")
    print(f"  mean pairwise cosine                     : {out['pairwise_cos_mean']:+.4f} +- {out['pairwise_cos_se']:.4f}   "
          f"(q10 {out['pairwise_cos_q10_q50_q90'][0]:+.3f}, median {out['pairwise_cos_q10_q50_q90'][1]:+.3f}, "
          f"q90 {out['pairwise_cos_q10_q50_q90'][2]:+.3f})")
    print(f"  split-half cosine (first {half} vs last {n - half}) : {out['split_half_cos']:+.4f}")
    print(f"  norm of the shared direction             : {out['shared_norm']:.4f}   "
          f"= {out['shared_share_of_batch16']:.1%} of a 16-prompt batch gradient's squared norm")
    return out


def cross(name: str, Za, na, Zb, nb) -> dict:
    n = len(na)
    D = Za @ Zb.T
    same = torch.diagonal(D) / (na * nb)
    mean_a = float(gram(Za, na).sum().clamp(min=0).sqrt() / n)
    mean_b = float(gram(Zb, nb).sum().clamp(min=0).sqrt() / n)
    out = {"same_prompt_cos_mean": float(same.mean()), "same_prompt_cos_se": float(same.std() / n ** 0.5),
           "same_prompt_cos_frac_negative": float((same < 0).float().mean()),
           "cos_of_means": float(D.mean() / (mean_a * mean_b))}
    print(f"  {name:14s}: same-prompt cosine {out['same_prompt_cos_mean']:+.4f} +- {out['same_prompt_cos_se']:.4f} "
          f"({out['same_prompt_cos_frac_negative']:.0%} negative) | cosine of the two mean gradients {out['cos_of_means']:+.4f}")
    return out


def group_direction_test(G: torch.Tensor, a: torch.Tensor, n_perm: int = 4000, seed: int = 0) -> dict:
    """Is the mean gradient of group ``a`` the same direction as the mean of the rest?

    The cosine between the two group means, against the same statistic for random
    splits of the same sizes. Both means carry per-prompt noise, so the cosine is
    below 1 even when the groups share one direction; the random splits say how far.
    """
    def cos(mask: torch.Tensor) -> float:
        b = ~mask
        na, nb = int(mask.sum()), int(b.sum())
        den = (G[mask][:, mask].sum().clamp(min=1e-30).sqrt() / na) * (G[b][:, b].sum().clamp(min=1e-30).sqrt() / nb)
        return float(G[mask][:, b].mean() / den)

    gen = torch.Generator().manual_seed(seed)
    n, k = len(a), int(a.sum())
    null = []
    for _ in range(n_perm):
        m = torch.zeros(n, dtype=torch.bool)
        m[torch.randperm(n, generator=gen)[:k]] = True
        null.append(cos(m))
    null_t = torch.tensor(null)
    obs = cos(a)
    return {"cos": obs, "random_split_mean": float(null_t.mean()),
            "random_split_q05_q95": [float(null_t.quantile(0.05)), float(null_t.quantile(0.95))],
            "p_lower": float((null_t <= obs).float().mean())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--dataset", default="svamp_eq")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--graph-rows", choices=["all", "answer", "last"], default="last")
    ap.add_argument("--graph-edge-topk", type=int, default=0)
    ap.add_argument("--neuron-selection", choices=["norm", "influence"], default="influence")
    ap.add_argument("--supernode-type", choices=["arg", "position"], default="position")
    ap.add_argument("--position-bands", type=int, default=1)
    ap.add_argument("--teacher-prop", type=float, default=0.01)
    ap.add_argument("--student-prop", type=float, default=0.003)
    ap.add_argument("--nodes-per-label", type=int, default=10)
    ap.add_argument("--aggregation", choices=["normalised", "raw-signed"], default="normalised",
                    help="How neuron edges are pooled into supernode edges: |.|-normalised shares (the trainer's "
                         "default) or signed sums.")
    ap.add_argument("--restrict-rows", action="store_true",
                    help="Attribute only the neurons whose rows --graph-rows scores; same loss, far fewer "
                         "backward passes, which is what makes hundreds of neurons per node affordable.")
    ap.add_argument("--token-source-columns", action="store_true",
                    help="Append the token-embedding nodes as source columns (the signed aggregation expects them).")
    ap.add_argument("--loss-type", default="jsd",
                    help="Row distance; jsd compares |.|-normalised rows, rel-mse compares the signed entries.")
    ap.add_argument("--teacher-graph-batch-size", type=int, default=32)
    ap.add_argument("--student-graph-batch-size", type=int, default=8)
    ap.add_argument("--sketch-chunk", type=int, default=4096)
    ap.add_argument("--tag", default="sft-student")
    ap.add_argument("--out", default=None)
    ap.add_argument("--transfer-dataset", default=None,
                    help="Also take the gold cross-entropy gradient on N prompts of this other dataset, and report "
                         "how far a step down each objective's mean gradient on --dataset lowers that loss "
                         "(to first order), next to the same for --dataset's own gold cross-entropy.")
    ap.add_argument("--build-missing", action="store_true",
                    help="Build (and cache) the teacher target for prompts that have none, instead of "
                         "using only cached prompts.")
    args = ap.parse_args()
    set_bos_mode("on")
    logging.getLogger("graph_loss.training").setLevel(logging.ERROR)
    dev = torch.device("cuda")
    out_path = os.path.join(DIR_ROOT, args.out or (
        f"FINAL_RESULTS/{args.dataset}/graph_grad_coherence_{args.neuron_selection}_{args.supernode_type}"
        f"_{args.graph_rows}_{args.tag}.json"))

    config = GT.GraphAuxConfig(
        graph_dtype=torch.bfloat16, teacher_prop_neurons_per_layer=args.teacher_prop,
        student_prop_neurons_per_layer=args.student_prop, top_k_logits=0.95, temperature=1.0,
        teacher_graph_batch_size=args.teacher_graph_batch_size,
        student_graph_batch_size=args.student_graph_batch_size,
        student_nodes_per_label=args.nodes_per_label, teacher_nodes_per_label=args.nodes_per_label,
        graph_loss_type=args.loss_type, supergraph_aggregation=args.aggregation,
        token_source_columns=args.token_source_columns, restrict_target_rows=args.restrict_rows,
        graph_node_labels=None, token_path_rows="weighted",
        dataset_name=args.dataset, graph_rows=args.graph_rows, graph_edge_topk=args.graph_edge_topk,
        neuron_selection=args.neuron_selection, supernode_type=args.supernode_type,
        position_bands=args.position_bands)
    key = GT.teacher_target_cache_key(config, args.teacher)
    cache = GT.TeacherTargetCache(os.path.join(DIR_ROOT, "cache/teacher_targets", f"{key}.pt"))
    train, _ = load_data(args.dataset)
    if args.build_missing:
        # a spread of the sorted prompts, so a grid-ordered dataset is not sampled from one corner
        every = sorted(train.items())
        stride = max(1, len(every) // args.n)
        items = every[::stride][: args.n]
    else:
        items = [(p, a) for p, a in sorted(train.items()) if p in cache.entries][: args.n]
    print(f"teacher cache {key}: {len(cache)} prompts cached, using {len(items)} prompts "
          f"({sum(p in cache.entries for p, _ in items)} of them cached)", flush=True)
    if len(items) < 8:
        raise SystemExit("too few cached teacher targets for this configuration; pass --build-missing, "
                         "or run the trainer or the precompute first")

    student, tok = load_student(args.student, "bfloat16")
    student.train()
    s_ad = HFLlamaGraphAdapter(student, tok, dev)
    teacher, _ = load_model(args.teacher)
    teacher.eval()
    for p_ in teacher.parameters():
        p_.requires_grad_(False)
    t_ad = HFLlamaGraphAdapter(teacher, tok, dev) if args.build_missing else None
    sk = Sketcher(student, args.sketch_chunk)
    print(f"sketch dim {sk.dim} (chunk {args.sketch_chunk}) over {sum(p.numel() for _, p in sk.named):,} weights", flush=True)

    ZX, NX = {}, {}
    Z = {k: [] for k in ("graph", "kl", "kl1")}
    N = {k: [] for k in ("graph", "kl", "kl1")}
    BY = {k: {} for k in ("graph", "kl", "kl1")}
    recs = []
    t_start = time.perf_counter()
    for i, (prompt, answer) in enumerate(items):
        # ---- graph term, as backward_batch_graph_loss does it for one prompt
        student.zero_grad(set_to_none=True)
        entry = cache.get(prompt)
        if entry is None:
            t_sg, logit_ids, dla = GT._compute_teacher_target(prompt, answer, t_ad, config, dev)
            cache.put(prompt, GT._teacher_target_entry(t_sg, logit_ids, dla, answer=answer))
        else:
            t_sg, logit_ids, dla = GT._teacher_target_from_entry(entry, dev)
        loss, _ = GT.compute_prompt_graph_loss(
            prompt=prompt, student_adapter=s_ad, config=config, teacher_supergraph=t_sg,
            logit_token_ids=logit_ids, teacher_dla_logits=dla, answer=answer)
        if not loss.requires_grad or not torch.isfinite(loss):
            print(f"  [{i}] graph loss unusable ({float(loss.detach())}); skipped", flush=True)
            continue
        loss.backward()
        zg, ng, bg = sk.grab()
        graph_loss = float(loss.detach())
        del loss, t_sg
        gc.collect()
        torch.cuda.empty_cache()

        # ---- KD term on the same prompt: the trainer's sequence, mask and loss
        ds = PromptAnswerDataset(args.dataset, {prompt: answer}, tok)
        batch = collate_fn([ds[0]], pad_id=tok.eos_token_id)
        ids, am, rm = (batch[k].to(dev) for k in ("input_ids", "attention_mask", "response_mask"))
        with torch.no_grad():
            t_logits = teacher(ids, attention_mask=am).logits
        resp = kl_position_mask(am, rm, "resp", input_ids=ids, bos_token_id=tok.bos_token_id)
        first = int(resp[0].nonzero()[0])
        only_first = torch.zeros_like(resp)
        only_first[0, first] = True
        got = {}
        for name, mask in (("kl", resp), ("kl1", only_first)):
            student.zero_grad(set_to_none=True)
            with student_autocast():
                s_logits = student(ids, attention_mask=am).logits
            kl = kl_loss(s_logits, t_logits, mask, 1.0)
            kl.backward()
            got[name] = (*sk.grab(), float(kl.detach()))
            hit = bool(s_logits[0, first].argmax() == ids[0, first + 1])
            t_hit = bool(t_logits[0, first].argmax() == ids[0, first + 1])
            del s_logits, kl
        extra = {}
        if args.transfer_dataset:
            # KD on the teacher's top-1 token, and the gold cross-entropy, on the same response positions
            pos = resp[0].nonzero().squeeze(1)
            for name in ("hard", "ce"):
                student.zero_grad(set_to_none=True)
                with student_autocast():
                    s_logits = student(ids, attention_mask=am).logits
                obj = (argmax_kd_loss(s_logits, t_logits, resp) if name == "hard"
                       else F.cross_entropy(s_logits[0, pos].float(), ids[0, pos + 1]))
                obj.backward()
                z_, n_, _ = sk.grab()
                extra[name] = (z_, n_)
                del s_logits, obj
        del t_logits
        vals = {"graph": (zg, ng, bg), "kl": got["kl"][:3], "kl1": got["kl1"][:3]}
        if not all(torch.isfinite(v[0]).all() and v[1] > 0 for v in list(vals.values()) + list(extra.values())):
            print(f"  [{i}] non-finite or zero gradient; skipped", flush=True)
            continue
        for k, (z_, n_) in extra.items():
            ZX.setdefault(k, []).append(z_)
            NX.setdefault(k, []).append(n_)
        for k, (z, nrm, by) in vals.items():
            Z[k].append(z)
            N[k].append(nrm)
            for grp, q in by.items():
                BY[k][grp] = BY[k].get(grp, 0.0) + q / (nrm ** 2)
        recs.append({"i": i, "graph_loss": graph_loss, "kl": got["kl"][3], "kl1": got["kl1"][3],
                     "resp_positions": int(resp.sum()), "student_first_token_hit": hit, "teacher_first_token_hit": t_hit,
                     "norm_graph": ng, "norm_kl": got["kl"][1], "norm_kl1": got["kl1"][1]})
        print(f"  [{i}] graph {graph_loss:.4f} |g| {ng:.3f} | kl {got['kl'][3]:.4f} |g| {got['kl'][1]:.2f} | "
              f"kl1 {got['kl1'][3]:.4f} |g| {got['kl1'][1]:.2f} | first token S {int(hit)} T {int(t_hit)} "
              f"| {time.perf_counter() - t_start:.0f}s", flush=True)
    student.zero_grad(set_to_none=True)
    if args.build_missing:
        cache.flush()

    # ---- gold cross-entropy gradients on another dataset (what a step here does to that task)
    ZT, NT = [], []
    if args.transfer_dataset:
        other_train, other_test = load_data(args.transfer_dataset)   # a test-only family has no train split
        every = sorted({**other_test, **other_train}.items())
        for prompt, answer in every[::max(1, len(every) // args.n)][: args.n]:
            ds = PromptAnswerDataset(args.transfer_dataset, {prompt: answer}, tok)
            batch = collate_fn([ds[0]], pad_id=tok.eos_token_id)
            ids, am, rm = (batch[k].to(dev) for k in ("input_ids", "attention_mask", "response_mask"))
            pos = kl_position_mask(am, rm, "resp", input_ids=ids, bos_token_id=tok.bos_token_id)[0].nonzero().squeeze(1)
            student.zero_grad(set_to_none=True)
            with student_autocast():
                s_logits = student(ids, attention_mask=am).logits
            obj = F.cross_entropy(s_logits[0, pos].float(), ids[0, pos + 1])
            obj.backward()
            z_, n_, _ = sk.grab()
            if torch.isfinite(z_).all() and n_ > 0:
                ZT.append(z_)
                NT.append(n_)
            del s_logits, obj
        student.zero_grad(set_to_none=True)

    Zt = {k: torch.stack(v) for k, v in Z.items()}
    Nt = {k: torch.tensor(v, dtype=torch.float64) for k, v in N.items()}
    n = len(recs)
    summary = {"n": n, "cache_key": key,
               "graph_loss_mean": sum(r["graph_loss"] for r in recs) / n,
               "kl_mean": sum(r["kl"] for r in recs) / n, "kl1_mean": sum(r["kl1"] for r in recs) / n,
               "student_first_token_acc": sum(r["student_first_token_hit"] for r in recs) / n}
    print(f"\n{n} prompts | graph loss {summary['graph_loss_mean']:.4f} | kl {summary['kl_mean']:.4f} | "
          f"kl at first response position {summary['kl1_mean']:.4f} | student first-token acc {summary['student_first_token_acc']:.3f}")
    titles = {"graph": "graph term", "kl": "KD, response positions", "kl1": "KD, first response position"}
    for k in ("graph", "kl", "kl1"):
        summary[k] = coherence(titles[k], Zt[k], Nt[k])
    # the same, on the prompts whose first response token the student gets wrong / right
    miss = torch.tensor([not r["student_first_token_hit"] for r in recs])
    for label, sel in (("student first token wrong", miss), ("student first token right", ~miss)):
        if int(sel.sum()) >= 6:
            summary[f"graph | {label}"] = coherence(f"graph term | {label}", Zt["graph"][sel], Nt["graph"][sel])

    # does the graph term push the same way where the student is wrong as where it is right?
    if 4 <= int(miss.sum()) <= n - 4:
        print("\n== mean gradient on first-token-wrong prompts vs on first-token-right prompts ==")
        for k in ("graph", "kl1"):
            t = group_direction_test(gram(Zt[k], Nt[k]), miss)
            summary[f"{k}_wrong_vs_right_direction"] = t
            print(f"  {titles[k]:30s}: cosine {t['cos']:+.3f} | random splits of the same sizes: mean "
                  f"{t['random_split_mean']:+.3f}, 5-95% [{t['random_split_q05_q95'][0]:+.3f}, "
                  f"{t['random_split_q05_q95'][1]:+.3f}] | share of random splits at or below: {t['p_lower']:.3f}")

    print("\n== graph term against KD ==")
    summary["graph_vs_kl"] = cross("graph vs kl", Zt["graph"], Nt["graph"], Zt["kl"], Nt["kl"])
    summary["graph_vs_kl1"] = cross("graph vs kl1", Zt["graph"], Nt["graph"], Zt["kl1"], Nt["kl1"])
    summary["kl_vs_kl1"] = cross("kl vs kl1", Zt["kl"], Nt["kl"], Zt["kl1"], Nt["kl1"])
    for label, sel in (("first token wrong", miss), ("first token right", ~miss)):
        if int(sel.sum()) >= 6:
            print(f"  on prompts with the student's {label} ({int(sel.sum())}):")
            summary[f"graph_vs_kl1 | {label}"] = cross(
                "graph vs kl1", Zt["graph"][sel], Nt["graph"][sel], Zt["kl1"][sel], Nt["kl1"][sel])
            summary[f"graph_vs_kl | {label}"] = cross(
                "graph vs kl", Zt["graph"][sel], Nt["graph"][sel], Zt["kl"][sel], Nt["kl"][sel])
    # inner products of every pair of gradients, so further questions need no GPU
    torch.save({"gram": {k: gram(Zt[k], Nt[k]) for k in Zt},
                "cross": {"graph_kl": Zt["graph"] @ Zt["kl"].T, "graph_kl1": Zt["graph"] @ Zt["kl1"].T,
                          "kl_kl1": Zt["kl"] @ Zt["kl1"].T},
                "norms": Nt, "records": recs}, os.path.splitext(out_path)[0] + ".pt")

    if args.transfer_dataset and ZT and ZX:
        mk = lambda zs, ns: (torch.stack(zs), torch.tensor(ns, dtype=torch.float64))
        mnorm = lambda Zm, Nm: float(gram(Zm, Nm).sum().clamp(min=0).sqrt() / len(Nm))
        srcs = {"graph term": (Zt["graph"], Nt["graph"]), "KD, soft": (Zt["kl"], Nt["kl"]),
                "KD, teacher's top answer": mk(ZX["hard"], NX["hard"]), "gold labels": mk(ZX["ce"], NX["ce"])}
        refs = {f"{args.dataset} (trained on)": mk(ZX["ce"], NX["ce"]), f"{args.transfer_dataset} (not trained on)": mk(ZT, NT)}
        print(f"\n== what a step down each mean gradient on {args.dataset} does to the gold cross-entropy "
              f"({n} prompts of {args.dataset}, {len(NT)} of {args.transfer_dataset}) ==")
        print("cos: cosine with that task's mean gold-CE gradient; drop: first-order fall in that loss per unit step")
        print(f"{'':26s} " + " ".join(f"{r[:30]:>32s}" for r in refs))
        print(f"{'':26s} " + " ".join(f"{'cos':>16s}{'drop':>16s}" for _ in refs))
        summary["transfer"] = {}
        for sname, (Zs, Ns) in srcs.items():
            cells, row = [], {}
            for rname, (Zr, Nr) in refs.items():
                dot = float(Zs.mean(0) @ Zr.mean(0))
                c, drop = dot / (mnorm(Zs, Ns) * mnorm(Zr, Nr)), dot / mnorm(Zs, Ns)
                row[rname] = {"cos": c, "drop_per_unit_step": drop}
                cells.append(f"{c:+16.3f}{drop:+16.2f}")
            summary["transfer"][sname] = row
            print(f"{sname:26s} " + " ".join(cells))

    print("\n== where each gradient sits (share of squared norm, mean over prompts) ==")
    groups = sorted(set().union(*[set(b) for b in BY.values()]))
    summary["by_group"] = {k: {g: BY[k].get(g, 0.0) / n for g in groups} for k in BY}
    coarse = {}
    for k in BY:
        c = {}
        for g, v in summary["by_group"][k].items():
            m = re.match(r"L(\d+)\.(\w+)", g)
            if m:
                band = "early" if int(m.group(1)) < 5 else ("middle" if int(m.group(1)) < 11 else "late")
                name = f"layers {band} {m.group(2)}"
            else:
                name = g
            c[name] = c.get(name, 0.0) + v
        coarse[k] = c
    summary["by_band"] = coarse
    names = sorted(set().union(*[set(c) for c in coarse.values()]))
    print(f"{'':24s} " + " ".join(f"{k:>8s}" for k in coarse))
    for nm in names:
        print(f"{nm:24s} " + " ".join(f"{coarse[k].get(nm, 0.0):8.3f}" for k in coarse))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "summary": summary, "records": recs}, f, indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

"""How much of a model's answer do the neurons an attribution graph keeps actually carry?

The attribution graph on svamp_eq keeps, per token position, the 10 MLP neurons
with the highest selection score (graph.select_position_supernodes), drawn from
a per-layer pool of the top ``prop`` (position, neuron) units. On a 60-token
prompt that is ~600 of 7.9M units in the 1B student and of 27.5M in the 8B
teacher. This measures, with no graph built, what those units account for, how
that changes with the number kept, and whether a different scoring rule picks
better ones.

For each prompt, at the prompt-end prediction (target = the logit of the model's
own top next token minus the mean logit), for EVERY MLP neuron at every position
and layer:

    a           its activation (the input of mlp.down_proj)
    w.g         d target / d a = w_out . (d target / d mlp_out), one backward pass
    attribution a * (w.g), the quantity the graph's logit edges are made of

Scoring rules (higher = picked first, top k per position pooled over layers,
BOS position excluded, as the pipeline does):

    write       |a| * ||w_out||             pipeline --neuron-selection norm
    influence   |a * (w.g)|                 pipeline --neuron-selection influence
    contrast    |(a - mean a) * (w.g)|      new: the neuron's activation relative
                to its average over all positions of all the prompts, so a neuron
                that is on for every problem scores nothing

Reported per rule and k, averaged over prompts:

    kept        share of all (position, neuron) units picked
    coverage    sum |attribution| of the picked units / of all units
    knock-out   set the picked units to their mean activation in a real forward
                pass: the fall in the target as a share of the fall when every
                unit is set to its mean; next to what the attributions predict
                for it (first order) and to the same number of random units
    keep-only   set every unit EXCEPT the picked ones to its mean: the share of
                the target that survives; next to random units

``--pool-prop`` adds the pipeline's own rows: the same top-k per position but
restricted to the per-layer pool of the top ``prop`` units, exactly as the graph
is built today.

    cd src && python -u -m experiments.neuron_selection_eval --model <path> --label teacher \\
        --dataset svamp_eq --n 24 --pool-prop 0.01
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RULES = ("write", "influence", "contrast")


class Tap:
    """Capture every MLP's neuron activations and output; optionally overwrite activations."""

    def __init__(self, model) -> None:
        self.layers = model.model.layers
        self.acts: dict[int, torch.Tensor] = {}
        self.outs: dict[int, torch.Tensor] = {}
        self.replace: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None  # layer -> (mask [T, d], value [d])
        self.handles = []
        for i, layer in enumerate(self.layers):
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(self._pre(i)))
            self.handles.append(layer.mlp.register_forward_hook(self._post(i)))

    def _pre(self, i):
        def hook(module, args):
            x = args[0]
            if self.replace is not None and i in self.replace:
                mask, value = self.replace[i]
                x = torch.where(mask.unsqueeze(0), value.to(x.dtype).view(1, 1, -1), x)
            self.acts[i] = x
            return (x,)
        return hook

    def _post(self, i):
        def hook(module, args, output):
            self.outs[i] = output
        return hook

    def close(self) -> None:
        for h in self.handles:
            h.remove()


def target_of(logits: torch.Tensor, token: int) -> torch.Tensor:
    return logits[token] - logits.mean()


def attribute(model, tap: Tap, ids: torch.Tensor):
    """One forward and one backward: activations and d target / d activation for every unit.

    Returns acts [L, T, d_mlp], wg [L, T, d_mlp] (float32), the target token and its clean value.
    """
    tap.replace = None
    embed = model.get_input_embeddings()
    handle = embed.register_forward_hook(lambda m, i, o: o.detach().requires_grad_(True))
    try:
        with torch.enable_grad():
            logits = model(ids.unsqueeze(0)).logits[0, -1].float()
            token = int(logits.argmax())
            target = target_of(logits, token)
            L = len(tap.layers)
            grads = torch.autograd.grad(target, [tap.outs[i] for i in range(L)])
    finally:
        handle.remove()
    acts = torch.stack([tap.acts[i][0].detach().float() for i in range(L)])                     # [L, T, d_mlp]
    wg = torch.stack([grads[i][0].detach().float() @ tap.layers[i].mlp.down_proj.weight.detach().float()
                      for i in range(L)])                                                       # [L, T, d_mlp]
    return acts, wg, token, float(target.detach())


def run_with(model, tap: Tap, ids: torch.Tensor, token: int, mask: torch.Tensor, mean: torch.Tensor) -> float:
    """The target with the units in ``mask`` [L, T, d_mlp] set to their mean activation."""
    tap.replace = {i: (mask[i], mean[i]) for i in range(mask.size(0)) if bool(mask[i].any())}
    with torch.no_grad():
        value = float(target_of(model(ids.unsqueeze(0)).logits[0, -1].float(), token))
    tap.replace = None
    return value


def top_per_position(score: torch.Tensor, k: int) -> torch.Tensor:
    """Bool mask [L, T, d]: at each position after BOS, the k highest-scoring units pooled over layers."""
    L, T, d = score.shape
    flat = score.permute(1, 0, 2).reshape(T, L * d)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    idx = flat[1:].topk(min(k, L * d), dim=1).indices
    mask[1:].scatter_(1, idx, True)
    return mask.view(T, L, d).permute(1, 0, 2)


def pool_of(score: torch.Tensor, prop: float) -> torch.Tensor:
    """Bool mask [L, T, d]: per layer, the top ``prop`` of its (position, neuron) units, BOS excluded."""
    L, T, d = score.shape
    s = score.clone()
    s[:, 0] = -1.0
    flat = s.view(L, T * d)
    k = min(max(1, int(T * d * prop)), T * d)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, flat.topk(k, dim=1).indices, True)
    return mask.view(L, T, d) & (s >= 0)


def random_like(mask: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """As many random units per position as ``mask`` has, BOS excluded."""
    L, T, d = mask.shape
    counts = mask.permute(1, 0, 2).reshape(T, L * d).sum(1)
    out = torch.zeros(T, L * d, dtype=torch.bool, device=mask.device)
    for p in range(1, T):
        c = int(counts[p])
        if c >= L * d:
            out[p] = True
        elif c:
            # draws with replacement: a repeat costs one unit in a few hundred thousand
            out[p, torch.randint(L * d, (c,), generator=gen).to(mask.device)] = True
    return out.view(T, L, d).permute(1, 0, 2)


def evaluate(model, prompts: list[torch.Tensor], ks: list[int], pool_prop: float | None, seed: int = 0) -> dict:
    """The table described in the module docstring, for one model. ``prompts``: 1-D id tensors on the model's device."""
    tap = Tap(model)
    gen = torch.Generator().manual_seed(seed)
    try:
        # mean activation of every neuron over every non-BOS position of every prompt
        total, count = None, 0
        for ids in prompts:
            tap.replace = None
            with torch.no_grad():
                model(ids.unsqueeze(0))
            a = torch.stack([tap.acts[i][0, 1:].float().sum(0) for i in range(len(tap.layers))])  # [L, d]
            total = a if total is None else total + a
            count += ids.numel() - 1
        mean = total / count

        rows: dict[str, dict[str, float]] = {}
        n = 0
        for ids in prompts:
            acts, wg, token, clean = attribute(model, tap, ids)
            L, T, d = acts.shape
            everything = torch.ones(L, T, d, dtype=torch.bool, device=acts.device)
            everything[:, 0] = False
            floor = run_with(model, tap, ids, token, everything, mean)
            span = clean - floor
            if span < 1.0:
                # the answer does not depend on the MLPs' prompt-specific activity enough to take shares of
                print(f"  skipped a prompt: target {clean:.2f}, with every unit at its mean {floor:.2f}", flush=True)
                continue
            n += 1
            attr = acts * wg
            delta = (acts - mean.unsqueeze(1)) * wg          # first-order effect of setting a unit to its mean
            mass = attr[:, 1:].abs().sum()
            w_norm = torch.stack([tap.layers[i].mlp.down_proj.weight.detach().float().norm(dim=0) for i in range(L)])
            scores = {"write": acts.abs() * w_norm.unsqueeze(1), "influence": attr.abs(), "contrast": delta.abs()}
            picks: dict[str, torch.Tensor] = {}
            for rule in RULES:
                for k in ks:
                    picks[f"{rule} k={k}"] = top_per_position(scores[rule], k)
                if pool_prop:
                    pool = pool_of(scores[rule], pool_prop)
                    picks[f"{rule} k={ks[0]} pool={pool_prop:g} (as built today)"] = top_per_position(
                        torch.where(pool, scores[rule], torch.full_like(scores[rule], -1.0)), ks[0]) & pool
            for name, mask in picks.items():
                rnd = random_like(mask, gen)
                rec = {
                    "kept": float(mask.sum()) / float(everything.sum()),
                    "coverage": float(attr[mask].abs().sum() / mass),
                    "knockout": (clean - run_with(model, tap, ids, token, mask, mean)) / span,
                    "knockout_predicted": float(delta[mask].sum()) / span,
                    "knockout_random": (clean - run_with(model, tap, ids, token, rnd, mean)) / span,
                    "keep_only": (run_with(model, tap, ids, token, everything & ~mask, mean) - floor) / span,
                    "keep_only_random": (run_with(model, tap, ids, token, everything & ~rnd, mean) - floor) / span,
                }
                acc = rows.setdefault(name, {key: 0.0 for key in rec})
                for key, value in rec.items():
                    acc[key] += value
            rows.setdefault("_all", {"predicted_all": 0.0, "span": 0.0})
            rows["_all"]["predicted_all"] += float(delta[everything].sum()) / span
            rows["_all"]["span"] += span
        for rec in rows.values():
            for key in rec:
                rec[key] /= max(n, 1)
        return {"n_prompts": n, "rows": rows}
    finally:
        tap.close()


def _rank(x: torch.Tensor) -> torch.Tensor:
    return x.argsort().argsort().float()


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = float(a.norm() * b.norm())
    return float((a * b).sum()) / den if den > 0 else float("nan")


def by_position(model, prompts: list[torch.Tensor], ks: list[int]) -> dict:
    """Does a node's edge to the answer track what the node really does?

    A position node is the k highest-influence units at one position (k = 0: every
    unit there). Its edge to the answer is built from its units' attributions:
    ``sum |a * (w.g)|`` is what the |.|-normalised graph row uses, and the signed
    first-order sum is what the attributions predict for knocking the node out.
    The real effect is the fall in the target when the node's units are set to
    their means in a forward pass, one node at a time. Reported per k, averaged
    over prompts: the rank correlation across positions of the real effect with
    each of the two edge weights, how often the predicted sign is right on the
    nodes that matter (real effect above 1% of the span), and the share of the
    summed real effect that sits on the 5 nodes with the largest edge weight.
    """
    tap = Tap(model)
    try:
        total, count = None, 0
        for ids in prompts:
            tap.replace = None
            with torch.no_grad():
                model(ids.unsqueeze(0))
            a = torch.stack([tap.acts[i][0, 1:].float().sum(0) for i in range(len(tap.layers))])
            total = a if total is None else total + a
            count += ids.numel() - 1
        mean = total / count
        out = {k: {"rank_abs": 0.0, "rank_signed": 0.0, "sign_right": 0.0, "top5_share": 0.0,
                   "largest_node": 0.0, "n": 0, "n_sign": 0} for k in ks}
        for pi, ids in enumerate(prompts):
            acts, wg, token, clean = attribute(model, tap, ids)
            L, T, d = acts.shape
            everything = torch.ones(L, T, d, dtype=torch.bool, device=acts.device)
            everything[:, 0] = False
            span = clean - run_with(model, tap, ids, token, everything, mean)
            if span < 1.0:
                continue
            attr = acts * wg
            delta = (acts - mean.unsqueeze(1)) * wg
            for k in ks:
                picked = everything if k == 0 else top_per_position(attr.abs(), k)
                real, w_abs, w_signed = [], [], []
                for p in range(1, T):
                    node = torch.zeros_like(picked)
                    node[:, p] = picked[:, p]
                    real.append((clean - run_with(model, tap, ids, token, node, mean)) / span)
                    w_abs.append(float(attr[node].abs().sum()))
                    w_signed.append(float(delta[node].sum()) / span)
                real, w_abs, w_signed = (torch.tensor(v) for v in (real, w_abs, w_signed))
                rec = out[k]
                rec["rank_abs"] += _corr(_rank(real), _rank(w_abs))
                rec["rank_signed"] += _corr(_rank(real), _rank(w_signed))
                matter = real.abs() > 0.01
                if bool(matter.any()):
                    rec["sign_right"] += float((torch.sign(real[matter]) == torch.sign(w_signed[matter])).float().mean())
                    rec["n_sign"] += 1
                pos = real.clamp(min=0)
                if float(pos.sum()) > 0:
                    rec["top5_share"] += float(pos[w_abs.topk(min(5, len(w_abs))).indices].sum() / pos.sum())
                rec["largest_node"] += float(real.max())
                rec["n"] += 1
            print(f"  by position: prompt {pi + 1}/{len(prompts)} done", flush=True)
        for rec in out.values():
            n = max(rec["n"], 1)
            for key in ("rank_abs", "rank_signed", "top5_share", "largest_node"):
                rec[key] /= n
            rec["sign_right"] /= max(rec["n_sign"], 1)
        return {str(k): v for k, v in out.items()}
    finally:
        tap.close()


def show_by_position(label: str, result: dict) -> None:
    print(f"\n== {label}: does a position node's edge to the answer track its real effect? ==")
    print(f"{'units per node':>15s} {'rank corr, |edge|':>18s} {'rank corr, signed':>18s} {'sign right':>11s} "
          f"{'top-5 nodes hold':>17s} {'largest real effect':>20s}")
    for k, r in result.items():
        name = "all" if k == "0" else k
        print(f"{name:>15s} {r['rank_abs']:18.3f} {r['rank_signed']:18.3f} {r['sign_right']:11.3f} "
              f"{r['top5_share']:17.3f} {r['largest_node']:20.3f}")


def show(label: str, result: dict) -> None:
    rows = result["rows"]
    print(f"\n== {label}: {result['n_prompts']} prompts | target falls {rows['_all']['span']:.2f} when every unit is set to "
          f"its mean; the attributions predict {rows['_all']['predicted_all']:.2f} of that ==")
    print(f"{'rule':50s} {'kept':>9s} {'coverage':>9s} | {'knock-out':>9s} {'predicted':>9s} {'random':>7s} | "
          f"{'keep-only':>9s} {'random':>7s}")
    for name, r in rows.items():
        if name == "_all":
            continue
        print(f"{name:50s} {r['kept']:9.5f} {r['coverage']:9.3f} | {r['knockout']:9.3f} {r['knockout_predicted']:9.3f} "
              f"{r['knockout_random']:7.3f} | {r['keep_only']:9.3f} {r['keep_only_random']:7.3f}")


def main() -> None:
    from utils import DIR_ROOT, PromptAnswerDataset, load_data, load_model, set_bos_mode

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="model")
    ap.add_argument("--dataset", default="svamp_eq")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--k", default="10,30,100,300,1000,3000")
    ap.add_argument("--pool-prop", type=float, default=None)
    ap.add_argument("--by-position", default=None,
                    help="Comma-separated node sizes (0 = every unit at the position) for the per-node test of "
                         "edge weight against real effect, run INSTEAD of the main table on the first --n prompts.")
    args = ap.parse_args()

    set_bos_mode("on")
    model, tok = load_model(args.model)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    train, _ = load_data(args.dataset)
    every = sorted(train.items())
    items = every[::max(1, len(every) // args.n)][: args.n]
    data = PromptAnswerDataset(args.dataset, dict(items), tok)
    dev = next(model.parameters()).device
    prompts = [data[i]["input_ids"][: data[i]["prompt_len"]].to(dev) for i in range(len(data))]
    if args.by_position:
        result = {"by_position": by_position(model, prompts, [int(x) for x in args.by_position.split(",")])}
        show_by_position(args.label, result["by_position"])
        name = f"neuron_selection_by_position_{args.label}.json"
    else:
        result = evaluate(model, prompts, [int(x) for x in args.k.split(",")], args.pool_prop)
        show(args.label, result)
        name = f"neuron_selection_eval_{args.label}.json"
    path = os.path.join(DIR_ROOT, f"FINAL_RESULTS/{args.dataset}/{name}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"model": args.model, "dataset": args.dataset, **result}, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

"""Step-1 gradient anatomy across BOS conventions: what each loss pushes on, before any training.

Why graph distillation gains ~0.5 on 21_mult under --bos-mode legacy and nothing
under the standard convention (or under `off`). Under legacy the KD term trains
sequences without BOS while the graph term (and eval) see BOS; under `on`/`off`
both terms see the same sequences. This measures, on the untrained student and
the same prompts:

* the graph gradient under `legacy` and under `on` -- the graph term tokenises its
  own input, so the two should be the same gradient (cos ~1, same loss);
* the KD gradient under each requested (bos_mode, kl_tokens) variant;
* cos(KD, graph) per variant, globally and per parameter group, in fp32 (the
  trainer's --track-grad-metrics cosine is a bf16 subtraction and drifts with
  lambda);
* cos between the KD variants themselves -- is the no-BOS KD gradient a
  different direction from the BOS one?
* the BOS embedding row: each term's gradient there, and its share of the
  embedding gradient. Under legacy the KD term reaches that row only through
  the tied lm_head (BOS as an output token, p ~ 0);
* Adam's first step is ~sign(g), so per element the graph term decides the
  update where |lambda g_graph| > |g_KD|. ``dominated`` is that fraction and
  ``flipped`` the fraction whose sign it flips, per variant and lambda.

Takes every graph_kd argument (pass the run's own: labels, prop neurons, lambda
is ignored -- the graph gradient is taken at scale 1 and scaled here). The
student should be fp32 (--dtype float32) so the gradients are not rounded.

    PYTHONPATH=src python -m experiments.bos_grad_anatomy \\
        --model meta-llama/Llama-3.2-1B-Instruct --teacher meta-llama/Meta-Llama-3-8B-Instruct \\
        --dataset 22_add --dtype float32 --bos-mode legacy --seeds 1 \\
        --graph-node-labels "arg1 range" "arg1 units" "arg2 range" "arg2 units" "sum range" "sum units" \\
        --teacher-prop-neurons 0.1 --nodes-per-label 10 --anova-cache-device cuda \\
        --n-batches 2 --out FINAL_RESULTS/bos_mechanism/grad_anatomy.json

Read: if cos(KD legacy, graph) ~ 0 while cos(KD on, graph) is clearly positive,
the graph term is redundant with KD only when both see BOS. If the BOS row's
``dominated`` fraction is ~1 under legacy and ~0 under on, the row is a
parameter the graph term owns only under legacy (the hypothesis the
--bos-embedding freeze / graph-only runs then test causally).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DIR_ROOT, collate_fn, load_data, set_bos_mode, tokenize_prompt_answer  # noqa: E402
from graph_loss.training import backward_batch_graph_loss  # noqa: E402
from training.graph_kd import GraphKDTrainer, build_parser, config_from_args  # noqa: E402
from training.utils import kl_loss, kl_position_mask, student_autocast, sweep_apply  # noqa: E402

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _group(name: str) -> str:
    if "embed_tokens" in name or name.endswith("lm_head.weight"):
        return "embed"
    m = re.search(r"layers\.(\d+)\.(self_attn|mlp)", name)
    if m:
        return f"L{int(m.group(1)):02d}.{'attn' if m.group(2) == 'self_attn' else 'mlp'}"
    if "norm" in name:
        return "norms"
    return "other"


def _coarse(group: str) -> str:
    return group.split(".")[-1] if group.startswith("L") else group


def _grads(model, device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for n, p in model.named_parameters():
        if p.grad is not None:
            out[n] = p.grad.detach().float().to(device, copy=True)
    return out


def _add(acc: Optional[Dict[str, torch.Tensor]], g: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if acc is None:
        return g
    for n, t in g.items():
        if n in acc:
            acc[n] += t
        else:
            acc[n] = t
    return acc


def _batch_for_mode(tokenizer, prompts: List[str], answers: list, mode: str) -> dict:
    set_bos_mode(mode)
    examples = []
    for p, a in zip(prompts, answers):
        p_ids, a_ids = tokenize_prompt_answer(tokenizer, p, str(a))
        examples.append({"input_ids": torch.cat([p_ids, a_ids]), "prompt_len": int(p_ids.numel()),
                         "prompt": p, "answer": a})
    return collate_fn(examples, pad_id=tokenizer.eos_token_id)


def _kd_grad(trainer: GraphKDTrainer, prompts, answers, mode: str, kl_tokens: str,
             store: torch.device) -> tuple[Dict[str, torch.Tensor], float, float]:
    model, teacher, tok = trainer.model, trainer.teacher, trainer.tokenizer
    batch = _batch_for_mode(tok, prompts, answers, mode)
    ids = batch["input_ids"].to(_DEVICE)
    attn = batch["attention_mask"].to(_DEVICE)
    resp = batch["response_mask"].to(_DEVICE)
    mask = kl_position_mask(attn, resp, kl_tokens, input_ids=ids, bos_token_id=tok.bos_token_id)
    model.zero_grad(set_to_none=True)
    with student_autocast():
        s_logits = model(ids, attention_mask=attn).logits
    with torch.no_grad():
        t_logits = teacher(ids, attention_mask=attn).logits
    loss = kl_loss(s_logits, t_logits, mask, trainer.config.temperature, trainer.config.kl_token_chunk_size)
    # Teacher-forced first-answer-token accuracy, the trainer's step_tf_acc.
    first = resp.int().argmax(dim=1)
    rows = torch.arange(ids.size(0), device=ids.device)
    tf_acc = float((s_logits.detach()[rows, first - 1].argmax(-1) == ids[rows, first]).float().mean())
    loss.backward()
    del s_logits, t_logits
    return _grads(model, store), float(loss.item()), tf_acc


def _graph_grad(trainer: GraphKDTrainer, prompts, answers, mode: str,
                store: torch.device) -> tuple[Dict[str, torch.Tensor], float]:
    set_bos_mode(mode)
    trainer.model.zero_grad(set_to_none=True)
    loss, _ = backward_batch_graph_loss(
        prompts=prompts, answers=answers,
        student_adapter=trainer.student_adapter, teacher_adapter=trainer.teacher_adapter,
        config=trainer.graph_config, device=_DEVICE, loss_scale=1.0,
    )
    return _grads(trainer.model, store), float(loss.item())


def _pair_stats(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> dict:
    """cos and norms of a vs b, globally, per group and per coarse group."""
    acc: Dict[str, List[float]] = {}
    for n in a:
        if n not in b:
            continue
        x = a[n].to(_DEVICE).flatten()
        y = b[n].to(_DEVICE).flatten()
        vals = [float(x @ y), float(x @ x), float(y @ y)]
        g = _group(n)
        for key in ("all", g, _coarse(g)):
            cur = acc.setdefault(key, [0.0, 0.0, 0.0])
            for i in range(3):
                cur[i] += vals[i]
    out = {}
    for key, (dot, xx, yy) in acc.items():
        denom = (xx ** 0.5) * (yy ** 0.5)
        out[key] = {"cos": dot / denom if denom else float("nan"), "norm_a": xx ** 0.5, "norm_b": yy ** 0.5}
    return out


def _bos_row(g: Dict[str, torch.Tensor], bos_id: int) -> tuple[float, float]:
    """(|g[BOS row]|, |g[BOS row]| / |g[embedding]|)."""
    for n, t in g.items():
        if _group(n) == "embed":
            row = float(t[bos_id].norm())
            return row, row / max(float(t.norm()), 1e-30)
    return float("nan"), float("nan")


def _adam_influence(kd: Dict[str, torch.Tensor], graph: Dict[str, torch.Tensor], lam: float, bos_id: int) -> dict:
    """Fraction of elements where lam*graph outweighs KD, and where it flips sign(KD)."""
    tot = {"all": [0, 0, 0]}
    for n, x in kd.items():
        if n not in graph:
            continue
        x = x.to(_DEVICE)
        y = graph[n].to(_DEVICE) * lam
        dom = int((y.abs() > x.abs()).sum())
        flip = int((torch.sign(x + y) != torch.sign(x)).sum())
        g = _group(n)
        for key in ("all", _coarse(g)):
            cur = tot.setdefault(key, [0, 0, 0])
            cur[0] += dom
            cur[1] += flip
            cur[2] += x.numel()
        if g == "embed":
            xr, yr = x[bos_id], y[bos_id]
            tot["bos_row"] = [int((yr.abs() > xr.abs()).sum()), int((torch.sign(xr + yr) != torch.sign(xr)).sum()),
                              xr.numel()]
    return {k: {"dominated": d / n, "flipped": f / n} for k, (d, f, n) in tot.items() if n}


def main() -> None:
    parser = build_parser()
    g = parser.add_argument_group("bos_grad_anatomy")
    g.add_argument("--n-batches", type=int, default=2, dest="n_batches",
                   help="Training batches to sum the gradients over (default 2).")
    g.add_argument("--variants", nargs="+", default=["legacy:all", "legacy:resp", "on:resp", "on:all", "on:all-no-bos"],
                   help="KD variants as bos_mode:kl_tokens.")
    g.add_argument("--lambdas", type=float, nargs="+", default=[0.3, 1.0, 3.0, 10.0])
    g.add_argument("--grad-device", choices=["cpu", "cuda"], default="cpu", dest="grad_device",
                   help="Where the summed fp32 gradients are held: ~5 GB each on the 1B student, "
                        "two graph and one per KD variant.")
    g.add_argument("--out", type=str, default="FINAL_RESULTS/bos_mechanism/grad_anatomy.json")
    args = parser.parse_args()
    next(iter(sweep_apply(args)))  # collapse every list-valued argument to its first value
    if args.dtype != "float32":
        print(f"WARN: --dtype {args.dtype}; gradients are only exact in float32")
    store = torch.device(args.grad_device if args.grad_device == "cpu" or torch.cuda.is_available() else "cpu")
    seed = args.seeds[0]

    set_bos_mode("legacy")  # the adapter prepends BOS under legacy and on alike
    train_data, test_data = load_data(args.dataset, test_limit=args.test_limit)
    cfg = config_from_args(args, seed, os.path.join(DIR_ROOT, "FINAL_RESULTS", "bos_mechanism", "_anatomy_scratch"))
    cfg.eval_datasets = []
    trainer = GraphKDTrainer(cfg, train_data, test_data)
    trainer.model.train()
    bos_id = trainer.tokenizer.bos_token_id

    batches = []
    for i, b in enumerate(trainer.loader):
        if i >= args.n_batches:
            break
        batches.append((list(b["prompts"]), list(b["answers"])))

    variants = [tuple(v.split(":")) for v in args.variants]
    kd_sum: Dict[str, Optional[Dict[str, torch.Tensor]]] = {v: None for v in args.variants}
    kd_loss: Dict[str, List[float]] = {v: [] for v in args.variants}
    kd_tf: Dict[str, List[float]] = {v: [] for v in args.variants}
    graph_sum: Dict[str, Optional[Dict[str, torch.Tensor]]] = {"legacy": None, "on": None}
    graph_loss: Dict[str, List[float]] = {"legacy": [], "on": []}

    for bi, (prompts, answers) in enumerate(batches):
        t0 = time.time()
        for mode in ("legacy", "on"):
            gr, gl = _graph_grad(trainer, prompts, answers, mode, store)
            graph_sum[mode] = _add(graph_sum[mode], gr)
            graph_loss[mode].append(gl)
        for name, (mode, kl_tokens) in zip(args.variants, variants):
            gr, kl, tf = _kd_grad(trainer, prompts, answers, mode, kl_tokens, store)
            kd_sum[name] = _add(kd_sum[name], gr)
            kd_loss[name].append(kl)
            kd_tf[name].append(tf)
        print(f"batch {bi + 1}/{len(batches)} done in {time.time() - t0:.0f}s | graph loss "
              f"legacy {graph_loss['legacy'][-1]:.4f} on {graph_loss['on'][-1]:.4f} | KD "
              + " ".join(f"{v}={kd_loss[v][-1]:.3f}" for v in args.variants))
    trainer.model.zero_grad(set_to_none=True)

    G = graph_sum["legacy"]
    res: dict = {
        "config": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, list, type(None)))},
        "n_prompts": sum(len(p) for p, _ in batches),
        "graph_loss": graph_loss,
        "graph_on_vs_legacy": _pair_stats(graph_sum["on"], G)["all"],
        "kd_loss": kd_loss,
        "kd_tf_acc": kd_tf,
        "bos_row": {"graph": _bos_row(G, bos_id)},
        "kd_vs_graph": {},
        "kd_vs_kd": {},
        "adam_influence": {},
    }
    for v in args.variants:
        res["kd_vs_graph"][v] = _pair_stats(kd_sum[v], G)
        res["bos_row"][f"kd {v}"] = _bos_row(kd_sum[v], bos_id)
        res["adam_influence"][v] = {str(lam): _adam_influence(kd_sum[v], G, lam, bos_id) for lam in args.lambdas}
    for i, a in enumerate(args.variants):
        for b in args.variants[i + 1:]:
            res["kd_vs_kd"][f"{a} | {b}"] = _pair_stats(kd_sum[a], kd_sum[b])

    out = args.out if os.path.isabs(args.out) else os.path.join(DIR_ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    # ---- readable summary ----
    print("\n=== graph term across modes (should be the same gradient) ===")
    gm = res["graph_on_vs_legacy"]
    print(f"loss legacy {graph_loss['legacy']} on {graph_loss['on']} | cos(on, legacy) {gm['cos']:+.4f} "
          f"| |g| on {gm['norm_a']:.4f} legacy {gm['norm_b']:.4f}")
    print("\n=== KD vs graph (fp32, summed over batches) ===")
    print(f"{'variant':16s} {'KD loss':>8s} {'tf_acc':>7s} {'|g_KD|':>9s} {'|g_gr|':>8s} {'cos':>7s} "
          f"{'cos attn':>8s} {'cos mlp':>8s} {'cos emb':>8s}")
    for v in args.variants:
        s = res["kd_vs_graph"][v]
        print(f"{v:16s} {sum(kd_loss[v]) / len(kd_loss[v]):8.3f} {sum(kd_tf[v]) / len(kd_tf[v]):7.3f} "
              f"{s['all']['norm_a']:9.3f} {s['all']['norm_b']:8.4f} {s['all']['cos']:+7.3f} "
              f"{s.get('attn', {}).get('cos', float('nan')):+8.3f} {s.get('mlp', {}).get('cos', float('nan')):+8.3f} "
              f"{s.get('embed', {}).get('cos', float('nan')):+8.3f}")
    print("\n=== KD variants against each other ===")
    for k, s in res["kd_vs_kd"].items():
        print(f"{k:36s} cos {s['all']['cos']:+.3f} (attn {s.get('attn', {}).get('cos', float('nan')):+.3f}, "
              f"mlp {s.get('mlp', {}).get('cos', float('nan')):+.3f}, embed {s.get('embed', {}).get('cos', float('nan')):+.3f})")
    print("\n=== BOS embedding row: |g[BOS]| and its share of the embedding gradient ===")
    for k, (row, share) in res["bos_row"].items():
        print(f"{k:22s} |g[BOS]| {row:.3e}  share {share:.3e}")
    print("\n=== Adam step-1 influence of lambda*graph: dominated / flipped fraction ===")
    for v in args.variants:
        for lam in args.lambdas:
            a = res["adam_influence"][v][str(lam)]
            parts = " ".join(f"{k} {a[k]['dominated']:.3f}/{a[k]['flipped']:.3f}"
                             for k in ("all", "attn", "mlp", "embed", "norms", "bos_row") if k in a)
            print(f"{v:16s} lambda {lam:<5g} {parts}")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()

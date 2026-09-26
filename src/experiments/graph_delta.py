"""Where in the weights does the graph term's effect live? Swap blocks between seed-matched students.

A standard-KD student and a graph-KD student trained with the same seed share
initialisation, data order and every KD gradient's inputs, so their weight
difference is the graph term's cumulative effect (plus whatever divergence it
seeds). This script loads the KD student, and for each decoder layer and module
(attention, MLP, both) copies in the graph student's weights for just that block,
then scores the answer position teacher-forced with BOS (the eval format):

  * ``acc`` / ``commit``: eval_model's greedy exact match, and the share of greedy
    continuations that open with an integer (a leading space token is allowed, as
    eval_model allows it; the graph students often emit one);
  * ``p_q``: teacher-forced probability mass on tokens starting with ``?``.

The reverse direction (KD block into the graph student) is run too, so a block
that is sufficient and one that is necessary can be told apart. ``delta`` is
||W_graph - W_kd|| / ||W_kd - W_base|| per block when ``--base`` is given.

Usage (from the repository root, GPU):
    PYTHONPATH=src python -m experiments.graph_delta --kd <dir> --graph <dir> --base meta-llama/Llama-3.2-1B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DIR_ROOT, load_data, set_bos_mode  # noqa: E402
from training.utils import load_student, student_autocast  # noqa: E402
from experiments.case_studies import DEVICE, greedy_rows  # noqa: E402
from experiments.format_decomp import token_classes  # noqa: E402


@torch.no_grad()
def score(model, tokenizer, data: dict, masks, batch_size: int = 512) -> dict:
    num, q, text = masks
    acc, rows = greedy_rows(model, tokenizer, "swap", data, 1024, 5)
    tokenizer.padding_side = "right"
    prompts = list(data)
    p_q = 0.0
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=True).to(DEVICE)
        with student_autocast():
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        last = enc["attention_mask"].sum(1) - 1
        probs = torch.softmax(logits[torch.arange(len(chunk), device=DEVICE), last].float(), -1)
        p_q += float(probs[:, q].sum())
    n = len(prompts)
    return {"acc": acc, "commit": sum(r["pred"] is not None for r in rows.values()) / n, "p_q": p_q / n}


def blocks(n_layers: int) -> dict[str, list[str]]:
    """Block name -> parameter-name prefixes."""
    out = {"embed": ["model.embed_tokens."], "final_norm+head": ["model.norm.", "lm_head."]}
    for i in range(n_layers):
        out[f"L{i:02d}.attn"] = [f"model.layers.{i}.self_attn.", f"model.layers.{i}.input_layernorm."]
        out[f"L{i:02d}.mlp"] = [f"model.layers.{i}.mlp.", f"model.layers.{i}.post_attention_layernorm."]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kd", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--datasets", nargs="+", default=["21_mult", "22_add"])
    ap.add_argument("--test-limit", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(DIR_ROOT, "results", "graph_delta", "summary.json"))
    args = ap.parse_args()
    set_bos_mode("legacy")  # BOS at eval, as the recorded runs were scored

    sets = {ds: load_data(ds, test_limit=args.test_limit)[1] for ds in args.datasets}
    models = {}
    for role in ("kd", "graph"):
        models[role], tokenizer = load_student(getattr(args, role), "bfloat16")
        models[role].eval()
    masks = token_classes(tokenizer)
    params = {r: dict(m.named_parameters()) for r, m in models.items()}
    base = None
    if args.base:
        bm, _ = load_student(args.base, "bfloat16")
        base = {k: v.detach().to("cpu", torch.float32) for k, v in bm.named_parameters()}
        del bm
    if "lm_head.weight" not in params["kd"]:
        print("note: lm_head is tied to embed_tokens; the embed block swaps both")

    out = {"args": vars(args), "reference": {}, "swap": {}, "delta": {}}
    for role, m in models.items():
        out["reference"][role] = {ds: score(m, tokenizer, d, masks) for ds, d in sets.items()}
        print(role, out["reference"][role])

    for name, prefixes in blocks(models["kd"].config.num_hidden_layers).items():
        keys = [k for k in params["kd"] if any(k.startswith(p) for p in prefixes)]
        if not keys:
            continue
        if base is not None:
            num = sum(float((params["graph"][k].float().cpu() - params["kd"][k].float().cpu()).norm() ** 2) for k in keys)
            den = sum(float((params["kd"][k].float().cpu() - base[k]).norm() ** 2) for k in keys)
            out["delta"][name] = (num / max(den, 1e-30)) ** 0.5
        res = {}
        for into, src in (("kd", "graph"), ("graph", "kd")):
            saved = {k: params[into][k].detach().clone() for k in keys}
            with torch.no_grad():
                for k in keys:
                    params[into][k].copy_(params[src][k])
            res[f"{src}->{into}"] = {ds: score(models[into], tokenizer, d, masks) for ds, d in sets.items()}
            with torch.no_grad():
                for k in keys:
                    params[into][k].copy_(saved[k])
        out["swap"][name] = res
        g2k = res["graph->kd"][args.datasets[0]]
        k2g = res["kd->graph"][args.datasets[0]]
        print(f"{name:16s} delta {out['delta'].get(name, float('nan')):.3f} | graph->kd acc {g2k['acc']:.3f} "
              f"commit {g2k['commit']:.3f} p_q {g2k['p_q']:.3f} | kd->graph acc {k2g['acc']:.3f} commit {k2g['commit']:.3f} p_q {k2g['p_q']:.3f}")

    # Spans of whole layers: the first k (graph->kd), to find how much of the network carries the effect.
    n_layers = models["kd"].config.num_hidden_layers
    out["span"] = {}
    for lo, hi in [(0, k) for k in (1, 2, 4, 8, 12, n_layers)] + [(k, n_layers) for k in (2, 4, 8)]:
        keys = [k for k in params["kd"] if any(k.startswith(f"model.layers.{i}.") for i in range(lo, hi))]
        saved = {k: params["kd"][k].detach().clone() for k in keys}
        with torch.no_grad():
            for k in keys:
                params["kd"][k].copy_(params["graph"][k])
        res = {ds: score(models["kd"], tokenizer, d, masks) for ds, d in sets.items()}
        with torch.no_grad():
            for k in keys:
                params["kd"][k].copy_(saved[k])
        out["span"][f"L{lo}-{hi - 1}"] = res
        r = res[args.datasets[0]]
        print(f"graph layers {lo}-{hi - 1} -> kd: acc {r['acc']:.3f} commit {r['commit']:.3f} p_q {r['p_q']:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

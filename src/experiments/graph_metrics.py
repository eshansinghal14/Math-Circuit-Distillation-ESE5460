"""Which distance between attribution graphs keeps correctness information across model sizes?

Collect phase (one model at a time): for each test problem of a fixed-length
dataset (mix3: BOS a op b op c =, seven positions, so positions line up across
models and problems), build the model's graph exactly as the trainer does
(position supernodes, influence selection, N neurons per node, only the scored
rows attributed) and keep three position-indexed vectors:

    last    the trainer's scored row: the |.|-normalised inbound edges of the
            last-position node, one entry per source position
    answer  each position node's signed attribution to the answer, plus that
            token embedding's own, one entry per position
    input   gradient x input at the token embeddings for the answer logit (no
            neuron selection at all), one entry per position

and whether the model's greedy answer token is right. Saved per model.

Analyse phase: for a reference model and every other model, per-problem
distances, and the AUC of each distance for "the model is wrong on this
problem" (0.5 = no information):

    last_jsd        JSD of the |.|-normalised last rows (what the trainer uses)
    answer_jsd      the same for the answer rows
    answer_rank     1 - Spearman correlation of the signed answer rows
    input_jsd       JSD of the |.|-normalised input attributions
    input_rank      1 - Spearman of the signed input attributions
    *_contrast      the same vector with each model's mean vector over all
                    problems subtracted first, compared by 1 - cosine: removes
                    a layout that a model uses on every problem

    cd src && python -u -m experiments.graph_metrics collect --model ft8b=<path>:0.01 --n 200
    cd src && python -u -m experiments.graph_metrics analyse --ref ft8b
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DIR_ROOT, PromptAnswerDataset, collate_fn, load_data, set_bos_mode  # noqa: E402
from training.utils import load_student, student_autocast  # noqa: E402
from graph_loss.create_graph import create_graph  # noqa: E402
from graph_loss.hf_adapter import HFLlamaGraphAdapter  # noqa: E402

OUT = os.path.join(DIR_ROOT, "FINAL_RESULTS/mix3/graph_metrics")


def items_for(dataset: str, n: int):
    _, test = load_data(dataset)
    every = sorted(test.items())
    return every[::max(1, len(every) // n)][:n]


def collect(args) -> None:
    set_bos_mode("on")
    dev = torch.device("cuda")
    items = items_for(args.dataset, args.n)
    os.makedirs(OUT, exist_ok=True)
    for spec in args.model:
        label, rest = spec.split("=", 1)
        path, prop = rest.rsplit(":", 1)
        model, tok = load_student(path, "bfloat16")
        model.train()
        ad = HFLlamaGraphAdapter(model, tok, dev)
        rec = {"label": label, "path": path, "last": [], "answer": [], "input": [], "hit": [], "T": []}
        t0 = time.perf_counter()
        for prompt, answer in items:
            b = collate_fn([PromptAnswerDataset(args.dataset, {prompt: answer}, tok)[0]], pad_id=tok.eos_token_id)
            ids = b["input_ids"].to(dev)
            first = int(b["response_mask"][0].nonzero()[0])
            gold = int(ids[0, first])
            # graph, as the trainer builds it, attributing the gold answer token
            res = create_graph(ad, prompt, attribution_targets=torch.tensor([gold]), top_k_logits=0.95,
                               temperature=1.0, dtype=torch.bfloat16, node_labels=[],
                               nodes_per_label=args.nodes_per_label, no_grad_supergraph=True,
                               build_create_graph=False, neuron_selection="influence", supernode_type="position",
                               target_rows="last", batch_size=32, prop_neurons_per_layer=float(prop),
                               detach_result=True)
            g, sg = res.graph, res.supergraph
            T, nn_ = g.n_tokens, g.n_neurons
            locs = g.neuron_locations[:, 1].tolist()
            A = g.adjacency_matrix.float()
            pos_of = []
            for members in sg.supernodes:
                pos_of.append(max(locs[m] for m in members) if members else -1)
            last_sid = max(range(len(sg.supernodes)), key=lambda s: pos_of[s])
            W = sg.supernode_adjacency_matrix.float()
            last = torch.zeros(T)
            for s, p in enumerate(pos_of):
                if p >= 0:
                    last[p] += float(W[last_sid, s])
            arow = A[nn_ + T, : nn_ + T]                       # the single attributed logit
            answer = torch.zeros(T)
            for j in range(nn_):
                answer[locs[j]] += float(arow[j])
            answer += arow[nn_: nn_ + T].cpu()
            del res, g, sg, A
            # gradient x input at the embeddings
            emb = model.get_input_embeddings()(ids).detach().requires_grad_(True)
            with student_autocast():
                logits = model(inputs_embeds=emb).logits[0, first - 1].float()
            (logits[gold] - logits.mean()).backward()
            inp = (emb.grad[0].float() * emb[0].float()).sum(-1).detach().cpu()
            hit = int(logits.argmax()) == gold
            model.zero_grad(set_to_none=True)
            rec["last"].append(last)
            rec["answer"].append(answer)
            rec["input"].append(inp[: first])
            rec["hit"].append(hit)
            rec["T"].append(T)
        print(f"{label}: {len(items)} problems in {time.perf_counter() - t0:.0f}s, accuracy "
              f"{sum(rec['hit']) / len(rec['hit']):.3f}, positions {sorted(set(rec['T']))}", flush=True)
        torch.save(rec, os.path.join(OUT, f"{label}.pt"))
        del ad, model
        gc.collect()
        torch.cuda.empty_cache()


def jsd(a: torch.Tensor, b: torch.Tensor) -> float:
    p = a.abs() / a.abs().sum().clamp(min=1e-12)
    q = b.abs() / b.abs().sum().clamp(min=1e-12)
    m = 0.5 * (p + q)
    kl = lambda x, y: float((x * ((x + 1e-12) / (y + 1e-12)).log()).sum())  # noqa: E731
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def rank_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    ra, rb = a.argsort().argsort().float(), b.argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return 1.0 - float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp(min=1e-12))


def cos_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    return 1.0 - float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-12))


def auc(wrong: list[float], right: list[float]) -> float:
    if not wrong or not right:
        return float("nan")
    return sum((w > r) + 0.5 * (w == r) for w in wrong for r in right) / (len(wrong) * len(right))


def analyse(args) -> None:
    recs = {os.path.basename(p)[:-3]: torch.load(os.path.join(OUT, p))
            for p in sorted(os.listdir(OUT)) if p.endswith(".pt")}
    ref = recs[args.ref]
    T0 = ref["T"][0]
    keep = [i for i in range(len(ref["T"])) if ref["T"][i] == T0 and all(r["T"][i] == T0 for r in recs.values())]

    def stack(rec, key):
        return torch.stack([rec[key][i][:T0] for i in keep])

    metrics = {}
    for key in ("last", "answer", "input"):
        metrics[f"{key}_jsd"] = (key, jsd, False)
        if key != "last":
            metrics[f"{key}_rank"] = (key, rank_dist, False)
        metrics[f"{key}_contrast"] = (key, cos_dist, True)
    out = {"ref": args.ref, "n": len(keep), "models": {}}
    print(f"reference {args.ref} (accuracy {sum(ref['hit'][i] for i in keep) / len(keep):.3f}), {len(keep)} problems")
    head = f"{'model':12s} {'acc':>5s} " + " ".join(f"{m:>16s}" for m in metrics)
    print("AUC of the distance for 'wrong' (0.5 = none), then mean distance on right / wrong problems")
    print(head)
    for label, rec in recs.items():
        if label == args.ref:
            continue
        hits = [rec["hit"][i] for i in keep]
        row_auc, row_mean, res = [], [], {}
        for name, (key, fn, contrast) in metrics.items():
            A, B = stack(ref, key), stack(rec, key)
            if contrast:
                A, B = A - A.mean(0), B - B.mean(0)
            d = [fn(A[j], B[j]) for j in range(len(keep))]
            w = [x for x, h in zip(d, hits) if not h]
            r = [x for x, h in zip(d, hits) if h]
            a = auc(w, r)
            mr = sum(r) / len(r) if r else float("nan")
            mw = sum(w) / len(w) if w else float("nan")
            res[name] = {"auc": a, "mean_right": mr, "mean_wrong": mw}
            row_auc.append(f"{a:16.2f}")
            row_mean.append(f"{mr:7.3f}/{mw:<8.3f}")
        acc = sum(hits) / len(hits)
        out["models"][label] = {"accuracy": acc, **res}
        print(f"{label:12s} {acc:5.2f} " + " ".join(row_auc))
        print(f"{'':12s} {'':5s} " + " ".join(f"{x:>16s}" for x in row_mean))
    path = os.path.join(OUT, f"analysis_ref-{args.ref}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--model", action="append", required=True, help="label=path:prop; repeatable")
    c.add_argument("--dataset", default="mix3")
    c.add_argument("--n", type=int, default=200)
    c.add_argument("--nodes-per-label", type=int, default=100)
    a = sub.add_parser("analyse")
    a.add_argument("--ref", required=True)
    args = ap.parse_args()
    collect(args) if args.cmd == "collect" else analyse(args)


if __name__ == "__main__":
    main()

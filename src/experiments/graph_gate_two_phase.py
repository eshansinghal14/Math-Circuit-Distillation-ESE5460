"""Does a graph loss point a student at fixing its own errors? Teacher and student loaded one at a time.

experiments/graph_grad_coherence.py holds teacher and student together, which an
8B teacher on a 22 GB card does not allow. This runs the same comparison in two
phases: the teacher alone builds everything the student will be scored against
and is then freed; the student alone takes the gradients.

Graph losses, per prompt (position supernodes, N neurons per node, only the
scored rows attributed; neurons picked by --neuron-selection):

    graph         the trainer's loss with --graph-rows last: the |.|-normalised
                  input rows of the last-position node and the DLA node, JSD
    graph_signed  (--signed-rows) the same two rows pooled signed (raw-signed
                  aggregation, token columns) and compared by relative squared
                  error; the pooling divides the matrix by its total |mass|, so
                  sign is kept and overall scale is not
    signed        the answer row (experiments/answer_row_check.py): each position
                  node's signed attribution to the answer, relative squared error
    signed_unit   the answer row with each model's row scaled to unit |mass|:
                  sign kept, scale removed
    shares        the answer row, |.|-normalised, JSD

Reference directions on the same prompt: KD on the response (kl) and on its
first token (kl1), gold cross-entropy on the response (ce) and its first token
(ce1). The table is given for all prompts, for those whose first response token
the student gets wrong, for those where it differs from the teacher's, and for
those it gets right.

Each graph gradient is also split into the part every prompt shares (the mean
over the run's prompts) and the part specific to the prompt (what is left), and
each part is compared with the first-token references on its own. A loss whose
useful signal sits in the specific part while most of its gradient is the shared
part is being diluted, and removing the shared part is worth building.

    cd src && python -u -m experiments.graph_gate_two_phase --teacher <8B> --student <kd-trained 1B> \\
        --dataset svamp_eq --split sft --n 150 --nodes-per-label 100 --signed-rows
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DIR_ROOT, PromptAnswerDataset, collate_fn, load_data, load_model, load_split, set_bos_mode  # noqa: E402
from training.utils import kl_loss, kl_position_mask, load_student, student_autocast  # noqa: E402
import graph_loss.training as GT  # noqa: E402
from graph_loss.attribution.attribute import neuron_activation_means  # noqa: E402
from graph_loss.create_graph import create_graph  # noqa: E402
from graph_loss.hf_adapter import HFLlamaGraphAdapter  # noqa: E402
from graph_loss.loss import edge_similarity  # noqa: E402
from experiments.answer_row_check import answer_row  # noqa: E402
from experiments.graph_grad_coherence import Sketcher, coherence  # noqa: E402

ROW_LOSSES = ("signed", "signed_unit", "shares")
REFS = ("kl", "kl1", "ce", "ce1")
REF_NAMES = {"kl": "KD, response", "kl1": "KD, first token", "ce": "gold, response", "ce1": "gold, first token"}


OPERATORS = "+-*/"


def graph_cut(prompt: str, answer, tok, dataset: str, where: str):
    """Where in the answer the graph is built.

    Returns ``(graph_prompt, next_text, target)``: the text the graph is built on
    (prompt plus the gold response before the attributed token), that token's
    text, and the index of that token in the full training sequence. None when
    the cut does not fall on a token boundary of the training sequence.

    prompt-end  the first response token (the trainer's default)
    operator    the first arithmetic operator of the gold equation
    result      the first token of the number after "=" (the computed answer)
    """
    response = str(answer)
    if where == "prompt-end":
        prefix = ""
    elif where == "operator":
        i = next((k for k, ch in enumerate(response) if ch in OPERATORS and k > 0), None)
        if i is None:
            return None
        prefix = response[:i].rstrip(" ")
    elif where == "result":
        j = response.find("=")
        if j < 0:
            return None
        prefix = response[: j + 1] + (" " if response[j + 1: j + 2] == " " else "")
    else:
        raise ValueError(where)
    full = PromptAnswerDataset(dataset, {prompt: answer}, tok)[0]
    ids, plen = full["input_ids"].tolist(), full["prompt_len"]
    bos = tok.bos_token or ""
    g_ids = tok(prompt + prefix, add_special_tokens=not (bos and prompt.startswith(bos)))["input_ids"]
    t = len(g_ids)
    if t < plen or t >= len(ids) or ids[:t] != g_ids:
        return None
    return prompt + prefix, tok.decode(ids[t:t + 1]), t


def unit_mass_error(t_row: torch.Tensor, s_row: torch.Tensor) -> torch.Tensor:
    """Relative squared error between the two rows after scaling each to unit |mass|."""
    t = t_row.float() / t_row.float().abs().sum().clamp(min=1e-8)
    s = s_row.float() / s_row.float().abs().sum().clamp(min=1e-8)
    return (s - t.detach()).pow(2).sum() / t.detach().pow(2).sum().clamp(min=1e-12)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--dataset", default="svamp_eq")
    ap.add_argument("--split", default="sft", help="Dataset split the prompts come from (one the student did not train on).")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--nodes-per-label", type=int, default=100)
    ap.add_argument("--teacher-prop", type=float, default=0.003)
    ap.add_argument("--student-prop", type=float, default=0.01)
    ap.add_argument("--neuron-selection", choices=["influence", "contrast"], default="influence",
                    help="'contrast' scores a neuron by its activation relative to its mean over --means-n training "
                         "prompts (times the gradient), so always-on neurons are not picked.")
    ap.add_argument("--means-n", type=int, default=64)
    ap.add_argument("--signed-rows", action="store_true", help="Also score the trainer's two rows pooled signed.")
    ap.add_argument("--graph-position", choices=["prompt-end", "operator", "result"], default="prompt-end",
                    help="Which answer token the graph attributes: the first one (as the trainer does), the first "
                         "operator of the equation, or the computed result after '='. 'Wrong' and the first-token "
                         "references then refer to that token.")
    ap.add_argument("--sketch-chunk", type=int, default=4096)
    ap.add_argument("--tag", default="default")
    args = ap.parse_args()

    set_bos_mode("on")
    dev = torch.device("cuda")
    every = sorted(load_split(args.dataset, args.split).items())
    items = every[::max(1, len(every) // args.n)][: args.n]
    train = sorted(load_data(args.dataset)[0])
    means_prompts = train[::max(1, len(train) // args.means_n)][: args.means_n]
    K = args.nodes_per_label
    config = GT.GraphAuxConfig(
        graph_dtype=torch.bfloat16, teacher_prop_neurons_per_layer=args.teacher_prop,
        student_prop_neurons_per_layer=args.student_prop, top_k_logits=0.95, temperature=1.0,
        teacher_graph_batch_size=16, student_graph_batch_size=32, student_nodes_per_label=K, teacher_nodes_per_label=K,
        graph_loss_type="jsd", graph_node_labels=None, token_path_rows="weighted", dataset_name=args.dataset,
        graph_rows="last", graph_edge_topk=0, neuron_selection=args.neuron_selection, supernode_type="position",
        position_bands=1, restrict_target_rows=True)
    configs = {"graph": config}
    if args.signed_rows:
        configs["graph_signed"] = dataclasses.replace(
            config, supergraph_aggregation="raw-signed", token_source_columns=True, graph_loss_type="rel-mse")
    losses = tuple(configs) + ROW_LOSSES
    row_kw = dict(top_k_logits=0.95, temperature=1.0, dtype=torch.bfloat16, node_labels=[], nodes_per_label=K,
                  no_grad_supergraph=True, build_create_graph=False, neuron_selection=args.neuron_selection,
                  supernode_type="position", target_rows="none", batch_size=16)

    # ---------------- phase 1: the teacher alone
    teacher, tok = load_model(args.teacher)
    teacher.eval()
    for p_ in teacher.parameters():
        p_.requires_grad_(False)
    t_ad = HFLlamaGraphAdapter(teacher, tok, dev)
    if args.neuron_selection == "contrast":
        t_ad.neuron_act_means = neuron_activation_means(t_ad, means_prompts, torch.bfloat16)
    cuts = [graph_cut(p, a, tok, args.dataset, args.graph_position) for p, a in items]
    kept = [(it, c) for it, c in zip(items, cuts) if c is not None]
    print(f"graph at {args.graph_position}: {len(kept)} of {len(items)} problems have that token on a clean token "
          f"boundary; e.g. {kept[0][1][0][-40:]!r} -> {kept[0][1][1]!r}", flush=True)
    items = [it for it, _ in kept]
    cuts = [c for _, c in kept]
    stored = []
    t0 = time.perf_counter()
    for i, ((prompt, answer), (g_prompt, g_next, tgt)) in enumerate(zip(items, cuts)):
        entries = {}
        for name, cfg in configs.items():
            t_sg, logit_ids, dla = GT._compute_teacher_target(g_prompt, g_next, t_ad, cfg, dev)
            entries[name] = GT._teacher_target_entry(t_sg, logit_ids, dla, answer=answer)
            del t_sg
        with torch.enable_grad():
            t_res = create_graph(t_ad, g_prompt, prop_neurons_per_layer=args.teacher_prop, detach_result=True, **row_kw)
        probs = t_res.graph.logit_probabilities.detach().float()
        nodes, toks = answer_row(t_res, probs)
        ds = PromptAnswerDataset(args.dataset, {prompt: answer}, tok)
        batch = collate_fn([ds[0]], pad_id=tok.eos_token_id)
        ids, am, rm = (batch[k].to(dev) for k in ("input_ids", "attention_mask", "response_mask"))
        pos = kl_position_mask(am, rm, "resp", input_ids=ids, bos_token_id=tok.bos_token_id)[0].nonzero().squeeze(1)
        with torch.no_grad():
            t_logits = teacher(ids, attention_mask=am).logits[0, pos].float().cpu()
        j = int((pos == tgt - 1).nonzero()[0])
        stored.append({
            "entries": entries, "ids_t": t_res.graph.logit_token_ids.cpu(), "probs": probs.cpu(),
            "end_logits": t_res.target_position_logits.float().cpu(),
            "nodes": {k: v.detach().float().cpu() for k, v in nodes.items()}, "toks": toks.detach().float().cpu(),
            "t_logits": t_logits, "t_top": int(t_logits[j].argmax()),
            "t_hit": bool(t_logits[j].argmax() == ids[0, tgt].cpu()),
        })
        del t_res, nodes, toks
        if i % 16 == 0:
            print(f"  teacher [{i}] {time.perf_counter() - t0:.0f}s | mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB",
                  flush=True)
    del t_ad, teacher
    gc.collect()
    torch.cuda.empty_cache()
    print(f"teacher phase done: {len(stored)} prompts, {time.perf_counter() - t0:.0f}s", flush=True)

    # ---------------- phase 2: the student alone
    student, _ = load_student(args.student, "bfloat16")
    student.train()
    s_ad = HFLlamaGraphAdapter(student, tok, dev)
    if args.neuron_selection == "contrast":
        s_ad.neuron_act_means = neuron_activation_means(s_ad, means_prompts, torch.bfloat16)
    sk = Sketcher(student, args.sketch_chunk)
    names = losses + REFS
    Z = {k: [] for k in names}
    N = {k: [] for k in names}
    recs = []
    t0 = time.perf_counter()
    for i, ((prompt, answer), (g_prompt, g_next, tgt), st) in enumerate(zip(items, cuts, stored)):
        got, vals = {}, {}
        ok = True
        for name, cfg in configs.items():
            student.zero_grad(set_to_none=True)
            t_sg, logit_ids, dla = GT._teacher_target_from_entry(st["entries"][name], dev)
            loss, _ = GT.compute_prompt_graph_loss(
                prompt=g_prompt, student_adapter=s_ad, config=cfg, teacher_supergraph=t_sg,
                logit_token_ids=logit_ids, teacher_dla_logits=dla, answer=g_next)
            if loss.requires_grad and torch.isfinite(loss):
                loss.backward()
                got[name] = sk.grab()[:2]
                vals[name] = float(loss.detach())
            else:
                ok = False
            del loss, t_sg
            gc.collect()
            torch.cuda.empty_cache()

        student.zero_grad(set_to_none=True)
        s_res = create_graph(s_ad, g_prompt, attribution_targets=st["ids_t"], dla_model_logits=st["end_logits"].to(dev),
                             prop_neurons_per_layer=args.student_prop, detach_result=False,
                             skip_logit_attribution=False, **row_kw)
        s_nodes, s_tok = answer_row(s_res, st["probs"])
        labels = [k for k in st["nodes"] if k in s_nodes]
        t_row = torch.cat([torch.stack([st["nodes"][k] for k in labels]), st["toks"]]).to(dev)
        s_row = torch.cat([torch.stack([s_nodes[k] for k in labels]), s_tok])
        rows = {"signed": edge_similarity(t_row[None], s_row[None], "rel-mse"),
                "signed_unit": unit_mass_error(t_row, s_row),
                "shares": edge_similarity(t_row[None], s_row[None], "jsd")}
        ok = ok and all(v.requires_grad and torch.isfinite(v) for v in rows.values())
        if ok:
            for j, name in enumerate(ROW_LOSSES):
                student.zero_grad(set_to_none=True)
                rows[name].backward(retain_graph=j + 1 < len(ROW_LOSSES))
                got[name] = sk.grab()[:2]
                vals[name] = float(rows[name].detach())
        del rows, s_res, s_nodes, s_tok, s_row
        gc.collect()
        torch.cuda.empty_cache()
        if not ok:
            print(f"  [{i}] a graph loss is unusable; skipped", flush=True)
            continue
        # Printed before the reference gradients: against itself a model's KD gradient is zero
        # and the prompt is dropped below, but its graph loss is the floor being measured.
        print(f"  [{i}] losses | " + " | ".join(f"{k} {vals[k]:.5f}" for k in losses), flush=True)

        ds = PromptAnswerDataset(args.dataset, {prompt: answer}, tok)
        batch = collate_fn([ds[0]], pad_id=tok.eos_token_id)
        ids, am, rm = (batch[k].to(dev) for k in ("input_ids", "attention_mask", "response_mask"))
        resp = kl_position_mask(am, rm, "resp", input_ids=ids, bos_token_id=tok.bos_token_id)
        pos = resp[0].nonzero().squeeze(1)
        first = tgt - 1           # the position that predicts the attributed token
        only_first = torch.zeros_like(resp)
        only_first[0, first] = True
        hit, s_top = False, -1
        for name in REFS:
            student.zero_grad(set_to_none=True)
            with student_autocast():
                s_logits = student(ids, attention_mask=am).logits
            if name in ("kl", "kl1"):
                t_full = torch.zeros_like(s_logits, dtype=torch.float32)
                t_full[0, pos] = st["t_logits"].to(dev)
                obj = kl_loss(s_logits, t_full, resp if name == "kl" else only_first, 1.0)
                del t_full
            elif name == "ce":
                obj = F.cross_entropy(s_logits[0, pos].float(), ids[0, pos + 1])
            else:
                obj = F.cross_entropy(s_logits[0, first: first + 1].float(), ids[0, first + 1: first + 2])
            s_top = int(s_logits[0, first].argmax())
            hit = bool(s_top == int(ids[0, first + 1]))
            obj.backward()
            got[name] = sk.grab()[:2]
            vals[name] = float(obj.detach())
            del s_logits, obj
        if not all(torch.isfinite(z).all() and nrm > 0 for z, nrm in got.values()):
            print(f"  [{i}] non-finite or zero gradient; skipped", flush=True)
            continue
        for k, (z, nrm) in got.items():
            Z[k].append(z)
            N[k].append(nrm)
        recs.append({"i": i, **vals, "student_first_token_hit": hit, "teacher_first_token_hit": st["t_hit"],
                     "agrees_with_teacher": s_top == st["t_top"]})
        if i % 10 == 0:
            print(f"  [{i}] kl {vals['kl']:.4f} | ce1 {vals['ce1']:.3f} | first token S {int(hit)} T {int(st['t_hit'])} "
                  f"| {time.perf_counter() - t0:.0f}s", flush=True)
    student.zero_grad(set_to_none=True)

    n = len(recs)
    if n < 4:
        raise SystemExit(f"only {n} prompts gave usable gradients; nothing to compare")
    Zt = {k: torch.stack(v) for k, v in Z.items()}
    Nt = {k: torch.tensor(v, dtype=torch.float64) for k, v in N.items()}
    hit = torch.tensor([r["student_first_token_hit"] for r in recs])
    t_hit = torch.tensor([r["teacher_first_token_hit"] for r in recs])
    agree = torch.tensor([r["agrees_with_teacher"] for r in recs])
    summary = {"args": vars(args), "n_prompts": n, "records": recs,
               "student_first_token_acc": float(hit.float().mean()), "teacher_first_token_acc": float(t_hit.float().mean()),
               "student_agrees_with_teacher": float(agree.float().mean())}
    print(f"\n{n} prompts from {args.dataset}/{args.split} | {K} neurons per node, picked by {args.neuron_selection} | "
          f"first-token acc: student {summary['student_first_token_acc']:.3f}, teacher "
          f"{summary['teacher_first_token_acc']:.3f}; student = teacher on {summary['student_agrees_with_teacher']:.3f}")
    for name in losses:
        mean = lambda sel: sum(r[name] for r, s in zip(recs, sel) if s) / max(int(sum(sel)), 1)  # noqa: E731
        print(f"  {name:12s} loss: all {mean([True] * n):.4f} | student wrong {mean((~hit).tolist()):.4f} | "
              f"student right {mean(hit.tolist()):.4f}")
        summary[name] = coherence(f"{name}", Zt[name], Nt[name])

    def mean_norm(k, idx) -> float:
        G = Zt[k][idx] @ Zt[k][idx].T
        d = torch.arange(len(idx))
        G[d, d] = Nt[k][idx] ** 2
        return float(G.sum().clamp(min=0).sqrt() / len(idx))

    def cell(v: torch.Tensor) -> str:
        return f"{float(v.mean()):+.3f} +- {float(v.std() / len(v) ** 0.5):.3f}"

    groups = {"all prompts": torch.arange(n), "student's first token wrong": (~hit).nonzero().squeeze(1),
              "student wrong, teacher right": ((~hit) & t_hit).nonzero().squeeze(1),
              "student's first token differs from the teacher's": (~agree).nonzero().squeeze(1),
              "student's first token right": hit.nonzero().squeeze(1)}
    summary["alignment"], summary["shared_vs_specific"] = {}, {}
    for gname, idx in groups.items():
        if len(idx) < 3:
            print(f"\n== {gname}: {len(idx)} prompts, too few ==")
            continue
        print(f"\n== {gname} ({len(idx)} prompts): cosine of the mean gradients (same-prompt mean +- se) ==")
        print(f"{'':12s} " + " ".join(f"{REF_NAMES[r]:>26s}" for r in REFS))
        summary["alignment"][gname] = {}
        for name in losses + ("kl",):
            cells, row = [], {}
            for ref in REFS:
                c = float((Zt[name][idx].mean(0) @ Zt[ref][idx].mean(0)) / (mean_norm(name, idx) * mean_norm(ref, idx)))
                same = torch.diagonal(Zt[name][idx] @ Zt[ref][idx].T) / (Nt[name][idx] * Nt[ref][idx])
                if name == ref:
                    same = torch.ones_like(same)
                se = float(same.std() / len(idx) ** 0.5)
                row[ref] = {"cos_of_means": c, "same_prompt_mean": float(same.mean()), "same_prompt_se": se}
                cells.append(f"{c:+7.3f} ({float(same.mean()):+.3f} +- {se:.3f})")
            summary["alignment"][gname][name] = row
            print(f"{name:12s} " + " ".join(f"{c:>26s}" for c in cells))

    # ---- each graph gradient split into what every prompt shares and what is specific to the prompt
    print("\n== shared part vs prompt-specific part: same-prompt cosine with the first-token references ==")
    print("   full = the gradient as is | specific = minus the mean over all prompts | shared = that mean alone")
    for gname in ("student's first token wrong", "student's first token differs from the teacher's",
                  "student's first token right"):
        idx = groups[gname]
        if len(idx) < 3:
            continue
        print(f"\n   -- {gname} ({len(idx)} prompts) --")
        print(f"   {'':12s} {'shared share':>12s} | {'gold first: full':>18s} {'specific':>18s} {'shared':>18s} | "
              f"{'KD first: full':>18s} {'specific':>18s} {'shared':>18s}")
        summary["shared_vs_specific"][gname] = {}
        for name in losses:
            mean = Zt[name].mean(0)
            Zc = Zt[name] - mean
            share = float(mean.dot(mean) / (Zt[name] * Zt[name]).sum(1).mean())
            row, cells = {"shared_share_of_squared_norm": share}, []
            for ref in ("ce1", "kl1"):
                full = torch.diagonal(Zt[name][idx] @ Zt[ref][idx].T) / (Nt[name][idx] * Nt[ref][idx])
                spec = torch.diagonal(Zc[idx] @ Zt[ref][idx].T) / (Zc[idx].norm(dim=1) * Nt[ref][idx])
                shared = (Zt[ref][idx] @ mean) / (mean.norm() * Nt[ref][idx])
                row[ref] = {k: [float(v.mean()), float(v.std() / len(v) ** 0.5)]
                            for k, v in (("full", full), ("specific", spec), ("shared", shared))}
                cells.append(f"{cell(full):>18s} {cell(spec):>18s} {cell(shared):>18s}")
            summary["shared_vs_specific"][gname][name] = row
            print(f"   {name:12s} {share:12.3f} | " + " | ".join(cells))

    path = os.path.join(DIR_ROOT, f"FINAL_RESULTS/{args.dataset}/graph_gate_two_phase_{args.tag}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

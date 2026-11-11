"""Where the arg/DLA graph loss comes from, and how much of it is noise.

Graph-only training on svamp_cot (arg/DLA supernodes, --graph-position
final-answer) took the SFT student from 0.576 to 0.246 in 20 steps while the
graph loss stayed flat at ~0.30. The loss is the mean row JSD over ~73 rows,
one per token position plus the DLA row, equally weighted. This measures, on
training prompts, with the student and the cached teacher targets exactly as the
trainer builds them (compute_prompt_graph_loss is run and its compute_graph_loss
call intercepted):

* per-row student-vs-teacher JSD, grouped by what the row's position is
  (chat template, question words, question numbers, instruction, CoT response,
  DLA row), and each group's share of the total loss;
* a per-row noise floor: a second teacher graph for the same prompt at a
  different pre-selection fraction (--noise-prop), aligned by label. A row whose
  teacher-vs-teacher JSD is close to its student-vs-teacher JSD carries little the
  student can learn;
* edge concentration of the teacher rows: share of a row's mass in its top-k
  edges (k = 1, 3, 10), the input to any edge-pruning decision;
* the teacher-vs-teacher JSD on the top-k edges only, i.e. whether pruning to the
  strong edges removes the noise.

    cd src && python -m experiments.graph_row_noise --teacher <sft teacher dir> \\
        --student <sft student dir> --dataset svamp_cot --n 24
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DIR_ROOT, load_data, load_model, set_bos_mode, split_final_answer  # noqa: E402
from training.utils import load_student  # noqa: E402
import graph_loss.training as GT  # noqa: E402
from graph_loss.hf_adapter import HFLlamaGraphAdapter  # noqa: E402

EPS = 1e-8


def row_dist(m: torch.Tensor) -> torch.Tensor:
    a = m.float().abs()
    return a / a.sum(dim=1, keepdim=True).clamp(min=EPS)


def row_jsd(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    lp, lq, lm = (p + EPS).log(), (q + EPS).log(), (m + EPS).log()
    return 0.5 * ((p * (lp - lm)).sum(1) + (q * (lq - lm)).sum(1))


def align(W_ref, labels_ref, W_other, labels_other):
    """W_other re-indexed into W_ref's (row, col) label order; rows/cols missing in other stay 0.
    Returns (ref_rows, other_rows, row_labels) over the labels both have."""
    idx_o = {l: i for i, l in enumerate(labels_other)}
    common = [i for i, l in enumerate(labels_ref) if l in idx_o]
    r = torch.tensor(common, dtype=torch.long)
    o = torch.tensor([idx_o[labels_ref[i]] for i in common], dtype=torch.long)
    other = torch.zeros(len(common), W_ref.shape[1])
    other[:, r] = W_other.float().cpu()[o][:, o]
    return W_ref.float().cpu()[r], other, [labels_ref[i] for i in common]


def topk_mask(dist: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, dist.shape[1])
    idx = dist.topk(k, dim=1).indices
    return torch.zeros_like(dist).scatter_(1, idx, 1.0)


def categorize(tok, prompt: str, graph_prompt: str, n_rows_tok: int) -> list[str]:
    """Category of each arg row (position 1..n, BOS excluded), by character spans of the prompt."""
    enc = tok(graph_prompt, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    q_start = prompt.index("<|end_header_id|>") + len("<|end_header_id|>\n\n")
    instr = prompt.index("\nWrite the equation")
    asst = prompt.index("<|eot_id|>")
    cats = []
    for (s, e), tid in zip(offs, enc["input_ids"]):
        text = graph_prompt[s:e]
        if tid == tok.bos_token_id:
            continue
        if s >= len(prompt):
            cats.append("response")
        elif s < q_start or s >= asst:
            cats.append("template")
        elif s >= instr:
            cats.append("instruction")
        elif any(c.isdigit() for c in text):
            cats.append("question-number")
        else:
            cats.append("question-word")
    if len(cats) != n_rows_tok:
        raise RuntimeError(f"{len(cats)} categorised positions vs {n_rows_tok} arg rows")
    return cats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--dataset", default="svamp_cot")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--teacher-prop", type=float, default=0.01)
    ap.add_argument("--student-prop", type=float, default=0.003)
    ap.add_argument("--noise-prop", type=float, default=0.005)
    ap.add_argument("--nodes-per-label", type=int, default=10)
    ap.add_argument("--teacher-graph-batch-size", type=int, default=16)
    ap.add_argument("--student-graph-batch-size", type=int, default=8)
    ap.add_argument("--out", default="FINAL_RESULTS/svamp_cot/graph_row_noise.json")
    args = ap.parse_args()
    set_bos_mode("on")
    dev = torch.device("cuda")

    train, _ = load_data(args.dataset)
    items = sorted(train.items())[: args.n]

    student, tok = load_student(args.student, "bfloat16")
    student.eval()
    teacher, _ = load_model(args.teacher)
    teacher.eval()
    for p_ in teacher.parameters():
        p_.requires_grad_(False)
    s_ad, t_ad = HFLlamaGraphAdapter(student, tok, dev), HFLlamaGraphAdapter(teacher, tok, dev)

    def cfg(tprop):
        return GT.GraphAuxConfig(
            graph_dtype=torch.bfloat16, teacher_prop_neurons_per_layer=tprop,
            student_prop_neurons_per_layer=args.student_prop, top_k_logits=0.95, temperature=1.0,
            teacher_graph_batch_size=args.teacher_graph_batch_size,
            student_graph_batch_size=args.student_graph_batch_size,
            student_nodes_per_label=args.nodes_per_label, teacher_nodes_per_label=args.nodes_per_label,
            graph_loss_type="jsd", graph_node_labels=None, token_path_rows="weighted",
            dataset_name=args.dataset)

    config = cfg(args.teacher_prop)
    key = GT.teacher_target_cache_key(config, args.teacher)
    cache = GT.TeacherTargetCache(os.path.join(DIR_ROOT, "cache/teacher_targets", f"{key}.pt"))
    print(f"teacher cache {key}: {len(cache)} prompts")
    noise_cfg = cfg(args.noise_prop)

    captured = {}
    orig = GT.compute_graph_loss

    def spy(W_T, W_S, mapping, teacher_ids, student_ids, **kw):
        captured.update(W_T=W_T.detach().float().cpu(), W_S=W_S.detach().float().cpu(),
                        mapping=mapping, teacher_ids=teacher_ids, student_ids=student_ids)
        return orig(W_T, W_S, mapping, teacher_ids, student_ids, **kw)
    GT.compute_graph_loss = spy

    rows = defaultdict(lambda: defaultdict(list))
    per_prompt = []
    for i, (prompt, response) in enumerate(items):
        prefix, number = split_final_answer(str(response))
        gp = prompt + prefix
        entry = cache.get(gp)
        if entry is None:
            print(f"  [{i}] not cached, skipped")
            continue
        t_sg, logit_ids, dla = GT._teacher_target_from_entry(entry, dev)
        loss, metrics = GT.compute_prompt_graph_loss(
            prompt=gp, student_adapter=s_ad, config=config, teacher_supergraph=t_sg,
            logit_token_ids=logit_ids, teacher_dla_logits=dla, answer=number)
        W_T, W_S = captured["W_T"], captured["W_S"]
        t_labels = [l[0] for l in t_sg.supernode_labels]
        # student labels from the mapping: teacher id -> student id
        s_of_t = {t: next(iter(s)) for t, s in captured["mapping"].items()}
        s_labels = [None] * W_S.shape[0]
        for t, s in s_of_t.items():
            s_labels[s] = t_labels[t]
        T_rows, S_rows, lab = align(W_T, t_labels, W_S, [l if l is not None else f"__s{j}" for j, l in enumerate(s_labels)])
        tD, sD = row_dist(T_rows), row_dist(S_rows)
        jsd_ts = row_jsd(tD, sD)

        # second teacher graph at another pre-selection fraction (noise floor)
        with torch.no_grad():
            t2_sg, _, _ = GT._compute_teacher_target(gp, number, t_ad, noise_cfg, dev)
        t2_labels = [l[0] for l in t2_sg.supernode_labels]
        T_a, T2_rows, lab2 = align(W_T, t_labels, t2_sg.supernode_adjacency_matrix, t2_labels)
        jsd_tt = row_jsd(row_dist(T_a), row_dist(T2_rows))
        tt_by_label = dict(zip(lab2, jsd_tt.tolist()))

        n_arg = sum(1 for l in t_labels if l.startswith("arg:"))
        cats = categorize(tok, prompt, gp, n_arg)
        cat_of = {l: (cats[j] if l.startswith("arg:") else "dla") for j, l in enumerate(t_labels)}
        for r, l in enumerate(lab):
            c = cat_of[l]
            d = tD[r]
            nz = int((T_rows[r].abs() > 0).sum())
            rows[c]["jsd_ts"].append(float(jsd_ts[r]))
            if l in tt_by_label:
                rows[c]["jsd_tt"].append(tt_by_label[l])
            if nz > 0:
                srt = d.sort(descending=True).values
                for k in (1, 3, 10):
                    rows[c][f"top{k}_mass"].append(float(srt[:k].sum()))
                # teacher-vs-teacher on the top-3 teacher edges only (renormalised)
                if l in tt_by_label:
                    j2 = lab2.index(l)
                    m = topk_mask(row_dist(T_a)[j2:j2 + 1], 3)
                    a = row_dist(T_a)[j2:j2 + 1] * m
                    b = row_dist(T2_rows)[j2:j2 + 1] * m
                    a, b = a / a.sum().clamp(min=EPS), b / b.sum().clamp(min=EPS)
                    rows[c]["jsd_tt_top3"].append(float(row_jsd(a, b)))
                    ms = topk_mask(tD[r:r + 1], 3)
                    a, b = tD[r:r + 1] * ms, sD[r:r + 1] * ms
                    a, b = a / a.sum().clamp(min=EPS), b / b.sum().clamp(min=EPS)
                    rows[c]["jsd_ts_top3"].append(float(row_jsd(a, b)))
            rows[c]["n_sources"].append(nz)
        per_prompt.append({"loss": float(loss), "rows": len(lab), "aligned": metrics.get("aligned_teacher_supernodes")})
        print(f"  [{i}] loss {float(loss):.4f} rows {len(lab)} | tokens {len(cats)}", flush=True)
        del loss
        torch.cuda.empty_cache()

    GT.compute_graph_loss = orig
    total = sum(sum(v["jsd_ts"]) for v in rows.values())
    summary = {}
    print(f"\n{'category':16s} {'rows':>5s} {'loss share':>10s} {'JSD s-t':>8s} {'JSD t-t':>8s} "
          f"{'t-t/s-t':>7s} {'top1':>5s} {'top3':>5s} {'top10':>5s} {'srcs':>5s} {'s-t top3':>8s} {'t-t top3':>8s}")
    for c in ["template", "question-word", "question-number", "instruction", "response", "dla"]:
        v = rows.get(c)
        if not v:
            continue
        mean = lambda k: (sum(v[k]) / len(v[k])) if v.get(k) else float("nan")
        s = {k: mean(k) for k in ("jsd_ts", "jsd_tt", "top1_mass", "top3_mass", "top10_mass", "n_sources",
                                  "jsd_ts_top3", "jsd_tt_top3")}
        s["rows"] = len(v["jsd_ts"])
        s["loss_share"] = sum(v["jsd_ts"]) / total if total else float("nan")
        summary[c] = s
        print(f"{c:16s} {s['rows']:5d} {s['loss_share']:10.3f} {s['jsd_ts']:8.4f} {s['jsd_tt']:8.4f} "
              f"{s['jsd_tt'] / s['jsd_ts']:7.2f} {s['top1_mass']:5.2f} {s['top3_mass']:5.2f} {s['top10_mass']:5.2f} "
              f"{s['n_sources']:5.1f} {s['jsd_ts_top3']:8.4f} {s['jsd_tt_top3']:8.4f}")
    out = os.path.join(DIR_ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"args": vars(args), "summary": summary, "per_prompt": per_prompt,
                   "rows": {c: dict(v) for c, v in rows.items()}}, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

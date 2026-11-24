"""Gate for a graph-loss target: is it about the student's errors, and does its gradient help fix them?

A 50-step KD + graph run costs ~2 GPU-hours; this costs ~15 s a prompt. It answers,
for several candidate losses computed from the SAME teacher and student supergraphs
(position supernodes, the final position's row and the DLA row), at a given student:

1. Does the loss value predict the student's first-token errors (AUC)?
2. Do the per-prompt gradients share a direction, and is that direction different
   on prompts the student gets wrong from prompts it gets right? (The full-row JSD
   failed this: one generic direction, 94% of the loss on prompts already right.)
3. Does a step down the candidate's average gradient, taken from the OTHER prompts,
   lower the first-response-token loss of a prompt the student gets wrong? (Leave-
   one-out, so it measures what transfers across prompts, the only part of a
   gradient that can generalise.) The KD gradient's own transfer is the yardstick.

Candidates (each averaged over the final-position row and the DLA row; t and s are
the teacher's and student's row as shares of the whole row):

    full       JSD over every source column (the loss trained so far)
    numrest    JSD over the question-number columns plus one "everything else" bin:
               how much the row draws on each number, ignoring how the rest is split
    numonly    JSD over the question-number columns, renormalised among themselves:
               which numbers, ignoring how much
    point      -log of the student's share, among the numbers, of the teacher's
               most-attributed number: point at the number the teacher points at
    pointfull  -log of the student's whole-row share on that number

Reference gradients on the same prompt: KD on the response positions (kl), KD at
the first response position (kl1), cross-entropy on the gold first response token
(ce1, the direction that fixes that token).

Every gradient is kept as a count sketch (see graph_grad_coherence.py); all pairwise
inner products are saved to <out>.pt before any statistic is computed, and
``--analyze-only`` recomputes the report from that file without a GPU.

    cd src && python -u -m experiments.graph_target_gate --teacher <sft teacher> \\
        --student <sft student> --dataset svamp_eq --n 128
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DIR_ROOT  # noqa: E402

NUM = re.compile(r"\d+(?:\.\d+)?")
GRAPH_OBJ = ("full", "numrest", "numonly", "point", "pointfull")
REF_OBJ = ("kl", "kl1", "ce1")
NAMES = GRAPH_OBJ + REF_OBJ


def norm_num(s: str) -> str:
    return str(int(float(s))) if float(s).is_integer() else str(float(s))


def label_number(label: str) -> str | None:
    """The number a position supernode's token spells, or None."""
    if not label.startswith(("pos:", "arg:")):
        return None
    tok = label[4:].split("#")[0].split("/b")[0].strip()
    return norm_num(tok) if tok and tok.replace(".", "", 1).isdigit() else None


def jsd_rows(t: torch.Tensor, s: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    m = 0.5 * (t + s)
    lt, ls, lm = (t + eps).log(), (s + eps).log(), (m + eps).log()
    return 0.5 * ((t * (lt - lm)).sum(1) + (s * (ls - lm)).sum(1))


def candidate_losses(W_T, W_S, mapping, t_labels, keep_rows, eps: float = 1e-8):
    """Per-row candidate losses on the aligned rows, as _compute_edge_loss aligns them.

    Returns ({name: [rows] tensor with grad}, info) or (None, reason).
    """
    n_T = len(t_labels)
    t2s = {t: next(iter(s)) for t, s in mapping.items() if s}
    valid = sorted(t2s)
    dev = W_S.device
    tv = torch.tensor(valid, device=dev, dtype=torch.long)
    sv = torch.tensor([t2s[t] for t in valid], device=dev, dtype=torch.long)
    S = torch.zeros(len(valid), n_T, device=dev, dtype=W_S.dtype)
    S[:, tv] = W_S[sv][:, sv]
    T = W_T[tv]
    sel = [r for r, t in enumerate(valid) if t in keep_rows]
    num = sorted(c for c in valid if label_number(t_labels[c]) is not None)
    if not sel:
        return None, "no scored row aligned"
    if len(num) < 2:
        return None, f"{len(num)} aligned number column(s)"
    sel_t = torch.tensor(sel, device=dev, dtype=torch.long)
    T, S = T[sel_t].float().abs().detach(), S[sel_t].float().abs()
    t = T / T.sum(1, keepdim=True).clamp(min=eps)
    s = S / S.sum(1, keepdim=True).clamp(min=eps)
    is_num = torch.zeros(n_T, dtype=torch.bool, device=dev)
    is_num[num] = True
    tn, sn = t[:, is_num], s[:, is_num]
    t_rest, s_rest = t[:, ~is_num].sum(1, keepdim=True), s[:, ~is_num].sum(1, keepdim=True)
    tnn = tn / tn.sum(1, keepdim=True).clamp(min=eps)
    snn = sn / sn.sum(1, keepdim=True).clamp(min=eps)
    top = tnn.argmax(1, keepdim=True)
    losses = {
        "numrest": jsd_rows(torch.cat([tn, t_rest], 1), torch.cat([sn, s_rest], 1)),
        "numonly": jsd_rows(tnn, snn),
        "point": -(snn.gather(1, top) + eps).log().squeeze(1),
        "pointfull": -(sn.gather(1, top) + eps).log().squeeze(1),
    }
    info = {
        "rows": ["dla" if t_labels[valid[r]] == "dla" else "LAST" for r in sel],
        "n_number_columns": len(num),
        "teacher_top_number": [label_number(t_labels[num[int(k)]]) for k in top.squeeze(1)],
        "student_top_number": [label_number(t_labels[num[int(k)]]) for k in snn.argmax(1)],
        "teacher_share_on_top": [float(x) for x in tn.gather(1, top).squeeze(1)],
        "student_share_on_top": [float(x) for x in sn.detach().gather(1, top).squeeze(1)],
        "teacher_number_mass": [float(x) for x in tn.sum(1)],
        "student_number_mass": [float(x) for x in sn.detach().sum(1)],
    }
    return losses, info


# ───────────────────────────── collection (GPU) ─────────────────────────────

def collect(args, pt_path: str) -> None:
    import torch.nn.functional as F
    from utils import PromptAnswerDataset, collate_fn, load_data, load_model, set_bos_mode
    from training.utils import kl_loss, kl_position_mask, load_student, student_autocast
    import graph_loss.training as GT
    from graph_loss.hf_adapter import HFLlamaGraphAdapter
    from experiments.graph_grad_coherence import Sketcher

    set_bos_mode("on")
    logging.getLogger("graph_loss.training").setLevel(logging.ERROR)
    dev = torch.device("cuda")
    config = GT.GraphAuxConfig(
        graph_dtype=torch.bfloat16, teacher_prop_neurons_per_layer=args.teacher_prop,
        student_prop_neurons_per_layer=args.student_prop, top_k_logits=0.95, temperature=1.0,
        teacher_graph_batch_size=args.teacher_graph_batch_size,
        student_graph_batch_size=args.student_graph_batch_size,
        student_nodes_per_label=args.nodes_per_label, teacher_nodes_per_label=args.nodes_per_label,
        graph_loss_type="jsd", graph_node_labels=None, token_path_rows="weighted",
        dataset_name=args.dataset, graph_rows="last", graph_edge_topk=0,
        neuron_selection=args.neuron_selection, supernode_type="position", position_bands=1)
    key = GT.teacher_target_cache_key(config, args.teacher)
    cache = GT.TeacherTargetCache(os.path.join(DIR_ROOT, "cache/teacher_targets", f"{key}.pt"))
    train, _ = load_data(args.dataset)
    items = [(p, a) for p, a in sorted(train.items()) if p in cache.entries][: args.n]
    print(f"teacher cache {key}: {len(cache)} prompts cached, using {len(items)} of them", flush=True)
    if len(items) < 8:
        raise SystemExit("too few cached teacher targets for this configuration")

    student, tok = load_student(args.student, "bfloat16")
    student.train()
    s_ad = HFLlamaGraphAdapter(student, tok, dev)
    teacher, _ = load_model(args.teacher)
    teacher.eval()
    for p_ in teacher.parameters():
        p_.requires_grad_(False)
    sk = Sketcher(student, args.sketch_chunk)

    grab = {}
    orig = GT.compute_graph_loss

    def spy(W_T, W_S, mapping, teacher_ids, student_ids, **kw):
        grab.update(W_T=W_T, W_S=W_S, mapping=mapping, keep_rows=kw.get("keep_rows"))
        return orig(W_T, W_S, mapping, teacher_ids, student_ids, **kw)
    GT.compute_graph_loss = spy

    Z = {k: [] for k in NAMES}
    N = {k: [] for k in NAMES}
    recs = []
    t_start = time.perf_counter()
    for i, (prompt, answer) in enumerate(items):
        student.zero_grad(set_to_none=True)
        t_sg, logit_ids, dla = GT._teacher_target_from_entry(cache.get(prompt), dev)
        full, _ = GT.compute_prompt_graph_loss(
            prompt=prompt, student_adapter=s_ad, config=config, teacher_supergraph=t_sg,
            logit_token_ids=logit_ids, teacher_dla_logits=dla, answer=answer)
        t_labels = [l[0] if l else "" for l in t_sg.supernode_labels]
        cand, info = candidate_losses(grab["W_T"], grab["W_S"], grab["mapping"], t_labels, grab["keep_rows"] or set())
        if cand is None or not full.requires_grad:
            print(f"  [{i}] skipped: {info}", flush=True)
            del full
            grab.clear()
            gc.collect()
            torch.cuda.empty_cache()
            continue
        graph_losses = {"full": full, **{k: v.mean() for k, v in cand.items()}}
        got = {}
        ok = all(bool(torch.isfinite(v)) for v in graph_losses.values())
        if ok:
            for j, k in enumerate(GRAPH_OBJ):
                student.zero_grad(set_to_none=True)
                graph_losses[k].backward(retain_graph=j < len(GRAPH_OBJ) - 1)
                z, nrm, _ = sk.grab()
                got[k] = (z, nrm)
        values = {k: float(v.detach()) for k, v in graph_losses.items()}
        per_row = {k: [float(x) for x in v.detach()] for k, v in cand.items()}
        del full, cand, graph_losses, t_sg
        grab.clear()
        gc.collect()
        torch.cuda.empty_cache()

        # ---- reference objectives on the trainer's sequence
        ds = PromptAnswerDataset(args.dataset, {prompt: answer}, tok)
        batch = collate_fn([ds[0]], pad_id=tok.eos_token_id)
        ids, am, rm = (batch[k].to(dev) for k in ("input_ids", "attention_mask", "response_mask"))
        with torch.no_grad():
            t_logits = teacher(ids, attention_mask=am).logits
        resp = kl_position_mask(am, rm, "resp", input_ids=ids, bos_token_id=tok.bos_token_id)
        first = int(resp[0].nonzero()[0])
        only_first = torch.zeros_like(resp)
        only_first[0, first] = True
        hit = False
        for k in REF_OBJ:
            student.zero_grad(set_to_none=True)
            with student_autocast():
                s_logits = student(ids, attention_mask=am).logits
            if k == "ce1":
                obj = F.cross_entropy(s_logits[0, first].float()[None], ids[0, first + 1][None])
            else:
                obj = kl_loss(s_logits, t_logits, resp if k == "kl" else only_first, 1.0)
            obj.backward()
            z, nrm, _ = sk.grab()
            got[k] = (z, nrm)
            values[k] = float(obj.detach())
            hit = bool(s_logits[0, first].argmax() == ids[0, first + 1])
            del s_logits, obj
        t_hit = bool(t_logits[0, first].argmax() == ids[0, first + 1])
        del t_logits
        if not ok or not all(bool(torch.isfinite(z).all()) and nrm > 0 for z, nrm in got.values()):
            print(f"  [{i}] non-finite or zero gradient; skipped", flush=True)
            continue
        for k, (z, nrm) in got.items():
            Z[k].append(z)
            N[k].append(nrm)
        gold = NUM.search(str(answer))
        gold_first = norm_num(gold.group()) if gold else None
        rec = {"i": i, "student_first_token_hit": hit, "teacher_first_token_hit": t_hit,
               "first_token_paren": str(answer).lstrip().startswith("("), "gold_first_number": gold_first,
               "values": values, "per_row": per_row, "norms": {k: got[k][1] for k in NAMES}, **info}
        rec["teacher_top_is_gold"] = [x == gold_first for x in info["teacher_top_number"]]
        rec["student_top_is_gold"] = [x == gold_first for x in info["student_top_number"]]
        recs.append(rec)
        print(f"  [{i}] hit S{int(hit)} T{int(t_hit)} | " + " ".join(f"{k} {values[k]:.3f}" for k in GRAPH_OBJ)
              + f" | ce1 {values['ce1']:.3f} | T top {info['teacher_top_number']} S top {info['student_top_number']} "
              f"gold {gold_first} | {time.perf_counter() - t_start:.0f}s", flush=True)
    GT.compute_graph_loss = orig
    student.zero_grad(set_to_none=True)

    Zt = {k: torch.stack(v) for k, v in Z.items()}
    torch.save({"cross": {f"{a}|{b}": Zt[a] @ Zt[b].T for a in NAMES for b in NAMES},
                "norms": {k: torch.tensor(v, dtype=torch.float64) for k, v in N.items()},
                "records": recs, "cache_key": key, "args": vars(args)}, pt_path)
    print(f"saved {pt_path}", flush=True)


# ───────────────────────────── analysis (no GPU) ─────────────────────────────

def auc(scores, labels):
    """P(score of a positive > score of a negative); ties count half."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def auc_se(a, n_pos, n_neg):
    """Hanley-McNeil standard error of an AUC."""
    if not (a == a) or not n_pos or not n_neg:
        return float("nan")
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return max((a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg), 0.0) ** 0.5


def exact_gram(raw: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
    G = raw.clone()
    G[range(len(norms)), range(len(norms))] = norms ** 2
    return G


def shared_direction(G: torch.Tensor, norms: torch.Tensor) -> dict:
    n = len(norms)
    C = G / (norms[:, None] * norms[None, :])
    off = ~torch.eye(n, dtype=torch.bool)
    half = n // 2
    A, B = slice(0, half), slice(half, n)
    mn = lambda sl: float(G[sl, sl].sum().clamp(min=0).sqrt() / G[sl, sl].shape[0])
    shared_sq = max(float(G[off].mean()), 0.0)
    var = float((norms ** 2).mean())
    return {"norm_per_prompt": float(norms.mean()), "norm_mean_of_16": mn(slice(0, min(16, n))),
            "norm_mean_of_all": mn(slice(0, n)), "pairwise_cos": float(C[off].mean()),
            "split_half_cos": float(G[A, B].mean() / (mn(A) * mn(B))), "shared_norm": shared_sq ** 0.5,
            "shared_share_of_batch16": shared_sq / (shared_sq + max(var - shared_sq, 0.0) / 16)}


def group_direction_test(G: torch.Tensor, a: torch.Tensor, n_perm: int = 4000, seed: int = 0) -> dict:
    """Cosine between the mean gradient of group ``a`` and of the rest, against random splits of the same sizes."""
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


def loo_transfer(D: torch.Tensor, Gx: torch.Tensor, ny: torch.Tensor, idx: list[int]) -> dict:
    """For each prompt j in idx: the mean x-gradient of the OTHER prompts against prompt j's y-gradient.

    D[i, j] = g_x,i . g_y,j. cos: cosine between the two. slope: first-order drop of
    prompt j's y-loss per unit step down the others' mean x-gradient.
    """
    n = Gx.shape[0]
    tot = Gx.sum()
    cos, slope = [], []
    for j in idx:
        dot = (D[:, j].sum() - D[j, j]) / (n - 1)
        sq = tot - 2 * Gx[j].sum() + Gx[j, j]          # |sum over i != j|^2
        nrm = sq.clamp(min=1e-30).sqrt() / (n - 1)
        cos.append(float(dot / (nrm * ny[j])))
        slope.append(float(dot / nrm))
    if not idx:
        return {"n": 0, "cos": float("nan"), "cos_se": float("nan"), "slope": float("nan"),
                "slope_se": float("nan"), "frac_positive": float("nan")}
    c, sl = torch.tensor(cos), torch.tensor(slope)
    k = len(idx)
    se = lambda x: float(x.std() / k ** 0.5) if k > 1 else float("nan")
    return {"n": k, "cos": float(c.mean()), "cos_se": se(c), "slope": float(sl.mean()), "slope_se": se(sl),
            "frac_positive": float((c > 0).float().mean())}


def analyze(pt_path: str, out_path: str) -> None:
    d = torch.load(pt_path, weights_only=False)
    cross, Nt, recs = d["cross"], d["norms"], d["records"]
    n = len(recs)
    G = {k: exact_gram(cross[f"{k}|{k}"], Nt[k]) for k in NAMES}
    miss = torch.tensor([not r["student_first_token_hit"] for r in recs])
    numstart = torch.tensor([not r["first_token_paren"] for r in recs])
    wrong_idx = [j for j in range(n) if miss[j]]
    right_idx = [j for j in range(n) if not miss[j]]
    wrong_num_idx = [j for j in range(n) if miss[j] and numstart[j]]
    mean = lambda xs: sum(xs) / max(len(xs), 1)
    summary = {"n": n, "cache_key": d.get("cache_key"), "student_first_token_acc": 1 - float(miss.float().mean()),
               "teacher_first_token_acc": mean([r["teacher_first_token_hit"] for r in recs]),
               "n_wrong": len(wrong_idx), "n_wrong_number_start": len(wrong_num_idx)}
    print(f"\n{n} prompts | first-token acc: student {summary['student_first_token_acc']:.3f}, teacher "
          f"{summary['teacher_first_token_acc']:.3f} | student wrong on {len(wrong_idx)} "
          f"({len(wrong_num_idx)} of them where the gold response starts with a number)")
    for row in ("LAST", "dla"):
        has = [r for r in recs if row in r["rows"]]
        tt = [r["teacher_top_is_gold"][r["rows"].index(row)] for r in has]
        ss = [r["student_top_is_gold"][r["rows"].index(row)] for r in has]
        sw = [r["student_top_is_gold"][r["rows"].index(row)] for r in has if not r["student_first_token_hit"]]
        tw = [r["teacher_top_is_gold"][r["rows"].index(row)] for r in has if not r["student_first_token_hit"]]
        summary[f"{row}_top_is_gold"] = {"teacher": mean(tt), "student": mean(ss), "teacher_on_student_wrong": mean(tw),
                                         "student_on_student_wrong": mean(sw)}
        print(f"  {row:4s} row: most-attributed number is the gold first operand: teacher {mean(tt):.3f}, student {mean(ss):.3f}"
              f" | where the student's first token is wrong: teacher {mean(tw):.3f}, student {mean(sw):.3f}")
    sketch_err = mean([float(((cross[f'{k}|{k}'].diagonal().clamp(min=0).sqrt() / Nt[k]) - 1).abs().mean()) for k in NAMES])
    print(f"  sketch norm vs exact norm, mean |rel err| over objectives: {sketch_err:.4f}")

    print("\n== 1. does the loss value predict a first-token miss? (AUC; 0.5 = no) ==")
    print(f"{'':10s} {'mean':>7s} {'mean ok':>8s} {'mean err':>8s} {'AUC all':>8s} {'se':>6s} {'AUC num-start':>13s} {'se':>6s}")
    lab = miss.tolist()
    sub = [j for j in range(n) if numstart[j]]
    for k in GRAPH_OBJ:
        v = [r["values"][k] for r in recs]
        a = auc(v, lab)
        a2 = auc([v[j] for j in sub], [lab[j] for j in sub])
        s = {"mean": mean(v), "mean_ok": mean([x for x, m in zip(v, lab) if not m]),
             "mean_err": mean([x for x, m in zip(v, lab) if m]),
             "auc": a, "auc_se": auc_se(a, sum(lab), n - sum(lab)), "auc_number_start": a2,
             "auc_number_start_se": auc_se(a2, sum(lab[j] for j in sub), len(sub) - sum(lab[j] for j in sub))}
        summary[f"value_{k}"] = s
        print(f"{k:10s} {s['mean']:7.4f} {s['mean_ok']:8.4f} {s['mean_err']:8.4f} {s['auc']:8.3f} {s['auc_se']:6.3f} "
              f"{s['auc_number_start']:13.3f} {s['auc_number_start_se']:6.3f}")

    print("\n== 2. gradient: is there a shared direction, and is it different where the student is wrong? ==")
    print(f"{'':10s} {'|g| prompt':>10s} {'|mean16|':>9s} {'pair cos':>9s} {'split-half':>10s} {'shared@16':>9s} | "
          f"{'wrong-vs-right cos':>18s} {'random splits':>22s} {'p':>6s}")
    for k in NAMES:
        c = shared_direction(G[k], Nt[k])
        t = group_direction_test(G[k], miss) if 2 <= len(wrong_idx) <= n - 2 else None
        summary[f"direction_{k}"] = {**c, "wrong_vs_right": t}
        tail = (f"{t['cos']:+18.3f} {t['random_split_mean']:+7.3f} [{t['random_split_q05_q95'][0]:+.2f},{t['random_split_q05_q95'][1]:+.2f}] "
                f"{t['p_lower']:6.3f}") if t else ""
        print(f"{k:10s} {c['norm_per_prompt']:10.3f} {c['norm_mean_of_16']:9.3f} {c['pairwise_cos']:+9.4f} "
              f"{c['split_half_cos']:+10.3f} {c['shared_share_of_batch16']:9.1%} | {tail}")

    print("\n== 3. does the candidate's average gradient (from the other prompts) fix a prompt's first token? ==")
    print("against that prompt's gold first-token cross-entropy gradient. cos: cosine; slope: drop in that loss per unit "
          "step; pos: share of prompts with cos > 0")
    print(f"{'':10s} | {'student wrong':^36s} | {'wrong, gold starts with a number':^36s} | {'student right':^17s}")
    hdr = f"{'cos':>8s} {'se':>6s} {'slope':>8s} {'se':>6s} {'pos':>4s}"
    print(f"{'':10s} | {hdr} | {hdr} | {'cos':>8s} {'slope':>8s}")
    for k in NAMES:
        D = cross[f"{k}|ce1"]
        tw = loo_transfer(D, G[k], Nt["ce1"], wrong_idx)
        tn_ = loo_transfer(D, G[k], Nt["ce1"], wrong_num_idx)
        tr = loo_transfer(D, G[k], Nt["ce1"], right_idx)
        summary[f"transfer_{k}"] = {"wrong": tw, "wrong_number_start": tn_, "right": tr}
        f = lambda t: f"{t['cos']:+8.4f} {t['cos_se']:6.4f} {t['slope']:+8.3f} {t['slope_se']:6.3f} {t['frac_positive']:4.2f}"
        print(f"{k:10s} | {f(tw)} | {f(tn_)} | {tr['cos']:+8.4f} {tr['slope']:+8.3f}")

    print("\nsame-prompt cosine of each candidate's gradient with the gold first-token gradient (ce1) and with KD (kl)")
    print(f"{'':10s} {'ce1, wrong':>18s} {'ce1, all':>18s} {'kl, wrong':>18s} {'kl, all':>18s}")
    for k in GRAPH_OBJ:
        row = {}
        for ref in ("ce1", "kl"):
            same = torch.diagonal(cross[f"{k}|{ref}"]) / (Nt[k] * Nt[ref])
            w = same[miss]
            row[ref] = {"wrong": float(w.mean()) if len(w) else float("nan"),
                        "wrong_se": float(w.std() / len(w) ** 0.5) if len(w) > 1 else float("nan"),
                        "all": float(same.mean()), "all_se": float(same.std() / n ** 0.5)}
        summary[f"same_prompt_{k}"] = row
        print(f"{k:10s} " + " ".join(f"{row[r][g]:+9.4f} +-{row[r][g + '_se']:6.4f}" for r in ("ce1", "kl") for g in ("wrong", "all")))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f_:
        json.dump({"args": d.get("args"), "summary": summary, "records": recs}, f_, indent=1)
    print(f"\nwrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teacher")
    ap.add_argument("--student")
    ap.add_argument("--dataset", default="svamp_eq")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--neuron-selection", choices=["norm", "influence"], default="influence")
    ap.add_argument("--teacher-prop", type=float, default=0.01)
    ap.add_argument("--student-prop", type=float, default=0.003)
    ap.add_argument("--nodes-per-label", type=int, default=10)
    ap.add_argument("--teacher-graph-batch-size", type=int, default=32)
    ap.add_argument("--student-graph-batch-size", type=int, default=8)
    ap.add_argument("--sketch-chunk", type=int, default=4096)
    ap.add_argument("--tag", default="sft-student")
    ap.add_argument("--out", default=None)
    ap.add_argument("--analyze-only", action="store_true", help="Recompute the report from the saved .pt; no GPU.")
    args = ap.parse_args()
    out_path = os.path.join(DIR_ROOT, args.out or f"FINAL_RESULTS/{args.dataset}/graph_target_gate_{args.tag}.json")
    pt_path = os.path.splitext(out_path)[0] + ".pt"
    if not args.analyze_only:
        if not args.teacher or not args.student:
            ap.error("--teacher and --student are required unless --analyze-only")
        collect(args, pt_path)
    analyze(pt_path, out_path)


if __name__ == "__main__":
    main()

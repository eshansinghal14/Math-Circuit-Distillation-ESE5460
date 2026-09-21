"""Is the token-path target learnable, and can any distance tell it from a wrong one?

The token-path term has now been run at lambda 0, 1 and 10, on two datasets, with
both row sets, and it has never improved accuracy. The lambda=10 scramble control
showed the target is not noise -- training on a scrambled target drives the
distance to the *real* target up -- so the question is no longer "is there
signal" but "is the loss shaped so a student can use it".

This script answers that offline. It builds the teacher and student token-path
profiles for N prompts, with no training and no double backward, and reports four
groups of statistics. Each one settles a specific decision.

Group 1, dynamic range. The distance from the student profile to the teacher one,
against three references: a within-prompt permutation of the teacher profile (the
scramble control), *another prompt* teacher profile (the harder and more
realistic control), and the mean profile over all prompts (what a student that
ignored the prompt would match). If the student already sits as close to the
right teacher profile as a different prompt profile does, there is no
prompt-specific range for the loss to exploit and no lambda or distance recovers
it.

Group 2, the distances themselves -- jsd, kld and rel-mse, at full width and at
several top-k widths. The number that matters is the *discrimination ratio*:
distance to a wrong target over distance to the right one. A distance that cannot
separate them cannot teach anything, however well conditioned it is. The gradient
norm is reported alongside, because the two trade off: kld discriminates best and
is also the one whose unboundedness produced a 48x swing in |g_graph|/|g_KL| in
training.

Group 3, structure of the top-k set. What fraction of the teacher mass the top-k
captures (is k large enough), how much the top-k *positions* agree across
different prompts (are they the generic BOS/recency prior rather than content),
and how much they agree across answer positions within one prompt (does the
per-position selection in _one_position_loss capture real position-dependence, or
is it just noise around one evidence set).

Group 4, evidence localisation. For extractive QA the answer is a span of the
context, so its token positions are known without any extra labels. If the
teacher attribution concentrates there and the student one does not, there is a
measured gap worth distilling and a metric that can move independently of exact
match. If the teacher does not concentrate there either, "distil where the
teacher looks" has nothing to transfer and every null result so far is explained.

Usage (from src/):
    python -m experiments.token_path_targets
        --model sft_models/Llama-3.2-1B-Instruct/hotpotqa/final_checkpoint
        --teacher sft_models/Meta-Llama-3-8B-Instruct/hotpotqa/final_checkpoint
        --dataset hotpotqa --n-prompts 300 --temperature 2.0
        --out results/token_path_targets
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DIR_ROOT, load_data, load_model, tokenize_prompt_answer  # noqa: E402
from training.utils import load_student  # noqa: E402
from graph_loss.hf_adapter import HFLlamaGraphAdapter  # noqa: E402
from graph_loss.loss import edge_similarity  # noqa: E402
from graph_loss.token_attribution import salient_logits  # noqa: E402
import graph_loss.training as GT  # noqa: E402
from graph_loss.training import (  # noqa: E402
    GraphAuxConfig,
    _graph_autocast,
    _position_profiles,
    _token_path_chunks,
    _top_token_view,
    scramble_teacher_rows,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DISTANCES = ("jsd", "kld", "rel-mse")
TOPKS = (0, 8, 16, 32, 64)
EPS = 1e-10


_PUNCT = set(""".,:;!?()[]{}<>"'`-/\\|*#&%$@~+= """)
_TEMPLATE_TOKENS = {"Q", "A"}


def structural_keep_mask(prompt_ids: torch.Tensor, tok: Any) -> torch.Tensor:
    """True at positions worth scoring; False at punctuation, template and specials.

    Group 3 of the first run found 31-53% of the teacher top-k slots were these:
    colon, period, 'A', BOS, newline-period, newline-question, 'Q', comma. They
    are also positions where teacher and student agree trivially -- both models
    attend to BOS -- so they inflate the free agreement and compress the
    distance's dynamic range while contributing nothing about which passage holds
    the answer.

    Masked positions are zeroed and the row renormalised rather than deleted, so
    every index still refers to the same prompt token and the answer-span mapping
    in Group 4 stays valid. A zeroed position can never enter a top-k.

    'Q' and 'A' are masked as the prompt template markers, which also masks them
    where they occur as ordinary words. On a context/question/answer prompt that
    is a handful of positions and worth the simplicity.
    """
    keep = torch.ones(int(prompt_ids.numel()), dtype=torch.bool)
    specials = {i for i in (getattr(tok, "bos_token_id", None),
                            getattr(tok, "eos_token_id", None),
                            getattr(tok, "pad_token_id", None)) if i is not None}
    for pos in range(int(prompt_ids.numel())):
        tid = int(prompt_ids[pos])
        if tid in specials:
            keep[pos] = False
            continue
        txt = tok.decode([tid])
        stripped = txt.strip()
        if stripped == "" or stripped in _TEMPLATE_TOKENS or all(c in _PUNCT for c in stripped):
            keep[pos] = False
    return keep


def apply_mask(profiles: list[torch.Tensor], keep: torch.Tensor) -> list[torch.Tensor]:
    """Zero the masked columns and renormalise each row."""
    out = []
    for prof in profiles:
        m = keep.to(prof.device).unsqueeze(0).to(prof.dtype)
        x = prof.abs() * m
        tot = x.sum(dim=1, keepdim=True)
        out.append(torch.where(tot > EPS, x / tot.clamp(min=EPS), prof.abs()))
    return out


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect(
    teacher_adapter: HFLlamaGraphAdapter,
    student_adapter: HFLlamaGraphAdapter,
    items: list[tuple[str, str]],
    config: GraphAuxConfig,
    micro_tokens: int,
    micro_batch: int,
    mask_structural: bool = True,
    drop_eos_position: bool = True,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Teacher and student token-path profiles for each prompt.

    Mirrors _token_path_batched exactly -- same tokenisation, same read
    positions, same teacher-chosen salient logits for both models, same prompt
    slice and renormalisation -- but with create_graph=False throughout, so it is
    a forward and one cheap backward per (position, row) rather than the double
    backward the trainer pays for.
    """
    tok = teacher_adapter.tokenizer
    rows = config.token_path_rows
    pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", 0) or 0

    seqs, p_lens, reads, golds, prompts = [], [], [], [], []
    for prompt, answer in items:
        p_ids, a_ids = tokenize_prompt_answer(tok, prompt, str(answer))
        seqs.append(torch.cat([p_ids, a_ids]).to(DEVICE))
        n_p = int(p_ids.numel())
        p_lens.append(n_p)
        golds.append(int(a_ids[0]))
        # The last answer position predicts EOS. "Which prompt token influences
        # the decision to stop" is not an evidence question, and on a 3-token
        # answer it is a third of the equally weighted mean over positions.
        n_read = int(a_ids.numel())
        if drop_eos_position and n_read > 1:
            n_read -= 1
        reads.append([n_p - 1 + k for k in range(n_read)])
        prompts.append(prompt)

    out: list[dict[str, Any]] = []
    chunks = _token_path_chunks([int(x.numel()) for x in seqs],
                                max_prompts=micro_batch, max_tokens=micro_tokens)
    done = 0
    for c, idxs in enumerate(chunks):
        if verbose and c % 10 == 0:
            print(f"  chunk {c + 1}/{len(chunks)} ({done} prompts done)")
        sub = [seqs[i] for i in idxs]
        L = max(int(x.numel()) for x in sub)
        ids = torch.full((len(sub), L), pad_id, dtype=torch.long, device=DEVICE)
        attn = torch.zeros((len(sub), L), dtype=torch.long, device=DEVICE)
        for r, x in enumerate(sub):
            ids[r, : x.numel()] = x
            attn[r, : x.numel()] = 1
        pl = [p_lens[i] for i in idxs]
        rd = [reads[i] for i in idxs]
        gd = [golds[i] for i in idxs]

        with torch.no_grad():
            dla_all = teacher_adapter.model(ids, attention_mask=attn).logits.detach()
        lids, wts = [], []
        for r in range(len(idxs)):
            li, pr = salient_logits(
                dla_all[r, pl[r] - 1].float(), config.top_k_logits, config.temperature,
                gold_token=gd[r] if rows == "gold" else None)
            lids.append(li.to(DEVICE))
            wts.append(pr.to(DEVICE))
        del dla_all

        t_prof = _position_profiles(
            teacher_adapter.model, ids, attn, pl, rd, lids, wts, gd, rows=rows,
            autocast=_graph_autocast(teacher_adapter, config))
        s_prof = _position_profiles(
            student_adapter.model, ids, attn, pl, rd, lids, wts, gd, rows=rows,
            autocast=_graph_autocast(student_adapter, config))

        for r, i in enumerate(idxs):
            ids_cpu = seqs[i][: pl[r]].cpu()
            t_rows = [x.detach().float().cpu() for x in t_prof[r]]
            s_rows = [x.detach().float().cpu() for x in s_prof[r]]
            keep = (structural_keep_mask(ids_cpu, tok) if mask_structural
                    else torch.ones(int(ids_cpu.numel()), dtype=torch.bool))
            if mask_structural:
                t_rows = apply_mask(t_rows, keep)
                s_rows = apply_mask(s_rows, keep)
            out.append({
                "prompt": prompts[i],
                "prompt_len": pl[r],
                "n_positions": len(rd[r]),
                "prompt_ids": ids_cpu,
                "keep": keep,
                "n_kept": int(keep.sum()),
                "teacher": t_rows,
                "student": s_rows,
            })
        done += len(idxs)
        del t_prof, s_prof
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def aggregate(profiles: list[torch.Tensor]) -> torch.Tensor:
    """One row per prompt: the teacher profile summed over answer positions."""
    acc = None
    for p in profiles:
        a = p.abs().mean(dim=0, keepdim=True)
        acc = a if acc is None else acc + a
    return acc if acc is not None else torch.ones(1, 1)


def mean_profile(recs: list[dict], width: int) -> torch.Tensor:
    """The across-prompt mean profile on a common width, renormalised.

    Prompts differ in length, so each profile is truncated to the shortest common
    width before averaging. This is what a student that ignored the prompt
    entirely would be matching.
    """
    acc, n = None, 0
    for r in recs:
        for p in r["teacher"]:
            if p.shape[1] < width:
                continue
            a = p[:, :width].abs().mean(dim=0, keepdim=True)
            acc = a if acc is None else acc + a
            n += 1
    if acc is None or n == 0:
        return torch.ones(1, width) / width
    acc = acc / n
    return acc / acc.sum().clamp(min=EPS)


def common_width(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Both rows truncated to their shared width and renormalised.

    Cross-prompt comparisons need a common support and prompts are not the same
    length. Truncation keeps the earliest positions, which for a QA prompt is the
    context rather than the question, so the comparison stays over the part of the
    input the target is supposed to be about.
    """
    w = min(int(a.shape[1]), int(b.shape[1]))
    a, b = a[:, :w].abs(), b[:, :w].abs()
    return (a / a.sum(dim=1, keepdim=True).clamp(min=EPS),
            b / b.sum(dim=1, keepdim=True).clamp(min=EPS))


def dist(t: torch.Tensor, s: torch.Tensor, kind: str, k: int) -> float:
    tr, sr = _top_token_view(t, s, k)
    return float(edge_similarity(tr, sr, kind))


def grad_norm(t: torch.Tensor, s: torch.Tensor, kind: str, k: int) -> float:
    sg = s.clone().requires_grad_(True)
    tr, sr = _top_token_view(t, sg, k)
    val = edge_similarity(tr, sr, kind)
    if not val.requires_grad:
        return 0.0
    val.backward()
    return float(sg.grad.norm()) if sg.grad is not None else 0.0


def topk_idx(p: torch.Tensor, k: int) -> set[int]:
    k = min(k, int(p.shape[1]))
    return set(p.abs().mean(dim=0).topk(k).indices.tolist())


# ---------------------------------------------------------------------------
# Groups 1 and 2: dynamic range and the distance comparison
# ---------------------------------------------------------------------------

def distance_table(recs: list[dict], config: GraphAuxConfig, rng: random.Random) -> dict:
    """Every distance against every reference, at each top-k width.

    References, in increasing order of how hard they are to beat:
      scrambled   -- the teacher own profile, positions permuted. The control the
                     training runs use.
      other       -- a different prompt teacher profile. Harder, because it is a
                     real attribution profile with the right general shape and
                     only the wrong content.
      mean        -- the across-prompt mean profile. What a prompt-independent
                     student matches, so the student distance to the teacher
                     should be clearly below its distance to this.
    """
    width = min(r["prompt_len"] for r in recs)
    mp = mean_profile(recs, width)
    res: dict = {}
    for k in TOPKS:
        for kind in DISTANCES:
            real, scram, other, meanr, gn = [], [], [], [], []
            for i, r in enumerate(recs):
                j = rng.randrange(len(recs))
                while j == i and len(recs) > 1:
                    j = rng.randrange(len(recs))
                for p, (t, s) in enumerate(zip(r["teacher"], r["student"])):
                    real.append(dist(t, s, kind, k))
                    gn.append(grad_norm(t, s, kind, k))
                    scram.append(dist(scramble_teacher_rows(t, config), s, kind, k))
                    tj = recs[j]["teacher"][min(p, len(recs[j]["teacher"]) - 1)]
                    a, b = common_width(tj, s)
                    other.append(dist(a, b, kind, k))
                    a2, b2 = common_width(mp, s)
                    meanr.append(dist(a2, b2, kind, k))
            m = lambda v: float(np.mean(v))  # noqa: E731
            res[f"k={k}|{kind}"] = {
                "real": m(real), "scrambled": m(scram), "other_prompt": m(other),
                "mean_profile": m(meanr), "grad_norm": m(gn),
                "ratio_scrambled": m(scram) / max(m(real), 1e-12),
                "ratio_other": m(other) / max(m(real), 1e-12),
                "ratio_mean": m(meanr) / max(m(real), 1e-12),
            }
    return res


# ---------------------------------------------------------------------------
# Group 3: structure of the top-k set
# ---------------------------------------------------------------------------

def topk_structure(recs: list[dict], tok: Any, rng: random.Random) -> dict:
    """Mass captured, agreement across prompts, agreement across answer positions."""
    res: dict = {}
    for k in (8, 16, 32, 64):
        mass, cross, within, at_bos, at_tail = [], [], [], [], []
        toks: Counter = Counter()
        skipped = 0
        for i, r in enumerate(recs):
            # A k at or above the prompt width selects every position, which
            # drives every overlap statistic to 1.0 and says nothing. Those
            # prompts are dropped rather than allowed to inflate the mean.
            if k >= r["prompt_len"]:
                skipped += 1
                continue
            agg = aggregate(r["teacher"])
            per_pos = [topk_idx(t, k) for t in r["teacher"]]
            idx = topk_idx(agg, k)
            mass.append(float(agg[0, list(idx)].sum()) / max(float(agg.sum()), EPS))
            n = r["prompt_len"]
            at_bos.append(float(0 in idx))
            at_tail.append(len([p for p in idx if p >= n - 5]) / max(len(idx), 1))
            ids = r["prompt_ids"]
            for p in idx:
                if p < ids.numel():
                    toks[tok.decode([int(ids[p])])] += 1
            if len(per_pos) > 1:
                pw = [len(a & b) / max(len(a | b), 1)
                      for x, a in enumerate(per_pos) for b in per_pos[x + 1:]]
                within.append(float(np.mean(pw)))
            j = rng.randrange(len(recs))
            while j == i and len(recs) > 1:
                j = rng.randrange(len(recs))
            if k >= recs[j]["prompt_len"]:
                continue
            idxj = topk_idx(aggregate(recs[j]["teacher"]), k)
            cross.append(len(idx & idxj) / max(len(idx | idxj), 1))
        if not mass:
            res[f"k={k}"] = {"skipped_prompts": skipped, "usable_prompts": 0}
            continue
        res[f"k={k}"] = {
            "usable_prompts": len(mass),
            "skipped_prompts": skipped,
            "teacher_mass_in_topk": float(np.mean(mass)),
            "cross_prompt_jaccard": float(np.mean(cross)),
            "within_prompt_position_jaccard": float(np.mean(within)) if within else None,
            "frac_selecting_bos": float(np.mean(at_bos)),
            "frac_slots_in_last_5_positions": float(np.mean(at_tail)),
            "most_common_tokens": toks.most_common(12),
        }
    return res


# ---------------------------------------------------------------------------
# Group 4: evidence localisation
# ---------------------------------------------------------------------------

def answer_span_mass(recs: list[dict], items: list[tuple[str, str]], tok: Any, k: int) -> dict:
    """How much attribution lands on the answer own tokens inside the context.

    For extractive QA the answer is a span of the passage, so its token positions
    are known with no extra labels -- no supporting-fact annotation needed. This
    is the premise of the whole method in one number: if the teacher does not put
    more mass there than chance, "distil where the teacher looks" has nothing to
    transfer.
    """
    t_mass, s_mass, base, t_top, s_top = [], [], [], [], []
    hits = 0
    for r, (prompt, answer) in zip(recs, items):
        a = str(answer).strip()
        low, al = prompt.lower(), a.lower()
        if not al or al not in low:
            continue
        start = low.index(al)
        try:
            enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=True)
            offs = enc["offset_mapping"]
        except Exception:
            continue
        span = {p for p, (s0, s1) in enumerate(offs)
                if s1 > start and s0 < start + len(a) and s1 > s0}
        # Chance is over the positions that survive masking, and a span token that
        # was itself masked (punctuation inside the answer) cannot be hit.
        keep = r["keep"]
        span = {p for p in span if p < r["prompt_len"] and bool(keep[p])}
        if not span:
            continue
        hits += 1
        agg_t = aggregate(r["teacher"])
        agg_s = aggregate(r["student"])
        idx = list(span)
        t_mass.append(float(agg_t[0, idx].sum()) / max(float(agg_t.sum()), EPS))
        s_mass.append(float(agg_s[0, idx].sum()) / max(float(agg_s.sum()), EPS))
        base.append(len(span) / max(r["n_kept"], 1))
        t_top.append(len(topk_idx(agg_t, k) & span) / max(len(span), 1))
        s_top.append(len(topk_idx(agg_s, k) & span) / max(len(span), 1))
    if not hits:
        return {"prompts_with_answer_in_context": 0}
    m = lambda v: float(np.mean(v))  # noqa: E731
    return {
        "prompts_with_answer_in_context": hits,
        "span_frac_of_prompt": m(base),
        "teacher_mass_on_answer_span": m(t_mass),
        "student_mass_on_answer_span": m(s_mass),
        "teacher_lift_over_chance": m(t_mass) / max(m(base), EPS),
        "student_lift_over_chance": m(s_mass) / max(m(base), EPS),
        f"teacher_top{k}_recall_of_span": m(t_top),
        f"student_top{k}_recall_of_span": m(s_top),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(res: dict) -> None:
    print()
    print("=" * 84)
    print("GROUPS 1 and 2   distances and discrimination")
    print("=" * 84)
    print("  ratio = distance to a WRONG target / distance to the right one.")
    print("  >1 means the distance separates them; ~1 means it cannot.")
    print("  r_oth (another prompt) is the one that matters; r_scr is the easy control.")
    print()
    print(f"  {'k':>5} {'dist':<9} {'real':>8} {'scram':>8} {'other':>8} {'mean':>8}"
          f" {'r_scr':>7} {'r_oth':>7} {'r_mean':>7} {'|grad|':>9}")
    for key, v in res["distances"].items():
        k, kind = key.split("|")
        label = "full" if k.split("=")[1] == "0" else k.split("=")[1]
        print(f"  {label:>5} {kind:<9} {v['real']:>8.4f} {v['scrambled']:>8.4f}"
              f" {v['other_prompt']:>8.4f} {v['mean_profile']:>8.4f}"
              f" {v['ratio_scrambled']:>7.2f} {v['ratio_other']:>7.2f}"
              f" {v['ratio_mean']:>7.2f} {v['grad_norm']:>9.3f}")

    print()
    print("=" * 84)
    print("GROUP 3   structure of the top-k set")
    print("=" * 84)
    for k, v in res["topk"].items():
        if not v.get("usable_prompts"):
            print(f"  {k}  skipped: k is at or above every prompt width")
            continue
        print(f"  {k}   ({v['usable_prompts']} prompts, {v['skipped_prompts']} too short)")
        print(f"    teacher mass inside the top-k      : {v['teacher_mass_in_topk']:.3f}")
        print(f"    cross-prompt overlap (Jaccard)     : {v['cross_prompt_jaccard']:.3f}"
              "   high => a generic positional prior")
        w = v["within_prompt_position_jaccard"]
        ws = "n/a" if w is None else format(w, ".3f")
        print(f"    across answer positions (Jaccard)  : {ws}"
              "   high => per-position selection buys nothing")
        print(f"    selects BOS                        : {v['frac_selecting_bos']:.3f}")
        print(f"    slots in the last 5 positions      : "
              f"{v['frac_slots_in_last_5_positions']:.3f}")
        print(f"    most common selected tokens        : "
              f"{[t for t, _ in v['most_common_tokens'][:8]]}")

    print()
    print("=" * 84)
    print("GROUP 4   evidence localisation (answer span inside the context)")
    print("=" * 84)
    g = res["answer_span"]
    if not g.get("prompts_with_answer_in_context"):
        print("  no prompt had its answer as a literal span of the context; skipped.")
        return
    print(f"  prompts usable                       : {g['prompts_with_answer_in_context']}")
    print(f"  span as a fraction of the prompt     : {g['span_frac_of_prompt']:.4f}   (chance)")
    print(f"  teacher mass on the span             : {g['teacher_mass_on_answer_span']:.4f}"
          f"   ({g['teacher_lift_over_chance']:.2f}x chance)")
    print(f"  student mass on the span             : {g['student_mass_on_answer_span']:.4f}"
          f"   ({g['student_lift_over_chance']:.2f}x chance)")
    for key, val in g.items():
        if "recall_of_span" in key:
            print(f"  {key:<37}: {val:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train", choices=("train", "test"))
    ap.add_argument("--n-prompts", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--top-k-logits", type=float, default=0.95)
    ap.add_argument("--token-path-rows", default="weighted",
                    choices=("weighted", "gold", "all"))
    ap.add_argument("--micro-tokens", type=int, default=2048)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--mask-structural", action=argparse.BooleanOptionalAction, default=True,
                    help="Zero the punctuation, prompt-template ('Q', 'A', ':') and special-token "
                         "positions and renormalise. On by default: they were 31-53%% of the "
                         "teacher top-k in the first run and are positions where teacher and "
                         "student agree trivially, so they compress the distance's range while "
                         "saying nothing about which passage holds the answer. "
                         "--no-mask-structural restores the full profile.")
    ap.add_argument("--drop-eos-position", action=argparse.BooleanOptionalAction, default=True,
                    help="Drop the final answer position, the one predicting EOS, from the "
                         "equally weighted mean over answer positions. On by default: it is not "
                         "an evidence question and on a 3-token answer it is a third of the loss. "
                         "--no-drop-eos-position keeps it.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/token_path_targets")
    args = ap.parse_args()

    GT._TOPK_LOG_CALLS[0] = 10 ** 9   # the per-call top-k log is noise here
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    train, test = load_data(args.dataset)
    data = train if args.split == "train" else test
    items = list(data.items())[: args.n_prompts]
    print(f"{len(items)} prompts from {args.dataset}/{args.split}")

    student, tokenizer = load_student(args.model)
    teacher, _ = load_model(args.teacher)
    student.eval()
    teacher.eval()
    sa = HFLlamaGraphAdapter(student, tokenizer, DEVICE)
    ta = HFLlamaGraphAdapter(teacher, tokenizer, DEVICE)

    config = GraphAuxConfig(
        graph_loss_type="jsd",
        token_path_rows=args.token_path_rows,
        supergraph_aggregation="token-path",
        top_k_logits=args.top_k_logits,
        temperature=args.temperature,
        scramble_teacher_graph=True,
    )
    config.scramble_seed = args.seed
    config.scramble_permutations = {}

    print("collecting profiles (teacher and student, no create_graph)...")
    recs = collect(ta, sa, items, config, args.micro_tokens, args.micro_batch,
                   mask_structural=args.mask_structural,
                   drop_eos_position=args.drop_eos_position)
    kept = float(np.mean([r["n_kept"] / max(r["prompt_len"], 1) for r in recs]))
    print(f"collected {len(recs)} prompts, "
          f"{sum(r['n_positions'] for r in recs)} answer positions")
    print(f"  mask_structural={args.mask_structural} -> {kept:.1%} of prompt positions kept")
    print(f"  drop_eos_position={args.drop_eos_position}")

    res = {
        "config": vars(args),
        "n_prompts": len(recs),
        "n_positions": sum(r["n_positions"] for r in recs),
        "prompt_len": {
            "min": min(r["prompt_len"] for r in recs),
            "median": float(np.median([r["prompt_len"] for r in recs])),
            "max": max(r["prompt_len"] for r in recs),
        },
        "mask_structural": args.mask_structural,
        "drop_eos_position": args.drop_eos_position,
        "frac_positions_kept": kept,
        "distances": distance_table(recs, config, rng),
        "topk": topk_structure(recs, tokenizer, rng),
        "answer_span": answer_span_mass(recs, items, tokenizer, k=16),
    }
    report(res)

    out_dir = args.out if os.path.isabs(args.out) else os.path.join(DIR_ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    tag = ("masked" if args.mask_structural else "full")
    tag += "-noeos" if args.drop_eos_position else "-eos"
    path = os.path.join(out_dir, f"{args.dataset}_{args.token_path_rows}_{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

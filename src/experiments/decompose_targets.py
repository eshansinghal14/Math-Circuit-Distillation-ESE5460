"""Is the teacher's supernode target a circuit, or a constant?

The graph term has never moved accuracy, at any weight, under any aggregation or
distance. ``inspect_graphs`` suggested why on 12 prompt pairs: a *different*
prompt's teacher target scores about as well as the correct one. This script
settles that on every prompt in the teacher-target cache, offline, with no model
loaded -- the cache already holds one target matrix per prompt, and the teacher
is frozen, so its targets are a fixed function of the prompt.

It decomposes the across-prompt variance of the cached target matrices into

    target(prompt) = grand mean + condition effects + per-prompt residual

and -- the part that matters -- asks how well each of those parts *predicts a
held-out prompt's target*, in the same relative-squared-error units the trainer
minimises (``--graph-loss-type rel-mse``: ||pred - T||^2 / ||T||^2, so 0 is
exact, 1 is an empty matrix).

Predictors, all scored out-of-fold on 5-fold splits over prompts:

  * ``constant``     -- the grand mean of the training folds. What a student that
                        ignored the prompt entirely would achieve.
  * ``cond:<factor>``-- the mean target of the prompt's cell, per factor. The
                        factors include the six ANOVA categories the supernodes
                        are *labelled* with (arg1 units, sum range, ...), so a
                        target that does not vary with them is one whose
                        supernodes do not do what their labels claim.
  * ``probe``        -- ridge regression of the flattened target on cheap
                        arithmetic features (operand and sum digits, carries,
                        magnitudes), ridge strength picked on an inner split.
                        Catches prompt structure the named conditions miss.
  * ``other_prompt`` -- a randomly chosen other prompt's target. The
                        shuffled-prompt control from ``inspect_graphs``, now over
                        every available pair instead of 12.

How to read the result. ``constant`` is the bar to beat: it is the error of a
target with no prompt information at all. If ``cond:*`` and ``probe`` do not beat
it by a clear margin, then the per-prompt teacher graph carries no per-prompt
signal, the loss cannot teach circuit structure, and no distance function or
lambda fixes it. Pass ``--reference-loss`` (the training run's logged ``Graph=``
value) to print the student's actual distance on the same axis: if the student
sits *above* the constant predictor, the loss is not even asking it for
prompt-specific structure yet.

Usage (from src/):
    python -m experiments.decompose_targets \\
        --cache cache/teacher_targets/8578ee41689722ff.pt \\
        --reference-loss 0.28 --out results/decompose_targets

    python -m experiments.decompose_targets --list     # what is in the cache dir
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:  # keeps the script importable off-Drive, where utils' data root does not exist
    from utils import DIR_ROOT  # noqa: E402
except Exception:  # pragma: no cover
    DIR_ROOT = ""

EPS = 1e-12
TAGS = ("carry_units", "sum_ge_100", "single_digit")


# ---------------------------------------------------------------------------
# Prompt features
# ---------------------------------------------------------------------------

def parse_prompt(prompt: str) -> list[int] | None:
    """Operands of an N-ary addition prompt ('12+34=' or '12+34+56='), else None.

    Mirrors ``inspect_graphs.parse_prompt`` so both scripts bucket identically.
    """
    m = re.fullmatch(r"\s*(\d+(?:\s*\+\s*\d+)+)\s*=\s*", prompt)
    return [int(x) for x in re.findall(r"\d+", m.group(1))] if m else None


def factors(prompt: str, n_ops: int) -> dict[str, Any] | None:
    """Categorical factors for one prompt: the three structural tags plus the six
    ANOVA categories the supernodes are named after. ``None`` if unparseable."""
    args = parse_prompt(prompt)
    if args is None or len(args) != n_ops:
        return None
    total = sum(args)
    out: dict[str, Any] = {
        "carry_units": sum(a % 10 for a in args) >= 10,
        "sum_ge_100": total >= 100,
        "single_digit": any(a < 10 for a in args),
        "sum units": total % 10,
        "sum range": total // 10,
    }
    for i, a in enumerate(args, 1):
        out[f"arg{i} units"] = a % 10
        out[f"arg{i} range"] = a // 10
    return out


def feature_vector(prompt: str, n_ops: int) -> np.ndarray | None:
    """Ridge-probe features: per-operand and sum digit one-hots, carries, magnitudes."""
    args = parse_prompt(prompt)
    if args is None or len(args) != n_ops:
        return None
    total = sum(args)
    eye = np.eye(10)
    f: list[float] = [1.0]
    for a in args:
        f += list(eye[a % 10])                    # units digit
        f += list(eye[min(a // 10, 9)])           # tens digit, clipped
        f.append(a / 100.0)
    f += list(eye[total % 10])                    # sum units
    f += list(eye[min((total // 10) % 10, 9)])    # sum tens
    f.append(total / 100.0)
    f.append(float(total >= 100))
    f.append(float(sum(a % 10 for a in args) >= 10))
    f.append(float(any(a < 10 for a in args)))
    f.append(sum(1 for a in args if a % 10 >= 5) / len(args))
    return np.asarray(f, dtype=np.float64)


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def load_cache(path: str) -> tuple[dict[str, dict], dict[str, Any]]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "entries" in data and "__meta__" in data:
        return dict(data["entries"]), dict(data["__meta__"])
    return dict(data), {}


def resolve_cache(arg: str | None, cache_dir: str) -> str:
    """Accept a full path, a bare cache key, or nothing (largest file in the dir)."""
    if arg and os.path.isfile(arg):
        return arg
    if arg:
        cand = os.path.join(cache_dir, arg if arg.endswith(".pt") else f"{arg}.pt")
        if os.path.isfile(cand):
            return cand
        raise SystemExit(f"no cache file at {arg!r} or {cand!r}")
    files = [f for f in glob.glob(os.path.join(cache_dir, "*.pt")) if ".stale-" not in f]
    if not files:
        raise SystemExit(f"no teacher-target cache files in {cache_dir}")
    return max(files, key=os.path.getsize)


def list_caches(cache_dir: str) -> None:
    files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
    if not files:
        print(f"no cache files in {cache_dir}")
        return
    print(f"{cache_dir}:")
    for f in files:
        try:
            entries, meta = load_cache(f)
            shapes = sorted({tuple(e["adj"].shape) for e in entries.values() if e.get("adj") is not None})
            print(f"  {os.path.basename(f):<24} {len(entries):>6} prompts | shapes {shapes}"
                  f" | code {meta.get('code_digest', 'unknown')}")
        except Exception as e:  # a half-written file, or one from an older version
            print(f"  {os.path.basename(f):<24} unreadable ({e})")


def audit(entries: dict[str, dict]) -> dict[str, Any]:
    """Integrity checks on the cache itself, before any of its contents are believed.

    A degenerate cache -- one target written for every prompt, or all-zero
    matrices -- would look exactly like the research finding this script is
    testing for, so it has to be ruled out first. The stored answer is also
    re-derived from the prompt: an answer that does not match the arithmetic
    means the target was built against the wrong row.
    """
    n = len(entries)
    digests: dict[bytes, int] = {}
    member_sets: set[tuple] = set()
    zero_rows = n_nonfinite = n_no_logit_ids = 0
    answer_checked = answer_wrong = 0
    absmax: list[float] = []
    wrong_examples: list[str] = []

    for prompt, e in entries.items():
        adj = e.get("adj")
        if adj is None:
            continue
        a = adj.detach().to(torch.float64)
        if not torch.isfinite(a).all():
            n_nonfinite += 1
            continue
        d = a.numpy().tobytes()
        digests[d] = digests.get(d, 0) + 1
        member_sets.add(tuple(tuple(sn) for sn in e.get("supernodes", [])))
        if float(a.abs().max()) == 0.0:
            zero_rows += 1
        absmax.append(float(a.abs().max()))
        if e.get("logit_ids") is None:
            n_no_logit_ids += 1
        args = parse_prompt(prompt)
        stored = e.get("answer")
        if args is not None and stored is not None:
            answer_checked += 1
            try:
                if int(str(stored)) != sum(args):
                    answer_wrong += 1
                    if len(wrong_examples) < 5:
                        wrong_examples.append(f"{prompt!r} -> stored {stored!r}, expected {sum(args)}")
            except ValueError:
                answer_wrong += 1
                if len(wrong_examples) < 5:
                    wrong_examples.append(f"{prompt!r} -> stored {stored!r}, not an integer")

    n_scored = sum(digests.values())
    duplicated = n_scored - len(digests)
    return {
        "n_entries": n,
        "n_distinct_matrices": len(digests),
        "n_duplicate_matrices": duplicated,
        "largest_duplicate_group": max(digests.values()) if digests else 0,
        "n_distinct_supernode_member_sets": len(member_sets),
        "n_all_zero": zero_rows,
        "n_nonfinite": n_nonfinite,
        "n_missing_logit_ids": n_no_logit_ids,
        "answers_checked": answer_checked,
        "answers_wrong": answer_wrong,
        "answer_mismatch_examples": wrong_examples,
        "abs_max_median": float(np.median(absmax)) if absmax else None,
    }


def report_audit(a: dict[str, Any]) -> list[str]:
    """Print the audit and return the problems that should stop interpretation."""
    print("\nCache audit:")
    print(f"  entries {a['n_entries']} | distinct matrices {a['n_distinct_matrices']}"
          f" | duplicates {a['n_duplicate_matrices']} (largest group {a['largest_duplicate_group']})")
    scale = "n/a" if a["abs_max_median"] is None else f"{a['abs_max_median']:.4g}"
    print(f"  distinct supernode member sets {a['n_distinct_supernode_member_sets']}"
          f" | all-zero {a['n_all_zero']} | non-finite {a['n_nonfinite']}"
          f" | median max|edge| {scale}")
    print(f"  stored answers re-derived from the prompt: {a['answers_checked']} checked, "
          f"{a['answers_wrong']} wrong")
    problems = []
    if a["n_distinct_matrices"] <= 1 and a["n_entries"] > 1:
        problems.append("every cached target is the same matrix -- the cache is degenerate, not the target")
    if a["largest_duplicate_group"] > max(2, 0.01 * max(a["n_entries"], 1)):
        problems.append(f"{a['largest_duplicate_group']} prompts share one identical matrix")
    if a["n_all_zero"]:
        problems.append(f"{a['n_all_zero']} targets are all zero")
    if a["answers_wrong"]:
        problems.append(f"{a['answers_wrong']} stored answers do not match their prompt: "
                        + "; ".join(a["answer_mismatch_examples"]))
    if a["n_distinct_supernode_member_sets"] == 0:
        problems.append("no supernode membership recorded")
    for p in problems:
        print(f"  PROBLEM: {p}")
    if not problems:
        print("  no integrity problems found")
    return problems


def collect(entries: dict[str, dict]) -> tuple[list[str], np.ndarray, tuple, tuple[int, int], int]:
    """Largest group of prompts sharing supernode labels and matrix shape.

    Returns (prompts, Y of shape [n, D], labels, shape, n_operands). Matrices are
    flattened; prompts with non-finite entries or the wrong operand count drop out.
    """
    groups: dict[tuple, list[str]] = {}
    for prompt, e in entries.items():
        adj = e.get("adj")
        if adj is None:
            continue
        labels = tuple(tuple(l) for l in e.get("labels", []))
        groups.setdefault((labels, tuple(adj.shape)), []).append(prompt)
    if not groups:
        raise SystemExit("cache holds no usable adjacency matrices")
    for (labels, shape), prompts in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        names = [l[0] if l else "?" for l in labels]
        print(f"group: {len(prompts):>6} prompts | shape {shape} | {len(labels)} supernodes {names}")
    (labels, shape), prompts = max(groups.items(), key=lambda kv: len(kv[1]))

    counts: dict[int, int] = {}
    for p in prompts:
        args = parse_prompt(p)
        if args:
            counts[len(args)] = counts.get(len(args), 0) + 1
    if not counts:
        raise SystemExit("no prompts in the largest group parse as N-ary addition")
    n_ops = max(counts, key=counts.get)

    keep, rows = [], []
    for p in sorted(prompts):
        args = parse_prompt(p)
        if args is None or len(args) != n_ops:
            continue
        y = entries[p]["adj"].detach().to(torch.float64).reshape(-1).numpy()
        if not np.isfinite(y).all():
            continue
        keep.append(p)
        rows.append(y)
    if len(keep) < 50:
        raise SystemExit(f"only {len(keep)} usable prompts; need at least 50")
    return keep, np.stack(rows), labels, shape, n_ops


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def rel_mse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-row ||pred - target||^2 / ||target||^2 -- the trainer's rel-mse loss."""
    num = ((pred - target) ** 2).sum(axis=1)
    den = np.maximum((target ** 2).sum(axis=1), EPS)
    return num / den


def cosine(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    num = (pred * target).sum(axis=1)
    den = np.maximum(np.linalg.norm(pred, axis=1) * np.linalg.norm(target, axis=1), EPS)
    return num / den


def eta_squared(Y: np.ndarray, cells: list[Any]) -> tuple[float, int]:
    """Share of total squared deviation explained by a cell partition, and #cells."""
    grand = Y.mean(axis=0)
    ss_total = float(((Y - grand) ** 2).sum())
    if ss_total <= EPS:
        return 0.0, 0
    keyed = list(map(str, cells))
    index: dict[str, list[int]] = {}
    for i, k in enumerate(keyed):
        index.setdefault(k, []).append(i)
    ss_between = 0.0
    for idx in index.values():
        ss_between += len(idx) * float(((Y[idx].mean(axis=0) - grand) ** 2).sum())
    return ss_between / ss_total, len(index)


def ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ridge on centred data: returns (W, x_mean, y_mean), predicting (X - x_mean) @ W + y_mean.

    Centring keeps the penalty off the intercept, so a large alpha shrinks the
    probe onto the training mean -- exactly the ``constant`` predictor -- rather
    than onto zero. Without it the probe can only ever overfit, because every
    alpha big enough to regularise also destroys the prediction's scale.
    """
    xm, ym = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - xm, Y - ym
    A = Xc.T @ Xc + alpha * np.eye(Xc.shape[1])
    return np.linalg.solve(A, Xc.T @ Yc), xm, ym


def probe_predict(fit: tuple[np.ndarray, np.ndarray, np.ndarray], X: np.ndarray) -> np.ndarray:
    W, xm, ym = fit
    return (X - xm) @ W + ym


def fit_probe(Xtr: np.ndarray, Ytr: np.ndarray, alphas: list[float],
              rng: np.random.Generator) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], float]:
    """Ridge with the strength chosen on an inner 80/20 split of the training fold."""
    n = len(Xtr)
    perm = rng.permutation(n)
    cut = int(0.8 * n)
    itr, iva = perm[:cut], perm[cut:]
    if len(iva) < 5 or len(itr) < 5:
        itr = iva = perm
    best, best_alpha = np.inf, alphas[0]
    for a in alphas:
        fit = ridge_fit(Xtr[itr], Ytr[itr], a)
        err = float(rel_mse(probe_predict(fit, Xtr[iva]), Ytr[iva]).mean())
        if err < best:
            best, best_alpha = err, a
    return ridge_fit(Xtr, Ytr, best_alpha), best_alpha


def kfold(Y: np.ndarray, X: np.ndarray, factor_cells: dict[str, list[Any]],
          n_folds: int, seed: int) -> dict[str, dict[str, Any]]:
    """Out-of-fold rel_mse and cosine for every predictor.

    Every cell mean, the grand mean and the ridge weights are fitted on the
    training folds only; a test prompt whose cell is unseen falls back to the
    training grand mean, so no predictor can see its own target.
    """
    rng = np.random.default_rng(seed)
    n = len(Y)
    folds = np.array_split(rng.permutation(n), n_folds)
    crossed = ["|".join(str(factor_cells[t][i]) for t in TAGS if t in factor_cells) for i in range(n)]
    cells_all = dict(factor_cells)
    cells_all["tags_crossed"] = crossed

    names = ["constant"] + [f"cond:{k}" for k in cells_all] + ["probe", "other_prompt"]
    acc: dict[str, list[np.ndarray]] = {k: [] for k in names}
    acc_cos: dict[str, list[np.ndarray]] = {k: [] for k in names}
    # The top of the grid shrinks the probe onto the training mean, so an
    # uninformative feature set degrades to ``constant`` instead of overfitting.
    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5]
    chosen_alphas: list[float] = []

    for f in range(n_folds):
        te = folds[f]
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != f])
        Yte = Y[te]
        grand = Y[tr].mean(axis=0)

        pred = np.repeat(grand[None, :], len(te), axis=0)
        acc["constant"].append(rel_mse(pred, Yte))
        acc_cos["constant"].append(cosine(pred, Yte))

        for name, cells in cells_all.items():
            keyed = list(map(str, cells))
            members: dict[str, list[int]] = {}
            for i in tr:
                members.setdefault(keyed[i], []).append(int(i))
            table = {c: Y[idx].mean(axis=0) for c, idx in members.items()}
            pred = np.stack([table.get(keyed[i], grand) for i in te])
            acc[f"cond:{name}"].append(rel_mse(pred, Yte))
            acc_cos[f"cond:{name}"].append(cosine(pred, Yte))

        fit, a = fit_probe(X[tr], Y[tr], alphas, rng)
        chosen_alphas.append(a)
        pred = probe_predict(fit, X[te])
        acc["probe"].append(rel_mse(pred, Yte))
        acc_cos["probe"].append(cosine(pred, Yte))

        other = tr[rng.integers(0, len(tr), size=len(te))]
        acc["other_prompt"].append(rel_mse(Y[other], Yte))
        acc_cos["other_prompt"].append(cosine(Y[other], Yte))

    out: dict[str, dict[str, Any]] = {}
    for k in names:
        v = np.concatenate(acc[k])
        c = np.concatenate(acc_cos[k])
        out[k] = {"rel_mse": float(v.mean()), "rel_mse_sd": float(v.std(ddof=1)),
                  "cos": float(c.mean()), "n": int(v.size)}
    out["probe"]["ridge_alphas"] = chosen_alphas
    return out


# ---------------------------------------------------------------------------
# Entry-level variance map
# ---------------------------------------------------------------------------

def variance_map(Y: np.ndarray, labels: tuple, shape: tuple[int, int], top: int = 12) -> dict[str, Any]:
    """Where the across-prompt variance sits, entry by entry, as a share of the total."""
    grand = Y.mean(axis=0)
    var = Y.var(axis=0)
    total = float(var.sum())
    rows, cols = shape
    names = [l[0] if l else f"sn{i}" for i, l in enumerate(labels)]
    names += [f"sn{i}" for i in range(len(names), rows)]
    col_names = list(names[:rows]) + [f"tok{j}" for j in range(max(cols - rows, 0))]

    entries = []
    for k in np.argsort(-var)[:top]:
        r, c = divmod(int(k), cols)
        entries.append({
            "target": names[r] if r < len(names) else f"sn{r}",
            "source": col_names[c] if c < len(col_names) else f"col{c}",
            "mean": float(grand[k]), "sd": float(np.sqrt(var[k])),
            "cv": float(np.sqrt(var[k]) / max(abs(grand[k]), EPS)),
            "share_of_variance": float(var[k] / max(total, EPS)),
        })
    return {
        "total_variance": total,
        "total_mean_sq": float((grand ** 2).sum()),
        "variance_over_mean_sq": float(total / max(float((grand ** 2).sum()), EPS)),
        "top_entries": entries,
    }


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cache", type=str, default=None,
                   help="Cache file path or bare key; default is the largest file in --cache-dir.")
    p.add_argument("--cache-dir", type=str, default="cache/teacher_targets", dest="cache_dir",
                   help="Searched when --cache is a bare key or absent (relative to the data root).")
    p.add_argument("--list", action="store_true", help="List the cache files with prompt counts and exit.")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top-entries", type=int, default=12, dest="top_entries")
    p.add_argument("--ignore-audit", action="store_true", dest="ignore_audit",
                   help="Run the decomposition even though the cache audit found problems.")
    p.add_argument("--reference-loss", type=float, default=None, dest="reference_loss",
                   help="A training run's logged Graph= value, printed on the same axis for comparison.")
    p.add_argument("--beat-threshold", type=float, default=0.10, dest="beat_threshold",
                   help="Fraction of the constant predictor's error that a prompt-aware predictor "
                        "must remove for the target to count as carrying prompt information.")
    p.add_argument("--out", type=str, default=None,
                   help="Directory for decompose_targets.json (default: alongside the cache file).")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cache_dir = args.cache_dir
    if not os.path.isabs(cache_dir):
        cache_dir = os.path.join(DIR_ROOT, cache_dir)
    if args.list:
        list_caches(cache_dir)
        return

    path = resolve_cache(args.cache, cache_dir)
    entries, meta = load_cache(path)
    print(f"Teacher target cache: {path} ({len(entries)} prompts, code {meta.get('code_digest', 'unknown')})")

    audit_result = audit(entries)
    problems = report_audit(audit_result)
    if problems and not args.ignore_audit:
        print("\nStopping: fix the cache before reading anything into the decomposition, or re-run\n"
              "with --ignore-audit if you have decided these are benign.")
        raise SystemExit(2)

    prompts, Y, labels, shape, n_ops = collect(entries)
    print(f"using {len(prompts)} prompts | {n_ops} operands | matrix {shape[0]}x{shape[1]} = {Y.shape[1]} entries")

    facs = [factors(p, n_ops) for p in prompts]
    feats = [feature_vector(p, n_ops) for p in prompts]
    ok = [i for i, (f, x) in enumerate(zip(facs, feats)) if f is not None and x is not None]
    prompts = [prompts[i] for i in ok]
    Y = Y[ok]
    facs = [facs[i] for i in ok]
    X = np.stack([feats[i] for i in ok])
    factor_cells = {k: [f[k] for f in facs] for k in facs[0]}

    print("\nIn-sample variance explained (eta^2), by factor:")
    etas: dict[str, dict[str, Any]] = {}
    crossed = ["|".join(str(factor_cells[t][i]) for t in TAGS if t in factor_cells) for i in range(len(Y))]
    for k, cells in list(factor_cells.items()) + [("tags_crossed", crossed)]:
        e, n_cells = eta_squared(Y, cells)
        etas[k] = {"eta_sq": e, "n_cells": n_cells}
        print(f"  {k:<16} {e:6.3f}  ({n_cells} cells)")

    vmap = variance_map(Y, labels, shape, top=args.top_entries)
    print(f"\nacross-prompt variance / squared mean: {vmap['variance_over_mean_sq']:.4f}"
          "   (<< 1 means the matrix is nearly the same for every prompt)")
    print("  entries carrying the most variance:")
    for e in vmap["top_entries"][:6]:
        print(f"    {e['target']:<12} <- {e['source']:<12} mean {e['mean']:+.4f} sd {e['sd']:.4f}"
              f" | {e['share_of_variance']:5.1%} of variance")

    print(f"\nOut-of-fold prediction of a held-out prompt's target ({args.folds} folds), "
          "in the trainer's rel-mse units:")
    res = kfold(Y, X, factor_cells, args.folds, args.seed)
    base = res["constant"]["rel_mse"]
    print(f"  {'predictor':<24} {'rel_mse':>9} {'sd':>8} {'cos':>7}   vs constant")
    for k in sorted(res, key=lambda k: res[k]["rel_mse"]):
        r = res[k]
        gain = (base - r["rel_mse"]) / max(base, EPS)
        mark = "" if k == "constant" else f"  {gain:+6.1%}"
        print(f"  {k:<24} {r['rel_mse']:9.4f} {r['rel_mse_sd']:8.4f} {r['cos']:7.3f}{mark}")

    prompt_aware = {k: v["rel_mse"] for k, v in res.items() if k not in ("constant", "other_prompt")}
    best_name = min(prompt_aware, key=prompt_aware.get)
    best = prompt_aware[best_name]
    gain = (base - best) / max(base, EPS)

    print("\n" + "-" * 70)
    print(f"constant predictor          : {base:.4f}")
    print(f"best prompt-aware predictor : {best:.4f}  ({best_name}, {gain:+.1%} vs constant)")
    print(f"another prompt's target     : {res['other_prompt']['rel_mse']:.4f}")
    if args.reference_loss is not None:
        print(f"student in training         : {args.reference_loss:.4f}  (--reference-loss)")
        if args.reference_loss > base:
            print("  the student is further from the teacher than a constant matrix is:")
            print("  the loss is not yet asking it for anything prompt-specific.")
    verdict = "prompt-dependent" if gain >= args.beat_threshold else "effectively constant"
    print(f"\nverdict: the teacher target is {verdict.upper()} "
          f"(threshold {args.beat_threshold:.0%} of the constant predictor's error).")
    if gain < args.beat_threshold:
        print("  No per-prompt circuit is being distilled. A different distance or lambda")
        print("  cannot change this; the supernode aggregation itself is the constraint.")
    print("-" * 70)

    out_dir = args.out or os.path.dirname(path)
    if out_dir and not os.path.isabs(out_dir):
        out_dir = os.path.join(DIR_ROOT, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "cache": path, "code_digest": meta.get("code_digest"),
        "n_prompts": len(prompts), "n_operands": n_ops, "matrix_shape": list(shape),
        "supernode_labels": [list(l) for l in labels],
        "audit": audit_result, "audit_problems": problems,
        "eta_squared": etas, "variance_map": vmap, "out_of_fold": res,
        "constant": base,
        "best_prompt_aware": {"name": best_name, "rel_mse": best, "gain_vs_constant": gain},
        "other_prompt": res["other_prompt"]["rel_mse"],
        "reference_loss": args.reference_loss, "beat_threshold": args.beat_threshold,
        "verdict": verdict,
    }
    out_path = os.path.join(out_dir, "decompose_targets.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

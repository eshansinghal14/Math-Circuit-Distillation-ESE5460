"""Render the lambda-sweep figure and table rows for the paper appendix.

Reads every graph-KD history in the lambda study directory that carries a
``config`` record (the fp32-regime runs; bf16-era histories have no such key and
are skipped with a notice), orders them by ``lambda_graph``, and treats a history
whose config has no ``lambda_graph`` as the standard-KD control (lambda = 0).

Produces ``lambda_sweep.pdf``, three panels: in-distribution and mean
out-of-distribution accuracy against training step, and peak mean
out-of-distribution accuracy against lambda on a log axis, the conventional
sensitivity view. Also prints the LaTeX table rows, so the numbers in the
appendix are generated from the histories rather than transcribed.

Encoding: lambda is ordered, so the runs take an ordinal ramp of one hue (light
to dark with increasing lambda) rather than unrelated categorical hues, and the
control is a dashed grey reference, clipped to the graph runs' last step so all
share an axis. Every run is direct-labelled at its right-hand end.

Usage (from the repository root):
    PYTHONPATH=src python -m experiments.plot_lambda_sweep
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OOD = ["222_add", "2222_add", "33_add", "21_mult"]

# Ordinal ramp, one hue, light -> dark with increasing lambda: steps 250 / 450 /
# 650 of the reference blue ramp, the lightest of which still clears 2:1 on white.
RAMP = ["#86b6ef", "#2a78d6", "#104281"]
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8983"
GRID = "#e9e8e5"
RULE_UNTRAINED = dict(color=INK_3, linewidth=0.9, linestyle=(0, (2, 2)), zorder=1)
RULE_TEACHER = dict(color=INK_3, linewidth=0.9, linestyle=(0, (1, 2)), zorder=1)
CONTROL = dict(color=INK_2, linewidth=1.2, linestyle=(0, (4, 2)), dash_capstyle="round", zorder=2)


def _style(ax) -> None:
    ax.set_facecolor("white")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, solid_capstyle="butt")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, length=2, width=0.8)


def _load(results_dir: str) -> tuple[list[tuple[float, dict]], dict | None]:
    runs, control = [], None
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            hist = json.load(f)
        cfg = hist.get("config")
        if not cfg:
            print(f"skipping {os.path.basename(path)}: no config record (pre-fp32 history)")
            continue
        if "lambda_graph" not in cfg:
            if control is not None:
                raise SystemExit("more than one standard-KD history in the directory")
            control = hist
            continue
        runs.append((float(cfg["lambda_graph"]), hist))
    if not runs:
        raise SystemExit(f"no fp32-regime histories under {results_dir}")
    runs.sort(key=lambda r: r[0])
    if len(runs) > len(RAMP):
        raise SystemExit(f"{len(runs)} runs but the ordinal ramp has {len(RAMP)} steps")
    return runs, control


def _mean_ood(hist: dict) -> list[float]:
    return [sum(hist[f"accuracy_{o}"][i] for o in OOD) / len(OOD) for i in range(len(hist["accuracy"]))]


def _series(hist: dict, key: str, last_step: int | None = None) -> tuple[list[int], list[float]]:
    """An eval series by recorded step, optionally clipped so a longer run shares the axis."""
    xs = hist["accuracy_step"]
    ys = _mean_ood(hist) if key == "ood" else hist[key]
    if last_step is not None:
        xs = [s for s in xs if s <= last_step]
        ys = ys[: len(xs)]
    return xs, ys


def _end_labels(ax, ends: list[list], dx: float = 1.5) -> None:
    """Direct labels at line ends, spread apart and kept inside the panel."""
    ends = sorted(ends, key=lambda e: e[1])
    lo, hi = ax.get_ylim()
    min_gap = 0.06 * (hi - lo)
    for i in range(1, len(ends)):
        if ends[i][1] - ends[i - 1][1] < min_gap:
            ends[i][1] = ends[i - 1][1] + min_gap
    overshoot = ends[-1][1] - (hi - 0.03 * (hi - lo))
    if overshoot > 0:
        for e in ends:
            e[1] -= overshoot
    for x, y, text in ends:
        ax.annotate(text, (x + dx, y), color=INK_2, fontsize=6, ha="left", va="center")


def _summary(hist: dict, baseline: float, last_step: int) -> dict:
    st = hist["accuracy_step"]
    acc = dict(zip(st, hist["accuracy"]))
    ood = dict(zip(st, _mean_ood(hist)))
    early = {s: v for s, v in acc.items() if 1 <= s <= 30}
    dip = min(early, key=early.get)
    rec = next((s for s in sorted(acc) if s > dip and acc[s] >= baseline), None)
    at = max(s for s in st if s <= last_step)
    pk, opk = max(acc, key=acc.get), max(ood, key=ood.get)
    fam = {o: dict(zip(st, hist[f"accuracy_{o}"]))[at] for o in OOD}
    return dict(dip=early[dip], dip_step=dip, recovery=rec, id_peak=acc[pk], id_peak_step=pk,
                ood_peak=ood[opk], ood_peak_step=opk, ood_at=ood[at], at=at, fam=fam,
                last=(st[-1], acc[st[-1]], ood[st[-1]]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default="results/lambda_graph_study")
    ap.add_argument("--out", default="latex/graph_distillation/figures/lambda_sweep.pdf")
    args = ap.parse_args()

    runs, control = _load(args.results_dir)
    colours = RAMP[-len(runs):] if len(runs) < len(RAMP) else RAMP
    base = runs[0][1]
    baseline, teacher = base["student_baseline"], base["teacher_baseline"]
    ood_baseline = _mean_ood(base)[0]
    last_step = max(h["train_step"][-1] for _, h in runs)

    plt.rcParams.update({
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "axes.titlesize": 7.5, "pdf.fonttype": 42,
    })
    fig, (ax_id, ax_ood, ax_lam) = plt.subplots(1, 3, figsize=(5.5, 2.05))
    for ax in (ax_id, ax_ood, ax_lam):
        _style(ax)

    # ---- (a), (b): trajectories ---------------------------------------------
    for ax, key in ((ax_id, "accuracy"), (ax_ood, "ood")):
        ax.set_xlim(-1, last_step + 12)
        ax.set_xticks(range(0, last_step + 1, 25))
        ax.set_xlabel("training step", color=INK_2)
        ends = []
        if control is not None:
            xs, ys = _series(control, key, last_step)
            ax.plot(xs, ys, **CONTROL)
            ends.append([xs[-1], ys[-1], "KD only"])
        for (lam, hist), colour in zip(runs, colours):
            xs, ys = _series(hist, key)
            ax.plot(xs, ys, color=colour, linewidth=1.4, zorder=3)
            ends.append([xs[-1], ys[-1], f"{lam:g}"])
        ax.axhline(baseline if key == "accuracy" else ood_baseline, **RULE_UNTRAINED)
        if key == "accuracy":
            ax.set_ylim(0, 1.02)
            ax.axhline(teacher, **RULE_TEACHER)
            ax.annotate("teacher", (1, teacher), color=INK_3, fontsize=6, ha="left", va="bottom")
        # Rule label at the right-hand end, below the rule: every curve is well
        # above both baselines by then, so nothing collides. Checked on the render.
        ax.annotate("untrained student", (last_step, baseline if key == "accuracy" else ood_baseline),
                    color=INK_3, fontsize=6, ha="right", va="top")
        _end_labels(ax, ends)
    ax_id.set_title("in-distribution (22_add)", color=INK, pad=4)
    ax_ood.set_title("mean out-of-distribution", color=INK, pad=4)
    ax_id.set_ylabel("accuracy", color=INK_2)

    # ---- (c): peak OOD against lambda, the conventional sensitivity view -------
    summaries = {lam: _summary(h, baseline, last_step) for lam, h in runs}
    lams = [lam for lam, _ in runs]
    peaks = [summaries[lam]["ood_peak"] for lam in lams]
    ax_lam.set_xscale("log")
    ax_lam.plot(lams, peaks, color=INK_2, linewidth=1.0, zorder=2)
    for lam, pk, colour in zip(lams, peaks, colours):
        ax_lam.plot([lam], [pk], marker="o", markersize=5, color=colour, markeredgewidth=0, zorder=3)
    ax_lam.axhline(ood_baseline, **RULE_UNTRAINED)
    if control is not None:
        c = _summary(control, baseline, last_step)
        ax_lam.axhline(c["ood_peak"], **CONTROL)
        ax_lam.annotate("KD only, peak", (lams[0] * 0.75, c["ood_peak"]), color=INK_2, fontsize=6,
                        ha="left", va="bottom")
    ax_lam.annotate("untrained student", (lams[0] * 0.75, ood_baseline), color=INK_3, fontsize=6,
                    ha="left", va="bottom")
    ax_lam.set_xlim(lams[0] * 0.6, lams[-1] * 1.8)
    ax_lam.set_xticks(lams)
    ax_lam.set_xticklabels([f"{lam:g}" for lam in lams])
    ax_lam.set_ylim(0.1, max(peaks) + 0.04)
    ax_lam.set_title("peak mean OOD vs. $\\lambda$", color=INK, pad=4)
    ax_lam.set_xlabel("$\\lambda_{\\mathrm{graph}}$ (log)", color=INK_2)

    handles = [Line2D([], [], color=c, linewidth=1.4) for c in colours]
    labels = [f"$\\lambda={lam:g}$" for lam in lams]
    if control is not None:
        handles.insert(0, Line2D([], [], color=INK_2, linewidth=1.2, linestyle=(0, (4, 2))))
        labels.insert(0, "KD only ($\\lambda=0$)")
    leg = fig.legend(handles, labels, loc="lower center", ncol=len(handles), frameon=False,
                     fontsize=6.5, handlelength=2.6, columnspacing=1.6, bbox_to_anchor=(0.5, -0.02))
    for text in leg.get_texts():
        text.set_color(INK)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=220, bbox_inches="tight", facecolor="white")
    print("wrote", args.out)

    # ---- Table rows ------------------------------------------------------------
    def fmt(v: float, ref: float, best: bool = False) -> str:
        s = f"{v:.4f}"
        if v < ref:
            return f"\\emph{{{s}}}"
        return f"\\textbf{{{s}}}" if best else s

    rows = ([("KD only ($\\lambda = 0$)", control)] if control is not None else []) + \
           [(f"$\\lambda = {lam:g}$", h) for lam, h in runs]
    sums = [(name, _summary(h, baseline, last_step)) for name, h in rows]
    best_fam = {o: max(s["fam"][o] for _, s in sums) for o in OOD}
    best_id = max(s["id_peak"] for _, s in sums)
    best_ood = max(s["ood_peak"] for _, s in sums)
    best_at = max(s["ood_at"] for _, s in sums)
    print(f"\n% rows: run & 22_add peak (step) & OOD peak (step) & OOD @{last_step} & " + " & ".join(OOD) + " & dip / recovery")
    print("    student, untrained & 0.6404 & " + f"{ood_baseline:.4f} & --- & "
          + " & ".join(f"{base[f'accuracy_{o}'][0]:.4f}" for o in OOD) + " & --- \\\\")
    for name, s in sums:
        fam = " & ".join(fmt(s["fam"][o], base[f"accuracy_{o}"][0], s["fam"][o] == best_fam[o]) for o in OOD)
        idp = ("\\textbf{%.4f}" if s["id_peak"] == best_id else "%.4f") % s["id_peak"]
        odp = ("\\textbf{%.4f}" if s["ood_peak"] == best_ood else "%.4f") % s["ood_peak"]
        oat = ("\\textbf{%.4f}" if s["ood_at"] == best_at else "%.4f") % s["ood_at"]
        tail = f"   % @{s['last'][0]}: {s['last'][1]:.4f} / {s['last'][2]:.4f}" if s["last"][0] != last_step else ""
        print(f"    {name} & {idp} ({s['id_peak_step']}) & {odp} ({s['ood_peak_step']}) & {oat} & {fam} & "
              f"{s['dip']:.3f} / {s['recovery'] if s['recovery'] is not None else '---'} \\\\{tail}")
    for lam, h in runs:
        r1 = h["step_grad_ratio"][0]
        print(f"% lambda={lam:g}: step1 |g_KD|={h['step_kl_gnorm'][0]:.4f} |g_graph|/lambda={h['step_graph_gnorm'][0] / lam:.3f} "
              f"cos={h['step_grad_cosine'][0]:+.4f} ratio/lambda={r1 / lam:.4f} | ratio last={h['step_grad_ratio'][-1]:.4f} "
              f"| canary={h.get('params_changed_step1')}")


if __name__ == "__main__":
    main()

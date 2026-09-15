"""Render the lambda-sweep figures and table rows for the paper appendix.

Reads every graph-KD history in the lambda study directory that carries a
``config`` record (the fp32-regime runs; bf16-era histories have no such key and
are skipped with a notice), orders them by ``lambda_graph``, and produces:

* ``lambda_sweep.pdf`` -- four panels against training step: in-distribution
  accuracy, mean out-of-distribution accuracy, the graph loss, and the ratio of
  the graph gradient norm to the KD gradient norm.
* ``lambda_sweep_ood.pdf`` -- the four out-of-distribution families as small
  multiples, so the per-family shape is visible rather than averaged away.
* LaTeX table rows on stdout, so the numbers in the appendix are generated from
  the histories rather than transcribed.

Encoding: lambda is an ordered quantity, so the runs take an ordinal ramp of one
hue (light to dark with increasing lambda) rather than unrelated categorical
hues. Every run is also direct-labelled at its right-hand end. Runs evaluated
every other step are plotted against their recorded ``accuracy_step``, so
different eval cadences share an axis.

Usage:
    python -m experiments.plot_lambda_sweep
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OOD = ["222_add", "2222_add", "33_add", "21_mult"]
# Two-line titles: four panels share 5.5in, so one-line descriptions collide.
OOD_TITLES = {
    "222_add": "222_add\nthree two-digit operands",
    "2222_add": "2222_add\nfour two-digit operands",
    "33_add": "33_add\nthree-digit addition",
    "21_mult": "21_mult\nmultiplication",
}
SMOOTH = 9  # rolling-mean window for the per-step graph loss

# Ordinal ramp, one hue, light -> dark with increasing lambda: steps 250 / 450 /
# 650 of the reference blue ramp, the lightest of which still clears 2:1 on white.
RAMP = ["#86b6ef", "#2a78d6", "#104281"]
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8983"
GRID = "#e9e8e5"


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
    """Graph-KD runs ordered by lambda, plus the standard-KD control if present.

    A standard-KD history is recognised by its config having no ``lambda_graph``;
    it is the lambda = 0 reference and is drawn as a control, not as a ramp step.
    """
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


def _clip(hist: dict, key: str, last_step: int) -> tuple[list[int], list[float]]:
    """The eval series ``key`` up to ``last_step``, so a longer control shares the axis."""
    xs = [s for s in hist["accuracy_step"] if s <= last_step]
    ys = hist[key] if key != "ood" else _mean_ood(hist)
    return xs, ys[: len(xs)]


def _mean_ood(hist: dict) -> list[float]:
    return [sum(hist[f"accuracy_{o}"][i] for o in OOD) / len(OOD) for i in range(len(hist["accuracy"]))]


def _lam_label(lam: float) -> str:
    return f"$\\lambda={lam:g}$"


def _rolling(y: list[float], window: int) -> list[float]:
    """Centred rolling mean; the window shrinks at the ends so no point is dropped."""
    half = window // 2
    return [statistics.mean(y[max(0, i - half): i + half + 1]) for i in range(len(y))]


def _end_labels(ax, runs, series_fn, colours, dx: float = 1.5, nudge: bool = True,
                extra: list[tuple[float, float, str]] | None = None) -> None:
    """Direct-label each run at its last point.

    With ``nudge`` the labels are spread apart vertically and then, if the spread
    pushed the top one past the axis, the whole cluster is shifted back down so
    every label stays inside the panel. Off for log axes, where the lines are far
    apart anyway and a linear gap would be meaningless. ``extra`` adds labels for
    series that are not lambda runs (the control) to the same collision pass.
    """
    ends = []
    for (lam, hist), colour in zip(runs, colours):
        y = series_fn(hist)
        x = hist["accuracy_step"][-1] if len(y) == len(hist["accuracy_step"]) else hist["train_step"][-1]
        ends.append([x, y[-1], f"{lam:g}"])
    for x, y, text in extra or []:
        ends.append([x, y, text])
    if nudge:
        ends.sort(key=lambda e: e[1])
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default="results/lambda_graph_study")
    ap.add_argument("--out", default="latex/graph_distillation/figures/lambda_sweep.pdf")
    args = ap.parse_args()

    runs, control = _load(args.results_dir)
    colours = RAMP[-len(runs):] if len(runs) < len(RAMP) else RAMP
    CONTROL_STYLE = dict(color=INK_2, linewidth=1.2, linestyle=(0, (4, 2)), dash_capstyle="round", zorder=2)
    base = runs[0][1]
    baseline = base["student_baseline"]
    teacher = base["teacher_baseline"]
    ood_baseline = _mean_ood(base)[0]
    last_step = max(h["train_step"][-1] for _, h in runs)

    plt.rcParams.update({
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "axes.titlesize": 7.5, "pdf.fonttype": 42,
    })

    # ---- Figure 1: accuracy, graph loss, gradient ratio --------------------
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.6))
    (ax_id, ax_ood), (ax_gl, ax_ratio) = axes
    for ax in axes.flat:
        _style(ax)
        ax.set_xlim(-1, last_step + 9)
        ax.set_xticks(range(0, last_step + 1, 25))

    ax_id.axhline(baseline, color=INK_3, linewidth=0.9, linestyle=(0, (2, 2)), zorder=1)
    ax_id.axhline(teacher, color=INK_3, linewidth=0.9, linestyle=(0, (1, 2)), zorder=1)
    ax_ood.axhline(ood_baseline, color=INK_3, linewidth=0.9, linestyle=(0, (2, 2)), zorder=1)

    control_end: dict[str, list[tuple[float, float, str]]] = {}
    if control is not None:
        for ax, key in ((ax_id, "accuracy"), (ax_ood, "ood")):
            xs, ys = _clip(control, key, last_step)
            ax.plot(xs, ys, **CONTROL_STYLE)
            control_end[key] = [(xs[-1], ys[-1], "KD only")]
    for (lam, hist), colour in zip(runs, colours):
        st = hist["accuracy_step"]
        ax_id.plot(st, hist["accuracy"], color=colour, linewidth=1.4, zorder=3)
        ax_ood.plot(st, _mean_ood(hist), color=colour, linewidth=1.4, zorder=3)
        ts = hist["train_step"]
        # The per-step JSD is noisy at the +/-0.03 level; the rolling mean is what
        # the caption describes. The window is stated there.
        ax_gl.plot(ts, _rolling(hist["step_graph_loss"], SMOOTH), color=colour, linewidth=1.3, zorder=3)
        ax_ratio.plot(ts, hist["step_grad_ratio"], color=colour, linewidth=1.1, zorder=3)

    ax_id.set_title("in-distribution (22_add)", color=INK, pad=4)
    ax_ood.set_title("mean out-of-distribution", color=INK, pad=4)
    ax_gl.set_title(r"graph loss $\mathcal{L}_{\mathrm{graph}}$ (JSD)", color=INK, pad=4)
    ax_ratio.set_title(r"$\|g_{\mathrm{graph}}\| \,/\, \|g_{\mathrm{KD}}\|$ (log)", color=INK, pad=4)
    ax_ratio.set_yscale("log")
    ax_id.set_ylabel("accuracy", color=INK_2)
    ax_gl.set_ylabel("loss", color=INK_2)
    for ax in (ax_gl, ax_ratio):
        ax.set_xlabel("training step", color=INK_2)
    ax_id.set_ylim(0, 1.02)
    # Rule labels go at the right-hand end, below the rule: by then every curve
    # sits well above both baselines, so nothing collides. Checked on the render.
    ax_id.annotate("teacher", (1, teacher), color=INK_3, fontsize=6, ha="left", va="bottom")
    ax_id.annotate("untrained student", (last_step, baseline), color=INK_3, fontsize=6,
                   ha="right", va="top")
    ax_ood.annotate("untrained student", (last_step, ood_baseline), color=INK_3, fontsize=6,
                    ha="right", va="top")

    _end_labels(ax_id, runs, lambda h: h["accuracy"], colours, extra=control_end.get("accuracy"))
    _end_labels(ax_ood, runs, _mean_ood, colours, extra=control_end.get("ood"))
    _end_labels(ax_gl, runs, lambda h: _rolling(h["step_graph_loss"], SMOOTH), colours)
    _end_labels(ax_ratio, runs, lambda h: h["step_grad_ratio"], colours, nudge=False)

    handles = [Line2D([], [], color=c, linewidth=1.4) for c in colours]
    labels = [_lam_label(lam) for lam, _ in runs]
    if control is not None:
        handles.insert(0, Line2D([], [], color=INK_2, linewidth=1.2, linestyle=(0, (4, 2))))
        labels.insert(0, r"KD only ($\lambda=0$)")
    leg = fig.legend(handles, labels,
                     title=r"$\lambda_{\mathrm{graph}}$", loc="lower center", ncol=len(handles),
                     frameon=False, fontsize=6.5, handlelength=2.6, columnspacing=1.6,
                     bbox_to_anchor=(0.5, -0.01))
    leg.get_title().set_fontsize(6.5)
    leg.get_title().set_color(INK_2)
    for text in leg.get_texts():
        text.set_color(INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=220, bbox_inches="tight", facecolor="white")
    print("wrote", args.out)

    # ---- Figure 2: the four OOD families -----------------------------------
    fig2, axes2 = plt.subplots(1, 4, figsize=(5.5, 1.9))
    for ax, fam in zip(axes2, OOD):
        _style(ax)
        ax.set_xlim(-1, last_step + 9)
        ax.set_xticks(range(0, last_step + 1, 50))
        ax.axhline(base[f"accuracy_{fam}"][0], color=INK_3, linewidth=0.9, linestyle=(0, (2, 2)), zorder=1)
        if control is not None:
            xs, ys = _clip(control, f"accuracy_{fam}", last_step)
            ax.plot(xs, ys, **CONTROL_STYLE)
        for (lam, hist), colour in zip(runs, colours):
            ax.plot(hist["accuracy_step"], hist[f"accuracy_{fam}"], color=colour, linewidth=1.3, zorder=3)
        ax.set_title(OOD_TITLES[fam], color=INK, pad=4, fontsize=6.5)
        ax.set_xlabel("training step", color=INK_2)
        _end_labels(ax, runs, lambda h, f=fam: h[f"accuracy_{f}"], colours)
    axes2[0].set_ylabel("accuracy", color=INK_2)
    leg2 = fig2.legend(handles, labels,
                       title=r"$\lambda_{\mathrm{graph}}$", loc="lower center", ncol=len(handles),
                       frameon=False, fontsize=6.5, handlelength=2.6, columnspacing=1.6,
                       bbox_to_anchor=(0.5, -0.06))
    leg2.get_title().set_fontsize(6.5)
    leg2.get_title().set_color(INK_2)
    for text in leg2.get_texts():
        text.set_color(INK)
    fig2.tight_layout(rect=(0, 0.14, 1, 1))
    out2 = args.out.replace(".pdf", "_ood.pdf")
    fig2.savefig(out2, bbox_inches="tight", facecolor="white")
    fig2.savefig(out2.replace(".pdf", ".png"), dpi=220, bbox_inches="tight", facecolor="white")
    print("wrote", out2)

    # ---- Table rows ----------------------------------------------------------
    rows = ([("KD only", control)] if control is not None else []) + [(f"{lam:g}", h) for lam, h in runs]
    print(f"\n% accuracy summary rows: run & dip min (step) & recovery step & 22_add peak (step) & 22_add @{last_step} & OOD peak (step) & OOD @{last_step}  [control also: @its last step]")
    for name, hist in rows:
        st, acc = hist["accuracy_step"], hist["accuracy"]
        d = dict(zip(st, acc))
        early = {s: v for s, v in d.items() if 1 <= s <= 30}
        mn = min(early, key=early.get)
        rec = next((s for s in sorted(d) if s > mn and d[s] >= baseline), None)
        ood = dict(zip(st, _mean_ood(hist)))
        pk, opk = max(d, key=d.get), max(ood, key=ood.get)
        at = max(s for s in st if s <= last_step)
        extra = f"   % @{st[-1]}: {d[st[-1]]:.4f} / {ood[st[-1]]:.4f}" if st[-1] != last_step else ""
        print(f"    {name} & {early[mn]:.3f} ({mn}) & {rec if rec is not None else '---'} & "
              f"{d[pk]:.4f} ({pk}) & {d[at]:.4f} & {ood[opk]:.4f} ({opk}) & {ood[at]:.4f} \\\\{extra}")
    print(f"\n% per-family rows at step {last_step}: run & 222_add & 2222_add & 33_add & 21_mult   [peak (step) in comment]")
    print("    untrained & " + " & ".join(f"{base[f'accuracy_{o}'][0]:.4f}" for o in OOD) + " \\\\")
    for name, hist in rows:
        st = hist["accuracy_step"]
        i = max(j for j, s in enumerate(st) if s <= last_step)
        peaks = ", ".join(f"{o}={max(hist[f'accuracy_{o}']):.4f}@{st[hist[f'accuracy_{o}'].index(max(hist[f'accuracy_{o}']))]}" for o in OOD)
        print(f"    {name} & " + " & ".join(f"{hist[f'accuracy_{o}'][i]:.4f}" for o in OOD) + f" \\\\   % peaks: {peaks}")
    print("\n% gradient rows: lambda & |g_KD|@1 & |g_graph|@1 & (ratio/lambda)@1 & cos@1 & mean ratio & mean cos & mean sign flips")
    for lam, hist in runs:
        r1 = hist["step_grad_ratio"][0]
        print(f"    {lam:g} & {hist['step_kl_gnorm'][0]:.4f} & {hist['step_graph_gnorm'][0]:.4f} & {r1 / lam:.4f} & "
              f"${hist['step_grad_cosine'][0]:+.4f}$ & {statistics.mean(hist['step_grad_ratio']):.4f} & "
              f"${statistics.mean(hist['step_grad_cosine']):+.4f}$ & {statistics.mean(hist['step_grad_signflip']):.4f} \\\\")
    for lam, hist in runs:
        gl = hist["step_graph_loss"]
        print(f"% lambda={lam:g}: graph loss first10={statistics.mean(gl[:10]):.4f} last10={statistics.mean(gl[-10:]):.4f}; "
              f"KL first={hist['step_kl_loss'][0]:.4f} last={hist['step_kl_loss'][-1]:.4f}; "
              f"|g_KD| first={hist['step_kl_gnorm'][0]:.2f} last={hist['step_kl_gnorm'][-1]:.2f}; "
              f"ratio last={hist['step_grad_ratio'][-1]:.4f}; canary={hist.get('params_changed_step1')}")


if __name__ == "__main__":
    main()

"""Figures and table rows for the audit of where the graph term's gain lives.

Reads the JSONs the analysis scripts wrote under results/overnight/:
  format_decomp/summary.json   teacher, base, kd1-3, sft1-3, graph1-3 (experiments.format_decomp)
  format_decomp/controls.json  scramble, commit/ce controls at three weights, seed 1
  graph_delta/seed{1,2,3}.json layer-span transplants (experiments.graph_delta)
  pos_patch/pos_patch_seed{1,2}.json residual patching by layer and position (experiments.pos_patch)

Writes figures/audit_{decomp,localise,controls}.pdf and tables/audit_{decomp,controls}.tex.

Usage: PYTHONPATH=src python -m experiments.plot_audit
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments._paper_plots import (AQUA, BLUE, FIG_DIR, GRID, INK, INK_2, INK_3, ORANGE, RC,  # noqa: E402
                                      TABLE_DIR, save, style)

R = "results/overnight"
FAMILIES = ["22_add", "222_add", "2222_add", "33_add", "21_mult"]
GROUPS = {"teacher": ["teacher"], "base": ["base"], "KD": ["kd1", "kd2", "kd3"], "SFT": ["sft1", "sft2", "sft3"],
          "graph": ["graph1", "graph2", "graph3"]}
CONTROLS = {"scrambled graph": "scramble", "commit, 0.01": "commit0.01", "commit, 0.1": "commit0.1",
            "gold CE, 0.1": "ce0.1"}


def load(path):
    with open(os.path.join(R, path), encoding="utf-8") as fh:
        return json.load(fh)


def stat(d, names, mode, ds, key):
    xs = [d[n]["results"][mode][ds][key] for n in names]
    return float(np.mean(xs)), (float(np.std(xs, ddof=1)) if len(xs) > 1 else None)


def fig_decomp(fd, ctl):
    """21_mult: right / answered-wrong / no number, with BOS; and accuracy with vs without BOS."""
    rows = [(g, fd, ms) for g, ms in GROUPS.items()] + [(g, ctl, [n]) for g, n in CONTROLS.items()]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [1.5, 1]})
    y = np.arange(len(rows))[::-1]
    for yi, (label, d, ms) in zip(y, rows):
        acc, _ = stat(d, ms, "bos", "21_mult", "acc")
        commit, _ = stat(d, ms, "bos", "21_mult", "commit")
        first = yi == y[0]
        a.barh(yi, acc, color=BLUE, height=0.62, label="right" if first else None)
        a.barh(yi, commit - acc, left=acc, color=AQUA, height=0.62, label="answered, wrong" if first else None)
        a.barh(yi, 1 - commit, left=commit, color=ORANGE, height=0.62, label="no number (hedge)" if first else None)
        nob, _ = stat(d, ms, "nobos", "21_mult", "acc")
        b.plot([nob, acc], [yi, yi], color=GRID, linewidth=1.6, zorder=1)
        b.scatter([nob], [yi], color=INK_3, s=12, zorder=2)
        b.scatter([acc], [yi], color=BLUE, s=12, zorder=3)
    for ax in (a, b):
        style(ax)
        ax.set_yticks(y)
        ax.set_xlim(0, 1)
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.yaxis.grid(False)
    a.set_yticklabels([r[0] for r in rows], color=INK)
    b.set_yticklabels([])
    a.set_title("21_mult, BOS at eval", color=INK, pad=4)
    a.legend(frameon=False, fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    b.set_title("accuracy without BOS (grey) and with BOS (blue)", color=INK, pad=4)
    save(fig, os.path.join(FIG_DIR, "audit_decomp.pdf"))


def fig_localise():
    spans = ["L0-0", "L0-1", "L0-3", "L0-7", "L0-15"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for s, colour in zip((1, 2, 3), (BLUE, AQUA, ORANGE)):
        d = load(f"graph_delta/seed{s}.json")
        kd = d["reference"]["kd"]["21_mult"]["acc"]
        graph = d["reference"]["graph"]["21_mult"]["acc"]
        xs = [0] + [int(k.split("-")[1]) + 1 for k in spans]
        ys = [kd] + [d["span"][k]["21_mult"]["acc"] for k in spans]
        a.plot(xs, ys, color=colour, marker="o", markersize=2.5, linewidth=1.3, label=f"seed {s}")
        late = d["span"]["L8-15"]["21_mult"]["acc"]
        a.scatter([8], [late], color=colour, marker="x", s=14, zorder=4)
        a.axhline(graph, color=colour, linewidth=0.6, linestyle=(0, (1, 2)))
    style(a)
    a.set_xlabel("graph layers 0..k-1 copied into KD (x: layers 8-15 instead)", color=INK_2)
    a.set_ylabel("21_mult accuracy", color=INK_2)
    a.set_title("layer transplant, seed-matched", color=INK, pad=4)
    a.legend(frameon=False, fontsize=6)
    p = load("pos_patch/pos_patch_seed1.json")
    layers, pos = [1, 3, 7, 11], ["BOS", "a", "*", "b", "=", "all"]
    grid = np.array([[p[f"L{L}/{q}"]["commit"] for q in pos] for L in layers])
    im = b.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(layers)):
        for j in range(len(pos)):
            b.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=6,
                   color="white" if grid[i, j] > 0.7 else INK)
    b.set_xticks(range(len(pos)), pos)
    b.set_yticks(range(len(layers)), [f"after L{L}" for L in layers])
    b.set_title(f"KD commit rate, graph residual patched in (KD {p['kd']['commit']:.2f}, graph {p['graph']['commit']:.2f})",
                color=INK, pad=4)
    b.tick_params(colors=INK_2, length=0)
    save(fig, os.path.join(FIG_DIR, "audit_localise.pdf"))


def fig_controls(fd, ctl):
    arms = [("KD", fd, GROUPS["KD"], INK_3), ("graph", fd, GROUPS["graph"], BLUE),
            ("commit, 0.01", ctl, ["commit0.01"], AQUA), ("gold CE, 0.1", ctl, ["ce0.1"], ORANGE),
            ("scrambled graph", ctl, ["scramble"], INK_2)]
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    w = 0.16
    for i, (label, d, ms, colour) in enumerate(arms):
        m = [stat(d, ms, "bos", ds, "acc")[0] for ds in FAMILIES]
        ax.bar(np.arange(len(FAMILIES)) + (i - 2) * w, m, width=w, color=colour, label=label)
    style(ax)
    ax.set_xticks(range(len(FAMILIES)), FAMILIES)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("accuracy (BOS at eval)", color=INK_2)
    ax.legend(frameon=False, fontsize=6, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    save(fig, os.path.join(FIG_DIR, "audit_controls.pdf"))


def f(x, sd=None):
    return f"${x:.3f}$" if sd is None or sd < 0.01 else f"${x:.3f} \\pm {sd:.3f}$"


def tables(fd, ctl):
    rows = []
    for g, ms in GROUPS.items():
        cells = [f(*stat(fd, ms, "bos", "21_mult", k)) for k in ("acc", "commit", "acc_given_commit", "num_argmax", "p_q")]
        cells.append(f(*stat(fd, ms, "nobos", "21_mult", "acc")))
        rows.append(" & ".join([g] + cells) + r" \\")
    for g, n in CONTROLS.items():
        cells = [f(*stat(ctl, [n], "bos", "21_mult", k)) for k in ("acc", "commit", "acc_given_commit", "num_argmax", "p_q")]
        cells.append(f(*stat(ctl, [n], "nobos", "21_mult", "acc")))
        rows.append(" & ".join([g] + cells) + r" \\")
    out = {"audit_decomp": rows, "audit_controls": []}
    for g, d, ms in [(g, fd, ms) for g, ms in GROUPS.items()] + [(g, ctl, [n]) for g, n in CONTROLS.items()]:
        out["audit_controls"].append(" & ".join([g] + [f(*stat(d, ms, "bos", ds, "acc")) for ds in FAMILIES]) + r" \\")
    os.makedirs(TABLE_DIR, exist_ok=True)
    for name, rs in out.items():
        with open(os.path.join(TABLE_DIR, f"{name}.tex"), "w", encoding="utf-8") as fh:
            fh.write(f"% generated by experiments.plot_audit from {R}\n" + "\n".join(rs) + "\n")
        print("wrote", os.path.join(TABLE_DIR, f"{name}.tex"))


def main():
    plt.rcParams.update(RC)
    fd, ctl = load("format_decomp/summary.json"), load("format_decomp/controls.json")
    fig_decomp(fd, ctl)
    fig_localise()
    fig_controls(fd, ctl)
    tables(fd, ctl)


if __name__ == "__main__":
    main()

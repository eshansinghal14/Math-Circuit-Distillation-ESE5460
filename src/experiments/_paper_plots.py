"""Shared loading, statistics and styling for the paper figures.

Every ``plot_*.py`` in this package reads training histories written by the
trainers in ``training/`` and renders one figure for ``latex/graph_distillation``,
writing the LaTeX table rows it reports alongside as a ``tables/*.tex`` fragment
so the numbers in the paper are generated, never transcribed.

Histories: a file holds one run at top level and, since multi-seed support,
every seed that wrote to its folder under ``runs`` (keyed by seed as a string).
:func:`load_run` returns all seeds; statistics are seed means with a sample
standard deviation when there is more than one seed.

Palette: the validated categorical slots (blue, orange, aqua) and the one-hue
blue ramp for ordered series; ink and grid tokens for everything that is text or
scaffolding. Colour carries identity only, so every figure also has a legend and
direct labels.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

OOD = ["222_add", "2222_add", "33_add", "21_mult"]
FAMILY_NAMES = {
    "22_add": "22_add", "222_add": "222_add", "2222_add": "2222_add",
    "33_add": "33_add", "21_mult": "21_mult",
}

# Categorical slots 1-3 of the validated palette; ordinal ramp for lambda.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#86b6ef", "#2a78d6", "#104281"]
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8983"
GRID = "#e9e8e5"
RULE_UNTRAINED = dict(color=INK_3, linewidth=0.9, linestyle=(0, (2, 2)), zorder=1)
RULE_TEACHER = dict(color=INK_3, linewidth=0.9, linestyle=(0, (1, 2)), zorder=1)
CONTROL = dict(color=INK_2, linewidth=1.2, linestyle=(0, (4, 2)), dash_capstyle="round", zorder=2)

RC = {
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "axes.titlesize": 7.5, "pdf.fonttype": 42,
    "legend.fontsize": 6.5,
}

FIG_DIR = "latex/graph_distillation/figures"
TABLE_DIR = "latex/graph_distillation/tables"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass
class Run:
    """One experimental arm: every seed's history for one config."""
    label: str
    hists: List[Dict[str, Any]]
    path: str = ""
    seeds: List[str] = field(default_factory=list)

    @property
    def config(self) -> Dict[str, Any]:
        return self.hists[0].get("config", {})

    @property
    def n(self) -> int:
        return len(self.hists)

    @property
    def steps(self) -> np.ndarray:
        return np.asarray(self.hists[0]["accuracy_step"])

    @property
    def baseline(self) -> float:
        return float(self.hists[0]["student_baseline"])

    @property
    def teacher(self) -> Optional[float]:
        t = self.hists[0].get("teacher_baseline")
        return None if t is None else float(t)

    def _key_series(self, h: Dict[str, Any], key: str) -> np.ndarray:
        if key == "ood":
            return np.mean([h[f"accuracy_{o}"] for o in OOD], axis=0)
        if key == "accuracy":
            return np.asarray(h["accuracy"], dtype=float)
        return np.asarray(h[f"accuracy_{key}"], dtype=float)

    def series(self, key: str, last_step: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """(steps, seed mean, seed std or None) of an eval series; ``key`` is
        ``accuracy``, ``ood`` or a family name."""
        ys = np.stack([self._key_series(h, key) for h in self.hists])
        xs = self.steps
        if last_step is not None:
            keep = xs <= last_step
            xs, ys = xs[keep], ys[:, keep]
        std = ys.std(axis=0, ddof=1) if self.n > 1 else None
        return xs, ys.mean(axis=0), std

    def step_series(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        """(train steps, seed mean) of a per-step training metric."""
        ys = np.stack([np.asarray(h[key], dtype=float) for h in self.hists])
        return np.asarray(self.hists[0]["train_step"]), ys.mean(axis=0)

    def baseline_of(self, key: str) -> float:
        xs, mean, _ = self.series(key)
        assert xs[0] == 0, "history has no step-0 eval"
        return float(mean[0])

    def summary(self, at_step: int) -> Dict[str, Any]:
        """Seed-level statistics used by every table.

        ``at``: values at ``at_step`` (or the last eval before it), mean and std.
        ``peak``: each seed's own maximum, mean and std, with the step at which the
        seed-mean curve peaks. ``dip``: minimum of the seed-mean in-distribution
        curve over steps 1-40 and the first later step back at or above the
        untrained student.
        """
        xs = self.steps
        at = int(xs[xs <= at_step].max())
        i_at = int(np.where(xs == at)[0][0])
        out: Dict[str, Any] = {"at": at, "n": self.n, "last": int(xs[-1])}
        for key in ["accuracy", "ood"] + OOD:
            ys = np.stack([self._key_series(h, key) for h in self.hists])
            mean = ys.mean(axis=0)
            out[key] = {
                "at": (ys[:, i_at].mean(), ys[:, i_at].std(ddof=1) if self.n > 1 else None),
                "peak": (ys.max(axis=1).mean(), ys.max(axis=1).std(ddof=1) if self.n > 1 else None),
                "peak_step": int(xs[int(mean.argmax())]),
                "last": (ys[:, -1].mean(), ys[:, -1].std(ddof=1) if self.n > 1 else None),
            }
        acc = self.series("accuracy")[1]
        early = (xs >= 1) & (xs <= 40)
        i_dip = int(np.where(early)[0][int(acc[early].argmin())])
        later = [int(s) for s, a in zip(xs[i_dip + 1:], acc[i_dip + 1:]) if a >= self.baseline]
        out["dip"] = float(acc[i_dip])
        out["dip_step"] = int(xs[i_dip])
        out["recovery"] = later[0] if later else None
        return out


def _runs_in(hist: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    runs = hist.get("runs")
    if isinstance(runs, dict) and runs:
        return dict(runs)
    return {str(hist.get("config", {}).get("seed", 42)): {k: v for k, v in hist.items() if k != "runs"}}


def load_run(path: str, label: str, seeds: Optional[Sequence[str]] = None) -> Run:
    """Every seed in a history file (or the ``seeds`` named), as one :class:`Run`.

    Refuses a history without a ``config`` record (a bf16-era file) and warns when
    seeds in one file were evaluated with different ``max_eval_tokens``, since
    their accuracies are then not comparable.
    """
    with open(path, encoding="utf-8") as f:
        hist = json.load(f)
    runs = _runs_in(hist)
    if seeds is not None:
        runs = {s: runs[str(s)] for s in seeds}
    hists = []
    for seed, h in sorted(runs.items(), key=lambda kv: int(kv[0])):
        if not h.get("config"):
            raise SystemExit(f"{path} seed {seed}: no config record (pre-fp32 history); refusing to plot it")
        hists.append(h)
    tokens = {h["config"].get("max_eval_tokens") for h in hists}
    if len(tokens) > 1:
        print(f"WARN {path}: seeds evaluated with different max_eval_tokens {tokens}; their accuracies are not comparable")
    steps = {tuple(h["accuracy_step"]) for h in hists}
    if len(steps) > 1:
        raise SystemExit(f"{path}: seeds have different eval schedules; cannot average them")
    return Run(label=label, hists=hists, path=path, seeds=[str(s) for s in sorted(map(int, runs))])


def optional_run(path: str, label: str) -> Optional[Run]:
    """A run if its history exists, else None (a slot the paper still reserves)."""
    return load_run(path, label) if os.path.exists(path) else None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def pm(stat: Tuple[float, Optional[float]], digits: int = 3, best: bool = False, below: Optional[float] = None) -> str:
    """``mean`` or ``mean ± std`` for LaTeX; bold when best, italic when below a reference."""
    mean, std = stat
    s = f"{mean:.{digits}f}" if std is None else f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"
    if below is not None and mean < below:
        s = f"\\emph{{{s}}}"
    if best:
        s = f"\\textbf{{{s}}}"
    return s


def _hdr(text: str) -> str:
    """A column header; a newline in ``text`` stacks it on two lines."""
    if "\n" in text:
        return "\\multicolumn{1}{c}{\\shortstack{" + " \\\\ ".join(text.split("\n")) + "}}"
    return "\\multicolumn{1}{c}{" + text + "}"


def metric_table(name: str, columns: List[Tuple[str, Optional["Run"]]], ref: "Run", at_step: int,
                 teacher_column: bool = False, header_comment: str = "") -> str:
    """Write ``tables/<name>.tex`` as a complete tabular: one column per arm, one row per metric.

    ``columns`` are (header, run) pairs; a None run is a reserved slot whose cells
    read \\pending{}. Rows: accuracy at ``at_step`` on the training family and each
    held-out family, the mean over held-out families and its peak (step), the
    peak in-distribution accuracy (step), and the in-distribution dip / recovery.
    Cells are seed mean $\\pm$ sd where a run holds several seeds. The best mean
    among the arms in a row is bold; a mean below the untrained student is italic.
    """
    sums = [(h, r, r.summary(at_step) if r is not None else None) for h, r in columns]
    ncol = 2 + len(columns) + (1 if teacher_column else 0)
    lines = ["\\begin{tabular}{l" + "c" * (ncol - 1) + "}", "  \\toprule",
             "  & " + _hdr("untrained\nstudent") + (" & " + _hdr("teacher") if teacher_column else "")
             + " & " + " & ".join(_hdr(h) for h, _ in columns) + " \\\\", "  \\midrule"]

    def row(label: str, key: str, stat: str, ref_cell: str, teacher_cell: str = "---") -> str:
        vals = [s[key][stat] if s is not None else None for _, _, s in sums]
        best = max(v[0] for v in vals if v is not None) if any(v is not None for v in vals) else None
        below = ref.baseline_of(key) if stat == "at" else None
        cells = []
        for v, (_, _, s) in zip(vals, sums):
            if v is None:
                cells.append("\\pending{}")
                continue
            c = pm(v, best=v[0] == best, below=below)
            if stat == "peak":
                c += f" ({s[key]['peak_step']})"
            cells.append(c)
        return f"  {label} & {ref_cell}" + (f" & {teacher_cell}" if teacher_column else "") + " & " + " & ".join(cells) + " \\\\"

    fam_label = {"accuracy": "22\\_add (train)", "222_add": "222\\_add", "2222_add": "2222\\_add",
                 "33_add": "33\\_add", "21_mult": "21\\_mult"}
    teacher = f"{ref.teacher:.3f}" if ref.teacher is not None else "---"
    lines.append(row(fam_label["accuracy"], "accuracy", "at", f"{ref.baseline:.3f}", teacher))
    for o in OOD:
        lines.append(row(fam_label[o], o, "at", f"{ref.baseline_of(o):.3f}"))
    lines.append(row("mean OOD", "ood", "at", f"{ref.baseline_of('ood'):.3f}"))
    lines.append("  \\midrule")
    lines.append(row("mean OOD, peak (step)", "ood", "peak", "---"))
    lines.append(row("22\\_add, peak (step)", "accuracy", "peak", "---"))
    dip = []
    for _, _, s in sums:
        if s is None:
            dip.append("\\pending{}")
        else:
            dip.append(f"{s['dip']:.3f} / {s['recovery'] if s['recovery'] is not None else '---'}")
    lines.append("  dip / recovery & ---" + (" & ---" if teacher_column else "") + " & " + " & ".join(dip) + " \\\\")
    lines += ["  \\bottomrule", "\\end{tabular}"]
    return write_table(name, lines, header_comment or f"arms as columns, metrics as rows; accuracies at step {at_step}, seed mean +- sd")


def write_table(name: str, rows: List[str], header_comment: str) -> str:
    """Write ``tables/<name>.tex`` holding only tabular rows, for ``\\input``."""
    os.makedirs(TABLE_DIR, exist_ok=True)
    path = os.path.join(TABLE_DIR, f"{name}.tex")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"% generated by experiments/plot_*.py -- do not edit by hand\n% {header_comment}\n")
        f.write("\n".join(rows) + "\n")
    print("wrote", path)
    return path


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def style(ax, title: Optional[str] = None) -> None:
    ax.set_facecolor("white")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, solid_capstyle="butt")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, length=2, width=0.8)
    if title:
        ax.set_title(title, color=INK, pad=4)


def line_with_band(ax, run: Run, key: str, colour: str, last_step: Optional[int] = None,
                   linestyle="solid", linewidth: float = 1.4, zorder: int = 3, marker: bool = False):
    """Seed-mean line, with a translucent ±1 sd band when there is more than one seed."""
    xs, mean, std = run.series(key, last_step)
    if std is not None:
        ax.fill_between(xs, mean - std, mean + std, color=colour, alpha=0.16, linewidth=0, zorder=zorder - 1)
    kw = dict(color=colour, linewidth=linewidth, linestyle=linestyle, zorder=zorder,
              solid_capstyle="round", dash_capstyle="round")
    if marker:
        kw.update(marker="o", markersize=2.0, markeredgewidth=0)
    ax.plot(xs, mean, **kw)
    return xs, mean


def rules(ax, baseline: float, teacher: Optional[float] = None) -> None:
    ax.axhline(baseline, **RULE_UNTRAINED)
    if teacher is not None:
        ax.axhline(teacher, **RULE_TEACHER)


def end_labels(ax, ends: List[List], dx: float = 1.5, fontsize: float = 6) -> None:
    """Direct labels at line ends, spread apart and kept inside the panel."""
    ends = sorted(ends, key=lambda e: e[1])
    lo, hi = ax.get_ylim()
    min_gap = 0.065 * (hi - lo)
    for i in range(1, len(ends)):
        if ends[i][1] - ends[i - 1][1] < min_gap:
            ends[i][1] = ends[i - 1][1] + min_gap
    overshoot = ends[-1][1] - (hi - 0.03 * (hi - lo))
    if overshoot > 0:
        for e in ends:
            e[1] -= overshoot
    for x, y, text in ends:
        ax.annotate(text, (x + dx, y), color=INK_2, fontsize=fontsize, ha="left", va="center")


def legend(fig, handles, labels, ncol: int, title: Optional[str] = None, y: float = -0.02):
    leg = fig.legend(handles, labels, title=title, loc="lower center", ncol=ncol, frameon=False,
                     fontsize=6.5, handlelength=2.6, columnspacing=1.4, bbox_to_anchor=(0.5, y))
    if title:
        leg.get_title().set_fontsize(6.5)
        leg.get_title().set_color(INK_2)
    for t in leg.get_texts():
        t.set_color(INK)
    return leg


def save(fig, out: str) -> None:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.replace(".pdf", ".png"), dpi=220, bbox_inches="tight", facecolor="white")
    print("wrote", out)

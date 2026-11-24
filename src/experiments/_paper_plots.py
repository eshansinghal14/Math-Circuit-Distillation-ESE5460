"""Shared loading, statistics and styling for the paper figures.

Every ``plot_*.py`` in this package reads training histories written by the
trainers in ``training/`` and renders one figure for ``latex/graph_distillation``,
writing the LaTeX table rows it reports alongside as a ``tables/*.tex`` fragment
so the numbers in the paper are generated, never transcribed.

Histories. A file's top level is whichever run wrote last; every run it holds
is under ``runs``. Since 2026-09-23 a run is keyed ``#<digest>|seed=<n>`` (older
keys carry an axis prefix, ``lambda_graph=3#<digest>|seed=1``, and the very
oldest are bare seeds). Never read the top level: :func:`select_runs` picks the
runs whose recorded config matches a filter, de-duplicates the prefixed and
unprefixed spellings of one run, and returns every seed as one :class:`Run`.
Seeds whose evaluation stopped early (a run cut off after its peak) are kept
and padded with NaN, so seed statistics are over the seeds present at a step.

Reporting convention. Under the constant learning rate every arm peaks and then
decays, at a different step per seed, so "accuracy at step 100" measures the
decay more than the method. Every table therefore reports each seed at its own
best in-distribution step (checkpoint selection on the training family's
held-out test split, which uses no out-of-distribution data), with the peak
mean OOD over the run alongside.

Palette: the validated categorical slots (blue, orange, aqua) and the one-hue
blue ramp for ordered series; ink and grid tokens for everything that is text or
scaffolding. Colour carries identity only, so every figure also has a legend and
direct labels.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

OOD = ["222_add", "2222_add", "33_add", "21_mult"]
# Task names as the paper prints them (the keys are the dataset identifiers of
# the code and histories): short forms for figure titles and table headers.
FAMILY_NAMES = {
    "22_add": "2-digit add", "222_add": "3-operand add", "2222_add": "4-operand add",
    "33_add": "3-digit add", "21_mult": "2×1-digit mult",
}
FAM_TEX = {"22_add": "2-digit add", "222_add": "3-operand add", "2222_add": "4-operand add",
           "33_add": "3-digit add", "21_mult": "$2{\\times}1$-digit mult"}
OOD_MEAN = "held-out mean"


def panel_title(key: str, train: str = "22_add") -> str:
    """Panel title for an eval series key: the task name, the training family marked."""
    if key == "accuracy":
        return f"{FAMILY_NAMES[train]} (train)"
    if key == "ood":
        return OOD_MEAN
    return FAMILY_NAMES[key]

# Categorical slots 1-3 of the validated palette; ordinal ramp for lambda.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#86b6ef", "#2a78d6", "#104281"]
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8983"
GRID = "#e9e8e5"
RULE_UNTRAINED = dict(color=INK_3, linewidth=0.9, linestyle=(0, (2, 2)), zorder=1)
RULE_TEACHER = dict(color=INK_3, linewidth=0.9, linestyle=(0, (1, 2)), zorder=1)
CONTROL = dict(color=INK_2, linewidth=1.2, linestyle=(0, (4, 2)), dash_capstyle="round", zorder=2)
DOT = (0, (1.2, 1.6))

# Type sits one step under the 9pt caption at print size (figures are drawn at
# their printed width, so no scaling happens in LaTeX).
RC = {
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.titlesize": 8, "pdf.fonttype": 42,
    "legend.fontsize": 7,
    # STIX ships with matplotlib and is metric-compatible with the Times body
    # text the ICLR style sets, so figure text matches the page.
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "Times"],
    "mathtext.fontset": "stix",
}

FIG_DIR = "latex/graph_distillation/figures"
TABLE_DIR = "latex/graph_distillation/tables"

# The runs the paper is built from, as (history file, config filter). Every
# script imports these rather than spelling paths, so a re-run that lands in a
# different file changes one line.
R = "results"
ANOVA_LABELS = ["arg1 range", "arg1 units", "arg2 range", "arg2 units", "sum range", "sum units"]
# Shared optimiser protocol of every arm in the paper (2026-09-25): bf16, lr 1e-5,
# 10 warmup steps, cosine to 0.1 of the peak, 100 steps, BOS-free training (the
# trainer's bos_mode "legacy": no BOS in the training sequence, BOS at evaluation).
PROTOCOL = dict(dataset="22_add", steps=100, learning_rate=1e-5, dtype="bfloat16",
                warmup_steps=10, lr_floor=0.1, bos_mode="legacy")
KD_PROTOCOL = dict(PROTOCOL, temperature=1.0, kl_tokens="all")
GRAPH_FIXED = dict(nodes_per_label=10, teacher_prop_neurons_per_layer=0.1, graph_loss_type="jsd",
                   graph_node_labels=ANOVA_LABELS)
GRAPH_SWEEP = f"{R}/anova_study/22_add_anova-sweep.json"
# The main arm has its own three-seed folder; GRAPH_SWEEP keeps the one-seed screen
# points the ablation panels read.
GRAPH_FINAL = f"{R}/anova_study/22_add_anova-final.json"
GRAPH_MAIN = (GRAPH_FINAL, dict(KD_PROTOCOL, **GRAPH_FIXED, lambda_graph=3.0, scramble_teacher_graph=False))
SCRAMBLED_MAIN = (f"{R}/anova_study/22_add_anova-scramble.json",
                  dict(KD_PROTOCOL, **GRAPH_FIXED, lambda_graph=3.0, scramble_teacher_graph=True))
KD_SWEEP = f"{R}/standard_kd/22_add-sweep.json"
KD_MAIN = (KD_SWEEP, KD_PROTOCOL)
# Equal-compute baselines (Section 4.2): standard KD with the cosine schedule
# stretched over a longer budget, one run each. 280 and 1400 steps are the KD
# budgets whose FLOPs match the graph arm at its peak step (20) and at step 100.
KD_STRETCHED = {n: (KD_SWEEP, dict(KD_PROTOCOL, steps=n)) for n in (280, 1400)}
BERT_SWEEP = f"{R}/baselines/22_add_bert-sweep.json"
CKA_SWEEP = f"{R}/baselines/22_add_cka-sweep.json"
BERT_MAIN = (f"{R}/baselines/22_add_bert-final.json", dict(KD_PROTOCOL, lambda_rep=0.3))
CKA_MAIN = (f"{R}/baselines/22_add_cka-final.json", dict(KD_PROTOCOL, lambda_rep=0.3))
# SFT has no KD term, so its config carries no temperature or kl_tokens.
SFT_MAIN = (f"{R}/sft/22_add.json", PROTOCOL)
# The second student (Section 4.3): Llama-3.2-3B-Instruct under the identical
# protocol, lambda and supernode categories; standard KD is its only baseline.
# The 1B files hold only 1B runs, so the main specs need no model filter.
STUDENT_1B = "meta-llama/Llama-3.2-1B-Instruct"
STUDENT_3B = "meta-llama/Llama-3.2-3B-Instruct"
KD_SWEEP_3B = f"{R}/standard_kd/3b_22_add-sweep.json"
KD_MAIN_3B = (KD_SWEEP_3B, dict(KD_PROTOCOL, model=STUDENT_3B))
GRAPH_MAIN_3B = (f"{R}/anova_study/3b_22_add_anova-final.json",
                 dict(KD_PROTOCOL, **GRAPH_FIXED, model=STUDENT_3B, lambda_graph=3.0, scramble_teacher_graph=False))
# Kept for the old constant-lr runs; nothing in the paper reads them.
MAIN_PROTOCOL = KD_PROTOCOL


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
    def dataset(self) -> str:
        return self.config.get("dataset", "22_add")

    @property
    def ood(self) -> List[str]:
        """Held-out families: every family evaluated that is not the training one."""
        return [o for o in OOD if o != self.dataset and f"accuracy_{o}" in self.hists[0]]

    @property
    def steps(self) -> np.ndarray:
        longest = max(self.hists, key=lambda h: len(h["accuracy_step"]))
        return np.asarray(longest["accuracy_step"])

    @property
    def baseline(self) -> float:
        return float(self.hists[0]["student_baseline"])

    @property
    def teacher(self) -> Optional[float]:
        t = self.hists[0].get("teacher_baseline")
        return None if t is None else float(t)

    def teacher_of(self, key: str) -> Optional[float]:
        if key == "accuracy":
            return self.teacher
        if key == "ood":
            vals = [self.hists[0].get(f"teacher_baseline_{o}") for o in self.ood]
            return None if any(v is None for v in vals) else float(np.mean(vals))
        t = self.hists[0].get(f"teacher_baseline_{key}")
        return None if t is None else float(t)

    def _key_series(self, h: Dict[str, Any], key: str) -> np.ndarray:
        n = len(self.steps)

        def padded(name: str) -> np.ndarray:
            v = np.asarray(h[name], dtype=float)
            out = np.full(n, np.nan)
            out[: len(v)] = v
            return out

        if key == "ood":
            if not self.ood:
                return np.full(n, np.nan)
            return np.nanmean(np.stack([padded(f"accuracy_{o}") for o in self.ood]), axis=0)
        if key == "accuracy":
            return padded("accuracy")
        return padded(f"accuracy_{key}")

    def matrix(self, key: str) -> np.ndarray:
        """[seed, step] array of an eval series, NaN where a seed has no eval."""
        return np.stack([self._key_series(h, key) for h in self.hists])

    def series(self, key: str, last_step: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """(steps, seed mean, seed std or None) of an eval series; ``key`` is
        ``accuracy``, ``ood`` or a family name. Statistics are over the seeds
        that have an eval at each step."""
        ys = self.matrix(key)
        xs = self.steps
        if last_step is not None:
            keep = xs <= last_step
            xs, ys = xs[keep], ys[:, keep]
        with np.errstate(all="ignore"):
            mean = np.nanmean(ys, axis=0)
            std = None
            if self.n > 1:
                cnt = np.sum(~np.isnan(ys), axis=0)
                std = np.where(cnt > 1, np.nanstd(ys, axis=0, ddof=1), np.nan)
        return xs, mean, std

    def step_series(self, key: str, seeds: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """(train steps, seed mean) of a per-step training metric; ``seeds=True``
        returns the [seed, step] matrix (NaN-padded) instead of the mean."""
        longest = max(self.hists, key=lambda h: len(h["train_step"]))
        n = len(longest["train_step"])
        ys = np.full((self.n, n), np.nan)
        for i, h in enumerate(self.hists):
            v = np.asarray(h[key], dtype=float)
            ys[i, : len(v)] = v
        xs = np.asarray(longest["train_step"])
        if seeds:
            return xs, ys
        with np.errstate(all="ignore"):
            return xs, np.nanmean(ys, axis=0)

    def baseline_of(self, key: str) -> float:
        xs, mean, _ = self.series(key)
        assert xs[0] == 0, "history has no step-0 eval"
        return float(mean[0])

    def summary(self, until: Optional[int] = None) -> Dict[str, Any]:
        """Seed-level statistics used by every table, over steps <= ``until``.

        ``sel``: each seed at its own best in-distribution step (mean, sd), and
        ``sel_steps`` those steps. ``peak``: each seed's own maximum of the series
        (mean, sd) with the step at which the seed-mean curve peaks. ``last``:
        the last eval each seed has.
        """
        xs = self.steps
        keep = np.ones_like(xs, dtype=bool) if until is None else xs <= until
        acc = self.matrix("accuracy")[:, keep]
        xk = xs[keep]
        i_sel = [int(np.nanargmax(a)) for a in acc]
        out: Dict[str, Any] = {"n": self.n, "sel_steps": [int(xk[i]) for i in i_sel],
                               "last_steps": [int(xk[int(np.max(np.where(~np.isnan(a))))]) for a in acc]}

        def stat(v: np.ndarray) -> Tuple[float, Optional[float]]:
            v = v[~np.isnan(v)]
            if len(v) == 0:
                return float("nan"), None
            return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else None)

        for key in ["accuracy", "ood"] + self.ood:
            ys = self.matrix(key)[:, keep]
            with np.errstate(all="ignore"):
                mean = np.nanmean(ys, axis=0)
            sel = np.array([ys[s, i] for s, i in enumerate(i_sel)])
            last = np.array([ys[s, int(np.max(np.where(~np.isnan(ys[s]))))] if np.any(~np.isnan(ys[s])) else np.nan
                             for s in range(self.n)])
            out[key] = {
                "sel": stat(sel),
                "peak": stat(np.nanmax(ys, axis=1)),
                "peak_step": int(xk[int(np.nanargmax(mean))]) if np.any(~np.isnan(mean)) else -1,
                "last": stat(last),
            }
        return out


VOLATILE = {"seed", "save_dir", "device", "torch_version", "transformers_version", "eval_batch_size", "resume",
            "save_every_n_steps", "graph_verbose", "mlp_cache_batch_size", "teacher_target_cache_dir",
            "anova_cache_device", "kl_token_chunk_size", "token_path_micro_batch", "token_path_micro_tokens",
            "teacher_graph_batch_size", "student_graph_batch_size", "track_flops", "track_grad_metrics",
            "student_n_params", "optimizer_class", "eval_every_n_steps"}


def config_ident(cfg: Dict[str, Any]) -> str:
    """A run's identity: its config without the fields that do not change the result."""
    return json.dumps({k: v for k, v in cfg.items() if k not in VOLATILE}, sort_keys=True)


def _parse_key(key: str) -> Tuple[str, str]:
    """(identity, seed) of a run key: ``#<digest>|seed=<n>`` with any axis prefix
    stripped, or a bare seed for the oldest files."""
    m = re.search(r"#([0-9a-f]+)\|seed=(\d+)$", key)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r"^(.*)\|seed=(\d+)$", key)
    if m:
        return m.group(1), m.group(2)
    return "", key


def _runs_in(hist: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    runs = hist.get("runs")
    if isinstance(runs, dict) and runs:
        return dict(runs)
    return {str(hist.get("config", {}).get("seed", 42)): {k: v for k, v in hist.items() if k != "runs"}}


def _matches(config: Dict[str, Any], where: Dict[str, Any]) -> bool:
    for k, v in where.items():
        c = config.get(k)
        if isinstance(v, float) or isinstance(c, float):
            if c is None or not np.isclose(float(c), float(v), rtol=1e-6):
                return False
        elif c != v:
            return False
    return True


def select_runs(path: str, where: Dict[str, Any], label: str = "", seeds: Optional[Sequence[int]] = None,
                required: bool = True) -> Optional[Run]:
    """Every run in ``path`` whose config matches ``where``, one seed each, as a :class:`Run`.

    A run stored under both a prefixed and an unprefixed key counts once. When
    the filter matches two different configs for one seed the call fails, since
    that means the filter is under-specified. ``required=False`` returns None
    for a missing file or an empty match (a slot the paper still reserves).
    """
    if not os.path.exists(path):
        if required:
            raise SystemExit(f"{path}: missing")
        return None
    with open(path, encoding="utf-8") as f:
        hist = json.load(f)
    chosen: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for key, h in _runs_in(hist).items():
        cfg = h.get("config")
        if not cfg or not _matches(cfg, where):
            continue
        _, seed = _parse_key(key)
        ident = config_ident(cfg)
        seed = str(cfg.get("seed", seed))
        if seeds is not None and int(seed) not in [int(s) for s in seeds]:
            continue
        if seed in chosen and chosen[seed][0] != ident:
            raise SystemExit(f"{path}: filter {where} matches two configs for seed {seed} "
                             f"({chosen[seed][0]} and {ident}); add a field to the filter")
        # Prefer the longer history when the same run is stored twice.
        if seed not in chosen or len(h["accuracy_step"]) > len(chosen[seed][1]["accuracy_step"]):
            chosen[seed] = (ident, h)
    if not chosen:
        if required:
            raise SystemExit(f"{path}: no run matches {where}")
        return None
    order = sorted(chosen, key=int)
    hists = [chosen[s][1] for s in order]
    tokens = {h["config"].get("max_eval_tokens") for h in hists}
    if len(tokens) > 1:
        print(f"WARN {path}: seeds evaluated with different max_eval_tokens {tokens}")
    for h in hists[1:]:
        a, b = h["accuracy_step"], hists[0]["accuracy_step"]
        n = min(len(a), len(b))
        if list(a[:n]) != list(b[:n]):
            raise SystemExit(f"{path}: seeds have different eval schedules; cannot average them")
    return Run(label=label, hists=hists, path=path, seeds=order)


def load_run(path: str, label: str) -> Run:
    """Every run in a file regardless of config (old single-config files)."""
    return select_runs(path, {}, label)


def all_runs(path: str) -> List[Run]:
    """Every distinct config in a sweep file as its own single-config :class:`Run`."""
    with open(path, encoding="utf-8") as f:
        hist = json.load(f)
    by_ident: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for key, h in _runs_in(hist).items():
        _, seed = _parse_key(key)
        ident = config_ident(h["config"])
        seed = str(h["config"].get("seed", seed))
        cur = by_ident.setdefault(ident, {})
        if seed not in cur or len(h["accuracy_step"]) > len(cur[seed]["accuracy_step"]):
            cur[seed] = h
    out = []
    for ident, runs in by_ident.items():
        order = sorted(runs, key=int)
        out.append(Run(label=ident, hists=[runs[s] for s in order], path=path, seeds=order))
    return out


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


def hdr(text: str) -> str:
    """A column header; a newline in ``text`` stacks it on two lines."""
    if "\n" in text:
        return "\\multicolumn{1}{c}{\\shortstack{" + " \\\\ ".join(text.split("\n")) + "}}"
    return "\\multicolumn{1}{c}{" + text + "}"


def steps_str(steps: List[int]) -> str:
    return "/".join(str(s) for s in steps)


def metric_table(name: str, columns: List[Tuple[str, Optional[Run]]], ref: Run, until: Optional[int] = None,
                 teacher_column: bool = True, header_comment: str = "", bold: bool = True) -> str:
    """Write ``tables/<name>.tex`` as a complete tabular: one column per arm, one row per metric.

    ``columns`` are (header, run) pairs; a None run is a reserved slot whose cells
    read \\pending{}. Rows: peak in-distribution accuracy with each seed's step,
    then each held-out family and their mean read at that step, then the peak
    mean OOD over the run (seed-mean curve's step). Cells are seed mean $\\pm$ sd
    where a run holds several seeds. The best mean among the arms in a row is
    bold; a mean below the untrained student is italic.
    """
    sums = [(h, r, r.summary(until) if r is not None else None) for h, r in columns]
    ncol = 2 + len(columns) + (1 if teacher_column else 0)
    lines = ["\\begin{tabular}{l" + "c" * (ncol - 1) + "}", "  \\toprule",
             "  & " + hdr("untrained\nstudent") + (" & " + hdr("teacher") if teacher_column else "")
             + " & " + " & ".join(hdr(h) for h, _ in columns) + " \\\\", "  \\midrule"]

    def row(label: str, key: str, stat: str, ref_cell: str, teacher_cell: str = "---", with_step: bool = False) -> str:
        vals = [s[key][stat] if s is not None else None for _, _, s in sums]
        best = max(v[0] for v in vals if v is not None) if any(v is not None for v in vals) else None
        below = ref.baseline_of(key) if stat == "sel" else None
        cells = []
        for v, (_, _, s) in zip(vals, sums):
            if v is None:
                cells.append("\\pending{}")
                continue
            c = pm(v, best=bold and v[0] == best, below=below)
            if with_step and stat == "sel":
                c += f" ({steps_str(s['sel_steps'])})"
            if with_step and stat == "peak":
                c += f" ({s[key]['peak_step']})"
            cells.append(c)
        return f"  {label} & {ref_cell}" + (f" & {teacher_cell}" if teacher_column else "") + " & " + " & ".join(cells) + " \\\\"

    def t(key):
        v = ref.teacher_of(key)
        return "---" if v is None else f"{v:.3f}"

    lines.append(row(FAM_TEX[ref.dataset] + " (train)", "accuracy", "sel", f"{ref.baseline:.3f}", t("accuracy")))
    steps_cells = [("\\pending{}" if s is None else steps_str(s["sel_steps"])) for _, _, s in sums]
    lines.append("  \\quad selected step, per seed & ---" + (" & ---" if teacher_column else "") + " & " + " & ".join(steps_cells) + " \\\\")
    for o in ref.ood:
        lines.append(row(FAM_TEX[o], o, "sel", f"{ref.baseline_of(o):.3f}", t(o)))
    lines.append(row(OOD_MEAN, "ood", "sel", f"{ref.baseline_of('ood'):.3f}", t("ood")))
    lines.append("  \\midrule")
    lines.append(row(f"peak {OOD_MEAN} (step)", "ood", "peak", "---", "---", with_step=True))
    lines.append(row(FAM_TEX[ref.dataset] + " (train), last step", "accuracy", "last", "---", "---"))
    lines += ["  \\bottomrule", "\\end{tabular}"]
    return write_table(name, lines, header_comment or "arms as columns, metrics as rows; each seed at its own best in-distribution step, seed mean +- sd")


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
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelcolor=INK, length=2.5, width=0.6)
    if title:
        ax.set_title(title, color=INK, pad=4)


def line_with_band(ax, run: Run, key: str, colour: str, last_step: Optional[int] = None,
                   linestyle="solid", linewidth: float = 1.4, zorder: int = 3, marker: bool = False):
    """Seed-mean line, with a translucent ±1 sd band when there is more than one seed."""
    xs, mean, std = run.series(key, last_step)
    if std is not None:
        ok = ~np.isnan(std)
        ax.fill_between(xs[ok], (mean - std)[ok], (mean + std)[ok], color=colour, alpha=0.16, linewidth=0, zorder=zorder - 1)
    kw = dict(color=colour, linewidth=linewidth, linestyle=linestyle, zorder=zorder,
              solid_capstyle="round", dash_capstyle="round")
    if marker:
        kw.update(marker="o", markersize=2.0, markeredgewidth=0)
    ax.plot(xs, mean, **kw)
    return xs, mean


def seed_lines(ax, run: Run, key: str, colour: str, last_step: Optional[int] = None,
               linestyle="solid", linewidth: float = 0.7, zorder: int = 3):
    """Every seed as its own thin line, and no band.

    For an arm whose seeds do not cluster, a mean with a +-1 sd band describes an
    outcome no seed produced: the scrambled control has two seeds that never leave
    the untrained student and one that trains, so its mean sits in the empty middle
    and the band spans almost the whole axis. Drawing the seeds is both honest and
    quieter. Returns the last point of the seed that ran furthest, for the end
    label.
    """
    xs = run.steps
    ys = run.matrix(key)
    if last_step is not None:
        keep = xs <= last_step
        xs, ys = xs[keep], ys[:, keep]
    for row in ys:
        ok = ~np.isnan(row)
        ax.plot(xs[ok], row[ok], color=colour, linewidth=linewidth, linestyle=linestyle,
                zorder=zorder, alpha=0.85, solid_capstyle="round", dash_capstyle="round")
    with np.errstate(all="ignore"):
        # median, not mean: the caller uses this only to place the end label, and
        # with three seeds the median sits on one of the drawn lines whereas the
        # mean of a bimodal arm sits in the gap between them.
        mid = np.nanmedian(ys, axis=0)
    return xs, mid


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


def reference_handles(teacher: bool = True, student: str = "untrained student"):
    """Legend entries for the two reference rules every accuracy panel draws, so
    the rules are named in the legend rather than by text laid over the lines."""
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], **{k: v for k, v in RULE_UNTRAINED.items() if k != "zorder"})]
    labels = [student]
    if teacher:
        handles.append(Line2D([], [], **{k: v for k, v in RULE_TEACHER.items() if k != "zorder"}))
        labels.append("teacher")
    return handles, labels


def legend(fig, handles, labels, ncol: int, title: Optional[str] = None, y: float = -0.02):
    leg = fig.legend(handles, labels, title=title, loc="lower center", ncol=ncol, frameon=False,
                     fontsize=RC["legend.fontsize"], handlelength=2.6, columnspacing=1.4, bbox_to_anchor=(0.5, y))
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

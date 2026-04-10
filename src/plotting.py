import argparse
import json
import os
import re
import sys
from typing import Any, Collection, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

_NEURIPS_RC: Dict[str, Any] = {
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "DejaVu Serif",
        "Bitstream Vera Serif",
        "Computer Modern Roman",
    ],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "mathtext.fontset": "dejavuserif",
}


def default_plots_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "plots"))


def default_neuron_cossim_topk_dir() -> str:
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "results",
            "circuit-discovery",
            "neuron_cossim_topk",
        ),
    )


def repo_root() -> str:
    """Parent of ``src/`` (directory containing ``plots/``, ``results/``)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _json_files_in_folder(folder: str) -> List[str]:
    root = os.path.abspath(folder)
    if not os.path.isdir(root):
        raise NotADirectoryError(f"Not a directory: {root}")
    names = [f for f in os.listdir(root) if f.lower().endswith(".json")]
    names.sort()
    return [os.path.join(root, f) for f in names]


def _safe_stem(name: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", name.strip())
    return s or "curve"


def _load_training_history(path: str) -> Dict[str, List]:
    """Load JSON as column-oriented dict ``{key: [values per step]}``."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        if not data:
            return {}
        keys = list(data[0].keys())
        return {k: [row.get(k) for row in data] for k in keys}
    if isinstance(data, dict):
        return data
    raise TypeError(f"Expected list or dict in {path!r}, got {type(data)}")


def _series_x_y(
    hist: Dict[str, List],
    metric_key: str,
    x_key: str,
) -> Tuple[np.ndarray, np.ndarray]:
    y = hist.get(metric_key)
    if y is None:
        raise KeyError(f"Metric {metric_key!r} not in history (keys: {list(hist.keys())!r})")
    x = hist.get(x_key)
    if x is None or len(x) != len(y):
        x = list(range(len(y)))

    def _num(v: Any) -> float:
        if v is None:
            return float("nan")
        return float(v)

    xs = np.array([_num(v) for v in x], dtype=float)
    ys = np.array([_num(v) for v in y], dtype=float)
    return xs, ys


def _trailing_moving_average(
    xs: np.ndarray,
    ys: np.ndarray,
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Trailing (causal) moving average; ``x`` aligned to the end of each window."""
    w = int(window)
    if w <= 1 or len(ys) < w:
        return xs, ys
    out_y = np.array(
        [float(np.nanmean(ys[i : i + w])) for i in range(len(ys) - w + 1)],
        dtype=float,
    )
    out_x = xs[w - 1 :]
    return out_x, out_y


def _format_param_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def plot_training_histories_param_sweep(
    json_folder: str,
    param_name: str,
    param_values: List[Any],
    metrics: List[str],
    *,
    x_key: str = "epoch",
    output_dir: Optional[str] = None,
    figure_width_in: float = 5.5,
    figure_height_in: float = 2.4,
    save_pdf: bool = True,
    save_png: bool = True,
    smooth_window: Optional[int] = None,
    smooth_metrics: Optional[Collection[str]] = None,
) -> str:
    """Plot several training histories (one JSON per run) as multiple lines per subplot.

    All ``*.json`` files in ``json_folder`` are loaded (sorted by filename). One colored
    line per file; legend ``{param_name} = value``.
    """
    if len(metrics) == 0:
        raise ValueError("metrics must be non-empty")
    if not save_pdf and not save_png:
        raise ValueError("At least one of save_pdf, save_png must be True")
    sw = smooth_window
    if sw is not None and sw <= 1:
        sw = None
    if sw is not None:
        if smooth_metrics is None:
            smooth_set: Set[str] = {"accuracy"}
        else:
            smooth_set = set(smooth_metrics)
    else:
        smooth_set = set()

    json_paths = _json_files_in_folder(json_folder)
    if not json_paths:
        raise ValueError(f"No .json files found in {os.path.abspath(json_folder)!r}")
    if len(json_paths) != len(param_values):
        raise ValueError(
            f"Found {len(json_paths)} JSON file(s) in folder but param_values has length "
            f"{len(param_values)}; they must match (one value per file, sorted filename order).",
        )
    loaded: List[Dict[str, List]] = []
    for p in json_paths:
        loaded.append(_load_training_history(p))

    out_dir = os.path.abspath(output_dir) if output_dir else default_plots_dir()
    os.makedirs(out_dir, exist_ok=True)
    base = _safe_stem(param_name) + "_testing"
    stem = os.path.join(out_dir, base)

    n = len(metrics)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(loaded))]

    with plt.rc_context(_NEURIPS_RC):
        fig, axes = plt.subplots(
            1,
            n,
            figsize=(figure_width_in, figure_height_in),
            squeeze=False,
        )
        ax_row = axes[0, :]

        legend_handles: Optional[List[Any]] = None
        legend_labels: Optional[List[str]] = None

        for mi, metric in enumerate(metrics):
            ax = ax_row[mi]
            do_smooth = sw is not None and metric in smooth_set
            for j, hist in enumerate(loaded):
                xs, ys = _series_x_y(hist, metric, x_key)
                if do_smooth:
                    assert sw is not None
                    xs, ys = _trailing_moving_average(xs, ys, sw)
                label = f"{param_name} = {_format_param_value(param_values[j])}"
                (line,) = ax.plot(
                    xs,
                    ys,
                    color=colors[j],
                    label=label,
                    clip_on=False,
                    linewidth=1.2,
                )
                if mi == 0:
                    if legend_handles is None:
                        legend_handles = []
                        legend_labels = []
                    legend_handles.append(line)
                    legend_labels.append(label)

            ax.set_xlabel(x_key.replace("_", " "))
            ylab = metric.replace("_", " ")
            if do_smooth and sw is not None:
                ylab = f"{ylab} (MA-{sw})"
            ax.set_ylabel(ylab)

        if legend_handles and legend_labels:
            ncol = min(len(legend_labels), 4)
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=ncol,
                frameon=False,
                handlelength=2.2,
                columnspacing=1.0,
            )
            fig.subplots_adjust(bottom=0.26, top=0.94, left=0.07, right=0.98, wspace=0.35)
        else:
            fig.tight_layout()

        if save_pdf:
            fig.savefig(f"{stem}.pdf", format="pdf")
        if save_png:
            fig.savefig(f"{stem}.png", format="png", dpi=300)
        plt.close(fig)

    primary = f"{stem}.pdf" if save_pdf else f"{stem}.png"
    return primary


def _mean_topk_curve(cossim: Sequence[float], num_points: int) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(cossim, dtype=np.float64)
    d = int(arr.size)
    if d < 1:
        raise ValueError("mean_pairwise_cossim is empty")
    c_desc = np.sort(arr)[::-1]
    cumsum = np.cumsum(c_desc)
    ks = np.arange(1, d + 1, dtype=np.float64)
    fracs = ks / float(d)
    means = cumsum / ks
    n = min(max(num_points, 2), d)
    idx = np.linspace(0, d - 1, n, dtype=int)
    return fracs[idx], means[idx]


def plot_frac_activated_vs_mean_cossim(
    json_folder: Optional[str] = None,
    *,
    json_paths: Optional[Sequence[str]] = None,
    towers: Collection[str] = ("1b", "8b"),
    output_dir: Optional[str] = None,
    out_name: str = "frac_activated_vs_mean_cossim",
    figure_width_in: float = 5.5,
    figure_height_in: float = 2.8,
    num_points: int = 800,
    anchor_zero: bool = True,
    x_as_percent: bool = True,
    save_pdf: bool = True,
    save_png: bool = True,
) -> str:
    if not save_pdf and not save_png:
        raise ValueError("At least one of save_pdf, save_png must be True")
    if json_paths is not None:
        paths = [os.path.abspath(p) for p in json_paths]
    else:
        root = json_folder if json_folder is not None else default_neuron_cossim_topk_dir()
        paths = _json_files_in_folder(root)
    if not paths:
        raise ValueError("No .json files to plot")

    out_dir = os.path.abspath(output_dir) if output_dir else default_plots_dir()
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, _safe_stem(out_name))

    tower_list = [t.lower() for t in towers]
    for t in tower_list:
        if t not in ("1b", "8b"):
            raise ValueError(f"Unknown tower {t!r}; use '1b' and/or '8b'")

    cmap = plt.get_cmap("tab10")
    linestyles = {"1b": "-", "8b": "--"}

    with plt.rc_context(_NEURIPS_RC):
        fig, ax = plt.subplots(figsize=(figure_width_in, figure_height_in))

        for pi, path in enumerate(paths):
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
            label_base = rec.get("dataset_prefix") or os.path.splitext(os.path.basename(path))[0]
            for tower in tower_list:
                block = rec.get(tower)
                if not isinstance(block, dict):
                    continue
                cossim = block.get("mean_pairwise_cossim")
                if not isinstance(cossim, list) or len(cossim) == 0:
                    continue
                fracs, means = _mean_topk_curve(cossim, num_points)
                if anchor_zero:
                    fracs = np.concatenate([[0.0], fracs])
                    means = np.concatenate([[0.0], means])
                xs = fracs * 100.0 if x_as_percent else fracs
                color = cmap(pi % 10)
                ax.plot(
                    xs,
                    means,
                    color=color,
                    linestyle=linestyles[tower],
                    label=f"{label_base} ({tower})",
                    clip_on=False,
                )

        ax.set_xlabel("Percent of neurons activated" if x_as_percent else "Fraction of neurons activated")
        ax.set_ylabel(r"Mean pairwise cosine similarity (top-$k$)")
        if not x_as_percent:
            ax.set_xlim(0.0, 1.0)
        else:
            ax.set_xlim(0.0, 100.0)
        ax.set_ylim(bottom=0.0)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ncol = min(len(handles), 3)
            ax.legend(
                handles,
                labels,
                loc="best",
                frameon=False,
                ncol=ncol,
                handlelength=2.4,
            )
        fig.tight_layout()

        if save_pdf:
            fig.savefig(f"{stem}.pdf", format="pdf")
        if save_png:
            fig.savefig(f"{stem}.png", format="png", dpi=300)
        plt.close(fig)

    primary = f"{stem}.pdf" if save_pdf else f"{stem}.png"
    return primary


def plot_k_vs_loss(model_name: str) -> None:
    base_dir = os.path.join("..", "results", "neuron-clustering", model_name)
    json_path = os.path.join(base_dir, "k_gs_testing.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find {json_path}. Run neuron_clustering.py first to generate it.")

    with open(json_path, "r") as f:
        k_gs_testing = json.load(f)

    os.makedirs(base_dir, exist_ok=True)

    for subclass_str, k_dict in k_gs_testing.items():
        ks = sorted(int(k) for k in k_dict.keys())
        losses = [k_dict[str(k)] for k in ks]

        plt.figure(figsize=(6, 4))
        plt.plot(ks, losses, marker="o")
        plt.xlabel("k (number of clusters)")
        plt.ylabel("Mean cosine distance to centroids (loss)")
        plt.title(f"k-means loss vs k for {model_name.split('/')[1]}, subclass {subclass_str}")
        plt.grid(True, alpha=0.3)

        out_path = os.path.join(base_dir, f"k_vs_loss_subclass_{subclass_str}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    plot_training_histories_param_sweep(
        json_folder="results/distillation/2d_add/lambda_cluster",
        param_name="lambda_cluster",
        param_values=[0.1, 0.5, 1.0, 5.0],
        metrics=["kl_loss", "cluster_loss", "accuracy"],
        smooth_window=10,
    )

    plot_training_histories_param_sweep(
        json_folder="results/distillation/2d_add/frac_activated",
        param_name="frac_activated",
        param_values=[0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
        metrics=["kl_loss", "cluster_loss", "accuracy"],
        smooth_window=10,
    )

    plot_frac_activated_vs_mean_cossim(
        json_folder="results/circuit-discovery/neuron_cossim_topk",
        out_name="frac_activated_vs_mean_cossim",
        figure_width_in=5.5,
        figure_height_in=2.8,
        num_points=800,
        anchor_zero=True,
        x_as_percent=True,
        save_pdf=True,
        save_png=True,
    )


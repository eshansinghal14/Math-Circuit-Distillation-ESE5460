import argparse
import json
import os
import re
import sys
from typing import Any, Collection, Dict, List, Optional, Sequence, Set, TextIO, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from cka_loss import stable_rank_centered_gram
from utils import NEURON_CLUSTERING_STUDENT_SUBPATH, NEURON_CLUSTERING_TEACHER_SUBPATH

# Paths passed into this module should be absolute (or cwd-relative); nothing here assumes a repo ``src/`` root.

_TOWER_NEURON_CLUSTER_SUBPATH: Dict[str, str] = {
    "1b": NEURON_CLUSTERING_STUDENT_SUBPATH,
    "8b": NEURON_CLUSTERING_TEACHER_SUBPATH,
}

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


def _output_dir_or_cwd(output_dir: Optional[str]) -> str:
    """Save figures here; if ``output_dir`` is omitted, use the process current working directory."""
    return os.path.abspath(output_dir) if output_dir else os.path.abspath(os.getcwd())


def _plotting_src_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _repo_root_from_plotting() -> str:
    """Repository root (parent of ``src/``)."""
    return os.path.abspath(os.path.join(_plotting_src_dir(), ".."))


def _load_random_ablation_xy(path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return sorted ``(fraction_ablated, performance_drop)`` or ``None`` if missing/empty."""
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    pts = data.get("points")
    if not isinstance(pts, list) or not pts:
        return None
    xs = np.array([float(p["fraction_ablated"]) for p in pts], dtype=np.float64)
    ys = np.array([float(p["performance_drop"]) for p in pts], dtype=np.float64)
    order = np.argsort(xs)
    return xs[order], ys[order]


def _poly_regression_curve(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    max_deg: int = 16,
    grid_n: int = 256,
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """Least-squares polynomial fit; returns ``(x_grid, y_grid, degree_used)`` or ``None``."""
    n = int(xs.shape[0])
    if n < 2:
        return None
    deg = min(max_deg, n - 1)
    deg = max(1, deg)
    coef = np.polyfit(xs.astype(np.float64), ys.astype(np.float64), deg=deg)
    xg = np.linspace(float(xs.min()), float(xs.max()), grid_n)
    yg = np.polyval(coef, xg)
    return xg, yg, deg


def save_random_ablation_1b_8b_plot() -> None:
    """Plot 1B vs 8B random-ablation JSONs and one polynomial regression curve per model; save under ``plots/``.

    Reads (no arguments):

    - ``src/experiments/random_ablation_vs_fraction/results/random_ablation_vs_fraction_1b.json``
    - ``src/experiments/random_ablation_vs_fraction/results/random_ablation_vs_fraction_8b.json``

    After each experiment run, copy ``random_ablation_vs_fraction.json`` to the matching
    ``*_1b.json`` / ``*_8b.json`` filename so both models are present.

    Saves ``plots/random_ablation_1b_8b.png``. For each available JSON, draws a least-squares
    polynomial (degree at most 3, capped by number of points) fit to that model’s points alone;
    both fit curves are drawn in black (scatters stay blue / orange).
    """
    res_dir = os.path.join(
        _plotting_src_dir(), "experiments", "random_ablation_vs_fraction", "results",
    )
    p1 = os.path.join(res_dir, "random_ablation_vs_fraction_1b.json")
    p8 = os.path.join(res_dir, "random_ablation_vs_fraction_8b.json")
    out_dir = os.path.join(_repo_root_from_plotting(), "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "random_ablation_1b_8b.png")

    xy1 = _load_random_ablation_xy(p1)
    xy8 = _load_random_ablation_xy(p8)
    if xy1 is None:
        print(f"save_random_ablation_1b_8b_plot: missing or empty {p1!r}")
    if xy8 is None:
        print(f"save_random_ablation_1b_8b_plot: missing or empty {p8!r}")
    if xy1 is None and xy8 is None:
        print("save_random_ablation_1b_8b_plot: nothing to plot; skipping save.")
        return

    with plt.rc_context(_NEURIPS_RC):
        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        if xy1 is not None:
            x1, y1 = xy1
            ax.scatter(
                x1,
                y1,
                s=16,
                alpha=0.45,
                c="tab:blue",
                edgecolors="none",
                label="1B",
                zorder=2,
            )
        if xy8 is not None:
            x8, y8 = xy8
            ax.scatter(
                x8,
                y8,
                s=16,
                alpha=0.45,
                c="tab:orange",
                edgecolors="none",
                label="8B",
                zorder=2,
            )

        _max_poly_deg = 8
        if xy1 is not None:
            fit1 = _poly_regression_curve(x1, y1, max_deg=_max_poly_deg)
            if fit1 is not None:
                xg, yg, deg_used = fit1
                ax.plot(
                    xg,
                    yg,
                    color="black",
                    linewidth=1.4,
                    zorder=3,
                    label=f"1B poly fit (deg {deg_used})",
                )
        if xy8 is not None:
            fit8 = _poly_regression_curve(xy8[0], xy8[1], max_deg=_max_poly_deg)
            if fit8 is not None:
                xg, yg, deg_used = fit8
                ax.plot(
                    xg,
                    yg,
                    color="black",
                    linewidth=1.4,
                    zorder=3,
                    label=f"8B poly fit (deg {deg_used})",
                )
        ax.set_xlabel("Fraction of neurons ablated")
        ax.set_ylabel("Performance drop (baseline − accuracy)")
        ax.set_title("Random ablation: 1B vs 8B")
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    print(f"Saved {out_path}")


def _cluster_neuron_counts_from_pt(path: str) -> Tuple[int, Dict[int, int]]:
    """Load a cluster assignment ``.pt``; return ``(k, {cluster_id -> num_neurons})``."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    c2i = d["cluster_to_indices"]
    k = int(d.get("k", len(c2i)))
    counts: Dict[int, int] = {}
    for cid, t in c2i.items():
        cid_i = int(cid)
        if hasattr(t, "numel"):
            counts[cid_i] = int(t.numel())
        else:
            counts[cid_i] = len(t)
    return k, counts


def _cluster_pt_paths_under_data_dir(
    data_dir: str,
    k_by_tower: Dict[str, int],
    *,
    subclass: int = 0,
) -> Tuple[str, str]:
    """``.../neuron-clustering/meta-llama/Llama-3.2-1B|Meta-Llama-3-8B/clusters/.../k{k}.pt``."""
    for key in ("1b", "8b"):
        if key not in k_by_tower:
            raise KeyError(f"k map must include {key!r}, got {list(k_by_tower.keys())!r}")

    base = os.path.abspath(data_dir)
    out: Dict[str, str] = {}
    for tw in ("1b", "8b"):
        rel = _TOWER_NEURON_CLUSTER_SUBPATH[tw]
        k = int(k_by_tower[tw])
        p = os.path.join(
            base,
            "neuron-clustering",
            *rel.split("/"),
            "clusters",
            f"subclass_{subclass}_clusters",
            f"k{k}.pt",
        )
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Missing {tw} cluster checkpoint under neuron-clustering/{rel}: {p!r}",
            )
        out[tw] = p
    return out["1b"], out["8b"]


def print_clustering_cluster_counts_table(
    data_dirs: Sequence[str],
    num_clusters: Sequence[Dict[str, int]],
    *,
    subclass: int = 0,
    row_labels: Optional[Sequence[str]] = None,
    row_label_header: str = "run",
    out: Optional[TextIO] = None,
    return_text: bool = False,
) -> Optional[str]:
    """Print a text table: rows = one per ``data_dir``, columns = cluster id, cells = ``1b/8b`` counts.

    For each ``data_dir``, ``num_clusters[i]`` is ``{'1b': k1, '8b': k2}``. Cluster ``.pt`` paths are
    under ``neuron-clustering/`` using :data:`utils.NEURON_CLUSTERING_STUDENT_SUBPATH` and
    :data:`utils.NEURON_CLUSTERING_TEACHER_SUBPATH`. Prefer **absolute** paths for each ``data_dir``.

    By default this only prints and returns ``None``. In Jupyter/IPython, returning the same string
    you print causes a second display with ``\\n`` shown as escapes; set ``return_text=True`` when you
    need the table string (e.g. to write a file).

    Returns:
        The table string if ``return_text`` is true; otherwise ``None``.
    """
    if len(num_clusters) != len(data_dirs):
        raise ValueError(
            f"num_clusters length {len(num_clusters)} != data_dirs length {len(data_dirs)}",
        )

    max_k = 0
    for m in num_clusters:
        for key in ("1b", "8b"):
            if key not in m:
                raise KeyError(f"each num_clusters entry must include {key!r}, got {list(m.keys())!r}")
        max_k = max(max_k, int(m["1b"]), int(m["8b"]))

    if row_labels is not None and len(row_labels) != len(data_dirs):
        raise ValueError(
            f"row_labels length {len(row_labels)} != data_dirs length {len(data_dirs)}",
        )

    parsed: List[Tuple[str, Dict[int, int], Dict[int, int]]] = []
    for i, raw in enumerate(data_dirs):
        root = os.path.abspath(raw)
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Not a directory: {root!r}")
        kmap = num_clusters[i]
        p1, p8 = _cluster_pt_paths_under_data_dir(
            root,
            kmap,
            subclass=subclass,
        )
        _, c1 = _cluster_neuron_counts_from_pt(p1)
        _, c2 = _cluster_neuron_counts_from_pt(p8)
        label = row_labels[i] if row_labels is not None else os.path.basename(root.rstrip(os.sep))
        parsed.append((label, c1, c2))

    if not parsed:
        msg = "data_dirs is empty"
        print(msg, file=out or sys.stdout)
        return msg if return_text else None

    rows: List[List[str]] = []
    for label, c1, c2 in parsed:
        row = [label]
        for cid in range(max_k):
            n1 = c1.get(cid, 0)
            n2 = c2.get(cid, 0)
            row.append(f"{n1}/{n2}")
        rows.append(row)

    headers = [row_label_header] + [f"cluster_{j}" for j in range(max_k)]
    col_widths = [max(len(headers[i]), max((len(r[i]) for r in rows), default=0)) for i in range(len(headers))]

    def fmt_row(cells: List[str]) -> str:
        return "  ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells))

    lines = [fmt_row(headers), "  ".join("-" * col_widths[i] for i in range(len(headers)))]
    for r in rows:
        lines.append(fmt_row(r[: len(headers)]))
    text = "\n".join(lines) + "\n"
    print(text, end="", file=out or sys.stdout)
    return text if return_text else None


def _subclass_features_pt_from_cluster_pt(cluster_pt_path: str) -> str:
    """``.../<model_dir>/clusters/subclass_*_clusters/k*.pt`` → ``.../<model_dir>/subclass_features.pt``."""
    model_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(cluster_pt_path))))
    return os.path.join(model_dir, "subclass_features.pt")


def _neuron_rows_for_cluster(
    cluster_neuron_indices: torch.Tensor,
    inv: Dict[int, int],
) -> List[int]:
    """Map global neuron indices to row indices into ``features_per_subclass[subclass]`` using ``inv``."""
    rows: List[int] = []
    for x in cluster_neuron_indices.reshape(-1).tolist():
        xi = int(x)
        if xi in inv:
            rows.append(inv[xi])
    return rows


def _neuron_id_to_feature_row_inv(indices_per_subclass: torch.Tensor) -> Dict[int, int]:
    """Build once per ``subclass_features`` load; reused for every cluster."""
    return {int(indices_per_subclass[i]): i for i in range(int(indices_per_subclass.numel()))}


def _cuda_device_for_stable_rank() -> torch.device:
    """CUDA device for per-cluster :func:`cka_loss.stable_rank_centered_gram` (required)."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "teacher stable-rank helpers require CUDA (torch.cuda.is_available() is False).",
        )
    return torch.device("cuda")


def teacher_cluster_stable_ranks_centered_gram(
    cluster_pt_path: str,
    subclass_features_pt_path: str,
    *,
    subclass: Optional[int] = None,
) -> Tuple[int, Dict[int, float]]:
    """Stable rank of the teacher's centered Gram matrix per cluster (tokens × neurons in cluster).

    Uses the same column centering as linear CKA (:func:`cka_loss.stable_rank_centered_gram`).
    ``cluster_pt_path`` should be the **8b (teacher)** cluster checkpoint. ``subclass_features_pt_path``
    must be that model's ``subclass_features.pt`` in the same ``neuron-clustering/.../<model_dir>/`` tree.

    Requires CUDA; raises :exc:`RuntimeError` if ``torch.cuda.is_available()`` is false.

    Returns:
        ``(k, {cluster_id -> stable_rank})`` with ``nan`` for empty clusters.
    """
    d = torch.load(cluster_pt_path, map_location="cpu", weights_only=False)
    sc = int(d["subclass"]) if subclass is None else int(subclass)
    k = int(d.get("k", len(d["cluster_to_indices"])))
    c2i = d["cluster_to_indices"]

    ckpt = torch.load(subclass_features_pt_path, map_location="cpu", weights_only=False)
    fp = ckpt["features_per_subclass"]
    ip = ckpt["indices_per_subclass"]
    if sc not in fp or sc not in ip:
        raise KeyError(f"subclass {sc} not in features checkpoint {subclass_features_pt_path!r}")
    feat = fp[sc].float()
    idx_feat = ip[sc]
    inv = _neuron_id_to_feature_row_inv(idx_feat)
    comp_dev = _cuda_device_for_stable_rank()

    out: Dict[int, float] = {}
    with torch.inference_mode():
        for j in range(k):
            cj = c2i.get(j)
            if cj is None or (hasattr(cj, "numel") and cj.numel() == 0):
                out[j] = float("nan")
                continue
            rows = _neuron_rows_for_cluster(cj, inv)
            if not rows:
                out[j] = float("nan")
                continue
            x_rows = feat[rows, :]  # [|C|, n_tokens]
            X = x_rows.T.contiguous().to(comp_dev, non_blocking=True)  # (n_tokens, |C|)
            sr = stable_rank_centered_gram(X)
            out[j] = float(sr.item())

    return k, out


def print_clustering_teacher_stable_rank_table(
    data_dirs: Sequence[str],
    num_clusters: Sequence[Dict[str, int]],
    *,
    subclass: int = 0,
    row_labels: Optional[Sequence[str]] = None,
    row_label_header: str = "run",
    out: Optional[TextIO] = None,
    float_fmt: str = "{:.4g}",
    return_text: bool = False,
) -> Optional[str]:
    """Print a text table: one row per ``data_dir``, columns = teacher (8b) cluster stable rank.

    Same path layout as :func:`print_clustering_cluster_counts_table` (8b under
    :data:`utils.NEURON_CLUSTERING_TEACHER_SUBPATH`). Loads ``subclass_features.pt`` next to the 8b
    cluster checkpoint.

    Stable rank is ``||G||_F^2 / ||G||_2^2`` for ``G = X_c X_c.T`` with column-centered activations
    ``X`` (tokens × neurons in cluster), matching :func:`cka_loss.stable_rank_centered_gram`.

    Requires CUDA (see :func:`teacher_cluster_stable_ranks_centered_gram`).

    By default this only prints and returns ``None``. In Jupyter/IPython, returning the same string
    you print causes a second display with ``\\n`` shown as escapes; set ``return_text=True`` when you
    need the table string (e.g. to write a file).

    Returns:
        The table string if ``return_text`` is true; otherwise ``None``.
    """
    if len(num_clusters) != len(data_dirs):
        raise ValueError(
            f"num_clusters length {len(num_clusters)} != data_dirs length {len(data_dirs)}",
        )
    for m in num_clusters:
        for key in ("1b", "8b"):
            if key not in m:
                raise KeyError(f"each num_clusters entry must include {key!r}, got {list(m.keys())!r}")

    max_k = max(int(m["8b"]) for m in num_clusters)

    if row_labels is not None and len(row_labels) != len(data_dirs):
        raise ValueError(
            f"row_labels length {len(row_labels)} != data_dirs length {len(data_dirs)}",
        )

    parsed: List[Tuple[str, int, Dict[int, float]]] = []
    for i, raw in enumerate(data_dirs):
        root = os.path.abspath(raw)
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Not a directory: {root!r}")
        kmap = num_clusters[i]
        _p1, p8 = _cluster_pt_paths_under_data_dir(
            root,
            kmap,
            subclass=subclass,
        )
        p_feat = _subclass_features_pt_from_cluster_pt(p8)
        if not os.path.isfile(p_feat):
            raise FileNotFoundError(
                f"Missing subclass_features.pt for teacher: {p_feat!r}",
            )
        kk, ranks = teacher_cluster_stable_ranks_centered_gram(
            p8, p_feat, subclass=subclass,
        )
        label = row_labels[i] if row_labels is not None else os.path.basename(root.rstrip(os.sep))
        parsed.append((label, kk, ranks))

    if not parsed:
        msg = "data_dirs is empty"
        print(msg, file=out or sys.stdout)
        return msg if return_text else None

    rows: List[List[str]] = []
    for label, _kk, ranks in parsed:
        row = [label]
        for cid in range(max_k):
            v = ranks.get(cid)
            if v is None or v != v:  # missing cluster or nan
                row.append("nan")
            else:
                row.append(float_fmt.format(v))
        rows.append(row)

    headers = [row_label_header] + [f"cluster_{j}" for j in range(max_k)]
    col_widths = [max(len(headers[i]), max((len(r[i]) for r in rows), default=0)) for i in range(len(headers))]

    def fmt_row(cells: List[str]) -> str:
        return "  ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells))

    lines = [fmt_row(headers), "  ".join("-" * col_widths[i] for i in range(len(headers)))]
    for r in rows:
        lines.append(fmt_row(r[: len(headers)]))
    text = "\n".join(lines) + "\n"
    print(text, end="", file=out or sys.stdout)
    return text if return_text else None


def _safe_stem(name: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", name.strip())
    return s or "curve"


def _single_json_in_dir(folder: str) -> str:
    """Return path to the sole ``*.json`` file in ``folder`` (sorted name for determinism)."""
    names = [f for f in os.listdir(folder) if f.lower().endswith(".json")]
    names.sort()
    if len(names) == 0:
        raise FileNotFoundError(f"No .json file found in {folder!r}")
    if len(names) > 1:
        raise ValueError(
            f"Expected exactly one .json file in {folder!r}, found {len(names)}: {names!r}",
        )
    return os.path.join(folder, names[0])


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
    data_dirs: Sequence[str],
    param_name: str,
    param_values: List[Any],
    metrics: List[str],
    *,
    history_filename: str = "training_history.json",
    x_key: str = "epoch",
    output_dir: Optional[str] = None,
    figure_width_in: float = 5.5,
    figure_height_in: float = 2.4,
    save_pdf: bool = True,
    save_png: bool = True,
    smooth_window: Optional[int] = None,
    smooth_metrics: Optional[Collection[str]] = None,
) -> str:
    """Plot several training histories (one run per folder) as multiple lines per subplot.

    Each entry in ``data_dirs`` is a directory containing ``history_filename`` (default
    ``training_history.json``). Order matches ``param_values`` (one legend line per directory).
    Pass **absolute** paths for ``data_dirs`` and ``output_dir`` when you can; if ``output_dir`` is
    omitted, figures are written under the process current working directory (not relative to ``src/``).
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

    json_paths: List[str] = []
    for d in data_dirs:
        root = os.path.abspath(d)
        if not os.path.isdir(root):
            raise NotADirectoryError(f"Not a directory: {root!r}")
        p = os.path.join(root, history_filename)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing {history_filename!r} in {root!r}")
        json_paths.append(p)
    if len(json_paths) != len(param_values):
        raise ValueError(
            f"Got {len(json_paths)} data dir(s) but param_values has length "
            f"{len(param_values)}; they must match (one value per directory, same order).",
        )
    loaded: List[Dict[str, List]] = []
    for p in json_paths:
        loaded.append(_load_training_history(p))

    out_dir = _output_dir_or_cwd(output_dir)
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
    data_dirs: Sequence[str],
    *,
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
    """Plot mean pairwise cossim curves for one or more runs.

    Each directory in ``data_dirs`` must contain **exactly one** ``*.json`` file (e.g. from
    ``circuit_discovery.neuron_cossim_topk``, schema ``neuron_mean_pairwise_cossim_v2``).
    Prefer **absolute** paths for ``data_dirs`` and ``output_dir``; if ``output_dir`` is omitted,
    output goes to the current working directory.
    """
    if not save_pdf and not save_png:
        raise ValueError("At least one of save_pdf, save_png must be True")
    paths: List[str] = []
    for d in data_dirs:
        root = os.path.abspath(d)
        if not os.path.isdir(root):
            raise NotADirectoryError(f"Not a directory: {root!r}")
        paths.append(_single_json_in_dir(root))
    if not paths:
        raise ValueError("data_dirs must be non-empty")

    out_dir = _output_dir_or_cwd(output_dir)
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
            label_base = rec.get("dataset_prefix") or os.path.basename(os.path.dirname(os.path.abspath(path)))
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
        plt.title(f"k-means loss vs k for {os.path.basename(base_dir)}, subclass {subclass_str}")
        plt.grid(True, alpha=0.3)

        out_path = os.path.join(base_dir, f"k_vs_loss_subclass_{subclass_str}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    # plot_training_histories_param_sweep(
    #     data_dirs=[
    #         "2d_add/1_class_0.01-lam0.1",
    #         "2d_add/1_class_0.01-lam0.5",
    #         "2d_add/1_class_0.01-lam1.0",
    #         "2d_add/1_class_0.01-lam5.0",
    #     ],
    #     param_name="lambda_cluster",
    #     param_values=[0.1, 0.5, 1.0, 5.0],
    #     metrics=["kl_loss", "cluster_loss", "accuracy"],
    #     smooth_window=10,
    # )

    # plot_training_histories_param_sweep(
    #   data_dirs=[
    #       "2d_add/1_class_0.001",
    #       "2d_add/1_class_0.01-lam0.1",
    #       "2d_add/1_class_0.05",
    #       "2d_add/1_class_0.10",
    #       "2d_add/1_class_0.25",
    #       "2d_add/1_class_0.50",
    #       "2d_add/1_class_1.0",
    #   ],
    #   param_name="frac_activated",
    #   param_values=[0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
    #   metrics=["kl_loss", "cluster_loss", "accuracy"],
    #   smooth_window=10,
    # )

    # plot_frac_activated_vs_mean_cossim(
    #   data_dirs=[
    #       "results/circuit-discovery/neuron_cossim_topk/frac0.001",
    #       "results/circuit-discovery/neuron_cossim_topk/frac0.01",
    #   ],
    #   out_name="frac_activated_vs_mean_cossim",
    #   figure_width_in=5.5,
    #   figure_height_in=2.8,
    #   num_points=800,
    #   anchor_zero=True,
    #   x_as_percent=True,
    #   save_pdf=True,
    #   save_png=True,
    # )

    # print_clustering_cluster_counts_table(
    #     data_dirs=[
    #         "results/distillation/2d_add/frac_activated/f0.01",
    #         "results/distillation/2d_add/frac_activated/f0.05",
    #     ],
    #     num_clusters=[
    #         {"1b": 5, "8b": 5},
    #         {"1b": 5, "8b": 7},
    #     ],
    #     subclass=0,
    #     row_label_header="frac_activated",
    #     out=sys.stdout,
    # )

    save_random_ablation_1b_8b_plot()
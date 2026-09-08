"""Render a built supergraph as a simple static node-link figure.

Replaces the old interactive web frontend: after ``python -m graph_loss`` builds
a supergraph, this draws every supernode (labeled with what it represents) and
every directed supernode-to-supernode edge annotated with its attribution
weight, saves the figure as a PNG/PDF, and opens it automatically.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import sys
import webbrowser
from pathlib import Path

from graph_loss.graph import Graph, SuperGraph
from graph_loss.static_figures import _describe_label

logger = logging.getLogger(__name__)


def slugify(value: str, *, fallback: str = "supergraph") -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower()).strip("-")
    return slug[:80] or fallback


def default_figure_name(graph: Graph, *, model_name: str | None = None) -> str:
    digest_src = f"{model_name or graph.cfg.model_name}:{graph.input_string}"
    digest = hashlib.sha1(digest_src.encode("utf-8")).hexdigest()[:10]
    prefix = slugify(model_name or graph.cfg.model_name or "supergraph")
    return f"{prefix}-{digest}"


def _supernode_labels(supergraph: SuperGraph) -> list[str]:
    labels = []
    for idx, members in enumerate(supergraph.supernodes):
        if supergraph.supernode_labels and idx < len(supergraph.supernode_labels):
            parts = [str(p) for p in supergraph.supernode_labels[idx] if p]
            if parts:
                labels.append(", ".join(parts))
                continue
        labels.append(f"supernode {idx} ({len(members)} neurons)")
    return labels


def _tier_positions(labels: list[str]) -> dict[int, tuple[float, float]]:
    """Arg-token supernodes on the bottom, sum/dla sinks on top, rest between."""
    low = [l.lower() for l in labels]
    bottom = [i for i, l in enumerate(low) if l.startswith("arg")]
    top = [i for i, l in enumerate(low) if "dla" in l or "sum" in l]
    mid = [i for i in range(len(labels)) if i not in bottom and i not in top]
    pos: dict[int, tuple[float, float]] = {}
    for y, members in ((0.14, bottom), (0.5, mid), (0.86, top)):
        m = len(members)
        for k, i in enumerate(members):
            pos[i] = (0.08 + 0.84 * (k + 0.5) / m if m else 0.5, y)
    return pos


def render_supergraph(
    graph: Graph,
    supergraph: SuperGraph,
    *,
    output_dir: str | os.PathLike[str],
    name: str | None = None,
    model_name: str | None = None,
    weight_threshold_frac: float = 0.02,
) -> Path:
    """Draw the supernode graph (labeled nodes, directed weighted edges).

    ``weight_threshold_frac`` hides edges below that fraction of the largest
    absolute edge weight so tiny attributions don't clutter the picture.
    Returns the path of the saved PNG (a PDF is written alongside it).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    labels = _supernode_labels(supergraph)
    n = len(labels)
    pos = _tier_positions(labels)
    adj = supergraph.supernode_adjacency_matrix.detach().float().cpu()

    fig, ax = plt.subplots(figsize=(max(8.0, 2.1 * n), 8.0))

    max_abs = float(adj.abs().max().item()) if adj.numel() else 0.0
    threshold = weight_threshold_frac * max_abs

    # ── Directed edges, annotated with their attribution weight ─────────────
    edges = []
    for tgt in range(n):
        for src in range(n):
            if tgt == src:
                continue
            w = float(adj[tgt, src].item())
            if abs(w) <= threshold:
                continue
            edges.append((abs(w), w, src, tgt))
    for mag, w, src, tgt in sorted(edges):
        rel = mag / max_abs if max_abs > 0 else 0.0
        color = "#2b6cb0" if w >= 0 else "#c62828"
        rad = 0.12
        ax.add_patch(FancyArrowPatch(
            pos[src], pos[tgt], arrowstyle="-|>", mutation_scale=13,
            lw=0.6 + 3.2 * rel, alpha=min(1.0, 0.35 + 0.65 * rel), color=color,
            connectionstyle=f"arc3,rad={rad}", shrinkA=26, shrinkB=26, zorder=1,
        ))
        # Weight label placed ~40% of the way along the curved edge (a midpoint
        # label often lands on top of a node that sits between source and target).
        (x0, y0), (x1, y1) = pos[src], pos[tgt]
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy) or 1.0
        px, py = -dy / dist, dx / dist
        cx = (x0 + x1) / 2 + px * rad * dist  # arc3 quadratic-Bezier control point
        cy = (y0 + y1) / 2 + py * rad * dist
        t = 0.38
        mx = (1 - t) ** 2 * x0 + 2 * t * (1 - t) * cx + t**2 * x1
        my = (1 - t) ** 2 * y0 + 2 * t * (1 - t) * cy + t**2 * y1
        ax.text(mx, my, f"{w:.2f}", ha="center", va="center", fontsize=7,
                color=color, zorder=2,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    # ── Nodes: label plus a plain-English description of what it represents ──
    for i, (x, y) in pos.items():
        ax.scatter([x], [y], s=2600, color="#cfe3ff", edgecolors="#2b6cb0",
                   linewidths=1.4, zorder=3)
        ax.text(x, y, labels[i], ha="center", va="center", fontsize=8,
                weight="bold", zorder=4, wrap=True)
        description = _describe_label(labels[i])
        if description != labels[i]:
            ax.text(x, y - 0.055, description, ha="center", va="top", fontsize=7,
                    color="#444444", style="italic", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="#cccccc", alpha=0.85))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    title = f"Attribution supergraph for “{graph.input_string}”"
    if model_name:
        title += f"  ({model_name})"
    ax.set_title(title, fontsize=12)
    fig.text(0.5, 0.01,
             "edge label = attribution weight (blue positive, red negative); "
             f"edges below {weight_threshold_frac:.0%} of the max are hidden",
             ha="center", fontsize=8, color="#666666")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = slugify(name) if name else default_figure_name(graph, model_name=model_name)
    png_path = out_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    return png_path


def show_figure(path: str | os.PathLike[str]) -> None:
    """Display the saved figure: inline in a notebook, else in the OS viewer."""
    path = Path(path)
    try:  # Jupyter/Colab: render inline instead of trying to open a window.
        from IPython import get_ipython
        from IPython.display import Image, display

        if get_ipython() is not None:
            display(Image(filename=str(path)))
            return
    except ImportError:
        pass
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # noqa: S606
        else:
            webbrowser.open(path.resolve().as_uri())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not auto-open %s (%s); open it manually.", path, exc)

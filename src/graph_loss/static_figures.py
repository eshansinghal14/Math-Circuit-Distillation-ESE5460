"""Static teacher-vs-student circuit figures for the write-up.

Builds teacher and student attribution supergraphs for a single prompt, aligns
their supernodes by label, and renders publication-ready static figures (PNG +
PDF) that show how the student circuit converges toward the teacher's. Nothing
here is interactive; everything is deterministic given the same checkpoints and
build args, so the figures are reproducible for the paper.

Quick start (run from ``src/``)::

    python -m graph_loss.static_figures \
        --teacher meta-llama/Llama-3.1-8B \
        --student results/graph_kd \
        --prompt "36+59=" \
        --out-dir results/figures \
        --history results/graph_kd/training_history.json \
        --teacher-heatmap-dir "/.../FINAL_RESULTS/36+59_heatmaps/llama_8b" \
        --student-heatmap-dir "/.../FINAL_RESULTS/36+59_heatmaps/llama_1b"

Required: ``--teacher``, ``--student``, ``--prompt``. Add ``--student-base
<id>`` for a third "before training" column. ``--history`` enables Fig D.
The shared ``add_graph_build_args`` flags (``--prop_neurons_per_layer``,
``--top_k_logits``, ``--temperature``, ``--nodes-per-label``,
``--graph-node-labels``, ...) control attribution exactly as in the trainers.

Figure guide (what each panel demonstrates in the write-up)
-----------------------------------------------------------
* Fig A  ``figA_adjacency_comparison`` — side-by-side row-normalized supernode
  adjacency heatmaps (teacher, student, optional base) plus a ``|teacher -
  student|`` panel annotated with the mean per-row JSD. This is the most direct
  picture of *what the graph loss matches*: each row is a target supernode's
  incoming routing distribution. A dark diff panel + low JSD ⇒ structural
  convergence.
* Fig B  ``figB_circuit_diagram`` — node-link diagram per model with arg-token
  supernodes on the bottom, intermediates in the middle, and the ``dla`` sink on
  top; edge width/opacity ∝ routing weight. Shows the student reproducing the
  teacher's arg→dla computation path.
* Fig C  ``figC_routing_distribution`` — grouped bar chart of one target
  supernode's incoming normalized edges (default: the ``dla`` row, override with
  ``--target-supernode``). This is the literal per-node KL operand: it shows the
  student putting mass on the same sources as the teacher.
* Fig D  ``figD_training_curves`` — KL loss, graph loss, and accuracy vs step
  read from a trainer's ``training_history.json`` (with the teacher accuracy
  baseline drawn in). Demonstrates convergence over training.
* Fig E  ``figE_heatmap_montage`` — per-supernode activation heatmaps, teacher
  row above student row, paired by supernode index. Two sources: pass
  ``--teacher-heatmap-dir``/``--student-heatmap-dir`` to reuse saved
  ``supernode_*.pdf`` files (rendered with PyMuPDF), or omit them and pass
  ``--dataset`` to regenerate aggregated activation grids from the MLP cache.

Notes
-----
* Figs A-D use the default arg-token + DLA supernodes and need no MLP cache.
  ANOVA labels (``--graph-node-labels``) and ``--dataset`` are only needed for
  the Fig E *regenerate* branch.
* The teacher DLA reference logits used to anchor the student's ``dla``
  supernode are the teacher's logits at the final prompt position (causal, so no
  answer string is required) — matching ``backward_batch_graph_loss``.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
from dataclasses import dataclass

import torch

from graph_loss.create_graph import GraphPipelineResult, create_graph
from graph_loss.graph import SuperGraph
from graph_loss.hf_adapter import HFLlamaGraphAdapter
from graph_loss.utils import add_graph_build_args
from utils import DIR_ROOT, load_model

logger = logging.getLogger(__name__)

EPS = 1e-8


# ---------------------------------------------------------------------------
# Build configuration
# ---------------------------------------------------------------------------


@dataclass
class BuildConfig:
    """Attribution-graph build parameters shared across all models."""

    top_k_logits: float = 0.95
    temperature: float = 2.0
    prop_neurons_per_layer: float = 0.1
    attribution_batch_size: int = 512
    nodes_per_label: int = 10
    anova_range_radius: int = 0
    graph_node_labels: list[str] | None = None
    dataset: str | None = None
    verbose: bool = False


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _model_device(adapter: HFLlamaGraphAdapter) -> torch.device:
    return next(adapter.model.parameters()).device


def last_token_logits(adapter: HFLlamaGraphAdapter, prompt: str) -> torch.Tensor:
    """Teacher reference logits for student DLA selection.

    The logits at the final prompt position predict the first answer token and
    depend only on the prompt (causal), so no answer string is needed.
    """
    input_ids = adapter.ensure_tokenized(prompt)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    with torch.no_grad():
        logits = adapter.model(input_ids.to(_model_device(adapter))).logits
    return logits.squeeze(0)[-1].detach()


def build_supergraph(
    adapter: HFLlamaGraphAdapter,
    prompt: str,
    cfg: BuildConfig,
    *,
    model_name: str | None = None,
    dla_model_logits: torch.Tensor | None = None,
    heatmap_output_dir: str | None = None,
) -> GraphPipelineResult:
    """Build one attribution supergraph (detached) for ``prompt``.

    Mirrors the non-CoT teacher/student build in ``backward_batch_graph_loss``:
    attribution runs under ``enable_grad`` (autograd.grad needs it) but the
    result is detached since figures never backprop.

    ``model_name`` is required when ANOVA labels are requested (it keys the MLP
    input cache that ANOVA labeling needs; without it the cache is never built
    and no ANOVA supernodes are produced).
    """
    with torch.enable_grad():
        result = create_graph(
            adapter,
            prompt,
            top_k_logits=cfg.top_k_logits,
            temperature=cfg.temperature,
            prop_neurons_per_layer=cfg.prop_neurons_per_layer,
            batch_size=cfg.attribution_batch_size,
            nodes_per_label=cfg.nodes_per_label,
            anova_range_radius=cfg.anova_range_radius,
            node_labels=cfg.graph_node_labels,
            dataset=cfg.dataset,
            model_name=model_name,
            dla_model_logits=dla_model_logits,
            detach_result=True,
            no_grad_supergraph=True,
            build_create_graph=False,
            supernode_heatmap_output_dir=heatmap_output_dir,
            verbose=cfg.verbose,
            logger=logger,
        )
    return result


# ---------------------------------------------------------------------------
# Alignment (mirror of the loss-side label matching)
# ---------------------------------------------------------------------------


def supernode_label(supergraph: SuperGraph, idx: int) -> str:
    labels = supergraph.supernode_labels
    if labels and idx < len(labels) and labels[idx]:
        return str(labels[idx][0])
    return f"supernode_{idx}"


def _label_to_index(supergraph: SuperGraph) -> dict[str, int]:
    return {
        supernode_label(supergraph, i): i
        for i in range(len(supergraph.supernodes))
    }


def _normalize_rows(adj: torch.Tensor) -> torch.Tensor:
    """L1-normalize |rows| -> per-target routing distribution (same as the loss)."""
    a = adj.abs()
    return a / a.sum(dim=1, keepdim=True).clamp(min=EPS)


def jsd_rows(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Per-row Jensen-Shannon divergence between two row-stochastic matrices."""
    m = 0.5 * (p + q)

    def _kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a * ((a + EPS).log() - (b + EPS).log())).sum(dim=1)

    return 0.5 * (_kl(p, m) + _kl(q, m))


@dataclass
class AlignedGraphs:
    labels: list[str]
    raw: dict[str, torch.Tensor]   # name -> [k, k] common-label-ordered adjacency
    norm: dict[str, torch.Tensor]  # name -> [k, k] row-normalized adjacency


def align_supergraphs(
    named: dict[str, SuperGraph], *, exclude_self_loops: bool = False
) -> AlignedGraphs:
    """Reorder every supergraph to the shared (intersection) label set.

    The first entry's label ordering is used as the canonical column/row order
    (pass the teacher first). Labels present in only some models are dropped
    with a warning, so figures always compare the same supernodes.

    If ``exclude_self_loops`` is set, the adjacency diagonal (a supernode routing
    into itself) is zeroed *before* row-normalization, so intra-supernode mass
    doesn't dominate the figures. This affects the figures only; the training
    loss still uses the full matrix.
    """
    label_maps = {name: _label_to_index(sg) for name, sg in named.items()}
    label_sets = [set(m) for m in label_maps.values()]
    common = set.intersection(*label_sets) if label_sets else set()

    order_name = next(iter(named))
    ordered = [
        lbl
        for lbl, _ in sorted(label_maps[order_name].items(), key=lambda kv: kv[1])
        if lbl in common
    ]

    dropped = {
        name: sorted(set(m) - common)
        for name, m in label_maps.items()
        if set(m) - common
    }
    if dropped:
        logger.warning("Dropping non-shared supernodes during alignment: %s", dropped)

    raw: dict[str, torch.Tensor] = {}
    norm: dict[str, torch.Tensor] = {}
    for name, sg in named.items():
        idx = [label_maps[name][lbl] for lbl in ordered]
        adj = sg.supernode_adjacency_matrix.detach().float().cpu()
        sub = (adj[idx][:, idx] if idx else adj[:0, :0]).clone()
        if exclude_self_loops and sub.numel():
            sub.fill_diagonal_(0.0)
        raw[name] = sub
        norm[name] = _normalize_rows(sub)
    return AlignedGraphs(labels=ordered, raw=raw, norm=norm)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    logger.info("Saved figure: %s", os.path.join(out_dir, f"{name}.png"))


def plot_adjacency_comparison(
    aligned: AlignedGraphs,
    *,
    teacher: str,
    student: str,
    base: str | None,
    out_dir: str,
) -> None:
    """Fig A: row-normalized supernode adjacency heatmaps + |teacher - student| panel."""
    plt = _import_pyplot()
    labels = aligned.labels
    if not labels:
        logger.warning("Fig A skipped: no shared supernodes to compare.")
        return

    names = [teacher, student] + ([base] if base else [])
    panels = names + ["|teacher - student|"]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6))
    if n == 1:
        axes = [axes]

    diff = (aligned.norm[teacher] - aligned.norm[student]).abs()
    per_row_jsd = jsd_rows(aligned.norm[teacher], aligned.norm[student])
    mean_jsd = float(per_row_jsd.mean().item())

    for ax, panel in zip(axes, panels):
        mat = diff if panel.startswith("|") else aligned.norm[panel]
        im = ax.imshow(mat.numpy(), cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("source supernode", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("target supernode", fontsize=8)
        title = panel if not panel.startswith("|") else f"|diff|  (mean row JSD={mean_jsd:.3f})"
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Row-normalized supernode routing (teacher vs student)", fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir, "figA_adjacency_comparison")
    plt.close(fig)


def _circuit_positions(labels: list[str]) -> dict[int, tuple[float, float]]:
    dla = [i for i, l in enumerate(labels) if "dla" in l.lower()]
    args = [i for i, l in enumerate(labels) if l.lower().startswith("arg")]
    others = [i for i in range(len(labels)) if i not in dla and i not in args]
    pos: dict[int, tuple[float, float]] = {}
    for row, members in ((0.12, args), (0.5, others), (0.9, dla)):
        for k, i in enumerate(members):
            x = (k + 1) / (len(members) + 1)
            pos[i] = (x, row)
    return pos


def _draw_circuit(ax, labels, norm_adj, *, title, weight_threshold=0.04):
    from matplotlib.patches import FancyArrowPatch

    pos = _circuit_positions(labels)
    k = len(labels)
    for tgt in range(k):
        for src in range(k):
            if tgt == src:
                continue
            w = float(norm_adj[tgt, src].item())
            if w < weight_threshold:
                continue
            arrow = FancyArrowPatch(
                pos[src], pos[tgt],
                arrowstyle="-|>", mutation_scale=12,
                lw=0.5 + 6.0 * w, alpha=min(1.0, 0.25 + w),
                color="#444", connectionstyle="arc3,rad=0.08",
                shrinkA=14, shrinkB=14, zorder=1,
            )
            ax.add_patch(arrow)
    for i, (x, y) in pos.items():
        ax.scatter([x], [y], s=900, color="#cfe3ff", edgecolors="#2b6cb0", zorder=2)
        ax.text(x, y, labels[i], ha="center", va="center", fontsize=7, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)


def plot_circuit_diagram(
    aligned: AlignedGraphs,
    *,
    teacher: str,
    student: str,
    base: str | None,
    out_dir: str,
) -> None:
    """Fig B: node-link circuit diagram per model (arg nodes -> dla sink)."""
    plt = _import_pyplot()
    labels = aligned.labels
    if not labels:
        logger.warning("Fig B skipped: no shared supernodes to draw.")
        return

    names = [teacher, student] + ([base] if base else [])
    fig, axes = plt.subplots(1, len(names), figsize=(5.0 * len(names), 5.0))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        _draw_circuit(ax, labels, aligned.norm[name], title=name)
    fig.suptitle("Supernode routing graph (edge width \u221d routing weight)", fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir, "figB_circuit_diagram")
    plt.close(fig)


def _resolve_target(labels: list[str], target: str | None) -> int | None:
    if not labels:
        return None
    if target:
        for i, l in enumerate(labels):
            if l == target or target.lower() in l.lower():
                return i
    for i, l in enumerate(labels):
        if "dla" in l.lower():
            return i
    return len(labels) - 1


def plot_routing_distributions(
    aligned: AlignedGraphs,
    *,
    teacher: str,
    student: str,
    base: str | None,
    out_dir: str,
    target: str | None = None,
) -> None:
    """Fig C: incoming routing distribution of one target supernode (grouped bars)."""
    plt = _import_pyplot()
    labels = aligned.labels
    tgt = _resolve_target(labels, target)
    if tgt is None:
        logger.warning("Fig C skipped: no shared supernodes.")
        return

    import numpy as np

    names = [teacher, student] + ([base] if base else [])
    rows = {name: aligned.norm[name][tgt].numpy() for name in names}
    x = np.arange(len(labels))
    width = 0.8 / len(names)

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(labels)), 4))
    for j, name in enumerate(names):
        ax.bar(x + (j - (len(names) - 1) / 2) * width, rows[name], width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("normalized incoming routing weight", fontsize=8)
    ax.set_title(f"Incoming routing into '{labels[tgt]}' supernode", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "figC_routing_distribution")
    plt.close(fig)


def plot_training_curves(history_path: str, out_dir: str) -> None:
    """Fig D: graph-loss / KL-loss / accuracy convergence from a saved history.json."""
    if not history_path or not os.path.isfile(history_path):
        logger.info("Fig D skipped: no history file at %s", history_path)
        return
    plt = _import_pyplot()
    with open(history_path, encoding="utf-8") as f:
        history = json.load(f)

    steps = history.get("train_step", [])
    graph_series = history.get("step_graph_loss", [])
    kl_series = history.get("step_kl_loss", [])
    acc_series = history.get("accuracy", [])
    acc_steps = history.get("accuracy_step", list(range(len(acc_series))))
    extra_acc = sorted(
        k for k in history if k.startswith("accuracy_") and k != "accuracy_step"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if steps and (kl_series or graph_series):
        if kl_series:
            axes[0].plot(steps[: len(kl_series)], kl_series, marker="o", markersize=2, label="KL loss")
        if graph_series:
            axes[0].plot(steps[: len(graph_series)], graph_series, marker="o", markersize=2, label="graph loss")
        axes[0].set_xlabel("step")
        axes[0].set_title("Training losses")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

    if acc_series:
        axes[1].plot(acc_steps[: len(acc_series)], acc_series, marker="o", markersize=2, label="main")
    for key in extra_acc:
        series = history.get(key, [])
        if series:
            axes[1].plot(acc_steps[: len(series)], series, marker="o", markersize=2, label=key[len("accuracy_"):])
    teacher_baseline = history.get("teacher_baseline")
    if isinstance(teacher_baseline, (int, float)):
        axes[1].axhline(teacher_baseline, color="gray", linestyle="--", linewidth=1, label="teacher")
    axes[1].set_xlabel("step")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Accuracy")
    axes[1].legend(fontsize=7, loc="lower right")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, out_dir, "figD_training_curves")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig E: per-supernode activation-heatmap montage
# ---------------------------------------------------------------------------


def _sorted_supernode_pdfs(directory: str) -> list[str]:
    files = glob.glob(os.path.join(directory, "supernode_*.pdf"))

    def _idx(path: str) -> int:
        m = re.search(r"supernode_(\d+)\.pdf$", os.path.basename(path))
        return int(m.group(1)) if m else 1_000_000

    return sorted(files, key=_idx)


def _render_pdf_pages(pdf_path: str, *, all_pages: bool, zoom: float = 2.0):
    import fitz  # PyMuPDF
    import numpy as np

    doc = fitz.open(pdf_path)
    pages = range(doc.page_count) if all_pages else [0]
    images = []
    for p in pages:
        pix = doc.load_page(p).get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        images.append(img[:, :, :3])
    doc.close()
    return images


def _heatmap_montage_from_dirs(
    teacher_dir: str,
    student_dir: str,
    out_dir: str,
    *,
    all_pages: bool,
) -> bool:
    try:
        import fitz  # noqa: F401
    except ImportError:
        logger.warning("Fig E reuse skipped: PyMuPDF (fitz) not installed. `pip install pymupdf`.")
        return False

    plt = _import_pyplot()
    t_pdfs = _sorted_supernode_pdfs(teacher_dir)
    s_pdfs = _sorted_supernode_pdfs(student_dir)
    if not t_pdfs and not s_pdfs:
        logger.warning("Fig E reuse skipped: no supernode_*.pdf in %s or %s", teacher_dir, student_dir)
        return False

    n = max(len(t_pdfs), len(s_pdfs))
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.4), squeeze=False)
    for row, (label, pdfs) in enumerate((("teacher", t_pdfs), ("student", s_pdfs))):
        for col in range(n):
            ax = axes[row][col]
            ax.set_axis_off()
            if col < len(pdfs):
                try:
                    img = _render_pdf_pages(pdfs[col], all_pages=all_pages)[0]
                    ax.imshow(img)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to render %s: %s", pdfs[col], exc)
                if row == 0:
                    ax.set_title(f"supernode {col}", fontsize=9)
            if col == 0:
                ax.text(-0.05, 0.5, label, transform=ax.transAxes,
                        rotation=90, va="center", ha="right", fontsize=11)
    fig.suptitle("Per-supernode activation heatmaps (teacher vs student)", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, "figE_heatmap_montage")
    plt.close(fig)
    return True


def _heatmap_montage_regenerate(
    results: dict[str, GraphPipelineResult],
    adapters: dict[str, HFLlamaGraphAdapter],
    *,
    teacher: str,
    student: str,
    dataset: str | None,
    out_dir: str,
) -> bool:
    if not dataset:
        logger.info("Fig E regenerate skipped: --dataset required to build the MLP cache.")
        return False
    from graph_loss.neuron_activation_heatmap import build_neuron_activation_write_result

    plt = _import_pyplot()
    names = [teacher, student]

    def _supernode_grids(name: str):
        result = results[name]
        graph = result.graph
        sg = result.supergraph
        locations = graph.neuron_locations.detach().cpu()
        grids = []
        for members in sg.supernodes:
            if not members:
                grids.append(None)
                continue
            member_locs = locations[[int(m) for m in members]]
            try:
                awr = build_neuron_activation_write_result(
                    adapters[name].model, member_locs, dataset=dataset
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Heatmap regen failed for %s supernode: %s", name, exc)
                grids.append(None)
                continue
            acts = awr.activations.float()
            grid = torch.nanmean(acts, dim=0)  # mean over member neurons
            while grid.dim() > 2:
                grid = torch.nanmean(grid, dim=-1)
            grids.append((grid, awr.arg_values))
        return sg, grids

    t_sg, t_grids = _supernode_grids(teacher)
    _, s_grids = _supernode_grids(student)
    n = max(len(t_grids), len(s_grids))
    if n == 0:
        return False

    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.4), squeeze=False)
    for row, (label, grids) in enumerate((("teacher", t_grids), ("student", s_grids))):
        for col in range(n):
            ax = axes[row][col]
            cell = grids[col] if col < len(grids) else None
            if cell is None:
                ax.set_axis_off()
            else:
                grid, arg_values = cell
                arr = grid.numpy()
                if arr.ndim == 2:
                    ax.imshow(arr.T, origin="lower", aspect="auto", cmap="viridis")
                    ax.set_xlabel("arg1", fontsize=7)
                    ax.set_ylabel("arg2", fontsize=7)
                else:
                    ax.plot(arg_values[0] if arg_values else range(arr.shape[0]), arr)
                ax.tick_params(labelsize=6)
            if row == 0:
                title = supernode_label(t_sg, col) if col < len(t_sg.supernodes) else f"supernode {col}"
                ax.set_title(title, fontsize=8)
            if col == 0:
                ax.text(-0.18, 0.5, label, transform=ax.transAxes,
                        rotation=90, va="center", ha="right", fontsize=11)
    fig.suptitle("Per-supernode activation heatmaps (teacher vs student, regenerated)", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, "figE_heatmap_montage")
    plt.close(fig)
    return True


def plot_heatmap_montage(
    *,
    teacher_heatmap_dir: str | None,
    student_heatmap_dir: str | None,
    results: dict[str, GraphPipelineResult],
    adapters: dict[str, HFLlamaGraphAdapter],
    teacher: str,
    student: str,
    dataset: str | None,
    out_dir: str,
    all_pages: bool,
) -> None:
    """Fig E: reuse saved heatmap PDFs if given, otherwise regenerate from the MLP cache."""
    if teacher_heatmap_dir and student_heatmap_dir:
        if _heatmap_montage_from_dirs(
            teacher_heatmap_dir, student_heatmap_dir, out_dir, all_pages=all_pages
        ):
            return
        logger.info("Fig E: falling back to regeneration.")
    _heatmap_montage_regenerate(
        results, adapters, teacher=teacher, student=student, dataset=dataset, out_dir=out_dir
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render static teacher-vs-student circuit figures for the write-up."
    )
    parser.add_argument("--teacher", required=True, help="Teacher model id or local path.")
    parser.add_argument("--student", required=True, help="Trained student model id or checkpoint path.")
    parser.add_argument("--student-base", "--student_base", dest="student_base", default=None,
                        help="Optional untrained base student (model id) for a before/after column.")
    parser.add_argument("--prompt", required=True, help="Prompt to attribute, e.g. '36+59='.")
    parser.add_argument("--dataset", default=None,
                        help="Dataset name (datasets/<name>) used only for ANOVA labels / Fig E regeneration.")
    parser.add_argument("--out-dir", "--out_dir", dest="out_dir", default="results/figures",
                        help="Directory to write figures into.")
    parser.add_argument("--target-supernode", "--target_supernode", dest="target_supernode", default=None,
                        help="Target supernode label for Fig C (default: the 'dla' node).")
    parser.add_argument("--history", default=None,
                        help="Path to a training_history.json for Fig D (training curves).")
    parser.add_argument("--teacher-heatmap-dir", "--teacher_heatmap_dir", dest="teacher_heatmap_dir", default=None,
                        help="Folder of teacher supernode_*.pdf heatmaps for Fig E.")
    parser.add_argument("--student-heatmap-dir", "--student_heatmap_dir", dest="student_heatmap_dir", default=None,
                        help="Folder of student supernode_*.pdf heatmaps for Fig E.")
    parser.add_argument("--heatmap-all-pages", "--heatmap_all_pages", dest="heatmap_all_pages",
                        action="store_true", help="Render all member pages per supernode PDF (default: first page).")
    parser.add_argument("--exclude-self-loops", "--exclude_self_loops", "--no-self-loops",
                        dest="exclude_self_loops", action="store_true",
                        help="Zero the adjacency diagonal (supernode->itself) before normalizing the "
                             "figures, so intra-supernode mass doesn't dominate Figs A/B/C.")
    add_graph_build_args(parser)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(DIR_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cfg = BuildConfig(
        top_k_logits=args.top_k_logits,
        temperature=args.temperature,
        prop_neurons_per_layer=args.prop_neurons_per_layer,
        attribution_batch_size=args.attribution_batch_size,
        nodes_per_label=args.nodes_per_label,
        anova_range_radius=args.anova_range_radius,
        graph_node_labels=args.graph_node_labels,
        dataset=args.dataset,
        verbose=args.verbose,
    )

    teacher_name = "teacher"
    student_name = "student"
    base_name = "base student" if args.student_base else None

    # ── Teacher ───────────────────────────────────────────────────────────
    logger.info("Loading teacher: %s", args.teacher)
    teacher_model, teacher_tok = load_model(args.teacher)
    teacher_model.eval()
    teacher_adapter = HFLlamaGraphAdapter(teacher_model, teacher_tok, _model_device_of(teacher_model))
    teacher_dla = last_token_logits(teacher_adapter, args.prompt)
    logger.info("Building teacher supergraph")
    teacher_result = build_supergraph(teacher_adapter, args.prompt, cfg, model_name=args.teacher)

    # ── Student (trained) ─────────────────────────────────────────────────
    logger.info("Loading student: %s", args.student)
    student_model, student_tok = load_model(args.student)
    student_model.eval()
    student_adapter = HFLlamaGraphAdapter(student_model, student_tok, _model_device_of(student_model))
    logger.info("Building student supergraph")
    student_result = build_supergraph(
        student_adapter, args.prompt, cfg,
        model_name=args.student,
        dla_model_logits=teacher_dla.to(_model_device_of(student_model)),
    )

    results = {teacher_name: teacher_result, student_name: student_result}
    adapters = {teacher_name: teacher_adapter, student_name: student_adapter}
    named_supergraphs = {
        teacher_name: teacher_result.supergraph,
        student_name: student_result.supergraph,
    }

    # ── Optional untrained base student ────────────────────────────────────
    if args.student_base:
        logger.info("Loading base student: %s", args.student_base)
        base_model, base_tok = load_model(args.student_base)
        base_model.eval()
        base_adapter = HFLlamaGraphAdapter(base_model, base_tok, _model_device_of(base_model))
        logger.info("Building base student supergraph")
        base_result = build_supergraph(
            base_adapter, args.prompt, cfg,
            model_name=args.student_base,
            dla_model_logits=teacher_dla.to(_model_device_of(base_model)),
        )
        results[base_name] = base_result
        adapters[base_name] = base_adapter
        named_supergraphs[base_name] = base_result.supergraph

    aligned = align_supergraphs(named_supergraphs, exclude_self_loops=args.exclude_self_loops)
    logger.info("Aligned %d shared supernodes: %s", len(aligned.labels), aligned.labels)

    # ── Figures (each guarded so one failure doesn't abort the rest) ───────
    figures = [
        ("Fig A adjacency", lambda: plot_adjacency_comparison(
            aligned, teacher=teacher_name, student=student_name, base=base_name, out_dir=out_dir)),
        ("Fig B circuit", lambda: plot_circuit_diagram(
            aligned, teacher=teacher_name, student=student_name, base=base_name, out_dir=out_dir)),
        ("Fig C routing", lambda: plot_routing_distributions(
            aligned, teacher=teacher_name, student=student_name, base=base_name,
            out_dir=out_dir, target=args.target_supernode)),
        ("Fig D curves", lambda: plot_training_curves(args.history, out_dir)),
        ("Fig E heatmaps", lambda: plot_heatmap_montage(
            teacher_heatmap_dir=args.teacher_heatmap_dir,
            student_heatmap_dir=args.student_heatmap_dir,
            results=results, adapters=adapters,
            teacher=teacher_name, student=student_name,
            dataset=args.dataset, out_dir=out_dir, all_pages=args.heatmap_all_pages)),
    ]
    for label, fn in figures:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s failed: %s", label, exc)

    logger.info("Done. Figures written to %s", out_dir)


def _model_device_of(model) -> torch.device:
    return next(model.parameters()).device


if __name__ == "__main__":
    main()

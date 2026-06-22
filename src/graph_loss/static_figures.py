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
from typing import TYPE_CHECKING

import torch

# Heavy model-stack imports (create_graph, hf_adapter, utils) are done lazily
# inside the functions that build graphs, so the rendering path (figures from a
# --plot-cache) can be imported and run with only numpy/matplotlib/torch.
if TYPE_CHECKING:
    from graph_loss.create_graph import GraphPipelineResult
    from graph_loss.graph import SuperGraph
    from graph_loss.hf_adapter import HFLlamaGraphAdapter

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


def model_prediction(
    adapter: HFLlamaGraphAdapter, prompt: str, *, dataset: str | None = None,
    max_new_tokens: int = 8,
) -> str:
    """What the model actually outputs for ``prompt`` under greedy decoding.

    Mirrors ``eval_model``: same raw prompt (BOS + tokens, no chat template for
    the local math datasets), greedy ``generate``, then ``parse_response`` to the
    integer answer when a ``dataset`` is given. Falls back to the decoded
    continuation so a broken model's junk output is shown verbatim.
    """
    tok = adapter.tokenizer
    input_ids = adapter.ensure_tokenized(prompt)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(_model_device(adapter))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with torch.no_grad():
        out = adapter.model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
    cont = tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
    if dataset:
        try:
            from utils import parse_response

            parsed = parse_response(prompt + cont, dataset)
            if parsed is not None:
                return str(parsed)
        except Exception:  # noqa: BLE001
            pass
    return cont.strip()


def true_answer(prompt: str) -> str | None:
    """The correct answer to an arithmetic prompt like '36+59=' (or None)."""
    lhs = prompt.split("=")[0]
    for op, fn in (("+", lambda xs: sum(xs)),
                   ("*", lambda xs: __import__("math").prod(xs)),
                   ("-", lambda xs: xs[0] - sum(xs[1:]))):
        parts = lhs.split(op)
        if len(parts) >= 2 and all(p.strip().lstrip("-").isdigit() for p in parts):
            return str(fn([int(p.strip()) for p in parts]))
    return None


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
    from graph_loss.create_graph import create_graph

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


def save_plot_cache(
    path: str,
    *,
    aligned: AlignedGraphs,
    grids: dict[str, dict[int, object]],
    names: list[str],
    prompt: str,
    answers: dict[str, str] | None = None,
    gold: str | None = None,
) -> None:
    """Persist everything the figures need (aligned adjacency + Fig F grids +
    predicted answers) so they can be redrawn without reloading models."""
    import pickle

    payload = {
        "labels": aligned.labels,
        "raw": {k: v.cpu() for k, v in aligned.raw.items()},
        "norm": {k: v.cpu() for k, v in aligned.norm.items()},
        "grids": grids,
        "names": names,
        "prompt": prompt,
        "answers": answers or {},
        "gold": gold,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    logger.info("Saved plot cache: %s", path)


def load_plot_cache(path: str):
    """Inverse of :func:`save_plot_cache`.

    Returns ``(aligned, grids, names, prompt, answers, gold)``.
    """
    import pickle

    with open(path, "rb") as f:
        payload = pickle.load(f)
    aligned = AlignedGraphs(
        labels=payload["labels"], raw=payload["raw"], norm=payload["norm"]
    )
    return (
        aligned,
        payload["grids"],
        payload["names"],
        payload["prompt"],
        payload.get("answers") or {},
        payload.get("gold"),
    )


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


def build_model_mlp_cache(
    adapter: HFLlamaGraphAdapter, model_name: str, dataset: str | None
) -> dict | None:
    """Build (or load from disk) the MLP-input cache for one model.

    Reuses the on-disk cache created during ANOVA graph building, so this is
    cheap on a second call. Returns None if no dataset is available.
    """
    if not dataset:
        return None
    try:
        from graph_loss.precompute_mlp_inputs import build_mlp_input_cache
        from utils import load_split

        data = load_split(dataset, "all")
        return build_mlp_input_cache(adapter, dataset, model_name, data_dict=data)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not build MLP cache for %s: %s", model_name, exc)
        return None


def aggregate_supernode_grid(
    adapter: HFLlamaGraphAdapter,
    graph,
    members,
    mlp_cache: dict | None,
):
    """Mean activation grid over a supernode's member neurons (2-D np array or None)."""
    from graph_loss.neuron_activation_heatmap import build_neuron_activation_write_result

    if not members or mlp_cache is None:
        return None
    locs = graph.neuron_locations.detach().cpu()[[int(m) for m in members]]
    try:
        awr = build_neuron_activation_write_result(adapter, locs, mlp_input_cache=mlp_cache)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Supernode activation grid failed: %s", exc)
        return None
    grid = torch.nanmean(awr.activations.float(), dim=0)  # mean over member neurons
    while grid.dim() > 2:
        grid = torch.nanmean(grid, dim=-1)
    return grid.numpy()


def _heatmap_montage_regenerate(
    results: dict[str, GraphPipelineResult],
    adapters: dict[str, HFLlamaGraphAdapter],
    *,
    teacher: str,
    student: str,
    dataset: str | None,
    out_dir: str,
    mlp_caches: dict[str, dict] | None = None,
) -> bool:
    if not dataset:
        logger.info("Fig E regenerate skipped: --dataset required to build the MLP cache.")
        return False

    plt = _import_pyplot()
    names = [teacher, student]
    mlp_caches = mlp_caches or {}

    def _supernode_grids(name: str):
        sg = results[name].supergraph
        graph = results[name].graph
        cache = mlp_caches.get(name) or build_model_mlp_cache(adapters[name], name, dataset)
        grids = [
            aggregate_supernode_grid(adapters[name], graph, members, cache)
            for members in sg.supernodes
        ]
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
            arr = grids[col] if col < len(grids) else None
            if arr is None or getattr(arr, "ndim", 0) != 2:
                ax.set_axis_off()
            else:
                ax.imshow(arr.T, origin="lower", aspect="auto", cmap="viridis")
                ax.set_xlabel("arg1", fontsize=7)
                ax.set_ylabel("arg2", fontsize=7)
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
    mlp_caches: dict[str, dict] | None = None,
) -> None:
    """Fig E: reuse saved heatmap PDFs if given, otherwise regenerate from the MLP cache."""
    if teacher_heatmap_dir and student_heatmap_dir:
        if _heatmap_montage_from_dirs(
            teacher_heatmap_dir, student_heatmap_dir, out_dir, all_pages=all_pages
        ):
            return
        logger.info("Fig E: falling back to regeneration.")
    _heatmap_montage_regenerate(
        results, adapters, teacher=teacher, student=student, dataset=dataset,
        out_dir=out_dir, mlp_caches=mlp_caches,
    )


# ---------------------------------------------------------------------------
# Fig F: computation flow with activation-heatmap nodes
# ---------------------------------------------------------------------------


# Plain-English meaning of each supernode label, for the Fig F legend.
CATEGORY_DESCRIPTIONS = {
    "arg1 range": "magnitude of the 1st operand",
    "arg1 units": "ones digit of the 1st operand",
    "arg2 range": "magnitude of the 2nd operand",
    "arg2 units": "ones digit of the 2nd operand",
    "arg3 range": "magnitude of the 3rd operand",
    "arg3 units": "ones digit of the 3rd operand",
    "arg1 units and arg2 units": "joint ones digits of both operands",
    "arg1 range and arg2 range": "joint magnitudes of both operands",
    "carry": "whether a carry occurs",
    "sum range": "magnitude of the sum",
    "sum units": "ones digit of the sum",
    "dla": "direct contribution to the output logits",
}


def _describe_label(label: str) -> str:
    key = label.lower().strip()
    if key in CATEGORY_DESCRIPTIONS:
        return CATEGORY_DESCRIPTIONS[key]
    if key.startswith("arg:"):
        return f"neurons aligned to input token '{label.split(':', 1)[1]}'"
    return label


def _wrap_label(label: str) -> str:
    return label.replace(" and ", "\n& ")


def _flow_positions(labels: list[str]) -> tuple[dict[int, tuple[float, float]], int]:
    """Tiered layout: arg inputs at bottom, sum/dla outputs at top, rest in the middle.

    Returns the positions plus the busiest tier size (for dynamic node sizing).
    """
    low = [l.lower() for l in labels]
    bottom = [i for i, l in enumerate(low) if l.startswith("arg")]
    top = [i for i, l in enumerate(low) if "sum" in l or "dla" in l]
    mid = [i for i in range(len(labels)) if i not in bottom and i not in top]
    pos: dict[int, tuple[float, float]] = {}
    # Tiers leave room for the input-prompt band (bottom) and answer band (top):
    # operand supernodes -> intermediate (carry) -> sum supernodes, read upward.
    for y, members in ((0.34, bottom), (0.56, mid), (0.76, top)):
        m = len(members)
        for k, i in enumerate(members):
            x = 0.07 + 0.86 * (k + 0.5) / m if m else 0.5
            pos[i] = (x, y)
    busiest = max((len(bottom), len(mid), len(top)), default=1)
    return pos, busiest


def _parse_operands(prompt: str) -> list[str]:
    """Operand strings from an arithmetic prompt, e.g. '36+59=' -> ['36','59']."""
    return re.findall(r"\d+", prompt)


def _operand_indices(label: str) -> list[int]:
    """1-based operand indices a supernode label refers to ('arg1 ...' -> [1])."""
    return [int(m) for m in re.findall(r"arg(\d+)", label.lower())]


def _canonical_edges(labels: list[str]) -> list[tuple[int, int]]:
    """Idealized two-digit-addition pathway as (src, tgt) supernode index pairs.

    Two parallel streams joined by the carry, with the joint "pair/lookup"
    supernodes as intermediate features (cf. Anthropic's lookup-table features):

      units stream:  arg{i} units -> arg1&arg2 units -> {sum units, carry}
                     (and arg{i} units -> sum units / carry directly, as skips)
      magnitude:     arg{i} range -> arg1&arg2 range -> sum range
                     (and arg{i} range -> sum range directly)
      carry bridge:  carry -> sum range ;  sum units -> sum range
    """
    low = [l.lower() for l in labels]

    def find(pred):
        return [i for i, l in enumerate(low) if pred(l)]

    units_indiv = find(lambda l: l.startswith("arg") and "units" in l and "and" not in l)
    range_indiv = find(lambda l: l.startswith("arg") and "range" in l and "and" not in l)
    joint_units = find(lambda l: "and" in l and "units" in l)
    joint_range = find(lambda l: "and" in l and "range" in l)
    sum_units = find(lambda l: "sum" in l and "units" in l)
    sum_range = find(lambda l: "sum" in l and "range" in l)
    carry = find(lambda l: l == "carry")

    units_all = units_indiv + joint_units
    range_all = range_indiv + joint_range

    edges: set[tuple[int, int]] = set()
    # individual operand features -> their joint "pair" feature
    for u in units_indiv:
        edges.update((u, t) for t in joint_units)
    for r in range_indiv:
        edges.update((r, t) for t in joint_range)
    # units stream -> ones digit of the sum and the carry
    for u in units_all:
        edges.update((u, t) for t in sum_units)
        edges.update((u, t) for t in carry)
    # magnitude stream -> magnitude of the sum
    for r in range_all:
        edges.update((r, t) for t in sum_range)
    # carry bumps the tens; ones-sum propagates into the tens
    for c in carry:
        edges.update((c, t) for t in sum_range)
    for su in sum_units:
        edges.update((su, t) for t in sum_range)
    return [(s, t) for s, t in edges if s != t]


def _curated_positions(labels: list[str]) -> tuple[dict[int, tuple[float, float]], int]:
    """Tiered layout for curated mode: individual operand features, then the
    joint pair/lookup features, then carry, then the sum features (read upward),
    so the skip-connection flow is legible."""
    low = [l.lower() for l in labels]
    indiv = [i for i, l in enumerate(low) if l.startswith("arg") and "and" not in l]
    joints = [i for i, l in enumerate(low) if "and" in l]
    carry = [i for i, l in enumerate(low) if l == "carry"]
    sums = [i for i, l in enumerate(low) if "sum" in l or "dla" in l]
    other = [i for i in range(len(labels))
             if i not in indiv and i not in joints and i not in carry and i not in sums]
    pos: dict[int, tuple[float, float]] = {}
    for y, members in ((0.27, indiv), (0.47, joints + other), (0.62, carry), (0.80, sums)):
        m = len(members)
        for k, i in enumerate(members):
            x = 0.08 + 0.84 * (k + 0.5) / m if m else 0.5
            pos[i] = (x, y)
    busiest = max((len(indiv), len(joints) + len(other), len(sums)), default=1)
    return pos, busiest


def _box_perimeter_point(
    center: tuple[float, float],
    toward: tuple[float, float],
    half_x: float,
    half_y: float,
    margin: float,
) -> tuple[float, float]:
    """Point on the (possibly non-square) node border toward another node, nudged
    out by ``margin`` along the connecting direction."""
    import math

    cx, cy = center
    dx, dy = toward[0] - cx, toward[1] - cy
    if dx == 0 and dy == 0:
        return center
    sx = abs(dx) / half_x if half_x > 0 else math.inf
    sy = abs(dy) / half_y if half_y > 0 else math.inf
    s = max(sx, sy)
    if s == 0:
        return center
    bx, by = cx + dx / s, cy + dy / s
    norm = math.hypot(dx, dy)
    return (bx + dx / norm * margin, by + dy / norm * margin)


def build_flow_grids(
    labels: list[str],
    *,
    results: dict[str, GraphPipelineResult],
    adapters: dict[str, HFLlamaGraphAdapter],
    mlp_caches: dict[str, dict],
    names: list[str],
) -> dict[str, dict[int, object]]:
    """Per-model {label_index -> mean activation grid} for Fig F. Heavy step
    (needs adapters + MLP cache); cache the result to redraw figures cheaply."""
    grids: dict[str, dict[int, object]] = {}
    for name in names:
        sg = results[name].supergraph
        graph = results[name].graph
        l2i = _label_to_index(sg)
        cache = mlp_caches.get(name)
        grids[name] = {
            li: aggregate_supernode_grid(
                adapters[name], graph, sg.supernodes[l2i[lbl]] if lbl in l2i else [], cache
            )
            for li, lbl in enumerate(labels)
        }
    return grids


def plot_circuit_with_heatmaps(
    aligned: AlignedGraphs,
    *,
    grids: dict[str, dict[int, object]],
    names: list[str],
    prompt: str,
    out_dir: str,
    answers: dict[str, str] | None = None,
    gold: str | None = None,
    mode: str = "faithful",
    weight_threshold: float = 0.04,
) -> None:
    """Fig F: per-model node-link circuit where each supernode node *is* its
    mean activation heatmap, read as an end-to-end flow from the input prompt
    (bottom) up through operand -> intermediate -> sum supernodes to the model's
    predicted answer (top). Mirrors Anthropic's addition attribution graph.

    ``mode``:
      * ``"faithful"`` (default, for the results comparison): draws every
        above-threshold supernode edge a model actually uses, plus a single
        prompt box fanning into the operand supernodes.
      * ``"curated"`` (for the intro/teaser): draws only the idealized addition
        pathway (edge width still = the model's measured weight, so a model that
        doesn't implement a step shows a faint/absent edge), and connects each
        operand token to *its own* supernodes (arg1 <- first operand, etc.),
        like Anthropic's intentional input wiring.

    ``answers`` maps each model name to its predicted first answer token; with a
    ``gold`` token the answer box is outlined green when correct and red when
    wrong, so a base student's broken computation *and* wrong answer show
    together.
    """
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    labels = aligned.labels
    if not labels:
        logger.warning("Fig F skipped: no shared supernodes.")
        return
    has_grid = any(
        any(v is not None for v in grids.get(n, {}).values()) for n in names
    )
    if not has_grid:
        logger.info("Fig F skipped: needs an MLP cache (pass --dataset).")
        return

    plt = _import_pyplot()

    curated = mode == "curated"
    pos, busiest = (_curated_positions(labels) if curated else _flow_positions(labels))
    low = [l.lower() for l in labels]
    bottom_idx = [i for i, l in enumerate(low) if l.startswith("arg")]
    top_idx = [i for i, l in enumerate(low) if "sum" in l or "dla" in l]
    PROMPT_Y, ANSWER_Y = 0.05, 0.93

    # Taller panels: extra vertical room for the prompt and answer bands
    # (curated mode adds an extra tier, so it gets a bit more height).
    col_w, col_h = 6.6, (8.2 if curated else 7.2)
    fig, axes = plt.subplots(1, len(names), figsize=(col_w * len(names), col_h))
    if len(names) == 1:
        axes = [axes]

    side_in = min(0.95, 0.72 * col_w / (busiest + 1))  # square thumbnail side, inches
    thumb_w, thumb_h = side_in / col_w, side_in / col_h
    half_x, half_y = thumb_w / 2.0, thumb_h / 2.0

    # Deliberate, discrete edge tiers (thickness + shade + opacity all step
    # together) so strong vs. weak routing is unmistakable at a glance.
    STRONG, MEDIUM = 0.20, 0.10
    edge_tiers = [
        ("strong",  STRONG, float("inf"), dict(lw=3.4, alpha=0.95, color="#1a1a1a", mscale=15)),
        ("medium",  MEDIUM, STRONG,       dict(lw=1.5, alpha=0.75, color="#5a5a5a", mscale=10)),
        ("weak", weight_threshold, MEDIUM, dict(lw=0.6, alpha=0.5, color="#aaaaaa", mscale=7)),
    ]

    def _edge_style(w: float):
        for _, lo, hi, style in edge_tiers:
            if lo <= w < hi:
                return style
        return None

    canonical = _canonical_edges(labels) if curated else None
    operands = _parse_operands(prompt) if curated else []
    # Operand-token x positions across the bottom band (curated input wiring).
    op_x = {}
    if curated and operands:
        for k in range(len(operands)):
            frac = (k + 0.5) / len(operands)
            op_x[k + 1] = 0.12 + 0.76 * frac

    label_box = dict(boxstyle="round,pad=0.32", fc="white", ec="#bcbcbc", lw=0.8)

    for ax, name in zip(axes, names):
        norm = aligned.norm[name]
        # Choose which edges to draw: the idealized pathway (curated) or every
        # above-threshold edge the model uses (faithful).
        edges = []
        if curated:
            # Only draw a canonical edge the model actually carries weight on,
            # so a model that skips a step simply has no arrow there.
            for src, tgt in canonical:
                w = float(norm[tgt, src].item())
                style = _edge_style(w)
                if style is not None:
                    edges.append((w, src, tgt, style))
        else:
            for tgt in range(len(labels)):
                for src in range(len(labels)):
                    if tgt == src:
                        continue
                    w = float(norm[tgt, src].item())
                    style = _edge_style(w)
                    if style is not None:
                        edges.append((w, src, tgt, style))
        for w, src, tgt, style in sorted(edges, key=lambda e: e[0]):
            start = _box_perimeter_point(pos[src], pos[tgt], half_x, half_y, 0.006)
            end = _box_perimeter_point(pos[tgt], pos[src], half_x, half_y, 0.006)
            ax.add_patch(FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=style["mscale"],
                lw=style["lw"], alpha=style["alpha"], color=style["color"],
                linestyle=style.get("ls", "-"),
                connectionstyle="arc3,rad=0.05", shrinkA=0, shrinkB=0, zorder=1,
            ))

        if curated and op_x:
            # Deliberate input wiring: each operand token feeds only its own
            # supernodes (arg1 <- first operand, joint nodes <- both).
            for i in bottom_idx:
                ops = _operand_indices(labels[i]) or list(op_x.keys())
                for o in ops:
                    if o not in op_x:
                        continue
                    sx = op_x[o]
                    end = _box_perimeter_point(pos[i], (sx, PROMPT_Y), half_x, half_y, 0.006)
                    ax.add_patch(FancyArrowPatch(
                        (sx, PROMPT_Y + 0.02), end, arrowstyle="-|>", mutation_scale=8,
                        lw=1.1, alpha=0.7, color="#7a7a7a",
                        connectionstyle="arc3,rad=0.04", shrinkA=0, shrinkB=0, zorder=1,
                    ))
            for o, sx in op_x.items():
                ax.text(sx, PROMPT_Y, operands[o - 1], ha="center", va="center",
                        fontsize=11, family="monospace", weight="bold", zorder=4,
                        bbox=dict(boxstyle="round,pad=0.45", fc="#f3f3f3",
                                  ec="#888888", lw=1.0))
        else:
            # Faithful: a single prompt box fans into all operand supernodes.
            for i in bottom_idx:
                end = _box_perimeter_point(pos[i], (pos[i][0], PROMPT_Y), half_x, half_y, 0.006)
                ax.add_patch(FancyArrowPatch(
                    (pos[i][0], PROMPT_Y + 0.02), end, arrowstyle="-|>", mutation_scale=7,
                    lw=0.7, alpha=0.45, color="#b0b0b0",
                    connectionstyle="arc3,rad=0.0", shrinkA=0, shrinkB=0, zorder=1,
                ))
            ax.text(0.5, PROMPT_Y, prompt, ha="center", va="center", fontsize=11,
                    family="monospace", weight="bold", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.5", fc="#f3f3f3", ec="#888888", lw=1.0))

        # Sum supernodes -> predicted answer (medium connectors).
        pred = (answers or {}).get(name)
        if pred is not None:
            for i in top_idx:
                start = _box_perimeter_point(pos[i], (pos[i][0], ANSWER_Y), half_x, half_y, 0.006)
                ax.add_patch(FancyArrowPatch(
                    start, (pos[i][0], ANSWER_Y - 0.02), arrowstyle="-|>", mutation_scale=10,
                    lw=1.6, alpha=0.8, color="#5a5a5a",
                    connectionstyle="arc3,rad=0.0", shrinkA=0, shrinkB=0, zorder=1,
                ))
            if gold is None:
                ec, tc = "#2b6cb0", "#1a1a1a"
            elif pred.strip() == gold.strip():
                ec, tc = "#2e7d32", "#2e7d32"   # correct -> green
            else:
                ec, tc = "#c62828", "#c62828"   # wrong -> red
            shown = repr(pred) if pred.strip() != pred else pred
            ax.text(0.5, ANSWER_Y, shown, ha="center", va="center", fontsize=13,
                    family="monospace", weight="bold", color=tc, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=ec, lw=2.0))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        ax.set_title(name, fontsize=12, pad=14)
        for li, (x, y) in pos.items():
            inset = ax.inset_axes([x - half_x, y - half_y, thumb_w, thumb_h])
            arr = grids[name][li]
            if arr is not None and getattr(arr, "ndim", 0) == 2:
                # Robust per-image contrast so each activation pattern pops.
                a = np.asarray(arr, dtype=float)
                finite = a[np.isfinite(a)]
                vmin, vmax = (np.percentile(finite, [2, 98]) if finite.size
                              else (None, None))
                if vmin is not None and vmax <= vmin:
                    vmin, vmax = None, None
                inset.imshow(a.T, origin="lower", aspect="auto", cmap="viridis",
                             vmin=vmin, vmax=vmax, interpolation="nearest")
            else:
                inset.set_facecolor("#eeeeee")
            inset.set_xticks([])
            inset.set_yticks([])
            for sp in inset.spines.values():
                sp.set_edgecolor("#2b6cb0")
                sp.set_linewidth(1.3)
            # Node label in a clean caption box below the heatmap, staggered into
            # two rows on the busy bottom tier so adjacent boxes never collide.
            extra = 0.055 if (y < 0.45 and li % 2 == 1) else 0.0
            ax.text(x, y - half_y - 0.016 - extra, _wrap_label(labels[li]),
                    ha="center", va="top", fontsize=7.5, weight="bold",
                    color="#1a1a1a", zorder=3, bbox=label_box)

    legend_handles = [
        Line2D([0], [0], color=s["color"], lw=s["lw"]) for _, _, _, s in edge_tiers
    ]
    legend_labels = [
        f"strong (\u2265{STRONG:.2f})",
        f"medium ({MEDIUM:.2f}\u2013{STRONG:.2f})",
        f"weak (<{MEDIUM:.2f})",
    ]
    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=3,
               frameon=False, fontsize=8.5, title="routing weight (row-normalized)",
               title_fontsize=8.5, bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(f"Computation flow for  \u201c{prompt}\u201d", fontsize=13)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.07, wspace=0.06)
    _save(fig, out_dir, "figF_circuit_heatmaps")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render static teacher-vs-student circuit figures for the write-up."
    )
    parser.add_argument("--teacher", required=True, help="Teacher model id or local path.")
    parser.add_argument("--student", default=None,
                        help="Trained student model id or checkpoint path. If omitted, the figures "
                             "compare the teacher against --student-base only.")
    parser.add_argument("--student-base", "--student_base", dest="student_base", default=None,
                        help="Untrained base student (model id). Used as a before/after column when "
                             "--student is given, or as the sole student column when it isn't.")
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
    parser.add_argument("--plot-cache", "--plot_cache", dest="plot_cache", default=None,
                        help="Path to a .pkl of precomputed plot data. If it exists, models/graphs "
                             "are skipped and figures are redrawn from it; otherwise it is created "
                             "after building so later reruns are fast.")
    parser.add_argument("--refresh-cache", "--refresh_cache", dest="refresh_cache",
                        action="store_true",
                        help="Rebuild graphs even if --plot-cache exists, then overwrite the cache.")
    parser.add_argument("--flow-mode", "--flow_mode", dest="flow_mode",
                        choices=["faithful", "curated"], default="faithful",
                        help="Fig F style: 'faithful' draws every above-threshold edge a model "
                             "uses (results comparison); 'curated' draws only the idealized "
                             "addition pathway with per-operand input wiring (intro/teaser).")
    from graph_loss.utils import add_graph_build_args

    add_graph_build_args(parser)
    return parser


def main() -> None:
    from utils import DIR_ROOT

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

    cache_path = None
    if args.plot_cache:
        cache_path = (args.plot_cache if os.path.isabs(args.plot_cache)
                      else os.path.join(DIR_ROOT, args.plot_cache))

    results: dict[str, GraphPipelineResult] = {}
    adapters: dict[str, HFLlamaGraphAdapter] = {}
    mlp_caches: dict[str, dict] = {}

    # ── Fast path: redraw figures straight from a cached payload ───────────
    if cache_path and os.path.isfile(cache_path) and not args.refresh_cache:
        logger.info("Loading plot cache (skipping models/graphs): %s", cache_path)
        aligned, grids, fig_names, prompt, answers, gold = load_plot_cache(cache_path)
        teacher_name = fig_names[0]
        student_name = fig_names[1] if len(fig_names) > 1 else fig_names[0]
        base_name = fig_names[2] if len(fig_names) > 2 else None
        _run_figures(
            aligned=aligned, grids=grids, fig_names=fig_names, prompt=prompt,
            teacher_name=teacher_name, student_name=student_name, base_name=base_name,
            results=results, adapters=adapters, mlp_caches=mlp_caches,
            answers=answers, gold=gold, args=args, out_dir=out_dir,
        )
        logger.info("Done. Figures written to %s", out_dir)
        return

    if not args.student and not args.student_base:
        build_parser().error(
            "provide --student (trained checkpoint) and/or --student-base (base model)."
        )

    from graph_loss.hf_adapter import HFLlamaGraphAdapter
    from utils import load_model

    teacher_name = "teacher"
    # The second column is the trained student when given; otherwise the base
    # student stands in for it. A separate "base student" column is only added
    # when BOTH a trained student and a base model are supplied.
    if args.student:
        student_name = "student"
        student_id = args.student
    else:
        student_name = "base student"
        student_id = args.student_base
    base_name = "base student" if (args.student and args.student_base) else None

    # ── Teacher ───────────────────────────────────────────────────────────
    logger.info("Loading teacher: %s", args.teacher)
    teacher_model, teacher_tok = load_model(args.teacher)
    teacher_model.eval()
    teacher_adapter = HFLlamaGraphAdapter(teacher_model, teacher_tok, _model_device_of(teacher_model))
    teacher_dla = last_token_logits(teacher_adapter, args.prompt)
    logger.info("Building teacher supergraph")
    teacher_result = build_supergraph(teacher_adapter, args.prompt, cfg, model_name=args.teacher)

    # ── Student column (trained student, or base student if none given) ────
    logger.info("Loading %s: %s", student_name, student_id)
    student_model, student_tok = load_model(student_id)
    student_model.eval()
    student_adapter = HFLlamaGraphAdapter(student_model, student_tok, _model_device_of(student_model))
    logger.info("Building %s supergraph", student_name)
    student_result = build_supergraph(
        student_adapter, args.prompt, cfg,
        model_name=student_id,
        dla_model_logits=teacher_dla.to(_model_device_of(student_model)),
    )

    results = {teacher_name: teacher_result, student_name: student_result}
    adapters = {teacher_name: teacher_adapter, student_name: student_adapter}
    named_supergraphs = {
        teacher_name: teacher_result.supergraph,
        student_name: student_result.supergraph,
    }
    model_ids = {teacher_name: args.teacher, student_name: student_id}

    # ── Optional untrained base student (before/after column) ──────────────
    if base_name:
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
        model_ids[base_name] = args.student_base

    aligned = align_supergraphs(named_supergraphs, exclude_self_loops=args.exclude_self_loops)
    logger.info("Aligned %d shared supernodes: %s", len(aligned.labels), aligned.labels)

    # ── MLP caches (shared by Fig E regen + Fig F); only when --dataset given ──
    fig_names = [teacher_name, student_name] + ([base_name] if base_name else [])
    if args.dataset:
        for nm in fig_names:
            logger.info("Preparing MLP cache for %s", nm)
            cache = build_model_mlp_cache(adapters[nm], model_ids[nm], args.dataset)
            if cache is not None:
                mlp_caches[nm] = cache

    # Precompute Fig F grids (the heavy per-supernode aggregation) once.
    grids = build_flow_grids(
        aligned.labels, results=results, adapters=adapters,
        mlp_caches=mlp_caches, names=fig_names,
    )

    # What each model actually generates (greedy, parsed like eval); correctness
    # is judged against the true arithmetic answer, falling back to the teacher's.
    answers = {nm: model_prediction(adapters[nm], args.prompt, dataset=args.dataset)
               for nm in fig_names}
    gold = true_answer(args.prompt) or answers.get(teacher_name)
    logger.info("Predicted answers: %s (gold=%r)", answers, gold)

    if cache_path:
        save_plot_cache(
            cache_path, aligned=aligned, grids=grids, names=fig_names,
            prompt=args.prompt, answers=answers, gold=gold,
        )

    _run_figures(
        aligned=aligned, grids=grids, fig_names=fig_names, prompt=args.prompt,
        teacher_name=teacher_name, student_name=student_name, base_name=base_name,
        results=results, adapters=adapters, mlp_caches=mlp_caches,
        answers=answers, gold=gold, args=args, out_dir=out_dir,
    )
    logger.info("Done. Figures written to %s", out_dir)


def _run_figures(
    *,
    aligned: AlignedGraphs,
    grids: dict[str, dict[int, object]],
    fig_names: list[str],
    prompt: str,
    teacher_name: str,
    student_name: str,
    base_name: str | None,
    results: dict[str, GraphPipelineResult],
    adapters: dict[str, HFLlamaGraphAdapter],
    mlp_caches: dict[str, dict],
    args,
    out_dir: str,
    answers: dict[str, str] | None = None,
    gold: str | None = None,
) -> None:
    """Render every figure (each guarded so one failure doesn't abort the rest).

    Figs A/B/C/F + Fig E-from-PDFs work from cached data alone; Fig E
    regeneration additionally needs live ``results``/``adapters``.
    """
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
            dataset=args.dataset, out_dir=out_dir, all_pages=args.heatmap_all_pages,
            mlp_caches=mlp_caches)),
        ("Fig F flow+heatmaps", lambda: plot_circuit_with_heatmaps(
            aligned, grids=grids, names=fig_names, prompt=prompt, out_dir=out_dir,
            answers=answers, gold=gold, mode=getattr(args, "flow_mode", "faithful"))),
    ]
    for label, fn in figures:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s failed: %s", label, exc)


def _model_device_of(model) -> torch.device:
    return next(model.parameters()).device


if __name__ == "__main__":
    main()

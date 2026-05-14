"""Inspect and compare teacher/student supergraphs for a single prompt.

Loads the teacher supergraph from a TeacherDataCache and the student
supergraph from a .pt file written by ``python -m graph_loss --supergraph_output_path``.
Prints a side-by-side table of supernodes (label, size, top edges) and a
focused view on the "carry" supernode (or any label of interest).

Usage:
    python -m graph_loss.inspect_supergraph \
        --teacher-cache "/path/to/teacher_cache/full_search_v2/22_add_tight_5000" \
        --student-supergraph "/path/to/supergraph_1b.pt" \
        --prompt "77+87=" --answer 164 \
        --top-k 3 --focus-label carry

If the teacher cache uses a different prompt key, run with --list-prompts
to print available cached prompts.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import torch

from .teacher_data_cache import TeacherDataCache


def _normalize_supergraph_payload(obj: Any) -> dict[str, Any]:
    """Both teacher cache and standalone .pt save use the same payload keys."""
    if isinstance(obj, dict):
        return obj
    payload = {
        "supernode_adjacency_matrix": obj.supernode_adjacency_matrix,
        "supernodes": obj.supernodes,
        "supernode_prob_deltas": getattr(obj, "supernode_prob_deltas", None),
        "supernode_labels": getattr(obj, "supernode_labels", None),
    }
    return payload


def _load_student_supergraph(path: str) -> dict[str, Any]:
    raw = torch.load(path, map_location="cpu")
    return _normalize_supergraph_payload(raw)


def _supernode_size(supernode_entry: Any) -> int:
    """Supernodes can be a list of node-id sets, dicts, or similar."""
    if hasattr(supernode_entry, "__len__"):
        try:
            return len(supernode_entry)
        except TypeError:
            pass
    return 0


def _format_label(labels: list[list[str]] | None, idx: int) -> str:
    if labels is None or idx >= len(labels) or not labels[idx]:
        return "(no labels)"
    return ", ".join(labels[idx])


def _normalize_row(row: torch.Tensor) -> torch.Tensor:
    abs_row = row.abs().float()
    total = abs_row.sum().clamp(min=1e-12)
    return abs_row / total


def _top_k_edges(row_or_col: torch.Tensor, k: int) -> list[tuple[int, float, float]]:
    """Return list of (idx, raw_value, pct_of_l1_total) for top-k by |value|."""
    abs_vals = row_or_col.abs().float()
    total = abs_vals.sum().clamp(min=1e-12).item()
    k = min(k, abs_vals.numel())
    top_vals, top_idx = torch.topk(abs_vals, k)
    out = []
    for i in range(k):
        ix = int(top_idx[i].item())
        raw = float(row_or_col[ix].item())
        pct = 100.0 * abs_vals[ix].item() / total if total > 0 else 0.0
        out.append((ix, raw, pct))
    return out


def _print_supernode_summary(
    title: str,
    payload: dict[str, Any],
    *,
    top_k: int,
) -> None:
    W = payload["supernode_adjacency_matrix"]
    if not isinstance(W, torch.Tensor):
        W = torch.tensor(W)
    W = W.cpu().float()
    supernodes = payload.get("supernodes", []) or []
    labels = payload.get("supernode_labels")
    n = W.shape[0]

    print(f"\n=== {title} ===")
    print(f"  n_supernodes: {n}")
    print(f"  adjacency shape: {tuple(W.shape)}")
    print(f"  adjacency |.| sum: {W.abs().sum().item():.4f}")
    print(f"  adjacency mean |.|: {W.abs().mean().item():.6f}")

    print(f"\n  {'idx':>4} {'size':>5} {'out_total':>11} {'in_total':>11}  label(s)")
    for i in range(n):
        size = _supernode_size(supernodes[i]) if i < len(supernodes) else 0
        out_t = W[i, :].abs().sum().item()
        in_t = W[:, i].abs().sum().item()
        print(f"  {i:>4} {size:>5} {out_t:>11.4f} {in_t:>11.4f}  {_format_label(labels, i)}")

    print(f"\n  Top-{top_k} edges per supernode (|.| as % of L1 row/col total):")
    for i in range(n):
        lbl = _format_label(labels, i)
        out_edges = _top_k_edges(W[i, :], top_k)
        in_edges = _top_k_edges(W[:, i], top_k)
        out_str = ", ".join(
            f"->{j}({_format_label(labels, j)[:14]}) {pct:5.1f}%"
            for j, _, pct in out_edges
        )
        in_str = ", ".join(
            f"<-{j}({_format_label(labels, j)[:14]}) {pct:5.1f}%"
            for j, _, pct in in_edges
        )
        print(f"  [{i}] {lbl}")
        print(f"      out: {out_str}")
        print(f"      in : {in_str}")


def _focus_on_label(
    title: str,
    payload: dict[str, Any],
    label: str,
    *,
    top_k: int,
) -> None:
    """Drill into a supernode whose first label matches ``label`` (case-insensitive)."""
    labels = payload.get("supernode_labels")
    W = payload["supernode_adjacency_matrix"]
    if not isinstance(W, torch.Tensor):
        W = torch.tensor(W)
    W = W.cpu().float()

    matches: list[int] = []
    if labels is not None:
        for i, lab in enumerate(labels):
            if lab and lab[0].lower() == label.lower():
                matches.append(i)

    print(f"\n--- {title} | focus label = '{label}' ---")
    if not matches:
        print(f"  (no supernode has first label '{label}')")
        return

    for i in matches:
        out_edges = _top_k_edges(W[i, :], top_k)
        in_edges = _top_k_edges(W[:, i], top_k)
        out_total = W[i, :].abs().sum().item()
        in_total = W[:, i].abs().sum().item()
        max_other_out = max(
            (W[j, :].abs().sum().item() for j in range(W.shape[0]) if j != i),
            default=0.0,
        )
        max_other_in = max(
            (W[:, j].abs().sum().item() for j in range(W.shape[0]) if j != i),
            default=0.0,
        )

        print(f"  supernode #{i} labels={_format_label(labels, i)}")
        print(f"  out_total={out_total:.4f}  (max non-this out_total in graph={max_other_out:.4f})")
        print(f"  in_total ={in_total:.4f}  (max non-this in_total  in graph={max_other_in:.4f})")
        print(f"  outgoing edges (this -> X, top-{top_k}):")
        for j, raw, pct in out_edges:
            print(f"    -> #{j} {_format_label(labels, j):30s}  raw={raw:+.4e}  {pct:5.1f}%")
        print(f"  incoming edges (X -> this, top-{top_k}):")
        for j, raw, pct in in_edges:
            print(f"    <- #{j} {_format_label(labels, j):30s}  raw={raw:+.4e}  {pct:5.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-cache", required=True)
    parser.add_argument("--student-supergraph", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--answer", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--focus-label", default="carry")
    parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="Print all (prompt, answer) keys present in the teacher cache and exit.",
    )
    args = parser.parse_args()

    cache = TeacherDataCache(args.teacher_cache)

    if args.list_prompts:
        keys = sorted(cache._samples.keys())
        print(f"Teacher cache has {len(keys)} prompts:")
        for p, a in keys[:30]:
            print(f"  {p!r} -> {a}")
        if len(keys) > 30:
            print(f"  ... ({len(keys) - 30} more)")
        return

    if (args.prompt, args.answer) not in cache._samples:
        print(f"ERROR: prompt={args.prompt!r} answer={args.answer} not in teacher cache.", file=sys.stderr)
        print("Run with --list-prompts to see available keys.", file=sys.stderr)
        sys.exit(1)

    teacher_payload = cache.load_teacher_supergraph(args.prompt, args.answer)
    student_payload = _load_student_supergraph(args.student_supergraph)

    print(f"Prompt: {args.prompt!r}  Answer: {args.answer}")
    _print_supernode_summary("TEACHER (8B, from cache)", teacher_payload, top_k=args.top_k)
    _print_supernode_summary("STUDENT (1B, from .pt)", student_payload, top_k=args.top_k)
    _focus_on_label("TEACHER (8B)", teacher_payload, args.focus_label, top_k=args.top_k)
    _focus_on_label("STUDENT (1B)", student_payload, args.focus_label, top_k=args.top_k)


if __name__ == "__main__":
    main()

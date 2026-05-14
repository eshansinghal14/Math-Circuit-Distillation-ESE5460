"""Compare teacher vs trained-student supergraphs after training.

Loads the trained student model from an HF checkpoint, builds its attribution
graph + supergraph for each specified prompt using the pre-computed fixed-label
clustering, and compares against the teacher supergraph from the teacher data
cache.

Outputs per prompt:
  1. Student model accuracy (generate vs expected answer parsed from prompt)
  2. Side-by-side supernode summary (sizes, edge totals, top-k edges)
  3. Focus-label breakdown for --focus-label (default "carry")
  4. JSD alignment score per matched label pair + mean across all pairs
  5. Top-3 outgoing edges for focus-label: teacher vs student

Usage:
    python -m graph_loss.compare_circuits \\
        --student-checkpoint SAVE_DIR/student_model \\
        --student-model meta-llama/Llama-3.2-1B-Instruct \\
        --teacher-cache /content/local_caches/teacher_cache/full_search_v2/22_add_tight_5000 \\
        --fixed-labels /content/local_caches/fixed_labels/fixed_labels_1b.json \\
        --prompts "77+87=" "36+59=" \\
        --top-k 5 --focus-label carry
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Ensure src/ is importable when run as -m from the src/ directory
# ---------------------------------------------------------------------------
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


# ===========================================================================
# Helpers copied/adapted from inspect_supergraph.py
# ===========================================================================

def _normalize_supergraph_payload(obj: Any) -> dict[str, Any]:
    """Accept both raw dicts (from cache) and SuperGraph NamedTuples."""
    if isinstance(obj, dict):
        return obj
    return {
        "supernode_adjacency_matrix": obj.supernode_adjacency_matrix,
        "supernodes": obj.supernodes,
        "supernode_prob_deltas": getattr(obj, "supernode_prob_deltas", None),
        "supernode_labels": getattr(obj, "supernode_labels", None),
    }


def _supernode_size(supernode_entry: Any) -> int:
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
) -> list[tuple[int, float, float]] | None:
    """Drill into a supernode whose first label matches ``label`` (case-insensitive).

    Returns the top-k outgoing edges of the first matching supernode (for the
    carry-check below), or None if no match.
    """
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
        return None

    first_out_edges = None
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

        if first_out_edges is None:
            first_out_edges = out_edges

    return first_out_edges


# ===========================================================================
# Supergraph adjacency aggregation (mirrored from training._aggregate_supergraph_adjacency)
# ===========================================================================

def _aggregate_supergraph_adjacency_local(graph, supernodes: list[list[int]]):
    """Build [n_sn, n_sn] supernode adjacency from a fine-grained graph."""
    from graph_loss.graph import normalize_matrix, SuperGraph

    adj_matrix_norm = normalize_matrix(graph.adjacency_matrix)
    num_supernodes = len(supernodes)
    rows = []
    for t in range(num_supernodes):
        if not supernodes[t]:
            row_entries = [
                torch.zeros((), device=adj_matrix_norm.device, dtype=adj_matrix_norm.dtype)
                for _ in range(num_supernodes)
            ]
            rows.append(torch.stack(row_entries))
            continue
        total_input = torch.abs(adj_matrix_norm[:, supernodes[t]]).sum(dim=0)
        internal_input = torch.abs(adj_matrix_norm[supernodes[t]][:, supernodes[t]]).sum(dim=0)
        frac_external = (total_input - internal_input) / total_input.clamp(min=1e-10)
        row_entries = []
        for s in range(num_supernodes):
            if not supernodes[s]:
                row_entries.append(torch.zeros((), device=adj_matrix_norm.device, dtype=adj_matrix_norm.dtype))
                continue
            sum_A = adj_matrix_norm[supernodes[t]][:, supernodes[s]].sum(dim=1)
            entry = (
                (frac_external * sum_A).sum(dim=0)
                / frac_external.sum(dim=0).clamp(min=1e-10)
            )
            row_entries.append(entry)
        rows.append(torch.stack(row_entries))
    supernode_adj = torch.stack(rows)
    return SuperGraph(
        supernode_adjacency_matrix=supernode_adj,
        supernodes=supernodes,
    )


# ===========================================================================
# JSD alignment computation
# ===========================================================================

def _compute_jsd_per_label_pair(
    teacher_payload: dict[str, Any],
    student_payload: dict[str, Any],
    *,
    epsilon: float = 1e-8,
) -> tuple[dict[str, float], float]:
    """Compute JSD between teacher and student adjacency rows, matched by label.

    For each label L that appears as first label in both teacher and student
    supernodes, project both adjacency rows onto the common label space (summing
    mass from supernodes sharing a label) and compute JSD between the two
    resulting distributions.

    Returns:
        label_jsd: dict mapping label → JSD value
        mean_jsd: mean JSD across all matched pairs (alignment score)
    """
    t_labels: list[list[str]] | None = teacher_payload.get("supernode_labels")
    s_labels: list[list[str]] | None = student_payload.get("supernode_labels")

    W_T = teacher_payload["supernode_adjacency_matrix"]
    W_S = student_payload["supernode_adjacency_matrix"]
    if not isinstance(W_T, torch.Tensor):
        W_T = torch.tensor(W_T)
    if not isinstance(W_S, torch.Tensor):
        W_S = torch.tensor(W_S)
    W_T = W_T.cpu().float()
    W_S = W_S.cpu().float()

    if t_labels is None or s_labels is None:
        print("  (cannot compute label-based JSD — supernode_labels missing in one or both payloads)")
        return {}, float("nan")

    # Build label → index maps (first label of each supernode)
    t_label_to_idx: dict[str, int] = {}
    for i, lab in enumerate(t_labels):
        if lab:
            t_label_to_idx.setdefault(lab[0], i)

    s_label_to_idx: dict[str, int] = {}
    for i, lab in enumerate(s_labels):
        if lab:
            s_label_to_idx.setdefault(lab[0], i)

    common_labels = sorted(set(t_label_to_idx) & set(s_label_to_idx))
    if not common_labels:
        print("  (no common labels between teacher and student — cannot compute JSD)")
        return {}, float("nan")

    label_jsd: dict[str, float] = {}

    for src_label in common_labels:
        t_src = t_label_to_idx[src_label]
        s_src = s_label_to_idx[src_label]

        # Project teacher row onto common-label space
        t_row_proj = torch.zeros(len(common_labels))
        for j, dst_label in enumerate(common_labels):
            t_dst = t_label_to_idx[dst_label]
            t_row_proj[j] = W_T[t_src, t_dst].abs()

        # Project student row onto common-label space
        s_row_proj = torch.zeros(len(common_labels))
        for j, dst_label in enumerate(common_labels):
            s_dst = s_label_to_idx[dst_label]
            s_row_proj[j] = W_S[s_src, s_dst].abs()

        # L1-normalise → distributions
        t_dist = t_row_proj / t_row_proj.sum().clamp(min=epsilon)
        s_dist = s_row_proj / s_row_proj.sum().clamp(min=epsilon)

        # JSD = 0.5 * KL(t||m) + 0.5 * KL(s||m), m = (t+s)/2
        m_dist = 0.5 * (t_dist + s_dist)
        kl_t_m = (t_dist * ((t_dist + epsilon).log() - (m_dist + epsilon).log())).sum()
        kl_s_m = (s_dist * ((s_dist + epsilon).log() - (m_dist + epsilon).log())).sum()
        jsd = 0.5 * (kl_t_m + kl_s_m)
        label_jsd[src_label] = float(jsd.item())

    mean_jsd = sum(label_jsd.values()) / len(label_jsd) if label_jsd else float("nan")
    return label_jsd, mean_jsd


# ===========================================================================
# Accuracy check helpers
# ===========================================================================

def _parse_answer_from_prompt(prompt: str) -> int | None:
    """Parse the expected answer from a prompt like '77+87=' → 164."""
    expr = prompt.rstrip("= ").strip()
    try:
        result = eval(expr, {"__builtins__": {}})  # noqa: S307
        return int(result)
    except Exception:
        return None


def _check_student_accuracy(
    model,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int = 8,
    device: torch.device,
) -> tuple[int, int]:
    """Run model.generate() for each prompt and check vs expected answer.

    Returns (n_correct, n_total).
    """
    n_correct = 0
    n_total = 0

    for prompt in prompts:
        expected = _parse_answer_from_prompt(prompt)
        if expected is None:
            print(f"  [accuracy] Could not parse expected answer from {prompt!r} — skipping")
            continue

        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(device)

        with torch.no_grad():
            out_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = out_ids[0, input_ids.shape[1]:]
        generated_str = tokenizer.decode(generated, skip_special_tokens=True).strip()
        expected_str = str(expected)
        correct = generated_str.startswith(expected_str)
        n_total += 1
        if correct:
            n_correct += 1
        status = "✓" if correct else "✗"
        print(f"  {status}  {prompt!r}  →  got {generated_str!r}  (expected {expected_str!r})")

    return n_correct, n_total


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--student-checkpoint", required=True,
        help="Path to saved HF student model directory (e.g. SAVE_DIR/student_model)",
    )
    parser.add_argument(
        "--student-model", required=True,
        help="Base HF model ID for tokenizer (e.g. meta-llama/Llama-3.2-1B-Instruct)",
    )
    parser.add_argument(
        "--teacher-cache", required=True,
        help="Path to teacher data cache directory",
    )
    parser.add_argument(
        "--fixed-labels", required=True,
        help="Path to JSON fixed-label mapping from precompute_fixed_labels.py",
    )
    parser.add_argument(
        "--prompts", nargs="+", required=True,
        help="One or more prompt strings (e.g. '77+87=' '36+59=')",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top edges to show")
    parser.add_argument("--focus-label", default="carry", help="Label to focus on in breakdown")
    parser.add_argument(
        "--prop-neurons-per-layer", type=float, default=5e-4,
        help="Fraction of neurons per layer to select during attribution",
    )
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--device", default=None, help="Device override (e.g. 'cuda', 'cpu')")
    parser.add_argument(
        "--list-prompts", action="store_true",
        help="List all (prompt, answer) keys in the teacher cache and exit",
    )
    args = parser.parse_args()

    # --- Device / dtype ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # --- Teacher cache ---
    print(f"Loading teacher cache from: {args.teacher_cache!r}")
    from graph_loss.teacher_data_cache import TeacherDataCache
    cache = TeacherDataCache(args.teacher_cache)

    if args.list_prompts:
        keys = sorted(cache._samples.keys())
        print(f"Teacher cache has {len(keys)} prompts:")
        for p, a in keys[:40]:
            print(f"  {p!r} -> {a}")
        if len(keys) > 40:
            print(f"  ... ({len(keys) - 40} more)")
        return

    # --- Fixed labels ---
    print(f"Loading fixed labels from: {args.fixed_labels!r}")
    with open(args.fixed_labels, encoding="utf-8") as f:
        fixed_labels_dict: dict[str, str] = json.load(f)
    print(f"  {len(fixed_labels_dict)} labeled neurons")

    # --- Student model ---
    print(f"\nLoading trained student from checkpoint: {args.student_checkpoint!r}")
    from utils.hf_models import load_student_model_for_distillation
    student_model, tokenizer = load_student_model_for_distillation(
        student_source=args.student_checkpoint,
        student_model_id=args.student_model,
        device=device,
    )
    student_model = student_model.to(dtype=dtype)
    student_model.eval()

    from graph_loss.hf_adapter import HFLlamaGraphAdapter
    adapter = HFLlamaGraphAdapter(student_model, tokenizer, device)

    from graph_loss.graph import build_super_graph

    # === Accuracy pass ===
    print("\n" + "=" * 70)
    print("STUDENT ACCURACY ON SPECIFIED PROMPTS")
    print("=" * 70)
    n_correct, n_total = _check_student_accuracy(
        student_model, tokenizer, args.prompts, device=device
    )
    if n_total > 0:
        print(f"\n  Accuracy: {n_correct}/{n_total} = {100.0 * n_correct / n_total:.1f}%")
    else:
        print("  (no prompts with parseable answers)")

    # === Per-prompt circuit comparison ===
    all_mean_jsds: list[float] = []

    for prompt in args.prompts:
        expected_answer = _parse_answer_from_prompt(prompt)

        print("\n" + "=" * 70)
        print(f"PROMPT: {prompt!r}   EXPECTED ANSWER: {expected_answer}")
        print("=" * 70)

        # --- Load teacher supergraph ---
        if expected_answer is None:
            print("  WARNING: Could not parse expected answer — skipping circuit comparison.")
            continue

        key = (str(prompt), int(expected_answer))
        if key not in cache._samples:
            print(
                f"  WARNING: ({prompt!r}, {expected_answer}) not in teacher cache.\n"
                "  Run with --list-prompts to see available keys."
            )
            continue

        print("\n[1/4] Loading teacher supergraph from cache...")
        teacher_payload = cache.load_teacher_supergraph(prompt, expected_answer)
        teacher_payload = _normalize_supergraph_payload(teacher_payload)

        # --- Build student graph ---
        print(f"[2/4] Building student attribution graph (prop_neurons={args.prop_neurons_per_layer})...")
        with torch.no_grad():
            student_graph = adapter.build_graph(
                prompt,
                prop_neurons_per_layer=args.prop_neurons_per_layer,
                batch_size=1,
                dtype=dtype,
                verbose=False,
                create_graph=False,
                detach_result=True,
                fast=False,
                skip_logit_attribution=True,
            )

        # --- Build student supergraph ---
        print("[3/4] Building student supergraph (fixed_labels clustering)...")
        with torch.no_grad():
            student_supergraph_structure = build_super_graph(
                student_graph,
                adapter,
                cluster_method="fixed_labels",
                fixed_labels=fixed_labels_dict,
                activation_forward_batch_size=500,
                computation_eps=0.05,
                embedding_eps=0.1,
            )

        # Aggregate adjacency
        student_supergraph = _aggregate_supergraph_adjacency_local(
            student_graph,
            student_supergraph_structure.supernodes,
        )

        student_payload = _normalize_supergraph_payload(student_supergraph)
        # Carry over labels from the structure (not stored in the raw SuperGraph from aggregation)
        student_payload["supernode_labels"] = student_supergraph_structure.supernode_labels

        # --- Side-by-side comparison ---
        print("[4/4] Printing comparison...")
        _print_supernode_summary(
            f"TEACHER (from cache) — {prompt!r}",
            teacher_payload,
            top_k=args.top_k,
        )
        _print_supernode_summary(
            f"STUDENT (trained checkpoint) — {prompt!r}",
            student_payload,
            top_k=args.top_k,
        )

        # --- Focus label breakdown ---
        teacher_top_out = _focus_on_label(
            f"TEACHER — {prompt!r}", teacher_payload, args.focus_label, top_k=args.top_k
        )
        student_top_out = _focus_on_label(
            f"STUDENT — {prompt!r}", student_payload, args.focus_label, top_k=args.top_k
        )

        # --- Carry-node alignment check ---
        print(f"\n--- CARRY-NODE CHECK: '{args.focus_label}' dominant-edge match ---")
        t_labels = teacher_payload.get("supernode_labels")
        s_labels = student_payload.get("supernode_labels")
        if teacher_top_out and student_top_out and t_labels and s_labels:
            t_dominant_j, _, _ = teacher_top_out[0]
            s_dominant_j, _, _ = student_top_out[0]
            t_dominant_label = _format_label(t_labels, t_dominant_j)
            s_dominant_label = _format_label(s_labels, s_dominant_j)
            match = t_dominant_label == s_dominant_label
            status = "MATCH ✓" if match else "MISMATCH ✗"
            print(f"  Teacher dominant out: -> #{t_dominant_j} ({t_dominant_label})")
            print(f"  Student dominant out: -> #{s_dominant_j} ({s_dominant_label})")
            print(f"  {status}")
        else:
            print(f"  (skipped — '{args.focus_label}' not found in one or both models)")

        # --- JSD alignment score ---
        print(f"\n--- JSD ALIGNMENT SCORE (lower = more aligned) ---")
        label_jsd, mean_jsd = _compute_jsd_per_label_pair(
            teacher_payload, student_payload
        )
        if label_jsd:
            max_label_width = max(len(l) for l in label_jsd)
            for lbl, jsd_val in sorted(label_jsd.items(), key=lambda kv: -kv[1]):
                print(f"  {lbl:{max_label_width}s}  JSD = {jsd_val:.4f}")
            print(f"\n  MEAN JSD (alignment score): {mean_jsd:.4f}")
            all_mean_jsds.append(mean_jsd)
        else:
            print("  (no matched label pairs — cannot compute JSD)")

    # === Summary across all prompts ===
    if all_mean_jsds:
        overall_mean = sum(all_mean_jsds) / len(all_mean_jsds)
        print("\n" + "=" * 70)
        print(f"OVERALL MEAN JSD across {len(all_mean_jsds)} prompt(s): {overall_mean:.4f}")
        print("(lower = student circuit structure more aligned to teacher)")
        print("=" * 70)


if __name__ == "__main__":
    main()

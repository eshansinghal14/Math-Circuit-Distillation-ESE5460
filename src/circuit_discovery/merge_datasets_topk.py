"""
Merge per-neuron cossim rankings across multiple datasets, then export a sparse
binary circuit checkpoint with the top-k neurons.

The merge operates on **percentile ranks** per dataset rather than raw cossim
values so different datasets can be combined more robustly. Two aggregation
methods are supported:

``min``
    Strict intersection over dataset percentile ranks.

``geo_mean``
    Geometric mean of dataset percentile ranks.

Example::

    python -m circuit_discovery.merge_datasets_topk \
        --datasets 222_add 23_add 34_add \
        --method geo_mean \
        --name addition_shared \
        --frac-activated 0.01
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import torch

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _src not in sys.path:
    sys.path.insert(0, _src)

from circuit_discovery.neuron_cossim_topk import (
    K_CLASSES,
    _neuron_cossim_workspace_root,
    build_sparse_circuit_model,
    load_cossim_record,
    mean_cossim_topk_neurons,
    topk_binary_mask,
)


def _resolve_dataset_json(dataset_entry: str) -> str:
    """Resolve a dataset folder name or JSON path to a cossim JSON file."""
    entry = dataset_entry.strip()
    if not entry:
        raise ValueError("Dataset entries must be non-empty")

    if os.path.isfile(entry):
        if not entry.lower().endswith(".json"):
            raise ValueError(f"Expected a cossim JSON path, got file {entry!r}")
        return os.path.abspath(entry)

    base = entry
    if not os.path.isabs(base):
        base = os.path.join(_neuron_cossim_workspace_root(), base)
    base = os.path.abspath(base)
    if not os.path.isdir(base):
        raise FileNotFoundError(
            f"Dataset folder {dataset_entry!r} does not exist under "
            f"{_neuron_cossim_workspace_root()!r} and is not a valid path"
        )

    matches = sorted(glob.glob(os.path.join(base, "neuron_mean_pairwise_cossim*.json")))
    if not matches:
        raise FileNotFoundError(
            f"No neuron cossim JSON found in dataset folder {base!r}; "
            "expected a file matching 'neuron_mean_pairwise_cossim*.json'"
        )
    if len(matches) > 1:
        shown = ", ".join(os.path.basename(m) for m in matches[:5])
        raise ValueError(
            f"Dataset folder {base!r} contains multiple cossim JSON files: {shown}. "
            "Pass the JSON path directly or keep only one matching file in the folder."
        )
    return os.path.abspath(matches[0])


def _percentile_ranks(scores: Sequence[float]) -> torch.Tensor:
    """Return percentile ranks in ``[0, 1]`` with larger scores getting larger ranks."""
    t = torch.as_tensor(scores, dtype=torch.float64)
    if t.ndim != 1 or t.numel() == 0:
        raise ValueError("scores must be a non-empty 1D sequence")
    n = int(t.numel())
    if n == 1:
        return torch.ones(1, dtype=torch.float32)

    order = torch.argsort(t, stable=True)
    sorted_t = t[order]
    sorted_ranks = torch.empty(n, dtype=torch.float64)

    start = 0
    while start < n:
        end = start + 1
        value = float(sorted_t[start].item())
        while end < n and float(sorted_t[end].item()) == value:
            end += 1
        avg_rank = 0.5 * (start + end - 1)
        sorted_ranks[start:end] = avg_rank / float(n - 1)
        start = end

    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks.to(torch.float32)


def _aggregate_rank_lists(rank_lists: List[torch.Tensor], method: str) -> torch.Tensor:
    """Aggregate percentile ranks across datasets."""
    if not rank_lists:
        raise ValueError("Need at least one dataset to merge")
    stack = torch.stack(rank_lists, dim=0).float()
    if method == "min":
        return stack.min(dim=0).values
    if method == "geo_mean":
        eps = 1e-12
        return torch.exp(torch.log(stack.clamp_min(eps)).mean(dim=0))
    raise ValueError(f"Unsupported method {method!r}")


def _validate_and_collect_records(json_paths: Sequence[str]) -> List[Dict[str, Any]]:
    """Load records and verify they are compatible for merging."""
    records: List[Dict[str, Any]] = []
    expected_dim_1b: Optional[int] = None
    expected_dim_8b: Optional[int] = None
    expected_res_token: Optional[int] = None
    expected_traj_mode: Optional[str] = None

    for path in json_paths:
        record = load_cossim_record(path)
        c1 = record["1b"]["mean_pairwise_cossim"]
        c8 = record["8b"]["mean_pairwise_cossim"]
        d1 = len(c1)
        d8 = len(c8)
        if d1 == 0 or d8 == 0:
            raise ValueError(f"Cossim JSON {path!r} has an empty neuron score vector")

        res_token = record.get("res_token")
        traj_mode = record.get("trajectory_mode")
        if expected_dim_1b is None:
            expected_dim_1b = d1
            expected_dim_8b = d8
            expected_res_token = res_token
            expected_traj_mode = traj_mode
        else:
            if d1 != expected_dim_1b or d8 != expected_dim_8b:
                raise ValueError(
                    f"Incompatible neuron dimensions while merging {path!r}: "
                    f"got (1b={d1}, 8b={d8}), expected "
                    f"(1b={expected_dim_1b}, 8b={expected_dim_8b})"
                )
            if res_token != expected_res_token or traj_mode != expected_traj_mode:
                raise ValueError(
                    f"Incompatible trajectory setting while merging {path!r}: "
                    f"got res_token={res_token!r}, trajectory_mode={traj_mode!r}, expected "
                    f"res_token={expected_res_token!r}, trajectory_mode={expected_traj_mode!r}"
                )

        record["_source_json_path"] = os.path.abspath(path)
        records.append(record)

    return records


def _merge_scores(records: Sequence[Dict[str, Any]], method: str) -> Dict[str, torch.Tensor]:
    """Build merged percentile-rank scores for 1B and 8B towers."""
    ranks_1b = [_percentile_ranks(r["1b"]["mean_pairwise_cossim"]) for r in records]
    ranks_8b = [_percentile_ranks(r["8b"]["mean_pairwise_cossim"]) for r in records]
    return {
        "1b": _aggregate_rank_lists(ranks_1b, method),
        "8b": _aggregate_rank_lists(ranks_8b, method),
    }


def _output_root(name: str) -> str:
    """Resolve output folder under the cossim workspace root unless absolute."""
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("--name must be non-empty")
    if os.path.isabs(cleaned):
        return os.path.abspath(cleaned)
    return os.path.abspath(os.path.join(_neuron_cossim_workspace_root(), cleaned))


def run(
    datasets: Sequence[str],
    method: str,
    name: str,
    frac_activated: float,
) -> str:
    if len(datasets) < 1:
        raise ValueError("Need at least one dataset folder or JSON path")
    if not (0.0 < frac_activated <= 1.0):
        raise ValueError("frac_activated must be in (0, 1]")

    json_paths = [_resolve_dataset_json(ds) for ds in datasets]
    records = _validate_and_collect_records(json_paths)
    merged = _merge_scores(records, method)

    score_1b = merged["1b"]
    score_8b = merged["8b"]
    d1 = int(score_1b.numel())
    d8 = int(score_8b.numel())
    k1 = max(1, int(round(frac_activated * d1)))
    k8 = max(1, int(round(frac_activated * d8)))

    print(f"Merging {len(records)} datasets with method={method!r} in percentile-rank space")
    for rec in records:
        print(
            f"  - {rec.get('dataset_prefix', '<unknown>')} "
            f"({rec['_source_json_path']})"
        )
    print(f"Building binary masks: top-{k1}/{d1} (1b), top-{k8}/{d8} (8b)")

    score_1b_list = score_1b.tolist()
    score_8b_list = score_8b.tolist()
    mask_1b = topk_binary_mask(d1, score_1b_list, k1)
    mask_8b = topk_binary_mask(d8, score_8b_list, k8)

    avg_1b = mean_cossim_topk_neurons(score_1b_list, k1)
    avg_8b = mean_cossim_topk_neurons(score_8b_list, k8)
    print(
        "Mean merged percentile-rank score (activated / top-k neurons): "
        f"1b={avg_1b:.6f}, 8b={avg_8b:.6f}"
    )

    model = build_sparse_circuit_model(
        mask_1b=mask_1b.cpu(),
        mask_8b=mask_8b.cpu(),
        mask_temperature=1.0,
    )

    out_dir = os.path.join(_output_root(name), f"frac{frac_activated:g}")
    os.makedirs(out_dir, exist_ok=True)
    out_pt = os.path.join(
        out_dir,
        f"merged_topk_{method}_frac{frac_activated:g}_k1b{k1}_k8b{k8}.pt",
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    torch.save(
        {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics_log": [],
            "sparse_binary_metadata": {
                "k_classes": K_CLASSES,
                "merge_method": method,
                "merge_space": "percentile_rank",
                "datasets": [str(d) for d in datasets],
                "source_cossim_jsons": [r["_source_json_path"] for r in records],
                "source_dataset_prefixes": [r.get("dataset_prefix") for r in records],
                "trajectory_mode": records[0].get("trajectory_mode"),
                "res_token": records[0].get("res_token"),
                "frac_activated": frac_activated,
                "k_1b": k1,
                "k_8b": k8,
                "dim_1b": d1,
                "dim_8b": d8,
                "mask_1b": mask_1b.cpu().clone(),
                "mask_8b": mask_8b.cpu().clone(),
            },
        },
        out_pt,
    )
    print(f"Saved merged sparse binary circuit model to {out_pt}")
    print("  Load with: utils.load_model_checkpoint(path, k_classes=1, lr=1e-3)")
    return out_pt


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Merge neuron cossim JSONs from multiple dataset folders using percentile-rank "
            "aggregation, then export a sparse binary circuit checkpoint."
        ),
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        metavar="DATASET_DIR",
        help=(
            "Dataset folders under results/circuit-discovery (or direct JSON paths) containing "
            "a single neuron_mean_pairwise_cossim*.json to merge"
        ),
    )
    p.add_argument(
        "--method",
        type=str,
        choices=("min", "geo_mean"),
        required=True,
        help="Dataset-merge rule in percentile-rank space",
    )
    p.add_argument(
        "--name",
        type=str,
        required=True,
        help=(
            "Output folder for the merged sparse .pt checkpoint; relative paths are placed under "
            "results/circuit-discovery/"
        ),
    )
    p.add_argument(
        "--frac-activated",
        type=float,
        required=True,
        help="Fraction of neurons to keep (top-k by merged score per tower)",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    run(
        datasets=args.datasets,
        method=args.method,
        name=args.name,
        frac_activated=args.frac_activated,
    )


if __name__ == "__main__":
    main()

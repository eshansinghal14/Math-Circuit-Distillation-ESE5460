"""ANOVA-style labels for neuron activation heatmaps."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class BasisRule:
    label: str
    mask: torch.Tensor
    category: str


@dataclass(frozen=True)
class NodeLabel:
    labels: list[str]
    scores: dict[str, float]
    categories: dict[str, str] = field(default_factory=dict)
    category_scores: dict[str, float] = field(default_factory=dict)
    category_specificity: dict[str, float] = field(default_factory=dict)


ANOVA_LABEL_CATEGORIES = [
    "arg1 range",
    "arg1 units",
    "arg2 range",
    "arg2 units",
    "arg1 units and arg2 units",
    "arg1 range and arg2 range",
    "carry",
    "sum range",
    "sum units",
]

BASE_ANOVA_LABEL_CATEGORIES = [
    "arg1 range",
    "arg1 units",
    "arg2 range",
    "arg2 units",
    "carry",
    "sum range",
    "sum units",
]

CATEGORY_COMPONENTS = {
    "arg1 units and arg2 units": {"arg1 units", "arg2 units"},
}


def _axis_values_grid(arg_values: list[list[int]], dim: int) -> torch.Tensor:
    shape = [1] * len(arg_values)
    shape[dim] = len(arg_values[dim])
    values = torch.tensor(arg_values[dim], dtype=torch.long).reshape(shape)
    return values.expand(*(len(values_for_dim) for values_for_dim in arg_values))


def _format_interval_label(prefix: str, lo: int, hi: int) -> str:
    return f"{prefix} {lo}-{hi}"


def _mask_interval_label(prefix: str, values: torch.Tensor, mask: torch.Tensor) -> str:
    selected = values[mask]
    if selected.numel() == 0:
        return _format_interval_label(prefix, 0, 0)
    return _format_interval_label(
        prefix,
        int(selected.min().item()),
        int(selected.max().item()),
    )


def _joint_range_label(
    arg1_values: torch.Tensor,
    arg2_values: torch.Tensor,
    mask: torch.Tensor,
) -> str:
    arg1_label = _mask_interval_label("arg1", arg1_values, mask)
    arg2_label = _mask_interval_label("arg2", arg2_values, mask)
    return f"{arg1_label} and {arg2_label}"


def parse_numeric_args(prompt: str) -> tuple[int, ...]:
    """Parse numeric arguments from the left side of an arithmetic prompt."""
    left = prompt.split("=", 1)[0]
    values = re.findall(r"-?\d+", left)
    if not values:
        raise ValueError(f"No numeric arguments found in prompt {prompt!r}")
    return tuple(int(value) for value in values)


def build_anova_basis_rules(
    arg_values: list[list[int]],
    *,
    target_args: tuple[int, ...] | None = None,
    anova_range_radius: int = 0,
) -> list[BasisRule]:
    """Build one binary basis mask per category."""
    if anova_range_radius < 0:
        raise ValueError("anova_range_radius must be non-negative")
    if not arg_values:
        return []

    grids = [_axis_values_grid(arg_values, dim) for dim in range(len(arg_values))]
    rules: list[BasisRule] = []

    for dim in range(min(2, len(arg_values))):
        arg_name = f"arg{dim + 1}"
        values = grids[dim]
        centers = (
            [int(target_args[dim])]
            if target_args is not None and dim < len(target_args)
            else [int(value) for value in arg_values[dim]]
        )
        unit_digits = (
            [int(target_args[dim]) % 10]
            if target_args is not None and dim < len(target_args)
            else list(range(10))
        )
        for center in centers:
            if anova_range_radius:
                mask = (values >= center - anova_range_radius) & (
                    values <= center + anova_range_radius
                )
                label = _mask_interval_label(arg_name, values, mask)
            else:
                mask = values == center
                label = _format_interval_label(arg_name, center, center)
            rules.append(BasisRule(label, mask, category=f"{arg_name} range"))
        for unit_digit in unit_digits:
            mask = torch.remainder(values, 10) == unit_digit
            if bool(mask.any().item()):
                rules.append(BasisRule(f"{arg_name} units {unit_digit}", mask, category=f"{arg_name} units"))

    if target_args is not None and len(target_args) >= 2 and len(arg_values) >= 2:
        arg1_values = grids[0]
        arg2_values = grids[1]
        arg1_center = int(target_args[0])
        arg2_center = int(target_args[1])
        if anova_range_radius:
            arg1_box = (arg1_values >= arg1_center - anova_range_radius) & (
                arg1_values <= arg1_center + anova_range_radius
            )
            arg2_box = (arg2_values >= arg2_center - anova_range_radius) & (
                arg2_values <= arg2_center + anova_range_radius
            )
        else:
            arg1_box = arg1_values == arg1_center
            arg2_box = arg2_values == arg2_center
        mask = arg1_box & arg2_box
        rules.append(BasisRule(
            _joint_range_label(arg1_values, arg2_values, mask),
            mask,
            category="arg1 range and arg2 range",
        ))

    if len(arg_values) >= 2:
        sums = grids[0] + grids[1]
        sum_centers = (
            [int(target_args[0] + target_args[1])]
            if target_args is not None and len(target_args) >= 2
            else sorted({int(value) for value in sums.flatten().tolist()})
        )
        for center in sum_centers:
            if anova_range_radius:
                mask = (sums >= center - anova_range_radius) & (
                    sums <= center + anova_range_radius
                )
                label = _mask_interval_label("sum", sums, mask)
            else:
                mask = sums == center
                label = _format_interval_label("sum", center, center)
            rules.append(BasisRule(label, mask, category="sum range"))
        sum_unit_digits = (
            [int(target_args[0] + target_args[1]) % 10]
            if target_args is not None and len(target_args) >= 2
            else list(range(10))
        )
        for unit_digit in sum_unit_digits:
            mask = torch.remainder(sums, 10) == unit_digit
            if bool(mask.any().item()):
                rules.append(BasisRule(f"sum units {unit_digit}", mask, category="sum units"))

        arg1_units = torch.remainder(grids[0], 10)
        arg2_units = torch.remainder(grids[1], 10)
        carry_mask = (arg1_units + arg2_units) >= 10
        if bool(carry_mask.any().item()) and bool((~carry_mask).any().item()):
            rules.append(BasisRule(label="carry", mask=carry_mask, category="carry"))

    return rules


def explained_variance_score(activation_grid: torch.Tensor, mask: torch.Tensor) -> float:
    """Return variance explained by the centered binary mask projection."""
    if activation_grid.shape != mask.shape:
        raise ValueError(
            f"Activation grid shape {tuple(activation_grid.shape)} does not match "
            f"basis mask shape {tuple(mask.shape)}"
        )

    activations = activation_grid.detach().float().flatten()
    basis = mask.detach().float().flatten()
    valid = ~torch.isnan(activations)
    if int(valid.sum().item()) < 2:
        return 0.0

    y = activations[valid]
    x = basis[valid]
    y_centered = y - y.mean()
    x_centered = x - x.mean()
    total_variance = y_centered.square().sum()
    basis_variance = x_centered.square().sum()
    if float(total_variance.item()) <= 0.0 or float(basis_variance.item()) <= 0.0:
        return 0.0

    projection = torch.dot(y_centered, x_centered)
    score = projection.square() / (total_variance * basis_variance)
    return float(score.clamp(min=0.0, max=1.0).item())


def _batch_explained_variance_scores(
    acts_flat: torch.Tensor,
    masks_flat: torch.Tensor,
) -> torch.Tensor:
    """Return [N, R] explained-variance scores via a single batched matmul.

    acts_flat:  [N, M]  — pre-flattened float activations (no NaNs)
    masks_flat: [R, M]  — pre-flattened float basis masks (one per rule)
    """
    y_c = acts_flat - acts_flat.mean(dim=-1, keepdim=True)    # [N, M]
    X_c = masks_flat - masks_flat.mean(dim=-1, keepdim=True)   # [R, M]
    total_var = y_c.square().sum(dim=-1, keepdim=True)         # [N, 1]
    basis_var = X_c.square().sum(dim=-1).unsqueeze(0)          # [1, R]
    proj = y_c @ X_c.T                                         # [N, R]
    denom = total_var * basis_var                              # [N, R]
    return torch.where(denom > 0, proj.square() / denom, torch.zeros_like(proj)).clamp(0, 1)


def build_gpu_anova_state(
    rules: list[BasisRule],
    device: torch.device,
) -> dict:
    """Pre-allocate GPU tensors for the ANOVA scoring pipeline.
    Call once before the batch loop; pass the result to gpu_label_activation_heatmaps.
    """
    masks_flat = torch.stack(
        [r.mask.detach().float().flatten() for r in rules], dim=0
    ).to(device)  # [R, M]
    return {"masks_flat_gpu": masks_flat}


def _make_node_labels(scores_cpu: torch.Tensor, rules: list[BasisRule]) -> list[NodeLabel]:
    """Convert [N, R] CPU scores into NodeLabels using rule.label directly."""
    N = int(scores_cpu.shape[0])
    out: list[NodeLabel] = []
    for n in range(N):
        category_scores_n: dict[str, float] = {}
        category_labels: dict[str, str] = {}
        for r, rule in enumerate(rules):
            s = float(scores_cpu[n, r].item())
            if s > category_scores_n.get(rule.category, float("-inf")):
                category_scores_n[rule.category] = s
                category_labels[rule.category] = rule.label

        if "arg1 units" in category_scores_n and "arg2 units" in category_scores_n:
            combo = "arg1 units and arg2 units"
            category_scores_n[combo] = min(
                category_scores_n["arg1 units"], category_scores_n["arg2 units"]
            )
            category_labels[combo] = (
                f"{category_labels['arg1 units']} and {category_labels['arg2 units']}"
            )

        category_specificity: dict[str, float] = {}
        for category, target_score in category_scores_n.items():
            excluded = {category} | CATEGORY_COMPONENTS.get(category, set())
            competitors = [
                category_scores_n[c]
                for c in BASE_ANOVA_LABEL_CATEGORIES
                if c not in excluded and c in category_scores_n
            ]
            category_specificity[category] = target_score - (max(competitors) if competitors else 0.0)

        labels = [category_labels[c] for c in ANOVA_LABEL_CATEGORIES if c in category_labels]
        scores = {
            category_labels[c]: category_scores_n[c]
            for c in ANOVA_LABEL_CATEGORIES
            if c in category_labels
        }
        out.append(NodeLabel(
            labels=labels,
            scores=scores,
            categories=category_labels,
            category_scores=category_scores_n,
            category_specificity=category_specificity,
        ))
    return out


def gpu_label_activation_heatmaps(
    acts_flat_gpu: torch.Tensor,
    gpu_state: dict,
    rules: list[BasisRule],
) -> list[NodeLabel]:
    """Score and label N neuron activation heatmaps with GPU-resident tensors.

    acts_flat_gpu: [N, M] on GPU — flattened activation grids, may contain NaN.
    gpu_state:     dict from build_gpu_anova_state.
    """
    N = int(acts_flat_gpu.shape[0])
    if not rules:
        empty = NodeLabel(labels=[], scores={}, categories={}, category_scores={}, category_specificity={})
        return [empty] * N

    masks_flat_gpu = gpu_state["masks_flat_gpu"]
    valid_mask = ~torch.isnan(acts_flat_gpu[0])
    acts_v = acts_flat_gpu[:, valid_mask].float()
    masks_v = masks_flat_gpu[:, valid_mask]

    scores_cpu = _batch_explained_variance_scores(acts_v, masks_v).cpu()
    return _make_node_labels(scores_cpu, rules)



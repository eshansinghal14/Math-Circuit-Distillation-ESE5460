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
    range_values: torch.Tensor | None = None
    range_center: int | None = None
    range_prefix: str | None = None
    dynamic_range_label: bool = True


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
    """Build binary basis masks for argument and sum rules.

    Arg range labels can be scored with a target-centered radius. Sum ranges
    are still scored against the exact sum target.
    """
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
                dynamic_range_label = False
            else:
                mask = values == center
                label = _format_interval_label(arg_name, center, center)
                dynamic_range_label = True
            rules.append(
                BasisRule(
                    label,
                    mask,
                    category=f"{arg_name} range",
                    range_values=values,
                    range_center=center,
                    range_prefix=arg_name,
                    dynamic_range_label=dynamic_range_label,
                )
            )
        for unit_digit in unit_digits:
            mask = torch.remainder(values, 10) == unit_digit
            if bool(mask.any().item()):
                rules.append(
                    BasisRule(
                        f"{arg_name} units {unit_digit}",
                        mask,
                        category=f"{arg_name} units",
                    )
                )

    if target_args is not None and len(target_args) >= 2 and len(arg_values) >= 2:
        arg1_values = grids[0]
        arg2_values = grids[1]
        arg1_center = int(target_args[0])
        arg2_center = int(target_args[1])
        arg1_mask = (arg1_values >= arg1_center - anova_range_radius) & (
            arg1_values <= arg1_center + anova_range_radius
        )
        arg2_mask = (arg2_values >= arg2_center - anova_range_radius) & (
            arg2_values <= arg2_center + anova_range_radius
        )
        mask = arg1_mask & arg2_mask
        rules.append(
            BasisRule(
                _joint_range_label(arg1_values, arg2_values, mask),
                mask,
                category="arg1 range and arg2 range",
                dynamic_range_label=False,
            )
        )

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
                dynamic_range_label = False
            else:
                mask = sums == center
                label = _format_interval_label("sum", center, center)
                dynamic_range_label = True
            rules.append(
                BasisRule(
                    label,
                    mask,
                    category="sum range",
                    range_values=sums,
                    range_center=center,
                    range_prefix="sum",
                    dynamic_range_label=dynamic_range_label,
                )
            )
        sum_unit_digits = (
            [int(target_args[0] + target_args[1]) % 10]
            if target_args is not None and len(target_args) >= 2
            else list(range(10))
        )
        for unit_digit in sum_unit_digits:
            mask = torch.remainder(sums, 10) == unit_digit
            if bool(mask.any().item()):
                rules.append(BasisRule(f"sum units {unit_digit}", mask, category="sum units"))

        # Carry detection: (arg1 % 10) + (arg2 % 10) >= 10.  Distinguishes
        # neurons that fire when the addition produces a carry from neurons
        # that track units/decades.  Critical for addition circuit analysis.
        arg1_units = torch.remainder(grids[0], 10)
        arg2_units = torch.remainder(grids[1], 10)
        carry_mask = (arg1_units + arg2_units) >= 10
        if bool(carry_mask.any().item()) and bool((~carry_mask).any().item()):
            rules.append(
                BasisRule(
                    label="carry",
                    mask=carry_mask,
                    category="carry",
                )
            )

    return rules


def _abs_zscore_grid(activation_grid: torch.Tensor) -> torch.Tensor:
    grid = activation_grid.detach().float()
    valid_values = grid[~torch.isnan(grid)]
    if valid_values.numel() == 0:
        return grid

    std = valid_values.std(unbiased=False)
    if float(std.item()) == 0.0:
        out = grid.clone()
        out[~torch.isnan(out)] = 0.0
        return out

    mean = valid_values.mean()
    return (grid - mean).abs() / std.clamp(min=1e-6)


def _high_activation_range_label(
    activation_grid: torch.Tensor,
    rule: BasisRule,
    *,
    z_threshold: float = 1.0,
) -> str:
    if (
        not rule.dynamic_range_label
        or
        rule.range_values is None
        or rule.range_center is None
        or rule.range_prefix is None
    ):
        return rule.label

    zscores = _abs_zscore_grid(activation_grid)
    values_grid = rule.range_values.to(dtype=torch.long)
    valid = ~torch.isnan(zscores)
    candidate_values = sorted(
        {
            int(value)
            for value in values_grid[valid].flatten().tolist()
        }
    )
    if int(rule.range_center) not in candidate_values:
        return rule.label

    value_scores: dict[int, float] = {}
    for value in candidate_values:
        value_mask = valid & (values_grid == value)
        if bool(value_mask.any().item()):
            value_scores[value] = float(zscores[value_mask].mean().item())
        else:
            value_scores[value] = float("-inf")

    center_idx = candidate_values.index(int(rule.range_center))
    left_idx = center_idx
    right_idx = center_idx
    while left_idx > 0 and value_scores[candidate_values[left_idx - 1]] >= z_threshold:
        left_idx -= 1
    while (
        right_idx + 1 < len(candidate_values)
        and value_scores[candidate_values[right_idx + 1]] >= z_threshold
    ):
        right_idx += 1

    return _format_interval_label(
        rule.range_prefix,
        candidate_values[left_idx],
        candidate_values[right_idx],
    )


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


def label_activation_heatmaps(
    activations: torch.Tensor,
    arg_values: list[list[int]],
    *,
    target_args: tuple[int, ...] | None = None,
    anova_range_radius: int = 0,
) -> list[NodeLabel]:
    """Score ANOVA label categories for each activation heatmap."""
    if activations.ndim != len(arg_values) + 1:
        raise ValueError(
            f"Expected activations with {len(arg_values) + 1} dims for arg_values, "
            f"got shape {tuple(activations.shape)}"
        )

    rules = build_anova_basis_rules(
        arg_values,
        target_args=target_args,
        anova_range_radius=anova_range_radius,
    )
    out: list[NodeLabel] = []
    for activation_grid in activations.detach().float().cpu():
        category_labels: dict[str, str] = {}
        category_scores: dict[str, float] = {}
        for rule in rules:
            score = explained_variance_score(activation_grid, rule.mask)
            label = _high_activation_range_label(
                activation_grid,
                rule,
            )
            if score > category_scores.get(rule.category, float("-inf")):
                category_scores[rule.category] = score
                category_labels[rule.category] = label

        if "arg1 units" in category_scores and "arg2 units" in category_scores:
            category = "arg1 units and arg2 units"
            category_scores[category] = min(
                category_scores["arg1 units"],
                category_scores["arg2 units"],
            )
            category_labels[category] = (
                f"{category_labels['arg1 units']} and {category_labels['arg2 units']}"
            )
        if (
            "arg1 range and arg2 range" not in category_scores
            and "arg1 range" in category_scores
            and "arg2 range" in category_scores
        ):
            category = "arg1 range and arg2 range"
            category_scores[category] = min(
                category_scores["arg1 range"],
                category_scores["arg2 range"],
            )
            category_labels[category] = (
                f"{category_labels['arg1 range']} and {category_labels['arg2 range']}"
            )

        category_specificity: dict[str, float] = {}
        for category, target_score in category_scores.items():
            excluded_categories = {category} | CATEGORY_COMPONENTS.get(category, set())
            competitor_scores = [
                category_scores[competitor]
                for competitor in BASE_ANOVA_LABEL_CATEGORIES
                if competitor not in excluded_categories and competitor in category_scores
            ]
            max_competitor_score = max(competitor_scores) if competitor_scores else 0.0
            category_specificity[category] = target_score - max_competitor_score

        labels = [
            category_labels[category]
            for category in ANOVA_LABEL_CATEGORIES
            if category in category_labels
        ]
        scores = {
            category_labels[category]: category_scores[category]
            for category in ANOVA_LABEL_CATEGORIES
            if category in category_labels
        }
        out.append(
            NodeLabel(
                labels=labels,
                scores=scores,
                categories=category_labels,
                category_scores=category_scores,
                category_specificity=category_specificity,
            )
        )
    return out

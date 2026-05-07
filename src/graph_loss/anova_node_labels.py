"""ANOVA-style labels for neuron activation heatmaps."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BasisRule:
    label: str
    mask: torch.Tensor


@dataclass(frozen=True)
class NodeLabel:
    labels: list[str]
    scores: dict[str, float]


def _axis_values_grid(arg_values: list[list[int]], dim: int) -> torch.Tensor:
    shape = [1] * len(arg_values)
    shape[dim] = len(arg_values[dim])
    values = torch.tensor(arg_values[dim], dtype=torch.long).reshape(shape)
    return values.expand(*(len(values_for_dim) for values_for_dim in arg_values))


def _format_range_label(prefix: str, center: int, radius: int) -> str:
    return f"{prefix} {center - radius}-{center + radius}"


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
    range_radius: int = 10,
) -> list[BasisRule]:
    """Build binary basis masks for argument and sum rules."""
    if not arg_values:
        return []
    if range_radius < 0:
        raise ValueError("range_radius must be non-negative")

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
            mask = (values >= center - range_radius) & (values <= center + range_radius)
            rules.append(BasisRule(_format_range_label(arg_name, center, range_radius), mask))
        for unit_digit in unit_digits:
            mask = torch.remainder(values, 10) == unit_digit
            if bool(mask.any().item()):
                rules.append(BasisRule(f"{arg_name} units {unit_digit}", mask))

    if len(arg_values) >= 2:
        sums = grids[0] + grids[1]
        sum_centers = (
            [int(target_args[0] + target_args[1])]
            if target_args is not None and len(target_args) >= 2
            else sorted({int(value) for value in sums.flatten().tolist()})
        )
        for center in sum_centers:
            mask = (sums >= center - range_radius) & (sums <= center + range_radius)
            rules.append(BasisRule(_format_range_label("sum", center, range_radius), mask))
        sum_unit_digits = (
            [int(target_args[0] + target_args[1]) % 10]
            if target_args is not None and len(target_args) >= 2
            else list(range(10))
        )
        for unit_digit in sum_unit_digits:
            mask = torch.remainder(sums, 10) == unit_digit
            if bool(mask.any().item()):
                rules.append(BasisRule(f"sum units {unit_digit}", mask))

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


def label_activation_heatmaps(
    activations: torch.Tensor,
    arg_values: list[list[int]],
    *,
    threshold: float,
    target_args: tuple[int, ...] | None = None,
    range_radius: int = 10,
) -> list[NodeLabel]:
    """Assign all basis labels whose explained-variance score reaches threshold."""
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be in [0, 1]")
    if activations.ndim != len(arg_values) + 1:
        raise ValueError(
            f"Expected activations with {len(arg_values) + 1} dims for arg_values, "
            f"got shape {tuple(activations.shape)}"
        )

    rules = build_anova_basis_rules(
        arg_values,
        target_args=target_args,
        range_radius=range_radius,
    )
    out: list[NodeLabel] = []
    for activation_grid in activations.detach().float().cpu():
        scores = {
            rule.label: explained_variance_score(activation_grid, rule.mask)
            for rule in rules
        }
        labels = [
            label
            for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
            if score >= threshold
        ]
        out.append(NodeLabel(labels=labels, scores={label: scores[label] for label in labels}))
    return out

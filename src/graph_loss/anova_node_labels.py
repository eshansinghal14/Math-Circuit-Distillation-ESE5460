"""ANOVA-style labels for neuron activation heatmaps.

Activation heatmaps are N-D where N = number of args in the prompt/dataset.
2-arg prompts → 2D heatmaps; 3-arg prompts → 3D heatmaps.
The grid dimensionality comes from the MLP input cache (dataset), so the
dataset passed via --dataset must have the same arg count as the prompt.
"""

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
    return values.expand(*(len(v) for v in arg_values))


def _format_interval_label(prefix: str, lo: int, hi: int) -> str:
    return f"{prefix} {lo}-{hi}"


def _mask_interval_label(prefix: str, values: torch.Tensor, mask: torch.Tensor) -> str:
    selected = values[mask]
    if selected.numel() == 0:
        return _format_interval_label(prefix, 0, 0)
    return _format_interval_label(prefix, int(selected.min().item()), int(selected.max().item()))


def _joint_range_label(arg1_values: torch.Tensor, arg2_values: torch.Tensor, mask: torch.Tensor) -> str:
    return f"{_mask_interval_label('arg1', arg1_values, mask)} and {_mask_interval_label('arg2', arg2_values, mask)}"


def parse_numeric_args(prompt: str) -> tuple[int, ...]:
    """Parse numeric arguments from the left side of an arithmetic prompt."""
    left = prompt.split("=", 1)[0]
    values = re.findall(r"-?\d+", left)
    if not values:
        raise ValueError(f"No numeric arguments found in prompt {prompt!r}")
    return tuple(int(v) for v in values)


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

    for dim in range(len(arg_values)):
        arg_name = f"arg{dim + 1}"
        values = grids[dim]
        centers = (
            [int(target_args[dim])]
            if target_args is not None and dim < len(target_args)
            else [int(v) for v in arg_values[dim]]
        )
        unit_digits = (
            [int(target_args[dim]) % 10]
            if target_args is not None and dim < len(target_args)
            else list(range(10))
        )
        for center in centers:
            if anova_range_radius:
                mask = (values >= center - anova_range_radius) & (values <= center + anova_range_radius)
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
        arg1_values, arg2_values = grids[0], grids[1]
        arg1_c, arg2_c = int(target_args[0]), int(target_args[1])
        if anova_range_radius:
            arg1_box = (arg1_values >= arg1_c - anova_range_radius) & (arg1_values <= arg1_c + anova_range_radius)
            arg2_box = (arg2_values >= arg2_c - anova_range_radius) & (arg2_values <= arg2_c + anova_range_radius)
        else:
            arg1_box = arg1_values == arg1_c
            arg2_box = arg2_values == arg2_c
        mask = arg1_box & arg2_box
        rules.append(BasisRule(_joint_range_label(arg1_values, arg2_values, mask), mask, category="arg1 range and arg2 range"))

    # Pairwise sum ranges: heatmap-scored, specificity-ranked (not DLA).
    for _i in range(len(arg_values)):
        for _j in range(_i + 1, len(arg_values)):
            pair_sums = grids[_i] + grids[_j]
            pair_category = f"arg{_i + 1} arg{_j + 1} sum range"
            pair_prefix = f"arg{_i + 1}+arg{_j + 1}"
            if target_args is not None and len(target_args) > _j:
                center = int(target_args[_i] + target_args[_j])
                if anova_range_radius:
                    mask = (pair_sums >= center - anova_range_radius) & (pair_sums <= center + anova_range_radius)
                    label = _mask_interval_label(pair_prefix, pair_sums, mask)
                else:
                    mask = pair_sums == center
                    label = _format_interval_label(pair_prefix, center, center)
                rules.append(BasisRule(label, mask, category=pair_category))
            else:
                for center in sorted({int(v) for v in pair_sums.flatten().tolist()}):
                    if anova_range_radius:
                        mask = (pair_sums >= center - anova_range_radius) & (pair_sums <= center + anova_range_radius)
                        label = _mask_interval_label(pair_prefix, pair_sums, mask)
                    else:
                        mask = pair_sums == center
                        label = _format_interval_label(pair_prefix, center, center)
                    rules.append(BasisRule(label, mask, category=pair_category))

    if len(arg_values) >= 2:
        sums = grids[0] + grids[1]
        sum_centers = (
            [int(target_args[0] + target_args[1])]
            if target_args is not None and len(target_args) >= 2
            else sorted({int(v) for v in sums.flatten().tolist()})
        )
        for center in sum_centers:
            if anova_range_radius:
                mask = (sums >= center - anova_range_radius) & (sums <= center + anova_range_radius)
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
    y_c = y - y.mean()
    x_c = x - x.mean()
    total_var = y_c.square().sum()
    basis_var = x_c.square().sum()
    if float(total_var.item()) <= 0.0 or float(basis_var.item()) <= 0.0:
        return 0.0
    return float((torch.dot(y_c, x_c).square() / (total_var * basis_var)).clamp(0.0, 1.0).item())


def _batch_explained_variance_scores(acts_flat: torch.Tensor, masks_flat: torch.Tensor) -> torch.Tensor:
    """Return [N, R] explained-variance scores via a single batched matmul.

    acts_flat:  [N, M]  — pre-flattened float activations (no NaNs)
    masks_flat: [R, M]  — pre-flattened float basis masks (one per rule)
    """
    y_c = acts_flat - acts_flat.mean(dim=-1, keepdim=True)    # [N, M]
    X_c = masks_flat - masks_flat.mean(dim=-1, keepdim=True)   # [R, M]
    total_var = y_c.square().sum(dim=-1, keepdim=True)         # [N, 1]
    basis_var = X_c.square().sum(dim=-1).unsqueeze(0)          # [1, R]
    proj = y_c @ X_c.T                                         # [N, R]
    denom = total_var * basis_var
    return torch.where(denom > 0, proj.square() / denom, torch.zeros_like(proj)).clamp(0, 1)


def build_gpu_anova_state(rules: list[BasisRule], device: torch.device) -> dict:
    """Pre-allocate GPU tensors and precompute category mappings for the ANOVA pipeline.

    Derives the category list dynamically from the supplied rules so that datasets
    with more than 2 numeric arguments (e.g. "arg3 range", "arg3 units") work without
    any changes to the fixed ANOVA_LABEL_CATEGORIES constant.

    Call once before the batch loop; pass the result to gpu_label_activation_heatmaps.
    """
    # Build an ordered category list: known categories first (preserving their order),
    # then any extra categories introduced by rules (e.g. "arg3 range").
    _known = set(ANOVA_LABEL_CATEGORIES)
    _seen_extra: dict[str, None] = {}  # ordered-set via dict
    for r in rules:
        if r.category not in _known:
            _seen_extra[r.category] = None
    all_categories: list[str] = list(ANOVA_LABEL_CATEGORIES) + list(_seen_extra)

    # Base categories: original base list + the same extra dynamic categories
    # (dynamic arg-range/unit categories are base-level for specificity purposes).
    base_categories: list[str] = list(BASE_ANOVA_LABEL_CATEGORIES) + list(_seen_extra)

    C = len(all_categories)
    B = len(base_categories)
    cat_to_idx = {c: i for i, c in enumerate(all_categories)}

    masks_flat = torch.stack([r.mask.detach().float().flatten() for r in rules]).to(device)  # [R, M]

    # [R] — which all_categories column each rule belongs to
    rule_cat_ids = torch.tensor(
        [cat_to_idx[r.category] for r in rules], dtype=torch.long, device=device
    )

    # [B] — indices into all_categories for the B base categories
    base_to_all = torch.tensor(
        [cat_to_idx[c] for c in base_categories], dtype=torch.long, device=device
    )

    # Label string per category slot (empty string = category not present in these rules)
    cat_label_strs: list[str] = [""] * C
    for rule in rules:
        cat_label_strs[cat_to_idx[rule.category]] = rule.label

    # Composite "arg1 units and arg2 units" (only valid for 2-arg problems)
    arg1u_idx = cat_to_idx.get("arg1 units", -1)
    arg2u_idx = cat_to_idx.get("arg2 units", -1)
    combo_idx = cat_to_idx.get("arg1 units and arg2 units", -1)
    has_combo = bool(
        arg1u_idx >= 0 and arg2u_idx >= 0 and combo_idx >= 0
        and cat_label_strs[arg1u_idx] and cat_label_strs[arg2u_idx]
    )
    if has_combo:
        cat_label_strs[combo_idx] = f"{cat_label_strs[arg1u_idx]} and {cat_label_strs[arg2u_idx]}"

    # Base-category columns that are competitors for the combo specificity
    combo_excluded = CATEGORY_COMPONENTS.get("arg1 units and arg2 units", set())
    combo_competitor_cols = torch.tensor(
        [i for i, c in enumerate(base_categories) if c not in combo_excluded],
        dtype=torch.long, device=device,
    )

    return {
        "masks_flat_gpu": masks_flat,
        "rule_cat_ids": rule_cat_ids,
        "base_to_all": base_to_all,
        "cat_label_strs": cat_label_strs,
        "C": C, "B": B,
        "arg1u_idx": arg1u_idx, "arg2u_idx": arg2u_idx, "combo_idx": combo_idx,
        "has_combo": has_combo,
        "combo_competitor_cols": combo_competitor_cols,
        "all_categories": all_categories,
        "base_categories": base_categories,
    }


def gpu_label_activation_heatmaps(
    acts_flat_gpu: torch.Tensor,
    gpu_state: dict,
    rules: list[BasisRule],
) -> list[NodeLabel]:
    """Score and label N neuron activation heatmaps entirely on GPU.

    acts_flat_gpu: [N, M] on GPU — flattened activation grids, may contain NaN.
    gpu_state:     dict from build_gpu_anova_state.

    All scoring (explained variance, scatter-max, specificity) runs on GPU.
    One .tolist() call moves the results to CPU; the final loop is pure dict
    construction with no tensor operations inside it.
    """
    N = int(acts_flat_gpu.shape[0])
    if not rules:
        empty = NodeLabel(labels=[], scores={}, categories={}, category_scores={}, category_specificity={})
        return [empty] * N

    masks_flat_gpu   = gpu_state["masks_flat_gpu"]
    rule_cat_ids     = gpu_state["rule_cat_ids"]
    base_to_all      = gpu_state["base_to_all"]
    cat_label_strs   = gpu_state["cat_label_strs"]
    C                = gpu_state["C"]
    B                = gpu_state["B"]
    arg1u_idx        = gpu_state["arg1u_idx"]
    arg2u_idx        = gpu_state["arg2u_idx"]
    combo_idx        = gpu_state["combo_idx"]
    has_combo        = gpu_state["has_combo"]
    combo_cols       = gpu_state["combo_competitor_cols"]
    all_categories   = gpu_state["all_categories"]
    base_categories  = gpu_state["base_categories"]
    dev              = acts_flat_gpu.device

    # --- GPU: explained variance [N, R] ---
    valid_mask = ~torch.isnan(acts_flat_gpu[0])
    acts_v  = acts_flat_gpu[:, valid_mask].float()
    masks_v = masks_flat_gpu[:, valid_mask]
    scores  = _batch_explained_variance_scores(acts_v, masks_v)  # [N, R]

    # --- GPU: scatter-max rules → categories [N, C] ---
    cat_scores = torch.zeros(N, C, device=dev)
    cat_scores.scatter_reduce_(
        1, rule_cat_ids.unsqueeze(0).expand(N, -1), scores, reduce="amax", include_self=True
    )

    # --- GPU: composite category score ---
    if has_combo:
        cat_scores[:, combo_idx] = torch.minimum(cat_scores[:, arg1u_idx], cat_scores[:, arg2u_idx])

    # --- GPU: specificity for base categories [N, B] ---
    # specificity[n, b] = cat_scores[n, b] - max(cat_scores[n, other base cats])
    base_scores          = cat_scores[:, base_to_all]                           # [N, B]
    top1_vals, top1_idx  = base_scores.max(dim=1)                               # [N]
    second               = base_scores.clone().scatter_(1, top1_idx.unsqueeze(1), float("-inf"))
    top2_vals            = second.max(dim=1).values.clamp(min=0.0)              # [N]
    is_top1              = torch.arange(B, device=dev) == top1_idx.unsqueeze(1) # [N, B]
    competitor_max       = torch.where(is_top1, top2_vals.unsqueeze(1), top1_vals.unsqueeze(1))
    base_specificity     = base_scores - competitor_max                         # [N, B]

    # --- GPU: combo specificity ---
    if has_combo:
        combo_competitor_max = (
            base_scores[:, combo_cols].max(dim=1).values.clamp(min=0.0)
            if combo_cols.numel() > 0 else torch.zeros(N, device=dev)
        )
        combo_specificity = cat_scores[:, combo_idx] - combo_competitor_max     # [N]

    # --- Single D2H transfer ---
    cs = cat_scores.tolist()       # [N][C]
    bs = base_specificity.tolist() # [N][B]
    combo_s = combo_specificity.tolist() if has_combo else None  # [N]

    # --- Dict construction: no tensor ops below this line ---
    out: list[NodeLabel] = []
    for n in range(N):
        row_cs = cs[n]
        row_bs = bs[n]

        category_scores_n = {
            all_categories[c]: row_cs[c]
            for c in range(C)
            if cat_label_strs[c] and row_cs[c] > 0.0
        }
        category_specificity_n = {
            base_categories[b]: row_bs[b]
            for b in range(B)
            if base_categories[b] in category_scores_n
        }
        if has_combo and "arg1 units and arg2 units" in category_scores_n:
            category_specificity_n["arg1 units and arg2 units"] = combo_s[n]

        labels = [
            cat_label_strs[c]
            for c, cat in enumerate(all_categories)
            if cat in category_scores_n
        ]
        scores_n = {
            cat_label_strs[c]: row_cs[c]
            for c, cat in enumerate(all_categories)
            if cat in category_scores_n
        }
        categories_n = {
            cat: cat_label_strs[c]
            for c, cat in enumerate(all_categories)
            if cat in category_scores_n
        }

        out.append(NodeLabel(
            labels=labels,
            scores=scores_n,
            categories=categories_n,
            category_scores=category_scores_n,
            category_specificity=category_specificity_n,
        ))

    return out

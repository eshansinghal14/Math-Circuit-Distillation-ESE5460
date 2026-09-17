"""ANOVA-style labels for neuron activation heatmaps.

Activation heatmaps are N-D where N = number of args in the prompt/dataset.
2-arg prompts → 2D heatmaps; 3-arg prompts → 3D heatmaps.
The grid dimensionality comes from the MLP input cache (dataset), so the
dataset passed via --dataset must have the same arg count as the prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations

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

    # Combination sum ranges for all subsets of size 2..N-1.
    # The full N-arg sum is already covered by the dedicated "sum range" category below.
    for combo_size in range(2, len(arg_values)):
        for combo_indices in combinations(range(len(arg_values)), combo_size):
            combo_sums = sum(grids[i] for i in combo_indices)
            combo_category = " ".join(f"arg{i + 1}" for i in combo_indices) + " sum range"
            combo_prefix = "+".join(f"arg{i + 1}" for i in combo_indices)
            all_known = target_args is not None and len(target_args) > max(combo_indices)
            if all_known:
                center = int(sum(int(target_args[i]) for i in combo_indices))
                if anova_range_radius:
                    mask = (combo_sums >= center - anova_range_radius) & (combo_sums <= center + anova_range_radius)
                    label = _mask_interval_label(combo_prefix, combo_sums, mask)
                else:
                    mask = combo_sums == center
                    label = _format_interval_label(combo_prefix, center, center)
                rules.append(BasisRule(label, mask, category=combo_category))
            else:
                for center in sorted({int(v) for v in combo_sums.flatten().tolist()}):
                    if anova_range_radius:
                        mask = (combo_sums >= center - anova_range_radius) & (combo_sums <= center + anova_range_radius)
                        label = _mask_interval_label(combo_prefix, combo_sums, mask)
                    else:
                        mask = combo_sums == center
                        label = _format_interval_label(combo_prefix, center, center)
                    rules.append(BasisRule(label, mask, category=combo_category))

    if len(arg_values) >= 2:
        # Sum of ALL arguments across every dimension.
        sums = sum(grids)
        all_args_known = target_args is not None and len(target_args) >= len(arg_values)
        sum_centers = (
            [int(sum(int(target_args[d]) for d in range(len(arg_values))))]
            if all_args_known
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
            [int(sum(int(target_args[d]) for d in range(len(arg_values)))) % 10]
            if all_args_known
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


def gpu_score_activation_heatmaps(
    acts_flat_gpu: torch.Tensor,
    gpu_state: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Score N activation heatmaps against the ANOVA rules, entirely on GPU.

    Returns ``(cat_scores [N, C], base_specificity [N, B], combo_specificity [N] or
    None)``: the per-category explained variance (scatter-max over that category's
    rules), the specificity of each base category (its score minus the best other
    base category) and the specificity of the composite "arg1 units and arg2
    units" category when it exists. This is the whole numerical content of a
    NodeLabel; node_labels_from_scores turns rows into NodeLabel objects.
    """
    N = int(acts_flat_gpu.shape[0])
    masks_flat_gpu   = gpu_state["masks_flat_gpu"]
    rule_cat_ids     = gpu_state["rule_cat_ids"]
    base_to_all      = gpu_state["base_to_all"]
    C                = gpu_state["C"]
    B                = gpu_state["B"]
    arg1u_idx        = gpu_state["arg1u_idx"]
    arg2u_idx        = gpu_state["arg2u_idx"]
    combo_idx        = gpu_state["combo_idx"]
    has_combo        = gpu_state["has_combo"]
    combo_cols       = gpu_state["combo_competitor_cols"]
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
    combo_specificity: torch.Tensor | None = None
    if has_combo:
        combo_competitor_max = (
            base_scores[:, combo_cols].max(dim=1).values.clamp(min=0.0)
            if combo_cols.numel() > 0 else torch.zeros(N, device=dev)
        )
        combo_specificity = cat_scores[:, combo_idx] - combo_competitor_max     # [N]

    return cat_scores, base_specificity, combo_specificity


def node_labels_from_scores(
    cat_scores: torch.Tensor,
    base_specificity: torch.Tensor,
    combo_specificity: torch.Tensor | None,
    *,
    all_categories: list[str],
    base_categories: list[str],
    cat_label_strs: list[str],
    has_combo: bool,
    rows: list[int] | torch.Tensor | None = None,
) -> list[NodeLabel]:
    """NodeLabel objects for ``rows`` (every row if None) of the score tensors.

    One D2H transfer, then pure dict construction. Building these for every
    labelled neuron was the cost of the old label path (hundreds of thousands
    of objects per prompt on the teacher); the training path now builds them
    only for the few neurons that end up in a supernode.
    """
    if rows is None:
        cs = cat_scores.tolist()
        bs = base_specificity.tolist()
        combo_s = combo_specificity.tolist() if has_combo and combo_specificity is not None else None
    else:
        rows_t = torch.as_tensor(rows, dtype=torch.long, device=cat_scores.device)
        cs = cat_scores[rows_t].tolist()
        bs = base_specificity[rows_t].tolist()
        combo_s = combo_specificity[rows_t].tolist() if has_combo and combo_specificity is not None else None
    C = len(all_categories)
    B = len(base_categories)

    out: list[NodeLabel] = []
    for n in range(len(cs)):
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
        if has_combo and combo_s is not None and "arg1 units and arg2 units" in category_scores_n:
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


def gpu_label_activation_heatmaps(
    acts_flat_gpu: torch.Tensor,
    gpu_state: dict,
    rules: list[BasisRule],
) -> list[NodeLabel]:
    """Score and label N neuron activation heatmaps; one NodeLabel per row.

    The CLI / heatmap path. Training uses LabelTable instead, which keeps the
    same scores as tensors and materialises NodeLabels only for selected rows.
    """
    N = int(acts_flat_gpu.shape[0])
    if not rules:
        empty = NodeLabel(labels=[], scores={}, categories={}, category_scores={}, category_specificity={})
        return [empty] * N
    cat_scores, base_specificity, combo_specificity = gpu_score_activation_heatmaps(acts_flat_gpu, gpu_state)
    return node_labels_from_scores(
        cat_scores, base_specificity, combo_specificity,
        all_categories=gpu_state["all_categories"],
        base_categories=gpu_state["base_categories"],
        cat_label_strs=gpu_state["cat_label_strs"],
        has_combo=gpu_state["has_combo"],
    )


@dataclass
class LabelTable:
    """ANOVA scores of N neurons as tensors: what a ``list[NodeLabel]`` holds, without the objects.

    Row ``i`` corresponds to neuron ``i`` of the ``neuron_locations`` it was built
    for. A row of zeros is an unlabelled neuron (the old ``empty`` NodeLabel):
    it has no positive category score, so it is never a candidate.
    """

    cat_scores: torch.Tensor                 # [N, C]
    base_specificity: torch.Tensor           # [N, B]
    combo_specificity: torch.Tensor | None   # [N]
    all_categories: list[str]
    base_categories: list[str]
    cat_label_strs: list[str]
    has_combo: bool

    @classmethod
    def zeros(cls, n: int, gpu_state: dict | None, device) -> "LabelTable":
        if gpu_state is None:
            return cls(torch.zeros(n, 0, device=device), torch.zeros(n, 0, device=device), None, [], [], [], False)
        return cls(
            torch.zeros(n, gpu_state["C"], device=device),
            torch.zeros(n, gpu_state["B"], device=device),
            torch.zeros(n, device=device) if gpu_state["has_combo"] else None,
            list(gpu_state["all_categories"]),
            list(gpu_state["base_categories"]),
            list(gpu_state["cat_label_strs"]),
            bool(gpu_state["has_combo"]),
        )

    @classmethod
    def from_scores(cls, cat_scores, base_specificity, combo_specificity, gpu_state: dict) -> "LabelTable":
        return cls(
            cat_scores, base_specificity, combo_specificity,
            list(gpu_state["all_categories"]), list(gpu_state["base_categories"]),
            list(gpu_state["cat_label_strs"]), bool(gpu_state["has_combo"]),
        )

    def __len__(self) -> int:
        return int(self.cat_scores.shape[0])

    def category_scores_column(self, category: str) -> torch.Tensor | None:
        """``[N]`` scores of ``category``, or None if the table has no such category."""
        if category not in self.all_categories:
            return None
        c = self.all_categories.index(category)
        if not self.cat_label_strs[c]:
            return None
        return self.cat_scores[:, c]

    def specificity_column(self, category: str) -> torch.Tensor:
        """``[N]`` specificity of ``category``; zeros for categories that have none
        (the NodeLabel path's ``category_specificity.get(category, 0.0)``)."""
        if category in self.base_categories:
            return self.base_specificity[:, self.base_categories.index(category)]
        if self.has_combo and self.combo_specificity is not None and category == "arg1 units and arg2 units":
            return self.combo_specificity
        return torch.zeros(len(self), device=self.cat_scores.device)

    def label_for(self, category: str) -> str:
        """The rule label a selected neuron of ``category`` is tagged with (NodeLabel.categories)."""
        if category in self.all_categories:
            label = self.cat_label_strs[self.all_categories.index(category)]
            if label:
                return label
        return category

    def node_label(self, row: int) -> NodeLabel:
        return self.node_labels([row])[0]

    def node_labels(self, rows) -> list[NodeLabel]:
        if len(self.all_categories) == 0:
            empty = NodeLabel(labels=[], scores={}, categories={}, category_scores={}, category_specificity={})
            return [empty] * len(rows)
        return node_labels_from_scores(
            self.cat_scores, self.base_specificity, self.combo_specificity,
            all_categories=self.all_categories, base_categories=self.base_categories,
            cat_label_strs=self.cat_label_strs, has_combo=self.has_combo, rows=rows,
        )

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from graph_loss.anova_node_labels import (  # noqa: E402
    _axis_values_grid,
    _gaussian_template,
    _joint_gaussian_template,
    label_activation_heatmaps,
    template_correlation_score,
)


def test_gaussian_template_peaks_at_center() -> None:
    values = torch.arange(10)
    template = _gaussian_template(values, center=4, cov=4.0)

    assert int(template.argmax().item()) == 4
    assert torch.isclose(template[3], template[5])
    assert template[4] > template[3] > template[0]


def test_label_correlation_prefers_matching_gaussian() -> None:
    arg_values = [list(range(10)), list(range(10))]
    arg1_values = _axis_values_grid(arg_values, 0)
    arg2_values = _axis_values_grid(arg_values, 1)
    matching = _joint_gaussian_template(arg1_values, arg2_values, 4, 6, 4.0)
    shifted = _joint_gaussian_template(arg1_values, arg2_values, 1, 1, 4.0)
    labels = label_activation_heatmaps(
        torch.stack((matching, shifted)),
        arg_values,
        target_args=(4, 6),
        anova_gaussian_cov=4.0,
    )

    target_category = "arg1 range and arg2 range"
    assert labels[0].category_scores[target_category] > labels[1].category_scores[target_category]
    assert labels[0].categories[target_category] == "arg1 2-6 and arg2 4-8"


def test_radius_alias_maps_to_covariance_when_cov_omitted() -> None:
    arg_values = [list(range(10)), list(range(10))]
    arg1_values = _axis_values_grid(arg_values, 0)
    activation = _gaussian_template(arg1_values, 4, 9.0).unsqueeze(0)
    labels = label_activation_heatmaps(
        activation,
        arg_values,
        target_args=(4, 6),
        anova_range_radius=3,
    )

    assert labels[0].categories["arg1 range"] == "arg1 1-7"
    assert labels[0].category_scores["arg1 range"] > 0.99


def test_template_correlation_handles_nans_and_constants() -> None:
    template = torch.tensor([[0.0, 1.0], [2.0, 3.0]])

    assert template_correlation_score(torch.full_like(template, 1.0), template) == 0.0
    assert template_correlation_score(torch.full_like(template, float("nan")), template) == 0.0

    activations = template.clone()
    activations[0, 0] = float("nan")
    assert template_correlation_score(activations, template) > 0.99


if __name__ == "__main__":
    test_gaussian_template_peaks_at_center()
    test_label_correlation_prefers_matching_gaussian()
    test_radius_alias_maps_to_covariance_when_cov_omitted()
    test_template_correlation_handles_nans_and_constants()

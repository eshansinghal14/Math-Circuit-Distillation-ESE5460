import json

import numpy as np


def expected_performance_drop_from_random_ablation_poly(
    fraction_ablated: float,
    poly_equation_json_path: str,
) -> float:
    """Evaluate the saved random-ablation polynomial at ``fraction_ablated``.

    Expects JSON written by :func:`plotting.save_random_ablation_1b_8b_plot` (under
    ``results/random_ablation_poly/<dataset>/``, e.g. ``random_ablation_poly_1b.json`` or
    ``random_ablation_poly_8b.json``; see :func:`random_ablation_poly_output_dir`), with
    ``coefficients`` in ``numpy.polyfit`` order (highest degree first).

    Returns the predicted performance drop (same units as the ablation JSON ``performance_drop``).
    Values outside the fitted ``fraction_ablated_range`` are extrapolated.
    """
    with open(poly_equation_json_path, encoding="utf-8") as f:
        data = json.load(f)
    fmt = data.get("format")
    if fmt is not None and fmt != "numpy_polyfit":
        raise ValueError(
            f"Unsupported poly JSON format {fmt!r} in {poly_equation_json_path!r}",
        )
    coeffs = data.get("coefficients")
    if not isinstance(coeffs, list) or len(coeffs) == 0:
        raise ValueError(
            f"Missing or invalid 'coefficients' list in {poly_equation_json_path!r}",
        )
    coef = np.asarray(coeffs, dtype=np.float64)
    return float(np.polyval(coef, float(fraction_ablated)))

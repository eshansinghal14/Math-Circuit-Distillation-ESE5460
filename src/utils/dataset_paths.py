import os
from typing import Optional, Tuple

from .config import (
    DATASET_ALL_SUFFIX,
    DATASET_TEST_SUFFIX,
    DATASET_TRAIN_SUFFIX,
    REPO_ROOT,
    _DEFAULT_DATASETS_DIR_DRIVE,
)


def default_datasets_dir() -> str:
    """Absolute path to the datasets root.

    If ``/content/drive/My Drive/Math Circuit Distillation (ESE 5460)/datasets`` exists
    (e.g. Colab with Drive mounted), that path is used; otherwise the repository
    ``datasets/`` directory.
    """
    drive = os.path.abspath(_DEFAULT_DATASETS_DIR_DRIVE)
    if os.path.isdir(drive):
        return drive
    return os.path.abspath(os.path.join(REPO_ROOT, "datasets"))


def _safe_dataset_segment(name: Optional[str]) -> Optional[str]:
    """Single path segment for ``random_ablation_poly/<segment>/`` (no ``..`` or separators)."""
    if not name:
        return None
    s = os.path.basename(str(name).strip())
    if not s or s in (".", ".."):
        return None
    return s


def random_ablation_poly_output_dir(
    plot_input_directory: Optional[str] = None,
    *,
    dataset: Optional[str] = None,
) -> str:
    """Directory for polynomial JSONs produced by :func:`plotting.save_random_ablation_1b_8b_plot`.

    Preferred layout: ``<repo>/results/random_ablation_poly/<dataset>/`` with JSON files
    ``random_ablation_poly_1b.json`` / ``random_ablation_poly_8b.json`` inside. Pass
    ``dataset`` (e.g. training prefix ``222_add``) for that layout.

    If ``dataset`` is set, its sanitized basename is used and ``plot_input_directory`` is
    ignored for the output path.

    If ``dataset`` is omitted, falls back to ``<leaf>/`` under ``random_ablation_poly/`` where
    ``leaf`` is the last segment of ``plot_input_directory``, unless that segment is named
    ``results`` (then no extra subfolder). If ``plot_input_directory`` is also omitted,
    returns ``random_ablation_poly`` with no leaf.
    """
    poly_base = os.path.join(REPO_ROOT, "results", "random_ablation_poly")
    ds = _safe_dataset_segment(dataset)
    if ds:
        return os.path.join(poly_base, ds)
    if plot_input_directory is None:
        return poly_base
    plot_input_directory = os.path.abspath(plot_input_directory)
    leaf = os.path.basename(os.path.normpath(plot_input_directory))
    if leaf == "results":
        return poly_base
    return os.path.join(poly_base, leaf)


def dataset_train_json_path(prefix: str, datasets_dir: Optional[str] = None) -> str:
    d = os.path.abspath(datasets_dir) if datasets_dir else default_datasets_dir()
    return os.path.join(d, f"{prefix}{DATASET_TRAIN_SUFFIX}.json")


def dataset_test_json_path(prefix: str, datasets_dir: Optional[str] = None) -> str:
    d = os.path.abspath(datasets_dir) if datasets_dir else default_datasets_dir()
    return os.path.join(d, f"{prefix}{DATASET_TEST_SUFFIX}.json")


def dataset_all_json_path(prefix: str, datasets_dir: Optional[str] = None) -> str:
    d = os.path.abspath(datasets_dir) if datasets_dir else default_datasets_dir()
    return os.path.join(d, f"{prefix}{DATASET_ALL_SUFFIX}")


def _resolve_dataset_output_path(
    dataset_fname: str,
    datasets_dir: Optional[str] = None,
) -> str:
    """Place bare filenames (no directory) under ``datasets/``; otherwise absolute path."""
    expanded = os.path.expanduser(dataset_fname)
    if os.path.dirname(expanded) == "":
        root = os.path.abspath(datasets_dir) if datasets_dir else default_datasets_dir()
        return os.path.join(root, os.path.basename(expanded))
    return os.path.abspath(expanded)


def require_dataset_prefix(
    dataset: Optional[str],
    datasets_dir: Optional[str],
) -> str:
    """Return stripped ``--dataset`` PREFIX or exit with an error (no interactive prompt)."""
    prefix = (dataset or "").strip()
    if prefix:
        return prefix
    d = os.path.abspath(datasets_dir) if datasets_dir else default_datasets_dir()
    raise SystemExit(
        "ERROR: --dataset PREFIX is required. "
        f"Example: --dataset 2d_add for <PREFIX>{DATASET_TRAIN_SUFFIX}.json and "
        f"<PREFIX>{DATASET_TEST_SUFFIX}.json under {d}."
    )


def resolve_train_test_paths(
    *,
    dataset: Optional[str],
    datasets_dir: Optional[str],
) -> Tuple[str, str, str]:
    """Resolve train/test JSON paths from ``--dataset PREFIX``.

    Returns:
        ``(train_path, test_path, prefix)`` with absolute paths.
    """
    prefix = require_dataset_prefix(dataset, datasets_dir)
    d = os.path.abspath(datasets_dir) if datasets_dir else default_datasets_dir()
    train = dataset_train_json_path(prefix, d)
    test = dataset_test_json_path(prefix, d)
    for label, p in (("train", train), ("test", test)):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Dataset {label} file not found for prefix {prefix!r}: {p}",
            )
    return train, test, prefix


def resolve_test_path(
    *,
    dataset: Optional[str],
    datasets_dir: Optional[str],
) -> Tuple[str, str]:
    """Resolve test JSON for eval-only scripts. Returns ``(test_path, prefix)``."""
    prefix = require_dataset_prefix(dataset, datasets_dir)
    d = os.path.abspath(datasets_dir) if datasets_dir else default_datasets_dir()
    p = dataset_test_json_path(prefix, d)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Test file not found for prefix {prefix!r}: {p}")
    return p, prefix

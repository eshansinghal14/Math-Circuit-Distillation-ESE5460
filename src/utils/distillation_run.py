import os
from typing import Optional, Tuple

from .config import STUDENT_MODEL_DIR, STUDENT_WEIGHTS_FILE
from .fs import most_recent_subdirectory


def resolve_distillation_run_dir(
    save_dir: str,
    *,
    resume: bool,
    checkpoint_run: Optional[str],
    runs_subdir: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Return ``(run_dir, student_source)``.

    All training outputs go **directly** under ``<save_dir>`` (or ``<save_dir>/<runs_subdir>``
    if set). There is no timestamp or extra run subfolder.

    New run: ``run_dir`` is that directory; ``student_source`` is None.

    Resume: load from ``<run_dir>/student_model/`` (or legacy ``student_weights.pt``).
    Pass ``checkpoint_run`` as a path relative to ``save_dir`` to resume a **legacy** nested
    run (e.g. ``2026-04-07_22-15-56`` or ``neuron-cluster/2026-04-07_22-15-56``). If omitted,
    uses ``save_dir`` (or the ``runs_subdir`` base) when checkpoints exist there; otherwise
    picks the most recently modified subfolder under that base (legacy multi-run layouts).
    """
    save_dir = os.path.abspath(save_dir)
    sub = (runs_subdir or "").strip().strip("/").strip("\\")
    base = os.path.join(save_dir, sub) if sub else save_dir

    if not resume:
        return base, None

    if checkpoint_run:
        cr = checkpoint_run.replace("\\", "/").strip("/")
        if sub and not cr.startswith(f"{sub}/"):
            cr = f"{sub}/{cr}"
        run_dir = os.path.join(save_dir, cr)
    else:
        hf_here = os.path.join(base, STUDENT_MODEL_DIR)
        wt_here = os.path.join(base, STUDENT_WEIGHTS_FILE)
        if os.path.isdir(hf_here) or os.path.isfile(wt_here):
            run_dir = base
        else:
            run_dir = most_recent_subdirectory(base)
            if run_dir is None:
                raise SystemExit(
                    f"No checkpoints in {base} and no run subfolders.\n"
                    "Train here first or pass --checkpoint-run <path under --save-dir>."
                )
            print(f"Auto-detected most recent run folder: {run_dir}")

    hf_path = os.path.join(run_dir, STUDENT_MODEL_DIR)
    wt_path = os.path.join(run_dir, STUDENT_WEIGHTS_FILE)
    if os.path.isdir(hf_path):
        student_source = hf_path
        print(f"Loading student from {student_source}")
    elif os.path.isfile(wt_path):
        student_source = wt_path
        print(f"Loading student weights from {student_source} (fast checkpoint)")
    else:
        raise SystemExit(
            f"Resume expected saved weights at {hf_path} or {wt_path}. "
            "Train a run first."
        )
    return run_dir, student_source

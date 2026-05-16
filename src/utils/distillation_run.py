import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .config import (
    EVAL_MAX_NEW_TOKENS,
    LLAMA_1B_MODEL_NAME,
    LLAMA_8B_MODEL_NAME,
    STUDENT_MODEL_DIR,
    STUDENT_WEIGHTS_FILE,
)
from .device import get_default_device
from .fs import most_recent_subdirectory


@dataclass
class DistillationConfig:
    teacher_model: str = LLAMA_8B_MODEL_NAME
    student_model: str = LLAMA_1B_MODEL_NAME
    steps: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    temperature: float = 2.0
    grad_clip: float = 1.0
    lambda_graph: float = 0.1
    graph_dtype: Optional[torch.dtype] = None
    graph_top_k_logits: Optional[int] = 20
    graph_prop_neurons_per_layer: float = 0.1
    graph_gen_batch_size: int = 1
    teacher_graph_batch_size: int = 512
    student_graph_batch_size: int = 1
    graph_verbose: bool = False
    graph_prune: bool = False
    graph_node_threshold: float = 0.8
    graph_edge_threshold: float = 0.98
    graph_node_weight: float = 1.0
    graph_edge_weight: float = 0.0
    graph_similarity_threshold: float = 0.7
    graph_max_fan_out: int = 4
    fast_teacher_graph: bool = False
    student_computation_eps: float = 0.1
    student_embedding_eps: float = 0.1
    student_activation_forward_batch_size: int = 32
    student_skip_logit_attribution: bool = True
    align_diagnostic: bool = False
    graph_focus_weight: float = 0.0
    graph_grad_mode: str = "approx"
    graph_true_grad_chunk_size: int = 4
    fast_student_graph: bool = False
    ablation_batch_size: int = 32
    align_by_label: bool = False
    student_cluster_method: str = "ablation"
    student_dataset: Optional[str] = None
    student_mlp_input_cache_path: Optional[str] = None
    student_activation_write_cache_path: Optional[str] = None
    student_anova_range_radius: int = 0
    student_anova_nodes_per_label: int = 10
    student_sum_min_specificity: float = 0.0
    lambda_kl: float = 1.0
    graph_grad_norm_scale: bool = False
    graph_start_step: int = 1
    student_fixed_labels_path: Optional[str] = None
    eval_batch_size: int = 50
    step_log_interval: int = 50
    save_best: bool = False
    eval_max_new_tokens: int = EVAL_MAX_NEW_TOKENS
    track_loss_grads: bool = False
    save_dir: str = "results/distillation"
    teacher_data_cache: Optional[str] = None
    seed: int = 42
    device: torch.device = get_default_device()
    save_interval: int = 0
    label_refresh_interval: int = 0
    label_refresh_n_prompts: int = 64


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

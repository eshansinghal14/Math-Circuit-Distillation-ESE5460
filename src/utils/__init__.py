from .answer_parsing import parse_answer
from .config import (
    EVAL_MAX_NEW_TOKENS,
    HF_READ_TOKEN,
    LLAMA_1B_MODEL_NAME,
    LLAMA_8B_MODEL_NAME,
    NEURON_CLUSTERING_STUDENT_SUBPATH,
    NEURON_CLUSTERING_SUBDIR,
    NEURON_CLUSTERING_TEACHER_SUBPATH,
    STUDENT_MODEL_DIR,
)
from .dataset_json import json_to_prompt_answer_dict, load_prompt_answer_json
from .dataset_paths import (
    dataset_all_json_path,
    default_datasets_dir,
    random_ablation_poly_output_dir,
    resolve_test_path,
    resolve_train_test_paths,
)
from .device import get_default_device, seed_all
from .distillation_batch import AddDataset, collate_fn, masked_kl_loss
from .distillation_run import DistillationConfig, resolve_distillation_run_dir
from .distillation_trainer import DistillationTrainer
from .eval_inference import (
    evaluate_prompt_answer_dict,
    run_hf_benchmark,
    test_model,
)
from .fs import most_recent_subdirectory, rm_dir_tree
from .hf_models import (
    load_model,
    load_student_model_for_distillation,
    patch_tokenizer_no_special_tokens,
)
from .synthetic_chain_math import generate_math_dataset, normalize_op_patterns

__all__ = [
    "AddDataset",
    "DistillationConfig",
    "DistillationTrainer",
    "EVAL_MAX_NEW_TOKENS",
    "HF_READ_TOKEN",
    "LLAMA_1B_MODEL_NAME",
    "LLAMA_8B_MODEL_NAME",
    "NEURON_CLUSTERING_STUDENT_SUBPATH",
    "NEURON_CLUSTERING_SUBDIR",
    "NEURON_CLUSTERING_TEACHER_SUBPATH",
    "STUDENT_MODEL_DIR",
    "collate_fn",
    "dataset_all_json_path",
    "default_datasets_dir",
    "evaluate_prompt_answer_dict",
    "generate_math_dataset",
    "get_default_device",
    "json_to_prompt_answer_dict",
    "load_model",
    "load_prompt_answer_json",
    "load_student_model_for_distillation",
    "masked_kl_loss",
    "most_recent_subdirectory",
    "normalize_op_patterns",
    "parse_answer",
    "patch_tokenizer_no_special_tokens",
    "random_ablation_poly_output_dir",
    "resolve_distillation_run_dir",
    "resolve_test_path",
    "resolve_train_test_paths",
    "rm_dir_tree",
    "run_hf_benchmark",
    "seed_all",
    "test_model",
]

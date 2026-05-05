import os

try:
    import constants as _constants

    HF_READ_TOKEN = getattr(_constants, "HF_READ_TOKEN", "") or getattr(
        _constants, "HF_TOKEN", ""
    )
except ModuleNotFoundError:
    HF_READ_TOKEN = ""

if not HF_READ_TOKEN:
    HF_READ_TOKEN = os.environ.get("HF_READ_TOKEN", "") or os.environ.get("HF_TOKEN", "")

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(SRC_ROOT, ".."))

# Prompts end with "=" and all 2-3 digit answers (20-198) are single tokens in Llama-3 BPE.
EVAL_MAX_NEW_TOKENS = 1

# --- Distillation run dirs / checkpoints ---------------------------------------------
STUDENT_MODEL_DIR = "student_model"
# Legacy fast-checkpoint filename (older runs may still contain this file).
STUDENT_WEIGHTS_FILE = "student_weights.pt"

LLAMA_1B_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA_8B_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# Legacy neuron-clustering result locations kept for plotting/analysis helpers.
NEURON_CLUSTERING_SUBDIR = "neuron_clustering"
# HF model ids as nested path segments under ``<run>/neuron_clustering/``.
NEURON_CLUSTERING_STUDENT_SUBPATH = LLAMA_1B_MODEL_NAME
NEURON_CLUSTERING_TEACHER_SUBPATH = LLAMA_8B_MODEL_NAME

# --- Dataset file naming: ``{prefix}_train.json``, ``{prefix}_test.json``, ``{prefix}_all.json`` ---
DATASET_TRAIN_SUFFIX = "_train"
DATASET_TEST_SUFFIX = "_test"
DATASET_ALL_SUFFIX = "_all.json"  # ``{prefix}_all.json``

# Colab + Google Drive default when the folder is mounted; otherwise use repo ``datasets/``.
_DEFAULT_DATASETS_DIR_DRIVE = (
    "/content/drive/My Drive/Math Circuit Distillation (ESE 5460)/datasets"
)

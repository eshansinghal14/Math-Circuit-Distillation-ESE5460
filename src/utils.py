import itertools
import json
import math
import random
import re
import os
import glob
import shutil
import subprocess
import torch
import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple, Union

try:
    import constants as _constants
    HF_READ_TOKEN = getattr(_constants, "HF_READ_TOKEN", "") or getattr(
        _constants, "HF_TOKEN", ""
    )
    CIRCUIT_DISCOVERY_CKPT_DIR = getattr(_constants, "CIRCUIT_DISCOVERY_CKPT_DIR", "")
except ModuleNotFoundError:
    HF_READ_TOKEN = ""
    CIRCUIT_DISCOVERY_CKPT_DIR = os.environ.get("CIRCUIT_DISCOVERY_CKPT_DIR", "")

if not HF_READ_TOKEN:
    HF_READ_TOKEN = os.environ.get("HF_READ_TOKEN", "") or os.environ.get("HF_TOKEN", "")

from transformers.utils import logging as hf_logging

logged_in = False

# Prompts end with "=" and all 2-3 digit answers (20-198) are single tokens in Llama-3 BPE.
EVAL_MAX_NEW_TOKENS = 1

# --- Distillation run dirs / checkpoints (used by ``neuron_distillation``) ------------
STUDENT_MODEL_DIR = "student_model"
# Legacy fast-checkpoint filename (older runs may still contain this file).
STUDENT_WEIGHTS_FILE = "student_weights.pt"

LLAMA_1B_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA_8B_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# Top-level folder under each run root where :mod:`neuron_distillation.clustering` writes
# (must match ``_neuron_clustering_run_dir``).
NEURON_CLUSTERING_SUBDIR = "neuron_clustering"
# HF model ids as nested path segments under ``<run>/neuron_clustering/``.
NEURON_CLUSTERING_STUDENT_SUBPATH = LLAMA_1B_MODEL_NAME
NEURON_CLUSTERING_TEACHER_SUBPATH = LLAMA_8B_MODEL_NAME


def rm_dir_tree(path: str) -> None:
    """Delete a directory tree (shell ``rm -rf`` on Unix, :func:`shutil.rmtree` fallback)."""
    try:
        result = subprocess.run(["rm", "-rf", path], capture_output=True)
        if result.returncode != 0:
            raise OSError(result.stderr.decode().strip())
    except Exception:
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass


def training_state_path(save_dir: str) -> str:
    return os.path.join(save_dir, "training_state.pt")


def save_training_state(
    save_dir: str,
    optimizer: torch.optim.Optimizer,
    next_epoch: int,
    best_acc: float,
) -> None:
    """``next_epoch`` = number of epochs already completed (resume starts at this index)."""
    path = training_state_path(save_dir)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "next_epoch": next_epoch,
            "best_acc": best_acc,
        },
        path,
    )


def load_training_state(
    path: str, optimizer: torch.optim.Optimizer, map_location,
) -> Tuple[int, float]:
    try:
        chk = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        chk = torch.load(path, map_location=map_location)
    optimizer.load_state_dict(chk["optimizer"])
    return int(chk["next_epoch"]), float(chk["best_acc"])


def most_recent_subdirectory(parent_dir: str) -> Optional[str]:
    """Most recently modified immediate subdirectory of ``parent_dir``."""
    if not os.path.isdir(parent_dir):
        return None
    try:
        entries = os.listdir(parent_dir)
    except OSError:
        return None
    best_mtime, best_path = None, None
    for name in entries:
        full = os.path.join(parent_dir, name)
        if not os.path.isdir(full):
            continue
        try:
            mtime = os.path.getmtime(full)
            if best_mtime is None or mtime > best_mtime:
                best_mtime, best_path = mtime, full
        except OSError:
            pass
    return best_path


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


def json_to_prompt_answer_dict(raw: object) -> Dict[str, int]:
    """Normalize math-dataset JSON to ``{prompt: int answer}``.

    Supports:

    - **Flat dict** ``{"12+34=": 46, ...}`` (string or int values).
    - **List of records** ``[{"q_str": "...", "a_str": "..."}, ...]`` (current repo format).
    """
    if isinstance(raw, dict):
        return {str(k): int(v) for k, v in raw.items()}
    if isinstance(raw, list):
        out: Dict[str, int] = {}
        for i, row in enumerate(raw):
            if not isinstance(row, dict):
                raise TypeError(f"Dataset row {i} must be a dict, got {type(row)}")
            if "q_str" not in row or "a_str" not in row:
                raise ValueError(
                    "List-format rows must include q_str and a_str; "
                    f"row {i} has keys: {list(row.keys())}",
                )
            out[str(row["q_str"])] = int(row["a_str"])
        return out
    raise TypeError(f"Dataset JSON must be a dict or list, got {type(raw)!r}")


def load_prompt_answer_json(path: str) -> Dict[str, int]:
    """Load train/test JSON from disk into ``{prompt: int answer}``."""
    with open(path, "r", encoding="utf-8") as f:
        return json_to_prompt_answer_dict(json.load(f))


def _load_dataset_rows(path: str) -> List[Dict]:
    """Load a dataset JSON as a list of row dicts for mix/export helpers."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise TypeError(
            f"Expected list-format dataset at {path!r} for mixing, got {type(raw)}."
        )
    rows: List[Dict] = []
    for i, row in enumerate(raw):
        if not isinstance(row, dict):
            raise TypeError(f"Dataset row {i} in {path!r} must be a dict, got {type(row)}")
        if "q_str" not in row or "a_str" not in row:
            raise ValueError(
                f"Dataset row {i} in {path!r} must include q_str and a_str; "
                f"got keys {list(row.keys())}",
            )
        rows.append(dict(row))
    return rows


def mix_datasets(
    dataset_stems: Sequence[str],
    output_stem: str,
    *,
    datasets_dir: Optional[str] = None,
    shuffle: bool = True,
) -> Dict[str, str]:
    """Mix existing dataset families into combined ``_all``, ``_train``, and ``_test`` JSONs.

    Args:
        dataset_stems: Source dataset prefixes, e.g. ``["22_add", "222_add_tight"]``.
        output_stem: Prefix for the mixed outputs, e.g. ``"mixed_add"``.
        datasets_dir: Optional datasets root; defaults to :func:`default_datasets_dir`.
        shuffle: If True (default), shuffle each combined split after concatenation.

    Returns:
        Mapping ``{"all": ..., "train": ..., "test": ...}`` with output file paths.
    """
    stems = [str(stem).strip() for stem in dataset_stems if str(stem).strip()]
    if not stems:
        raise ValueError("dataset_stems must contain at least one non-empty dataset prefix.")
    out_stem = str(output_stem).strip()
    if not out_stem:
        raise ValueError("output_stem must be a non-empty dataset prefix.")

    d = os.path.abspath(datasets_dir) if datasets_dir else default_datasets_dir()
    mixed = {"all": [], "train": [], "test": []}
    split_paths = {
        "all": dataset_all_json_path,
        "train": dataset_train_json_path,
        "test": dataset_test_json_path,
    }

    for stem in stems:
        for split, path_fn in split_paths.items():
            path = path_fn(stem, d)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Dataset {split} file not found for prefix {stem!r}: {path}",
                )
            mixed[split].extend(_load_dataset_rows(path))

    if shuffle:
        for rows in mixed.values():
            random.shuffle(rows)

    out_paths = {
        "all": dataset_all_json_path(out_stem, d),
        "train": dataset_train_json_path(out_stem, d),
        "test": dataset_test_json_path(out_stem, d),
    }

    for split, path in out_paths.items():
        out_dir = os.path.dirname(os.path.abspath(path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mixed[split], f, indent=4)

    return out_paths


def _extract_int_after_equals(text: str) -> Optional[int]:
    m = re.search(r"=\s*(\d+)", text)
    return int(m.group(1)) if m else None


def _normalize_numeric_str(s: str) -> str:
    """Canonical string for comparing numeric answers (e.g. ``3.0`` → ``3``)."""
    s = str(s).strip()
    if not s:
        return s
    try:
        x = float(s)
        if math.isfinite(x) and x == int(x):
            return str(int(x))
        return str(x)
    except ValueError:
        return s


def extract_numeric_answer_from_text(text: str) -> Optional[str]:
    """Parse a numeric answer from model output (GSM8K/SVAMP-style).

    Prefers phrases like ``The answer is``, ``Answer:``, ``is``; otherwise last number in text.
    Returns normalized string (see :func:`_normalize_numeric_str`) or ``None``.
    """
    text = (text or "").strip()
    if not text:
        return None
    m = re.search(
        r"(?:(?:The\s+answer\s+is)|(?:^|\s)Answer\s*:?|(?:^|\s)is\s*:?)\s*(-?\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    if m:
        return _normalize_numeric_str(m.group(1))
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if matches:
        return _normalize_numeric_str(matches[-1])
    return None


def _gsm8k_gold_answer_str(answer_field: str) -> Optional[str]:
    m = re.search(r"####\s*(\d+)", answer_field)
    return _normalize_numeric_str(m.group(1)) if m else None


def _slice_text_after_reasoning(full_decoded: str) -> str:
    """Use the segment after ``Reasoning:`` for extraction (model repeats prompt)."""
    idx = full_decoded.find("Reasoning:")
    if idx != -1:
        return full_decoded[idx + len("Reasoning:") :].strip()
    return full_decoded.strip()


def load_hf_benchmark_rows(
    name: str,
    *,
    limit: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """``load_dataset`` for ``gsm8k`` or ``svamp``; return ``(prompt, gold_str)`` rows.

    Prompt format: ``Question: …\\nReasoning:``. Gold is a normalized numeric string.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "gsm8k/svamp require `datasets`. Install with: pip install datasets",
        ) from e

    key = name.strip().lower()
    rows: List[Tuple[str, str]] = []

    if key == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        for ex in ds:
            q = ex["question"].strip()
            gold = _gsm8k_gold_answer_str(ex["answer"])
            if gold is None:
                continue
            prompt = f"Question: {q}\nReasoning:"
            rows.append((prompt, gold))
            if limit is not None and len(rows) >= limit:
                break
    elif key == "svamp":
        ddict = load_dataset("ChilleD/SVAMP")
        split = "test" if "test" in ddict else "train"
        for ex in ddict[split]:
            if ex.get("question_concat"):
                q = str(ex["question_concat"]).strip()
            else:
                q = f"{str(ex.get('Body', '')).strip()}\n{str(ex.get('Question', '')).strip()}"
            raw_a = ex.get("Answer", ex.get("answer"))
            if raw_a is None:
                continue
            try:
                gold = _normalize_numeric_str(str(raw_a).strip())
            except Exception:
                continue
            prompt = f"Question: {q}\nReasoning:"
            rows.append((prompt, gold))
            if limit is not None and len(rows) >= limit:
                break
    else:
        raise ValueError(f"Unknown HF benchmark dataset: {name!r}")

    return rows


@torch.no_grad()
def run_hf_benchmark(
    model,
    tokenizer,
    name: str,
    results_fname: str,
    *,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    limit: Optional[int] = None,
    log: bool = True,
) -> Tuple[List[Dict], float]:
    """GSM8K / SVAMP via ``load_dataset``; greedy generate; save JSON like :func:`test_model`.

    Returns ``(results, accuracy)`` where each result has ``response``, ``answer`` (gold str),
    and ``parsed`` (predicted numeric str or ``null``).
    """
    rows = load_hf_benchmark_rows(name, limit=limit)
    n = len(rows)
    results: List[Dict] = []
    correct = 0
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    try:
        for i in range(0, n, batch_size):
            batch = rows[i : i + batch_size]
            prompts = [p for p, _ in batch]
            golds = [g for _, g in batch]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True,
            ).to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for j, full_text in enumerate(decoded):
                tail = _slice_text_after_reasoning(full_text)
                pred = extract_numeric_answer_from_text(tail)
                gold = golds[j]
                if pred is not None and pred == gold:
                    correct += 1
                results.append(
                    {
                        "response": full_text,
                        "answer": gold,
                        "parsed": pred,
                    },
                )
            if log:
                end = min(i + batch_size, n)
                print(f"processing {end}/{n}")
    finally:
        tokenizer.padding_side = original_side

    acc = correct / n if n else 0.0
    with open(results_fname, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    return results, acc


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    test_path: str,
    batch_size: int = 32,
    max_new_tokens: Optional[int] = None,
    debug_decode: int = 0,
    debug_tag: Optional[str] = None,
) -> float:
    """Greedy generation accuracy on a math test JSON (right padding; int after ``=``).

    Uses the same padding side as :class:`neuron_distillation.distillation.AddDataset` /
    ``collate_fn`` so RoPE positions match distillation training.
    """
    if max_new_tokens is None:
        max_new_tokens = EVAL_MAX_NEW_TOKENS
    with open(test_path, "r", encoding="utf-8") as f:
        data = json_to_prompt_answer_dict(json.load(f))
    prompts = list(data.keys())
    answers = list(data.values())

    model.eval()
    correct = total = 0
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    for i in range(0, len(prompts), batch_size):
        batch_p = prompts[i : i + batch_size]
        batch_a = answers[i : i + batch_size]
        inputs = tokenizer(
            batch_p,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        ).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if debug_decode > 0 and i == 0:
            n = min(debug_decode, len(decoded))
            tag = f" [{debug_tag}]" if debug_tag else ""
            print(f"--- decode debug{tag} (first batch, max_new_tokens={max_new_tokens}) ---")
            for j in range(n):
                text = decoded[j]
                gold = batch_a[j]
                pred = _extract_int_after_equals(text)
                print(f"  gold={gold}  pred={pred}  decoded={text!r}")
            print("---")

        for text, gold in zip(decoded, batch_a):
            pred = _extract_int_after_equals(text)
            if pred == gold:
                correct += 1
            total += 1

    tokenizer.padding_side = original_side
    return correct / max(total, 1)


# --- Dataset file naming: ``{prefix}_train.json``, ``{prefix}_test.json``, ``{prefix}_all.json`` ---
DATASET_TRAIN_SUFFIX = "_train"
DATASET_TEST_SUFFIX = "_test"
DATASET_ALL_SUFFIX = "_all.json"  # ``{prefix}_all.json``


# Colab + Google Drive default when the folder is mounted; otherwise use repo ``datasets/``.
_DEFAULT_DATASETS_DIR_DRIVE = (
    "/content/drive/My Drive/Math Circuit Distillation (ESE 5460)/datasets"
)


def default_datasets_dir() -> str:
    """Absolute path to the datasets root.

    If ``/content/drive/My Drive/Math Circuit Distillation (ESE 5460)/datasets`` exists
    (e.g. Colab with Drive mounted), that path is used; otherwise the repository
    ``datasets/`` directory (sibling of ``src/``).
    """
    drive = os.path.abspath(_DEFAULT_DATASETS_DIR_DRIVE)
    if os.path.isdir(drive):
        return drive
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))


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
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    poly_base = os.path.join(repo_root, "results", "random_ablation_poly")
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


def patch_tokenizer_no_special_tokens(tokenizer):
    """Default ``add_special_tokens=False`` on ``__call__`` and ``encode`` (idempotent).

    Avoids silently prepending BOS / appending EOS on encode paths so training and eval
    match raw string tokenization.
    """
    if getattr(tokenizer, "_math_circuit_no_special_tokens_patched", False):
        return tokenizer
    orig_call = tokenizer.__call__
    orig_encode = tokenizer.encode

    def _call(*args, **kwargs):
        kwargs.setdefault("add_special_tokens", False)
        return orig_call(*args, **kwargs)

    def _encode(*args, **kwargs):
        kwargs.setdefault("add_special_tokens", False)
        return orig_encode(*args, **kwargs)

    tokenizer.__call__ = _call
    tokenizer.encode = _encode
    tokenizer._math_circuit_no_special_tokens_patched = True
    return tokenizer


def load_model(model_name):

    hf_logging.set_verbosity_error()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import login

    global logged_in
    if not logged_in and HF_READ_TOKEN:
        login(HF_READ_TOKEN)
        logged_in = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if torch.cuda.is_available() else None,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # Match neuron_distillation AddDataset/collate_fn (right-pad); eval/generate use same side.
    tokenizer.padding_side = "right"
    return model, patch_tokenizer_no_special_tokens(tokenizer)


def load_student_model_for_distillation(
    student_source: Optional[str],
    student_model_id: str,
    device: Union[str, torch.device],
):
    """Load student LM + tokenizer for distillation (HF id, saved HF dir, or ``.pt`` state_dict)."""
    print("\n" + "=" * 60)
    print("Loading student")
    print("=" * 60)
    if student_source and student_source.endswith(".pt"):
        print(f"  Loading base model + fast weights from: {student_source!r}")
        student, tokenizer = load_model(student_model_id)
        state_dict = torch.load(student_source, map_location="cpu", weights_only=True)
        student.load_state_dict(state_dict)
        del state_dict
    elif student_source:
        print(f"  From checkpoint dir: {student_source!r}")
        student, tokenizer = load_model(student_source)
    else:
        print(f"  From Hugging Face: {student_model_id!r}")
        student, tokenizer = load_model(student_model_id)
    student = student.to("cpu").float().to(device)
    tokenizer.padding_side = "right"
    return student, tokenizer


def test_model(model, tokenizer, dataset_fname, results_fname, batch_size=50, max_new_tokens=EVAL_MAX_NEW_TOKENS, log=True):
    """Greedy eval on a math JSON file.

    ``dataset_fname`` uses the same formats as :func:`json_to_prompt_answer_dict`:
    flat ``{prompt: answer}`` or a list of ``{q_str, a_str}`` rows (extra keys allowed).
    """
    model.eval()
    with open(dataset_fname, encoding="utf-8") as f:
        raw = json.load(f)
    data = json_to_prompt_answer_dict(raw)
    prompts = list(data.keys())
    answers = [int(v) for v in data.values()]
    n = len(prompts)
    results = []
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for i in range(0, n, batch_size):
            with torch.no_grad():
                if log:
                    print(f"processing {i}/{n}")
                end = min(i + batch_size, n)
                batched_prompts = prompts[i:end]
                batched_answers = answers[i:end]
                input_ids = tokenizer(
                    batched_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    add_special_tokens=True,
                ).to(model.device)
                outputs = model.generate(
                    **input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                for k, resp in enumerate(responses):
                    results.append({"response": resp, "answer": str(batched_answers[k])})
    finally:
        tokenizer.padding_side = original_side

    with open(results_fname, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    return results

def parse_answer(resp):
    return _extract_int_after_equals(resp)

def eval_model(results_fname, log: bool = True):
    with open(results_fname, "r", encoding="utf-8") as f:
        results = json.load(f)
    if not results:
        return 0.0

    correct = 0
    for res in results:
        if parse_answer(res["response"]) == int(res["answer"]):
            correct += 1

    acc = correct / len(results)
    if log:
        print("Accuracy: ", acc)
    return acc


def mlp_flatten_dim_from_model(model) -> int:
    """Total flattened MLP width (``layers × intermediate_size``) for Llama-style causal LMs."""
    cfg = model.config
    return int(cfg.num_hidden_layers) * int(cfg.intermediate_size)


def mlp_flatten_dim_from_pretrained_id(model_id: str) -> int:
    """Same as :func:`mlp_flatten_dim_from_model` but from a HuggingFace id (no model load)."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, token=HF_READ_TOKEN or None)
    return int(cfg.num_hidden_layers) * int(cfg.intermediate_size)


def normalize_op_patterns(
    operations: Sequence[Sequence[str]],
    num_gaps: int,
) -> List[List[str]]:
    """Validate 2D ``operations``: each row is one full operator sequence between operands.

    Each row must have length ``num_gaps`` (``len(digits) - 1``), or length 1 to broadcast that
    operator to every gap. Only ``+``, ``*``, and ``×`` are allowed.
    """
    if num_gaps < 1:
        raise ValueError("num_gaps must be >= 1.")
    if not operations:
        raise ValueError("operations must be a non-empty list of operator rows.")
    allowed = {"+", "*", "×"}
    out: List[List[str]] = []
    for row in operations:
        ops = [o.strip() for o in row if o and str(o).strip()]
        if len(ops) == 1 and num_gaps > 1:
            ops = [ops[0]] * num_gaps
        elif len(ops) != num_gaps:
            raise ValueError(
                f"each operator row must have length {num_gaps} "
                f"(or length 1 to broadcast); got {len(ops)} in {list(row)!r}",
            )
        for o in ops:
            if o not in allowed:
                raise ValueError(
                    f"operations must use only '+' or '*' (or '×'); got {o!r}.",
                )
        out.append(ops)
    return out


def _eval_pemdas_plus_mul(numbers: Sequence[int], op_syms: Sequence[str]) -> int:
    """Evaluate with multiplication before addition (PEMDAS for ``+`` and ``*`` only)."""
    values = list(numbers)
    ops: List[str] = []
    for o in op_syms:
        s = o.strip()
        if s == "×":
            s = "*"
        ops.append(s)

    i = 0
    while i < len(ops):
        if ops[i] == "*":
            values[i] = values[i] * values[i + 1]
            del values[i + 1]
            del ops[i]
        else:
            i += 1
    return sum(values)


def _format_chain_prompt(
    nums: Sequence[int],
    op_syms: Sequence[str],
    mod_n: Optional[int],
    *,
    spaces: bool = True,
) -> str:
    parts: List[str] = [str(nums[0])]
    for i, sym in enumerate(op_syms):
        if spaces:
            parts.append(f" {sym} {nums[i + 1]}")
        else:
            parts.append(f"{sym}{nums[i + 1]}")
    body = "".join(parts)
    if mod_n is not None:
        if spaces:
            return f"{body} mod {mod_n} = "
        return f"{body}mod{mod_n}="
    if spaces:
        return f"{body} = "
    return f"{body}="


def _iter_chain_pairs(
    digits: Sequence[int],
    operations: Sequence[Sequence[str]],
    mod_n: Optional[int],
    *,
    spaces: bool = True,
) -> List[Tuple[str, int]]:
    """Cartesian product over operand ranges; each row picks one operator pattern at random.

    **PEMDAS:** ``*`` / ``×`` before ``+``. ``operations`` is a 2D list: each inner list is one
    allowed ordering of ``len(digits)-1`` operators; one ordering is chosen uniformly at random
    per problem. If ``mod_n`` is set, the gold answer is ``(value) % mod_n`` and the prompt
    ends with ``... mod n = ``.
    """
    if len(digits) < 2:
        raise ValueError("digits must list at least two operand widths.")
    if any(d < 1 for d in digits):
        raise ValueError("each entry in digits must be >= 1.")
    num_gaps = len(digits) - 1
    op_patterns = normalize_op_patterns(operations, num_gaps)
    if mod_n is not None and mod_n < 1:
        raise ValueError("mod_n must be >= 1 when set.")

    ranges = [range(10**int(d)) for d in digits]
    pairs: List[Tuple[str, int]] = []
    for nums in itertools.product(*ranges):
        nlist = list(nums)
        op_syms = list(random.choice(op_patterns))
        inner = _eval_pemdas_plus_mul(nlist, op_syms)
        ans = inner % mod_n if mod_n is not None else inner
        pairs.append((_format_chain_prompt(nlist, op_syms, mod_n, spaces=spaces), ans))
    return pairs


def _sample_chain_pairs(
    digits: Sequence[int],
    operations: Sequence[Sequence[str]],
    mod_n: Optional[int],
    samples: int,
    shuffle: bool,
    *,
    spaces: bool = True,
) -> List[Tuple[str, int]]:
    """Draw ``samples`` unique random problems without enumerating the full Cartesian product."""
    if samples < 1:
        raise ValueError("samples must be >= 1.")
    if len(digits) < 2:
        raise ValueError("digits must list at least two operand widths.")
    if any(d < 1 for d in digits):
        raise ValueError("each entry in digits must be >= 1.")
    num_gaps = len(digits) - 1
    op_patterns = normalize_op_patterns(operations, num_gaps)
    if mod_n is not None and mod_n < 1:
        raise ValueError("mod_n must be >= 1 when set.")

    selected: List[Tuple[str, int]] = []
    seen = set()
    cap = max(1_000_000, samples * 10_000)
    for _ in range(cap):
        if len(selected) >= samples:
            break
        nlist = [random.randrange(10**int(d)) for d in digits]
        op_syms = list(random.choice(op_patterns))
        inner = _eval_pemdas_plus_mul(nlist, op_syms)
        ans = inner % mod_n if mod_n is not None else inner
        prompt = _format_chain_prompt(nlist, op_syms, mod_n, spaces=spaces)
        key = (prompt, ans)
        if key in seen:
            continue
        seen.add(key)
        selected.append((prompt, ans))
    if len(selected) < samples:
        raise ValueError(
            f"Could only draw {len(selected)} unique problems (requested {samples}); "
            "the search space may be too small for that count.",
        )
    if shuffle:
        random.shuffle(selected)
    return selected


def generate_math_dataset(
    dataset_fname: str,
    tokenizer,
    *,
    digits: List[int],
    operations: List[List[str]],
    mod_n: Optional[int] = None,
    shuffle: bool = True,
    samples: Optional[int] = None,
    split_test_frac: Optional[float] = None,
    datasets_dir: Optional[str] = None,
    spaces: bool = True,
) -> None:
    """Build a math JSON dataset ``{{q_str, a_str, ids}}`` compatible with the rest of the repo.

    Chained prompts: operand ``i`` uses ``digits[i]`` decimal digits (values ``0 .. 10**digits[i]-1``).
    ``operations`` is a list of rows; each row is one allowed operator ordering (length
    ``len(digits)-1``). For each problem, one row is chosen uniformly at random.
    Evaluation uses **PEMDAS** for ``+`` and ``*`` (or ``×``): multiplication before addition.
    If ``mod_n`` is set, the prompt ends with ``... mod n = `` and the answer is
    ``(PEMDAS value) % mod_n``.

    Args:
        dataset_fname: Primary output path. If the path has no directory (e.g. ``2d_add_all.json``),
            it is written under the repo ``datasets/`` directory (or ``datasets_dir`` if given).
            If ``split_test_frac`` is set, must end with ``_all.json``; train/test files are written
            alongside using ``_train.json`` / ``_test.json`` suffixes.
        tokenizer: HuggingFace tokenizer (``encode`` for ``ids``).
        digits: Length ``>= 2``: decimal width per operand.
        operations: Non-empty list of operator rows; each row has one ``+`` / ``*`` / ``×`` per gap
            between operands (or a single op broadcast to every gap).
        mod_n: Optional modulus appended before ``= ``; gold answer uses ``% mod_n``.
        shuffle: Shuffle the final row list (after sampling or after full enumeration).
        samples: If set and smaller than the number of generated pairs, keep a random subset
            of exactly this many (without replacement).
        split_test_frac: If set (e.g. ``0.2``), write ``dataset_fname`` as the full set, then
            write train/test JSON by splitting the shuffled list (first ``1-fraction`` train).
        datasets_dir: Optional root for bare ``dataset_fname``; defaults to repo ``datasets/``.
        spaces: If True (default), prompts use spaces around operators, e.g. ``12 + 34 = ``.
            If False, tight formatting, e.g. ``12+34=``.
    """
    dataset_fname = _resolve_dataset_output_path(dataset_fname, datasets_dir)
    if samples is not None:
        selected = _sample_chain_pairs(
            digits, operations, mod_n, samples, shuffle, spaces=spaces,
        )
    else:
        selected = _iter_chain_pairs(digits, operations, mod_n, spaces=spaces)
        if shuffle:
            random.shuffle(selected)

    rows: List[Dict] = []
    for prompt, answer in selected:
        q_str = prompt
        a_str = str(answer)
        ids = tokenizer.encode(q_str + a_str, add_special_tokens=False)
        rows.append({"q_str": q_str, "a_str": a_str, "ids": ids})

    out_dir = os.path.dirname(os.path.abspath(dataset_fname))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    def _write(path: str, data: List[Dict]) -> None:
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    _write(dataset_fname, rows)

    if split_test_frac is not None:
        if not (0.0 < split_test_frac < 1.0):
            raise ValueError("split_test_frac must be in (0, 1).")
        if not dataset_fname.endswith("_all.json"):
            raise ValueError(
                "split_test_frac requires dataset_fname to end with '_all.json' "
                "(e.g. datasets/2d_add_all.json).",
            )
        n = len(rows)
        split_i = int(n * (1.0 - split_test_frac))
        train_path = dataset_fname.replace("_all.json", f"{DATASET_TRAIN_SUFFIX}.json")
        test_path = dataset_fname.replace("_all.json", f"{DATASET_TEST_SUFFIX}.json")
        _write(train_path, rows[:split_i])
        _write(test_path, rows[split_i:])


def _rand_expr_number(allow_decimals: bool, max_int: int) -> str:
    if allow_decimals and random.random() < 0.45:
        return f"{random.randint(0, max_int)}.{random.randint(0, 9)}"
    return str(random.randint(0, max_int))


def _safe_eval_bin(a: float, op: str, b: float) -> Optional[float]:
    try:
        if op == "+":
            out = a + b
        elif op == "-":
            out = a - b
        elif op == "*":
            out = a * b
        elif op == "/":
            if abs(b) < 1e-12:
                return None
            out = a / b
        else:
            return None
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _fmt_expr_answer(v: float) -> str:
    if abs(v - round(v)) < 1e-10:
        return str(int(round(v)))
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


def generate_expression_json_dataset(
    dataset_fname: str,
    tokenizer,
    *,
    n_examples: int = 1000,
    max_depth: int = 2,
    allow_decimals: bool = False,
    max_int: int = 999,
    seed: Optional[int] = None,
    ensure_diversity: bool = True,
    allowed_ops: str = "+-*/",
    max_parentheses_pairs: int = 1,
    require_integer_answers: bool = True,
    max_answer_abs: int = 10000,
    min_parentheses_frac: float = 0.30,
    split_test_frac: Optional[float] = None,
    datasets_dir: Optional[str] = None,
) -> None:
    """Generate expression dataset rows: ``{q_str, a_str, ids}``.

    ``q_str`` is compact and ends with ``=`` (no spaces, no words), e.g. ``(19+4)*15=``.
    ``a_str`` stores the computed result as a compact numeric string.
    """
    if n_examples < 1:
        raise ValueError("n_examples must be >= 1")
    if max_depth < 1:
        raise ValueError("max_depth must be >= 1")
    if max_int < 1:
        raise ValueError("max_int must be >= 1")
    if max_parentheses_pairs < 0 or max_parentheses_pairs > 1:
        raise ValueError("max_parentheses_pairs must be 0 or 1")
    if max_answer_abs < 1:
        raise ValueError("max_answer_abs must be >= 1")
    if not (0.0 <= min_parentheses_frac <= 1.0):
        raise ValueError("min_parentheses_frac must be in [0, 1]")
    ops = [op for op in allowed_ops if op in {"+", "-", "*", "/"}]
    if not ops:
        raise ValueError("allowed_ops must include at least one of '+', '-', '*', '/'")

    dataset_fname = _resolve_dataset_output_path(dataset_fname, datasets_dir)
    old_state = random.getstate()
    if seed is not None:
        random.seed(seed)

    seen_q = set()
    rows: List[Dict] = []
    template_cycle = ["two_term", "three_term", "paren_left", "paren_right"]
    paren_count = 0

    def _n() -> str:
        # Mix easy + medium + harder magnitudes (not only 3-digit operands).
        r = random.random()
        if r < 0.50:
            lim = min(max_int, 20)
        elif r < 0.85:
            lim = min(max_int, 99)
        else:
            lim = max_int
        return _rand_expr_number(allow_decimals, lim)

    def _sample_expression(template: str) -> Tuple[str, Optional[float]]:
        # single-step: a+b
        if template == "two_term":
            a, b = _n(), _n()
            op = random.choice(ops)
            expr = f"{a}{op}{b}"
            return expr, _safe_eval_bin(float(a), op, float(b))

        # multi-step, no parentheses: a+b*c
        if template == "three_term":
            a, b, c = _n(), _n(), _n()
            op1, op2 = random.choice(ops), random.choice(ops)
            expr = f"{a}{op1}{b}{op2}{c}"
            try:
                return expr, float(eval(expr))
            except Exception:
                return expr, None

        if max_parentheses_pairs == 0:
            return _sample_expression("three_term")

        # one parenthesis pair max: (a+b)*c
        if template == "paren_left":
            a, b, c = _n(), _n(), _n()
            op1, op2 = random.choice(ops), random.choice(ops)
            expr = f"({a}{op1}{b}){op2}{c}"
            try:
                return expr, float(eval(expr))
            except Exception:
                return expr, None

        # one parenthesis pair max: a*(b+c)
        a, b, c = _n(), _n(), _n()
        op1, op2 = random.choice(ops), random.choice(ops)
        expr = f"{a}{op1}({b}{op2}{c})"
        try:
            return expr, float(eval(expr))
        except Exception:
            return expr, None

    max_attempts = max(20_000, n_examples * 120)
    attempts = 0
    while len(rows) < n_examples and attempts < max_attempts:
        attempts += 1
        if max_depth <= 1:
            tpl = "two_term"
        else:
            candidates = ["two_term", "three_term"]
            if max_parentheses_pairs > 0:
                candidates += ["paren_left", "paren_right"]
            tpl = random.choice(candidates)
            if ensure_diversity and len(rows) < len(template_cycle):
                tpl = template_cycle[len(rows)]
            # Ensure we actually get parenthesized forms.
            target_paren = int(min_parentheses_frac * max(1, n_examples))
            if max_parentheses_pairs > 0 and paren_count < target_paren and random.random() < 0.6:
                tpl = random.choice(["paren_left", "paren_right"])
        e, val = _sample_expression(tpl)
        if val is None:
            continue
        q = f"{e}="
        if q in seen_q:
            continue
        if require_integer_answers and abs(val - round(val)) > 1e-10:
            continue
        if abs(val) > max_answer_abs:
            continue
        a = _fmt_expr_answer(val)
        ids = tokenizer.encode(q + a, add_special_tokens=False)
        rows.append({"q_str": q, "a_str": a, "ids": ids})
        seen_q.add(q)
        if "(" in q:
            paren_count += 1

    if len(rows) < n_examples:
        raise ValueError(f"Could only generate {len(rows)} unique expressions; requested {n_examples}.")

    def _write(path: str, data: List[Dict]) -> None:
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    _write(dataset_fname, rows)

    if split_test_frac is not None:
        if not (0.0 < split_test_frac < 1.0):
            raise ValueError("split_test_frac must be in (0, 1).")
        if not dataset_fname.endswith("_all.json"):
            raise ValueError(
                "split_test_frac requires dataset_fname to end with '_all.json' "
                "(e.g. datasets/svamp_style_expr_all.json).",
            )
        split_i = int(len(rows) * (1.0 - split_test_frac))
        train_path = dataset_fname.replace("_all.json", f"{DATASET_TRAIN_SUFFIX}.json")
        test_path = dataset_fname.replace("_all.json", f"{DATASET_TEST_SUFFIX}.json")
        _write(train_path, rows[:split_i])
        _write(test_path, rows[split_i:])

    if seed is not None:
        random.setstate(old_state)


def _resolve_ckpt_path(checkpoint: str) -> str:
    """
    Resolve a checkpoint spec to a local .pt file.

    Accepted forms:
    - absolute/relative filepath to a .pt file
    - "latest"
    - "1500" / "epoch_1500" / "epoch_1500.pt"
    """
    if os.path.exists(checkpoint):
        return checkpoint

    ckpt_root = CIRCUIT_DISCOVERY_CKPT_DIR or os.path.join(os.path.dirname(__file__), "..", "results", "circuit-discovery", "checkpoints")
    ckpt_root = os.path.abspath(ckpt_root)

    if checkpoint == "latest":
        cand = glob.glob(os.path.join(ckpt_root, "epoch_*.pt"))
        if not cand:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_root}")
        def _epoch_num(p: str) -> int:
            m = re.search(r"epoch_(\d+)\.pt$", os.path.basename(p))
            return int(m.group(1)) if m else -1
        return max(cand, key=_epoch_num)

    m = re.search(r"(\d+)", checkpoint)
    if m:
        epoch = int(m.group(1))
        cand = os.path.join(ckpt_root, f"epoch_{epoch}.pt")
        if os.path.exists(cand):
            return cand

    raise FileNotFoundError(
        f"Could not resolve checkpoint '{checkpoint}'. "
        f"Provide a path to a .pt file, 'latest', or an epoch like '1500'. "
        f"Looked in {ckpt_root}."
    )


def _extract_circuit_model_state_dict(ckpt_data, ckpt_path: str):
    """Resolve model weights from a .pt file saved in different layouts."""
    if not isinstance(ckpt_data, dict):
        raise TypeError(
            f"Checkpoint {ckpt_path!r} must load to a dict, got {type(ckpt_data)}."
        )
    for key in ("model_state_dict", "state_dict"):
        if key in ckpt_data:
            return ckpt_data[key]
    # torch.save(model.state_dict(), path) — weights only, no wrapper dict
    if any(k.startswith("classifier.") for k in ckpt_data.keys()):
        return ckpt_data
    # Common mistake: neuron cluster / feature files
    if "features_per_subclass" in ckpt_data or "cluster_to_indices" in ckpt_data:
        raise ValueError(
            f"{ckpt_path!r} is a neuron-cluster or feature file, not a circuit-discovery "
            "checkpoint. Pass the circuit training checkpoint (e.g. epoch_*.pt with "
            "model weights from circuit_discovery), not k*.pt under clusters/."
        )
    keys_preview = list(ckpt_data.keys())[:12]
    extra = "..." if len(ckpt_data) > 12 else ""
    raise ValueError(
        "No model weights found: expected keys 'model_state_dict' or 'state_dict', "
        "or a raw state_dict with 'classifier.*' keys (from circuit discovery). "
        f"File has keys: {keys_preview}{extra}"
    )


def load_model_checkpoint(checkpoint, k_classes, lr):
    from circuit_discovery.models import CircuitDiscoveryModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = None
    try:
        ckpt_path = _resolve_ckpt_path(checkpoint)
    except FileNotFoundError:
        ckpt_path = None

    if ckpt_path is None:
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint!r}. "
            "Provide a valid path to a .pt file, 'latest', or an epoch number."
        )

    ckpt_data = torch.load(ckpt_path, map_location=device)

    # Auto-detect k_classes from checkpoint weights
    state = _extract_circuit_model_state_dict(ckpt_data, ckpt_path)
    ckpt_k = None
    if "classifier.classifier.4.weight" in state:
        ckpt_k = state["classifier.classifier.4.weight"].shape[0]

    if ckpt_k is not None and ckpt_k != k_classes:
        raise RuntimeError(
            f"Checkpoint was trained with k_classes={ckpt_k} but you "
            f"requested k_classes={k_classes}. Use a checkpoint that "
            f"matches your experiment, or pass --k-classes {ckpt_k}.\n"
            f"  Checkpoint: {ckpt_path}"
        )

    model = CircuitDiscoveryModel(k_classes=k_classes, mask_temperature=1.0).to(device)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        print(
            "Warning: checkpoint missing keys (using model defaults):",
            incompatible.missing_keys,
        )
    if incompatible.unexpected_keys:
        print("Warning: checkpoint unexpected keys ignored:", incompatible.unexpected_keys)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    opt_state = ckpt_data.get("optimizer_state_dict")
    if opt_state is not None:
        optimizer.load_state_dict(opt_state)

    epoch = ckpt_data.get("epoch", ckpt_data.get("step", 0))
    metrics_log = ckpt_data.get("metrics_log", [])
    return model, optimizer, metrics_log, epoch

def _stack_layer_activations(batch_activations):
    if not batch_activations:
        raise ValueError("batch_activations is empty")

    layers = sorted(batch_activations.keys())
    tensors = [batch_activations[i] for i in layers]
    return torch.cat(tensors, dim=-1)


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

if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = patch_tokenizer_no_special_tokens(
        AutoTokenizer.from_pretrained(LLAMA_1B_MODEL_NAME),
    )

    generate_math_dataset(
        dataset_all_json_path("2222_add"),
        tokenizer,
        digits=[2, 2, 2, 2],
        operations=[["+", "+", "+"]],
        spaces=True,
        shuffle=True,
        samples=10000,
        split_test_frac=0.5,
    )

    # mix_datasets(
    #     dataset_stems=["22_add_tight", "222_add_tight"],
    #     output_stem="2-3_add",
    #     datasets_dir=default_datasets_dir(),
    #     shuffle=True,
    # )

import random
import json
import re
import os
import sys
import glob
import torch
from typing import List, Optional, Tuple

try:
    import constants as _constants
    HF_TOKEN = getattr(_constants, "HF_TOKEN", "")
    CIRCUIT_DISCOVERY_CKPT_DIR = getattr(_constants, "CIRCUIT_DISCOVERY_CKPT_DIR", "")
except ModuleNotFoundError:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    CIRCUIT_DISCOVERY_CKPT_DIR = os.environ.get("CIRCUIT_DISCOVERY_CKPT_DIR", "")

from transformers.utils import logging as hf_logging
from transformers import AutoTokenizer

logged_in = False

# Prompts end with "=" and all 2-3 digit answers (20-198) are single tokens in Llama-3 BPE.
EVAL_MAX_NEW_TOKENS = 1

# --- Dataset file naming: ``{prefix}_train_80.json``, ``{prefix}_test_20.json``, ``{prefix}_all.json`` ---
DATASET_TRAIN_SUFFIX = "_train_80"
DATASET_TEST_SUFFIX = "_test_20"
DATASET_ALL_SUFFIX = "_all.json"  # ``{prefix}_all.json``


def default_datasets_dir() -> str:
    """Absolute path to the repo ``datasets/`` directory (sibling of ``src/``)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))


def dataset_train_json_path(prefix: str, datasets_dir: Optional[str] = None) -> str:
    d = os.path.abspath(datasets_dir) if datasets_dir else default_datasets_dir()
    return os.path.join(d, f"{prefix}{DATASET_TRAIN_SUFFIX}.json")


def dataset_test_json_path(prefix: str, datasets_dir: Optional[str] = None) -> str:
    d = os.path.abspath(datasets_dir) if datasets_dir else default_datasets_dir()
    return os.path.join(d, f"{prefix}{DATASET_TEST_SUFFIX}.json")


def dataset_all_json_path(prefix: str, datasets_dir: Optional[str] = None) -> str:
    d = os.path.abspath(datasets_dir) if datasets_dir else default_datasets_dir()
    return os.path.join(d, f"{prefix}{DATASET_ALL_SUFFIX}")


def list_dataset_prefixes(datasets_dir: Optional[str] = None) -> List[str]:
    """Prefixes found from existing ``*_train_80.json`` files under ``datasets_dir``."""
    d = os.path.abspath(datasets_dir) if datasets_dir else default_datasets_dir()
    if not os.path.isdir(d):
        return []
    pat = os.path.join(d, f"*{DATASET_TRAIN_SUFFIX}.json")
    out: List[str] = []
    for p in glob.glob(pat):
        base = os.path.basename(p)
        suf = f"{DATASET_TRAIN_SUFFIX}.json"
        if base.endswith(suf):
            out.append(base[: -len(suf)])
    return sorted(set(out))


def prompt_dataset_prefix(datasets_dir: Optional[str] = None) -> str:
    """Interactive prompt when ``--dataset`` is omitted (requires a TTY)."""
    d = os.path.abspath(datasets_dir) if datasets_dir else default_datasets_dir()
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Datasets directory not found: {d}")
    if not sys.stdin.isatty():
        raise SystemExit(
            "No dataset specified. Pass --dataset PREFIX (e.g. 2d_add). "
            f"Expected files: <prefix>{DATASET_TRAIN_SUFFIX}.json and <prefix>{DATASET_TEST_SUFFIX}.json under {d}."
        )
    choices = list_dataset_prefixes(d)
    print(f"Datasets directory: {d}")
    if choices:
        print("Available prefixes (from existing *_train_80.json files):")
        for c in choices:
            print(f"  {c}")
    else:
        print("No *_train_80.json files found in that directory; enter a prefix anyway (e.g. 2d_add).")
    prefix = input("Enter dataset prefix: ").strip()
    if not prefix:
        raise SystemExit("Empty dataset prefix.")
    return prefix


def resolve_train_test_paths(
    *,
    dataset: Optional[str],
    datasets_dir: Optional[str],
) -> Tuple[str, str, str]:
    """Resolve train/test JSON paths from ``--dataset PREFIX`` (or interactive prompt).

    Returns:
        ``(train_path, test_path, prefix)`` with absolute paths.
    """
    d = os.path.abspath(datasets_dir) if datasets_dir else default_datasets_dir()
    prefix = (dataset or "").strip() or prompt_dataset_prefix(d)
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
    d = os.path.abspath(datasets_dir) if datasets_dir else default_datasets_dir()
    prefix = (dataset or "").strip() or prompt_dataset_prefix(d)
    p = dataset_test_json_path(prefix, d)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Test file not found for prefix {prefix!r}: {p}")
    return p, prefix


def resolve_ablation_all_path(
    *,
    dataset: Optional[str],
    ablation_path: Optional[str],
    datasets_dir: Optional[str],
    prefix: Optional[str] = None,
) -> str:
    """Full ``*_all.json`` for ablation: ``--ablation-dataset`` path, or ``{prefix}_all.json``."""
    if ablation_path:
        return os.path.abspath(ablation_path)
    d = os.path.abspath(datasets_dir) if datasets_dir else default_datasets_dir()
    pre = (prefix or "").strip() or (dataset or "").strip() or prompt_dataset_prefix(d)
    p = dataset_all_json_path(pre, d)
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"Ablation/all-in-one dataset not found for prefix {pre!r}: {p}",
        )
    return p


def load_model(model_name):

    hf_logging.set_verbosity_error()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import login

    global logged_in
    if not logged_in:
        login(HF_TOKEN)
        logged_in = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_size = 'left'
    return model, tokenizer

def test_model(model, tokenizer, dataset_fname, results_fname, batch_size=50, max_new_tokens=EVAL_MAX_NEW_TOKENS, log=True):
    model.eval()
    with open(dataset_fname, 'r') as f:
        dataset = json.load(f)
    prompts = []
    for s in dataset:
        prompts.append(s['q_str'])
    results = []
    for i in range(0, len(prompts), batch_size):
        with torch.no_grad():
            if log:
                print(f'processing {i}/{len(prompts)}')
            batched_prompts = prompts[i:min(i + batch_size, len(prompts))]   
            input_ids = tokenizer(batched_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            outputs = model.generate(**input_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for k, resp in enumerate(responses):
                results.append({'response': resp, 'answer': dataset[i + k]['a_str']})

    with open(results_fname, 'w') as f:
        json.dump(results, f, indent=4)

    return results

def parse_answer(resp):
    match = re.search(r'=\s*(\d+)', resp)
    return int(match.group(1)) if match else None

def eval_model(results_fname):
    with open(results_fname, 'r') as f:
        results = json.load(f)

    correct = 0
    for res in results:
        if parse_answer(res['response']) == int(res['answer']):
            correct += 1

    print('Accuracy: ', correct / len(results))
    return correct / len(results)

# samples=None means all 2-digit addition pairs are added; otherwise sample without replacement
def gen_2d_add_dataset(dataset_fname, samples, tokenizer):
    all_pairs = [(f'{num1}+{num2}=', num1 + num2) for num1 in range(100) for num2 in range(100)]

    if samples is None or samples >= len(all_pairs):
        selected = all_pairs
        random.shuffle(selected)
    else:
        selected = random.sample(all_pairs, samples)

    dataset = []
    for prompt, answer in selected:
        q_str = prompt
        a_str = str(answer)
        ids = tokenizer.encode(q_str + a_str, add_special_tokens=False)
        dataset.append(
            {
                "q_str": q_str,
                "a_str": a_str,
                "ids": ids,
            }
        )

    with open(dataset_fname, 'w') as f:
        json.dump(dataset, f, indent=4)

def gen_3d_add_dataset(dataset_fname, samples, tokenizer):
    all_pairs = [(f'{num1}+{num2}=', num1 + num2) for num1 in range(1000) for num2 in range(1000)]

    if samples is None or samples >= len(all_pairs):
        selected = all_pairs
        random.shuffle(selected)
    else:
        selected = random.sample(all_pairs, samples)

    dataset = []
    for prompt, answer in selected:
        q_str = prompt
        a_str = str(answer)
        ids = tokenizer.encode(q_str + a_str, add_special_tokens=False)
        dataset.append(
            {
                "q_str": q_str,
                "a_str": a_str,
                "ids": ids,
            }
        )

    with open(dataset_fname, 'w') as f:
        json.dump(dataset, f, indent=4)

def gen_2d1d_mult_dataset(dataset_fname, samples, tokenizer):
    all_pairs = [(f'{num1}*{num2}=', num1 * num2) for num1 in range(100) for num2 in range(10)]
    all_pairs += [(f'{num1}*{num2}=', num1 * num2) for num1 in range(10) for num2 in range(100)]

    if samples is None or samples >= len(all_pairs):
        selected = all_pairs
        random.shuffle(selected)
    else:
        selected = random.sample(all_pairs, samples)

    dataset = []
    for prompt, answer in selected:
        q_str = prompt
        a_str = str(answer)
        ids = tokenizer.encode(q_str + a_str, add_special_tokens=False)
        dataset.append(
            {
                "q_str": q_str,
                "a_str": a_str,
                "ids": ids,
            }
        )

    with open(dataset_fname, 'w') as f:
        json.dump(dataset, f, indent=4)

def gen_mix_dataset(dataset_fname, files):
    dataset = []
    for file in files:
        with open(file, 'r') as f:
            dataset.extend(json.load(f))
    
    random.shuffle(dataset)
    
    with open(dataset_fname, 'w') as f:
        json.dump(dataset, f, indent=4)

def split_dataset(dataset_fname, test_frac=0.1):
    with open(dataset_fname, 'r') as f:
        dataset = json.load(f)

    split = int(len(dataset) * (1 - test_frac))
    train = dataset[:split]
    test = dataset[split:]
    
    with open(f"{dataset_fname.replace('_all.json', f'_train_{100 - int(test_frac * 100)}.json')}", 'w') as f:
        json.dump(train, f, indent=4)
    with open(f"{dataset_fname.replace('_all.json', f'_test_{int(test_frac * 100)}.json')}", 'w') as f:
        json.dump(test, f, indent=4)

def _safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")

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

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    # gen_2d1d_mult_dataset("datasets/2d1d_mult_all.json", None, tokenizer) 
    # gen_mix_dataset("datasets/add_mult_all.json", ["datasets/2d_add_all.json", "datasets/2d1d_mult_all.json"])
    # split_dataset("datasets/2d1d_mult_all.json", test_frac=0.2)
    split_dataset("datasets/2d_add_all.json", test_frac=0.2)
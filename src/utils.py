import random
import json
import re
import os
import glob
import torch

try:
    # Preferred: repo-local constants (in the same directory as this file).
    from constants import (  # type: ignore
        HF_TOKEN,
        CIRCUIT_DISCOVERY_CKPT_DIR,
        ACTIVATIONS_DIR,
        BUCKET_NAME,
        USE_S3,
    )
except ModuleNotFoundError:
    # Colab/back-compat fallback if `constants.py` isn't present in the checkout.
    HF_TOKEN = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGINGFACE_TOKEN", "")
    BUCKET_NAME = os.environ.get("S3_BUCKET", "circuit-distillation")
    USE_S3 = os.environ.get("USE_S3", "0") == "1"
    CIRCUIT_DISCOVERY_CKPT_DIR = os.environ.get("CIRCUIT_DISCOVERY_CKPT_DIR", "")
    ACTIVATIONS_DIR = os.environ.get("ACTIVATIONS_DIR", "")
from transformers.utils import logging as hf_logging

try:
    import boto3  # type: ignore
except Exception:  # pragma: no cover
    boto3 = None

def _get_s3_client():
    if boto3 is None:
        raise ImportError("boto3 is not installed. Install boto3 or disable S3 mode (USE_S3=0).")
    return boto3.client("s3")

logged_in = False

def get_model_name(argv):
    if len(argv) > 1:
        return argv[1]
    else:
        print('Please provide model name')
        exit()

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

def test_model(model, tokenizer, dataset_fname, results_fname, batch_size=50, max_new_tokens=5, log=True):
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

def _safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


def list_keys(model_name: str, activations_dir: str | None = None):
    """
    Local-first listing of activation batch files for `model_name`.

    - If `USE_S3=1`, returns legacy S3 keys under `mlp_activations/{model_name}/`.
    - Otherwise returns local filepaths under `results/activations/` (or `ACTIVATIONS_DIR`).

    Returns a list of filepaths to activation batch .pt files for `model_name`.
    We keep the old name `list_keys` so existing code keeps working.
    """
    if USE_S3:
        s3 = _get_s3_client()
        prefix = f"mlp_activations/{model_name}/"
        keys = []
        token = None
        while True:
            kwargs = {"Bucket": BUCKET_NAME, "Prefix": prefix}
            if token is not None:
                kwargs["ContinuationToken"] = token
            resp = s3.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []):
                keys.append(obj["Key"])
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        return keys

    safe = _safe_model_name(model_name)
    base_dir = activations_dir or (ACTIVATIONS_DIR or os.path.join(os.path.dirname(__file__), "..", "results", "activations"))
    base_dir = os.path.abspath(base_dir)

    patterns = [
        # New local convention (recommended): results/activations/activations_<safe>_<batch>.pt
        os.path.join(base_dir, f"activations_{safe}_*.pt"),
        # Back-compat: results/activations/<safe>/activations_<safe>_<batch>.pt
        os.path.join(base_dir, safe, f"activations_{safe}_*.pt"),
        # Looser match if user saved elsewhere under base_dir
        os.path.join(base_dir, "**", f"activations_{safe}_*.pt"),
    ]

    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))

    # De-dup + stable sort
    files = sorted(set(files))
    return files

def suffix_map(keys):
    # Works for both local filepaths and S3 keys
    return {str(k).split("/")[-1]: k for k in keys}

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


def load_model_checkpoint(checkpoint, k_classes, lr):
    from circuit_discovery.models import CircuitDiscoveryModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = None
    try:
        ckpt_path = _resolve_ckpt_path(checkpoint)
    except FileNotFoundError:
        ckpt_path = None

    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location=device)
    else:
        # Legacy fallback: allow loading from S3 if enabled.
        if not USE_S3:
            raise
        s3 = _get_s3_client()
        import io
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=f"circuit-discovery/{checkpoint}")
        bytestream = io.BytesIO(obj["Body"].read())
        checkpoint = torch.load(bytestream, map_location=device)

    model = CircuitDiscoveryModel(k_classes=k_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", checkpoint.get("step", 0))
    metrics_log = checkpoint.get("metrics_log", [])
    return model, optimizer, metrics_log, epoch

def _stack_layer_activations(batch_activations):
    if not batch_activations:
        raise ValueError("batch_activations is empty")

    layers = sorted(batch_activations.keys())
    tensors = [batch_activations[i] for i in layers]
    return torch.cat(tensors, dim=-1)

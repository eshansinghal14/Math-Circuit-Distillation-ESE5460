import random
import json
import re
import os
import glob
import torch

import constants as _constants  # type: ignore
HF_TOKEN = getattr(_constants, "HF_TOKEN", "")
CIRCUIT_DISCOVERY_CKPT_DIR = getattr(_constants, "CIRCUIT_DISCOVERY_CKPT_DIR", "")

from transformers.utils import logging as hf_logging
from transformers import AutoTokenizer

logged_in = False

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

    checkpoint = torch.load(ckpt_path, map_location=device)

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

if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    # gen_2d1d_mult_dataset("datasets/2d1d_mult_all.json", None, tokenizer) 
    # gen_mix_dataset("datasets/add_mult_all.json", ["datasets/2d_add_all.json", "datasets/2d1d_mult_all.json"])
    split_dataset("datasets/2d1d_mult_all.json", test_frac=0.2)
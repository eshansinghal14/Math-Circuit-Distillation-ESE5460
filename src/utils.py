import random
import json
import re
import torch
import boto3
import io

from constants import HF_TOKEN, BUCKET_NAME
from transformers.utils import logging as hf_logging

s3 = boto3.client("s3")
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

def list_keys(model_name):
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

def suffix_map(keys):
    return {k.split("/")[-1]: k for k in keys}

def load_model_checkpoint(model_name, k_classes, lr):
    from circuit_discovery.models import CircuitDiscoveryModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=f"circuit-discovery/{model_name}")
    bytestream = io.BytesIO(obj["Body"].read())

    checkpoint = torch.load(bytestream, map_location=device)

    model = CircuitDiscoveryModel(k_classes=k_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer, checkpoint["metrics_log"], checkpoint["epoch"]

def _stack_layer_activations(batch_activations):
    if not batch_activations:
        raise ValueError("batch_activations is empty")

    layers = sorted(batch_activations.keys())
    tensors = [batch_activations[i] for i in layers]
    return torch.cat(tensors, dim=-1)

import random
import json
import re
import torch
import boto3
import io

from circuit_discovery.models import CircuitDiscoveryModel
from constants import HF_TOKEN, BUCKET_NAME

s3 = boto3.client("s3")

def get_model_name(argv):
    if len(argv) > 1:
        return argv[1]
    else:
        print('Please provide model name')
        exit()

def load_model(argv):
    model_name = get_model_name(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import login

    login(HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_size = 'left'
    return model, tokenizer

def test_model(model, tokenizer, dataset_fname, results_fname, batch_size=50):
    model.eval()
    with open(dataset_fname, 'r') as f:
        dataset = json.load(f)
    prompts = list(dataset.keys())
    results = []
    for i in range(0, len(prompts), batch_size):
        with torch.no_grad():
            print(f'processing {i}/{len(prompts)}')
            batched_prompts = prompts[i:i + batch_size]   
            input_ids = tokenizer(batched_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            outputs = model.generate(**input_ids, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for k, resp in enumerate(responses):
                results.append({'response': resp, 'answer': dataset[batched_prompts[k]]})

    with open(results_fname, 'w') as f:
        json.dump(results, f, indent=4)

def parse_answer(resp):
    match = re.search(r'=\s*(\d+)', resp)
    return int(match.group(1)) if match else None

def eval_model(results_fname):
    with open(results_fname, 'r') as f:
        results = json.load(f)

    correct = 0
    for res in results:
        if parse_answer(res['response']) == res['answer']:
            correct += 1

    print('Accuracy: ', correct / len(results))

# samples=None means all 2-digit addition pairs are added; otherwise sample without replacement
def gen_2d_add_dataset(dataset_fname, samples):
    all_pairs = [(f'{num1}+{num2}=', num1 + num2) for num1 in range(100) for num2 in range(100)]

    if samples is None or samples >= len(all_pairs):
        selected = all_pairs
        random.shuffle(selected)
    else:
        selected = random.sample(all_pairs, samples)

    dataset = {prompt: answer for prompt, answer in selected}

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

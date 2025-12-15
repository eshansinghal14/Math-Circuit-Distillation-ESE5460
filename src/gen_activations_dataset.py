import json
import sys
from collections import deque
import torch
import boto3
from utils import load_model, parse_answer
from constants import BUCKET_NAME

model, tokenizer = load_model(sys.argv)
model_name = sys.argv[1]
model.eval()

s3 = boto3.client('s3')

with open('../datasets/2d_add_all.json', 'r') as f:
    dataset = json.load(f)

with open('../datasets/2d_add_test.json', 'r') as f:
    test_dataset = json.load(f)

inputs = []
prompts = list(dataset.keys())
answers = list(dataset.values())
for i, prompt in enumerate(prompts):
    ans = str(answers[i])
    for j in range(1, len(ans) + 1):
        inputs.append(prompt + ans[:j])

input_ids = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).input_ids.to(model.device)

# Prepare containers for activations: layer_idx -> list of [batch, hidden]
layer_activations = {}

# Set up hooks on each MLP layer (Llama-style: model.model.layers[i].mlp)
handles = []

def make_hook(layer_idx):
    def hook(module, inputs, output):
        # output: [batch, seq_len, hidden_size] (for LlamaMLP)
        # We store the last token's activations
        activ = output[:, :, :].detach().cpu()
        layer_activations.setdefault(layer_idx, []).append(activ)
    return hook

# Register hooks
for i, layer in enumerate(model.model.layers):
    h = layer.mlp.register_forward_hook(make_hook(i))
    handles.append(h)

# Iterate through prompts in batches and run deterministic generation,
# saving each batch's activations to a separate file
batch_size = 50
with torch.no_grad():
    for i in range(0, len(inputs), batch_size):
        batch_inputs = input_ids[i: i + min(batch_size, len(inputs) - i)]

        print(f'processing batch {i}/{len(inputs)}')

        layer_activations = {}

        # Teacher-forced forward pass on prefixes
        _ = model(input_ids=batch_inputs)

        # Stack activations per layer for this batch: [batch_size, hidden]
        batch_activations = {}
        for layer_idx, chunks in layer_activations.items():
            batch_activations[layer_idx] = torch.cat(chunks, dim=0)

        activations = {
            'prompts': inputs[i: i + min(batch_size, len(inputs) - i)],
            'activations': batch_activations,
        }
        torch.save(activations, '/tmp/activations.pt')
        s3.upload_file('/tmp/activations.pt', BUCKET_NAME, f'mlp_activations/{model_name}/{i}_{len(inputs)}.pt')

# Remove hooks
for h in handles:
    h.remove()

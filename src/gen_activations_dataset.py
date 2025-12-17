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

ids = []
for record in dataset:
    ids.append(record['ids'])
ids = torch.tensor(ids).to(model.device)

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
    h = layer.mlp.up_proj.register_forward_hook(make_hook(i))
    handles.append(h)

# Iterate through prompts in batches and run deterministic generation,
# saving each batch's activations to a separate file
batch_size = 50
with torch.no_grad():
    for i in range(0, ids.shape[0], batch_size):
        batch_inputs = ids[i: i + min(batch_size, ids.shape[0] - i)]

        print(f'processing batch {i}/{ids.shape[0]}')

        layer_activations = {}

        # Teacher-forced forward pass on prefixes
        _ = model(input_ids=batch_inputs)

        # Stack activations per layer for this batch: [batch_size, hidden]
        batch_activations = {}
        for layer_idx, chunks in layer_activations.items():
            batch_activations[layer_idx] = torch.cat(chunks, dim=0)
        
        activations = {
            'ids': ids[i: i + min(batch_size, ids.shape[0] - i)],
            'activations': batch_activations,
        }
        torch.save(activations, '/tmp/activations.pt')
        s3.upload_file('/tmp/activations.pt', BUCKET_NAME, f'mlp_activations/{model_name}/{i}_{ids.shape[0]}.pt')

# Remove hooks
for h in handles:
    h.remove()

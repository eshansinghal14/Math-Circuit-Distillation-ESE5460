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
layer_activations = {}

handles = []

def make_hook(layer_idx):
    def hook(module, inputs, output):
        activ = output[:, :, :].detach().cpu()
        layer_activations.setdefault(layer_idx, []).append(activ)
    return hook

for i, layer in enumerate(model.model.layers):
    h = layer.mlp.up_proj.register_forward_hook(make_hook(i))
    handles.append(h)

batch_size = 50
with torch.no_grad():
    for i in range(0, ids.shape[0], batch_size):
        batch_inputs = ids[i: i + min(batch_size, ids.shape[0] - i)]

        print(f'processing batch {i}/{ids.shape[0]}')

        layer_activations = {}

        _ = model(input_ids=batch_inputs)

        batch_activations = {}
        for layer_idx, chunks in layer_activations.items():
            batch_activations[layer_idx] = torch.cat(chunks, dim=0)
        
        activations = {
            'ids': ids[i: i + min(batch_size, ids.shape[0] - i)],
            'activations': batch_activations,
        }
        torch.save(activations, '/tmp/activations.pt')
        s3.upload_file('/tmp/activations.pt', BUCKET_NAME, f'mlp_activations/{model_name}/{i}_{ids.shape[0]}.pt')

for h in handles:
    h.remove()

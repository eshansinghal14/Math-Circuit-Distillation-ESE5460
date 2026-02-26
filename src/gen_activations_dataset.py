import json
import torch
from utils import load_model

class NeuronActivationsGenerator:

    def __init__(self, model_name, batch_size=50):
        self.model, self.tokenizer = load_model(model_name)
        self.model.eval()
        self.model_name = model_name
        self.batch_size = batch_size

        with open('datasets/2d_add_all.json', 'r') as f:
            dataset = json.load(f)

        ids = []
        for record in dataset:
            ids.append(record['ids'])
        self.ids = torch.tensor(ids).to(self.model.device)
        self.layer_activations = {}

        self.handles = []
        for i, layer in enumerate(self.model.model.layers):
            h = layer.mlp.up_proj.register_forward_hook(self.make_hook(i))
            self.handles.append(h)

    def make_hook(self, layer_idx):
        def hook(module, inputs, output):
            activ = output[:, :, :].detach().cpu()
            self.layer_activations.setdefault(layer_idx, []).append(activ)
        return hook
    
    def generate_batch_activations(self, batch, log=True):
        with torch.no_grad():
            start_prob = batch * self.batch_size
            batch_inputs = self.ids[start_prob: start_prob + min(self.batch_size, self.ids.shape[0] - start_prob)]
            
            if log and batch % 100 == 0: 
                print(f'processing batch {batch}/{self.ids.shape[0] // self.batch_size}')

            self.layer_activations = {}
            _ = self.model(input_ids=batch_inputs)
            batch_activations = {}
            for layer_idx, chunks in self.layer_activations.items():
                batch_activations[layer_idx] = torch.cat(chunks, dim=0)
            
            activations = {
                'ids': batch_inputs,
                'activations': batch_activations,
            }
            return activations

    def remove_handles(self):
        for h in self.handles:
            h.remove()
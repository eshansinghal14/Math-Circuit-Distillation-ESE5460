import json
import sys
from collections import deque
import torch
from utils import get_model_name, load_model, parse_answer
import os
from constants import BUCKET_NAME, UPLOAD_ACTIVATIONS_TO_S3

try:
    import boto3  # type: ignore
except Exception:  # pragma: no cover
    boto3 = None

class NeuronActivationsGenerator:

    def __init__(self, model_name, batch_size=50):
        self.model, self.tokenizer = load_model(model_name)
        self.model.eval()
        self.model_name = model_name
        # filesystem-safe model name (avoid slashes in filenames)
        self.safe_model_name = model_name.replace('/', '_').replace(':', '_')
        self.batch_size = batch_size

        with open('../datasets/2d_add_all.json', 'r') as f:
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
            
            if log: 
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

            out_dir = os.environ.get("ACTIVATIONS_DIR", "") or os.path.join(os.path.dirname(__file__), "..", "results", "activations")
            os.makedirs(out_dir, exist_ok=True)
            out_fname = os.path.join(out_dir, f'activations_{self.safe_model_name}_{batch}.pt')
            torch.save(activations, out_fname)
            return out_fname

    def generate_all_activations(self):
        num_batches = (self.ids.shape[0] + self.batch_size - 1) // self.batch_size
        out_files = []
        s3 = None
        if UPLOAD_ACTIVATIONS_TO_S3:
            if boto3 is None:
                raise ImportError("UPLOAD_ACTIVATIONS_TO_S3=1 but boto3 is not installed.")
            s3 = boto3.client("s3")
        for batch in range(num_batches):
            out_fname = self.generate_batch_activations(batch)
            out_files.append(out_fname)
            if UPLOAD_ACTIVATIONS_TO_S3:
                # legacy S3 layout
                key = f"mlp_activations/{self.model_name}/{batch}_{self.ids.shape[0]}.pt"
                s3.upload_file(out_fname, BUCKET_NAME, key)
        return out_files

    def remove_handles(self):
        for h in self.handles:
            h.remove()

if __name__ == "__main__":
    model_name = get_model_name(sys.argv)
    activations_generator = NeuronActivationsGenerator(model_name)
    activations_generator.generate_all_activations()
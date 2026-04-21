import json
import os

import torch
from utils import dataset_test_json_path, load_model

class NeuronActivationsGenerator:

    def __init__(
        self,
        model_name,
        batch_size=50,
        *,
        dataset_prefix: str | None = None,
        res_token: int | None = None,
    ):
        if not dataset_prefix:
            raise ValueError("dataset_prefix is required; no default dataset is assumed")
        self.model, self.tokenizer = load_model(model_name)
        self.model.eval()
        self.model_name = model_name
        self.batch_size = batch_size
        self.res_token = res_token

        dataset_path = dataset_test_json_path(dataset_prefix, None)
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(
                f"Dataset not found for prefix {dataset_prefix!r}: {dataset_path}",
            )
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        ids = []
        for record in dataset:
            if res_token is None:
                ids.append(record["ids"])
                continue
            if "q_str" not in record or "a_str" not in record:
                raise ValueError(
                    "Dataset rows must include q_str and a_str when res_token is set.",
                )
            prompt_ids = self.tokenizer.encode(str(record["q_str"]), add_special_tokens=False)
            answer_ids = self.tokenizer.encode(str(record["a_str"]), add_special_tokens=False)
            if len(answer_ids) < res_token:
                continue
            ids.append(prompt_ids + answer_ids[: max(0, res_token - 1)])
        if not ids:
            raise ValueError(
                f"No dataset rows contain at least {res_token} response tokens."
                if res_token is not None
                else "Dataset is empty.",
            )
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
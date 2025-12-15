import os
import json
import torch
from torch import nn
from torch.nn import functional as F

from transformers import AutoConfig
import boto3
from constants import BUCKET_NAME

s3 = boto3.client('s3')

llama_1b = 'meta-llama/Llama-3.2-1B'
llama_8b = 'meta-llama/Meta-Llama-3-8B'

config = {
    '1b': AutoConfig.from_pretrained(llama_1b),
    '8b': AutoConfig.from_pretrained(llama_8b),
}

def parse_equation(probs, device=None):
    op1_list = []
    op2_list = []
    res_list = []
    term_encodings = []

    for prob in probs:
        add_idx = prob.index('+')
        equal_idx = prob.index('=')
        op1_str = prob[:add_idx]
        op2_str = prob[add_idx + 1 : equal_idx]
        res_str = prob[equal_idx + 1 :]

        op1_list.append(int(op1_str))
        op2_list.append(int(op2_str))
        res_list.append(int(res_str))

        term_encoding = [0] * len(op1_str) + [1] + [2] * len(op2_str) + [3] + [4] * len(res_str)
        term_encodings.append(torch.tensor(term_encoding, dtype=torch.long, device=device))

    op1 = torch.tensor(op1_list, dtype=torch.long, device=device)
    op2 = torch.tensor(op2_list, dtype=torch.long, device=device)
    res = torch.tensor(res_list, dtype=torch.long, device=device)

    # Pad term encodings to a single tensor of shape (batch, max_seq_len)
    term_encoding_padded = nn.utils.rnn.pad_sequence(
        term_encodings, batch_first=True, padding_value=0
    )

    return op1, op2, res, term_encoding_padded

class ProblemEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.op1_emb_layer = nn.Embedding(100, embedding_dim // 4)
        self.op2_emb_layer = nn.Embedding(100, embedding_dim // 4)
        self.sum_emb_layer = nn.Embedding(200, embedding_dim // 2)
        # step_emb_layer = nn.Embedding(3, embedding_dim // 8)

    def forward(self, op1, op2, res):
        op1_emb = self.op1_emb_layer(op1)
        op2_emb = self.op2_emb_layer(op2)
        sum_emb = self.sum_emb_layer(res)
        # step_emb = self.step_emb_layer(len(prob - equal_idx))
        return torch.cat((op1_emb, op2_emb, sum_emb), dim=-1)

class ActivationsEncoder(nn.Module):
    def __init__(self, model, input_dim, embedding_dim, output_dim, num_layers=4, num_heads=4, max_seq_len=9):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.max_seq_len = max_seq_len
        num_term_ids = 5
        self.pos_embedding = nn.Embedding(num_term_ids * max_seq_len, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(embedding_dim, output_dim)

    def forward(self, activations, term_encoding):
        x = self.input_proj(activations)
        _, seq_len, _ = x.shape

        positions = torch.arange(seq_len, device=activations.device)
        positions = positions.unsqueeze(0).expand_as(term_encoding)
        combined_ids = term_encoding * self.max_seq_len + positions
        pos_emb = self.pos_embedding(combined_ids)
        x = x + pos_emb

        x = self.transformer(x)
        x = self.output_proj(x)
        x = x.mean(dim=1)

        return x

class ProblemClassifier(nn.Module):
    def __init__(self, input_dim, k_classes, hidden1_dim=256, hidden2_dim=32):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, k_classes)
        )

    def forward(self, x):
        return self.classifier(x)

class NeuronMask(nn.Module):
    def __init__(self, k_classes, activations_dim):
        super().__init__()

        self.masks = nn.Parameter(torch.randn(k_classes, activations_dim))

    def forward(self, class_probs, activations_encoding):
        selected_mask = class_probs @ self.masks
        sigmoid_mask = torch.sigmoid(selected_mask)
        masked_activations = activations_encoding * sigmoid_mask
        return masked_activations, sigmoid_mask

class CircuitDiscoveryModel(nn.Module):
    def __init__(self, k_classes, problem_embedding_dim=256, activation_embedding_dim=1024, tau=0.5):
        super().__init__()

        self.tau = tau
        num_activations_1b = config['1b'].intermediate_size * config['1b'].num_hidden_layers
        num_activations_8b = config['8b'].intermediate_size * config['8b'].num_hidden_layers

        self.problem_encoder = ProblemEncoder(embedding_dim=problem_embedding_dim)
        self.activations_1b_encoder = ActivationsEncoder('1b', input_dim=num_activations_1b, embedding_dim=256, output_dim=activation_embedding_dim)
        self.activations_8b_encoder = ActivationsEncoder('8b', input_dim=num_activations_8b, embedding_dim=512, output_dim=activation_embedding_dim)
        self.classifier = ProblemClassifier(problem_embedding_dim + activation_embedding_dim, k_classes)

        self.neuron_masks_1b = NeuronMask(k_classes, activation_embedding_dim)
        self.neuron_masks_8b = NeuronMask(k_classes, activation_embedding_dim)

    def forward(self, op1, op2, res, activations_1b, activations_8b, term_encoding):
        problem_encoding = self.problem_encoder(op1, op2, res)
        activations_1b_encoding = self.activations_1b_encoder(activations_1b, term_encoding)
        activations_8b_encoding = self.activations_8b_encoder(activations_8b, term_encoding)

        combined_encoding = torch.cat((problem_encoding, activations_1b_encoding, activations_8b_encoding), dim=-1)
        logits = self.classifier(combined_encoding)
        hard_class_probs = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)

        masked_activations_1b, mask_1b = self.neuron_masks_1b(hard_class_probs, activations_1b_encoding)
        masked_activations_8b, mask_8b = self.neuron_masks_8b(hard_class_probs, activations_8b_encoding)

        with torch.no_grad():
            soft_class_probs = F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=-1)
            # Reuse CircuitLoss binary_entropy implementation for class entropy metric
            class_entropy = CircuitLoss().binary_entropy(soft_class_probs)
            frac_activated_1b = torch.sum(mask_1b) / torch.numel(mask_1b)
            frac_activated_8b = torch.sum(mask_8b) / torch.numel(mask_8b)

        return {
            "hard_class_probs": hard_class_probs,
            "masked_activations_1b": masked_activations_1b, 
            "masked_activations_8b": masked_activations_8b, 
            "mask_1b": mask_1b,
            "mask_8b": mask_8b,
            "frac_activated_1b": frac_activated_1b.detach(), 
            "frac_activated_8b": frac_activated_8b.detach(), 
            "class_entropy": class_entropy.detach()
        }

class CircuitLoss(nn.Module):
    def __init__(self, lambda_sim=1.0, lambda_sparsity=1e-3, eps=1e-8):
        super().__init__()
        self.lambda_sim = lambda_sim
        self.lambda_sparsity = lambda_sparsity
        self.eps = eps

    def classwise_pairwise_cossim(self, activations, hard_class_probs):
        _, k_classes = hard_class_probs.shape

        norm_acts = F.normalize(activations, p=2, dim=-1, eps=self.eps)

        per_class_sims = []
        for k in range(k_classes):
            class_mask = hard_class_probs[:, k].bool()
            idx = class_mask.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() < 2:
                continue

            acts_k = norm_acts[idx]
            sim_mat = acts_k @ acts_k.t()

            n_k = acts_k.size(0)
            if n_k > 1:
                triu_indices = torch.triu_indices(n_k, n_k, offset=1, device=acts_k.device)
                pair_sims = sim_mat[triu_indices[0], triu_indices[1]]
                if pair_sims.numel() > 0:
                    per_class_sims.append(pair_sims.mean())

        if not per_class_sims:
            return activations.new_tensor(0.0)

        return torch.stack(per_class_sims).mean()

    def binary_entropy(self, p):
        p = torch.clamp(p, self.eps, 1.0 - self.eps)
        entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
        return entropy.mean()

    def combined_loss(self, hard_class_probs, masked_activations, mask):
        sim_loss = - self.classwise_pairwise_cossim(masked_activations, hard_class_probs)
        sparsity_loss = self.binary_entropy(mask)
        total_loss = self.lambda_sim * sim_loss + self.lambda_sparsity * sparsity_loss
        return total_loss, sim_loss, sparsity_loss

    def forward(self, hard_class_probs, masked_activations_1b, masked_activations_8b, mask_1b, mask_8b):
        loss_1b, sim_loss_1b, sparsity_loss_1b = combined_loss(hard_class_probs, masked_activations_1b, mask_1b)
        loss_8b, sim_loss_8b, sparsity_loss_8b = combined_loss(hard_class_probs, masked_activations_8b, mask_8b)
        total_loss = loss_1b + loss_8b

        return {
            "loss": total_loss,
            "sim_1b": sim_loss_1b.detach(),
            "sim_8b": sim_loss_8b.detach(),
            "sparsity_1b": sparsity_loss_1b.detach(),
            "sparsity_8b": sparsity_loss_8b.detach()
        }


def _stack_layer_activations(batch_activations):
    """Stack per-layer activations dict into a single tensor.

    Expects a dict: layer_idx -> Tensor[batch, seq_len, hidden].
    Returns Tensor[batch, seq_len, hidden * num_layers].
    """

    if not batch_activations:
        raise ValueError("batch_activations is empty")

    layers = sorted(batch_activations.keys())
    tensors = [batch_activations[i] for i in layers]
    return torch.cat(tensors, dim=-1)


def _load_s3_activation_batches(model_name, prefix_root="mlp_activations"):
    """Yield activation batches (prompts, activations_dict) from S3 for a given model.

    Each object was saved by gen_activations_dataset.py as a torch file with keys:
        'prompts': List[str]
        'activations': Dict[int, Tensor[batch, seq_len, hidden]]
    """

    prefix = f"{prefix_root}/{model_name}/"
    continuation_token = None

    while True:
        if continuation_token is None:
            resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
        else:
            resp = s3.list_objects_v2(
                Bucket=BUCKET_NAME,
                Prefix=prefix,
                ContinuationToken=continuation_token,
            )

        for obj in resp.get("Contents", []):
            key = obj["Key"]
            # Download to memory and load with torch
            buf = torch.tensor([])  # placeholder to ensure torch is imported
            local_path = "/tmp/activations_tmp.pt"
            s3.download_file(BUCKET_NAME, key, local_path)
            batch = torch.load(local_path, map_location="cpu")
            yield key, batch["prompts"], batch["activations"]

        if not resp.get("IsTruncated"):
            break
        continuation_token = resp.get("NextContinuationToken")


def train_circuit_discovery(
    k_classes,
    epochs=1,
    lr=1e-3,
    device=None,
    prefix_root="mlp_activations",
):
    """Train CircuitDiscoveryModel from activation batches stored in S3.

    This assumes you have uploaded activations for both the 1B and 8B models
    under S3 prefixes:
        mlp_activations/<llama_1b>/...
        mlp_activations/<llama_8b>/...
    with identical batching / prompts ordering.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CircuitDiscoveryModel(k_classes=k_classes).to(device)
    criterion = CircuitLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    metrics_log = []
    # Helper to list S3 keys for each model without downloading contents
    def list_keys(model_name):
        prefix = f"{prefix_root}/{model_name}/"
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

    keys_1b = list_keys(llama_1b)
    keys_8b = list_keys(llama_8b)

    # Align by suffix so batches correspond across models
    def suffix_map(keys):
        return {k.split("/", 2)[-1]: k for k in keys}

    map_1b = suffix_map(keys_1b)
    map_8b = suffix_map(keys_8b)
    shared_suffixes = list(set(map_1b.keys()) & set(map_8b.keys()))
    if not shared_suffixes:
        raise ValueError("No overlapping activation batches found for 1B and 8B models in S3.")

    files_per_epoch = 5  # one epoch = 5 files ≈ 250 samples

    for epoch in range(epochs):
        # Use the first files_per_epoch batches deterministically for each epoch
        epoch_suffixes = shared_suffixes[:files_per_epoch]

        stats = {"loss": [], "class_usage_entropy": [], "frac_1b": [], "frac_8b": [], "class_entropy": []}

        for suffix in epoch_suffixes:
            key_1b = map_1b[suffix]
            key_8b = map_8b[suffix]

            # Fresh download for this batch
            local_1b = "/tmp/activations_1b.pt"
            local_8b = "/tmp/activations_8b.pt"
            s3.download_file(BUCKET_NAME, key_1b, local_1b)
            s3.download_file(BUCKET_NAME, key_8b, local_8b)

            batch_1b = torch.load(local_1b, map_location="cpu")
            batch_8b = torch.load(local_8b, map_location="cpu")

            prompts_1b, activations_dict_1b = batch_1b["prompts"], batch_1b["activations"]
            prompts_8b, activations_dict_8b = batch_8b["prompts"], batch_8b["activations"]

            if prompts_1b != prompts_8b:
                continue

            prompts = prompts_1b

            activations_1b = _stack_layer_activations(activations_dict_1b).to(device)
            activations_8b = _stack_layer_activations(activations_dict_8b).to(device)

            op1, op2, res, term_encoding = parse_equation(prompts, device=device)

            model.train()
            optimizer.zero_grad()

            outputs = model(op1, op2, res, activations_1b, activations_8b, term_encoding)

            hard_class_probs = outputs["hard_class_probs"]
            masked_1b = outputs["masked_activations_1b"]
            masked_8b = outputs["masked_activations_8b"]
            mask_1b = outputs["mask_1b"]
            mask_8b = outputs["mask_8b"]

            loss_dict = criterion(hard_class_probs, masked_1b, masked_8b, mask_1b, mask_8b)
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                class_freq = hard_class_probs.float().mean(dim=0)
                class_freq = torch.clamp(class_freq, 1e-8, 1.0)
                class_usage_entropy = - (class_freq * class_freq.log()).sum().item()
                frac_1b = float(outputs["frac_activated_1b"])
                frac_8b = float(outputs["frac_activated_8b"])
                class_ent = float(outputs["class_entropy"])

            stats["loss"].append(loss.item())
            stats["class_usage_entropy"].append(class_usage_entropy)
            stats["frac_1b"].append(frac_1b)
            stats["frac_8b"].append(frac_8b)
            stats["class_entropy"].append(class_ent)

        def avg(key):
            return float(sum(stats[key]) / len(stats[key])) if stats[key] else float("nan")

        epoch_metrics = {
            "epoch": epoch + 1,
            "loss": avg("loss"),
            "class_usage_entropy": avg("class_usage_entropy"),
            "frac_activated_1b": avg("frac_1b"),
            "frac_activated_8b": avg("frac_8b"),
            "class_entropy": avg("class_entropy"),
        }

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"loss: {epoch_metrics['loss']:.4f} - "
            f"class_usage_entropy: {epoch_metrics['class_usage_entropy']:.4f} - "
            f"frac_activated_1b: {epoch_metrics['frac_activated_1b']:.4f} - "
            f"frac_activated_8b: {epoch_metrics['frac_activated_8b']:.4f} - "
            f"class_entropy: {epoch_metrics['class_entropy']:.4f}"
        )

        metrics_log.append(epoch_metrics)

    # Save metrics to results/circuit-discovery/metrics.json
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "circuit-discovery")
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=4)


if __name__ == "__main__":
    # Example: train with 4 subclasses
    train_circuit_discovery(k_classes=10, epochs=1000)




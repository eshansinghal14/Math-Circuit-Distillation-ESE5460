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

        self.masks = nn.Parameter(torch.randn(k_classes, embedding_dim))

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
        soft_class_probs = F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=-1)
        hard_class_probs = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)

        masked_activations_1b = self.neuron_masks_1b(hard_class_probs, activations_1b_encoding)
        masked_activations_8b = self.neuron_masks_8b(hard_class_probs, activations_8b_encoding)

        return hard_class_probs, soft_class_probs, logits, masked_activations_1b, masked_activations_8b


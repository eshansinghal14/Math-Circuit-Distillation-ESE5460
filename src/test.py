import json
import sys
from collections import deque
import torch
import boto3
from utils import load_model, parse_answer

s3 = boto3.client('s3')
BUCKET_NAME = 'math-circuit-distillation-ese5460'
model_name = sys.argv[1]
s3.download_file(BUCKET_NAME, f'mlp_activations/{model_name}/{i}_{len(inputs)}.pt', './local_activations.pt')
activations = torch.load('./local_activations.pt')

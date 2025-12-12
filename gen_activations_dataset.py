import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import login

import sys
from utils import load_model, generate_2digit_addition_prob

model = load_model(sys.argv)
if model is None:
    exit()


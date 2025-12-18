from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from utils import test_model, eval_model
from constants import HF_TOKEN
from huggingface_hub import login

login(HF_TOKEN)

model_name = "vedantgaur/circuit-distilled-llama-1b-low-cka"
student = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_size = 'left'

test_model(student, tokenizer, "../datasets/2d_add_test_20.json", f"../results/{model_name}/2d_add_test_20.json")
eval_model(f"../results/{model_name}/2d_add_test_20.json")
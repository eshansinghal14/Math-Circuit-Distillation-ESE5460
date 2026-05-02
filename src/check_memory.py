import torch
import gc
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer

def print_mem(tag):
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"[{tag}] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

print_mem("Start")
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
student_name = "meta-llama/Llama-3.2-1B-Instruct"

hf_student = AutoModelForCausalLM.from_pretrained(student_name, torch_dtype=torch.bfloat16, device_map="cuda:0")
print_mem("After HF Student")

from graph_loss.replacement_model import TransformerLensReplacementModel
teacher = TransformerLensReplacementModel.from_pretrained(model_name, device="cuda:0", dtype=torch.bfloat16)
print_mem("After Teacher HookedTransformer")

gc.collect()
torch.cuda.empty_cache()
print_mem("After GC & Empty Cache")


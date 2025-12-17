# Math-Circuit-Distillation-ESE5460

Circuit Distillation for Mathematical Reasoning Tasks

## Models

| Model | Accuracy | Link |
|-------|----------|------|
| Teacher (Llama-3-8B) | 96.3% | [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) |
| Student (Llama-3.2-1B) | 50.5% (baseline) | [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) |
| **Distilled Student** | - | [vedantgaur/circuit-distilled-llama-1b](https://huggingface.co/vedantgaur/circuit-distilled-llama-1b) |

## Method

Uses CKA (Centered Kernel Alignment) loss to align student model representations with teacher circuit heads, combined with standard cross-entropy loss:

```
L = L_CE + λ * Σ L_CKA(paired_layers)
```

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("vedantgaur/circuit-distilled-llama-1b")
tokenizer = AutoTokenizer.from_pretrained("vedantgaur/circuit-distilled-llama-1b")
```
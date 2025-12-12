import random
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch

def load_model(argv):
    if len(argv) > 1:
        model_name = argv[1]
    else:
        print('Please provide model name')
        exit()

    login()
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_size = 'left'
    return model, tokenizer

def test_model(model, tokenizer, dataset_fname, results_fname, samples=1000, batch_size=50):
    with open(dataset_fname, 'r') as f:
        dataset = json.load(f)
    prompts = list(dataset.keys())
    results = []
    for i in range(0, samples, batch_size):
        with torch.no_grad():
            print(f'processing {i}/{samples}')
            batched_prompts = prompts[i:i + batch_size]   
            input_ids = tokenizer(batched_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            outputs = model.generate(**input_ids, max_length=512, pad_token_id=tokenizer.pad_token_id)
            responses = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

            for k, resp in enumerate(responses):
                results.append({'response': resp, 'answer': dataset[i + k]})

    with open(results_fname, 'w') as f:
        json.dump(results, f, indent=4)

def gen_2d_add_prob():
    num1 = random.randint(0, 99)
    num2 = random.randint(0, 99)
    prompt_str = f'{num1}+{num2}='
    return prompt_str, num1 + num2


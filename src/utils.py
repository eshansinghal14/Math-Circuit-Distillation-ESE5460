import random
import json
import re

def load_model(argv):
    if len(argv) > 1:
        model_name = argv[1]
    else:
        print('Please provide model name')
        exit()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import login

    login()
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_size = 'left'
    return model, tokenizer

def test_model(model, tokenizer, dataset_fname, results_fname, samples=1000, batch_size=50):
    model.eval()
    with open(dataset_fname, 'r') as f:
        dataset = json.load(f)
    prompts = list(dataset.keys())
    results = []
    for i in range(0, samples, batch_size):
        import torch
        with torch.no_grad():
            print(f'processing {i}/{samples}')
            batched_prompts = prompts[i:i + batch_size]   
            input_ids = tokenizer(batched_prompts, return_tensors="pt", do_sample=False, padding=True, truncation=True).to(model.device)
            outputs = model.generate(**input_ids, max_new_tokens=5, pad_token_id=tokenizer.pad_token_id)
            responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for k, resp in enumerate(responses):
                results.append({'response': resp, 'answer': dataset[batched_prompts[k]]})

    with open(results_fname, 'w') as f:
        json.dump(results, f, indent=4)

def parse_answer(resp):
    match = re.search(r'=\s*(\d+)', resp)
    return int(match.group(1)) if match else None

def eval_model(results_fname):
    with open(results_fname, 'r') as f:
        results = json.load(f)

    correct = 0
    for res in results:
        if parse_answer(res['response']) == res['answer']:
            correct += 1

    print('Accuracy: ', correct / len(results))

# samples=None means all 2-digit addition pairs are added; otherwise sample without replacement
def gen_2d_add_dataset(dataset_fname, samples):
    all_pairs = [(f'{num1}+{num2}=', num1 + num2) for num1 in range(100) for num2 in range(100)]

    if samples is None or samples >= len(all_pairs):
        selected = all_pairs
    else:
        selected = random.sample(all_pairs, samples)

    dataset = {prompt: answer for prompt, answer in selected}

    with open(dataset_fname, 'w') as f:
        json.dump(dataset, f, indent=4)

from utils import load_model, test_model, eval_model, gen_2d_add_dataset, gen_3d_add_dataset
import json
import argparse


parser = argparse.ArgumentParser(description="Benchmark model on arithmetic dataset")
parser.add_argument(
    "--model-name",
    type=str,
    required=True,
    help="HuggingFace model identifier (e.g. meta-llama/Llama-3.2-1B)",
)
args = parser.parse_args()

model_name = args.model_name
model, tokenizer = load_model(model_name)

# gen_2d_add_dataset('../datasets/2d_add_train.json', samples=5000)
# gen_2d_add_dataset('../datasets/2d_add_all.json', samples=None, tokenizer=tokenizer)
# gen_2d_add_dataset('../datasets/2d_add_test_20.json', samples=None, tokenizer=tokenizer)
# gen_3d_add_dataset('../datasets/3d_add_test.json', samples=1000, tokenizer=tokenizer)

dataset = '3d_add_test.json'
# test_model(model, tokenizer, f'../datasets/{dataset}', f'../results/{model_name}/{dataset}')

eval_model(f'../results/meta-llama/Meta-Llama-3-8B/{dataset}')
eval_model(f'../results/meta-llama/Llama-3.2-1B/{dataset}')
eval_model(f'../results/vedantgaur/circuit-distilled-llama-1b-fixed-loss/{dataset}')

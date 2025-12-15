from utils import load_model, test_model, eval_model, gen_2d_add_dataset
import json
import sys

# gen_2d_add_dataset('../datasets/2d_add_train.json', samples=5000)
# gen_2d_add_dataset('../datasets/2d_add_all.json', samples=None)

# model, tokenizer = load_model(sys.argv)
# model_name = sys.argv[1]
# batch_sizes = {
#     'meta-llama/Llama-3.2-1B': 50,
#     'meta-llama/Meta-Llama-3-8B': 100,
# }
# test_model(model, tokenizer, '../datasets/2d_add_all.json', f'../results/{model_name}/2d_add_all.json', batch_sizes[model_name])

eval_model('../results/meta-llama/Meta-Llama-3-8B/2d_add_all.json')
eval_model('../results/meta-llama/Llama-3.2-1B/2d_add_all.json')

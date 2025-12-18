from utils import load_model, get_model_name, test_model, eval_model, gen_2d_add_dataset
import json
import sys

# model, tokenizer = load_model(sys.argv)
# model_name = get_model_name(sys.argv)

# gen_2d_add_dataset('../datasets/2d_add_train.json', samples=5000)
# gen_2d_add_dataset('../datasets/2d_add_all.json', samples=None, tokenizer=tokenizer)
# gen_2d_add_dataset('../datasets/2d_add_test_20.json', samples=None, tokenizer=tokenizer)

# test_model(model, tokenizer, '../datasets/2d_add_test_20.json', f'../results/{model_name}/2d_add_test_20.json')

eval_model('../results/meta-llama/Meta-Llama-3-8B/2d_add_all.json')
eval_model('../results/meta-llama/Llama-3.2-1B/2d_add_all.json')

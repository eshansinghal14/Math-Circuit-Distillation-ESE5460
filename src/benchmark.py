from utils import load_model, test_model, eval_model, gen_2d_add_dataset
import json
import sys

if __name__ == '__main__':
    gen_2d_add_dataset('../datasets/2d_add_train.json', samples=5000)
    gen_2d_add_dataset('../datasets/2d_add_test.json', samples=None)

    # model, tokenizer = load_model(sys.argv)
    # test_model(model, tokenizer, 'datasets/2d_add_test.json', 'results/2d_add_test.json')

    # eval_model('../results/2d_add_test.json')

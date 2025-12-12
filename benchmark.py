from utils import load_model, test_model, gen_2d_add_prob
import json
import sys

def gen_2d_add_dataset(filename, samples=1000):
    dataset = {}
    for i in range(samples):
        if i % 100 == 0:
            print(f'processing {i}/{samples}')
        prompt, answer = gen_2d_add_prob()
        dataset[prompt] = answer
    
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=4)

if __name__ == '__main__':
    # gen_2d_add_dataset('datasets/2d_add_test.json')
    model, tokenizer = load_model(sys.argv)
    test_model(model, tokenizer, 'datasets/2d_add_test.json', 'results/2d_add_test.json')
"""What kind of mistakes does each model make on mix3? Greedy answer vs gold, same problems as graph_metrics.

For each model: accuracy, and among wrong answers the share that are a number
within 10 of the gold value (a near miss inside the right procedure), a number
equal to a partial result or an operand (a step skipped or misapplied), another
number, or not a number at all.

    cd src && python -u -m experiments.error_kinds --model kd150=<path> --model base8b=<path> --n 200
"""

import argparse
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_data, load_model, set_bos_mode  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    set_bos_mode("on")
    _, test = load_data("mix3")
    every = sorted(test.items())
    items = every[::max(1, len(every) // args.n)][: args.n]
    for spec in args.model:
        label, path = spec.split("=", 1)
        model, tok = load_model(path)
        model.eval()
        kinds = {"near (|d|<=10)": 0, "partial / operand": 0, "other number": 0, "not a number": 0}
        right, examples = 0, []
        for prompt, answer in items:
            ids = tok(prompt, return_tensors="pt")["input_ids"].to(model.device)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            m = re.match(r"\s*(-?\d+)", text)
            gold = int(answer)
            if m and int(m.group(1)) == gold:
                right += 1
                continue
            nums = [int(x) for x in re.findall(r"\d+", prompt)]
            a, b, c = nums[0], nums[1], nums[2]
            o1 = re.findall(r"[+-]", prompt)[0]
            partial = a + b if o1 == "+" else a - b
            if not m:
                kinds["not a number"] += 1
            else:
                v = int(m.group(1))
                if abs(v - gold) <= 10:
                    kinds["near (|d|<=10)"] += 1
                elif v in (partial, a, b, c):
                    kinds["partial / operand"] += 1
                else:
                    kinds["other number"] += 1
            if len(examples) < 4:
                examples.append(f"{prompt}{text.strip()!r} (gold {gold})")
        wrong = len(items) - right
        print(f"{label:8s} acc {right / len(items):.3f} | wrong {wrong}: "
              + ", ".join(f"{k} {v / max(wrong, 1):.2f}" for k, v in kinds.items()), flush=True)
        print("          e.g. " + " | ".join(examples), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

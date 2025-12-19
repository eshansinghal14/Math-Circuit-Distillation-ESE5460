import json
from transformers import AutoTokenizer

def convert_add_json(
    in_path: str,
    out_path: str,
    tokenizer_name: str = "meta-llama/Llama-3.2-1B",
) -> None:
    """
    Convert addition JSON of the form:
        { "55+94=": 149, ... }
    into a list of dicts:
        [{ "q_str": "55+94=", "a_str": "149", "ids": [...] }, ...]
    saved to out_path.
    """
    # Load original mapping
    with open(in_path, "r") as f:
        data = json.load(f)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # Keep it consistent with utils.test_model
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    converted = []
    for q_str, answer in data.items():
        a_str = str(answer)
        # Encode question + answer, no special tokens
        ids = tokenizer.encode(q_str + a_str, add_special_tokens=False)
        converted.append(
            {
                "q_str": q_str,
                "a_str": a_str,
                "ids": ids,
            }
        )

    with open(out_path, "w") as f:
        json.dump(converted, f, indent=4)

    print(f"Saved converted dataset to {out_path}")

convert_add_json(
    "../datasets/2d_add_train_80.json",
    "../datasets/2d_add_train_80_formatted.json",
)

convert_add_json(
    "../datasets/2d_add_test_20.json",
    "../datasets/2d_add_test_20_formatted.json",
)
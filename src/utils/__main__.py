from transformers import AutoTokenizer

from . import (
    LLAMA_1B_MODEL_NAME,
    dataset_all_json_path,
    generate_math_dataset,
    patch_tokenizer_no_special_tokens,
)


def main() -> None:
    tokenizer = patch_tokenizer_no_special_tokens(
        AutoTokenizer.from_pretrained(LLAMA_1B_MODEL_NAME),
    )

    generate_math_dataset(
        dataset_all_json_path("22_add_tight_5000"),
        tokenizer,
        digits=[2, 2],
        operations=[["+"]],
        spaces=False,
        shuffle=True,
        samples=10000,
        split_test_frac=0.5,
    )

if __name__ == "__main__":
    main()

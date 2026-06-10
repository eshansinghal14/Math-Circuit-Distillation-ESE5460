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
        dataset_all_json_path("21_mult_tight"),
        tokenizer,
        digits=[2, 1],
        operations=[["*"]],
        spaces=False,
        shuffle=True,
        samples=1000,
        split_test_frac=0.5,
    )

if __name__ == "__main__":
    main()

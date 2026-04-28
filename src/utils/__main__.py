from transformers import AutoTokenizer

from . import (
    LLAMA_1B_MODEL_NAME,
    dataset_all_json_path,
    default_datasets_dir,
    generate_math_dataset,
    mix_datasets,
    patch_tokenizer_no_special_tokens,
)


def main() -> None:
    tokenizer = patch_tokenizer_no_special_tokens(
        AutoTokenizer.from_pretrained(LLAMA_1B_MODEL_NAME),
    )

    generate_math_dataset(
        dataset_all_json_path("222_add_tight"),
        tokenizer,
        digits=[2, 2, 2],
        operations=[["+", "+"]],
        spaces=False,
        shuffle=True,
        samples=30000,
        split_test_frac=0.167,
    )

    # mix_datasets(
    #     dataset_stems=["22_add_tight", "222_add_tight"],
    #     output_stem="2-3_add",
    #     datasets_dir=default_datasets_dir(),
    #     shuffle=True,
    # )


if __name__ == "__main__":
    main()

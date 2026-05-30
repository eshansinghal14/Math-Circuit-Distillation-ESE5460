from typing import Optional, Union

import torch

from .config import HF_READ_TOKEN
from .device import get_default_device

logged_in = False


def patch_tokenizer_no_special_tokens(tokenizer):
    """Default ``add_special_tokens=False`` on ``__call__`` and ``encode`` (idempotent).

    Avoids silently prepending BOS / appending EOS on encode paths so training and eval
    match raw string tokenization.
    """
    if getattr(tokenizer, "_math_circuit_no_special_tokens_patched", False):
        return tokenizer
    orig_call = tokenizer.__call__
    orig_encode = tokenizer.encode

    def _call(*args, **kwargs):
        kwargs.setdefault("add_special_tokens", False)
        return orig_call(*args, **kwargs)

    def _encode(*args, **kwargs):
        kwargs.setdefault("add_special_tokens", False)
        return orig_encode(*args, **kwargs)

    tokenizer.__call__ = _call
    tokenizer.encode = _encode
    tokenizer._math_circuit_no_special_tokens_patched = True
    return tokenizer


def load_model(model_name):
    import os

    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    try:
        from huggingface_hub.utils import disable_progress_bars as _hfhub_disable
        _hfhub_disable()
    except Exception:
        pass

    from huggingface_hub import login
    from transformers import AutoModelForCausalLM, AutoTokenizer

    global logged_in
    if not logged_in and HF_READ_TOKEN:
        login(HF_READ_TOKEN)
        logged_in = True

    device = get_default_device()
    # Bypass hub repo-id validation for local paths.  Use prefix detection
    # rather than os.path.isdir() because FUSE mounts (e.g. Google Drive) can
    # return False for real directories if the mount is slow to respond.
    _is_local = model_name.startswith("/") or model_name.startswith(".") or os.path.isdir(model_name)
    local_kwargs = {"local_files_only": True} if _is_local else {}
    # Suppress the safetensors/tqdm loading bar that hf_logging.disable_progress_bar()
    # does not cover (safetensors uses tqdm directly, not the HF progress-bar layer).
    _prev_tqdm = os.environ.get("TQDM_DISABLE")
    os.environ["TQDM_DISABLE"] = "1"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            **local_kwargs,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, **local_kwargs)
    finally:
        if _prev_tqdm is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = _prev_tqdm
    tokenizer.pad_token = tokenizer.eos_token
    # Match distillation AddDataset/collate_fn (right-pad); eval/generate use same side.
    tokenizer.padding_side = "right"
    return model, patch_tokenizer_no_special_tokens(tokenizer)


def load_student_model_for_distillation(
    student_source: Optional[str],
    student_model_id: str,
    device: Union[str, torch.device],
):
    """Load student LM + tokenizer for distillation (HF id, saved HF dir, or ``.pt`` state_dict)."""
    print("\n" + "=" * 60)
    print("Loading student")
    print("=" * 60)
    if student_source and student_source.endswith(".pt"):
        print(f"  Loading base model + fast weights from: {student_source!r}")
        student, tokenizer = load_model(student_model_id)
        state_dict = torch.load(student_source, map_location="cpu", weights_only=True)
        student.load_state_dict(state_dict)
        del state_dict
    elif student_source:
        print(f"  From checkpoint dir: {student_source!r}")
        student, tokenizer = load_model(student_source)
    else:
        print(f"  From Hugging Face: {student_model_id!r}")
        student, tokenizer = load_model(student_model_id)
    student = student.to("cpu").float().to(device)
    tokenizer.padding_side = "right"
    return student, tokenizer

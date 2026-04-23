import torch


def get_default_device() -> torch.device:
    """Get the default device, preferring CUDA if available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

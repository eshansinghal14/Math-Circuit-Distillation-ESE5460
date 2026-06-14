"""Project constants -- fill in before running on Colab.

This file is imported by utils.py and circuit_discovery/utils.py.
Set your HuggingFace token and any other shared constants here.
"""

import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")

CIRCUIT_DISCOVERY_CKPT_DIR = os.environ.get(
    "CIRCUIT_DISCOVERY_CKPT_DIR", ""
)

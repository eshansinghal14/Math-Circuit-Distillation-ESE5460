import os
import shutil
import subprocess
from typing import Optional, Tuple

import torch


def rm_dir_tree(path: str) -> None:
    """Delete a directory tree (shell ``rm -rf`` on Unix, :func:`shutil.rmtree` fallback)."""
    try:
        result = subprocess.run(["rm", "-rf", path], capture_output=True)
        if result.returncode != 0:
            raise OSError(result.stderr.decode().strip())
    except Exception:
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass


def training_state_path(save_dir: str) -> str:
    return os.path.join(save_dir, "training_state.pt")


def save_training_state(
    save_dir: str,
    optimizer: torch.optim.Optimizer,
    next_epoch: int,
    best_acc: float,
) -> None:
    """``next_epoch`` = number of epochs already completed (resume starts at this index)."""
    path = training_state_path(save_dir)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "next_epoch": next_epoch,
            "best_acc": best_acc,
        },
        path,
    )


def load_training_state(
    path: str, optimizer: torch.optim.Optimizer, map_location,
) -> Tuple[int, float]:
    try:
        chk = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        chk = torch.load(path, map_location=map_location)
    optimizer.load_state_dict(chk["optimizer"])
    return int(chk["next_epoch"]), float(chk["best_acc"])


def most_recent_subdirectory(parent_dir: str) -> Optional[str]:
    """Most recently modified immediate subdirectory of ``parent_dir``."""
    if not os.path.isdir(parent_dir):
        return None
    try:
        entries = os.listdir(parent_dir)
    except OSError:
        return None
    best_mtime, best_path = None, None
    for name in entries:
        full = os.path.join(parent_dir, name)
        if not os.path.isdir(full):
            continue
        try:
            mtime = os.path.getmtime(full)
            if best_mtime is None or mtime > best_mtime:
                best_mtime, best_path = mtime, full
        except OSError:
            pass
    return best_path

import json
import os
from typing import Any

import torch


class TeacherDataCache:
    """Load per-prompt teacher logits and graph artifacts from `generate_teacher_data`."""

    def __init__(self, cache_dir: str):
        self.cache_dir = os.path.abspath(cache_dir)
        manifest_path = os.path.join(self.cache_dir, "manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest: dict[str, Any] = json.load(f)

        self.teacher_model = self.manifest.get("teacher_model")
        self.student_model = self.manifest.get("student_model")
        self.teacher_vocab_size = int(self.manifest["teacher_vocab_size"])
        self._samples: dict[tuple[str, int], dict[str, Any]] = {}
        for sample in self.manifest.get("samples", []):
            self._samples[(str(sample["prompt"]), int(sample["answer"]))] = sample

    def _artifact_path(self, sample: dict[str, Any], key: str) -> str:
        rel = sample[key]
        return os.path.join(self.cache_dir, rel)

    def _load_logits_record(self, prompt: str, answer: int) -> dict[str, Any]:
        key = (str(prompt), int(answer))
        if key not in self._samples:
            raise KeyError(f"No cached teacher data for prompt={prompt!r}, answer={answer!r}")
        return torch.load(
            self._artifact_path(self._samples[key], "teacher_logits"),
            map_location="cpu",
            weights_only=False,
        )

    def get_artifact_path(self, prompt: str, answer: int, artifact_name: str) -> str | None:
        key = (str(prompt), int(answer))
        if key not in self._samples:
            raise KeyError(f"No cached teacher data for prompt={prompt!r}, answer={answer!r}")
        rel = self._samples[key].get(artifact_name)
        return os.path.join(self.cache_dir, rel) if rel else None

    def load_teacher_supergraph(self, prompt: str, answer: int) -> dict[str, Any]:
        path = self.get_artifact_path(prompt, answer, "supergraph")
        if path is None:
            raise FileNotFoundError("Teacher cache sample has no supergraph artifact")
        return torch.load(path, map_location="cpu", weights_only=False)

    def load_teacher_supernode_dla(self, prompt: str, answer: int) -> dict[str, Any]:
        path = self.get_artifact_path(prompt, answer, "teacher_supernode_dla")
        if path is None:
            raise FileNotFoundError("Teacher cache sample has no teacher DLA artifact")
        return torch.load(path, map_location="cpu", weights_only=False)

    def get_batch_logits(
        self,
        *,
        prompts: list[str],
        answers: list[int],
        input_ids: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        records = [
            self._load_logits_record(prompt, answer)
            for prompt, answer in zip(prompts, answers, strict=True)
        ]
        batch_size, seq_len = input_ids.shape
        dtype = records[0]["logits"].dtype
        logits = torch.zeros(
            batch_size,
            seq_len,
            self.teacher_vocab_size,
            dtype=dtype,
        )
        input_ids_cpu = input_ids.detach().cpu()

        for row, record in enumerate(records):
            cached_ids = record["input_ids"].to(dtype=input_ids_cpu.dtype)
            length = int(cached_ids.numel())
            if length > seq_len:
                raise ValueError(
                    f"Cached sequence for prompt {prompts[row]!r} is longer than batch sequence"
                )
            if not torch.equal(input_ids_cpu[row, :length], cached_ids):
                raise ValueError(
                    "Teacher cache tokenization mismatch for prompt "
                    f"{prompts[row]!r}. Regenerate the cache with the same student tokenizer."
                )
            cached_logits = record["logits"]
            if cached_logits.shape[0] != length:
                raise ValueError(
                    f"Cached logits length mismatch for prompt {prompts[row]!r}: "
                    f"{cached_logits.shape[0]} vs {length}"
                )
            if cached_logits.shape[1] != self.teacher_vocab_size:
                raise ValueError(
                    f"Cached logits vocab mismatch for prompt {prompts[row]!r}: "
                    f"{cached_logits.shape[1]} vs manifest {self.teacher_vocab_size}"
                )
            logits[row, :length, :] = cached_logits

        return logits.to(device=device)

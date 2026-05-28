import json
import os
from typing import Any

import torch

_SHARD_PREFIX = "manifest_shard_"
_COT_METADATA_FILENAME = "cot_metadata.json"


def _load_merged_manifest(cache_dir: str) -> dict[str, Any]:
    """Load manifest.json, or merge all shard manifests on-the-fly if it doesn't exist."""
    manifest_path = os.path.join(cache_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    shard_files = sorted(
        f for f in os.listdir(cache_dir) if f.startswith(_SHARD_PREFIX) and f.endswith(".json")
    )
    if not shard_files:
        raise FileNotFoundError(
            f"No manifest.json or shard manifests found in {cache_dir!r}. "
            "Run generate_teacher_data.py first."
        )

    merged: dict[str, Any] = {}
    all_samples: list[dict[str, Any]] = []
    for fname in shard_files:
        with open(os.path.join(cache_dir, fname), "r", encoding="utf-8") as f:
            shard = json.load(f)
        if not merged:
            merged = {k: v for k, v in shard.items() if k != "samples"}
        all_samples.extend(shard.get("samples", []))

    all_samples.sort(key=lambda s: s["sample_index"])
    merged["samples"] = all_samples
    return merged


class TeacherDataCache:
    """Load per-prompt teacher logits and graph artifacts from `generate_teacher_data`."""

    def __init__(self, cache_dir: str):
        self.cache_dir = os.path.abspath(cache_dir)
        self.manifest: dict[str, Any] = _load_merged_manifest(self.cache_dir)

        self.teacher_model = self.manifest.get("teacher_model")
        self.student_model = self.manifest.get("student_model")
        self.teacher_vocab_size = int(self.manifest["teacher_vocab_size"])
        # Keys use str(answer) so both int and str answers are handled uniformly.
        self._samples: dict[tuple[str, str], dict[str, Any]] = {}
        for sample in self.manifest.get("samples", []):
            self._samples[(str(sample["prompt"]), str(sample["answer"]))] = sample

    def _artifact_path(self, sample: dict[str, Any], key: str) -> str:
        rel = sample[key]
        return os.path.join(self.cache_dir, rel)

    def _sample_dir(self, prompt: str, answer: int | str) -> str:
        key = (str(prompt), str(answer))
        if key not in self._samples:
            raise KeyError(f"No cached teacher data for prompt={prompt!r}, answer={answer!r}")
        return os.path.join(self.cache_dir, self._samples[key]["folder"])

    def _load_logits_record(self, prompt: str, answer: int | str) -> dict[str, Any]:
        key = (str(prompt), str(answer))
        if key not in self._samples:
            raise KeyError(f"No cached teacher data for prompt={prompt!r}, answer={answer!r}")
        return torch.load(
            self._artifact_path(self._samples[key], "teacher_logits"),
            map_location="cpu",
            weights_only=False,
        )

    def get_artifact_path(self, prompt: str, answer: int | str, artifact_name: str) -> str | None:
        key = (str(prompt), str(answer))
        if key not in self._samples:
            raise KeyError(f"No cached teacher data for prompt={prompt!r}, answer={answer!r}")
        rel = self._samples[key].get(artifact_name)
        return os.path.join(self.cache_dir, rel) if rel else None

    def load_teacher_supergraph(self, prompt: str, answer: int | str) -> dict[str, Any]:
        path = self.get_artifact_path(prompt, answer, "supergraph")
        if path is None:
            raise FileNotFoundError("Teacher cache sample has no supergraph artifact")
        return torch.load(path, map_location="cpu", weights_only=False)

    def load_logit_token_ids(self, prompt: str, answer: int | str) -> torch.Tensor | None:
        """Return the teacher's top-k logit token IDs saved during pregeneration, or None."""
        data = self.load_teacher_supergraph(prompt, answer)
        return data.get("logit_token_ids")

    def load_teacher_supernode_dla(self, prompt: str, answer: int | str) -> dict[str, Any]:
        path = self.get_artifact_path(prompt, answer, "teacher_supernode_dla")
        if path is None:
            raise FileNotFoundError("Teacher cache sample has no teacher DLA artifact")
        return torch.load(path, map_location="cpu", weights_only=False)

    def is_cot_cache(self, prompt: str, answer: int | str) -> bool:
        """Return True iff a CoT cache (cot_metadata.json) exists for this sample."""
        try:
            sample_dir = self._sample_dir(prompt, answer)
        except KeyError:
            return False
        return os.path.exists(os.path.join(sample_dir, _COT_METADATA_FILENAME))

    def load_cot_teacher_supergraphs(self, prompt: str, answer: int | str) -> list[dict[str, Any]]:
        """Load all K CoT supergraphs for one sample.

        Returns a list of dicts, each containing the same keys as
        ``load_teacher_supergraph`` plus ``"prefix"`` (str) and ``"dla_logits"``
        (Tensor).

        Raises KeyError if cot_metadata.json does not exist.
        """
        sample_dir = self._sample_dir(prompt, answer)
        meta_path = os.path.join(sample_dir, _COT_METADATA_FILENAME)
        if not os.path.exists(meta_path):
            raise KeyError(
                f"No CoT metadata found for prompt={prompt!r}, answer={answer!r}. "
                "Regenerate with --cot-k-positions > 0."
            )
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        K: int = int(meta["k_positions"])
        result: list[dict[str, Any]] = []
        for k in range(K):
            sg_path = os.path.join(sample_dir, f"supergraph_k{k}.pt")
            data = torch.load(sg_path, map_location="cpu", weights_only=False)
            result.append(data)
        return result

    def get_batch_logits(
        self,
        *,
        prompts: list[str],
        answers: list[int | str],
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

import os
import sys
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, SGD
from tqdm import tqdm

from transformers import AutoTokenizer

from utils import load_model, load_model_checkpoint, test_model, eval_model
from circuit_discovery.utils import parse_equation
from circuit_discovery.models import CircuitDiscoveryModel
from cluster_pairing import _load_single_ablation_performance, create_cluster_mapping, ClusterMapping


def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    X = X.float()
    Y = Y.float()

    if X.dim() == 3:
        X = X.reshape(-1, X.size(-1))
    if Y.dim() == 3:
        Y = Y.reshape(-1, Y.size(-1))

    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    XtX_norm = torch.norm(X.T @ X, "fro")
    YtY_norm = torch.norm(Y.T @ Y, "fro")
    YtX_norm_sq = torch.norm(Y.T @ X, "fro") ** 2

    denom = XtX_norm * YtY_norm + eps
    if denom < eps:
        return torch.tensor(0.0, device=X.device, dtype=torch.float32)

    cka = YtX_norm_sq / denom
    return torch.clamp(cka, 0.0, 1.0)


class CKALoss(nn.Module):
    def forward(self, student_acts: torch.Tensor, teacher_acts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cka = linear_cka(student_acts, teacher_acts)
        loss = 1.0 - cka
        return loss, cka


class ActivationCache:
    def __init__(self):
        self.activations: Dict[int, torch.Tensor] = {}
        self.hooks: List = []

    def create_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            self.activations[layer_idx] = output.detach()

        return hook

    def register_hooks(self, model, layer_indices: List[int]):
        self.clear()

        for idx in layer_indices:
            layer = model.model.layers[idx].mlp
            hook = layer.register_forward_hook(self.create_hook(idx))
            self.hooks.append(hook)

    def clear(self):
        self.activations = {}
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


@dataclass
class Config:
    teacher_model_name: str = "meta-llama/Meta-Llama-3-8B"
    student_model_name: str = "meta-llama/Llama-3.2-1B"
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    lambda_cka: float = 0.01
    grad_clip: float = 1.0
    top_k_pairs_per_subclass: Optional[int] = None
    train_path: str = "../datasets/2d_add_train_80.json"
    test_path: str = "../datasets/2d_add_test_20_formatted.json"
    teacher_cache_path: str = "../results/teacher_cache_train.pt"


class AddDataset(Dataset):
    def __init__(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)

        self.prompts = list(data.keys())
        self.answers = [str(data[p]) for p in self.prompts]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        answer = self.answers[idx]
        return {"idx": idx, "prompt": prompt, "answer": answer}


def collate_batch(examples, tokenizer):
    indices = [ex["idx"] for ex in examples]
    prompts = [ex["prompt"] for ex in examples]
    answers = [ex["answer"] for ex in examples]

    full_texts = [p + str(a) for p, a in zip(prompts, answers)]

    enc = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    input_ids = enc["input_ids"]

    labels = input_ids.clone()
    prompt_enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    prompt_lens = (prompt_enc["input_ids"] != tokenizer.pad_token_id).sum(dim=1)
    for i, l in enumerate(prompt_lens.tolist()):
        labels[i, :l] = -100

    return {
        "indices": torch.tensor(indices, dtype=torch.long),
        "input_ids": input_ids,
        "labels": labels,
        "prompts": prompts,
        "answers": answers,
    }


def eval_accuracy(model, tokenizer, test_path: str, device: str) -> float:
    # Use existing evaluation helpers from utils for consistency
    # test_model runs the model over the dataset and writes predictions,
    # eval_model reads that file and returns accuracy.
    results_path = os.path.join(os.path.dirname(test_path), "student_eval_results.json")
    # Use a conservative batch size in eval to avoid OOM while keeping eval fast
    test_model(model, tokenizer, test_path, results_path, batch_size=32, max_new_tokens=2, log=False)
    return eval_model(results_path)


def load_cluster_indices(model_name: str, k_classes: int = 8, class_clusters: Optional[List[int]] = None):
    if class_clusters is None:
        class_clusters = [6] * k_classes

    results_dir = os.path.join("..", "results", "neuron-clustering", model_name)
    cluster_indices: Dict[int, Dict[int, torch.Tensor]] = {}

    for subclass in range(k_classes):
        k = class_clusters[subclass]
        ckpt_path = os.path.join(results_dir, f"subclass_{subclass}_clusters", f"k{k}.pt")
        if not os.path.exists(ckpt_path):
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu")
        c2i = ckpt["cluster_to_indices"]
        cluster_indices[subclass] = {int(cid): idx for cid, idx in c2i.items()}

    return cluster_indices


def build_paired_clusters(student_model: str, teacher_model: str, top_k_per_subclass: Optional[int] = None):
    base_results_dir = os.path.join("..", "results", "circuit-discovery")
    s_path = os.path.join(base_results_dir, student_model, "ablation_performance.json")
    t_path = os.path.join(base_results_dir, teacher_model, "ablation_performance.json")

    delta_s = _load_single_ablation_performance(s_path)
    delta_t = _load_single_ablation_performance(t_path)

    mappings = create_cluster_mapping(delta_s, delta_t, top_k_student=top_k_per_subclass)

    paired: Dict[int, List[ClusterMapping]] = {}
    for m in mappings:
        paired.setdefault(m.subclass, []).append(m)
    return paired


def build_cluster_matrix(acts: Dict[int, torch.Tensor], neuron_indices: torch.Tensor, mask: torch.Tensor, intermediate_size: int, device: torch.device) -> torch.Tensor:
    neuron_indices = neuron_indices.to(device)
    rows = []

    for idx in neuron_indices.tolist():
        layer_id = idx // intermediate_size
        neuron_id = idx % intermediate_size
        if layer_id not in acts:
            continue
        layer_acts = acts[layer_id]
        # layer_acts can be [B, T, H] or [B, H] (cached mean over tokens)
        if neuron_id < 0 or neuron_id >= layer_acts.size(-1):
            continue
        if layer_acts.dim() == 3:
            sel = layer_acts[mask, :, neuron_id]  # [b_s, T]
            if sel.numel() == 0:
                continue
            mean_over_tokens = sel.mean(dim=1)  # [b_s]
        elif layer_acts.dim() == 2:
            sel = layer_acts[mask, neuron_id]  # [b_s]
            if sel.numel() == 0:
                continue
            mean_over_tokens = sel
        else:
            continue
        rows.append(mean_over_tokens)

    if not rows:
        return torch.empty(0, 0, device=device)

    mat = torch.stack(rows, dim=1)  # [b_s, num_neurons]
    return mat


def precompute_teacher_cache(config: Config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Precompute mode requires CUDA for efficient teacher forward passes.")

    teacher, tokenizer = load_model(config.teacher_model_name)
    teacher.to(device).eval()

    teacher_cache = ActivationCache()
    teacher_layers = list(range(teacher.config.num_hidden_layers))
    teacher_intermediate = teacher.config.intermediate_size

    dataset = AddDataset(config.train_path)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda ex: collate_batch(ex, tokenizer),
    )

    all_prompts: List[str] = []
    all_answers: List[str] = []
    logits_chunks: List[torch.Tensor] = []
    acts_per_layer: Dict[int, List[torch.Tensor]] = {}

    print("Precomputing teacher logits and mean MLP activations...")

    for batch in tqdm(loader, desc="Precompute teacher", ncols=100):
        input_ids = batch["input_ids"].to(device)
        prompts = batch["prompts"]
        answers = batch["answers"]

        all_prompts.extend(prompts)
        all_answers.extend(answers)

        teacher_cache.register_hooks(teacher, teacher_layers)
        with torch.no_grad():
            outputs = teacher(input_ids=input_ids)
            logits = outputs.logits.detach().cpu()  # [B, T, V]
        t_acts = teacher_cache.activations

        logits_chunks.append(logits)

        # For each hooked layer, compute mean over tokens -> [B, H]
        for layer_idx, act in t_acts.items():
            # act: [B, T, H]
            if act.dim() == 3:
                mean_act = act.mean(dim=1).cpu()  # [B, H]
            elif act.dim() == 2:
                mean_act = act.cpu()
            else:
                continue
            acts_per_layer.setdefault(layer_idx, []).append(mean_act)

        teacher_cache.clear()

    # Stack across batches
    all_logits = torch.cat(logits_chunks, dim=0) if logits_chunks else torch.empty(0)
    acts_stacked: Dict[int, torch.Tensor] = {}
    for layer_idx, chunks in acts_per_layer.items():
        acts_stacked[layer_idx] = torch.cat(chunks, dim=0)

    cache = {
        "prompts": all_prompts,
        "answers": all_answers,
        "logits": all_logits,
        "acts": acts_stacked,
        "intermediate_size": teacher_intermediate,
        "layers": teacher_layers,
    }

    os.makedirs(os.path.dirname(config.teacher_cache_path), exist_ok=True)
    torch.save(cache, config.teacher_cache_path)
    print(f"Saved teacher cache to {config.teacher_cache_path}")


def main():
    config = Config()

    if "--precompute-teacher" in sys.argv:
        precompute_teacher_cache(config)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Student and tokenizer for training
    # Load via utils (which may use fp16), then explicitly convert student to fp32 for stable training
    student, tokenizer = load_model(config.student_model_name)
    student = student.to("cpu")
    student = student.float()
    student = student.to(device)
    student.train()

    # Circuit discovery classifier for subclass labels (move to GPU if available)
    circuit_model, _, _, _ = load_model_checkpoint("model_5000", k_classes=8, lr=1e-3)
    circuit_model.to(device).eval()

    # Load precomputed teacher cache (logits + mean MLP activations)
    if not os.path.exists(config.teacher_cache_path):
        raise FileNotFoundError(
            f"Teacher cache not found at {config.teacher_cache_path}. "
            f"Run: python distillation.py --precompute-teacher"
        )

    teacher_cache = torch.load(config.teacher_cache_path, map_location="cpu")
    teacher_logits_all: torch.Tensor = teacher_cache["logits"]  # keep on CPU to save VRAM
    teacher_acts_all_cpu: Dict[int, torch.Tensor] = teacher_cache["acts"]
    teacher_intermediate = int(teacher_cache["intermediate_size"])
    teacher_layers = list(teacher_cache["layers"])

    student_intermediate = student.config.intermediate_size
    student_layers = list(range(student.config.num_hidden_layers))

    student_cache = ActivationCache()

    student_cluster_indices = load_cluster_indices(config.student_model_name)
    teacher_cluster_indices = load_cluster_indices(config.teacher_model_name)

    paired_clusters = build_paired_clusters(
        config.student_model_name,
        config.teacher_model_name,
        top_k_per_subclass=config.top_k_pairs_per_subclass,
    )

    train_dataset = AddDataset(config.train_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda ex: collate_batch(ex, tokenizer),
    )

    cka_loss_fn = CKALoss().to(device)
    # Use SGD instead of AdamW to reduce optimizer state memory for fp32 1B student
    optimizer = SGD(student.parameters(), lr=config.learning_rate, momentum=0.9)

    baseline_student = eval_accuracy(student, tokenizer, config.test_path, device)

    history = {
        "epoch": [],
        "ce_loss": [],
        "cka_loss": [],
        "total_loss": [],
        "accuracy": [],
    }

    step_losses = []
    step_ce_losses = []
    step_cka_losses = []
    global_step = 0

    print(f"\n{'=' * 60}")
    print(f"Starting training for {config.epochs} epochs")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Lambda CKA: {config.lambda_cka}")
    print(f"{'=' * 60}\n")

    best_accuracy = baseline_student
    temperature = 2.0
    k_classes = 8

    for epoch in range(config.epochs):
        student.train()
        epoch_ce, epoch_cka, epoch_total = 0.0, 0.0, 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        for step, batch in enumerate(pbar):
            indices = batch["indices"]  # [B]
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            prompts = batch["prompts"]
            answers = batch["answers"]

            if (labels != -100).sum().item() == 0:
                continue

            # Teacher logits from cache
            teacher_logits_batch = teacher_logits_all[indices].to(device)

            # Student forward and activations
            student_cache.register_hooks(student, student_layers)
            try:
                student_outputs = student(input_ids=input_ids)
                s_acts = dict(student_cache.activations)
                student_logits = student_outputs.logits

                # Match teacher/student logits shapes if needed
                min_len = min(student_logits.size(1), teacher_logits_batch.size(1))
                student_logits_trim = student_logits[:, :min_len, :]
                teacher_logits_trim = teacher_logits_batch[:, :min_len, :]

                ce_loss = F.kl_div(
                    F.log_softmax(student_logits_trim / temperature, dim=-1),
                    F.softmax(teacher_logits_trim / temperature, dim=-1),
                    reduction="batchmean",
                ) * (temperature ** 2)

                # Circuit discovery classifier expects full equations like "12+34=46".
                full_eqs = [f"{p}{a}" for p, a in zip(prompts, answers)]
                op1, op2, res = parse_equation(full_eqs, device=device)
                with torch.no_grad():
                    logits_cd = circuit_model.classify_problem(op1, op2, res)
                    subclass_ids = logits_cd.argmax(dim=-1).to(device)  # [batch]

                cka_losses = []

                # Build teacher activations for this batch from cache (mean over tokens per neuron)
                # Move the cached activations to GPU once per batch slice.
                t_acts_batch: Dict[int, torch.Tensor] = {
                    layer: acts_layer[indices].to(device)
                    for layer, acts_layer in teacher_acts_all_cpu.items()
                }

                for subclass in range(k_classes):
                    mask = subclass_ids == subclass
                    if mask.sum().item() == 0:
                        continue

                    if subclass not in paired_clusters:
                        continue

                    if subclass not in student_cluster_indices or subclass not in teacher_cluster_indices:
                        continue

                    for m in paired_clusters[subclass]:
                        s_cid = m.student_cluster_idx
                        t_cid = m.teacher_cluster_idx

                        s_neurons = student_cluster_indices[subclass].get(s_cid)
                        t_neurons = teacher_cluster_indices[subclass].get(t_cid)
                        if s_neurons is None or t_neurons is None:
                            continue

                        X = build_cluster_matrix(
                            s_acts,
                            s_neurons,
                            mask,
                            student_intermediate,
                            device=torch.device(device),
                        )
                        Y = build_cluster_matrix(
                            t_acts_batch,
                            t_neurons,
                            mask,
                            teacher_intermediate,
                            device=torch.device(device),
                        )

                        if X.numel() == 0 or Y.numel() == 0:
                            continue

                        if X.shape != Y.shape:
                            min_dim = min(X.shape[1], Y.shape[1])
                            X = X[:, :min_dim]
                            Y = Y[:, :min_dim]

                        loss_val, _ = cka_loss_fn(X, Y)
                        if not torch.isnan(loss_val):
                            cka_losses.append(loss_val)

                cka_loss = (
                    torch.stack(cka_losses).mean().to(device)
                    if cka_losses
                    else torch.tensor(0.0, device=device)
                )

                total_loss = ce_loss + config.lambda_cka * cka_loss

                if torch.isnan(total_loss):
                    continue

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), config.grad_clip)
                optimizer.step()

                epoch_ce += ce_loss.item()
                epoch_cka += cka_loss.item()
                epoch_total += total_loss.item()
                n_batches += 1

                step_losses.append(total_loss.item())
                step_ce_losses.append(ce_loss.item())
                step_cka_losses.append(cka_loss.item())
                global_step += 1

                pbar.set_postfix({"CE": f"{ce_loss.item():.3f}", "CKA": f"{cka_loss.item():.3f}"})
            finally:
                student_cache.clear()

        avg_ce = epoch_ce / n_batches if n_batches > 0 else 0.0
        avg_cka = epoch_cka / n_batches if n_batches > 0 else 0.0
        avg_total = epoch_total / n_batches if n_batches > 0 else 0.0

        accuracy = eval_accuracy(student, tokenizer, config.test_path, device)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            student.save_pretrained("./best_model")
            tokenizer.save_pretrained("./best_model")
            print("    New best model saved!")

        history["epoch"].append(epoch + 1)
        history["ce_loss"].append(avg_ce)
        history["cka_loss"].append(avg_cka)
        history["total_loss"].append(avg_total)
        history["accuracy"].append(accuracy)

        print(
            f"\nEpoch {epoch + 1}/{config.epochs}: CE={avg_ce:.4f}, CKA={avg_cka:.4f}, "
            f"Total={avg_total:.4f}, Acc={accuracy:.3f}"
        )

    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"Best accuracy: {best_accuracy:.3f} (baseline: {baseline_student:.3f})")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()


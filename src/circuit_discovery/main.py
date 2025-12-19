import os
import json
import torch

from transformers import AutoTokenizer

from .utils import (
    s3,
    BUCKET_NAME,
    llama_1b,
    llama_8b,
    parse_equation,
    _stack_layer_activations,
    log_epoch_metrics,
)
from .models import CircuitDiscoveryModel, CircuitLoss, _mean_pairwise_mask_cossim
from utils import list_keys, suffix_map, load_model_checkpoint


def train_circuit_discovery(
    k_classes,
    epochs=1,
    resume_model=None,
    lr=1e-3,
    device=None,
    cache_dir=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(llama_1b)

    if resume_model is None:
        model = CircuitDiscoveryModel(k_classes=k_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        metrics_log = []
        start_epoch = 0
    else:
        model, optimizer, metrics_log, start_epoch = load_model_checkpoint(resume_model, k_classes, lr)

    criterion = CircuitLoss().to(device)

    keys_1b = list_keys(llama_1b)
    keys_8b = list_keys(llama_8b)

    map_1b = suffix_map(keys_1b)
    map_8b = suffix_map(keys_8b)
    shared_suffixes = list(set(map_1b.keys()) & set(map_8b.keys()))
    if not shared_suffixes:
        raise ValueError("No overlapping activation batches found for 1B and 8B models in S3.")

    files_per_epoch = 5

    tokenizer = AutoTokenizer.from_pretrained(llama_1b)

    for epoch in range(start_epoch, epochs):
        start = (epoch * files_per_epoch) % len(shared_suffixes)
        end = start + files_per_epoch
        if end <= len(shared_suffixes):
            epoch_suffixes = shared_suffixes[start:end]
        else:
            epoch_suffixes = shared_suffixes[start:] + shared_suffixes[: end - len(shared_suffixes)]

        all_hard_class_probs = []
        all_masked_1b = []
        all_masked_8b = []
        all_mask_1b = []
        all_mask_8b = []

        frac_1b_list = []
        frac_8b_list = []
        class_ent_list = []

        model.train()
        optimizer.zero_grad()

        if cache_dir is None:
            if os.path.exists("/opt/dlami/nvme"):
                cache_dir_resolved = "/opt/dlami/nvme/activations_cache"
            else:
                cache_dir_resolved = "/mnt/activations_cache"
        else:
            cache_dir_resolved = cache_dir

        os.makedirs(cache_dir_resolved, exist_ok=True)

        for suffix in epoch_suffixes:
            key_1b = map_1b[suffix]
            key_8b = map_8b[suffix]

            local_1b = os.path.join(cache_dir_resolved, f"1b_{suffix}")
            local_8b = os.path.join(cache_dir_resolved, f"8b_{suffix}")

            if not os.path.exists(local_1b):
                s3.download_file(BUCKET_NAME, key_1b, local_1b)
            if not os.path.exists(local_8b):
                s3.download_file(BUCKET_NAME, key_8b, local_8b)

            batch_1b = torch.load(local_1b, map_location="cpu")
            batch_8b = torch.load(local_8b, map_location="cpu")

            ids_1b, activations_dict_1b = batch_1b["ids"], batch_1b["activations"]
            ids_8b, activations_dict_8b = batch_8b["ids"], batch_8b["activations"]

            if not torch.equal(ids_1b, ids_8b):
                continue

            prompts = tokenizer.batch_decode(ids_1b, skip_special_tokens=True)

            activations_1b = _stack_layer_activations(activations_dict_1b).to(device)
            activations_8b = _stack_layer_activations(activations_dict_8b).to(device)

            op1, op2, res = parse_equation(prompts, device=device)

            outputs = model(op1, op2, res, activations_1b, activations_8b)

            hard_class_probs = outputs["hard_class_probs"]
            masked_1b = outputs["masked_activations_1b"]
            masked_8b = outputs["masked_activations_8b"]
            mask_1b = outputs["mask_1b"]
            mask_8b = outputs["mask_8b"]

            all_hard_class_probs.append(hard_class_probs)
            all_masked_1b.append(masked_1b)
            all_mask_1b.append(mask_1b)
            all_masked_8b.append(masked_8b)
            all_mask_8b.append(mask_8b)

            with torch.no_grad():
                frac_1b_list.append(float((mask_1b > (1 - 1e-3)).float().mean()))
                frac_8b_list.append(float((mask_8b > (1 - 1e-3)).float().mean()))
                class_ent_list.append(float(outputs["class_entropy"]))

        if not all_hard_class_probs:
            continue

        hard_class_probs = torch.cat(all_hard_class_probs, dim=0)
        masked_1b = torch.cat(all_masked_1b, dim=0)
        masked_8b = torch.cat(all_masked_8b, dim=0)
        mask_1b = torch.cat(all_mask_1b, dim=0)
        mask_8b = torch.cat(all_mask_8b, dim=0)

        assert torch.isfinite(mask_1b).all(), "mask_1b non-finite"
        assert torch.isfinite(mask_8b).all(), "mask_8b non-finite"
        assert torch.isfinite(masked_1b).all(), "masked_1b non-finite"
        assert torch.isfinite(masked_8b).all(), "masked_8b non-finite"
        assert torch.isfinite(hard_class_probs).all(), "hard_class_probs non-finite"

        loss_dict = criterion(
            hard_class_probs,
            masked_1b,
            masked_8b,
            mask_1b,
            mask_8b,
            model.neuron_masks_1b.class_masks(),
            model.neuron_masks_8b.class_masks(),
        )
        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            class_usage_entropy = float(loss_dict["class_usage_entropy"])

            frac_1b = sum(frac_1b_list) / len(frac_1b_list) if frac_1b_list else float("nan")
            frac_8b = sum(frac_8b_list) / len(frac_8b_list) if frac_8b_list else float("nan")
            class_ent = sum(class_ent_list) / len(class_ent_list) if class_ent_list else float("nan")

            sim_loss_1b = float(loss_dict["sim_1b"])
            sim_loss_8b = float(loss_dict["sim_8b"])
            kl_bernoulli_1b = float(loss_dict["kl_bernoulli_1b"])
            kl_bernoulli_8b = float(loss_dict["kl_bernoulli_8b"])
            mask_cossim_1b_loss = float(loss_dict["mask_cossim_1b"])
            mask_cossim_8b_loss = float(loss_dict["mask_cossim_8b"])

            sparsity_1b = float(criterion.binary_entropy(mask_1b.detach()))
            sparsity_8b = float(criterion.binary_entropy(mask_8b.detach()))

        epoch_metrics = {
            "epoch": epoch + 1,
            "loss": float(loss.item()),
            "sim_loss_1b": float(sim_loss_1b),
            "sim_loss_8b": float(sim_loss_8b),
            "class_usage_entropy": float(class_usage_entropy),
            "frac_activated_1b": float(frac_1b),
            "frac_activated_8b": float(frac_8b),
            "class_entropy": float(class_ent),
            "sparsity_1b": float(sparsity_1b),
            "sparsity_8b": float(sparsity_8b),
            "kl_bernoulli_1b": float(kl_bernoulli_1b),
            "kl_bernoulli_8b": float(kl_bernoulli_8b),
            "mask_cossim_1b_loss": float(mask_cossim_1b_loss),
            "mask_cossim_8b_loss": float(mask_cossim_8b_loss),
        }

        log_epoch_metrics(epoch_metrics)

        metrics_log.append(epoch_metrics)

        results_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "circuit-discovery")
        os.makedirs(results_dir, exist_ok=True)
        metrics_path = os.path.join(results_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_log, f, indent=4)

        if (epoch + 1) % 500 == 0:
            if os.path.exists("/opt/dlami/nvme"):
                ckpt_root = "/opt/dlami/nvme/circuit_discovery_checkpoints"
            else:
                ckpt_root = os.path.join(results_dir, "checkpoints")

            os.makedirs(ckpt_root, exist_ok=True)
            ckpt_path = os.path.join(ckpt_root, f"epoch_{epoch+1}.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics_log": metrics_log,
                },
                ckpt_path,
            )

            s3.upload_file(ckpt_path, BUCKET_NAME, f"circuit-discovery/model_{epoch+1}")

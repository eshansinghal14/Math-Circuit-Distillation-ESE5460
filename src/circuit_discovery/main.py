import math
import os
import json
import torch

from transformers import AutoTokenizer

from .utils import (
    llama_1b,
    llama_8b,
    parse_equation,
    merge_activation_batches,
    _stack_layer_activations,
    log_epoch_metrics,
)
from .models import CircuitDiscoveryModel, CircuitLoss
from .loss_scheduler import LossScheduler


def train_circuit_discovery(
    k_classes,
    epochs=1,
    resume_model=None,
    lr=1e-3,
    device=None,
    files_per_epoch=5,
    use_loss_scheduler=True,
    orthogonality_threshold=0.35,
    balance_ratio=0.85,
    loss_scheduler_kwargs=None,
):
    from utils import load_model_checkpoint
    from gen_activations_dataset import NeuronActivationsGenerator

    def _generate_and_merge_batches(act_generator, batch_indices):
        batches = []
        for i in batch_indices:
            batches.append(act_generator.generate_batch_activations(i, log=False))
        return merge_activation_batches(batches)

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
    scheduler_kw = dict(loss_scheduler_kwargs or {})
    scheduler_kw.setdefault("orthogonality_threshold", orthogonality_threshold)
    scheduler_kw.setdefault("balance_ratio", balance_ratio)
    loss_scheduler = (
        LossScheduler(k_classes=k_classes, **scheduler_kw)
        if use_loss_scheduler
        else None
    )

    act_generator_1b = NeuronActivationsGenerator(llama_1b, batch_size=50)
    act_generator_8b = NeuronActivationsGenerator(llama_8b, batch_size=50)
    act_generators = {
        "1b": act_generator_1b,
        "8b": act_generator_8b,
    }

    num_examples = act_generator_1b.ids.shape[0]
    batch_size = act_generator_1b.batch_size
    num_batches = (num_examples + batch_size - 1) // batch_size

    for epoch in range(start_epoch, epochs):
        # Update loss weights from scheduler (use previous epoch metrics after first epoch)
        if loss_scheduler is not None:
            prev_metrics = metrics_log[-1] if metrics_log else {}
            lambdas = loss_scheduler.get_lambdas(epoch, prev_metrics)
            criterion.update_lambdas(**lambdas)
        else:
            lambdas = {}

        # choose files_per_epoch batch indices for this epoch (wrap around)
        start = (epoch * files_per_epoch) % num_batches
        batch_indices = [(start + offset) % num_batches for offset in range(files_per_epoch)]

        # Generate the individual batch activation files (they are saved to
        # `activations_{model_name}.pt` by the generator). Then read those
        # temporary files and merge them layer-wise so we can process a
        # single, larger batch in-place below.
        merged = {
            key: _generate_and_merge_batches(gen, batch_indices)
            for key, gen in act_generators.items()
        }

        # if end <= len(shared_suffixes):
        #     epoch_suffixes = shared_suffixes[start:end]
        # else:
        #     epoch_suffixes = shared_suffixes[start:] + shared_suffixes[: end - len(shared_suffixes)]

        model.train()
        optimizer.zero_grad()

        ids = {key: batch["ids"] for key, batch in merged.items()}
        ref_ids = ids["1b"]
        if not torch.equal(ref_ids, ids["8b"]):
            continue

        prompts = tokenizer.batch_decode(ref_ids, skip_special_tokens=True)

        stacked_activations = {
            key: _stack_layer_activations(batch["activations"]).to(device)
            for key, batch in merged.items()
        }

        op1, op2, res = parse_equation(prompts, device=device)

        outputs = model(op1, op2, res, stacked_activations["1b"], stacked_activations["8b"])

        hard_class_probs = outputs["hard_class_probs"]
        masked_1b = outputs["masked_activations_1b"]
        masked_8b = outputs["masked_activations_8b"]
        mask_1b = outputs["mask_1b"]
        mask_8b = outputs["mask_8b"]

        with torch.no_grad():
            frac_1b = float((mask_1b > (1 - 1e-3)).float().mean())
            frac_8b = float((mask_8b > (1 - 1e-3)).float().mean())
            class_ent = float(outputs["class_entropy"])

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

            sim_loss_1b = float(loss_dict["sim_1b"])
            sim_loss_8b = float(loss_dict["sim_8b"])
            kl_bernoulli_1b = float(loss_dict["kl_bernoulli_1b"])
            kl_bernoulli_8b = float(loss_dict["kl_bernoulli_8b"])
            mask_cossim_1b_loss = float(loss_dict["mask_cossim_1b"])
            mask_cossim_8b_loss = float(loss_dict["mask_cossim_8b"])

            sparsity_1b = float(criterion.binary_entropy(mask_1b.detach()))
            sparsity_8b = float(criterion.binary_entropy(mask_8b.detach()))

        max_class_usage_entropy = math.log(k_classes) if k_classes > 0 else 0.0
        epoch_metrics = {
            "epoch": epoch + 1,
            "loss": float(loss.item()),
            "sim_loss_1b": float(sim_loss_1b),
            "sim_loss_8b": float(sim_loss_8b),
            "class_usage_entropy": float(class_usage_entropy),
            "max_class_usage_entropy": float(max_class_usage_entropy),
            "frac_activated_1b": float(frac_1b),
            "frac_activated_8b": float(frac_8b),
            "class_entropy": float(class_ent),
            "sparsity_1b": float(sparsity_1b),
            "sparsity_8b": float(sparsity_8b),
            "kl_bernoulli_1b": float(kl_bernoulli_1b),
            "kl_bernoulli_8b": float(kl_bernoulli_8b),
            "mask_cossim_1b_loss": float(mask_cossim_1b_loss),
            "mask_cossim_8b_loss": float(mask_cossim_8b_loss),
            "loss_phase": loss_scheduler.phase if loss_scheduler is not None else -1,
            **lambdas,
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

            # Store checkpoints locally only (do not upload to S3)
            print(f"Saved checkpoint to {ckpt_path}")

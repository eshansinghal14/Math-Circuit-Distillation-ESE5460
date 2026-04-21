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
    tower_grad_balance_weights,
)
from .models import CircuitDiscoveryModel, CircuitLoss


def train_circuit_discovery(
    k_classes,
    dataset_prefix,
    epochs=1,
    resume_model=None,
    lr=1e-3,
    device=None,
    files_per_epoch=5,
    lambda_usage=0.15,
    lambda_mask_cossim=0.25,
    lambda_kl=0.15,
    lambda_sparsity=0.20,
    mask_temperature=1.0,
    mask_activate_threshold=0.99,
    grad_clip_norm=1.0,
    class_reweight=False,
    balance_tower_grads=True,
):
    from utils import load_model_checkpoint
    from neuron_distillation.activations import NeuronActivationsGenerator

    def _generate_and_merge_batches(act_generator, batch_indices):
        batches = []
        for i in batch_indices:
            batches.append(act_generator.generate_batch_activations(i, log=False))
        return merge_activation_batches(batches)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from utils import patch_tokenizer_no_special_tokens

    tokenizer = patch_tokenizer_no_special_tokens(
        AutoTokenizer.from_pretrained(llama_1b),
    )

    if resume_model is None:
        model = CircuitDiscoveryModel(k_classes=k_classes, mask_temperature=mask_temperature).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        metrics_log = []
        start_epoch = 0
    else:
        model, optimizer, metrics_log, start_epoch = load_model_checkpoint(resume_model, k_classes, lr)

    # Training hyperparameter (overrides value from checkpoint on resume)
    model.mask_temperature.fill_(float(mask_temperature))

    # Sim loss weight = 1 - sum of auxiliary weights so total weights sum to 1
    lambda_sim = 1.0 - (lambda_usage + lambda_mask_cossim + lambda_kl + lambda_sparsity)
    if lambda_sim <= 0:
        raise ValueError(
            "Auxiliary weights must sum to < 1 (lambda_usage + lambda_mask_cossim + lambda_kl + lambda_sparsity < 1)"
        )
    criterion = CircuitLoss(
        lambda_sim=lambda_sim,
        lambda_usage=lambda_usage,
        lambda_mask_cossim=lambda_mask_cossim,
        lambda_kl=lambda_kl,
        lambda_sparsity=lambda_sparsity,
        class_reweight=class_reweight,
    ).to(device)

    act_generator_1b = NeuronActivationsGenerator(
        llama_1b, batch_size=50, dataset_prefix=dataset_prefix,
    )
    act_generator_8b = NeuronActivationsGenerator(
        llama_8b, batch_size=50, dataset_prefix=dataset_prefix,
    )
    act_generators = {
        "1b": act_generator_1b,
        "8b": act_generator_8b,
    }

    num_examples = act_generator_1b.ids.shape[0]
    batch_size = act_generator_1b.batch_size
    num_batches = (num_examples + batch_size - 1) // batch_size

    results_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "circuit-discovery")
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.json")

    for epoch in range(start_epoch, epochs):
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
            del merged
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        prompts = tokenizer.batch_decode(ref_ids, skip_special_tokens=True)

        stacked_activations = {
            key: _stack_layer_activations(batch["activations"]).to(device)
            for key, batch in merged.items()
        }

        op1, op2, res = parse_equation(prompts, device=device)

        T = model.mask_temperature
        outputs = model(op1, op2, res, stacked_activations["1b"], stacked_activations["8b"])

        logits = outputs["logits"]
        hard_class_probs = outputs["hard_class_probs"]
        masked_1b = outputs["masked_activations_1b"]
        masked_8b = outputs["masked_activations_8b"]
        mask_1b = outputs["mask_1b"]
        mask_8b = outputs["mask_8b"]

        with torch.no_grad():
            thr = mask_activate_threshold
            frac_1b = float((mask_1b > thr).float().mean())
            frac_8b = float((mask_8b > thr).float().mean())
            class_ent = float(outputs["class_entropy"])

        assert torch.isfinite(mask_1b).all(), "mask_1b non-finite"
        assert torch.isfinite(mask_8b).all(), "mask_8b non-finite"
        assert torch.isfinite(masked_1b).all(), "masked_1b non-finite"
        assert torch.isfinite(masked_8b).all(), "masked_8b non-finite"
        assert torch.isfinite(hard_class_probs).all(), "hard_class_probs non-finite"

        loss_dict = criterion(
            logits,
            hard_class_probs,
            masked_1b,
            masked_8b,
            mask_1b,
            mask_8b,
            model.neuron_masks_1b.class_masks(T),
            model.neuron_masks_8b.class_masks(T),
        )
        lt1 = loss_dict["tower_loss_1b"]
        lt2 = loss_dict["tower_loss_8b"]
        usage_ent = loss_dict["class_usage_entropy"]
        if balance_tower_grads:
            w1, w2, gn1, gn2 = tower_grad_balance_weights(lt1, lt2, model.parameters())
            loss = w1 * lt1 + w2 * lt2 - criterion.lambda_usage * usage_ent
        else:
            loss = lt1 + lt2 - criterion.lambda_usage * usage_ent
            w1 = w2 = 0.5
            gn1 = gn2 = float("nan")
        # Always backward so the autograd graph is freed this iteration. Skipping backward
        # when loss is NaN leaves huge intermediates (1b/8b activations path) alive until the
        # loop body ends and can spike peak VRAM when the next epoch allocates.
        loss.backward()
        grad_finite = all(
            p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
        )
        loss_finite = torch.isfinite(loss).all().item()
        if grad_finite and grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        if loss_finite and grad_finite:
            optimizer.step()
        else:
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            class_usage_entropy = float(loss_dict["class_usage_entropy"].detach())

            sim_loss_1b = float(loss_dict["sim_1b"])
            sim_loss_8b = float(loss_dict["sim_8b"])
            kl_bernoulli_1b = float(loss_dict["kl_bernoulli_1b"])
            kl_bernoulli_8b = float(loss_dict["kl_bernoulli_8b"])
            mask_cossim_1b_loss = float(loss_dict["mask_cossim_1b"])
            mask_cossim_8b_loss = float(loss_dict["mask_cossim_8b"])

            sparsity_1b = float(criterion.binary_entropy(mask_1b.detach()))
            sparsity_8b = float(criterion.binary_entropy(mask_8b.detach()))

            # Number of problems assigned to each class using argmax over logits
            preds = logits.argmax(dim=-1)
            class_counts = torch.bincount(preds, minlength=k_classes).cpu().tolist()
            thr_cls = mask_activate_threshold
            cm1 = model.neuron_masks_1b.class_masks(T)
            cm8 = model.neuron_masks_8b.class_masks(T)
            prop_active_neurons_1b_per_class = [
                float((cm1[c] > thr_cls).float().mean().item()) for c in range(k_classes)
            ]
            prop_active_neurons_8b_per_class = [
                float((cm8[c] > thr_cls).float().mean().item()) for c in range(k_classes)
            ]

        max_class_usage_entropy = math.log(k_classes) if k_classes > 0 else 0.0
        epoch_metrics = {
            "epoch": epoch + 1,
            "loss": float(loss.item()),
            "loss_unweighted": float(loss_dict["loss"].detach().item()),
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
            "class_counts": class_counts,
            "prop_active_neurons_1b_per_class": prop_active_neurons_1b_per_class,
            "prop_active_neurons_8b_per_class": prop_active_neurons_8b_per_class,
        }
        if balance_tower_grads:
            epoch_metrics["tower_balance_w1"] = w1
            epoch_metrics["tower_balance_w2"] = w2
            epoch_metrics["tower_grad_norm_1b"] = gn1
            epoch_metrics["tower_grad_norm_8b"] = gn2

        log_epoch_metrics(epoch_metrics)

        # Overwrite existing epoch entry when resuming from checkpoint, else append
        if epoch < len(metrics_log):
            metrics_log[epoch] = epoch_metrics
        else:
            metrics_log.append(epoch_metrics)

        # Write metrics to JSON in real time (exclude verbose per-class breakdown)
        _skip_json = {
            "class_counts",
            "prop_active_neurons_1b_per_class",
            "prop_active_neurons_8b_per_class",
        }
        metrics_for_json = [{k: v for k, v in m.items() if k not in _skip_json} for m in metrics_log]
        with open(metrics_path, "w") as f:
            json.dump(metrics_for_json, f, indent=4)

        if (epoch + 1) % 100 == 0:
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

        del merged, stacked_activations, outputs, hard_class_probs
        del masked_1b, masked_8b, mask_1b, mask_8b, loss_dict, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

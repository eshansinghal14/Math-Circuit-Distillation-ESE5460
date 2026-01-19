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

from gen_activations_dataset import NeuronActivationsGenerator


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

    # keys_1b = list_keys(llama_1b)
    # keys_8b = list_keys(llama_8b)

    # map_1b = suffix_map(keys_1b)
    # map_8b = suffix_map(keys_8b)
    # shared_suffixes = list(set(map_1b.keys()) & set(map_8b.keys()))
    # if not shared_suffixes:
    #     raise ValueError("No overlapping activation batches found for 1B and 8B models in S3.")

    files_per_epoch = 5

    tokenizer = AutoTokenizer.from_pretrained(llama_1b)

    act_generator_1b = NeuronActivationsGenerator(llama_1b, batch_size=50)
    act_generator_8b = NeuronActivationsGenerator(llama_8b, batch_size=50)

    dataset_size = act_generator_1b.ids.shape[0]

    for epoch in range(start_epoch, epochs):
        start = (epoch * files_per_epoch) % dataset_size
        end = start + files_per_epoch

        # Generate the individual batch activation files (they are saved to
        # `activations_{model_name}.pt` by the generator). Then read those
        # temporary files and merge them layer-wise so we can process a
        # single, larger batch in-place below.
        batches_1b = []
        batches_8b = []
        for i in range(start, end):
            act_generator_1b.generate_batch_activations(i, log=False)
            batch1 = torch.load(f"activations_{llama_1b}.pt", map_location="cpu")
            batches_1b.append(batch1)

            act_generator_8b.generate_batch_activations(i, log=False)
            batch8 = torch.load(f"activations_{llama_8b}.pt", map_location="cpu")
            batches_8b.append(batch8)

        def _merge_batches(batches):
            # batches: list of dicts {'ids': Tensor, 'activations': {layer_idx: Tensor}}
            merged = {}
            ids_list = []
            for b in batches:
                ids_list.append(b['ids'])
                for layer_idx, t in b['activations'].items():
                    merged.setdefault(layer_idx, []).append(t)
            ids_cat = torch.cat(ids_list, dim=0) if ids_list else torch.empty(0, dtype=torch.long)
            for k, chunks in list(merged.items()):
                merged[k] = torch.cat(chunks, dim=0)
            return {'ids': ids_cat, 'activations': merged}

        merged_1b = _merge_batches(batches_1b)
        merged_8b = _merge_batches(batches_8b)

        # We'll process the merged pair as a single item in the same style as
        # the later loop did for each suffix/file pair.
        merged_pairs = [(merged_1b, merged_8b)]

        # if end <= len(shared_suffixes):
        #     epoch_suffixes = shared_suffixes[start:end]
        # else:
        #     epoch_suffixes = shared_suffixes[start:] + shared_suffixes[: end - len(shared_suffixes)]

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

        for merged_1b_batch, merged_8b_batch in merged_pairs:
            ids_1b, activations_dict_1b = merged_1b_batch['ids'], merged_1b_batch['activations']
            ids_8b, activations_dict_8b = merged_8b_batch['ids'], merged_8b_batch['activations']

            if not torch.equal(ids_1b, ids_8b):
                # If IDs don't match between the two models' merged batches,
                # skip this merged pair.
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

            # Store checkpoints locally only (do not upload to S3)
            print(f"Saved checkpoint to {ckpt_path}")

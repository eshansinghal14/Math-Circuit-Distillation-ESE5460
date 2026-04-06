import torch
from torch import nn
from torch.nn import functional as F

from .utils import config


class ProblemEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.op1_emb_layer = nn.Embedding(100, embedding_dim // 4)
        self.op2_emb_layer = nn.Embedding(100, embedding_dim // 4)
        self.sum_emb_layer = nn.Embedding(200, embedding_dim // 2)

    def forward(self, op1, op2, res):
        op1_emb = self.op1_emb_layer(op1)
        op2_emb = self.op2_emb_layer(op2)
        sum_emb = self.sum_emb_layer(res)
        return torch.cat((op1_emb, op2_emb, sum_emb), dim=-1)


class ProblemClassifier(nn.Module):
    def __init__(self, input_dim, k_classes, hidden1_dim=256, hidden2_dim=32):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, k_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class NeuronMask(nn.Module):
    _SIGMOID_CLAMP = 60.0  # keeps sigmoid and its backward numerically stable

    def __init__(self, k_classes, activations_dim):
        super().__init__()

        hidden_dim = 4
        self.k_classes = k_classes
        self.class_embedding = nn.Embedding(k_classes, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, activations_dim)

    def _sigmoid_mask(self, logits, mask_temperature):
        t = mask_temperature.clamp(min=1e-6)
        x = logits / t
        x = torch.nan_to_num(x, nan=0.0, posinf=self._SIGMOID_CLAMP, neginf=-self._SIGMOID_CLAMP)
        x = x.clamp(min=-self._SIGMOID_CLAMP, max=self._SIGMOID_CLAMP)
        out = torch.sigmoid(x)
        return torch.nan_to_num(out, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    def forward(self, class_probs, activations, mask_temperature):
        class_ids = class_probs.argmax(dim=-1)
        hidden = self.class_embedding(class_ids)
        hidden = F.relu(hidden)
        selected_mask = self.output_layer(hidden)
        sigmoid_mask = self._sigmoid_mask(selected_mask, mask_temperature)
        sigmoid_mask_expanded = sigmoid_mask.unsqueeze(1)

        masked_activations = activations * sigmoid_mask_expanded
        return masked_activations, sigmoid_mask

    def class_masks(self, mask_temperature):
        device = self.class_embedding.weight.device
        class_ids = torch.arange(self.k_classes, device=device)
        hidden = self.class_embedding(class_ids)
        hidden = F.relu(hidden)
        masks = self.output_layer(hidden)
        return self._sigmoid_mask(masks, mask_temperature)


class CircuitDiscoveryModel(nn.Module):
    def __init__(
        self,
        k_classes,
        problem_embedding_dim=256,
        activation_embedding_dim=1024,
        tau=0.5,
        mask_temperature=1.0,
    ):
        super().__init__()

        self.tau = tau
        num_activations_1b = config["1b"].intermediate_size * config["1b"].num_hidden_layers
        num_activations_8b = config["8b"].intermediate_size * config["8b"].num_hidden_layers

        self.problem_encoder = ProblemEncoder(embedding_dim=problem_embedding_dim)
        self.classifier = ProblemClassifier(problem_embedding_dim, k_classes)

        self.neuron_masks_1b = NeuronMask(k_classes, num_activations_1b)
        self.neuron_masks_8b = NeuronMask(k_classes, num_activations_8b)
        # sigmoid(logits / T); lower T sharpens masks. Constant for the lifetime of this model (saved in ckpt).
        self.register_buffer("mask_temperature", torch.tensor(float(mask_temperature)))

    def classify_problem(self, op1, op2, res):
        problem_encoding = self.problem_encoder(op1, op2, res)
        logits = self.classifier(problem_encoding)
        return logits

    def forward(self, op1, op2, res, activations_1b, activations_8b):
        logits = self.classify_problem(op1, op2, res)
        hard_class_probs = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)

        T = self.mask_temperature
        masked_activations_1b, mask_1b = self.neuron_masks_1b(hard_class_probs, activations_1b, T)
        masked_activations_8b, mask_8b = self.neuron_masks_8b(hard_class_probs, activations_8b, T)

        with torch.no_grad():
            soft_class_probs = F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=-1)
            class_entropy = -(soft_class_probs * torch.log(soft_class_probs)).sum(dim=-1).mean()

        return {
            "logits": logits,
            "hard_class_probs": hard_class_probs,
            "masked_activations_1b": masked_activations_1b,
            "masked_activations_8b": masked_activations_8b,
            "mask_1b": mask_1b,
            "mask_8b": mask_8b,
            "class_entropy": class_entropy.detach(),
        }


class CircuitLoss(nn.Module):
    def __init__(
        self,
        lambda_sim=1.0,
        lambda_sparsity=5e-0,
        lambda_usage=1,
        lambda_kl=1e-1,
        lambda_mask_cossim=5,
        eps=1e-8,
        class_reweight=False,
    ):
        super().__init__()
        self.lambda_sim = lambda_sim
        self.lambda_sparsity = lambda_sparsity
        self.lambda_usage = lambda_usage
        self.lambda_kl = lambda_kl
        self.lambda_mask_cossim = lambda_mask_cossim
        self.eps = eps
        self.class_reweight = class_reweight

    def update_lambdas(
        self,
        lambda_sim=None,
        lambda_sparsity=None,
        lambda_usage=None,
        lambda_kl=None,
        lambda_mask_cossim=None,
    ):
        """Update loss weights. Omitted keys are left unchanged."""
        if lambda_sim is not None:
            self.lambda_sim = lambda_sim
        if lambda_sparsity is not None:
            self.lambda_sparsity = lambda_sparsity
        if lambda_usage is not None:
            self.lambda_usage = lambda_usage
        if lambda_kl is not None:
            self.lambda_kl = lambda_kl
        if lambda_mask_cossim is not None:
            self.lambda_mask_cossim = lambda_mask_cossim

    @staticmethod
    def inverse_frequency_class_weights(hard_class_probs, eps=1e-6):
        """Per-class weights inversely related to batch counts (add-one smoothed), normalized to mean 1."""
        counts = hard_class_probs.float().sum(dim=0)
        w = 1.0 / (counts + 1.0)
        w = w * w.numel() / (w.sum() + eps)
        return w

    def mean_pairwise_mask_cossim(self, masks):
        if masks.dim() != 2:
            return masks.new_tensor(0.0)

        num_classes = masks.size(0)
        if num_classes < 2:
            return masks.new_tensor(0.0)

        norm_masks = F.normalize(masks, p=2, dim=-1, eps=self.eps)
        sim_mat = norm_masks @ norm_masks.t()

        triu_indices = torch.triu_indices(num_classes, num_classes, offset=1, device=masks.device)
        pair_sims = sim_mat[triu_indices[0], triu_indices[1]]
        if pair_sims.numel() == 0:
            return masks.new_tensor(0.0)

        return pair_sims.mean()

    def classwise_pairwise_cossim(self, activations, hard_class_probs, class_weights=None):
        _, k_classes = hard_class_probs.shape

        if activations.dim() != 3:
            raise ValueError(f"Expected activations to have shape [B, T, D] (dim=3), got {tuple(activations.shape)}")

        # Token-wise comparison: keep token dimension and normalize per token.
        # activations: [B, T, D]
        norm_acts = F.normalize(activations, p=2, dim=-1, eps=self.eps)

        per_class_sims = []
        included_weights = []
        for k in range(k_classes):
            class_mask = hard_class_probs[:, k].bool()
            idx = class_mask.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() < 2:
                continue

            acts_k = norm_acts[idx]
            n_k = acts_k.size(0)

            # acts_k: [n_k, T, D]
            # For each token t, compute mean pairwise cosine similarity among examples in this class.
            sum_vec = acts_k.sum(dim=0)  # [T, D]
            sum_sq = (sum_vec * sum_vec).sum(dim=-1)  # [T]

            total_pair_sum = sum_sq - n_k
            num_pairs = n_k * (n_k - 1)
            per_token_sim = total_pair_sum / num_pairs  # [T]
            per_class_sims.append(per_token_sim.mean())
            if class_weights is not None:
                included_weights.append(class_weights[k])

        if not per_class_sims:
            return activations.new_tensor(0.0)
        stacked = torch.stack(per_class_sims)
        if class_weights is None:
            return stacked.mean()
        iw = torch.stack(included_weights)
        return (stacked * iw).sum() / (iw.sum() + self.eps)

    def binary_entropy(self, p, sample_weights=None):
        p = torch.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
        p = torch.clamp(p, self.eps, 1.0 - self.eps)
        entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
        if sample_weights is None:
            return entropy.mean()
        per_ex = entropy.view(entropy.size(0), -1).mean(dim=-1)
        sw = sample_weights / (sample_weights.mean() + self.eps)
        return (per_ex * sw).mean()

    def class_usage_entropy(self, hard_class_probs):
        class_freq = hard_class_probs.float().mean(dim=0)
        class_freq = torch.clamp(class_freq, self.eps, 1.0)
        class_usage_entropy = -(class_freq * class_freq.log()).sum()
        return class_usage_entropy

    def soft_usage_entropy_from_logits(self, logits):
        probs = F.softmax(logits, dim=-1)
        class_freq = probs.mean(dim=0)
        class_freq = torch.clamp(class_freq, self.eps, 1.0)
        return -(class_freq * class_freq.log()).sum()

    def bernoulli_kl_to_prior(self, class_masks, class_weights=None, pi=0.10, eps=1e-8):
        """KL(Bernoulli(q_k) || Bernoulli(pi)) per class k, with q_k = row mean of class k."""
        p = torch.clamp(class_masks, eps, 1.0 - eps)
        if p.dim() != 2:
            raise ValueError(f"Expected class_masks [K, D], got shape {tuple(p.shape)}")
        q = p.mean(dim=-1)
        q = torch.clamp(q, eps, 1.0 - eps)

        pi_t = torch.clamp(torch.tensor(pi, device=p.device, dtype=p.dtype), eps, 1.0 - eps)
        kl = q * (q / pi_t).log() + (1 - q) * ((1 - q) / (1 - pi_t)).log()
        if class_weights is None:
            return kl.mean()
        cw = class_weights / (class_weights.mean() + self.eps)
        return (kl * cw).mean()

    def combined_loss(self, hard_class_probs, masked_activations, mask, class_masks):
        if self.class_reweight:
            class_weights = self.inverse_frequency_class_weights(hard_class_probs)
            class_ids = hard_class_probs.argmax(dim=-1)
            sample_weights = class_weights[class_ids]
        else:
            class_weights = None
            sample_weights = None

        sim_loss = -self.classwise_pairwise_cossim(
            masked_activations, hard_class_probs, class_weights=class_weights
        )
        mask_cossim = self.mean_pairwise_mask_cossim(class_masks)
        kl_bernoulli_loss = self.bernoulli_kl_to_prior(class_masks, class_weights=class_weights)
        entropy_loss = self.binary_entropy(mask, sample_weights=sample_weights)

        total_loss = (
            self.lambda_sim * sim_loss
            + self.lambda_mask_cossim * mask_cossim
            + self.lambda_kl * kl_bernoulli_loss
            + self.lambda_sparsity * entropy_loss
        )
        return total_loss, sim_loss, kl_bernoulli_loss, entropy_loss, mask_cossim

    def forward(self, logits, hard_class_probs, masked_activations_1b, masked_activations_8b, mask_1b, mask_8b, class_masks_1b, class_masks_8b):
        assert torch.isfinite(class_masks_1b).all(), "class_masks_1b non-finite"
        loss_1b, sim_loss_1b, kl_bernoulli_loss_1b, entropy_loss_1b, mask_cossim_1b = self.combined_loss(hard_class_probs, masked_activations_1b, mask_1b, class_masks_1b)
        loss_8b, sim_loss_8b, kl_bernoulli_loss_8b, entropy_loss_8b, mask_cossim_8b = self.combined_loss(hard_class_probs, masked_activations_8b, mask_8b, class_masks_8b)
        total_loss = loss_1b + loss_8b

        class_usage_entropy = self.soft_usage_entropy_from_logits(logits)
        total_loss = total_loss - self.lambda_usage * class_usage_entropy

        return {
            "loss": total_loss,
            "tower_loss_1b": loss_1b,
            "tower_loss_8b": loss_8b,
            "class_usage_entropy": class_usage_entropy,
            "sim_1b": sim_loss_1b.detach(),
            "sim_8b": sim_loss_8b.detach(),
            "kl_bernoulli_1b": kl_bernoulli_loss_1b.detach(),
            "kl_bernoulli_8b": kl_bernoulli_loss_8b.detach(),
            "entropy_1b": entropy_loss_1b.detach(),
            "entropy_8b": entropy_loss_8b.detach(),
            "mask_cossim_1b": mask_cossim_1b.detach(),
            "mask_cossim_8b": mask_cossim_8b.detach(),
        }

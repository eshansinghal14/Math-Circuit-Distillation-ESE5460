import json
import os
import matplotlib.pyplot as plt

metrics_path = "./metrics.json"
with open(metrics_path, "r") as f:
    metrics = sorted(json.load(f), key=lambda x: x["epoch"])

epochs = [m["epoch"] for m in metrics]

def series(key):
    return [m.get(key, float("nan")) for m in metrics]


frac_1b = series("frac_activated_1b")
frac_8b = series("frac_activated_8b")
sparsity_1b = series("sparsity_1b")
sparsity_8b = series("sparsity_8b")
kl_1b = series("kl_bernoulli_1b")
kl_8b = series("kl_bernoulli_8b")
sim_1b = series("sim_loss_1b")
sim_8b = series("sim_loss_8b")
mask_cossim_1b = series("mask_cossim_1b_loss")
mask_cossim_8b = series("mask_cossim_8b_loss")


out_dir = "figures"
os.makedirs(out_dir, exist_ok=True)

color_1b = '#2E86AB'
color_8b = '#A23B72'
color_kl_1b = '#F18F01'
color_kl_8b = '#C73E1D'

fig, ax = plt.subplots(figsize=(7.5, 4.5))

ax.plot(
    epochs, frac_1b,
    label="Frac. active (1B)",
    color=color_1b,
    linestyle="-",
    linewidth=2
)
ax.plot(
    epochs, frac_8b,
    label="Frac. active (8B)",
    color=color_8b,
    linestyle="-",
    linewidth=2
)
ax.plot(
    epochs, sparsity_1b,
    label="Sparsity (1B, bin-entropy)",
    color=color_1b,
    linestyle="--",
    linewidth=1.5,
    alpha=0.8
)
ax.plot(
    epochs, sparsity_8b,
    label="Sparsity (8B, bin-entropy)",
    color=color_8b,
    linestyle="--",
    linewidth=1.5,
    alpha=0.8
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Fraction / Binary entropy")

ax2 = ax.twinx()
ax2.plot(
    epochs, kl_1b,
    label="KL-to-prior (1B)",
    color=color_kl_1b,
    linestyle="-",
    linewidth=2,
    marker='o',
    markersize=3,
    markevery=max(1, len(epochs)//10)
)
ax2.plot(
    epochs, kl_8b,
    label="KL-to-prior (8B)",
    color=color_kl_8b,
    linestyle="-",
    linewidth=2,
    marker='s',
    markersize=3,
    markevery=max(1, len(epochs)//10)
)
ax2.set_ylabel(r"KL$(\mathrm{Bern}(q)\,\|\,\mathrm{Bern}(\pi))$")

lines = ax.get_lines() + ax2.get_lines()
labels = [line.get_label() for line in lines]
ax.legend(lines, labels, fontsize=9, loc="best", framealpha=0.9)

plt.title("Mask sparsification dynamics (target $\\pi \\approx 0.1$)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "cd_sparsify.png"), dpi=300)
plt.close()


color_sim_1b = '#06A77D'
color_sim_8b = '#005F73'
color_mask_1b = '#D62828'
color_mask_8b = '#F77F00'

fig, ax = plt.subplots(figsize=(7.5, 4.5))

ax.plot(
    epochs, sim_1b,
    label="Similarity loss (1B)",
    color=color_sim_1b,
    linestyle="-",
    linewidth=2
)
ax.plot(
    epochs, sim_8b,
    label="Similarity loss (8B)",
    color=color_sim_8b,
    linestyle="-",
    linewidth=2
)
ax.set_xlabel("Epoch")
ax.set_ylabel("Similarity loss")

ax2 = ax.twinx()
ax2.plot(
    epochs, mask_cossim_1b,
    label="Mask cosine sim (1B)",
    color=color_mask_1b,
    linestyle="-",
    linewidth=2,
    marker='o',
    markersize=3,
    markevery=max(1, len(epochs)//10)
)
ax2.plot(
    epochs, mask_cossim_8b,
    label="Mask cosine sim (8B)",
    color=color_mask_8b,
    linestyle="-",
    linewidth=2,
    marker='s',
    markersize=3,
    markevery=max(1, len(epochs)//10)
)
ax2.set_ylabel("Mean pairwise mask cosine similarity")

lines = ax.get_lines() + ax2.get_lines()
labels = [line.get_label() for line in lines]
ax.legend(lines, labels, fontsize=9, loc="best", framealpha=0.9)

plt.title("Representativeness vs orthogonality during circuit discovery")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "cd_rep_ortho.png"), dpi=300)
plt.close()
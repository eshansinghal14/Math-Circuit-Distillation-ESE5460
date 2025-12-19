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

plt.rcParams.update({
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

lw_main = 1.3
lw_aux = 1.1
alpha_1b = 0.4
alpha_8b = 1.0

FRAC     = "#1f77b4"  # blue
SPARSE   = "#9467bd"  # purple
KL       = "#ff7f0e"  # orange
SIM      = "#2ca02c"  # green
MASKCOS  = "#d62728"  # red

fig, ax = plt.subplots(figsize=(7.5, 4.5))

ax.plot(
    epochs, frac_1b,
    label="Frac. active (1B)",
    color=FRAC,
    linewidth=lw_main,
    alpha=alpha_1b
)
ax.plot(
    epochs, frac_8b,
    label="Frac. active (8B)",
    color=FRAC,
    linewidth=lw_main,
    alpha=alpha_8b
)

ax.plot(
    epochs, sparsity_1b,
    label="Sparsity (1B)",
    color=SPARSE,
    linewidth=lw_main,
    alpha=alpha_1b
)
ax.plot(
    epochs, sparsity_8b,
    label="Sparsity (8B)",
    color=SPARSE,
    linewidth=lw_main,
    alpha=alpha_8b
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Fraction active / Sparsity")

ax2 = ax.twinx()
ax2.plot(
    epochs, kl_1b,
    label="KL-to-prior (1B)",
    color=KL,
    linewidth=lw_aux,
    alpha=alpha_1b
)
ax2.plot(
    epochs, kl_8b,
    label="KL-to-prior (8B)",
    color=KL,
    linewidth=lw_aux,
    alpha=alpha_8b
)

ax2.set_ylabel(r"KL$(\mathrm{Bern}(q)\,\|\,\mathrm{Bern}(\pi))$")

lines = ax.get_lines() + ax2.get_lines()
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, fontsize=9, framealpha=0.95)

plt.title("Mask sparsification dynamics (target $\\pi \\approx 0.1$)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "cd_sparsify.png"), dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(7.5, 4.5))

ax.plot(
    epochs, sim_1b,
    label="Similarity loss (1B)",
    color=SIM,
    linewidth=lw_main,
    alpha=alpha_1b
)
ax.plot(
    epochs, sim_8b,
    label="Similarity loss (8B)",
    color=SIM,
    linewidth=lw_main,
    alpha=alpha_8b
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Similarity loss")

ax2 = ax.twinx()
ax2.plot(
    epochs, mask_cossim_1b,
    label="Mask cos sim (1B)",
    color=MASKCOS,
    linewidth=lw_aux,
    alpha=alpha_1b
)
ax2.plot(
    epochs, mask_cossim_8b,
    label="Mask cos sim (8B)",
    color=MASKCOS,
    linewidth=lw_aux,
    alpha=alpha_8b
)

ax2.set_ylabel("Mean pairwise mask cosine similarity")

lines = ax.get_lines() + ax2.get_lines()
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, fontsize=9, framealpha=0.95)

plt.title("Representativeness vs orthogonality during circuit discovery")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "cd_rep_ortho.png"), dpi=300)
plt.close()
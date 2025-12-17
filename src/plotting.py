import matplotlib.pyplot as plt
import json
import os
import argparse

def plot_k_vs_loss(model_name: str):
    base_dir = os.path.join("..", "results", "neuron-clustering", model_name)
    json_path = os.path.join(base_dir, "k_gs_testing.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find {json_path}. Run neuron_clustering.py first to generate it.")

    with open(json_path, "r") as f:
        k_gs_testing = json.load(f)

    # Keys are strings when loaded from JSON
    ks = sorted(int(k) for k in k_gs_testing.keys())
    losses = [k_gs_testing[str(k)] for k in ks]

    plt.figure(figsize=(6, 4))
    plt.plot(ks, losses, marker="o")
    plt.xlabel("k (number of clusters)")
    plt.ylabel("Mean cosine distance to centroids (loss)")
    plt.title(f"k-means loss vs k for {model_name}")
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(base_dir, "k_vs_loss.png")
    os.makedirs(base_dir, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Model name used when running neuron_clustering.py")
    args = parser.parse_args()
    plot_k_vs_loss(args.model_name)


if __name__ == "__main__":
    main()

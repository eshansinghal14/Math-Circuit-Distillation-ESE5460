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

    os.makedirs(base_dir, exist_ok=True)

    for subclass_str, k_dict in k_gs_testing.items():
        ks = sorted(int(k) for k in k_dict.keys())
        losses = [k_dict[str(k)] for k in ks]

        plt.figure(figsize=(6, 4))
        plt.plot(ks, losses, marker="o")
        plt.xlabel("k (number of clusters)")
        plt.ylabel("Mean cosine distance to centroids (loss)")
        plt.title(f"k-means loss vs k for {model_name.split('/')[1]}, subclass {subclass_str}")
        plt.grid(True, alpha=0.3)

        out_path = os.path.join(base_dir, f"k_vs_loss_subclass_{subclass_str}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Model name used when running neuron_clustering.py")
    args = parser.parse_args()
    plot_k_vs_loss(args.model_name)


if __name__ == "__main__":
    main()

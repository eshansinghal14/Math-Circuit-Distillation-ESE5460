"""
Serve the graph visualization frontend in Colab.

Usage (one cell):
    %run /content/Math-Circuit-Distillation-ESE5460/src/graph_loss/frontend_assets/__main__.py
"""
import json
import os
import subprocess

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))

graph_data_dir = os.path.join(ASSETS_DIR, "graph_data")
metadata_path = os.path.join(ASSETS_DIR, "data/graph-metadata.json")

os.makedirs(graph_data_dir, exist_ok=True)
os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

if os.path.exists(metadata_path):
    with open(metadata_path) as f:
        metadata = json.load(f)
else:
    metadata = {"graphs": []}

existing_slugs = {g.get("slug") for g in metadata.get("graphs", [])}

for filename in sorted(os.listdir(graph_data_dir)):
    if filename.endswith(".json"):
        slug = filename[:-5]
        if slug not in existing_slugs:
            print(f"Registering: {slug}")
            with open(os.path.join(graph_data_dir, filename)) as f:
                try:
                    data = json.load(f)
                    meta = data.get("metadata", {})
                    entry = {
                        "slug": slug,
                        "scan": meta.get("scan", "graph-loss-supergraph"),
                        "prompt": meta.get("prompt", slug),
                        "title_prefix": "",
                        "schema_version": meta.get("schema_version", 1),
                        "model": meta.get("model", "Unknown"),
                        "n_supernodes": meta.get("n_supernodes", 0),
                        "n_links": meta.get("n_links", 0),
                    }
                except Exception:
                    entry = {
                        "slug": slug,
                        "scan": "graph-loss-supergraph",
                        "prompt": slug,
                        "title_prefix": "",
                        "schema_version": 1,
                        "model": "Unknown",
                        "n_supernodes": 0,
                        "n_links": 0,
                    }
            metadata["graphs"].append(entry)
            existing_slugs.add(slug)

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Serving {len(metadata['graphs'])} graph(s)")

subprocess.Popen(["python3", "-m", "http.server", "8082"], cwd=ASSETS_DIR)

from google.colab.output import serve_kernel_port_as_iframe
serve_kernel_port_as_iframe(8082, height=800)

"""Export graph-loss supergraphs for the bundled attribution-graph frontend."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import torch

from graph_loss.graph import Graph, SuperGraph


def default_frontend_output_dir() -> Path:
    return Path(__file__).resolve().parent / "frontend_assets"


def slugify(value: str, *, fallback: str = "supergraph") -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower()).strip("-")
    return slug[:80] or fallback


def default_slug(graph: Graph, *, model_name: str | None = None) -> str:
    digest_src = f"{model_name or graph.cfg.model_name}:{graph.input_string}"
    digest = hashlib.sha1(digest_src.encode("utf-8")).hexdigest()[:10]
    prefix = slugify(model_name or graph.cfg.model_name or "supergraph")
    return f"{prefix}-{digest}"


def export_supergraph_frontend(
    graph: Graph,
    supergraph: SuperGraph,
    *,
    output_dir: str | os.PathLike[str] | None = None,
    slug: str | None = None,
    model_name: str | None = None,
    tokenizer: Any | None = None,
    title_prefix: str = "Generated",
) -> Path:
    """Write static frontend JSON for a built supergraph.

    Returns the path to the generated graph-data JSON file.
    """

    output_path = Path(output_dir) if output_dir is not None else default_frontend_output_dir()
    graph_slug = slugify(slug or default_slug(graph, model_name=model_name))
    data_dir = output_path / "data"
    graph_data_dir = output_path / "graph_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    graph_data_dir.mkdir(parents=True, exist_ok=True)

    graph_data = _build_graph_data(
        graph,
        supergraph,
        slug=graph_slug,
        model_name=model_name,
        tokenizer=tokenizer,
    )
    graph_data_path = graph_data_dir / f"{graph_slug}.json"
    _write_json(graph_data_path, graph_data)

    metadata_path = data_dir / "graph-metadata.json"
    metadata = _read_metadata(metadata_path)
    entry = _build_metadata_entry(graph_data["metadata"], title_prefix=title_prefix)
    graphs = [item for item in metadata.get("graphs", []) if item.get("slug") != graph_slug]
    graphs.append(entry)
    _write_json(metadata_path, {"graphs": graphs})

    return graph_data_path


def _build_graph_data(
    graph: Graph,
    supergraph: SuperGraph,
    *,
    slug: str,
    model_name: str | None,
    tokenizer: Any | None,
) -> dict[str, Any]:
    prompt_tokens = _prompt_tokens(graph, tokenizer)
    nodes = _supergraph_nodes(graph, supergraph)
    links = _supergraph_links(supergraph, nodes)
    metadata = {
        "schema_version": 1,
        "format": "graph_loss_supergraph",
        "slug": slug,
        "scan": "graph-loss-supergraph",
        "model": model_name or graph.cfg.model_name,
        "prompt": graph.input_string,
        "prompt_tokens": prompt_tokens,
        "n_neurons": graph.n_neurons,
        "n_supernodes": len(supergraph.supernodes),
        "n_links": len(links),
        "attribution_mode": graph.attribution_mode,
    }
    return {
        "metadata": metadata,
        "nodes": nodes,
        "links": links,
        "qParams": {},
    }


def _build_metadata_entry(metadata: dict[str, Any], *, title_prefix: str) -> dict[str, Any]:
    return {
        "slug": metadata["slug"],
        "scan": metadata["scan"],
        "prompt": metadata["prompt"],
        "title_prefix": title_prefix,
        "schema_version": metadata["schema_version"],
        "model": metadata["model"],
        "n_supernodes": metadata["n_supernodes"],
        "n_links": metadata["n_links"],
    }


def _prompt_tokens(graph: Graph, tokenizer: Any | None) -> list[str]:
    tokens = graph.input_tokens.detach().cpu().tolist()
    if tokenizer is None or not hasattr(tokenizer, "decode"):
        return [str(int(token_id)) for token_id in tokens]

    decoded = []
    for token_id in tokens:
        try:
            decoded.append(str(tokenizer.decode([int(token_id)])))
        except TypeError:
            decoded.append(str(tokenizer.decode(int(token_id))))
        except Exception:
            decoded.append(str(int(token_id)))
    return decoded


def _supergraph_nodes(graph: Graph, supergraph: SuperGraph) -> list[dict[str, Any]]:
    locations = graph.neuron_locations.detach().cpu()
    activations = graph.neuron_activations.detach().float().cpu()
    delta_norms = (
        supergraph.supernode_prob_deltas.detach().float().cpu().norm(dim=1)
        if supergraph.supernode_prob_deltas is not None
        else None
    )

    nodes = []
    for supernode_idx, members in enumerate(supergraph.supernodes):
        member_ids = [int(member) for member in members]
        member_locations = locations[member_ids] if member_ids else torch.empty((0, 3))
        layer = _mean_int(member_locations[:, 0]) if member_ids else 0
        ctx_idx = _mean_int(member_locations[:, 1]) if member_ids else 0
        neuron_ids = (
            member_locations[:, 2].to(dtype=torch.long).tolist() if member_ids else []
        )
        node_id = f"sg_{supernode_idx}"
        label = _supernode_label(supergraph, supernode_idx, member_ids)
        activation_values = activations[member_ids] if member_ids else torch.empty(0)
        nodes.append(
            {
                "node_id": node_id,
                "feature": supernode_idx,
                "feature_type": "supernode",
                "layer": layer,
                "ctx_idx": ctx_idx,
                "probe_location_idx": 0,
                "influence": float(delta_norms[supernode_idx].item())
                if delta_norms is not None and supernode_idx < int(delta_norms.numel())
                else 0.0,
                "activation": float(activation_values.mean().item())
                if activation_values.numel()
                else 0.0,
                "clerp": label,
                "member_node_ids": member_ids,
                "member_neuron_ids": [int(neuron_id) for neuron_id in neuron_ids],
                "member_count": len(member_ids),
                "is_supergraph_node": True,
            }
        )
    return nodes


def _supergraph_links(
    supergraph: SuperGraph,
    nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    adjacency = supergraph.supernode_adjacency_matrix.detach().float().cpu()
    node_ids = [node["node_id"] for node in nodes]
    links = []
    for target_idx, source_idx in adjacency.nonzero(as_tuple=False).tolist():
        if target_idx == source_idx:
            continue
        if target_idx >= len(node_ids) or source_idx >= len(node_ids):
            continue
        links.append(
            {
                "source": node_ids[source_idx],
                "target": node_ids[target_idx],
                "weight": float(adjacency[target_idx, source_idx].item()),
            }
        )
    links.sort(key=lambda item: abs(item["weight"]))
    return links


def _supernode_label(
    supergraph: SuperGraph,
    supernode_idx: int,
    member_ids: list[int],
) -> str:
    if supergraph.supernode_labels and supernode_idx < len(supergraph.supernode_labels):
        label_parts = [str(part) for part in supergraph.supernode_labels[supernode_idx] if part]
        if label_parts:
            return ", ".join(label_parts)

    if supergraph.node_labels:
        labels = []
        for member_id in member_ids:
            labels.extend(str(label) for label in supergraph.node_labels.get(member_id, []) if label)
        if labels:
            unique_labels = list(dict.fromkeys(labels))
            return ", ".join(unique_labels[:5])

    return f"Supernode {supernode_idx} ({len(member_ids)} neurons)"


def _mean_int(values: torch.Tensor) -> int:
    if values.numel() == 0:
        return 0
    return int(round(float(values.float().mean().item())))


def _read_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"graphs": []}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")

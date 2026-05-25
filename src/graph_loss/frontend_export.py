"""Export graph-loss supergraphs for the bundled attribution-graph frontend."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
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

    # Copy supernode PDF heatmaps into the frontend assets so the web server can serve them.
    heatmap_pdf_url_by_supernode: dict[int, str] = {}
    if supergraph.supernode_heatmap_pdf_paths:
        heatmap_dir = output_path / "heatmaps" / graph_slug
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        for supernode_idx, src_pdf in enumerate(supergraph.supernode_heatmap_pdf_paths):
            if src_pdf and os.path.isfile(src_pdf):
                dst_pdf = heatmap_dir / f"supernode_{supernode_idx}.pdf"
                shutil.copy2(src_pdf, dst_pdf)
                heatmap_pdf_url_by_supernode[supernode_idx] = f"heatmaps/{graph_slug}/supernode_{supernode_idx}.pdf"

    graph_data = _build_graph_data(
        graph,
        supergraph,
        slug=graph_slug,
        model_name=model_name,
        tokenizer=tokenizer,
        heatmap_pdf_url_by_supernode=heatmap_pdf_url_by_supernode,
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
    heatmap_pdf_url_by_supernode: dict[int, str] | None = None,
) -> dict[str, Any]:
    prompt_tokens = _prompt_tokens(graph, tokenizer)
    nodes = _supergraph_nodes(
        graph,
        supergraph,
        heatmap_pdf_url_by_supernode=heatmap_pdf_url_by_supernode,
        tokenizer=tokenizer,
    )
    links = _supergraph_links(graph, supergraph, nodes)
    metadata = {
        "schema_version": 1,
        "format": "graph_loss_supergraph",
        "slug": slug,
        "scan": "graph-loss-supergraph",
        "model": model_name or graph.cfg.model_name,
        "prompt": graph.input_string,
        "prompt_tokens": prompt_tokens,
        "n_neurons": graph.n_neurons,
        "n_tokens": graph.n_tokens,
        "n_logits": graph.n_logits,
        "n_supernodes": len(supergraph.supernodes),
        "n_links": len(links),
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
        "prompt": metadata["slug"],
        "title_prefix": "",
        "schema_version": metadata["schema_version"],
        "model": metadata["model"],
        "n_supernodes": metadata["n_supernodes"],
        "n_links": metadata["n_links"],
    }


def _prompt_tokens(graph: Graph, tokenizer: Any | None) -> list[str]:
    # prompt_tokens is a display-only field (x-axis token labels) and must always
    # be populated regardless of whether token embedding nodes are in the graph.
    # When include_token_nodes=False, graph.input_tokens is empty, so we
    # re-derive the IDs by re-encoding graph.input_string with the tokenizer.
    if graph.n_tokens > 0:
        token_id_list = graph.input_tokens.detach().cpu().tolist()
    elif tokenizer is not None and hasattr(tokenizer, "encode") and graph.input_string:
        try:
            token_id_list = tokenizer.encode(graph.input_string, add_special_tokens=False)
        except Exception:
            return []
    else:
        return []

    if tokenizer is None or not hasattr(tokenizer, "decode"):
        return [str(int(tid)) for tid in token_id_list]

    decoded = []
    for token_id in token_id_list:
        try:
            decoded.append(str(tokenizer.decode([int(token_id)])))
        except TypeError:
            decoded.append(str(tokenizer.decode(int(token_id))))
        except Exception:
            decoded.append(str(int(token_id)))
    return decoded


def _supergraph_nodes(
    graph: Graph,
    supergraph: SuperGraph,
    *,
    heatmap_pdf_url_by_supernode: dict[int, str] | None = None,
    tokenizer: Any | None = None,
) -> list[dict[str, Any]]:
    locations = graph.neuron_locations.detach().cpu()
    activations = graph.neuron_activations.detach().float().cpu()
    delta_norms = None

    max_layer = int(locations[:, 0].max().item()) if graph.n_neurons > 0 else 0

    nodes: list[dict[str, Any]] = []

    # ── 1. Supernode (neuron-cluster) nodes ──────────────────────────────────
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
        node_dict: dict[str, Any] = {
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
        if heatmap_pdf_url_by_supernode and supernode_idx in heatmap_pdf_url_by_supernode:
            node_dict["heatmap_pdf_url"] = heatmap_pdf_url_by_supernode[supernode_idx]
        nodes.append(node_dict)

    # ── 2. Token embedding nodes (present when include_token_nodes=True) ─────
    if graph.n_tokens > 0:
        token_ids = graph.input_tokens.detach().cpu().tolist()
        for pos, token_id in enumerate(token_ids):
            token_str = _decode_token(token_id, tokenizer)
            nodes.append({
                "node_id": f"tok_{pos}",
                "feature": pos,
                "feature_type": "token",
                "layer": -1,          # embedding layer — before all MLP layers
                "ctx_idx": pos,
                "probe_location_idx": 0,
                "influence": 0.0,
                "activation": 1.0,    # token embeddings are always "active"
                "clerp": token_str,
                "token_id": int(token_id),
                "is_token_node": True,
            })

    # ── 3. Logit target nodes (present when include_logit_nodes=True) ────────
    if graph.n_logits > 0:
        logit_probs = graph.logit_probabilities.detach().float().cpu().tolist()
        for logit_idx, target in enumerate(graph.logit_targets):
            prob = float(logit_probs[logit_idx]) if logit_idx < len(logit_probs) else 0.0
            nodes.append({
                "node_id": f"logit_{logit_idx}",
                "feature": logit_idx,
                "feature_type": "logit",
                "layer": max_layer + 1,       # after all MLP layers
                "ctx_idx": graph.n_pos - 1,   # last sequence position
                "probe_location_idx": 0,
                "influence": 0.0,
                "activation": prob,
                "clerp": target.token_str,
                "vocab_idx": target.vocab_idx,
                "probability": prob,
                "is_logit_node": True,
            })

    return nodes


def _supergraph_links(
    graph: Graph,
    supergraph: SuperGraph,
    nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    supernode_adj = supergraph.supernode_adjacency_matrix.detach().float().cpu()
    n_supernodes = len(supergraph.supernodes)
    links = []

    # ── 1. Supernode-to-supernode edges (from the supergraph adjacency matrix) ─
    for target_idx, source_idx in supernode_adj.nonzero(as_tuple=False).tolist():
        if target_idx == source_idx:
            continue
        if target_idx >= n_supernodes or source_idx >= n_supernodes:
            continue
        links.append({
            "source": f"sg_{source_idx}",
            "target": f"sg_{target_idx}",
            "weight": float(supernode_adj[target_idx, source_idx].item()),
        })

    # ── 2. Token→supernode cross-edges ───────────────────────────────────────
    # Adjacency layout: [neurons | tokens | logits]; rows=targets, cols=sources.
    # adjacency[neuron_row, n_neurons + token_pos] = token influence on that neuron.
    n_neurons = graph.n_neurons
    n_tokens = graph.n_tokens
    n_logits = graph.n_logits

    if n_tokens > 0:
        adj = graph.adjacency_matrix.detach().float().cpu()
        adj_cols = adj.shape[1]
        for pos in range(n_tokens):
            token_col = n_neurons + pos
            if token_col >= adj_cols:
                continue
            for sg_idx, members in enumerate(supergraph.supernodes):
                if not members:
                    continue
                member_t = torch.tensor(members, dtype=torch.long)
                weight = float(adj[member_t, token_col].sum().item())
                if weight != 0.0:
                    links.append({
                        "source": f"tok_{pos}",
                        "target": f"sg_{sg_idx}",
                        "weight": weight,
                    })

    # ── 3. Supernode→logit cross-edges ───────────────────────────────────────
    # adjacency[n_neurons + n_tokens + logit_idx, neuron_col] = neuron's contribution to that logit.
    if n_logits > 0:
        adj = graph.adjacency_matrix.detach().float().cpu()
        adj_rows = adj.shape[0]
        for logit_idx in range(n_logits):
            logit_row = n_neurons + n_tokens + logit_idx
            if logit_row >= adj_rows:
                continue
            for sg_idx, members in enumerate(supergraph.supernodes):
                if not members:
                    continue
                member_t = torch.tensor(members, dtype=torch.long)
                weight = float(adj[logit_row, member_t].sum().item())
                if weight != 0.0:
                    links.append({
                        "source": f"sg_{sg_idx}",
                        "target": f"logit_{logit_idx}",
                        "weight": weight,
                    })

    links.sort(key=lambda item: abs(item["weight"]))
    return links


def _decode_token(token_id: int, tokenizer: Any | None) -> str:
    """Decode a single token ID to a display string."""
    if tokenizer is None or not hasattr(tokenizer, "decode"):
        return str(int(token_id))
    try:
        return str(tokenizer.decode([int(token_id)]))
    except TypeError:
        return str(tokenizer.decode(int(token_id)))
    except Exception:
        return str(int(token_id))


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

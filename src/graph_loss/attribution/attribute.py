"""Neuron-level attribution graph builder for HF LLaMA models.

Mirrors the TL ``attribute`` / ``setup_attribution`` pipeline but uses
``HFLlamaGraphAdapter`` and PyTorch native hooks instead of TransformerLens.

Typical flow:

    adapter = HFLlamaGraphAdapter(hf_model, tokenizer, device)
    graph = attribute(adapter, prompt, top_k_logits=0.95)

``build_graph`` on the adapter is a thin wrapper around ``attribute``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from graph_loss.attribution.context import HFAttributionContext
from graph_loss.attribution.targets import AttributionTargets, TargetSpec
from graph_loss.graph import Graph

if TYPE_CHECKING:
    from graph_loss.hf_adapter import HFLlamaGraphAdapter


def setup_attribution(
    adapter: "HFLlamaGraphAdapter",
    input_ids: torch.Tensor,
    prop_neurons_per_layer: float,
    dtype: torch.dtype | None,
) -> HFAttributionContext:
    """Run the initial single forward pass and return an ``HFAttributionContext``.

    Captures MLP inputs and token embeddings via hooks, then selects the
    top-``prop_neurons_per_layer`` neurons per layer by source-vector norm.
    Equivalent to ``TransformerLensReplacementModel.setup_attribution``.
    """
    input_batch = input_ids.unsqueeze(0)
    mlp_inputs: dict[int, torch.Tensor] = {}
    embed_out: torch.Tensor | None = None
    handles = []

    def embed_hook(_module, _inputs, output):
        nonlocal embed_out
        embed_out = output
        return output

    handles.append(adapter.embed_tokens.register_forward_hook(embed_hook))

    for layer_idx, layer in enumerate(adapter.layers):
        def _pre(m, inputs, *, idx=layer_idx):
            mlp_inputs[idx] = inputs[0]

        handles.append(layer.mlp.register_forward_pre_hook(_pre))

    try:
        with adapter.autocast_context(dtype):
            out = adapter.model(
                input_ids=input_batch,
                attention_mask=torch.ones_like(input_batch, device=adapter.device),
                output_hidden_states=True,
                use_cache=False,
            )
    finally:
        for h in handles:
            h.remove()

    if embed_out is None:
        raise RuntimeError("Embedding hook did not fire.")
    if len(mlp_inputs) != adapter.n_layers:
        raise RuntimeError(
            f"Expected {adapter.n_layers} MLP input captures, got {len(mlp_inputs)}."
        )

    n_pos = int(input_ids.numel())
    positions = torch.arange(n_pos, device=adapter.device, dtype=torch.long)
    neuron_ids = torch.arange(adapter.d_mlp, device=adapter.device, dtype=torch.long)

    neuron_locations = []
    neuron_activations = []
    target_encoders = []
    source_vectors = []
    source_layer_by_node: list[int] = []

    for layer_idx in range(adapter.n_layers):
        layer_input = mlp_inputs[layer_idx].squeeze(0)
        layer_acts, layer_target_encoders, layer_source_vectors = (
            adapter._compute_layer_neuron_data(layer_idx, layer_input)
        )
        flat_norms = layer_source_vectors.norm(dim=-1).reshape(-1)
        k = min(max(1, int(flat_norms.numel() * prop_neurons_per_layer)), flat_norms.numel())
        keep = torch.topk(flat_norms, k, dim=0).indices

        layer_locations = torch.stack(
            [
                torch.full((n_pos * adapter.d_mlp,), layer_idx, device=adapter.device, dtype=torch.long),
                positions.repeat_interleave(adapter.d_mlp),
                neuron_ids.repeat(n_pos),
            ],
            dim=1,
        )
        neuron_locations.append(layer_locations[keep])
        neuron_activations.append(layer_acts.reshape(-1)[keep])
        target_encoders.append(layer_target_encoders.reshape(-1, adapter.d_model)[keep])
        source_vectors.append(layer_source_vectors.reshape(-1, adapter.d_model)[keep])
        source_layer_by_node.extend([layer_idx] * int(keep.numel()))

    return HFAttributionContext(
        adapter=adapter,
        input_ids=input_ids,
        neuron_locations=torch.cat(neuron_locations, dim=0),
        neuron_activations=torch.cat(neuron_activations, dim=0),
        target_encoders=torch.cat(target_encoders, dim=0),
        source_vectors=torch.cat(source_vectors, dim=0),
        source_layer_by_node=torch.tensor(source_layer_by_node, device=adapter.device, dtype=torch.long),
        embed_out=embed_out,
        logits=out.logits,
        dtype=dtype,
    )


def _attribute_from_context(
    ctx: HFAttributionContext,
    *,
    attribution_targets: Sequence[str] | Sequence[TargetSpec] | torch.Tensor | None = None,
    top_k_logits: float | None = 0.95,
    temperature: float = 2.0,
    batch_size: int = 512,
    create_graph: bool = False,
    detach_result: bool | None = None,
    skip_logit_attribution: bool = False,
    verbose: bool = False,
    include_token_nodes: bool = False,
    include_logit_nodes: bool = False,
) -> Graph:
    """Build a Graph from a pre-built HFAttributionContext (edge attribution phase only).

    Callers that want to run ANOVA labeling before edge attribution should call
    setup_attribution() → select_anova_supernodes() → ctx.filter(mask) → this function.

    Args:
        include_token_nodes: If True, include token embedding nodes in the adjacency
            matrix (rows and columns).  Default False — neuron-only graph.
        include_logit_nodes: If True, include logit target nodes in the adjacency
            matrix (rows and columns).  Default False — neuron-only graph.
    """
    from graph_loss.hf_adapter import _HFGraphConfig, detach_graph

    adapter = ctx.adapter

    targets = AttributionTargets(
        attribution_targets=attribution_targets,
        logits=ctx.logits[0, -1],
        unembed_proj=adapter.W_U.to(dtype=ctx.dtype) if ctx.dtype is not None else adapter.W_U,
        tokenizer=adapter.tokenizer,
        top_k_logits=top_k_logits,
        temperature=temperature,
    )

    n_neurons = ctx.n_neurons
    n_tokens = ctx.n_tokens
    n_logits = len(targets)
    total_nodes = n_neurons + n_tokens + n_logits
    source_count = ctx.source_count

    cfg = _HFGraphConfig(adapter)

    # Gradient-based edge attribution.
    neuron_row_chunks = []
    for start in range(0, n_neurons, max(1, batch_size)):
        end = min(start + max(1, batch_size), n_neurons)
        if verbose:
            print(f"    [graph] neuron rows {start}:{end} / {n_neurons}")
        neuron_row_chunks.append(ctx.compute_neuron_batch(start, end, create_graph=create_graph))
    neuron_rows = (
        torch.cat(neuron_row_chunks, dim=0)
        if neuron_row_chunks
        else torch.zeros(0, source_count, dtype=ctx.source_vectors.dtype, device=adapter.device)
    )

    # Skip logit attribution entirely when logit nodes won't be included anyway.
    effective_skip_logit = skip_logit_attribution or not include_logit_nodes
    if not effective_skip_logit:
        logit_row_chunks = []
        for start in range(0, n_logits, max(1, batch_size)):
            end = min(start + max(1, batch_size), n_logits)
            if verbose:
                print(f"    [graph] logit rows {start}:{end} / {n_logits}")
            logit_row_chunks.append(
                ctx.compute_logit_batch(start, end, targets.logit_vectors, create_graph=create_graph)
            )
        logit_rows_partial = (
            torch.cat(logit_row_chunks, dim=0)
            if logit_row_chunks
            else torch.zeros(0, source_count, dtype=ctx.source_vectors.dtype, device=adapter.device)
        )
    else:
        logit_rows_partial = torch.zeros(
            n_logits, source_count, dtype=ctx.source_vectors.dtype, device=adapter.device
        )

    # Pad source-count columns out to total_nodes (append zero logit-target cols).
    n_logit_cols = total_nodes - source_count
    if n_logit_cols > 0:
        neuron_rows = torch.cat(
            [neuron_rows, torch.zeros(n_neurons, n_logit_cols, dtype=neuron_rows.dtype, device=adapter.device)],
            dim=1,
        )
        logit_rows = torch.cat(
            [logit_rows_partial, torch.zeros(n_logits, n_logit_cols, dtype=logit_rows_partial.dtype, device=adapter.device)],
            dim=1,
        )
    else:
        logit_rows = logit_rows_partial

    token_rows = torch.zeros(n_tokens, total_nodes, dtype=neuron_rows.dtype, device=adapter.device)
    adjacency = torch.cat([neuron_rows, token_rows, logit_rows], dim=0)

    # Strip token and/or logit node rows/columns when not requested.
    # Node layout is [neurons | tokens | logits]; keep indices are built in that order.
    if not include_token_nodes or not include_logit_nodes:
        keep: list[int] = list(range(n_neurons))
        if include_token_nodes:
            keep.extend(range(n_neurons, n_neurons + n_tokens))
        if include_logit_nodes:
            keep.extend(range(n_neurons + n_tokens, total_nodes))
        keep_t = torch.tensor(keep, dtype=torch.long, device=adapter.device)
        adjacency = adjacency[keep_t][:, keep_t]

    # Update Graph fields to match the actual adjacency layout.
    # n_tokens / n_logits are inferred from input_tokens / logit_targets lengths,
    # so zero them out when those nodes were excluded.
    graph_input_tokens = (
        ctx.input_ids
        if include_token_nodes
        else torch.zeros(0, dtype=torch.long, device=adapter.device)
    )
    graph_logit_targets = targets.logit_targets if include_logit_nodes else []
    graph_logit_probs = (
        targets.logit_probabilities
        if include_logit_nodes
        else torch.zeros(0, dtype=targets.logit_probabilities.dtype, device=adapter.device)
    )

    graph = Graph(
        input_string=adapter.tokenizer.decode(ctx.input_ids.detach().cpu().tolist()),
        input_tokens=graph_input_tokens,
        neuron_locations=ctx.neuron_locations,
        adjacency_matrix=adjacency,
        cfg=cfg,
        neuron_activations=ctx.neuron_activations,
        logit_targets=graph_logit_targets,
        logit_probabilities=graph_logit_probs,
        vocab_size=targets.vocab_size,
        neuron_write_vectors=ctx.source_vectors.detach().clone(),
    )
    if detach_result is None:
        detach_result = not create_graph
    if detach_result:
        graph = detach_graph(graph)
    return graph


def attribute(
    adapter: "HFLlamaGraphAdapter",
    prompt: str | torch.Tensor | list[int],
    *,
    attribution_targets: Sequence[str] | Sequence[TargetSpec] | torch.Tensor | None = None,
    top_k_logits: float | None = 0.95,
    temperature: float = 2.0,
    prop_neurons_per_layer: float = 0.1,
    batch_size: int = 512,
    dtype: torch.dtype | None = None,
    verbose: bool = False,
    create_graph: bool = False,
    detach_result: bool | None = None,
    skip_logit_attribution: bool = False,
    include_token_nodes: bool = False,
    include_logit_nodes: bool = False,
) -> Graph:
    """Compute a neuron-level attribution graph for ``prompt``.

    Phase 0: setup_attribution — single forward pass, neuron selection.
    Phase 1-2: _attribute_from_context — gradient-based edge scoring.
    """
    if not (0.0 < prop_neurons_per_layer <= 1.0):
        raise ValueError("prop_neurons_per_layer must be in (0, 1]")

    input_ids = adapter.ensure_tokenized(prompt)
    ctx = setup_attribution(adapter, input_ids, prop_neurons_per_layer, dtype)
    return _attribute_from_context(
        ctx,
        attribution_targets=attribution_targets,
        top_k_logits=top_k_logits,
        temperature=temperature,
        batch_size=batch_size,
        create_graph=create_graph,
        detach_result=detach_result,
        skip_logit_attribution=skip_logit_attribution,
        verbose=verbose,
        include_token_nodes=include_token_nodes,
        include_logit_nodes=include_logit_nodes,
    )

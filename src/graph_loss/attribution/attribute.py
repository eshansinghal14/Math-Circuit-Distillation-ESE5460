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
from graph_loss.freeze import without_gradient_checkpointing
from graph_loss.attribution.targets import AttributionTargets, TargetSpec
from graph_loss.graph import Graph

if TYPE_CHECKING:
    from graph_loss.hf_adapter import HFLlamaGraphAdapter


def setup_attribution(
    adapter: "HFLlamaGraphAdapter",
    input_ids: torch.Tensor,
    prop_neurons_per_layer: float,
    dtype: torch.dtype | None,
    *,
    freeze_attention: bool = False,
    freeze_rms_norm: bool = False,
) -> HFAttributionContext:
    """Run the initial single forward pass and return an ``HFAttributionContext``.

    Captures MLP inputs and token embeddings via hooks, then selects the
    top-``prop_neurons_per_layer`` neurons per layer by source-vector norm.
    Equivalent to ``TransformerLensReplacementModel.setup_attribution``.

    ``freeze_attention`` / ``freeze_rms_norm`` stop-gradient the attention pattern
    and the RMSNorm denominator (see ``graph_loss.freeze``). They are stored on the
    returned context and applied *only* to the edge-attribution forwards; this
    forward deliberately runs unfrozen. Since the freeze is backward-only it cannot
    alter any value produced here, so the resulting graph is identical either way,
    while the autograd graph attached to ``mlp_inputs`` -- the student's only
    gradient path from the graph loss back to the model weights -- stays intact.
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
        # NOT wrapped in frozen_graph_edges -- see the docstring above and
        # graph_loss.freeze. The freeze is backward-only, so it cannot change any
        # value this pass produces (logits, embed_out, mlp_inputs, and hence the
        # source/target vectors and neuron pre-selection are bit-identical either
        # way). All it would change is the autograd graph hanging off mlp_inputs,
        # which on the student path is the sole route from the graph loss back to
        # the weights, since the edge gradients are constants when
        # create_graph=False. Freezing here adds nothing to the attribution and
        # only biases the training gradient: it zeroes the q/k path outright and
        # strips the radial correction from every RMSNorm.
        with without_gradient_checkpointing(adapter.model), adapter.autocast_context(dtype):
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

    # NOTE: do NOT wrap this loop in torch.no_grad() and do NOT CPU-offload the
    # kept tensors. setup_attribution runs in two contexts: the TEACHER path is
    # already under the caller's torch.no_grad(), while the STUDENT path must keep
    # gradients flowing from source_vectors/target_encoders back into the model
    # weights. The selective method respects the ambient autograd state and only
    # detaches the (non-differentiable) top-k scoring, so the kept [k, d_model]
    # tensors stay differentiable on the student path. A prior version that added
    # a no_grad wrapper + CPU offload here broke student gradients and was reverted.
    for layer_idx in range(adapter.n_layers):
        layer_input = mlp_inputs[layer_idx].squeeze(0)
        keep, layer_acts_kept, layer_te_kept, layer_sv_kept = (
            adapter._compute_layer_neuron_data_selective(
                layer_idx, layer_input, prop_neurons_per_layer
            )
        )

        layer_locations = torch.stack(
            [
                torch.full((n_pos * adapter.d_mlp,), layer_idx, device=adapter.device, dtype=torch.long),
                positions.repeat_interleave(adapter.d_mlp),
                neuron_ids.repeat(n_pos),
            ],
            dim=1,
        )
        neuron_locations.append(layer_locations[keep])
        neuron_activations.append(layer_acts_kept)
        target_encoders.append(layer_te_kept)
        source_vectors.append(layer_sv_kept)
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
        freeze_attention=freeze_attention,
        freeze_rms_norm=freeze_rms_norm,
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
    target_position: int = -1,
) -> Graph:
    """Build a Graph from a pre-built HFAttributionContext (edge attribution phase only).

    Always includes token embedding and logit target nodes in the adjacency matrix.
    Token and logit nodes are needed for the frac_external calculation in
    build_super_graph; they are excluded from the final supergraph and frontend
    visualization downstream.

    Callers that want to run ANOVA labeling before edge attribution should call
    setup_attribution() → select_anova_supernodes() → ctx.filter(mask) → this function.

    target_position: which sequence position to attribute from (default -1 = last).
    """
    from graph_loss.hf_adapter import _HFGraphConfig, detach_graph

    adapter = ctx.adapter

    targets = AttributionTargets(
        attribution_targets=attribution_targets,
        logits=ctx.logits[0, target_position],
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

    # Logit attribution: compute rows for logit target nodes.
    # Token and logit nodes are always included in the adjacency matrix so that
    # build_super_graph's frac_external calculation accounts for all influence paths.
    if not skip_logit_attribution:
        logit_row_chunks = []
        for start in range(0, n_logits, max(1, batch_size)):
            end = min(start + max(1, batch_size), n_logits)
            if verbose:
                print(f"    [graph] logit rows {start}:{end} / {n_logits}")
            logit_row_chunks.append(
                ctx.compute_logit_batch(start, end, targets.logit_vectors, create_graph=create_graph, target_position=target_position)
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
    # Node layout: [neurons | tokens | logits].
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

    graph = Graph(
        input_string=adapter.tokenizer.decode(ctx.input_ids.detach().cpu().tolist()),
        input_tokens=ctx.input_ids,
        neuron_locations=ctx.neuron_locations,
        adjacency_matrix=adjacency,
        cfg=cfg,
        neuron_activations=ctx.neuron_activations,
        logit_targets=targets.logit_targets,
        logit_probabilities=targets.logit_probabilities,
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
    target_position: int = -1,
    freeze_attention: bool = False,
    freeze_rms_norm: bool = False,
) -> Graph:
    """Compute a neuron-level attribution graph for ``prompt``.

    Always includes token embedding and logit target nodes in the adjacency matrix
    so that downstream frac_external calculations have full context.

    Phase 0: setup_attribution — single forward pass, neuron selection.
    Phase 1-2: _attribute_from_context — gradient-based edge scoring.
    """
    if not (0.0 < prop_neurons_per_layer <= 1.0):
        raise ValueError("prop_neurons_per_layer must be in (0, 1]")

    input_ids = adapter.ensure_tokenized(prompt)
    ctx = setup_attribution(
        adapter,
        input_ids,
        prop_neurons_per_layer,
        dtype,
        freeze_attention=freeze_attention,
        freeze_rms_norm=freeze_rms_norm,
    )
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
        target_position=target_position,
    )

"""Build dense-backed neuron-level attribution graphs for small prompts/models."""

import logging
import math
import time
from collections.abc import Sequence
from typing import Literal

import torch
from tqdm import tqdm

from graph_loss.attribution.targets import (
    AttributionTargets,
    TargetSpec,
    log_attribution_target_info,
)
from graph_loss.graph import Graph
from graph_loss.replacement_model import TransformerLensReplacementModel


def attribute(
    prompt: str | torch.Tensor | list[int],
    model: TransformerLensReplacementModel,
    *,
    attribution_targets: Sequence[str] | Sequence[TargetSpec] | torch.Tensor | None = None,
    top_k_logits: int | None = None,
    prop_neurons_per_layer: float = 0.1,
    batch_size: int = 512,
    max_feature_nodes: int | None = None,
    offload: Literal["cpu", "disk", None] = None,
    verbose: bool = False,
    update_interval: int = 4,
    fast: bool = False,
) -> Graph:
    """Compute a dense-backed neuron graph for a prompt.

    The dense adjacency layout is:
    ``[neurons, token_embeddings, logits]``.
    Token rows remain zero because tokens are source-only roots.

    When ``fast`` is True, skip Phase 3 backward attribution: fill only a linear
    logit readout block (demeaned unembed directions times residual write vectors)
    and use ``attribution_mode='fast'`` for pruning / supergraph influence proxies.
    """

    if max_feature_nodes is not None:
        raise ValueError("Neuron graphs always include all neurons; max_feature_nodes is unsupported.")
    if offload is not None:
        raise NotImplementedError("offload is not supported in the neuron graph builder.")
    if update_interval != 4:
        raise ValueError("update_interval is a legacy feature-selection parameter and is unsupported.")

    logger = logging.getLogger("attribution")
    logger.propagate = False
    handler = None
    if verbose and not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    try:
        return _run_attribution(
            model=model,
            prompt=prompt,
            attribution_targets=attribution_targets,
            top_k_logits=top_k_logits,
            prop_neurons_per_layer=prop_neurons_per_layer,
            batch_size=batch_size,
            logger=logger,
            verbose=verbose,
            fast=fast,
        )
    finally:
        if handler:
            logger.removeHandler(handler)


def _check_dense_graph_size(total_nodes: int, dtype: torch.dtype):
    element_size = torch.tensor([], dtype=dtype).element_size()
    total_bytes = total_nodes * total_nodes * element_size
    gib = total_bytes / (1024**3)
    if gib > 4:
        logging.getLogger("attribution").warning(
            "Dense adjacency is estimated to require about %.2f GiB; continuing anyway.",
            gib,
        )


def _run_target_batches(
    *,
    ctx,
    target_rows: torch.Tensor,
    target_layers: torch.Tensor,
    target_positions: torch.Tensor,
    inject_values: torch.Tensor,
    edge_row_chunks: list[torch.Tensor],
    edge_col_chunks: list[torch.Tensor],
    edge_value_chunks: list[torch.Tensor],
    batch_size: int,
    total_batches: int,
    processed_batches: int,
    progress_bar: tqdm | None = None,
) -> int:
    source_cols = ctx.source_node_count

    for start in range(0, len(target_rows), batch_size):
        end = min(start + batch_size, len(target_rows))
        retain_graph = processed_batches < total_batches - 1
        rows = ctx.compute_batch(
            layers=target_layers[start:end],
            positions=target_positions[start:end],
            inject_values=inject_values[start:end],
            retain_graph=retain_graph,
        )
        rows = rows.cpu()
        nonzero = rows.nonzero(as_tuple=False)
        if len(nonzero):
            batch_targets = target_rows[start:end].to(dtype=torch.long).cpu()
            edge_row_chunks.append(batch_targets[nonzero[:, 0]])
            edge_col_chunks.append(nonzero[:, 1].to(dtype=torch.long))
            edge_value_chunks.append(rows[nonzero[:, 0], nonzero[:, 1]])
        processed_batches += 1
        if progress_bar is not None:
            progress_bar.update(end - start)

    return processed_batches


def _run_attribution(
    *,
    model: TransformerLensReplacementModel,
    prompt,
    attribution_targets,
    top_k_logits,
    prop_neurons_per_layer,
    batch_size,
    logger,
    verbose,
    fast: bool = False,
):
    start_time = time.time()

    logger.info("Phase 0: Precomputing neuron activations and encoders")
    phase_start = time.time()
    input_ids = model.ensure_tokenized(prompt)
    ctx = model.setup_attribution(input_ids, prop_neurons_per_layer=prop_neurons_per_layer)

    logger.info(f"Precomputation completed in {time.time() - phase_start:.2f}s")
    logger.info(f"Enumerated {len(ctx.neuron_locations)} neuron nodes")
    if ctx.layer_capture_stats:
        avg_captured_fraction = sum(
            float(stats["captured_fraction"]) for stats in ctx.layer_capture_stats
        ) / len(ctx.layer_capture_stats)
        logger.info(
            "Average layer selection captured %.2f%% of residual write norm across the model",
            100.0 * avg_captured_fraction,
        )

    phase1_expand = 1 if fast else batch_size
    logger.info("Phase 1: Running hooked forward pass")
    phase_start = time.time()
    with ctx.install_hooks(model):
        residual = model.forward(
            input_ids.expand(phase1_expand, -1),
            stop_at_layer=model.cfg.n_layers,
        )
        ctx._resid_activations[-1] = model.ln_final(residual)
    logger.info(f"Forward pass completed in {time.time() - phase_start:.2f}s")

    logger.info("Phase 2: Building target specifications")
    phase_start = time.time()
    targets = AttributionTargets(
        attribution_targets=attribution_targets,
        logits=ctx.logits[0, -1],
        unembed_proj=model.unembed.W_U,
        tokenizer=model.tokenizer,
        top_k_logits=top_k_logits,
    )
    log_attribution_target_info(targets, attribution_targets, logger)

    n_neurons = len(ctx.neuron_locations)
    n_tokens = len(input_ids)
    n_logits = len(targets)
    total_nodes = n_neurons + n_tokens + n_logits
    logit_start = n_neurons + n_tokens

    logger.info(f"Neuron nodes: {n_neurons}, Token nodes: {n_tokens}, Logit nodes: {n_logits}")

    _check_dense_graph_size(total_nodes, ctx.source_vectors.dtype)
    logger.info(f"Target setup completed in {time.time() - phase_start:.2f}s")

    neuron_write_vectors: torch.Tensor | None = None

    if fast:
        logger.info("Phase 3 (fast): Skipping backward attribution; linear logit readout only")
        phase_start = time.time()
        neuron_write_vectors = ctx.source_vectors.clone()
        adjacency_matrix = torch.zeros(
            (total_nodes, total_nodes),
            dtype=ctx.source_vectors.dtype,
            device=ctx.source_vectors.device,
        )
        lv = targets.logit_vectors.to(
            device=ctx.source_vectors.device,
            dtype=ctx.source_vectors.dtype,
        )
        scores = lv @ ctx.source_vectors.T
        adjacency_matrix[logit_start : logit_start + n_logits, :n_neurons] = scores
        logger.info(f"Fast attribution completed in {time.time() - phase_start:.2f}s")
    else:
        edge_row_chunks: list[torch.Tensor] = []
        edge_col_chunks: list[torch.Tensor] = []
        edge_value_chunks: list[torch.Tensor] = []

        logger.info("Phase 3: Computing neuron and logit attribution rows")
        phase_start = time.time()
        neuron_batches = math.ceil(n_neurons / batch_size)
        logit_batches = math.ceil(n_logits / batch_size)
        total_batches = neuron_batches + logit_batches
        processed_batches = 0

        neuron_rows = torch.arange(n_neurons, dtype=torch.long)
        neuron_layers = ctx.neuron_locations[:, 0]
        neuron_positions = ctx.neuron_locations[:, 1]

        progress_bar = tqdm(
            total=n_neurons + n_logits,
            desc="Neuron graph attribution",
            disable=not verbose,
        )

        processed_batches = _run_target_batches(
            ctx=ctx,
            target_rows=neuron_rows,
            target_layers=neuron_layers,
            target_positions=neuron_positions,
            inject_values=ctx.target_encoders,
            edge_row_chunks=edge_row_chunks,
            edge_col_chunks=edge_col_chunks,
            edge_value_chunks=edge_value_chunks,
            batch_size=batch_size,
            total_batches=total_batches,
            processed_batches=processed_batches,
            progress_bar=progress_bar,
        )

        logit_rows = torch.arange(logit_start, logit_start + n_logits, dtype=torch.long)
        logit_layers = torch.full((n_logits,), model.cfg.n_layers, device=input_ids.device, dtype=torch.long)
        logit_positions = torch.full(
            (n_logits,),
            len(input_ids) - 1,
            device=input_ids.device,
            dtype=torch.long,
        )
        processed_batches = _run_target_batches(
            ctx=ctx,
            target_rows=logit_rows,
            target_layers=logit_layers,
            target_positions=logit_positions,
            inject_values=targets.logit_vectors,
            edge_row_chunks=edge_row_chunks,
            edge_col_chunks=edge_col_chunks,
            edge_value_chunks=edge_value_chunks,
            batch_size=batch_size,
            total_batches=total_batches,
            processed_batches=processed_batches,
            progress_bar=progress_bar,
        )

        progress_bar.close()
        logger.info(f"Attribution rows completed in {time.time() - phase_start:.2f}s")

        adjacency_matrix = torch.zeros(
            (total_nodes, total_nodes),
            dtype=ctx.source_vectors.dtype,
        )
        if edge_value_chunks:
            adjacency_indices = torch.stack(
                [torch.cat(edge_row_chunks), torch.cat(edge_col_chunks)],
                dim=0,
            )
            adjacency_values = torch.cat(edge_value_chunks)
            adjacency_matrix.index_put_(
                (adjacency_indices[0], adjacency_indices[1]),
                adjacency_values,
                accumulate=True,
            )

    graph = Graph(
        input_string=model.tokenizer.decode(input_ids.detach().cpu().tolist()),
        input_tokens=input_ids,
        neuron_locations=ctx.neuron_locations,
        adjacency_matrix=adjacency_matrix,
        cfg=model.cfg,
        neuron_activations=ctx.neuron_activations,
        logit_targets=targets.logit_targets,
        logit_probabilities=targets.logit_probabilities,
        vocab_size=targets.vocab_size,
        attribution_mode="fast" if fast else "full",
        neuron_write_vectors=neuron_write_vectors,
    )

    # Free memory immediately to prevent PyTorch graph accumulation
    ctx._resid_activations.clear()
    del ctx
    residual = None

    total_time = time.time() - start_time
    logger.info(f"Attribution completed in {total_time:.2f}s")
    return graph
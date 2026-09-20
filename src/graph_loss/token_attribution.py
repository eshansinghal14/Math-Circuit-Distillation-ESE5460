"""Which input tokens the prediction draws on, by gradient x input.

The attribution-graph route to the same quantity builds a neuron-level graph and
then collapses it to a vector over token positions, which throws away everything
the graph was for and costs ~1.8 s a prompt. This computes the vector directly:

    attr[p] = | (d target / d e_p) . e_p |        # then normalised over p

where ``e_p`` is token p's input embedding and ``target`` is a scalar read off the
logits at the last prompt position. One forward and one backward, tens of
milliseconds, and no pre-selection, supernodes, ANOVA labels, operand grids or
MLP-input cache anywhere in the path. That also means it ports unchanged to tasks
that have no operands.

It is *more* complete than the graph route, not less. The graph version sums the
direct residual path plus one MLP hop, so it silently drops multi-hop neuron
routes and every path through attention, and it can only route through neurons
that survived pre-selection. A single backward sums every path at once, to first
order, over the whole network.

What it does not reproduce is the graph's *structure*: the frozen, node-to-node
linearisation that lets an attribution graph be read as a circuit. These are
related but not equal quantities, and how closely they agree is an empirical
question rather than something to assume.

The student's side of a distillation loss has to be differentiable, and this
target is itself a gradient, so training on it needs double backward
(``create_graph=True`` on the inner pass). That costs roughly 2-3x a normal step.
It is also why occlusion, which is causal and needs only T forwards, cannot serve
as the student's side even though it is cheaper for the teacher.
"""

from __future__ import annotations

from typing import Any

import torch

TOKEN_PATH_ROWS = ("weighted", "gold", "all")


def salient_logits(
    logits: torch.Tensor, top_k_logits: float, temperature: float, gold_token: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(token ids, probabilities)`` for the fewest top logits reaching ``top_k_logits``
    cumulative mass, capped at 10, with ``gold_token`` appended when given and missing.

    Mirrors AttributionTargets._from_salient so a token-path run selects the same
    set the graph pipeline would have.
    """
    probs = torch.softmax(logits.float() / max(temperature, 1e-6), dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    k = int((torch.cumsum(sorted_probs, dim=-1) < top_k_logits).sum().item()) + 1
    k = min(k, 10, probs.numel())
    ids = sorted_idx[:k]
    if gold_token is not None and not bool((ids == int(gold_token)).any()):
        ids = torch.cat([ids, torch.tensor([int(gold_token)], device=ids.device, dtype=ids.dtype)])
    return ids.detach(), probs[ids].detach()


def token_attribution(
    model: Any,
    input_ids: torch.Tensor,
    read_position: int,
    logit_ids: torch.Tensor,
    *,
    rows: str = "weighted",
    logit_weights: torch.Tensor | None = None,
    gold_token: int | None = None,
    create_graph: bool = False,
    autocast: Any = None,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    """``[R, T]`` distribution(s) over input positions; R is 1 unless ``rows='all'``.

    ``read_position`` is the index whose logits are attributed from -- the last
    prompt position, the same one the KD term and the DLA reference use.

    ``rows``:
      * ``weighted`` -- one row, the logits combined with ``logit_weights`` (the
        *teacher's* probabilities, for both models; weighting each model by its own
        beliefs would confound the comparison with the difference being measured).
      * ``gold`` -- one row, attribution of the gold answer's logit alone.
      * ``all`` -- one row per entry of ``logit_ids``, which costs one backward each.

    Pass ``create_graph=True`` for the student so the loss can backprop through it.
    """
    if rows not in TOKEN_PATH_ROWS:
        raise ValueError(f"rows must be one of {TOKEN_PATH_ROWS}, got {rows!r}")
    ids = input_ids.reshape(1, -1)
    embed = model.get_input_embeddings()
    e = embed(ids)
    e = e.detach().clone().requires_grad_(True)

    ctx = autocast() if autocast is not None else torch.enable_grad()
    with ctx:
        out = model(inputs_embeds=e).logits[0, read_position]
    out = out.float()

    logit_ids = logit_ids.to(out.device).reshape(-1)
    if rows == "gold":
        if gold_token is None:
            raise ValueError("rows='gold' needs gold_token")
        targets = [out[int(gold_token)]]
    elif rows == "all":
        targets = [out[int(i)] for i in logit_ids]
    else:
        if logit_weights is None:
            raise ValueError("rows='weighted' needs logit_weights (the teacher's probabilities)")
        w = logit_weights.to(device=out.device, dtype=out.dtype).reshape(-1)
        if w.numel() != logit_ids.numel():
            raise ValueError(f"logit_weights has {w.numel()} entries for {logit_ids.numel()} logits")
        w = w / w.sum().clamp(min=epsilon)
        targets = [(w * out[logit_ids]).sum()]

    out_rows = []
    for t, target in enumerate(targets):
        retain = create_graph or t < len(targets) - 1
        g = torch.autograd.grad(target, e, create_graph=create_graph, retain_graph=retain)[0]
        attr = (g * e).sum(dim=-1).reshape(-1).abs()          # [T]
        out_rows.append(attr / attr.sum().clamp(min=epsilon))
    return torch.stack(out_rows)

"""Swap graph: a teacher/student graph whose edges are measured by interchange, in output units.

The attribution graph compared |row|-normalised gradient attributions over a
top-k subset of each model's neurons. Scored against a same-architecture
teacher its gradient had cosine -0.03 with the direction that makes the student
answer like the teacher (2026-11-13): the normalisation divides out the size of
each logit, which is what decides the answer, so a student can match the shares
without changing what it computes; and the neuron subsets, chosen separately in
each model, left an unknown part of the computation out of both graphs.

This graph keeps the supernode / edge-matrix / graph-loss recipe and replaces
how an edge is measured and what units it is in:

Nodes   No neurons are selected. A node is the whole residual-stream state of
        a set of positions at one relative depth, so everything every neuron
        and attention head wrote there is in it. For each number in the prompt
        and each depth there are two:
          number  the state at the number's own position;
          rest    the states of every later position (what has already been
                  read off the number by that depth).
        Positions before the number cannot depend on it, so at any one depth
        the two nodes together carry the number's entire influence: swapping
        both reproduces the number-changed prompt exactly (checked by
        experiments/swap_graph_inspect.py). Nothing is left unaccounted for.
Edge    node -> response: the model's next-token distributions on the response
        after the node's state is swapped for the state it has when that number
        is a different number (a "twin" prompt: the same tokens with one number
        token replaced). Depth 0 swaps the token embedding, which for the number
        node is the twin prompt itself and for the rest node changes nothing;
        depth f swaps the input of block round(f * n_layers), so the same f
        names the same relative depth in a 32-layer teacher and a 16-layer
        student.
Loss    KL(teacher's swapped distributions || student's), the divergence KD
        uses on the unswapped run. Teacher and student share the vocabulary, so
        an edge needs no normalisation and no neuron correspondence: it is zero
        exactly when the student answers the swap as the teacher does, and a
        model scored against itself gives zero at every node.

An edge costs two forward passes whatever the size of the node.

The depth-0 edges alone are KD on number-changed copies of the training
prompts, i.e. data augmentation. That is the baseline the deeper edges have to
beat before anything is credited to structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

NODE_KINDS = ("number", "number+rest")


class NumberTokens:
    """The single-token integers 0-999 and where a prompt's standalone numbers sit."""

    def __init__(self, tokenizer) -> None:
        self.id_of: Dict[int, int] = {}
        for n in range(1000):
            enc = tokenizer.encode(str(n), add_special_tokens=False)
            if len(enc) == 1:
                self.id_of[n] = int(enc[0])
        if len(self.id_of) < 900:
            raise RuntimeError(
                f"only {len(self.id_of)} of the integers 0-999 are single tokens for this tokenizer; "
                "the swap graph replaces one number token with another and needs them to be")
        self.value = torch.full((len(tokenizer),), -1, dtype=torch.long)
        for n, i in self.id_of.items():
            self.value[i] = n
        dot = tokenizer.encode(".", add_special_tokens=False)
        self.dot_id = int(dot[0]) if len(dot) == 1 else -1

    def standalone(self, ids: torch.Tensor, prompt_len: int) -> List[int]:
        """Prompt positions holding a whole integer: a number token that is not
        part of a longer digit run or a decimal."""
        toks = ids[:prompt_len].tolist()
        vals = self.value[ids[:prompt_len]].tolist()
        out = []
        for p, v in enumerate(vals):
            if v < 0:
                continue
            prev_num = p > 0 and vals[p - 1] >= 0
            next_num = p + 1 < prompt_len and vals[p + 1] >= 0
            dec_next = p + 2 < prompt_len and toks[p + 1] == self.dot_id and vals[p + 2] >= 0
            dec_prev = p >= 2 and toks[p - 1] == self.dot_id and vals[p - 2] >= 0
            if not (prev_num or next_num or dec_next or dec_prev):
                out.append(p)
        return out

    def replacement(self, value: int, taken: Sequence[int], gen: torch.Generator) -> int:
        """Another integer with as many digits as ``value`` that the prompt does not already hold."""
        digits = len(str(value))
        lo, hi = (10 ** (digits - 1) if digits > 1 else 1), 10 ** digits - 1
        taken = set(taken)
        pool = [n for n in range(lo, hi + 1) if n in self.id_of and n not in taken]
        return pool[int(torch.randint(len(pool), (1,), generator=gen))]


@dataclass
class Edges:
    """One edge per sequence: which number is swapped, for what, at which depth, through which node."""
    twin_ids: torch.Tensor  # [B, T] the sequences with that one number token replaced
    pos: torch.Tensor       # [B] position of the swapped number
    band: torch.Tensor      # [B] index into the depth list
    has: torch.Tensor       # [B] False where the prompt has no standalone number
    number: torch.Tensor    # [B] swap the state at the number's position
    rest: torch.Tensor      # [B] swap the states of every later position

    def to(self, device) -> "Edges":
        return Edges(*(t.to(device) for t in (self.twin_ids, self.pos, self.band, self.has, self.number, self.rest)))

    def at(self, band: int, number: bool = True, rest: bool = False) -> "Edges":
        """The same numbers and replacements, every sequence at one depth and node."""
        return Edges(self.twin_ids, self.pos, torch.full_like(self.band, band), self.has,
                     torch.full_like(self.has, number), torch.full_like(self.has, rest))


def pick_edges(input_ids: torch.Tensor, prompt_len: torch.Tensor, numbers: NumberTokens,
               depths: Sequence[float], gen: torch.Generator, nodes: str = "number") -> Edges:
    """Sample one (number, replacement, depth, node) per sequence. CPU tensors out.

    Under ``nodes="number+rest"`` half the edges go through the rest node, drawn
    from the depths above 0 only: at depth 0 that node holds the unchanged
    embeddings of the later tokens, so its edge is identically the unswapped run.
    """
    if nodes not in NODE_KINDS:
        raise ValueError(f"nodes must be one of {NODE_KINDS}, got {nodes!r}")
    ids = input_ids.cpu()
    B = ids.size(0)
    twin = ids.clone()
    pos = torch.zeros(B, dtype=torch.long)
    band = torch.zeros(B, dtype=torch.long)
    has = torch.zeros(B, dtype=torch.bool)
    number = torch.ones(B, dtype=torch.bool)
    rest = torch.zeros(B, dtype=torch.bool)
    deep = [i for i, d in enumerate(depths) if d > 0]
    draw = lambda n: int(torch.randint(n, (1,), generator=gen))  # noqa: E731
    for i in range(B):
        plen = int(prompt_len[i])
        cand = numbers.standalone(ids[i], plen)
        if not cand:
            continue
        p = cand[draw(len(cand))]
        taken = [int(numbers.value[ids[i, q]]) for q in cand]
        twin[i, p] = numbers.id_of[numbers.replacement(int(numbers.value[ids[i, p]]), taken, gen)]
        pos[i] = p
        has[i] = True
        if nodes == "number+rest" and deep and draw(2) == 1:
            number[i], rest[i] = False, True
            band[i] = deep[draw(len(deep))]
        else:
            band[i] = draw(len(depths))
    return Edges(twin, pos, band, has, number, rest)


def block_index(model, frac: float) -> int:
    """The decoder block whose input sits at relative depth ``frac`` (0 = the embeddings)."""
    n = len(model.model.layers)
    return min(n - 1, int(round(frac * n)))


def swap_forward(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, edges: Edges,
                 depths: Sequence[float]) -> torch.Tensor:
    """Logits of ``input_ids`` with each sequence's edge applied.

    One pass over the twin sequences records the residual stream entering every
    block; a second pass over the original sequences overwrites, for sequence i,
    its node's positions entering block ``depths[edges.band[i]]`` with the
    twin's states. Gradients flow through both passes when the caller allows them.
    """
    if not bool(edges.has.any()):
        return model(input_ids, attention_mask=attention_mask).logits
    layers = model.model.layers
    B, T = input_ids.shape
    block = torch.tensor([block_index(model, depths[int(b)]) for b in edges.band.tolist()],
                         device=input_ids.device)
    t = torch.arange(T, device=input_ids.device)[None, :]
    p = edges.pos[:, None]
    where = ((edges.number[:, None] & (t == p)) | (edges.rest[:, None] & (t > p))) & edges.has[:, None]
    source = model(edges.twin_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states
    swapped = {}
    for b in sorted(set(block[edges.has].tolist())):
        swapped[b] = layers[b]
        layers[b] = _SwapIn(layers[b], (where & (block == b)[:, None]).unsqueeze(-1), source[b])
    try:
        logits = model(input_ids, attention_mask=attention_mask).logits
    finally:
        for b, original in swapped.items():
            layers[b] = original
    return logits


class _SwapIn(torch.nn.Module):
    """Stands in for one decoder block for one forward: overwrites part of the
    incoming residual stream, then runs the block.

    A forward pre-hook on the block would do the same on the forward pass, but
    the student trains with gradient checkpointing, which replays each block's
    call during backward, after the hook is gone, and the replay then disagrees
    with the forward. Here the overwrite happens before the block's call, so the
    swapped stream is the input the checkpoint saves and replays.
    """

    def __init__(self, block, mask: torch.Tensor, state: torch.Tensor) -> None:
        super().__init__()
        self.block = block
        self.mask = mask
        self.state = state

    def forward(self, *args, **kwargs):
        if args:
            h = torch.where(self.mask, self.state.to(args[0].dtype), args[0])
            return self.block(h, *args[1:], **kwargs)
        h = kwargs.pop("hidden_states")
        return self.block(torch.where(self.mask, self.state.to(h.dtype), h), **kwargs)


SCORED = ("all", "consistent")


def consistent_positions(input_ids: torch.Tensor, response_mask: torch.Tensor, kd_mask: torch.Tensor,
                         edges: Edges) -> torch.Tensor:
    """The scored response positions whose teacher-forced context does not contradict the swap.

    The response fed to both models is the original problem's. Once it has
    spelled out the original number, the context says one number and the
    swapped state says another, and what either model predicts after that is
    its reaction to a contradiction, not its answer to the swap. Keep a position
    only while no response token up to and including it is the original number:
    that is every position up to the one that predicts the number's first
    appearance, and all of them when the response never mentions it.
    """
    rows = torch.arange(input_ids.size(0), device=input_ids.device)
    old = input_ids[rows, edges.pos]
    said = ((input_ids == old[:, None]) & response_mask.bool()).long().cumsum(dim=1) > 0
    return kd_mask & ~said


def prompt_end_effect(clean_logits: torch.Tensor, swapped_logits: torch.Tensor,
                      first_response: torch.Tensor) -> torch.Tensor:
    """[B] how far the swap moves the first response token's distribution: KL(clean || swapped), nats.

    This is the edge's weight for display; the loss compares the full swapped
    distributions between the two models.
    """
    rows = torch.arange(clean_logits.size(0), device=clean_logits.device)
    a = F.log_softmax(clean_logits[rows, first_response - 1].float(), dim=-1)
    b = F.log_softmax(swapped_logits[rows, first_response - 1].float(), dim=-1)
    return (a.exp() * (a - b)).sum(-1)

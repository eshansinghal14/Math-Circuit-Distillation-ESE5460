"""What does the attribution graph attend to? Share of attribution by kind of token.

For N problems and each graph position (the first answer token, the first
operator, the computed result), build a model's graph exactly as the gate does
(position supernodes, influence selection, N neurons per node) and split two
rows of it by the kind of input token each source node sits at:

    answer row   each position node's signed attribution to the attributed
                 token (summed |.|), plus the token embeddings' own
    last row     the trainer's scored row: the |.|-normalised inbound edges of
                 the last-position node

Kinds: number (a number in the problem), response (tokens of the answer written
so far), template (chat markers, the instruction "Write the equation and its
result.", the last position itself), word (every other word of the problem).

If most of the attribution sits on template tokens and little on the numbers
and words of the problem, the graph describes the format, not the reasoning.

    cd src && python -u -m experiments.graph_content --model <path> --label teacher --n 48
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DIR_ROOT, load_model, load_split, set_bos_mode  # noqa: E402
from graph_loss.create_graph import create_graph  # noqa: E402
from graph_loss.graph import normalize_matrix  # noqa: E402
from graph_loss.hf_adapter import HFLlamaGraphAdapter  # noqa: E402
from experiments.graph_gate_two_phase import graph_cut  # noqa: E402

KINDS = ("number", "response", "template", "word")
INSTRUCTION = {"Write", "the", "equation", "and", "its", "result", "."}


def kinds_of(tok, ids: list[int], prompt_len: int) -> list[str]:
    """One kind per position of the graph's input."""
    text = [tok.decode([i]) for i in ids]
    out = []
    # the instruction sentence sits just before the chat end marker
    try:
        eot = max(i for i, t in enumerate(text[:prompt_len]) if t == "<|eot_id|>")
    except ValueError:
        eot = prompt_len
    instr_start = eot
    for i in range(eot - 1, -1, -1):
        if text[i].strip() == "Write":
            instr_start = i
            break
    for i, t in enumerate(text):
        if i >= prompt_len:
            out.append("response")
        elif t.startswith("<|") or i >= instr_start or i == 0 or t.strip() in ("user", "assistant", ""):
            out.append("template")
        elif re.fullmatch(r"\s?\d+", t):
            out.append("number")
        else:
            out.append("word")
    out[-1] = "template" if len(ids) == prompt_len else out[-1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="model")
    ap.add_argument("--dataset", default="svamp_eq")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--nodes-per-label", type=int, default=100)
    ap.add_argument("--prop", type=float, default=0.003)
    args = ap.parse_args()

    set_bos_mode("on")
    dev = torch.device("cuda")
    model, tok = load_model(args.model)
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    ad = HFLlamaGraphAdapter(model, tok, dev)
    every = sorted(load_split(args.dataset, args.split).items())
    items = every[::max(1, len(every) // args.n)][: args.n]
    kw = dict(top_k_logits=0.95, temperature=1.0, dtype=torch.bfloat16, node_labels=[],
              nodes_per_label=args.nodes_per_label, no_grad_supergraph=True, build_create_graph=False,
              neuron_selection="influence", supernode_type="position", batch_size=16, detach_result=True)
    out = {}
    for where in ("prompt-end", "operator", "result"):
        tot = {row: {k: 0.0 for k in KINDS} for row in ("answer", "last")}
        n_pos = {k: 0 for k in KINDS}
        n = 0
        for prompt, answer in items:
            cut = graph_cut(prompt, answer, tok, args.dataset, where)
            if cut is None:
                continue
            g_prompt, _, _ = cut
            plen = len(tok(prompt, add_special_tokens=False)["input_ids"])
            with torch.enable_grad():
                res = create_graph(ad, g_prompt, prop_neurons_per_layer=args.prop, target_rows="last", **kw)
            g, sg = res.graph, res.supergraph
            ids = g.input_tokens.tolist()
            kinds = kinds_of(tok, ids, plen)
            nn_, nt = g.n_neurons, g.n_tokens
            A = g.adjacency_matrix.float()
            locs = g.neuron_locations[:, 1].tolist()
            probs = g.logit_probabilities.float().to(A.device)
            ans = ((probs / probs.sum())[:, None] * A[nn_ + nt:, : nn_ + nt]).sum(0).abs()
            E = normalize_matrix(A)
            # the trainer's last row: the last-position node's members' normalised inbound edges
            last = max(range(len(sg.supernodes)), key=lambda s: max((locs[m] for m in sg.supernodes[s]), default=-1)
                       if (sg.supernode_labels[s] or [""])[0].startswith("pos:") else -1)
            members = sg.supernodes[last]
            row = E[members].mean(0)
            for row_name, vec in (("answer", ans), ("last", row)):
                mass = {k: 0.0 for k in KINDS}
                for j in range(nn_):
                    mass[kinds[locs[j]]] += float(vec[j])
                for p in range(nt):
                    mass[kinds[p]] += float(vec[nn_ + p])
                s = sum(mass.values()) or 1.0
                for k in KINDS:
                    tot[row_name][k] += mass[k] / s
            for k in kinds:
                n_pos[k] += 1
            n += 1
            del res
        out[where] = {"n": n, **{row: {k: v / max(n, 1) for k, v in d.items()} for row, d in tot.items()},
                      "token_share": {k: v / max(sum(n_pos.values()), 1) for k, v in n_pos.items()}}
        r = out[where]
        print(f"\n== {args.label}, graph at {where} ({n} problems) ==")
        print(f"   {'':12s} " + " ".join(f"{k:>10s}" for k in KINDS))
        print(f"   {'tokens':12s} " + " ".join(f"{r['token_share'][k]:10.3f}" for k in KINDS))
        print(f"   {'answer row':12s} " + " ".join(f"{r['answer'][k]:10.3f}" for k in KINDS))
        print(f"   {'last row':12s} " + " ".join(f"{r['last'][k]:10.3f}" for k in KINDS), flush=True)
    path = os.path.join(DIR_ROOT, f"FINAL_RESULTS/{args.dataset}/graph_content_{args.label}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

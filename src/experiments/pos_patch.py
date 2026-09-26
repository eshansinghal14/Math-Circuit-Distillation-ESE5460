"""experiments.pos_patch: patch the graph student's residual stream (after layer L, at one position) into the KD student.

21_mult prompts "a*b=" tokenise to exactly [BOS, a, *, b, =] with Llama-3, so positions are fixed.
Usage: PYTHONPATH=src python -m experiments.pos_patch <kd dir> <graph dir> <out.json> [device]
Metrics on the KD student at the answer position: commit (top-1 token is a number or whitespace-only,
the graph students' lead-in) and p_q (mass on tokens starting with '?').
"""
import json, os, re, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_data
from training.utils import load_student

dev = torch.device(sys.argv[4] if len(sys.argv) > 4 else "cuda")
kd, tok = load_student(sys.argv[1], "bfloat16"); g, _ = load_student(sys.argv[2], "bfloat16")
kd.to(dev).eval(); g.to(dev).eval()
text = [tok.decode([i]) for i in range(len(tok))]
ok = torch.tensor([bool(re.fullmatch(r"\s?\d+|\s+", t)) for t in text], device=dev)
q = torch.tensor([t.lstrip().startswith("?") for t in text], device=dev)
data = load_data("21_mult", test_limit=1000)[1]
enc = tok(list(data), return_tensors="pt", padding=True, add_special_tokens=True).to(dev)
assert enc["input_ids"].shape[1] == 5 and bool(enc["attention_mask"].all()), "expected [BOS,a,*,b,=]"
ids = enc["input_ids"]

@torch.no_grad()
def run(model, patch=None):
    hooks = []
    if patch is not None:
        L, pos, src = patch
        def h(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            hs = hs.clone(); hs[:, pos] = src[:, pos].to(hs.dtype)
            return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
        hooks.append(model.model.layers[L].register_forward_hook(h))
    try:
        out = model(ids, output_hidden_states=True)
    finally:
        for hk in hooks: hk.remove()
    p = torch.softmax(out.logits[:, -1].float(), -1)
    return {"commit": float(ok[p.argmax(-1)].float().mean()), "p_q": float(p[:, q].sum(-1).mean())}, out.hidden_states

res = {}
res["kd"], _ = run(kd)
res["graph"], hs_g = run(g)
print("kd", res["kd"], "graph", res["graph"])
for L in (1, 3, 7, 11):
    for pos, name in [(0, "BOS"), (1, "a"), (2, "*"), (3, "b"), (4, "="), (slice(None), "all")]:
        r, _ = run(kd, (L, pos, hs_g[L + 1]))
        res[f"L{L}/{name}"] = r
        print(f"L{L:2d} {name:4s} commit {r['commit']:.3f} p_q {r['p_q']:.3f}")
json.dump(res, open(sys.argv[3], "w"), indent=1)

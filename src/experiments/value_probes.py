"""Where do models compute the intermediate values of mix3, and do those values explain their errors?

mix3 prompts are "BOS a o1 b o2 c =" (7 tokens). For each model, the residual
stream at every layer is recorded at positions b (3), o2 (4), c (5) and "=" (6)
for N_train training problems and the N_test test problems graph_metrics uses.
Linear ridge probes, fitted per (layer, position) on the training problems with
the model's own states, read off:

    partial   p = a o1 b                    (regression)
    result    r = p o2 c                    (regression)
    r_tens    tens digit of r               (10-way, one-vs-rest ridge, argmax)
    r_units   units digit of r              (10-way)
    p_tens    tens digit of p mod 100       (10-way)

Reported per model:
  1. probe quality on the test problems for each target at each position, at
     the best layer (and the relative depth of that layer);
  2. on the problems the model answers wrong: is the partial sum it holds
     (probe at the best layer for "partial") right, and is the result tens
     digit decodable at an intermediate layer even though the answer is wrong;
  3. AUC of per-problem probe error for "the model is wrong" (0.5 = none),
     to set against the attribution graph's 0.57-0.59 for these students.

    cd src && python -u -m experiments.value_probes --model kd150=<path> --model ft8b=<path>
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DIR_ROOT, load_data, load_model, set_bos_mode  # noqa: E402

POS = {"b": 3, "o2": 4, "c": 5, "=": 6}
TARGETS = ("partial", "result", "p_tens", "r_tens", "r_units", "p_mod100", "r_mod100")
FOURIER = (10, 100)


def parse(prompt: str, answer) -> dict:
    a, b, c = (int(x) for x in re.findall(r"\d+", prompt))
    o1, o2 = re.findall(r"[+-]", prompt)
    p = a + b if o1 == "+" else a - b
    r = int(answer)
    return {"partial": p, "result": r, "p_tens": (abs(p) // 10) % 10, "r_tens": (r // 10) % 10, "r_units": r % 10}


@torch.no_grad()
def states(model, tok, prompts: list[str], batch: int = 64) -> tuple[torch.Tensor, list[int]]:
    """[N, L+1, 4, d] float16 on cpu; and each model's greedy answer token id."""
    out, preds = [], []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors="pt", padding=False)
        ids = torch.stack([torch.tensor(x) for x in enc["input_ids"]]).to(model.device)
        assert ids.shape[1] == 7, (ids.shape, prompts[i])
        o = model(ids, output_hidden_states=True)
        hs = torch.stack(o.hidden_states, 1)[:, :, list(POS.values())]  # [B, L+1, 4, d]
        out.append(hs.to(torch.float16).cpu())
        preds += o.logits[:, -1].argmax(-1).tolist()
    return torch.cat(out), preds


def ridge(X: torch.Tensor, Y: torch.Tensor, lam: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu, sd = X.mean(0), X.std(0).clamp(min=1e-4)
    Z = (X - mu) / sd
    ym = Y.mean(0)
    A = Z.T @ Z + lam * torch.eye(Z.shape[1], device=Z.device)
    W = torch.linalg.solve(A, Z.T @ (Y - ym))
    return W, mu, sd, ym


def predict(X, W, mu, sd, ym):
    return ((X - mu) / sd) @ W + ym


def mod100(cs: torch.Tensor) -> torch.Tensor:
    """Value mod 100 from a predicted (cos, sin) pair of period 100."""
    ang = torch.atan2(cs[:, 1], cs[:, 0])
    return torch.round(ang / (2 * torch.pi) * 100).long() % 100


def auc(wrong, right):
    if not wrong or not right:
        return float("nan")
    return sum((w > r) + 0.5 * (w == r) for w in wrong for r in right) / (len(wrong) * len(right))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", action="append", required=True, help="label=path; repeatable")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--lam", type=float, default=100.0)
    args = ap.parse_args()
    set_bos_mode("on")
    dev = torch.device("cuda")
    train, test = load_data("mix3")
    tr = sorted(train.items())[: args.n_train]
    every = sorted(test.items())
    te = every[::max(1, len(every) // args.n_test)][: args.n_test]
    T_tr = [parse(p, a) for p, a in tr]
    T_te = [parse(p, a) for p, a in te]

    def ymat(T):
        cols = [torch.tensor([t["partial"] for t in T], dtype=torch.float32),
                torch.tensor([t["result"] for t in T], dtype=torch.float32)]
        for k in ("p_tens", "r_tens", "r_units"):
            cols.append(torch.nn.functional.one_hot(torch.tensor([t[k] for t in T]), 10).float())
        # periodic features: cos / sin of 2 pi v / period, for v in (partial, result)
        for key in ("partial", "result"):
            v = torch.tensor([t[key] for t in T], dtype=torch.float32)
            for per in FOURIER:
                ang = 2 * torch.pi * v / per
                cols.append(torch.stack([ang.cos(), ang.sin()], 1))
        return torch.cat([c if c.dim() == 2 else c[:, None] for c in cols], 1)

    Ytr = ymat(T_tr).to(dev)
    out = {}
    for spec in args.model:
        label, path = spec.split("=", 1)
        model, tok = load_model(path)
        model.eval()
        Htr, _ = states(model, tok, [p for p, _ in tr])
        Hte, preds = states(model, tok, [p for p, _ in te])
        gold_ids = [tok(" " + str(t["result"]), add_special_tokens=False)["input_ids"] for t in T_te]
        gold_ids2 = [tok(str(t["result"]), add_special_tokens=False)["input_ids"] for t in T_te]
        hit = [preds[i] in (gold_ids[i][0], gold_ids2[i][0]) for i in range(len(te))]
        pred_val = []
        for i, pid in enumerate(preds):
            s = tok.decode([pid]).strip()
            pred_val.append(int(s) if s.lstrip("-").isdigit() else None)
        L1 = Htr.shape[1]
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # fit every (layer, position); score on test
        score = {k: torch.full((L1, 4), float("nan")) for k in TARGETS}
        preds_te = {}
        for li in range(L1):
            for pj in range(4):
                X = Htr[:, li, pj].float().to(dev)
                W, mu, sd, ym = ridge(X, Ytr, args.lam)
                P = predict(Hte[:, li, pj].float().to(dev), W, mu, sd, ym).cpu()
                preds_te[(li, pj)] = P
                tp = torch.tensor([t["partial"] for t in T_te], dtype=torch.float32)
                trr = torch.tensor([t["result"] for t in T_te], dtype=torch.float32)
                r2 = lambda yhat, y: float(1 - ((yhat - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())  # noqa: E731
                score["partial"][li, pj] = r2(P[:, 0], tp)
                score["result"][li, pj] = r2(P[:, 1], trr)
                for k, off in (("p_tens", 2), ("r_tens", 12), ("r_units", 22)):
                    y = torch.tensor([t[k] for t in T_te])
                    score[k][li, pj] = float((P[:, off:off + 10].argmax(1) == y).float().mean())
                for k, key, off in (("p_mod100", "partial", 34), ("r_mod100", "result", 38)):
                    y = torch.tensor([t[key] % 100 for t in T_te])
                    score[k][li, pj] = float((mod100(P[:, off:off + 2]) == y).float().mean())
        rec = {"accuracy": sum(hit) / len(hit), "layers": L1 - 1, "best": {}}
        print(f"\n== {label}: answer accuracy {rec['accuracy']:.3f} on {len(te)} test problems, {L1 - 1} layers ==")
        print(f"   best probe on test (R^2 for values, accuracy for digits) and its relative depth, per position")
        print(f"   {'target':8s} " + " ".join(f"{p:>14s}" for p in POS))
        for k in TARGETS:
            cells = []
            for pj, pname in enumerate(POS):
                col = score[k][:, pj]
                li = int(torch.nan_to_num(col, nan=-9).argmax())
                cells.append(f"{float(col[li]):6.2f} @{li / (L1 - 1):4.2f}")
                rec["best"][f"{k}@{pname}"] = {"score": float(col[li]), "layer": li, "depth": li / (L1 - 1)}
            print(f"   {k:8s} " + " ".join(f"{c:>14s}" for c in cells))

        # error analysis on the problems the model gets wrong
        wrong = [i for i in range(len(te)) if not hit[i]]
        right = [i for i in range(len(te)) if hit[i]]
        bp = rec["best"]["partial@c"]["layer"]
        P = preds_te[(bp, 2)]
        p_err = [abs(float(P[i, 0]) - T_te[i]["partial"]) for i in range(len(te))]
        p_tens_ok = [int(P[i, 2:12].argmax()) == T_te[i]["p_tens"] for i in range(len(te))]
        # result tens at "=" across depth: first layer where it is decodable best
        bl = rec["best"]["r_tens@="]["layer"]
        R = preds_te[(bl, 3)]
        r_tens_hat = [int(R[i, 12:22].argmax()) for i in range(len(te))]
        r_tens_ok = [r_tens_hat[i] == T_te[i]["r_tens"] for i in range(len(te))]
        follows_answer = [pred_val[i] is not None and r_tens_hat[i] == (pred_val[i] // 10) % 10 for i in range(len(te))]
        res_err = [abs(float(R[i, 1]) - T_te[i]["result"]) for i in range(len(te))]
        frac = lambda v, idx: sum(v[i] for i in idx) / max(len(idx), 1)  # noqa: E731
        rec["wrong_n"] = len(wrong)
        rec["partial_tens_right_on_wrong"] = frac(p_tens_ok, wrong)
        rec["partial_tens_right_on_right"] = frac(p_tens_ok, right)
        rec["result_tens_probe_gold_on_wrong"] = frac(r_tens_ok, wrong)
        rec["result_tens_probe_follows_answer_on_wrong"] = frac(follows_answer, wrong)
        rec["auc_partial_error"] = auc([p_err[i] for i in wrong], [p_err[i] for i in right])
        rec["auc_result_error"] = auc([res_err[i] for i in wrong], [res_err[i] for i in right])
        rec["auc_result_tens_wrong"] = auc([1 - r_tens_ok[i] for i in wrong], [1 - r_tens_ok[i] for i in right])
        print(f"   on the {len(wrong)} wrong problems: partial-sum tens digit decoded right {rec['partial_tens_right_on_wrong']:.2f} "
              f"(right problems {rec['partial_tens_right_on_right']:.2f}); at '=' (depth {bl / (L1 - 1):.2f}) the result tens probe "
              f"gives the gold digit {rec['result_tens_probe_gold_on_wrong']:.2f}, the model's own wrong digit "
              f"{rec['result_tens_probe_follows_answer_on_wrong']:.2f}")
        print(f"   AUC for wrong: partial-sum probe error {rec['auc_partial_error']:.2f} | result probe error "
              f"{rec['auc_result_error']:.2f} | result tens probe wrong {rec['auc_result_tens_wrong']:.2f}")
        # the same questions with the periodic (mod 100) readouts, which carry the tens digit
        lp = rec["best"]["p_mod100@c"]["layer"]
        lr = rec["best"]["r_mod100@="]["layer"]
        pm = mod100(preds_te[(lp, 2)][:, 34:36])
        rm = mod100(preds_te[(lr, 3)][:, 38:40])
        p_ok = [int(pm[i]) == T_te[i]["partial"] % 100 for i in range(len(te))]
        r_ok = [int(rm[i]) == T_te[i]["result"] % 100 for i in range(len(te))]
        r_follow = [pred_val[i] is not None and int(rm[i]) == pred_val[i] % 100 for i in range(len(te))]
        rec["p_mod100_right_on_wrong"] = frac(p_ok, wrong)
        rec["p_mod100_right_on_right"] = frac(p_ok, right)
        rec["r_mod100_gold_on_wrong"] = frac(r_ok, wrong)
        rec["r_mod100_follows_answer_on_wrong"] = frac(r_follow, wrong)
        rec["auc_p_mod100_wrong"] = auc([1 - p_ok[i] for i in wrong], [1 - p_ok[i] for i in right])
        rec["auc_r_mod100_wrong"] = auc([1 - r_ok[i] for i in wrong], [1 - r_ok[i] for i in right])
        print(f"   periodic readouts: partial mod 100 (at c, depth {lp / (L1 - 1):.2f}) right on wrong problems "
              f"{rec['p_mod100_right_on_wrong']:.2f} vs right problems {rec['p_mod100_right_on_right']:.2f}; "
              f"result mod 100 (at '=', depth {lr / (L1 - 1):.2f}) on wrong problems = gold {rec['r_mod100_gold_on_wrong']:.2f}, "
              f"= the model's own answer {rec['r_mod100_follows_answer_on_wrong']:.2f}")
        print(f"   AUC for wrong: partial mod-100 probe wrong {rec['auc_p_mod100_wrong']:.2f} | result mod-100 probe wrong "
              f"{rec['auc_r_mod100_wrong']:.2f}")
        rec["score_by_layer"] = {k: v.tolist() for k, v in score.items()}
        out[label] = rec
        del Htr, Hte, preds_te
        gc.collect()

    path = os.path.join(DIR_ROOT, "FINAL_RESULTS/mix3/value_probes.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

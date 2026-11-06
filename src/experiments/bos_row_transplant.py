"""Is a trained student's gain carried by its BOS embedding row? Eval only, no training.

Under --bos-mode legacy the KD term never feeds BOS in, so the BOS row of the
embedding is the one parameter the graph term reaches that KD does not. If the
graph arm's out-of-distribution gain lives there, moving that one row (2048
numbers on the 1B) between the untrained and the trained student moves the gain
with it. Four models are scored on every split:

    base                    the untrained student
    trained                 the checkpoint
    base + trained BOS row  only the BOS row taken from the checkpoint
    trained + base BOS row  the checkpoint with its BOS row reset

Read: the gain follows the row -> the row is the mechanism; the gain stays with
``trained`` and ``base + trained row`` does nothing -> it lives in the shared
weights. The row is also reported numerically (relative change, fraction of
elements that moved at all -- bf16 drops Adam steps below half a ULP), next to
the same statistics for the rows of the tokens the prompts do use, so a moved
BOS row can be told from an embedding that moved everywhere.

Llama-3.2 ties the input embedding to the lm_head, so the row is also BOS's
output row; nothing the eval generates depends on that row being BOS-likely.

    PYTHONPATH=src python -m experiments.bos_row_transplant \\
        --trained "<save-dir>/final_checkpoint" --bos-mode legacy \\
        --datasets 22_add 222_add 2222_add 33_add 21_mult --test-limit 2000 \\
        --max-eval-tokens 5 --out FINAL_RESULTS/bos_mechanism/transplant_<name>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DIR_ROOT, BOS_MODES, PromptAnswerDataset, eval_model, load_data, load_model, set_bos_mode  # noqa: E402
from training.utils import DTYPES, student_autocast  # noqa: E402


def _row_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.float(), b.float()
    return {
        "rel_change": float((b - a).norm() / a.norm().clamp_min(1e-12)),
        "changed_frac": float((a != b).float().mean()),
        "cos": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
        "norm_base": float(a.norm()),
        "norm_trained": float(b.norm()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--trained", required=True, help="A final_checkpoint / checkpoint folder.")
    ap.add_argument("--datasets", nargs="+", default=["22_add", "222_add", "2222_add", "33_add", "21_mult"])
    ap.add_argument("--test-limit", type=int, default=2000, dest="test_limit")
    ap.add_argument("--max-eval-tokens", type=int, default=5, dest="max_eval_tokens")
    ap.add_argument("--eval-batch-size", type=int, default=1024, dest="eval_batch_size")
    ap.add_argument("--bos-mode", choices=sorted(BOS_MODES), default="legacy", dest="bos_mode",
                    help="Only decides whether eval prepends BOS (on and legacy both do).")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16",
                    help="Load precision for both models; the checkpoints were trained in bfloat16.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    set_bos_mode(args.bos_mode)
    dtype = DTYPES[args.dtype]
    base, tok = load_model(args.base, dtype=dtype)
    trained, _ = load_model(args.trained, dtype=dtype)
    bos = tok.bos_token_id
    e_base = base.get_input_embeddings().weight
    e_tr = trained.get_input_embeddings().weight
    row_base = e_base[bos].detach().clone()
    row_tr = e_tr[bos].detach().clone()

    # The rows the prompts actually use, for comparison with the BOS row.
    used = set()
    tests = {}
    for ds in args.datasets:
        _, test = load_data(ds, test_limit=args.test_limit)
        tests[ds] = PromptAnswerDataset(ds, test, tok)
        for s in tests[ds].samples[:200]:
            used.update(int(t) for t in tok(s["formatted_prompt"], add_special_tokens=False)["input_ids"])
            used.update(int(t) for t in tok(str(s["answer"]), add_special_tokens=False)["input_ids"])
    used.discard(bos)
    used_rows = sorted(used)
    per_row = [_row_stats(e_base[i].detach(), e_tr[i].detach()) for i in used_rows]
    med = lambda key: float(torch.tensor([r[key] for r in per_row]).median())  # noqa: E731
    with torch.no_grad():
        num = den = 0.0
        for (n, pb), (_, pt) in zip(base.named_parameters(), trained.named_parameters()):
            num += float((pt.float() - pb.float()).norm() ** 2)
            den += float(pb.float().norm() ** 2)
    stats = {
        "bos_row": _row_stats(row_base, row_tr),
        "prompt_token_rows_median": {k: med(k) for k in ("rel_change", "changed_frac", "cos")},
        "n_prompt_token_rows": len(used_rows),
        "whole_model_rel_change": (num / den) ** 0.5,
    }

    def score(model) -> dict:
        out = {}
        with student_autocast():
            for ds, td in tests.items():
                out[ds] = eval_model(model, tok, td, ds, args.eval_batch_size, args.max_eval_tokens)
        return out

    results = {}
    with torch.no_grad():
        results["base"] = score(base)
        results["trained"] = score(trained)
        e_base[bos] = row_tr.to(e_base.dtype)
        results["base + trained BOS row"] = score(base)
        e_base[bos] = row_base
        e_tr[bos] = row_base.to(e_tr.dtype)
        results["trained + base BOS row"] = score(trained)
        e_tr[bos] = row_tr

    ood = [d for d in args.datasets if d != args.datasets[0]]
    print("\n=== BOS row vs the prompt-token rows ===")
    b = stats["bos_row"]
    p = stats["prompt_token_rows_median"]
    print(f"BOS row: rel change {b['rel_change']:.3e}, elements moved {b['changed_frac']:.3f}, cos {b['cos']:.5f}")
    print(f"prompt-token rows (median of {stats['n_prompt_token_rows']}): rel change {p['rel_change']:.3e}, "
          f"elements moved {p['changed_frac']:.3f}")
    print(f"whole model: rel change {stats['whole_model_rel_change']:.3e}")
    print("\n=== accuracy ===")
    print(f"{'model':26s} " + " ".join(f"{d:>9s}" for d in args.datasets) + (f" {'OOD mean':>9s}" if ood else ""))
    for name, accs in results.items():
        mean = sum(accs[d] for d in ood) / len(ood) if ood else None
        print(f"{name:26s} " + " ".join(f"{accs[d]:9.3f}" for d in args.datasets)
              + (f" {mean:9.3f}" if mean is not None else ""))

    out = args.out or os.path.join("FINAL_RESULTS", "bos_mechanism",
                                   f"transplant_{os.path.basename(os.path.dirname(os.path.abspath(args.trained)))}.json")
    out = out if os.path.isabs(out) else os.path.join(DIR_ROOT, out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "row_stats": stats, "accuracy": results}, f, indent=2)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()

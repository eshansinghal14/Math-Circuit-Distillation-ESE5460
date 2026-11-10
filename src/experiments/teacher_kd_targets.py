"""What KD imitates: the teacher's distribution at the positions the KD term scores.

Graph distillation alone reaches 21_mult 0.97 under the standard BOS convention
while soft KD (KL to the teacher, --kl-tokens resp) stays at 0.33-0.39 although
it fits 22_add (KL 0.13, in-dist 0.98), and SFT on the gold answer -- which the
teacher's top-1 matches ~96% of the time -- reaches 0.93. So the suspect is the
part of the teacher's distribution that is *not* its top-1 answer. This script
measures it, with no training:

Per answer position k (k=0 predicts the first answer token, the last k predicts
the EOS after the answer; Llama-3 writes numbers up to 999 as one token, so on
22_add and 21_mult that is k=0 the number, k=1 the EOS):

* teacher entropy, top-1 probability, top-1 = gold rate, probability on gold;
* where the rest of the teacher's mass goes: the tokens with the most non-gold
  mass, summed over prompts;
* for each student (the untrained one by default; pass checkpoint dirs to
  compare trained ones): KL(teacher || student) -- exactly the KD loss at tau 1 --
  its split across positions, and the tokens that contribute most of it
  (sum over prompts of p_t(v) * (log p_t(v) - log q_s(v))), plus the student's
  own entropy / top-1 = gold / gold probability.

Datasets: the KD training prompts (``--dataset``, train split) and any
``--extra-datasets`` (test split), e.g. 21_mult, to see what the teacher's
distribution looks like where the students differ.

    cd src && python -m experiments.teacher_kd_targets \\
        --teacher meta-llama/Meta-Llama-3-8B-Instruct \\
        --students meta-llama/Llama-3.2-1B-Instruct \\
        --dataset 22_add --extra-datasets 21_mult --n 1000 --bos-mode on \\
        --out FINAL_RESULTS/bos_mechanism/teacher_kd_targets.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (  # noqa: E402
    DIR_ROOT, PromptAnswerDataset, collate_fn, load_data, load_split, set_bos_mode,
)
from training.utils import kl_position_mask, load_student  # noqa: E402

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prompts(dataset: str, split: str, n: int) -> Dict:
    if split == "train":
        data, _ = load_data(dataset)
    else:
        data = load_split(dataset, "test")
    items = list(data.items())[:n]
    return dict(items)


class _Acc:
    """Running sums for one (dataset, model, position k)."""

    def __init__(self, vocab: int) -> None:
        self.n = 0
        self.sums = defaultdict(float)
        self.vocab_mass = torch.zeros(vocab, dtype=torch.float64)  # teacher non-gold mass / KL contribution

    def add(self, key: str, values: torch.Tensor) -> None:
        self.sums[key] += float(values.double().sum())

    def mean(self, key: str) -> float:
        return self.sums[key] / max(self.n, 1)


def _top_tokens(vec: torch.Tensor, tok, n_prompts: int, k: int = 12) -> List[List]:
    vals, ids = torch.topk(vec, min(k, vec.numel()))
    return [[ascii(tok.decode([int(i)])), round(float(v) / max(n_prompts, 1), 5)] for v, i in zip(vals, ids)]


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--teacher", required=True)
    p.add_argument("--students", nargs="*", default=["meta-llama/Llama-3.2-1B-Instruct"],
                   help="HF ids or checkpoint dirs (relative to DIR_ROOT or absolute).")
    p.add_argument("--dataset", default="22_add")
    p.add_argument("--extra-datasets", nargs="*", default=["21_mult"])
    p.add_argument("--n", type=int, default=1000, help="Prompts per dataset.")
    p.add_argument("--bos-mode", default="on", choices=["on", "legacy", "off"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--out", default="FINAL_RESULTS/bos_mechanism/teacher_kd_targets.json")
    args = p.parse_args()
    set_bos_mode(args.bos_mode)

    teacher, tok = load_student(args.teacher, "bfloat16")
    teacher.eval()
    students = {}
    data_tok = None
    for s in args.students:
        path = s if (os.path.isabs(s) or not os.path.isdir(os.path.join(DIR_ROOT, s))) else os.path.join(DIR_ROOT, s)
        m, s_tok = load_student(path, "bfloat16")
        # Llama-3 and Llama-3.2 share every regular token id but name some reserved
        # special slots differently, so compare the base vocabulary and the ids the
        # sequences actually use (BOS, EOS), not the full map.
        base = lambda t: {k: v for k, v in t.get_vocab().items() if k not in t.get_added_vocab()}
        assert base(s_tok) == base(tok), f"{s}: regular vocabulary differs from the teacher's"
        if (s_tok.bos_token_id, s_tok.eos_token_id) != (tok.bos_token_id, tok.eos_token_id):
            # Not an error: the KD trainer builds sequences with the student's
            # tokenizer and feeds the same ids to the teacher, and so does this script.
            print(f"note: {s} BOS/EOS ids {(s_tok.bos_token_id, s_tok.eos_token_id)} vs teacher "
                  f"{(tok.bos_token_id, tok.eos_token_id)}; sequences use the student's, as in KD training")
        data_tok = data_tok or s_tok  # build sequences with the student's tokenizer, as the KD trainer does
        m.eval()
        students[s] = m

    sets = [(args.dataset, "train")] + [(d, "test") for d in args.extra_datasets]
    report: Dict = {"args": vars(args), "datasets": {}}
    for ds, split in sets:
        data = _prompts(ds, split, args.n)
        dataset = PromptAnswerDataset(ds, data, data_tok or tok)
        names = ["teacher"] + list(students)
        accs: Dict[str, Dict[int, _Acc]] = {nm: {} for nm in names}
        n_rows = 0
        for start in range(0, len(dataset), args.batch_size):
            batch = collate_fn([dataset[i] for i in range(start, min(start + args.batch_size, len(dataset)))],
                               pad_id=(data_tok or tok).eos_token_id)
            ids = batch["input_ids"].to(_DEVICE)
            att = batch["attention_mask"].to(_DEVICE)
            resp = batch["response_mask"].to(_DEVICE)
            mask = kl_position_mask(att, resp, "resp")             # [B, L] positions KD scores
            first = resp.int().argmax(dim=1)                       # index of the first answer token
            rows, cols = mask.nonzero(as_tuple=True)
            k_of = (cols + 1 - first[rows]).tolist()               # 0 = predicts first answer token
            gold = ids[rows, cols + 1]
            n_rows += ids.shape[0]

            t_logits = teacher(ids, attention_mask=att).logits[rows, cols].float()
            log_pt = F.log_softmax(t_logits, dim=-1)
            pt = log_pt.exp()
            V = pt.shape[-1]
            per_model = {"teacher": (log_pt, pt)}
            for nm, m in students.items():
                s_logits = m(ids, attention_mask=att).logits[rows, cols].float()[:, :V]
                ls = F.log_softmax(s_logits, dim=-1)
                per_model[nm] = (ls, ls.exp())

            for k in sorted(set(k_of)):
                sel = torch.tensor([i for i, kk in enumerate(k_of) if kk == k], device=_DEVICE)
                g = gold[sel]
                lt, ptk = log_pt[sel], pt[sel]
                for nm, (lq, q) in per_model.items():
                    acc = accs[nm].setdefault(k, _Acc(V))
                    lqk, qk = lq[sel], q[sel]
                    acc.n += sel.numel()
                    acc.add("entropy", -(qk * lqk).sum(-1))
                    acc.add("top1_prob", qk.max(-1).values)
                    acc.add("top1_is_gold", (qk.argmax(-1) == g).float())
                    acc.add("gold_prob", qk.gather(1, g[:, None]).squeeze(1))
                    if nm == "teacher":
                        nongold = ptk.clone()
                        nongold.scatter_(1, g[:, None], 0.0)
                        acc.add("nongold_mass", nongold.sum(-1))
                        acc.vocab_mass += nongold.sum(0).double().cpu()
                    else:
                        contrib = ptk * (lt - lqk)                 # per-token KL terms
                        kl = contrib.sum(-1)
                        acc.add("kl_to_teacher", kl)
                        # KL from the gold token alone vs everything else
                        acc.add("kl_gold_term", contrib.gather(1, g[:, None]).squeeze(1))
                        acc.vocab_mass += contrib.sum(0).double().cpu()

        out_ds: Dict = {"split": split, "n_prompts": n_rows, "models": {}}
        print(f"\n=== {ds} ({split}, {n_rows} prompts, bos-mode {args.bos_mode}) ===")
        for nm in names:
            out_m = {}
            for k, acc in sorted(accs[nm].items()):
                row = {key: acc.mean(key) for key in acc.sums}
                row["n_positions"] = acc.n
                label = "top non-gold teacher mass" if nm == "teacher" else "top KL contributors"
                row[label] = _top_tokens(acc.vocab_mass, tok, acc.n)
                out_m[str(k)] = row
                stats = " | ".join(f"{key}={row[key]:.3f}" for key in sorted(acc.sums))
                print(f"  [{nm[-40:]}] k={k} n={acc.n} | {stats}")
                print(f"      {label}: " + ", ".join(f"{t} {v:.4f}" for t, v in row[label][:8]))
            if nm != "teacher":
                tot = sum(a.sums["kl_to_teacher"] for a in accs[nm].values())
                share = {str(k): a.sums["kl_to_teacher"] / tot if tot else math.nan for k, a in accs[nm].items()}
                out_m["kl_share_by_position"] = share
                print(f"      KL share by position: " + ", ".join(f"k={k} {v:.2f}" for k, v in share.items()))
            out_ds["models"][nm] = out_m
        report["datasets"][ds] = out_ds

    out = args.out if os.path.isabs(args.out) else os.path.join(DIR_ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

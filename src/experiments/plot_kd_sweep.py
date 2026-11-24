"""Tabulate the baseline sweeps that fixed the paper's protocol and baseline weights.

``tables/kd_sweep.tex``: every standard-KD configuration in
``results/standard_kd/22_add-sweep.json`` (learning rate x BOS convention x KD
position set), one row each: settings, the fraction of parameters that moved on
step one, peak in-distribution accuracy with its step, 21_mult and mean OOD at
that step, and the last in-distribution eval. ``tables/rep_sweep.tex``: the
TinyBERT and CKA weight sweeps (learning rate x representation weight). The
rows the paper's arms use are shaded.

Usage (from the repository root):
    PYTHONPATH=src python -m experiments.plot_kd_sweep
"""

from __future__ import annotations

import argparse
from typing import List

from experiments._paper_plots import (
    OOD_MEAN,
    BERT_SWEEP, CKA_SWEEP, FAM_TEX, KD_PROTOCOL, KD_SWEEP, Run, all_runs, hdr, pm, write_table,
)


# The trainer's flag values, named by what they are in the paper: whether the
# training sequence carries the BOS token (evaluation and attribution always do).
BOS_NAME = {"legacy": "no (BOS-free)", "on": "yes (standard)", "off": "no, and none at evaluation"}


def lr_tex(lr: float) -> str:
    m, e = f"{lr:.0e}".split("e")
    return f"${int(m)} \\times 10^{{{int(e)}}}$" if int(m) != 1 else f"$10^{{{int(e)}}}$"


def kd_table() -> None:
    runs = all_runs(KD_SWEEP)
    runs.sort(key=lambda r: (r.config.get("learning_rate"), r.config.get("bos_mode"), r.config.get("kl_tokens")))
    ref = runs[0]
    cols = ["lr", "BOS in training", "KD positions", "seeds", FAM_TEX["22_add"] + " peak (step)", FAM_TEX["21_mult"], OOD_MEAN, FAM_TEX["22_add"] + " last"]
    lines = ["\\begin{tabular}{lllccccc}", "  \\toprule", "  " + " & ".join(hdr(c) for c in cols) + " \\\\", "  \\midrule"]
    for r in runs:
        c = r.config
        s = r.summary()
        marked = all(c.get(k) == v for k, v in KD_PROTOCOL.items())
        row = [lr_tex(c["learning_rate"]), BOS_NAME[c.get("bos_mode")], c.get("kl_tokens"), str(r.n),
               pm(s["accuracy"]["sel"]) + f" ({'/'.join(map(str, s['sel_steps']))})",
               pm(s["21_mult"]["sel"], below=ref.baseline_of("21_mult")),
               pm(s["ood"]["sel"], below=ref.baseline_of("ood")),
               pm(s["accuracy"]["last"])]
        line = "  " + " & ".join(row) + " \\\\"
        if marked:
            line = "  \\rowcolor{gray!12} " + line.strip()
        lines.append(line)
    lines += ["  \\bottomrule", "\\end{tabular}"]
    write_table("kd_sweep", lines, "standard-KD protocol sweep on 22_add; each seed at its own best in-distribution step; shaded row is the paper's protocol")
    print(f"kd_sweep: {len(runs)} configurations")


def rep_table() -> None:
    lines: List[str] = []
    cols = ["recipe", "lr", "weight", "seeds", FAM_TEX["22_add"] + " peak (step)", FAM_TEX["21_mult"], FAM_TEX["222_add"], OOD_MEAN, FAM_TEX["22_add"] + " last"]
    lines = ["\\begin{tabular}{lllcccccc}", "  \\toprule", "  " + " & ".join(hdr(c) for c in cols) + " \\\\", "  \\midrule"]
    ref = None
    for name, path in (("TinyBERT", BERT_SWEEP), ("CKA", CKA_SWEEP)):
        runs = all_runs(path)
        runs.sort(key=lambda r: (r.config.get("learning_rate"), r.config.get("lambda_rep")))
        ref = ref or runs[0]
        for r in runs:
            c = r.config
            s = r.summary()
            marked = c.get("learning_rate") == 1e-5 and c.get("lambda_rep") == 0.3
            row = [name, lr_tex(c["learning_rate"]), f"{c['lambda_rep']:g}", str(r.n),
                   pm(s["accuracy"]["sel"]) + f" ({'/'.join(map(str, s['sel_steps']))})",
                   pm(s["21_mult"]["sel"], below=ref.baseline_of("21_mult")),
                   pm(s["222_add"]["sel"], below=ref.baseline_of("222_add")),
                   pm(s["ood"]["sel"], below=ref.baseline_of("ood")),
                   pm(s["accuracy"]["last"])]
            line = "  " + " & ".join(row) + " \\\\"
            if marked:
                line = "  \\rowcolor{gray!12} " + line.strip()
            lines.append(line)
        lines.append("  \\midrule")
    lines = lines[:-1] + ["  \\bottomrule", "\\end{tabular}"]
    write_table("rep_sweep", lines, "TinyBERT and CKA weight sweeps, one seed each; shaded rows are the weights used in the main table")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args()
    kd_table()
    rep_table()


if __name__ == "__main__":
    main()

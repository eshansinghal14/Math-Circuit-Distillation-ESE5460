# Paper audit: what to fix, where to add, and how ICLR would grade it

Written 2026-09-26 against `Neuron_Level_Circuit_Distillation_for_Arithmetic.pdf` (15 pages, the version with λ = 3 as the main arm). References are **page:line** from the PDF's margin line numbers, e.g. `p6:317`.

**Companion files** (all in `~/Downloads`, and on branch `analysis/eval-format-audit`):

| File | What it is |
|---|---|
| `graph_distillation_companion.md` | Content source: every number with its data file, section outlines, the claim → citation map. |
| `graph_distillation_references.bib` | Merged, verified bibliography. Paste it over `references.bib`. |
| `graph_distillation_audit_evidence.html` | The evidence page with every table; opens offline. |
| `graph_distillation_figures/` | The three new figures and two table-row files. |

---

## 0. The plan for the flight, in order

1. **Decide the framing (5 min).** Recommendation: the audit paper (Option 2). The current framing makes claims our own data refutes (§1 below), and a reviewer who checks commit rates will find that in an afternoon.
2. **Retitle, and rewrite the abstract and intro** (§3.1–3.2 below; the companion's §3 has sentence-level outlines).
3. **Apply the line-level fixes** in §2 below. Most are deletions or one-sentence swaps.
4. **Add the new Section 5** from the companion's §5. Figures and tables are ready; LaTeX snippets are in §4 below.
5. **Cut to fit 9 pages** using §4.3 below.
6. **Swap the bibliography** and fix the one miscitation (§2, `p2:095`).
7. **Last pass:** search for `pending`, "above the teacher", "routing transfers" and "no mechanism".

---

## 1. Realistic ICLR grading

These are the ICLR 2025 form fields. The 2027 form may differ in scale, but the categories are stable:
- Soundness, Presentation, Contribution: 1 poor, 2 fair, 3 good, 4 excellent;
- Rating: 1 / 3 / 5 / 6 / 8 / 10, where 5 is marginally below and 6 marginally above the acceptance threshold;
- Confidence: 1–5.

Recent ICLR acceptance is about 30–32%. Accepted papers typically average about 6 or above, with no strong reject.

### A. The current draft, submitted as-is

| | Score | Why |
|---|---|---|
| Soundness | **2** | The main arm has one seed at λ = 3. The scrambled control, which is the paper's own test of the claim, is "pending". The protocol is unconventional (BOS-free training) and was *selected on held-out 21_mult accuracy* (`p11:540`), yet the effect vanishes under the standard convention with "no mechanism" offered (`p7:371–373`). There is one task and one teacher–student pair. |
| Presentation | **2** | The draft has visible `[pending]` markers throughout, a broken sentence (`p7:345–346`), and long dense sentences. Section 4.2 is titled "Routing transfer" but no routing is measured. |
| Contribution | **2** | The idea is novel and the functional supernodes are nice. But the closest prior work (Circuit Distillation, Causal Distillation) is missing, the related-work claim that "none constrains the causal relations among internal components" (`p2:085`) is false, and the scope is narrow. |
| Rating | **3** (range 1–5) | A careful reviewer asks "why is the student above the teacher?" and "why does BOS matter?", and the answer is our audit. If a reviewer finds it, the paper reads as an artefact. |
| Accept chance | < 10% | |

### B. Reframed as an audit, with the evidence we already have (no new runs)

| | Score | Why |
|---|---|---|
| Soundness | **3** | Every analysis claim is multi-seed or controlled: seed-matched students, 3-seed commit control, localisation replicated over 2–3 seeds. Remaining gaps: one task, one model pair, the audit at λ = 1 not λ = 3, and the scrambled control is 1 seed. |
| Presentation | **3** | Achievable if the rewrite follows the companion's structure and cuts the method pages. The figures are clear. |
| Contribution | **2–3** | The reviewer's likely question: "a negative result on a method nobody uses; format sensitivity is known." Our answer: it is the first audit of a *mechanism-matching* distillation objective, it gives a mechanism for the gain (layers 0–7, `=` position, the teacher's hedge), and it is a reusable protocol relevant to recent work (Circuit Distillation, Causal Distillation). |
| Rating | **5** median (range 3–6) | |
| Accept chance | about 15–25% | |

### C. B plus one or two generality experiments (§5 below; 2–4 GPU hours)

- Rating **6** median (range 5–8), accept chance **about 35–50%**.
- The single experiment that moves reviewers most is showing that the lesson is not specific to our loss: the TinyBERT/CKA term computed on BOS-prefixed inputs should also produce the "gain". That turns "our method had a bug-like confound" into "any auxiliary loss computed in the eval format does this, and here is how to detect it".

### Likely reviewer comments on B, and your answers

| Comment | Answer and where it lives |
|---|---|
| "Single task and single model pair." | Say it; the audit is a case study by design. Soften it with C's experiments. Limitations section. |
| "Isn't this just a bug in your training pipeline?" | It is the protocol the paper selected, on purpose, as the best KD baseline (0.512 vs 0.390). Both conventions are standard in practice (many SFT stacks omit BOS). The point is that the auxiliary loss's gain is invisible to standard evaluation. §4.1 + §5.5. |
| "Why should I care about a negative result?" | Mechanism-matching distillation is an active direction (Wu et al. 2022; Wadhwa et al. 2025). We give a cheap audit protocol and show the failure mode is silent: it improves every metric reported. Intro ¶4–5 + Discussion. |
| "Format sensitivity is known." | Known for prompts at inference (Sclar et al. 2024). New here: a *training* objective's apparent generalisation comes from the format it is computed in, and it is localised mechanistically. Related work ¶4. |
| "λ = 1 vs λ = 3." | Table 4: 0.950 vs 0.951; the conclusions rest on seed-matched comparisons. §5 opening sentence. |
| "Why do supernode size and λ matter if it's only format?" | The switch has a direction and a strength set by the target. See the companion's reviewer table. |

---

## 2. Line-by-line fixes to the current PDF

Each row gives the location, the problem, and what to do.

### Title and abstract

| Where | Problem | Fix |
|---|---|---|
| `p1:000–004` title | Promises the transfer we no longer claim. | New title; options in companion §1. |
| `p1:024–028` | "lifts ... from 0.51 to 0.95 [pending: seeds 2–3], above the teacher itself" | Remove "above the teacher". Replace [pending] with "three seeds". Then add the audit sentences (companion §3.1, points 4–5). |
| `p1:027–028` | "where matching hidden states does not move the baseline" | Remove it as evidence for the routing. The representation baselines never see the eval format (§5.4). |
| `p1:028–030` | "it vanishes under the standard tokenisation convention" (unexplained) | "...which we trace to the teacher's own hedge (§5.5)". |

### Introduction (Section 1)

| Where | Problem | Fix |
|---|---|---|
| `p1:039–042` | "one has the algorithm and the other a lookup" | Keep it, but mark it as *the hypothesis we test* ("If graph distillation transferred the algorithm, we would expect..."). |
| `p1:047–048` | "what that buys" | Add: "...and then audit what it actually changes." |
| `p1:049–p2:076` contributions | (ii) claims OOD transfer; the scrambled control is "in progress" (`p2:073–075`); (iii) calls the BOS result a boundary. | Rewrite as the three contributions in companion §3.2 ¶5. Delete "in progress". |
| after `p2:076` | Nothing tells the reader the twist is coming. | Add companion ¶4 ("We then asked where the gain lives") before the contributions. |

### Related work (Section 2)

| Where | Problem | Fix |
|---|---|---|
| `p2:085` | **False claim**: "none constrains the causal relations among internal components". | Causal Distillation (\citep{wu2022causal}) and Circuit Distillation (\citep{wadhwa2025circuit}) do. Add companion ¶1b and differentiate. A reviewer who knows Wadhwa 2025 will flag this first. |
| `p2:088–095` | Fine. | Add the patching and mediation citations the audit uses (companion §4 ¶3). |
| `p2:095–096` | **Miscitation**: "Arithmetic ... studied as a circuit (Wadhwa et al., 2024)". That paper is about CoT distillation. | Replace with \citep{stolfo2023mechanistic,nikankin2025arithmetic,zhou2024pretrained,lindsey2025biology}. Mention that Nikankin et al. study Llama3-8B, our teacher. Move Wadhwa 2024 to the distillation paragraph. |
| *(new)* | No paragraph on what training changes or on eval-format effects. | Add companion §4 ¶2 and ¶4 (fine-tuning wrappers, stitching, attention sinks, format sensitivity, shortcut learning). |

### Method (Section 3)

| Where | Problem | Fix |
|---|---|---|
| `p3:125–128` | "which is how a term a tenth the size of the KD gradient changes what the student learns" is written as an explanation of the gain. | Keep the orthogonality as an observation; drop the causal claim, or point to §5 ("we show in §5 what it changes"). |
| `p3:146` | "zero the BOS position so the attention sink is never selected" | Add: "The attribution forward pass itself includes BOS, as evaluation does; §5 shows why this matters." It plants the key fact early. |
| `p3:160–161` | "where it can be checked against a known circuit" | Overstated: LLM addition is a bag of heuristics (Nikankin et al.). Say "where candidate roles are well studied". |
| `p4:169–177` category table | Lists the composite "arg1 units and arg2 units" category, but training uses six supernodes without it. | Say it is scored for reading only, or drop it from the table. |
| Figure 1 (`p2:054–068`) | Doesn't show where BOS enters. | Optional but strong: label the graph path "BOS + prompt" and the KD path "prompt (no BOS)". It makes the audit's core fact visible in the first figure. The text is also very small; enlarge the resizebox. |
| Figure 2 (`p5:216–241`) | A low-resolution PNG with small edge labels. | Re-export at 300 dpi if possible. Keep it: it is the best visual in the paper. |

### Experiments (Section 4)

| Where | Problem | Fix |
|---|---|---|
| `p5:261` | "a fixed 2000-prompt subset of each held-out test set" | 21_mult has **1000**. Also fix Table 9 (`p15:768`). |
| `p5:265–269` | Scrambled control "running [pending: three seeds]"; graph arm "one [pending]". | Scrambled: report the λ = 1 seed-1 result (§5.4). Graph: add "at λ = 1, three seeds give 0.948 / 0.955 / 0.951 on 21_mult (best step)". |
| `p6:308–313` BOS-free paragraph | Hides the key fact. | Add: "Under this protocol the graph term is the only loss computed on BOS-prefixed inputs, the format every accuracy is scored in (§5)." |
| `p11:540` and `p6:309` | **Protocol selected on held-out accuracy** ("0.512 against 0.390 on 21_mult ... which is why BOS-free training was selected"). Checkpoint selection is ID-only, but protocol selection used OOD. | State it openly in §4.1. In the audit framing it helps: we chose the protocol under which the auxiliary term looked best, and then explain why. |
| Table 1 (`p6:270–281`) | Caption says "[pending]" and "the scrambled control is running". | Remove both. Replace the scrambled column with the λ = 1 scrambled numbers, or drop the column and put controls in the §5 table. Add a footnote that the teacher's 0.894 on 21_mult is 0.986 among answered prompts. |
| Figure 3 (`p6:283–305`) | End labels overlap ("graph SFT KD TinyBERT CKA" stacked); "graph distillation (1 seed)". | Increase the label gap or use a legend only. Keep the figure; it is "the gain to be explained". |
| `p6:317–p7:341` | "above the teacher's own 0.894"; "Graph distillation is ahead of every baseline on every held-out family". | Keep the numbers; remove the interpretation. Add one sentence: "§5 shows what produces this gain." |
| `p7:342–346` | "not a generic regulariser ... specific to the routing target"; "the teacher's routing transfers better than the labels". | **Delete.** Both are refuted (§5.4; SFT's advantage is also commitment, 0.894 vs 0.466). |
| `p7:345–346` | **Broken sentence**: "Whether the graph arm's seed variance is as small as its baselines' is large awaits the remaining seeds". | Replace with the 3-seed λ = 1 numbers (sd 0.002–0.004). |
| `p7:348–352` Sensitivity | Fine as data. | Add: "λ and supernode size set how strongly the target pushes the answer-or-hedge decision (§5.4)." |
| Figure 4 (`p7:324–338`) | "scrambled arm: pending"; the per-step scrambled history is lost. | Replace with `audit_controls.pdf` (in §5), or keep the real arm's panels and quote the scrambled eval curve in text. |
| `p7:354–359` "Is the content of the target what acts?" | Pending. | Replace with the §5.4 results paragraph. |
| `p7:364–373` BOS paragraph | "we have no mechanism for it and offer none" | Replace with §5.5 (the teacher's P(?) = 0.255 on 22_add under BOS, copied by KD). |
| `p7:375–377` | "The held-out evaluation shows the routing transfers across operand count, digit width and operation" | **Delete.** It is refuted. Move "it is a case study" to Limitations. |

### Limitations, conclusion, statements (Sections 5–6)

| Where | Problem | Fix |
|---|---|---|
| `p8:380–387` Limitations | Pending markers; framed around the positive result. | Rewrite (companion §6). |
| `p8:391–398` Conclusion | "carries the teacher's competence to an operation it never trained on, above the teacher's own accuracy" | Rewrite (companion §6, four sentences). |
| `p8:412` Ethics | "transfers a teacher's internal structure" | Change to "adds a loss built from the teacher's attribution graphs". |
| `p8:417–423` Reproducibility | Doesn't mention the audit. | Add `format_decomp`, `graph_delta`, `pos_patch`, `plot_audit` and the `--bos-aux` control flag. |

### Appendices

| Where | Problem | Fix |
|---|---|---|
| App. B `p11:590–593` | λ = 3 chosen as "the middle of the flat range" of held-out results. | That is selection on OOD accuracy too; say so in one clause. |
| App. B `p11:566` | "[pending: CKA, weight 10]" | Drop the pending slot. Mention that CKA at weight 10 reached 0.743 in one seed, which fits the story: a stronger auxiliary term moves the answer decision. |
| App. B `p11:593` | "[pending: λ = 10 to 100 steps]" | Drop it; the canary is enough. |
| App. B Table 6 intro | "[pending: seeds 2–3 for the λ = 1 arms]" | Drop. |
| App. C Precision (`p13`) | bf16 master weights with 83% of updates rounded away; honest but alarming. | Keep, and shorten to 3 sentences. |
| App. D `p14:740` | "A100-80GB" | The audit ran on an A100-40GB; state both. |
| App. E Table 9 | "2000"; seeds pending; λ = 3. | Change 21_mult to 1000; seeds "1–3 (λ = 1)"; add the control arms. |
| App. F (`p15:783–808`) | Five rounds of pipeline bugs plus withdrawn results: an honest but long confession, and reviewers read it as fragility. | Cut to one short "Pipeline checks" paragraph: the tokenisation-consistency check, the config-digest run keys, and the rule about coinciding losses. Delete the "Open runs" list. |
| *(new)* App. G | — | "Audit methods": exact procedures for commit/correctness, layer transplant, residual patching and the controls, plus the per-family decomposition table (all five families; companion §5.1). |

---

## 3. Where to add what

### 3.1 Proposed section order (9 pages)

| # | Section | Pages | Source |
|---|---|---|---|
| 1 | Introduction | 1.0 | companion §3.2 |
| 2 | Related Work (4–5 paragraphs, incl. mechanism distillation) | 0.7 | companion §4 |
| 3 | Graph Distillation: keep 3.1–3.3; move 3.4 details and the category table to the appendix | 2.0 | current §3 |
| 4 | Experiments: 4.1 setup (with the BOS fact up front), 4.2 "The gain to be explained" (Table 1, Fig. 3, sensitivity) | 1.3 | current §4.1–4.2, trimmed |
| 5 | **Where the gain lives**: 5.1 commitment vs correctness; 5.2 the same model outside the eval format; 5.3 layers, then position; 5.4 controls; 5.5 why standard BOS removes it; 5.6 routing probes (short) | 2.8 | companion §5 |
| 6 | Discussion: lessons for mechanism-matching objectives, and an audit checklist box | 0.4 | companion §5.7 |
| 7 | Limitations + Conclusion | 0.4 | companion §6 |

**The audit checklist box** is highly recommended: reviewers like reusable takeaways. Put it in §6 as a framed box:
1. List the input format each loss is computed on, and evaluate in every one of them.
2. Split accuracy into commitment and correctness-given-commitment.
3. Compare seed-matched runs and transplant layers to localise any difference.
4. Patch residuals by position to find where the difference is computed.
5. Run a content-free control computed in the same place as the auxiliary loss.

### 3.2 Figures and tables, and where they go

| Asset (in `figures/` / `tables/`) | Put it | Size |
|---|---|---|
| `audit_decomp.pdf` | Top of §5.1 | full width |
| `tables/audit_decomp.tex` | §5.1, or merged into Fig. `audit_decomp` as the main table | `\scriptsize` |
| `audit_localise.pdf` | §5.3 | full width |
| `audit_controls.pdf` + `tables/audit_controls.tex` | §5.4 (the figure) and the appendix (the table), or the reverse | full width |
| Routing/probe table (companion §5.6) | Appendix G | — |
| Per-family decomposition (companion §5.1) | Appendix G | — |

---

## 4. LaTeX you can paste

### 4.1 Figures

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/audit_decomp.pdf}
  \caption{Where the 21\_mult gain comes from. \emph{Left:} each arm's greedy outputs split into
  right, answered-but-wrong, and no number (the hedge ``?\textbackslash nTo find the answer'').
  KD students, three seeds, answer only 47\% of prompts but are right on 95\% of those, as the
  graph students are. \emph{Right:} accuracy without BOS at evaluation (grey), the format the KD
  term trains on, against with BOS (blue). Seed-matched KD and graph students agree within 0.005
  without BOS on every family.}
  \label{fig:audit-decomp}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/audit_localise.pdf}
  \caption{Localising the change. \emph{Left:} copying the graph student's layers $0..k{-}1$
  into its seed-matched KD student; layers 0--7 reproduce the full effect in all three seeds,
  layers 8--15 (crosses) almost none. \emph{Right:} patching the graph student's residual
  stream into KD after layer $L$ at one position; only the final prompt token (\texttt{=})
  carries the decision.}
  \label{fig:audit-localise}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/audit_controls.pdf}
  \caption{Controls, BOS at evaluation, step 100. A loss that only rewards committing to some
  number token, computed on the BOS-prefixed batch (``commit'', weight 0.01, three seeds),
  matches graph distillation on every family; a scrambled graph target of matched gradient share
  drives the student to hedge.}
  \label{fig:audit-controls}
\end{figure}
```

### 4.2 Decomposition table

```latex
\begin{table}[t]
  \centering\scriptsize\setlength{\tabcolsep}{3pt}
  \caption{21\_mult split into commitment and correctness, BOS at evaluation. Students are
  three-seed means ($\pm$ sd where above 0.01); controls one seed. \emph{Forced}: the most likely
  number token is the answer.}
  \label{tab:audit-decomp}
  \begin{tabular}{lcccccc}
    \toprule
    arm & acc & commit & acc$\mid$commit & forced & $P(\texttt{?})$ & acc, no BOS \\
    \midrule
    \tablerows{audit_decomp}
    \bottomrule
  \end{tabular}
\end{table}
```

### 4.3 Where to find the space

| Cut | Saves |
|---|---|
| Move §3.4 (supernode edge weights, eq. 5 and its paragraph) to the appendix | about 0.4 p |
| Move the category table (`p4:169–177`) to the appendix; describe it in one sentence | about 0.3 p |
| Shorten §3.2's second paragraph (`p3:147–152`) to two sentences | about 0.15 p |
| Cut §4.2's interpretive prose (`p7:341–346`) | about 0.2 p |
| Replace Figure 4 with `audit_controls.pdf` | about 0 p (a swap) |
| Shrink Figure 3 to a 2 × 3 grid at 0.9 width, or drop the 2222_add panel | about 0.2 p |

---

## 5. If you get GPU time: experiments ranked by reviewer impact per hour

| # | Experiment | Why | Cost |
|---|---|---|---|
| 1 | **TinyBERT/CKA computed on the BOS-prefixed batch** (KD otherwise unchanged) | Shows *any* auxiliary loss in the eval format reproduces the gain. This generalises the lesson and moves Contribution from 2 to 3. Needs `representation.py` to prepend BOS in its forward, about 15 lines, mirroring `--bos-aux`. | about 30 min |
| 2 | **Second model pair, KD vs KD + commit only** (e.g. Qwen2.5-7B → Qwen2.5-0.5B, same legacy protocol) | Shows the hedge/format confound isn't Llama-specific. No graph pipeline is needed. | about 1 h |
| 3 | **Scrambled seeds 2–3** | Completes the one single-seed control that carries weight. | about 2 h |
| 4 | **format_decomp on one n = 3 (3 neurons per label) checkpoint** | Answers "why does supernode size matter" (prediction: the shortfall is hedges). | about 40 min (retrain 50 steps) |
| 5 | **λ = 3 checkpoint through format_decomp + graph_delta** | Removes the λ = 1 vs λ = 3 caveat. | about 1.5 h |

Commands for 1–5 follow the ones in companion §0 and the `--bos-aux` flag. Every result lands in `results/overnight/` and is picked up by `plot_audit`.

---

## 6. Final-pass checklist

- [ ] No `pending` left: search the `.tex` for `\pending`.
- [ ] No "above the teacher", "routing transfers", "no mechanism", "in progress".
- [ ] 21_mult is "1000 prompts" everywhere (§4.1, Table 9).
- [ ] The Wadhwa 2024 citation moved; Wadhwa 2025 and Wu 2022 discussed in related work.
- [ ] `p2:085` "none constrains the causal relations" removed.
- [ ] The BOS fact stated in §3.2 or §4.1 *before* §5.
- [ ] Table 1 caption and Figure 4 no longer mention a running scramble.
- [ ] Bibliography replaced with the merged file; compile shows no `?`.
- [ ] Reproducibility statement lists the audit scripts.
- [ ] Main text ≤ 9 pages. The AI use, ethics and reproducibility statements don't count.
- [ ] Anonymity: no GitHub URL, no names in the PDF metadata. The filename `Neuron_Level_Circuit_Distillation_for_Arithmetic.pdf` doesn't match the title; rename it before upload.

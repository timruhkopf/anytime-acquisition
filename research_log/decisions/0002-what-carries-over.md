# 0002: What this repository carries over from AnytimeAcquisition and pfn-acqpolicy, and what it doesn't

**Context.** Two earlier repositories explored the idea.
- AnytimeAcquisition: a PFN, its BNN prior, an amortized-EI "action expert" (E1) on the smoke-scale PFN,
  and an RL stage.
- pfn-acqpolicy: the Phase-0 branching kill test, its review response (nulls, replicates, an advantage
  study), `pfn_large`, and the alpha-PFN.

Both grew to dozens of pipelines and ~50 experiment configs. Their logs contain careful, sometimes
self-correcting analysis; their code mixes experiments with infrastructure. This repository starts over
from research-template and reuses the *results and lessons*. It takes code only where it is a clean,
tested library. The models now live in pfn-repro (a pinned dependency).

**Carried over (and why).**

| What | From | Why |
|---|---|---|
| PFN, BNN prior, bar distribution, alpha-PFN, checkpoints | pfn-repro v0.1.0 (ported, reproduced bit-exactly) | the frozen surrogate is not a research variable here |
| Reporting standards S1–S9 (selection vs evaluation, tuned baselines, equal-arms null, power, scale-free paired effects, shared f\*, train/test separation, pre-registration, candidate generator) | pfn-acqpolicy log 2026-10-20 §22 | they were derived from mistakes made in this exact setting |
| The **non-anticipating rollout controller** (privileged and belief-only) | pfn-acqpolicy log 2026-10-15 §3 (proposed, never run) | it measures the value of state-dependent switching without the hindsight selection bias that made the branching oracle 64–90% luck |
| Budget scaled with d (2d initial points), context within pfn_large's training range (≤ 100) | pfn-acqpolicy design review, weakness 1 | conclusions at d = 6 on 4 + 16 evaluations rested on a regime with little optimization |
| fp32, CPU-drawn functions, a shared f\* per task | pfn-acqpolicy log §11 (review points 1, 1b, 10) | TF32 and device-side RNG made tasks irreproducible across devices |
| The VLA-style readout (π0 joint attention over the frozen PFN's per-layer context K/V) | AnytimeAcquisition log 2026-10-09 §4–5 | its key test (A4: does reading the PFN state beat learning from D?) failed on the smoke PFN and was never asked of pfn_large |

**Not carried over (and why).**

| What | Why not |
|---|---|
| The K^S branching tree as the main instrument | Its hindsight best-of-81 oracle is mostly luck (the null tree reproduces 81–95%). With switching at the segment boundaries, every policy is one of the leaves, so the tree only bounds; it doesn't measure |
| A fixed per-instance candidate pool with one incumbent-perturbation scale | Late in the old runs, 60–99% of picks came from the shared perturbations (σ = 0.05). Arms could only differ in which shared candidate they picked (S9). Here: a fresh Sobol set per (task, step) and perturbations at three scales |
| The smoke PFN for any conclusion | trained on n ≤ 20; miscalibrated at small n (notebook 03 in pfn-repro) |
| RL (PPO) for a controller | premature: no switching gain has yet been shown that a policy could learn. The rollout controller is a policy-improvement step that needs no training, and the right first measurement |
| The 3-arm portfolio {PI, EI, UCB(0.95)} | three myopic scores from the same marginal, two of them tied. Here: tuned EI and PI, UCB(0.99), MES, and the alpha-PFN's amortized ES (structurally different, information-based) |
| Per-step diagnostic sprawl in the BO step | kept to what a question needs |

**Consequences.** The headline experiment answers one question: how much could a switching policy gain,
with and without privileged information, against a tuned static baseline and an equal-arms null? Every
other experiment must state which question it serves.

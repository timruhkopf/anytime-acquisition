---
experiment: api_decoder
sweeps: []
---
# Amortized policy improvement, round 2: the 0006 value decoder (Phase A)

## Question

See the description at the top of `notebooks/09_2026-11-11_api_decoder.ipynb`; not retyped here. Issue #13.

## §1 Sanity (2026-11-11)

**What ran.** The pinned 4000 states plus 200 new label tasks (4200 states); 10 epochs; calibration on 30 tasks, test on 30 (`results/api_decoder/sanity_20261111-163331`, code 1c54997).

**Result** (a pipeline check, not an estimate):
- The pipeline ran end to end.
- Quantile bar borders collapsed to 25 bins: most single-seed advantages are within ±0.002.
- The validation within-state Spearman was −0.031.
- The quantile rules never deviated (test norm regret 1.0186 for every q rule and EI).
- The "mean" rule deviated in 99% of steps (calib 8.27 vs EI 0.545; test 9.17 vs 1.02): heavy tails make the predicted mean positive.
- Labeling took 459 s for 200 tasks on a shared GPU.

**Revision before the pilot (4274e77, notebook 09).**
- Signed-log borders: a zero bin of ±0.001 and 31 log-spaced bins per side, 63 in all.
- On the round-1 labels, 59% of the non-base single-seed targets lie in the zero bin.

**Next.** The pilot (8000 new label tasks; 12 000 states), launched when the probe pilot has finished (GPU memory).

## §2 Pilot (2026-11-11)

**What ran.**
- 12 000 labeled states: the pinned 4000 plus 8000 new label tasks.
- Calibration on 300 split-0 tasks; test on 1000 split-1 tasks, d = 2 (`results/api_decoder/pilot_20261111-165340`, code 4274e77).
- Labeling took 286 min on a shared GPU; training 13 epochs.
- Analysis: notebook 09.

**Result.**
- rule\* = q0.5, calibrated on split 0. It deviates at 0.2% of test steps.
- Δ norm_regret vs EI is −0.0013 [−0.014, 0.012], relative −0.1% [−1.1%, +0.9%].
- Validation within-state Spearman:
  - q0.5 0.060 [0.039, 0.081];
  - for comparison, untrained log EI scores 0.063 (api_probe §3), and the label reliability is 0.57.
- Validation NLL: decoder −3.519 vs null −3.509.
- The mean rule is catastrophic: calib 3.77 vs EI 1.06.
- Against round 1, on the same tasks with identical EI trajectories:
  - r2 − r1 = +0.009 [−0.012, 0.038];
  - r1 − EI = −0.010 [−0.033, 0.002].

**Interpretation.**
- The decoder learns no ranking beyond EI's own and correctly stays with EI.
- The evidence agrees across probes (api_probe §2–3), the PFN lookahead (api_gate §2) and this decoder: on noise-free P0, d = 2, T = 20, the clairvoyant headroom over EI's pick is not predictable from the frozen PFN's belief.
- One-step policy improvement over EI from this belief has nothing to amortize.

**Verdict: NO-GO** (pre-registered rule `api_verdict`; recorded with `decide`).

**Next.** Do not scale the decoder (Phase B, larger candidate sets) on this prior. Directions that change what the belief contains or what the labels measure:
- fine-tune the PFN on BO contexts (0004/0005, E1);
- a noisy prior;
- multi-step labels instead of one-step advantages over EI.

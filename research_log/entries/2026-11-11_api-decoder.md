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

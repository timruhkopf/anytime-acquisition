---
experiment: pfn_bo_inference
sweeps: []
---
# PFN inference on BO contexts: is the predictive calibrated where BO queries, and what does resolution do to the labels?

## Question

See the description at the top of `notebooks/10_2026-11-12_pfn_bo_inference.ipynb`; not retyped here.

## §1 Sanity (2026-11-12)

**What ran.** 200 states on ulysses (`results/pfn_bo_inference/sanity_20261112-083250`, code 8f9ba21). Collection took 3.9 s and the analysis 1.5 s.

**Result** (a pipeline check; the numbers come from a flawed design and are not interpreted):
- The pipeline ran end to end, and the tables have the expected shape.
- Two design errors:
  - The random design's local and EI-pick points were chosen from the BO data, so they are not ignorable for that context. It underpredicted improvement at those points (calibration-in-the-large −0.26 at EI picks), as the error predicts.
  - The PFN's target is noisy y (the prior's noise, about one bin), but the scores used noise-free f.

**Revision before the pilot** (notebook 10, "Revision after the sanity run"):
- each context gets its own local points and EI pick;
- primary scores against y from the task's noise model, with f-based scores kept as secondary;
- a check that the redrawn tasks reproduce the label set's observations.

**Next.** The pilot (4000 states, 200 bootstrap resamples).

## §2 Pilot (2026-11-12)

**What ran.**
- 4000 states (pinned api_round1 labels, split 4, d = 2) and 264 000 scored points (`results/pfn_bo_inference/pilot_20261112-084114`, code 3c475a7).
- Collection 59 s, analysis 34 s on ulysses (first run).
- The first pilot run, `pilot_20261112-083546` (code 03d8de2), produced a per-point table over the repo's 2 MB file limit. The per-point scores became a run artifact with aggregated tables for the notebook, and the pilot was repeated. Its calibration and strata tables reproduce the first run exactly (max abs diff 0). The first run's snapshot is not committed.
- Check: the redrawn tasks reproduce the labels' F (max |Δ| 1.2e-5). Analysis: notebook 10.

**Result, test 2 (calibration against y; BO context vs random design).**
- Random-design control: pooled cov90 0.900 [0.896, 0.904], cov50 0.544, every event ECE ≤ 0.008.
- BO contexts, pooled: cov90 0.881 [0.877, 0.885], cov50 0.523, ECE ≤ 0.013. Within tolerance.
- BO contexts, EI's picks:
  - cov50 0.537 [0.521, 0.551] (control 0.517); cov90 0.880 [0.869, 0.889];
  - improvement event (k = 0): calibration-in-the-large +0.036 [0.019, 0.049] (control +0.018); ECE difference vs control 0.019 [−0.001, 0.027].
- By step: cov90 at local points falls from 0.930 to 0.831 between t 0–4 and t 15–19 on BO contexts, against 0.927 to 0.879 for the control.

**Result, test 1 (by gap stratum in PFN bins).**
- 57% of states are in the < 1-bin stratum. All their non-base advantages are below one bin (91% overall), but they carry 2.1% of the headroom sum (mean 0.025 % of the gap).
- The 10–100 and > 100-bin strata (25% of states) carry 88% of it: means 2.06 [1.62, 2.62] and 3.55 [2.35, 5.04].
- Within-state Spearman of log EI with the label:
  - 0.073 in < 1 bin, 0.068 in 1–10;
  - 0.042 [0.018, 0.062] in 10–100, 0.054 [0.007, 0.097] in > 100.
  - PI: ≤ 0.02 in the ≥ 10-bin strata.
- Label reliability falls from 0.64 (< 1 bin) to 0.41 (> 100).
- Decoder q0.5 by stratum (328 joined validation states): 0.157 [0.074, 0.238] at 1–10 bins, 0.04–0.08 elsewhere.

**Interpretation.**
- E2 is supported by the pre-registered rule, narrowly. The trigger is cov50 at EI's picks; the BO-specific miscalibration is a few points of coverage, concentrated late in the run near the incumbent, plus mild optimism about improvement at EI's picks. Pooled over all points, the PFN is calibrated on BO data within tolerance.
- E3 is not supported: the ranking is no better where the gap is resolvable.
- E4 does not explain the headroom: the sub-bin labels carry almost none of it.
- What remains is E1 against the size of this E2, at the large-gap states where the headroom lives. That is 0007 test 3.

**Verdict: GO** (E2 supported; pre-registered rule `inference_verdict`; recorded with `decide`).

**Next.** 0007 test 3: a variance-reduced PFN lookahead at the large-gap states (≥ 10 bins), with its calibration slope against the labels.

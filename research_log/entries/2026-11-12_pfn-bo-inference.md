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

---
experiment: option_policy
sweeps: []
---
# Can a policy that reads only the context choose options that beat EI? One-deviation option policy from exact labels

## Question

See the description at the top of `notebooks/13_2026-11-12_option_policy.ipynb`; not retyped here.

## §1 Sanity (2026-11-12)

**What ran.** 100 tasks on ulysses (`results/option_policy/sanity_20261112-123006`, code c0921cd). Collection took 6.1 min, the evaluation 7 s.

**Result** (a pipeline check; 100 tasks are far too few for the decision):
- The pipeline ran end to end: 8 decision steps × 8 options × 2 seeds per task, no missing features, 11.8% of regrets at the ε floor.
- Mean advantage per option (seed 0): UCB and MES are positive at every early step, as options for 3 steps and for the rest; PI and mean "for the rest" are clearly negative (−0.45, −0.81 at step 0).
- Privileged one-deviation headroom (cross-fitted over seeds): 0.38 [0.24, 0.55] without MES, 0.37 with.
- Held-out fit of Q: Spearman 0.05–0.08 for the context feature sets, 0.00 for `time`; the labels' cross-seed correlation is 0.57.
- **Labels are exact only up to GPU floating-point effects.** Under seed 0 the base lane from every step should be the prefix itself. Its raw regret differs across steps by at most 7e-6 (ECDF units), but that is more than 0.01 log10 in 27% of tasks (1% above 0.3), because near-ties late in a run flip with the batch composition. It is mean-zero noise at the scale of ε, and also applied to notebook 12.

**Revision before the pilot:** none to the design or the rule. The advantage keeps using the base lane of the same step and batch. Notebook 13 reports the base lanes' spread across steps as the numerical noise scale.

**Next.** The pilot as pre-registered (2000 tasks).

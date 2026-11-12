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

## §2 Pilot (2026-11-12)

**What ran.** 2000 split-4 tasks on ulysses, 12:37–14:39 (`results/option_policy/pilot_20261112-123724`, code c36d0c3; 1 member, DONE). Collection took 121 min, the evaluation 0.6 min.

**Result** (log10 terminal regret, ε = 1e-6, positive = better; held-out tasks, seed 0, 95% bootstrap CI; numbers from the snapshot's `summary` table unless noted):
- Primary policy `nomes/belief/mlp` (no MES among the options) over EI: **+0.135 [0.096, 0.174]**; executed on the other candidate set (seed 1): +0.128 [0.089, 0.166].
- Best context-free schedule (UCB(0.995) for the rest of the run, from t = 2 or 4) over EI: +0.119 [0.074, 0.165].
- Policy over that schedule: **+0.017 [−0.018, 0.050]**; seed 1: +0.026 [−0.022, 0.069].
- Policy over its shuffled null: +0.043 [0.020, 0.066]; with data features only: +0.013 [−0.010, 0.037].
- Other variants over static: `nomes/pfn/mlp` +0.009 [−0.028, 0.044]; `nomes/belief/ridge` +0.016 [−0.021, 0.050]; with MES, `all/belief/mlp` +0.027 [−0.006, 0.060] and `all/belief/ridge` +0.042 [0.009, 0.073] (seed 1: +0.021 [−0.025, 0.066]).
- Privileged one-deviation headroom: 0.346 [0.308, 0.386] without MES, 0.365 [0.328, 0.404] with.
- Held-out Spearman of Q (`fit` table): 0.15 on average for `belief`; 0.14 for "UCB for the rest", whose label correlates 0.63 across seeds.
- Numerical noise (run artifact `labels.pt`): the seed-0 base lane differs across decision steps by 0.014 log10 on average, by more than 0.1 in 3.0% of tasks.

**Post hoc** (scratch on the run artifact `labels.pt`; not pre-registered): with EI's terminal log regret added as one privileged feature, the same pipeline gains +0.147 [0.109, 0.183] over the static schedule (MLP; ridge +0.137 [0.100, 0.172]) and deviates in 37% of tasks. With the current true regret instead: +0.010 [−0.027, 0.044]. By tercile of EI's terminal regret the static UCB schedule gives −0.18, −0.11, +0.68. EI's terminal regret is half of the label, so part of this is mechanical.

**Interpretation.**
- A deployable policy beats EI by about 27% less regret, but nearly all of that is the static part: "switch to UCB early".
- The belief features are used (shuffled null), worth about 0.04 log10, but that does not exceed one good schedule.
- The learner and the one-deviation class are not what limits: given whether EI's run will end badly, they find +0.15 over static. What the features lack is that prediction. Deviating hurts on the two thirds of tasks where EI ends well.
- The one-deviation class is also narrow: 3-step options are each worth about +0.10 at every step from t = 4, and one deviation takes only one.

**Verdict: INCONCLUSIVE** (pre-registered rule on `nomes/belief/mlp`, seed 0; `just decide` not yet run).

**Next** (to discuss with Tim).
1. Predict EI's failure from context: a target with far less luck than the option advantage (EI's terminal log regret correlates 0.93 across seeds in notebook 12's pilot). Probe it from the belief features and the PFN state at each step; gate the UCB schedule with it.
2. Repeated decisions (policy iteration round 2 on the improved policy), since 3-step options stay valuable at every step.
3. The MES-base variant only after 1: it would repeat this result one level up.

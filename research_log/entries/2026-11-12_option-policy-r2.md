---
experiment: option_policy_r2
sweeps: []
---
# Round 2 of policy iteration: one context-dependent deviation from the UCB schedule

## Question

See the description at the top of `notebooks/14_2026-11-12_option_policy_r2.ipynb`; not retyped here.

## §1 Sanity (2026-11-12)

**What ran.** 100 tasks on ulysses, 15:50–15:57 (`results/option_policy_r2/sanity_20261112-155053`, code 639fae6). Collection took 6.1 min, the evaluation 7 s.

**Result** (a pipeline check; 100 tasks are far too few for the decision):
- The pipeline ran end to end with UCB as the base: 8 options (EI, PI, MES, mean; 3 steps or the rest), no missing values, the `full:*` columns give EI and UCB on the same tasks.
- Mean advantage over UCB by step (seed 0, `options` table): leaving UCB early costs (EI for the rest −0.30 at t = 0 and 2); from t = 8 on it is at or above zero (+0.10 at t = 12).
- Privileged one-deviation headroom from UCB: 0.21 [0.08, 0.35] without MES.

**Revision before the pilot:** none.

**Next.** The pilot as pre-registered (4000 tasks), after the ei_failure pilot.

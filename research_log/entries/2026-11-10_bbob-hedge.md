---
experiment: bbob_hedge
sweeps: []
---
# GP-Hedge on BBOB: how much of the switching headroom does an online portfolio recover?

## Question

See the description at the top of `notebooks/05_2026-11-10_bbob_hedge.ipynb`; not retyped here.
Baseline for the learned selector; GP-Hedge implementation in PR #12 (issue #11).

## §1 Setup (2026-11-10)

- Runs from the worktree `anytime-acquisition-hedge`, branch `exp/bbob-hedge` (stacked on `feat/11-gp-hedge`),
  on the local CPU. ulysses is busy with pfn-repro #3.
- Timing probe (scratch, CPU, 24 tasks, d = 5, η = 1): GP 99 s, PFN 373 s.

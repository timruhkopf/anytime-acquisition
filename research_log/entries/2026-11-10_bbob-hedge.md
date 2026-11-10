---
experiment: bbob_hedge
sweeps: [pilot_20261110-180520]
---
# GP-Hedge on BBOB: how much of the switching headroom does an online portfolio recover?

## Question

See the description at the top of `notebooks/05_2026-11-10_bbob_hedge.ipynb`; not retyped here.
Baseline for the learned selector; GP-Hedge implementation in PR #12 (issue #11).

## §1 Setup (2026-11-10)

- Runs from the worktree `anytime-acquisition-hedge`, branch `exp/bbob-hedge` (stacked on `feat/11-gp-hedge`),
  on the local CPU. ulysses is busy with pfn-repro #3.
- Timing probe (scratch, CPU, 24 tasks, d = 5, η = 1): GP 99 s, PFN 373 s.
- Sanity `sanity_20261110-180449` (2 functions, 1 instance, d = 2): DONE.
- Sweep `pilot_20261110-180520` (8 members, local CPU, detached, pid 250314), launched 18:05. It runs on the
  evaluation split 1, as pre-registered. It is labelled `pilot` because the tracking CLI allows `confirm` only after
  a pilot is decided GO, and there is no selection step here: nothing is chosen from these runs.

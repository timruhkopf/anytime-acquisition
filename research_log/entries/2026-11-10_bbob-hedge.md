---
experiment: bbob_hedge
sweeps: [pilot_20261110-180520, bbob_hedge_nomean/pilot_20261110-211257]
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

## §2 Primary result and a defect in the portfolio (2026-11-10 21:10)

All 8 members DONE (21:06, 3.0 h wall). Snapshot `results/bbob_hedge/pilot_20261110-180520`; notebook 05 executed.

**Check:** no member ends with uniform probabilities, so the rewards carry signal.

**Gain over B\*** (1 − geometric-mean regret ratio, 192 paired tasks, 95% CI):

| surrogate | d | η = 1 | η = 10 | privileged (ceiling) |
|---|---|---|---|---|
| gp | 2 | −0.03 [−0.30, +0.18] | +0.06 [−0.22, +0.27] | +0.59 |
| gp | 5 | −0.30 [−0.78, −0.01] | −0.08 [−0.46, +0.15] | +0.19 |
| pfn | 2 | −0.35 [−1.07, +0.08] | −2.10 [−4.33, −0.90] | +0.66 |
| pfn | 5 | −0.38 [−0.64, −0.17] | −3.06 [−5.45, −1.72] | +0.29 |

Against `ei`, hedge is better for the GP (+0.42 to +0.55, all CIs above 0) and not for the PFN.

**What is off.** Hedge chooses the `mean` arm in 44–90% of steps, more at η = 10 and at d = 5.

- **Cause:** the reward −μ(nominee) is maximal at argmin μ, which is the `mean` arm's nominee by
  construction. Reconstructed from the cumulative gains:
  - `mean` earns the step's top reward in 46–85% of steps;
  - its reward is on average 0.002–0.02 below the step's best, against 0.06–0.37 for every other arm.
- So Hedge, as configured, measures "who proposes the lowest posterior mean", and pure exploitation wins it.
  This explains why η = 10 is worst for the PFN: it locks into `mean` (83–90% of steps).
- Brochu et al.'s portfolio (EI, PI, UCB) has no such arm; including `mean` was my choice, copied from the
  bbob_switching arm set.

**Fix under test.** `experiment=bbob_hedge_nomean`, the same sweep with `mean` removed. It is registered in
the notebook description before launch, and its criteria are there. Sweep
`bbob_hedge_nomean/pilot_20261110-211257` (local CPU, pid 278663), launched 21:12.

**Housekeeping.** `tables/hedge.parquet` (3.6 MB; per-step probabilities and gains) exceeds the 2000 KB
pre-commit limit and is not committed. `docs/process.md` says to move `results/` to Git LFS past a few MB; left
to Tim.

---
experiment: acq_distill
sweeps: []
---
# Can an acquisition branch on the frozen PFN reproduce every arm's ranking? Per-arm distillation

## Question

See the description at the top of `notebooks/16_2026-11-13_acq_distill.ipynb`; not retyped here.

## §1 Sanity, d = 1 (2026-11-14)

**What ran.** d = 1, 12000 steps on 1000 tasks, `task.branch.post=2 y_in=true inc_token=true aligned=true`, 100 evaluation tasks, on ulysses 2026-11-14 11:33–12:23, alone on the GPU (`results/acq_distill/sanity_20261114-113335`, code 9c091ab; 1 member, DONE). The same configuration on 2026-11-13 (`runs/acq_distill/sanity_20261113-113139`) trained and then failed in the evaluation with a CUDA out-of-memory error on a GPU shared three ways. Earlier sanity runs (400 and 3000 steps, no additions) were not snapshotted.

**Result** (`summary` table; notebook 16):
- Spearman with the arm over all candidates: EI 0.94, mean 0.98, MES 0.91, PI(0.01) 0.89, UCB(0.995) 0.97; the same on candidate set 1 and on the dense grid.
- Good picks (within the arm's top 1%) on candidate set 0: EI 0.63, mean 0.63, MES 0.54, PI 0.51, UCB 0.55. Exact argmax: 0.19–0.35.
- Closed loop, branch minus arm in log10 terminal regret: EI +0.315 [0.164, 0.479], UCB +0.248 [0.095, 0.417], MES +0.187 [0.043, 0.337], mean +0.128 [−0.035, 0.288], PI +0.119 [−0.103, 0.335].
- Training (`train` table): top-1 loss 3.19 at step 10000, 2.98 at the end; still falling.
- d = 1 pictures: the learned curves follow every arm over the whole domain, no artifacts between training points; misses are near-ties between peaks.

**Interpretation.** The branch can read the arms' rankings off the frozen PFN's train-context activations. What is missing is the sharpness at the top, and the run was not converged. Which of longer training and the four additions made the difference to the 3000-step run is not separated.

**Verdict: INCONCLUSIVE** at d = 1 (rule: GO at 0.9 good picks for every arm, NO-GO below 0.5).

**Next (to decide with Tim).** Longer training with more weight on the top of the ranking, then d = 2 and d = 4; or stop here, given decision 0010's reading that distillation is a start and not a prerequisite.

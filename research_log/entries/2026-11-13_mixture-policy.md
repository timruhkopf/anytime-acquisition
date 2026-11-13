---
experiment: mixture_policy
sweeps: []
---
# An in-context mixture of acquisition functions: does a budget- and context-dependent mixture beat every arm?

## Question

See the description at the top of `notebooks/17_2026-11-13_mixture_policy.ipynb`; not retyped here.

## §1 Sanity (2026-11-13)

**What ran.** d = 2, 128 training / validation / test tasks, 3 generations per phase, on ulysses 11:35–11:52 (`results/mixture_policy/sanity_20261113-113528`, code 7f233a3; 1 member, DONE). The GPU was shared with two other runs. Search took 13.8 min, the test 3.0 min (run summary `time/*`).

**Result** (a pipeline check; 3 generations and 128 tasks decide nothing; numbers from the snapshot's `summary` and `train` tables):
- The pipeline ran end to end; all tables written. No candidate was accepted in either phase, so `budget` = `context` = the start (83% of the weight on MES, the best arm on validation), and `vs_static` is exactly 0.
- Arms on the test tasks, J at floor 1e-3 (seed 0): MES −2.660, EI −2.653, UCB(0.995) −2.633, PI(0.01) −2.612, mean −2.244. **The four real arms lie within 0.05 of each other in d = 2**; the bootstrap CI of a paired difference at 128 tasks is about ±0.04.
- The uniform mixture, never trained: +0.025 [−0.018, 0.068] over MES in J (seed 0), −0.006 [−0.059, 0.042] on seed 1.
- Timing: about 100 s per generation (12 perturbed policies × 128 tasks × 20 steps) with the GPU shared three ways.

**Revision before the pilot:** none to the design or the rule. Order of the members: d = 4 first. In d = 2 at this floor the arms are nearly tied, so there is little for a mixture to gain; d = 4 with a tight budget is the regime the question is about.

**Next.** The pilot as pre-registered, member `task.d=4 task.n_init=8` first, then `task.d=2`.

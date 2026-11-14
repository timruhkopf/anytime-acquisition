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

## §2 Pilot, d = 4 member (2026-11-13)

**What ran.** `task.d=4 task.n_init=8`, T = 20, 512 training / 512 validation / 1536 test tasks, 50 generations per phase, on ulysses 11:57–15:27 (`results/mixture_policy/pilot_20261113-115739`, code 0789f24; 1 member, DONE). Search 179 min, test 31 min, GPU shared with two other runs.

**Result** (differences of log10(regret + 1e-3), positive = better; paired over the 1536 test tasks, 95% bootstrap CI; `summary` table):
- Context-dependent over budget-only mixture, J: **+0.007 [−0.007, 0.021]** (seed 0); other candidate set: −0.002 [−0.018, 0.013].
- Budget-only mixture over the best arm (MES), J: **+0.030 [0.016, 0.045]**; other candidate set: +0.013 [−0.002, 0.027].
- Uniform mixture (never trained) over MES, J: +0.016 [0.001, 0.031]; other candidate set: +0.014 [−0.001, 0.029].
- Over EI, J: budget +0.042 [0.026, 0.060], context +0.049 [0.033, 0.067]; final regret only: +0.069 [0.049, 0.091] and +0.083 [0.062, 0.105].
- Levels (seed 0), area / final: EI −2.124 / −2.391, MES −2.099 / −2.441, UCB(0.995) −2.056 / −2.426, PI(0.01) −2.065 / −2.337, uniform −2.133 / −2.439, budget −2.139 / −2.460, context −2.140 / −2.474.
- Search (`train` table): validation J −2.256 at the start, −2.273 accepted after the budget phase, −2.299 after the context phase.

**Interpretation.**
- Mixing the arms gets both ends at once: EI's early speed and MES's final regret. That is the Pareto corner asked for, but the step beyond the best arm is 3–7% less regret.
- About half of it comes from mixing at all (uniform weights); the learned budget schedule adds the rest on the trained candidate set and nothing on the other one.
- Context adds nothing measurable. The two learned policies use very different weights and reach the same outcome: the objective is flat inside the hull.
- The paired search does produce a usable signal (validation and test both move), so the objective can be optimized directly; the hull of these arms is simply small (the four real arms span 0.07 in J).

**Verdict: NO-GO** for the context-dependent mixture in d = 4 (pre-registered rule; `just decide` not yet run, d = 2 member pending).

**Next.** d = 2 member (`pilot_20261113-152733`, running). Then, per decision 0010: the gain has to come from outside the hull (the extra dimension on the branch), with the same paired objective and an optimizer that scales.

## §3 Pilot, d = 2 member, and both together (2026-11-14)

**What ran.** `task.d=2`, same sizes as §2, on ulysses 2026-11-13 15:27 onwards (`results/mixture_policy/pilot_20261113-152733`, code 0789f24; 1 member, DONE).

**Result** (as §2; `summary` table):
- Context over budget-only, J: **+0.014 [0.003, 0.025]**; other candidate set +0.007 [−0.004, 0.019]. In the area alone +0.018 [0.007, 0.029] (other set +0.014 [0.003, 0.025]); in the final regret +0.009 [−0.005, 0.025].
- Context over MES (best arm on validation): +0.045 [0.033, 0.057]; over EI (best arm on the test tasks): +0.034 [0.022, 0.046]; other candidate set +0.050 [0.036, 0.065] and +0.035 [0.023, 0.046].
- Budget-only over MES: +0.032 [0.018, 0.045]; uniform mixture over MES: +0.031 [0.018, 0.044].
- Search (`train` table): validation J −2.699 → −2.725 (budget) → −2.741 (context).

**Both members.**
- The in-context mixture beats every single arm on fresh tasks in d = 2 and d = 4, by 0.03–0.05 log10 (7–10% less regret); in d = 2 it holds on a second candidate set, in d = 4 it halves there.
- The context's share is +0.014 (d = 2) and +0.007 (d = 4); the budget schedule's share over a uniform mixture is about zero in d = 2 and +0.014 in d = 4.
- The arms' hull is small (the four real arms span 0.07 in J) and flat inside.

**Verdict:** d = 2 **INCONCLUSIVE**, d = 4 **NO-GO** for "context over budget-only" by the pre-registered rule (`just decide` not yet run).

**Next.** Per decision 0010: leave the hull. The paired objective can be optimized (validation and test both move); the gain available from reweighting these arms is under 0.05.

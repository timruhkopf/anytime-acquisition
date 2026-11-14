# 0004: Fine-tune the PFN on the BO trajectories that policy iteration produces anyway

Status: **proposed, assessed, not acted on** (2026-11-11). Proposed by Tim. Assessment written by Claude.
Companion of [0003](0003-amortized-policy-improvement.md) (§5.1 step 5, §7.6, §8.3, §9).

## Context

The PFN is trained on contexts sampled from the prior: x mostly uniform, y from the task. A BO run
produces a different kind of context. Points cluster around incumbents, a few points are far away, and the
y values are skewed toward the optimum. Several papers suggest that real BO trajectories are out of
distribution for a PFN trained this way. Which papers, and what each shows, still has to be collected
(`status: to-verify`; none is cited here). Notebook 02 and static_tune log §1 show the practical symptom:
near-tie choices are fragile on late, clustered contexts.

Policy iteration (0003 §5.2, §9) generates many ground-truth rollouts on prior draws just to label Q.
Every rollout is a BO-shaped context sequence on a function whose ground truth is known exactly.

## Proposal

1. **Store trajectories cheaply.** Store (task seed, x sequence, noise-stream keys), not y. The function is
   a deterministic CPU draw from its seed (`bo/tasks.py::draw_tasks`). Any y or f at any point can be
   recomputed, including points that were never queried.
2. **Fine-tune the PFN on a mixture.** Use prior contexts as in pretraining, to keep the PFN capable on
   them, plus contexts that are prefixes D_{1:n} of stored trajectories.
3. **Query anywhere.** For a trajectory context, draw fresh query points every epoch: uniform points, the
   candidate generator's distribution (Sobol plus perturbations around incumbents), and points near the
   context. Targets come from the ground truth: f(x_q) for the latent head, and a fresh noisy y for the
   noisy head. One stored trajectory therefore yields an unbounded number of training targets.

## Assessment

### 1. It doesn't bias the PFN's target, under one condition

The PFN's loss is minimized by the true posterior predictive p(y | x, D) at every context that has
positive density under the training context distribution. Changing that distribution reweights where the
approximation is good. It doesn't change what the approximation converges to.

The posterior itself also doesn't change when D is collected adaptively. If the design policy depends
only on the observed data and its own randomness, the design factor in the likelihood doesn't depend on
the task ω. So p(ω | D) is the same as for a fixed design: the design is ignorable (the likelihood
principle for sequential designs).

BO-shaped contexts are therefore legitimate training data for the same target: the Bayes posterior under
the prior. They are not a different task.

**The condition: only non-anticipating trajectories.** A trajectory is ineligible if its x choices
depended on ω beyond the observed y. Such a trajectory makes D informative about ω through the design,
so the PFN would learn a biased "posterior". Concretely:

| Trajectory source | Eligible? |
|---|---|
| base rollouts of π_k (EI, π_1, …) and off-policy heuristic rollouts (0003 §5.2 coverage) | yes |
| label continuations: query candidate c (chosen by stratification, not by ground truth), then π_k | yes |
| privileged controller / clairvoyant ceiling (choices use true-f rollouts) | **no** |
| any "best branch" or best-of-K trajectory selected on ground-truth regret | **no** (the whole trajectory; prefixes before the selection point are fine) |
| contexts from tasks kept or dropped by a ground-truth filter (0003 §5.3: f\* − med < 10⁻³) | changes the task distribution; small, but apply the same filter to prior contexts or skip it here |

This rule belongs in code: each stored trajectory gets a provenance tag, and the loader refuses
ineligible tags.

### 2. Does it reduce PFN error during policy iteration? Yes, with three qualifications

PFN error enters the method in three places: π_0 itself (EI on a wrong predictive), the value head's
inputs, and the model-based ceiling (0003 §8.3), whose continuations use the PFN's own fantasies. Fine-
tuning on the states the current policy visits targets exactly the contexts where these errors matter.
The three qualifications:

- **One round late.** The trajectories of round k cover the states of π_k. π_{k+1} visits other states.
  This is the same distribution-shift cycle as approximate policy iteration itself, and it closes the same
  way: every round adds the new policy's states.
- **Only between rounds.** The labels of round k describe π_k with the PFN it ran on (PFN_k). Fine-tuning
  inside a round turns the fixed target into a moving one, which is the stability argument of 0003 §9.2.
  So the order is: freeze PFN_k, run round k, then fine-tune to PFN_{k+1}. The head of round k+1 may read
  features from PFN_{k+1}, because features are recomputed on load (0003 §6.3) and any F_t-measurable
  input is legitimate. The base pick x_0 = π_k(s) must still come from the policy as it was labeled, i.e.
  from PFN_k. Two encodings per step until the round is relabeled.
- **It changes how the gain is attributed.** With ground-truth labels, part of the gain of π_1 over EI is
  "correction of PFN error" (0003 §3.4). A better PFN shrinks that part. This helps the measurement: the
  remaining gain is closer to the pure planning gap. But results must name the PFN version, and EI on
  PFN_{k+1} becomes a new, stronger baseline.

The value head already corrects PFN error, but only where it changes decisions, and only through the
labels. Fine-tuning corrects the belief itself. The two are complementary.

### 3. "Uses the rollouts more effectively": true for targets, not for functions

Querying anywhere makes the number of targets per context unbounded. That addresses the error of a
regression with few targets per context. It doesn't add task diversity.

Label generation creates 16 candidates × 4 seeds = 64 continuations per state, all on the same function.
They are strongly correlated contexts of one ω. The effective sample size is the number of functions
(about 50k per d in round 1). Pretraining saw about 38M (300k steps × 128 tasks).

Fine-tuning a 25M-parameter model with unbounded queries on 50k fixed functions can memorize them:
the model learns the functions rather than the posterior. Mitigations:

- keep fresh prior draws in the mixture (they also prevent forgetting);
- sample trajectory contexts uniformly over functions, not over continuations;
- refresh with new base rollouts, which are cheap compared with labels: one rollout per instance against
  64 continuations;
- hold out functions, not contexts, for validation.

"Reduces bias" is accurate in this sense: it reduces the approximation bias of the PFN on BO-shaped
contexts (covariate shift), not the bias of the Bayes target.

### 4. Practical points

- **Where the code lives.** PFN training is pfn-repro's job (project rule: a dependency, not a workspace).
  This repository writes the trajectory store and its provenance tags. pfn-repro gets a context source
  that reads such a store and draws functions from seeds, plus the mixture weight. Then bump the tag here.
- **Cost.** The contexts are free by-products. The fine-tuning cost is small next to label generation.
  Ground-truth recomputation is one BNN forward pass per query batch.
- **Choice of query points.** Weight the generator's proposal distribution, because that is where
  decisions are made, and keep uniform queries for calibration elsewhere.
- **Noise (P2 and later).** Latent targets f(x_q) are exact. That is the best possible supervision for
  the latent head, and it can't be had from real data. The noisy head gets fresh noise from the noise
  model. The context's own noise must be recomputed from the stored stream keys.
- **What it doesn't fix.** The PFN still outputs marginals only (0003 §3.1, no joint). Noisy EI and
  martingale consistency are unaffected.

### 5. Checks before adoption (a separate experiment, cheap, and needing no labels)

Run EI rollouts on split-4 functions, fine-tune on a mixture, and compare against the frozen PFN:

1. NLL and 50/90% coverage of f(x_q) on held-out functions, at BO contexts of EI binned by t/T. This is
   the target.
2. The same on uniform prior contexts. This is forgetting: no loss beyond a pre-registered tolerance.
3. Paired EI regret on fresh tasks (split 1 only for the final number), PFN_1 against PFN_0.
4. Contexts of π_1 (if available) as an off-distribution check on the "one round late" point.

GO if (1) improves clearly, (2) stays within tolerance, and (3) is not worse. A GO feeds both the PFN
(Stage 0 of 0003, round B) and every later policy-iteration round. It generalizes 0003 §5.1 step 5 (BO-like
contexts bootstrapped from latent EI) to every round's policy, with queries anywhere.

## Decision

Not adopted yet. Prerequisite: round 1 (#13) runs on the frozen PFN, so that the PFN version stays fixed
for the first measurement. Next step if pursued: an issue in pfn-repro for a trajectory-store context
source, and the experiment in §5 here.

## Relation to 0005 (added 2026-11-11)

[0005](0005-assessment-pfn-finetuning-on-rollout-contexts.md) is Tim's assessment of the same proposal, logged
verbatim. The two agree on the core points:
- design ignorability makes BO contexts valid training data for the same Bayesian target;
- trajectories must be non-anticipating, with no outcome-based selection;
- fine-tuning happens only at round boundaries;
- correlated prefixes mean the instance count, not the prefix count, is the sample size;
- forgetting is guarded by a prior-context mixture.

0005 adds what this note lacks:
- a separate test-noise stream, because common random numbers would otherwise leak the context's noise into
  test targets (rule 3);
- the four-arm attribution (original or fine-tuned PFN × EI or π_k);
- an error-size gate (E1) before any fine-tuning;
- held-out-policy evaluation.

One difference: 0005 redefines all policies on the fine-tuned PFN at the boundary and relabels. This note kept x₀ on
PFN_k until relabeling. 0005's schedule is the cleaner one; follow it.

<!-- Claude, 2026-11-12, recording the reasoning shared with Tim after the api pilots, pfn_bo_inference and ei_opt (issue #13). Status: open; supersedes the "next steps" of 0007 §4 where they conflict. -->

# Where the headroom over EI is: basin discovery, structured deviations, and luck-corrected switching

Nov 12, 2026 · Claude with Tim

## 1. The shared observation

Rounds 1 and 2 of amortized policy improvement (0003, 0006) improved nothing over EI:
- π₁ − EI: −0.010 [−0.033, 0.002] (api_round1);
- decoder − EI: −0.1% [−1.1%, +0.9%] (api_decoder §2).

The reasons, as we now understand them:

1. **The reward was linear where it should be logarithmic.**
   - Labels and decisions used final normalized regret (% of f_med − f\*).
   - EI's median regret falls from 10⁻¹·⁴ to 10⁻⁵·¹ (ECDF units) over the budget, and 80% of tasks end within one PFN bin of f\*. Source: the budget-vs-regret scratch figure, 2026-11-12 track record, on the 1000 split-1 tasks of `api_round1/pilot_20261111-133311` and `api_decoder/pilot_20261111-165340`.
   - On a linear scale, progress near the optimum is worth almost nothing. BO practice (log simple regret) is the reward to follow, so that results are comparable with other BO methods.
2. **The labels' noise was our own.**
   - π₀ maximized EI over a fresh random candidate set, so its picks are stochastic: replicate picks differ in 90% of states (ei_opt §2).
   - On noise-free tasks that is the only noise in a continuation.
   - A π₀ whose candidate set depends only on the data (`fixed`) is deterministic. It is indistinguishable from the sampled EI on normalized and log regret (post hoc, ei_opt §2), and makes one continuation an exact label.
   - Refining EI by gradient is worse on log regret: +0.056 [0.013, 0.100] log10.
3. **Log regret alone mixes two different things** (Tim's point).
   - **Strategic: which basin the incumbent is in, and when it got there.** An early decision can decide whether the global basin is found within the budget. Missing it costs orders of magnitude, and an under-explored surface is where EI's greed can fail.
   - **Tactical: polishing inside the basin.** Once the right basin is found, EI with local perturbations is a reasonable local optimizer. The remaining 10⁻³ → 10⁻⁵ differences depend on local geometry, the perturbation scale and tie-breaks, which are close to luck from the policy's view.
   - A log reward amplifies the tactical part. Most states at d = 2 are in the polishing regime, so most label variance would come from the part we cannot plan.
   - The plannable advantage is **finding the right basin earlier**, while reaching the same or lower final regret.
4. **One-step deviations are the weakest perturbation** (Tim's point).
   - **Long remaining horizon:** EI recovers from a single different action, so the advantage tends to 0. This is genuine: there is no value in deviating there.
   - **Short horizon (late):** the posterior is concentrated and EI is near-optimal.
   - **Signal therefore exists only where actions are hard to undo:** early, on an under-explored surface, where a choice decides whether a basin is found in time.
   - **Cutting the label's horizon short** would boost the signal but reward immediate improvement, the same exploitation bias as an "anytime" reward. Terminal reward at the known budget stays the objective. "Sooner" becomes an evaluation check: the same or better final regret, with an earlier time to basin or earlier time to reach EI's final regret.
   - **The structured fix:** a *larger* deviation. Follow a different acquisition rule for k steps, then continue with π₀. Its effect survives the horizon better, the action space is small (about 6 arms rather than 133 candidates), and it is the README's original policy π(arm | PFN state, budget left).

## 2. Luck-corrected switching headroom exists; recovering it from the belief has failed so far

- **On BBOB** (decided GO, `bbob_switching` §2):
  - switching with task knowledge gains over the best static arm +66% [+51, +76] at d = 2 and +29% [+18, +39] at d = 5 (PFN), and +59% / +19% (GP);
  - the equal-arms null is ≈ 0;
  - a per-*function* static arm choice gains nothing. The headroom depends on the state within a run, not on problem identity.
- **On the prior** (switching pilot `runs/switching/pilot_20261021-152818`, split 2, 48 tasks per d; not snapshotted or decided, indicative only, issue #6):
  - privileged switching gains over the best static arm +37% [+15, +60] (d = 2), +36% [+11, +56] (d = 4);
  - the null is clean;
  - the **belief controller loses**: −49% [−109, −11] at d = 2.
  - Issue #6 traced the loss to its estimator: the minimum of fantasized *noisy* y, a winner's curse rewarding sampling near the incumbent. The gate's lookahead in api_gate failed the same way: its estimate was dominated by Monte-Carlo noise (0007 §3).
- **Static arms on the prior** (static_tune §3–4): the averages are close; at d = 2 the best static arm is MES, not EI. Per-task rankings vary strongly (per-task log-ratio sd ≈ 0.55), but part of that is path-dependence luck, not predictable structure.
- **Reading.** Arms disagree on actions, their rankings vary across instances, and luck-corrected privileged switching headroom is large. Together that supports context-dependent acquisition selection (no free lunch for any single arm). What has never worked is the deployable estimator, and both failures were estimator failures, not failures of the principle.
- **Theory.** Rollout with **multiple base heuristics** (Bertsekas, rollout algorithms; reference to verify and add) has a cost-improvement property:
  - picking, per state, the heuristic whose rollout is best is at least as good as the best single heuristic, when the rollouts are evaluated exactly;
  - under the belief, this holds in expectation over the prior if the belief is the Bayes posterior. The PFN is close to it on its prior (pfn_bo_inference: calibrated, with only a small error at BO contexts);
  - a conservative acceptance test (accept only if better than the current policy on split 0) keeps the result at least as good as EI, up to estimation error.

## 3. Privileged basin labels: what they give, where they can mislead, how far they scale

**What they give.**
- For a task, run Sobol starts plus local descent on the true f to enumerate the local optima and their values, unless the function is very rugged.
- For each incumbent, local descent identifies its basin.
- Per run: τ, the first step at which the incumbent is in a *good* basin (or never), and whether the budget sufficed at all.
- Per policy: P(good basin by t). This splits log regret into a strategic part (τ) and a tactical part (polishing given the basin).

**Where they can mislead.**
1. **Near-equal optima.**
   - Two basins whose minima differ by less than the polishing error are not a meaningful "wrong basin".
   - Define "good basin" as a basin whose minimum is within δ of f\*, with δ at or above one PFN bin.
   - Report how much depends on δ.
2. **Ill-defined basins.**
   - Irrelevant dimensions make valleys, not points: descent ends anywhere along a flat direction. Compare minima by value, not by location.
   - Plateaus: the ECDF output transform saturates at the extremes, and a crude 64×64 grid check found no strict local minima for most tasks, which suggests ties.
   - Rugged draws.
   - Attraction basins also depend on the descent method. Fix one protocol and check that a second step size agrees.
3. **Diagnostic, not reward.**
   - The objective stays terminal log regret. Optimizing τ directly would reward reaching a good basin without polishing it, or prefer one of two near-equal basins.
   - τ explains *where* a gain comes from; it is not what we maximize.
4. **Luck in τ.**
   - Even with deterministic policies and noise-free tasks, whether an early evaluation happens to land in the basin is luck at the level of the instance and design: a fixed Sobol grid may cover it by chance.
   - Correction, reusing what worked for switching:
     - paired comparisons on the same instances;
     - an **equal-arms or uninformed-perturbation null**: a deviation of the same size without information, such as a random arm or a different fixed candidate set. Its effect on τ is the luck baseline;
     - **several deterministic π₀ variants** (different fixed seeds) as replicates, to estimate the spread of τ that is pure luck;
     - **cross-fitting** whenever a best option is selected (select on some replicates, score on others).
   - A gain in τ counts only above the null.
5. **Necessary, not sufficient.** Being in the global basin at T is necessary for low regret, but polishing still decides the final value. Report both.

**How far it scales.**
- **d.** Enumerating optima by Sobol plus descent needs many more starts as d and ruggedness grow, and the number of local minima can grow exponentially. f\* itself (Sobol plus Adam) is reliable at d = 2: a 16× stronger search never improved it by more than 1e-6 (2026-11-12 track record). At d = 4–6 it needs to be checked the same way.
  - pfn_large is valid only up to d = 6 anyway. Within that range, per-incumbent descent on the cheap BNN tasks is affordable. Full enumeration of optima may not be at d = 6, but identifying the incumbent's basin and checking it against f\*'s basin is.
- **Off the prior.** BBOB has known optima and basin structure, so the labels work there. Real problems have no ground truth.
- That is acceptable: privileged information is needed only for **training labels and analysis on synthetic priors**, never at decision time. The deployed policy reads only the belief.

## 4. Order of work

1. **Decompose (privileged, no new BO runs if possible).** For EI and the main static arms on the prior at d = 2:
   - P(good basin by t), τ, and the polishing given the basin;
   - how much of the arms' differences and of the switching headroom is basin discovery;
   - with the δ sensitivity and the luck nulls of §3.
2. **Exact option labels.**
   - At states, the action is "follow arm a for k steps (k ∈ {1, 3, 5}), then π₀". Every arm uses deterministic candidate sets, so one continuation is exact.
   - Outcomes: terminal log regret and τ.
   - Measures luck-corrected switching headroom with noise-free labels, by t and k.
3. **From the belief.**
   - Belief rollouts done properly: deterministic arms, values on the latent f rather than noisy y (issue #6), enough samples, variance reduction.
   - Or a value model per (arm, k) trained on step 2's labels.
   - Then conservative acceptance against EI and the best static arm on split 0, and the test on split 1.

Paused: the candidate-level relabeling round (one-step deviations, log reward) proposed earlier on 2026-11-12. Fine-tuning the PFN (0004/0005) stays deferred.

<!-- Assessment by Tim, 2026-11-11, added verbatim from the shared doc ("Assessment: fine-tuning the PFN on BO rollout contexts"). Status: proposed, not acted on. Companion of 0003 (main design) and 0004 (Claude's earlier assessment of the same idea; see the note at the end of 0004 on how the two relate). -->

# Assessment: fine-tuning the PFN on BO rollout contexts

Nov 11, 2026 · @Tim

Status: proposed, not acted on. Policy iteration produces millions of BO trajectories with ground truth attached; reusing them as PFN training contexts, mixed with prior contexts, is a valid Bayesian training target, costs almost nothing in data, and targets exactly the error source BO suffers from. It should be done between policy-iteration rounds, never within one, and assessed first by measuring how large the PFN's error on BO contexts actually is.

## 1. Proposal

Store every rollout as a seed plus an (x, y) sequence, and fine-tune the PFN on a mixture of prior contexts and prefixes of these rollouts, with fresh ground-truth test targets at arbitrary locations.

**What is stored per rollout** (a few kilobytes):

- the instance seed, from which the task ω (BNN weights, architecture, noise model, output warp) is regenerated exactly;
- the noise seed of the rollout;
- the policy that generated it (latent EI, π\_k, UCB, Sobol, …) and its round;
- the sequence (x\_1, y\_1), …, (x\_T, y\_T), in order.

**How a training example is built:**

1. Draw a stored rollout and a prefix length n ∈ {1..T}; the context is D\_n = (x\_i, y\_i) for i ≤ n.
2. Choose test points from three sources: uniform in X; Gaussian perturbations around context points; and the rollout's actual next query x\_{n+1}, the point that matters most for BO.
3. Targets: the latent value f\_ω(x̃), exact from the regenerated task; and a noisy ỹ drawn fresh from the instance's noise model with a separate noise stream (§2, rule 3).
4. Mix with ordinary prior examples (uniform and replicate contexts) at a ratio ρ, so the PFN keeps its capability on prior-shaped contexts.

**Relation to the existing plan.** §5.1 of the main document already bootstraps BO-like contexts from latent-EI runs (round B of pretraining). This proposal generalizes that step: it reuses all rollouts that policy iteration produces anyway, from every policy and round, and keeps doing so as the policies change.

## 2. Validity: the Bayesian target does not change

BO-built contexts lead to the same posterior predictive as uniform ones, because the design is ignorable: the policy chooses x from data already observed, so it carries no extra information about the task. Only the weighting of contexts in the loss changes, which is the point.

**Argument.** Under a policy π the context has probability

```latex
p(D_n \mid \omega) = \prod_{i=1}^{n} \pi(x_i \mid D_{i-1}, u_i)\, P_\omega(y_i \mid x_i)
```

where u\_i is the policy's own randomness. The design factors do not depend on ω, so they cancel in the posterior:

```latex
\Pi(d\omega \mid D_n) \propto \Pi(d\omega) \prod_{i=1}^{n} P_\omega(y_i \mid x_i)
```

The NLL minimizer over this training distribution is therefore the same posterior predictive as for uniform contexts, for any test-point rule that depends only on D\_n and independent randomness:

```latex
q^\star(\tilde y \mid \tilde x, D_n) = \int P_\omega(\tilde y \mid \tilde x)\, \Pi(d\omega \mid D_n)
```

The mixture over rollout and prior contexts changes only where the network spends its capacity. The function it should learn is unchanged.

**Rules that keep it valid.**

1. **Policies see only observed data.** Every policy generating stored rollouts must depend on ω only through observed (x, y). Never store or train on rollouts of clairvoyant policies (oracle selection, branches chosen by their ground-truth outcome).
2. **No outcome-based filtering of instances or rollouts.** Keeping only rollouts where BO failed, or instances where the PFN erred, selects on ω and changes the effective prior; the PFN would learn a posterior under that distorted prior. Reweighting is allowed only if declared and corrected, and is best avoided.
3. **Fresh, independent test noise.** The common-random-number scheme makes noise a deterministic function of (instance, seed, x, visit count). A test point at a location already in the context would then repeat the context's noise and leak it. PFN test targets must use a separate noise stream.
4. **Test-point rule independent of ω.** Uniform points, perturbations of context points and the policy's next query are fine. Points chosen near the true optimizer x\* are not.
5. **Same transforms.** Rollout contexts go through the same per-context standardization as prior contexts (main document §7.4).

## 3. What it buys

It targets context-design shift, the PFN error that falls exactly on the states BO visits. During policy iteration it lowers PFN error in four places, but not in the labels, which are ground truth and unbiased either way.

**The error source.** A PFN trained on uniform contexts matches the posterior predictive only where those contexts put mass. BO contexts are clustered near good regions, contain near-duplicates, and grow in a data-dependent order. Amortization error is therefore largest where decisions are made: near the incumbent, at the next query, in late-stage contexts. Rollout contexts put training mass exactly there.

**Free data.** Label generation produces about T steps per rollout, each a valid prefix, for tens of thousands of instances per round. With ground truth regenerable from the seed, every stored rollout yields unlimited test points. The marginal data cost is close to zero; only fine-tuning compute is added.

**Where PFN error drops during policy iteration:**

| Place | Effect of a better PFN on BO contexts |
| --- | --- |
| Readout and value-head inputs | features (μ, σ, EI, noise estimates) are more accurate on visited states, so the value head has less to compensate |
| Base and continuation policies | latent EI and every π\_k decide on a more accurate posterior, so they are stronger policies |
| Model-based gate and lookahead teacher | PFN-simulated continuations are closer to the truth; the gap between the model-based and the clairvoyant ceiling should shrink, and that shrinkage measures the effect |
| Martingale and calibration diagnostics | better coherence on BO contexts makes fantasies and depth-1 lookahead more trustworthy |

**Where it does not help.** The value-head labels are ground-truth returns, so they are unbiased with or without fine-tuning; a better PFN changes which policy is being improved, not the correctness of its labels. Fine-tuning also cannot fix prior mismatch: the PFN gets better at the posterior of the training prior, not at reality.

**Answer to the guiding question.** Yes: it should reduce PFN error during policy iteration, and with it the error in the policies, features and model-based estimates. It also shifts part of the gain from the value head into the PFN, which matters for how results are attributed (§4).

## 4. Interaction with policy iteration

Fine-tuning changes the PFN, and the PFN is inside every policy and every feature; so it may only happen at round boundaries, followed by relabeling.

**Why not within a round.** Labels for round k are returns of π\_k, and π\_k is defined through the PFN (latent EI uses its posterior; later rounds use its features). Changing the PFN mid-round changes the policy whose returns the stored labels describe. The labels then belong to a policy that no longer exists.

**Schedule.**

1. Round k: PFN frozen. Roll out π\_k, store rollouts, generate labels, fit the head, validate.
2. Boundary: fine-tune the PFN on the mixture of prior contexts and all stored rollouts so far (all policies and rounds, not only the last).
3. Redefine the policies on the new PFN (latent EI and π\_k now read the fine-tuned PFN), re-validate π\_k, and relabel for round k + 1.

Relabeling is needed anyway in policy iteration, so the schedule adds no labeling cost; it adds only the fine-tuning compute.

**Attribution.** Gains can now come from two places: a better PFN (better posterior, so better EI) and a better policy on top of it. Keep both arms in every comparison:

| Arm | PFN | Policy |
| --- | --- | --- |
| A | original | latent EI |
| B | fine-tuned | latent EI |
| C | original | π\_k |
| D | fine-tuned | π\_k |

B − A is the value of fine-tuning alone; C − A the value of the policy alone; D − B the policy's value on the better PFN. The planning-gap measurement of the main document should be reported on both PFNs: if fine-tuning closes much of the gap, the gain was PFN error, not planning.

**Moving baseline.** Latent EI on the fine-tuned PFN is a stronger baseline than on the original. Any claim of improvement over EI must name which PFN it uses.

## 5. Risks and mitigations

The main risks are forgetting, narrowing onto the current policy's habits, and leaking information through noise or selection; all have simple guards.

| Risk | What goes wrong | Mitigation |
| --- | --- | --- |
| Forgetting | accuracy on prior-shaped contexts drops; early BO steps and new policies suffer | mixture ratio ρ with prior contexts; gate on uniform-context NLL and coverage (§6) |
| Narrowing to one policy | the PFN becomes good on π\_k's states and worse elsewhere; when the policy changes, shift returns | train on rollouts of all policies and rounds (EI, UCB, Sobol, every π\_k); evaluate on rollouts of a held-out policy |
| Feedback loop | PFN fits π\_k's states, π\_k+1 concentrates further there, coverage shrinks | keep exploratory policies in the rollout mix; monitor fill distance of stored contexts across rounds |
| Correlated prefixes | prefixes of one rollout are highly correlated, so the effective sample size is smaller than the count | sample one or a few prefixes per rollout per epoch; count instances, not prefixes |
| Noise leakage | test target at a context location reuses the context's noise under the common-random-number scheme | separate noise stream for PFN test targets (§2, rule 3) |
| Outcome selection | filtering rollouts or instances by results changes the effective prior | no filtering; uniform over stored rollouts (§2, rule 2) |
| Calibration in the tails | BO contexts sit near optima, where few training examples existed; overfitting there is easy | early stopping on held-out instances; PIT and coverage stratified by distance to the incumbent |
| Moving target for attribution | gains from PFN and policy blur | four-arm comparison (§4) |

## 6. How to assess it

First measure how wrong the PFN is on BO contexts; if it is barely wrong, stop. Otherwise fine-tune and check that it helps downstream without forgetting.

**E1: Size of the error (no training, cheap).** On held-out instances, compare the original PFN on uniform contexts and on prefixes of stored rollouts at matched context size n.

- Metrics: NLL and CRPS of the latent f; 50% and 90% interval coverage; PIT histograms; the martingale self-test.
- Test points stratified: uniform, near context points, the rollout's actual next query.
- Per policy (EI, UCB, π\_k) and per stage (early, middle, late prefixes).
- **Gate:** the latent NLL gap between rollout and uniform contexts, at the next query, exceeds a pre-registered threshold (for example 0.1 nats), or 90% coverage on rollout contexts is off by more than 5 points. Otherwise there is little to gain.

**E2: Fine-tuning without forgetting.** Fine-tune with mixture ratios ρ ∈ {0.25, 0.5, 0.75} of rollout contexts.

- Evaluate on held-out instances, on rollouts of a held-out policy (generalization across policies), and on uniform contexts.
- **Gate:** E1's gap closes by at least half on held-out instances and held-out policies, while uniform-context NLL worsens by less than 2% and coverage stays within ±5 points.

**E3: Downstream value.** Paired, with shared noise, on fresh instances:

- latent EI on the original versus the fine-tuned PFN (arm B − A of §4);
- the gap between the model-based and the clairvoyant ceiling, before and after;
- the policy-iteration round run on both PFNs (arms C and D).

**Gate:** B − A is non-negative with a paired interval excluding a loss; fine-tuning is kept for later rounds only if this holds.

**Cost.** E1 needs only stored rollouts and forward passes. E2 is a fine-tuning run of a fraction of the pretraining budget. E3 reuses the evaluation protocol of the main document.

## 7. Verdict and recommendation

Recommended, with two conditions: run E1 before any fine-tuning, and fine-tune only at round boundaries with the four-arm attribution.

| Question | Assessment |
| --- | --- |
| Is it valid? | Yes. Design ignorability makes BO contexts a legitimate training distribution for the same Bayesian target, provided the five rules of §2 hold. |
| Is it cheap? | The data is free; ground truth is regenerable from seeds; only fine-tuning compute is added. |
| Does it reduce PFN error during policy iteration? | It should, on exactly the visited states: in features, in base and continuation policies, and in model-based estimates. It does not change label correctness, which never depended on the PFN. |
| What is the main cost? | Interpretation: part of the gain moves from the policy into the PFN, and EI on the better PFN is a stronger baseline. |
| Likelihood it pays (estimate) | Measurable error on BO contexts (E1 passes): about 70%. Downstream gain for latent EI (E3 passes): about 50%. |

**Open questions.**

- [ ] Which earlier work reports that BO-generated contexts are out of distribution for PFNs; collect and cite (not verified here).
- [ ] Mixture ratio ρ and whether to anneal it across rounds.
- [ ] Whether to fine-tune the whole PFN or only the last blocks and heads.
- [ ] Whether to weight the next-query test point more heavily, since it is the one BO acts on.

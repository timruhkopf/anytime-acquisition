<!-- Claude, 2026-11-13, recording Tim's framing from the discussion while mixture_policy (notebook 17) was in its sanity run; issue #13. Status: open. Builds on 0009. -->

# From the arms to a learned acquisition function: start inside the hull, then leave it

Nov 13, 2026 · Tim's framing, written up by Claude

## 1. The goal

A policy that depends on the context, reads the state from the frozen PFN, knows its remaining budget, and beats every single arm on the prior, in final regret and in how fast it gets there.

We know no single arm wins everywhere. So a context-dependent mixture of the arms is a starting point that is at least as good as the best arm, and everything better has to be reached from there.

## 2. The construction

The policy is the argmax of an acquisition value function α(x | D, budget). That function is what we optimize. It is built in three parts.

1. **The marginals: one network that holds every arm.** The acquisition branch of decision 0009, distilled in parallel across arms: output dimension k ranks candidates as arm k does. One network then contains every arm's candidates.
2. **Reweighting: an in-context mixture.** Weights w(context, remaining budget) over those K dimensions give a single acquisition function. A one-hot w is one arm exactly; mixed weights can pick a candidate no arm ranks first. This is GP-Hedge made context-aware and amortized, and it stays inside the convex hull of the arms.
3. **Leaving the hull: extra mass.** One more output dimension, initialized at zero and added to the mixed score. The policy starts as the mixture; only an outcome signal can move it towards options no arm proposes.

```
 frozen PFN layers ──► acquisition branch ──► K arm rankings ──┐
        (cross-attention, 0009)          └──► extra score (0) ─┤
 context, remaining budget ──► weights w ──────────────────────┴──► α(x | D, budget) ──► argmax
```

Why distillation at all: not as an end, and not as a prerequisite for the reward question. It makes the policy start strong, from actions known to be good but not optimal, in a network whose readout of the PFN is rich enough to go further.

Why a rich readout: the PFN's layers encode the BAMDP state. Hand-made belief features are a narrow view of it. Cross-attention into every layer is a far richer one, at the price of many more parameters to estimate.

## 3. What everything then depends on

A training signal that moves α towards better acquisition, beyond
- noise in the outcome (luck of the candidate set and of the task),
- measurement error of the reward (regrets below one PFN bin),
- a reward that does not distinguish what a better policy would do differently.

Evidence on how hard this is (all at d = 2, split 4):
- Per-decision advantage labels are almost all luck: the Q regression's held-out R² stays at 0.04 with 800 or 1600 training tasks, ridge and MLP alike (scratch on the run artifact `labels.pt` of `results/option_policy/pilot_20261112-123724`).
- The policy from those labels gains +0.017 [−0.018, 0.050] over the best static schedule (`results/option_policy/pilot_20261112-123724`, `summary`).
- Gating UCB by EI's predicted failure equals the static schedule, +0.000 (`results/ei_failure/pilot_20261113-075212`).

## 4. The steps, and what each one tests

| Step | Experiment | What it tests |
|---|---|---|
| Mixture on the true arm scores, 42 parameters, direct paired search on the outcome | `mixture_policy` (notebook 17), running | The reward and the pairing alone: is there a signal that moves a policy above the best arm? No readout to blame. |
| Per-arm distillation into the branch | `acq_distill` (notebook 16), running | Does the branch reproduce every arm, on candidate sets, dense grids and in closed loop? |
| Mixture head on the distilled branch | not built | Same weights on distilled scores: the start inside the hull, in one network. |
| Extra dimension, trained on the outcome | not built | Can we leave the hull? Needs an optimizer for many parameters. |

How to read the mixture result:
- It beats the best arm: the objective carries signal; move the same objective to the branch.
- It does not: reweighting known arms is not where the gain is, or the reward is too noisy. This does not say the extra dimension would fail; it says to fix the reward first.

## 5. Open

- **The optimizer for the branch.** Direct search works for tens of parameters, not for a transformer. Candidates: policy gradient over a softmax on the candidates with the best arm's run on the same task and candidate seed as control variate; horizon bootstrapping (Q₁ = EI's value as boundary condition, longer horizons from the PFN's one-step expectations); a privileged critic.
- **The reward.** Currently half final, half mean over steps of log10(regret + 1e-3); 1e-4 and 1e-6 reported. Not yet confirmed by Tim.
- **Distillation quality.** At 3000 steps in d = 1 the branch follows the arms' shape but does not yet reproduce their picks (`runs/acq_distill/sanity_20261113-103542`, not snapshotted). A 12000-step run with post-readout blocks, y input, incumbent marker and aligned readout is training (`runs/acq_distill/sanity_20261113-113139`).
- **Horizons.** A distribution over horizons as a curriculum (0009 §3) is not yet in any run.

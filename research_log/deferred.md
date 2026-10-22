# Deferred and future work

Directions considered and deliberately **not** pursued now, each with the reason and what would make it
worth revisiting. Issues are the actionable backlog; this file is the scientific judgement behind what was
left out. It feeds the paper's limitations and future-work sections, and keeps a narrowed experiment honest
about what it no longer tests.

Add an entry whenever a design choice, a pre-registration or a verdict narrows the scope. Entries are
append-only like the track record: a later outcome is a dated **Status** line, not an edit of the text.

<!--
## <Short title> (YYYY-MM-DD)

- **Idea.** What would be done, in two or three sentences.
- **Why deferred.** The reason (cost, a missing prerequisite, out of scope, a verdict), with its evidence.
- **Revisit when.** The concrete trigger: a result, a resource, a question that would need it.
- **Pointers.** Log entry § / notebook / issue / references.
- **Status.** open
-->

## Other tests of whether switching acquisition functions has merit (2026-10-21)

The switching experiment (notebook 02) tests one controller: rollout-based, non-anticipating, on the
frozen PFN's own prior. These complementary tests were considered and deferred until its verdict:

- **Idea 1: a trained policy's ceiling on the prior, with a null.** Amortize a switching policy over the
  prior (RL or imitation of the privileged rollout controller), and compare it with the best static arm and
  with the same policy trained on shuffled state features (the null). It measures what amortization buys
  beyond what the rollout controller can find online.
- **Idea 2: the value of task features.** Give the privileged controller the task's hyperparameters
  (lengthscale, noise, depth of the BNN). If knowing the task does not help choose the arm, no
  belief-conditioned policy can either.
- **Idea 3: variance decomposition.** Split the regret variance into task × arm interaction, arm main effect
  and seed noise, at matched seeds (CRN). Switching headroom is bounded by the task × arm (and the
  state × arm) interaction; a small interaction against the seed noise explains a small effect directly.
- **Idea 4: off the prior.** Repeat the comparison on real or standard benchmarks (HPO-B, YAHPO, BBOB) with
  the PFN as the surrogate, where the surrogate is misspecified and the acquisition functions' different
  robustness to it could matter more than on the prior.
- **Why deferred.** The rollout experiment is the cheapest test with a clear null; these need its effect
  size and noise level first (and ideas 1 and 4 need GPU time).
- **Revisit when.** After the switching confirm's verdict: idea 3 in any case (it explains the effect
  size); ideas 1 and 2 if the verdict is GO or INCONCLUSIVE; idea 4 if it is NO-GO on the prior.
- **Pointers.** Notebook 02; log `2026-10-21_switching.md`; issue for the switching experiment.
- **Status.** open

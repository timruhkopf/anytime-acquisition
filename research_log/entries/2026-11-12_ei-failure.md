---
experiment: ei_failure
sweeps: []
---
# Can the context predict how EI's run ends, and does gating the UCB deviation by it beat the static schedule?

## Question

See the description at the top of `notebooks/15_2026-11-12_ei_failure.ipynb`; not retyped here.

## §1 Sanity (2026-11-13)

**What ran.** 100 tasks on ulysses, 07:46–07:49 (`results/ei_failure/sanity_20261113-074557`, code ab5ab73). Collection took 2.3 min, the evaluation 18 s.

**Result** (a pipeline check; 100 tasks are far too few for the decision):
- The pipeline ran end to end: all tables written, 6 deployable and 3 privileged gates, the static schedule.
- Controls behave: the other seed's EI outcome correlates 0.82 (t = 0) to 0.95 (t = 14) with the target; the current true regret 0.19 to 0.92; the predictors 0.2–0.3 at t = 0 and 0.5–0.6 late (`probe` table).
- The primary gate chose the ungated static schedule in every fold (gain over static 0.000).

**Revision before the pilot:** none.

**Next.** The pilot as pre-registered (3000 tasks).

## §2 Pilot (2026-11-13)

**What ran.** 3000 split-4 tasks (offset 50000) on ulysses, 07:52–09:04 (`results/ei_failure/pilot_20261113-075212`, code daaa023; 1 member, DONE). Collection took 71 min, the evaluation 1.3 min.

**Result** (log10 terminal regret, ε = 1e-6; held-out tasks, seed 0; tables of the snapshot):
- Probe (`probe`, `probe_diff`): Spearman of the prediction with EI's terminal regret is 0.29, 0.46, 0.63, 0.74 at t = 0, 2, 6, 14 (`belief/mlp`). Belief over data features: +0.079 [0.053, 0.102] at t = 0, +0.112 [0.089, 0.136] at t = 2, ≈ 0 from t = 12. The PFN state adds nothing to the belief features (±0.01).
- Gate (`summary`): the primary `gate/belief/mlp` and every other deployable gate chose the ungated static schedule; gain over static +0.000. Static: +0.109 [0.084, 0.133] over EI.
- Ceiling: the gate on the other seed's EI outcome gains +0.038 [0.008, 0.067] over static.
- The same-seed outcome as a score gains +0.098 [0.068, 0.129] on seed 0 and −0.012 [−0.048, 0.023] on seed 1: mechanical.
- Strata (`strata`): "UCB for the rest" from t = 4 by tercile of the other seed's outcome: −0.09, −0.00, +0.41; by tercile of the predicted outcome: +0.16, +0.20, −0.03.

**Interpretation.**
- The exploration on notebook 13's labels replicates on fresh tasks.
- The context predicts task difficulty, which is common to all arms and cancels in the advantage. It does not predict on which tasks EI in particular fails.
- This corrects notebook 13's post hoc (log §2 there): the +0.147 from EI's own terminal regret as a feature was mechanical.
- The ceiling was not below the 0.05 stated in advance (upper bound 0.067), so the idea is small rather than formally closed.

**Verdict: NO-GO** (pre-registered rule on `gate/belief/mlp`, seed 0; `just decide` not yet run).

**Next.** No further work on gating by EI's predicted outcome. The probe result (belief > data early) is kept as evidence that the belief features carry state information.

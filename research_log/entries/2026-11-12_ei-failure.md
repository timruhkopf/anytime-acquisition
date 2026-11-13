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

---
experiment: bbob_switching
sweeps: []
---
# Luck-corrected switching headroom on BBOB with GP and PFN surrogates

## Question

See the description at the top of `notebooks/04_2026-10-22_bbob_switching.ipynb`; not retyped here.

## §1 Implementation and sanity (2026-10-22)

- BBOB tasks via `ioh` (`experimental/bbob.py`): f − f_opt, so f* = 0. Tests check that f ≈ 0 at the optimum
  and that lanes select the right task.
- Batched MAP-II GP with BoTorch's default priors, and a PFN on rank-transformed y (`experimental/surrogates.py`).
  Tests check that the GP fits a smooth function and the arms' preferences on toy predictives.
- `Engine.score` hook: the refactor is bit-identical. Old vs new code gave the same best tables on the prior
  (static, privileged, belief, equal_arms; 135 rows, pfn_smoke).
- Timing probe (scratch, ulysses, fbc7d0c): privileged controller, d = 5, 24 tasks: GP 274 s, PFN 173 s.
  This sized split 1 at 192 tasks/d (≈ 3 h for both confirm sweeps).
- Belief controller not run off the prior (notebook 04 description, issue #6).

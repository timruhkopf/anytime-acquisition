---
experiment: bbob_switching
sweeps:
- id: pilot_20261022-144551
  stage: pilot
  commit: 135eb56e9d71
  decision: GO
  reason: 'Selection done per the pre-registered rule (4/4 DONE, 48 tasks per surrogate
    and d): B* = ucb0.9 (GP, d=2,5), ucb0.99 (PFN, d=2,5)'
  members: 4/4 done
  decided: '2026-10-22'
- id: confirm_20261022-145708
  stage: confirm
  commit: 63900a28b7bf
  decision: GO
  reason: 'GP: privileged gain over B*=ucb0.9 +58.6% [+45.9,+68.8] (d=2), +18.8% [+3.1,+31.9]
    (d=5), n=192/d; null -5.0% [-43.5,+22.0], -5.7% [-20.9,+7.7] (CI incl. 0); headroom_verdict=GO'
  members: 6/6 done
  decided: '2026-10-23'
- id: confirm_20261023-075829
  stage: confirm
  commit: 63900a28b7bf
  decision: GO
  reason: 'PFN: privileged gain over B*=ucb0.99 +65.6% [+51.3,+76.1] (d=2), +29.2%
    [+17.8,+38.9] (d=5), n=192/d; null -1.4% [-28.4,+19.6], -1.9% [-16.7,+11.2]; headroom_verdict=GO'
  members: 6/6 done
  decided: '2026-10-23'
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

## §2 Selection and confirm (2026-10-22/23)

- **Selection** (`results/bbob_switching/pilot_20261022-144551`, split 0, 48 tasks per (surrogate, d), 4/4 DONE):
  B\* = ucb0.9 (GP, d = 2 and 5) and ucb0.99 (PFN, d = 2 and 5). Decided GO.
- **Confirm** (split 1, 192 tasks/d; GP `confirm_20261022-145708` 6/6 DONE; PFN `confirm_20261023-075829` 6/6 DONE).
  Gain over B\* (1 − geometric-mean regret ratio, 95% bootstrap CI):

  | surrogate | d | privileged | equal-arms null | per-function static choice |
  |---|---|---|---|---|
  | GP | 2 | +58.6% [+45.9, +68.8] | −5.0% [−43.5, +22.0] | −2.3% [−27.1, +17.9] |
  | GP | 5 | +18.8% [+3.1, +31.9] | −5.7% [−20.9, +7.7] | −9.5% [−32.7, +9.0] |
  | PFN | 2 | +65.6% [+51.3, +76.1] | −1.4% [−28.4, +19.6] | −13.7% [−49.6, +9.8] |
  | PFN | 5 | +29.2% [+17.8, +38.9] | −1.9% [−16.7, +11.2] | −24.0% [−72.5, +3.8] |

  `headroom_verdict`: GO for both surrogates; both confirm sweeps decided GO.
- **Interpretation.** On noise-free BBOB, luck-corrected headroom for state-dependent switching is large and
  detectable with both a GP and the PFN. It is not problem-dependent: a per-function static arm chosen on held-out
  instances gains nothing. Off the prior the GP is the better surrogate at d = 5 (its B\* is 25.5% [11.8, 37.3]
  better than the PFN's), so the headroom is not an artefact of a misspecified surrogate.
  The open question moves from "is there headroom?" to "can it be found without task knowledge?". In-prior the belief
  controller lost to B\* (#6).
- **Next.** #6 (belief value on the latent f; resolution-aware metric), then a belief controller off the prior
  (GP posterior sample paths need no rank-scale inverse). #9 (noise) and #7 (GP on the prior) stay open but are less
  urgent: headroom already exists without noise and with either surrogate.

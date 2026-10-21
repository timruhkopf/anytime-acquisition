---
experiment: static_tune
sweeps: []
---
# Which static acquisition arms are competitive with pfn_large, per d? (selects the switching portfolio)

## Question

See the description at the top of `notebooks/01_2026-10-21_static_tune.ipynb`; not retyped here.

## §1 Timing probe and numerics (2026-10-21, 14:50)

Scratch probe on ulysses (RTX 3090), `pfn_large`, 1216 candidates per lane, context n = 10d, 3 steps.
Throughput of `Engine.propose` (lane-steps per second):

| d | arm | 64 lanes | 2048 lanes |
|---|---|---|---|
| 2 | ei | 445 | 600 |
| 2 | es (alpha-PFN, two models) | 305 | 302 |
| 6 | ei | 530 | 516 |
| 6 | es | 266 | 261 |

The GPU is saturated from ~64 lanes; the cost is the 1216 candidate tokens per lane. Variants at d = 6,
2048 lanes, compared with fp32 on the same states:

| Numerics | lane-steps/s | same choice as fp32 |
|---|---|---|
| fp32 | 512 | — |
| TF32 matmuls | 817 | 93.4% |
| bf16 autocast | 1105 | 65.0% |
| fp32, 608 candidates | 961 | (different candidate set) |

**Decision: fp32.** bf16 changes a third of the choices and is out. TF32 flips 6.6% of choices among
near-tied candidates. Under common random numbers a flip doesn't bias a comparison, but it makes
trajectories diverge for reasons unrelated to the arm, which inflates the per-task variance of paired
differences. The predecessor switched its v2 runs to fp32 for the same reason (pfn-acqpolicy
`2026-10-15` log, near-tie review). The cost is ~1.6× GPU time. The candidate set stays at
1024 + 3 × 64 (the predecessor's standard S9).

**Sizing (fp32, ~500 lane-steps/s, es arms at half that).** A static arm over 200 tasks costs
200 · 10d lane-steps, i.e. a few minutes for all 13 arms at d = 6. The switching controller costs
B · K · R · T(J+1)/2 lane-steps: at d = 6 with B = 200, K = 6, R = 8 that is 1.7 M, ≈ 1 h per
controller.

## §2 Sweeps that were stopped (2026-10-21, 15:00–15:04)

- `pilot_20261021-150017` (commit 313e2ec, TF32 on): stopped by me after member 0 (d = 2) was DONE,
  once I decided on fp32 (§1). Not used.
- `pilot_20261021-150405`: launched by mistake from commit 313e2ec (TF32 on). My commit of the fp32
  change had failed a lint hook, and the launch was chained after it with `;` instead of `&&`.
  Stopped within a minute (exact PIDs). Not used.

Both sweep dirs stay on ulysses, and their tags exist. The tuning run is `pilot_20261021-150441`
(commit f234370, fp32).

## §3 Pilot: selection (2026-10-21)

What ran, the result (numbers from the snapshot), interpretation, next step.

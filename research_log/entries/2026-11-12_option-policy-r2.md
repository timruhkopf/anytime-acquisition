---
experiment: option_policy_r2
sweeps: []
---
# Round 2 of policy iteration: one context-dependent deviation from the UCB schedule

## Question

See the description at the top of `notebooks/14_2026-11-12_option_policy_r2.ipynb`; not retyped here.

## §1 Sanity (2026-11-12)

**What ran.** 100 tasks on ulysses, 15:50–15:57 (`results/option_policy_r2/sanity_20261112-155053`, code 639fae6). Collection took 6.1 min, the evaluation 7 s.

**Result** (a pipeline check; 100 tasks are far too few for the decision):
- The pipeline ran end to end with UCB as the base: 8 options (EI, PI, MES, mean; 3 steps or the rest), no missing values, the `full:*` columns give EI and UCB on the same tasks.
- Mean advantage over UCB by step (seed 0, `options` table): leaving UCB early costs (EI for the rest −0.30 at t = 0 and 2); from t = 8 on it is at or above zero (+0.10 at t = 12).
- Privileged one-deviation headroom from UCB: 0.21 [0.08, 0.35] without MES.

**Revision before the pilot:** none.

**Next.** The pilot as pre-registered (4000 tasks), after the ei_failure pilot.

## §2 Pilot (2026-11-14)

**What ran.** 4000 split-4 tasks on ulysses, 2026-11-13 09:04 to about 16:00, GPU shared with other runs from 11:30 (`results/option_policy_r2/pilot_20261113-090450`; 1 member, DONE).

**Result** (log10 terminal regret, ε = 1e-6, positive = better; held-out tasks, seed 0, 95% bootstrap CI; `summary` table and notebook 14's cell outputs):
- Primary policy `nomes/belief/mlp` over UCB: +0.090 [0.072, 0.110]; over the best static schedule from UCB: **−0.023 [−0.043, −0.003]**; seed 1: −0.017 [−0.040, 0.005]. Over its shuffled null: +0.087 [0.072, 0.103].
- Static schedule (UCB, then EI for the rest from the last decision step) over UCB: +0.113 [0.095, 0.134].
- Against EI on the same tasks (notebook 14 §4b): UCB +0.082 [0.046, 0.114]; static MES +0.121 [0.090, 0.153]; **UCB + static switch to EI +0.195 [0.164, 0.226]** (seed 1: +0.200 [0.165, 0.235]); round-2 policy +0.172 [0.140, 0.203].
- No variant beats static; best `all/belief/mlp` −0.013 [−0.033, 0.006].
- Privileged one-deviation headroom from UCB: 0.255 [0.232, 0.278] without MES, 0.260 [0.237, 0.283] with.
- Held-out fit of Q (`fit` table): R² 0.057 and Spearman 0.17 for `belief`/MLP.

**Interpretation.**
- Same outcome as round 1, one level up: the context is used (shuffled null) and the policy still loses to one fixed schedule.
- The schedule itself is the gain: explore with UCB, exploit with EI at the end, 36% less regret than EI and clearly more than any single arm.
- The policy switches to EI too early; per-decision labels are too noisy to learn when not to.

**Verdict: NO-GO** (pre-registered rule; `just decide` not yet run).

**Next.** No further rounds of one-deviation policy improvement. The budget-dependent schedule is what mixture_policy (notebook 17) learns directly.

# anytime-acquisition

Amortized acquisition control for Bayesian optimization: does switching between acquisition functions, conditioned on a frozen PFN's in-context belief, pay off, and how much could a policy amortized over the prior gain?

@docs/agent_rules.md

## Current phase

Phase 0: measure the value of switching before building a policy (README § Status).
- 0a `static_tune` (#3): select B* and the portfolio on tuning split 0.
- 0b `switching` (#1): non-anticipating rollout controller vs B* on fresh tasks; decision rule in
  `analysis/bo.py::switching_verdict`, pre-registered in notebook 02.
- 0c `ei_readout` (#2): PFN-VLA action expert learning EI; PFN-reading vs D-only.
Gate: no Phase-1 policy code unless 0b is GO.

## Project rules

- **pfn-repro is a dependency, not a workspace.** Never copy or patch its code here; change it in
  `../pfn-repro` (issue, branch, release a tag), then bump the tag (README § The PFN dependency).
- **Frozen PFNs, valid ranges.** `pfn_large` / alpha models: context n <= 100, d <= 6; `pfn_smoke`:
  n <= 20 (tests only). Runs outside these are separate `<name>_robust` experiments.
- **Minimization** everywhere; y on the training prior's ECDF scale [0, 1]; always pass `n_features`.
- **Tasks are prior draws identified by (split, d, i)** (`bo/tasks.py::task_seed`): split 0 tuning,
  1 evaluation, 2 pilot. Never select anything on split 1.
- **Common random numbers**: all nuisance comes from `bo/nuisance.py` keyed by (kind, step, replicate);
  replicate 0 is the evaluated trajectory, rollouts use ids > 0. CRN holds within one task list (the
  stream is indexed by row), so compare only members with the same `split`, `n_tasks`, `task_offset`.
- **fp32** for every BO run (`task.tf32=false`): TF32 flips near-tie choices (static_tune log §1).
- Every comparison is paired over tasks and reported with a null control and bootstrap CIs.

## Commands

`just` lists all recipes. Common: `just test`, `just lint`, `just smoke`, `just sweep <backend> <stage> experiment=<name> ...`,
`just sweep-status <dir>`, `just snapshot <dir>`, `just decide <dir> <verdict> "<reason>"`.
Process and conventions: `docs/process.md`. Machine notes: `docs/compute.local.md` (untracked).
GPU box: ulysses (`ANYTIME_ACQ_REMOTE_HOST`); `just push` then `just remote sweep detached <stage> ...`;
verify with `just remote-ps`; kill exact PIDs only (never `pkill -f`).

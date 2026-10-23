# anytime-acquisition

<!-- Project-facing README: this is what a reviewer opens from your CV (~60 s).
     The research process lives in docs/process.md (owned by the template, updated by `copier update`). -->

[![CI](https://github.com/timruhkopf/anytime-acquisition/actions/workflows/ci.yml/badge.svg)](https://github.com/timruhkopf/anytime-acquisition/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Can a policy amortized over the prior pick the right acquisition function at each step of a Bayesian
> optimization run, and how much would that be worth?** This repo measures the second question cleanly
> before anyone builds the first.

**Status:** Phase 0 (measure the value of switching) is running. See [Status](#status). **How we work:**
[docs/process.md](docs/process.md) · **Lab book and decisions:** [research_log/](research_log/README.md) ·
**Literature:** [references.md](references.md)

## The idea

Bayesian optimization (BO) picks the next point by maximizing an *acquisition function*: EI, PI, UCB,
entropy search, or the posterior mean. Which of them works best depends on the problem, the budget and
the stage of the run: explore early, refine late. In practice the choice is a hyperparameter fixed by
hand. Portfolios and hedging (GP-Hedge), and recent LLM- and RL-based selectors, switch between them
heuristically.

A **prior-data fitted network (PFN)** is a transformer trained on functions sampled from a prior. Given
the observations so far in its context, it returns the posterior predictive at any query point in one
forward pass. That makes two things cheap that used to be expensive:

1. **Simulating BO runs from the prior.** Tasks drawn from the PFN's training prior are, by
   construction, problems on which the PFN is the Bayes posterior. BO itself is then a POMDP whose
   belief state the PFN computes.
2. **Reading that belief state.** The PFN's per-layer keys and values over the context *are* its
   summary of the data.

So the **amortized acquisition policy** is π(arm | PFN state, budget left), trained over the prior: the
same prior the PFN was trained on, so the policy is matched to the belief it reads. The hoped-for
benefits:

- **No per-problem tuning** of the acquisition function or its exploration parameter.
- **Non-myopic behaviour for free.** A policy trained on whole runs can learn to trade exploration
  against the remaining budget, which a fixed one-step criterion cannot.
- **Amortized cost.** One forward pass of a small head per decision, instead of an inner optimization
  loop or lookahead rollouts at test time.
- **Bayes-consistency.** Trained on the prior the PFN represents, the policy is optimal *for that
  prior*, which is a well-defined target. The open question is how much better than a single tuned
  arm that optimum is.

### The PFN-VLA perspective

Vision-language-action (VLA) models put a small **action expert** on a frozen (or lightly tuned)
foundation model. The expert cross-attends to the backbone's per-layer keys and values and outputs
actions (π0 is the template). The analogue here:
- the frozen PFN is the backbone, a "world model" of the objective given the data;
- the action is the next query point (or the choice of arm);
- the action expert cross-attends to the PFN's context K/V.

The cheapest test of that architecture is to make it **learn a known acquisition function**: amortized
EI, a one-shot readout of EI's maximizer from the PFN state, with no candidate scoring
([issue #2](https://github.com/timruhkopf/anytime-acquisition/issues/2)). The predecessor found, on a
small PFN, that the readout works only once the expert also sees the PFN's prediction *at query
points* (a probe set). A matched encoder trained from scratch (D-only) did as well as reading the
frozen PFN. This repo repeats the key comparison (PFN-reading vs D-only) on the long-trained
`pfn_large`.

## What is measured, and how

Switching can only pay if some *state-dependent* choice of arms beats the best single arm. The
predecessor ([pfn-acqpolicy](https://github.com/timruhkopf/pfn-acqpolicy)) estimated this with a
hindsight oracle (the best of 81 arm sequences per task). That number turned out to be mostly
selection luck: a null tree of identical arms already reached 81–95% of it. The design here removes the
luck by construction ([notebook 02](notebooks/02_2026-10-21_switching.ipynb)):

| | |
|---|---|
| **Baseline** | the tuned best static arm B\*, selected on separate tasks ([notebook 01](notebooks/01_2026-10-21_static_tune.ipynb)) and re-measured on fresh ones |
| **Controller** | every 2d steps, roll each of K = 6 arms forward (then B\* to the horizon) R = 8 times on **fresh** randomness, and execute the best. Non-anticipating, so its gain is unbiased; it is one-step policy improvement over B\* |
| **privileged** rollouts | query the true function: switching with task knowledge, an upper reference |
| **belief** rollouts | sample observations from the PFN's posterior predictive: no task knowledge. This is what a policy amortized over the prior could learn |
| **Null control** | the same controller over 6 copies of B\* (random candidate halves): its gain must be 0 |
| **Statistics** | paired over tasks (common random numbers), geometric-mean regret ratio, bootstrap CIs, power from a pilot; decision rule fixed before the run |

## Status

| Phase | Question | Verdict | Evidence |
|---|---|---|---|
| 0a | Which static arms are competitive with `pfn_large`, per d? | _running_ | [notebook 01](notebooks/01_2026-10-21_static_tune.ipynb) |
| 0b | How much does non-anticipating switching gain over B\*? | _pre-registered_ | [notebook 02](notebooks/02_2026-10-21_switching.ipynb) |
| 0c | Does an action expert reading `pfn_large` learn EI better than one trained from scratch? | _planned_ | [issue #2](https://github.com/timruhkopf/anytime-acquisition/issues/2) |
| 0d | Is there luck-corrected switching headroom off the prior (BBOB, GP and PFN)? | **GO** (both): privileged +59/+19% (GP), +66/+29% (PFN) at d = 2/5, null ≈ 0; per-function static choice ≈ 0 | [notebook 04](notebooks/04_2026-10-22_bbob_switching.ipynb), [issue #8](https://github.com/timruhkopf/anytime-acquisition/issues/8) |
| 1 | Train the amortized policy | only if 0b is GO | — |

**Prior evidence (predecessor, pfn-acqpolicy).** No robust gain from switching. The upper bound was
about 5%. The one consistent effect is EI under-exploring at d ≥ 4, which a tuned static arm already
fixes. This repo is the sound re-measurement of that bound.

## Quickstart

```bash
git clone https://github.com/timruhkopf/anytime-acquisition.git && cd anytime-acquisition
just setup cpu local     # uv sync --extra cpu (pulls pfn-repro at its pinned tag), hooks, .env
just test                # unit tests run on the CPU with the small PFN bundled in pfn-repro
uv run anytime_acq experiment=static_tune task.device=cpu task.pfn=pfn_smoke task.n_tasks=4 \
  task.d=2 'task.arms=[ei,ucb0.9]'   # a scratch BO run in under a minute
```

The reported runs use `pfn_large` (76 MB, not in git; see below). Reproduce a reported result:

```bash
just reproduce results/<experiment>/<sweep_id>   # worktree at the sweep's tag, uv sync --frozen, rerun, compare
```

Requirements: [uv](https://docs.astral.sh/uv/), [just](https://github.com/casey/just). Always pass a torch extra
(`cpu` or `cu126`); see [docs/process.md § Environment](docs/process.md#environment).

## The PFN dependency: pfn-repro

The PFNs, the BNN prior, the bar distribution and the standard acquisition functions live in a
separate library, [pfn-repro](https://github.com/timruhkopf/pfn-repro). It holds frozen code, plus
checkpoints pinned by sha256 and golden-output tests. This repo never copies its code. Its
`docs/dependency_guide.md` weighs the options. What this repo does:

- **Default (every tracked run): a git tag.** `pyproject.toml` declares
  `pfn-repro = { git = "https://github.com/timruhkopf/pfn-repro", tag = "v0.1.0" }`, and `uv.lock` pins the
  commit. A commit of this repo therefore pins pfn-repro too, and provenance needs nothing extra.
- **Developing both at once:** on a branch of this repo, `uv add --editable ../pfn-repro`. Tracked
  runs record the editable checkout's commit and refuse uncommitted changes in it, and `confirm` /
  `final` refuse editable dependencies altogether. Before reporting, tag a pfn-repro release and switch
  back with `uv add "pfn-repro @ git+https://github.com/timruhkopf/pfn-repro" --tag vX.Y.Z`.
- **Checkpoints** are not in the package (except the smoke PFN and the ECDF). Point
  `PFN_REPRO_CKPT_ROOT` in `.env` at the directory that holds `pfn_large.pt` etc.
  `python -c "from pfn_repro.checkpoints import available; print(available())"` shows what is found.
- **A GPU box without GitHub credentials** (ulysses): git rewrites the URL to a local clone there,
  `git config --global url."/home/<user>/PycharmProjects/pfn-repro".insteadOf https://github.com/timruhkopf/pfn-repro`.
  That clone needs the tag: `git push <box> v0.1.0` from the laptop.

**PyCharm.** Open this repo as the project with `.venv` as its interpreter. Then attach pfn-repro
(*File → Open → pfn-repro → Attach*). That gives navigation, refactoring and search across both repos,
while the interpreter keeps importing the *locked* version from `.venv`. Navigation from this repo
into pfn-repro therefore lands in `.venv/.../site-packages/pfn_repro` unless the editable install is
active. To edit pfn-repro, open its own files (the attached project), commit there and release a tag.
Mark `src/` as *Sources Root*. Notebooks use the same interpreter.

## Repository map

| Path | What |
|---|---|
| `src/anytime_acq/` | the package; one entry point `anytime_acq` (Hydra), configs ship inside it |
| `src/anytime_acq/bo/` | vectorized BO with the frozen PFN: tasks from the prior, common random numbers, engine, rollout controller |
| `src/anytime_acq/tasks/bo.py` | the BO pipeline (static arms, random search, controllers, null control) |
| `src/anytime_acq/analysis/bo.py` | paired gains, power, and the pre-registered selection and decision rules |
| `src/anytime_acq/configs/experiment/` | one file per research question (config only) |
| `research_log/` | lab book: entries with decisions, append-only track record, design decisions (backlog: GitHub issues) |
| `results/` | committed snapshots of finished sweeps; notebooks and figures read only these |
| `notebooks/` | executed analyses, outputs committed |
| `references.md` / `.bib` | annotated literature, keyed to BibTeX |

## Citation

```bibtex
@misc{ruhkopf2026anytime_acq,
  title  = {...},
  author = {...},
  year   = {2026},
  url    = {https://github.com/timruhkopf/anytime-acquisition}
}
```

## License and acknowledgements

MIT, see [LICENSE](LICENSE). Datasets are subject to their own licenses (`data/README.md`).
Funding, compute resources, and code this project builds on.

# Track record

**Append-only.** New entries go at the bottom; old entries are never edited, a later entry corrects an
earlier one explicitly. Times are local. Each entry names the commit it refers to, what was done, what
was **verified and how**, and numbers with the snapshot or artifact they come from.

Autonomous agent sessions start with their mandate (goal, deadline, which launches are allowed) and end
with deliverables and open questions.

## Entries

### 2026-10-21 14:45 — Mandate and setup (e1799cd)

Mandate (Tim, 2026-10-21): rebuild the anytime-acquisition study from the template.
- Make pfn-repro the dependency.
- Communicate the idea, its benefits and the PFN-VLA perspective.
- Run an amortized-EI readout experiment.
- Above all, re-plan and re-run the switching experiment soundly. Notebooks for every run, a lab book, branches, provenance, MLflow.
- Runs on ulysses. The GPU is reserved for the pfn-repro TabPFN pilot from 20:00 to 07:30.

- The repo was generated from research-template v0.4.0, with pfn-repro v0.1.0 as a git-tag dependency (`uv.lock` → e3574b6). Decision 0002 records what carries over from pfn-acqpolicy and AnytimeAcquisition.
- ulysses: `just remote-setup` (cu126). `available()` shows all four checkpoints; CUDA works. pfn-repro resolves through `url.insteadOf` to the local clone on ulysses.
- Timing probe and numerics decision (fp32): static_tune log §1.
- Template gap found and fixed: scratch runs didn't load `.env`. Fixed here (313e2ec) and in research-template v0.4.1.

### 2026-10-21 15:00–15:30 — static_tune decided GO; switching pre-registered and piloted

- Two sweeps were stopped and not used (static_tune log §2). One of them was my mistake: a launch chained with `;` ran after a failed commit, on a TF32 commit.
- `pilot_20261021-150441` (pre-registered grid) and `pilot_20261021-151907` (extension past the grid edge, log §4). Both 3/3 DONE, both decided GO. B\* = mes / ucb0.995 / ucb0.995 at d = 2 / 4 / 6. Verified: notebook 01 executed and figures checked against the tables. Random search is 17–24× worse than B\* at every d.
- switching: decision rule revised before any switching run. NO-GO now needs the privileged CI to exclude a 10% gain, because the per-task log-ratio sd is ≈ 0.55 (notebook 02, revision note). Rollouts use 608 candidates; a test shows this leaves the evaluated trajectory unchanged when the choice cannot matter.
- PR #4 squash-merged (closes #3). The switching pilot `pilot_20261021-152818` was launched from d7c6d6f (12 members, split 2, 48 tasks/d).
- ei_readout (branch exp/2-ei-readout, 3cd9f3c): expert, data and pipeline, with 7 tests (normalized density, frozen PFN, padding invariance, targets, end to end). GPU scratch smoke passed. Pilot `pilot_20261021-153316` launched for the trunks pfn_all, d_only and pfn_last; the probe variant is deferred to tomorrow's GPU window.

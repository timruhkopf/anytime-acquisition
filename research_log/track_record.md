# Track record

**Append-only.** New entries go at the bottom; old entries are never edited, a later entry corrects an
earlier one explicitly. Times are local. Each entry names the commit it refers to, what was done, what
was **verified and how**, and numbers with the snapshot or artifact they come from.

Autonomous agent sessions start with their mandate (goal, deadline, which launches are allowed) and end
with deliverables and open questions.

## Entries

### 2026-10-22 14:30 — Mandate (autonomous): BBOB switching with GP and PFN (issue #8)

Mandate (Tim, 2026-10-22, ~13:30–15:00): run the luck-controlled switching design on BBOB with a GP and with the
PFN on ulysses, then write a notebook assessing the outcome. The question: is luck-corrected headroom for
switching still undetectable? Also: what does SAWEI do? Use ulysses without asking whenever it is idle.
"Finish and report by tomorrow 7am", no questions. Allowed launches: the bbob_switching pilot and confirm sweeps.
Earlier the same day: issues #6–#9 filed; pfn-repro#3 filed with the noise rationale; the repo pushed (exp/1, exp/2, sweep tags).

### 2026-10-22 14:45–16:01 — Implementation, selection pilot GO, GP confirm (fbc7d0c, 135eb56, 63900a2)

- Code: BBOB tasks (`ioh`), a batched MAP-II GP and a rank-transformed PFN behind `Engine.score`.
  Verified: the refactor is bit-identical on the prior (all 4 methods, pfn_smoke); 30 unit tests pass; pyright 35
  errors vs 36 before (all pre-existing).
- Selection pilot `pilot_20261022-144551`: 4/4 DONE, decided GO. B\* = ucb0.9 (GP), ucb0.99 (PFN) at d = 2 and 5.
- GP confirm `confirm_20261022-145708` launched 14:57, 6/6 DONE at 16:00 (queue.log).

### 2026-10-23 07:58 — Late: PFN confirm launched only now; GP confirm decided GO

- The session stopped after the GP sweep finished, so the PFN confirm sweep was not launched yesterday and the 07:00
  deadline was missed. This is my failure, not a compute limit: ulysses was idle all night (nvidia-smi 0%,
  4 MiB at 07:58).
- PFN confirm `confirm_20261023-075829` launched at 07:58 (pid 1272603, verified with ps).
- GP confirm snapshot `results/bbob_switching/confirm_20261022-145708`, decided GO. Privileged gain over B\*
  (n = 192/d): +58.6% [+45.9, +68.8] at d = 2, +18.8% [+3.1, +31.9] at d = 5. Null: −5.0% [−43.5, +22.0] and
  −5.7% [−20.9, +7.7].

### 2026-10-23 09:30 — PFN confirm DONE and decided GO; notebook 04 executed (deliverables)

- PFN confirm `confirm_20261023-075829`: 6/6 DONE at 09:05 (queue.log). Snapshotted and decided GO.
  Privileged +65.6% [+51.3, +76.1] (d = 2) and +29.2% [+17.8, +38.9] (d = 5); null −1.4% and −1.9%, CIs include 0.
- Notebook 04 executed against the three snapshots (`just notebook`). Verified: every cell ran without error; the two
  figures were checked against the tables; every number in the Observations was re-read from the cell outputs.
- **Deliverables:**
  - issues #6–#9, pfn-repro#3 (with the noise rationale);
  - notebook 04, log entry `2026-10-22_bbob-switching.md` §1–§2, a README status row;
  - references: SAWEI, HEBO, decoupled PFNs, martingale posteriors.
- **Open questions:**
  - Can a belief controller (no task knowledge) recover part of the headroom (#6)?
  - Does the headroom survive at a larger budget (SAWEI uses 35d) and with noise (#9)?
  - The PFN at d = 2: the top arm doesn't transfer from split 0 to split 1 (ucb0.9 beats the selected ucb0.99 by
    28%), so B\* selection on 48 tasks is noisy.

### 2026-11-11 13:30 — Mandate: amortized policy improvement, round 1 (issue #13)

Mandate (Tim, 2026-11-11): a new branch and worktree (another session works in this repository); add the design
document "Amortized policy improvement for PFN-based BO" to the research log; implement and run its first stage on
ulysses if the GPU has room; report with a notebook comparing the learned first policy-iteration result against EI,
with the conservative update rule in place; open the notebook in PyCharm. Added during the session: assess the
proposal to fine-tune the PFN on BO trajectories as a separate logged document, without acting on it.
Allowed launches: api_round1 sanity and pilot on ulysses (GPU checked idle: 0%, 4 MiB, 13:20).
Peers asked first: two sessions are in another checkout (AnytimeAcquisition, branch pfn-native-acquisition), one in
the anytime-acquisition-hedge worktree with nothing running.

### 2026-11-11 13:27 — Design logged, round 1 implemented, sanity launched (a929cf2, 171a48b)

- Worktree `../anytime-acquisition-api`, branch `exp/13-api-round1` from `feat/11-gp-hedge` (main lacks the engine).
- Design: `decisions/0003` (verbatim + provenance), `decisions/0004` (assessment of the PFN fine-tuning idea).
- Code: experimental/api.py, tasks/api.py, analysis/api.py; notebook 06 description (pre-registration) before any run.
  Scope: "first stage" = round 1 of 0003 (labels → head → conservative π₁ → calibrated c → test vs EI) on the frozen
  noise-free `pfn_large`; 0003's Stage 0 (two-headed PFN) needs pfn-repro work and is out of scope.
- Verified: 9 new tests (query path reproduces the PFN logits; x₀ equals Engine.propose's EI pick; anchoring;
  the rule; c = ∞ reproduces EI; paired labels; pipeline end to end); full suite 92 passed; ruff clean; no pyright
  errors in the new non-experimental files.
- Sanity sweep `runs/api_round1/sanity_20261111-132704` launched on ulysses at 13:27 (pid 1759250, verified with ps).

### 2026-11-11 13:40 — Sanity DONE, c grid extended, pilot launched; a git incident from the pre-push hook (c0f51e0)

- Sanity `sanity_20261111-132704`: 1/1 DONE in ~5 min (log entry §1). Change before the pilot: c grid {0, …, 4}.
- Pilot `runs/api_round1/pilot_20261111-133311` (d = 2, 4000 label states) launched on ulysses at 13:33 (pid 1759943,
  verified with ps at 13:36).
- **Incident.** `git push` from this linked worktree ran the pre-push hook (pytest). The template's provenance tests
  ran their throwaway-repo git commands with the hook's inherited `GIT_*` environment and wrote to this repository:
  - the worktree's index was emptied (files intact);
  - the shared `.git/config` got `core.bare = true` and a `[user] test <test@example.com>` section.
  This broke the main checkout between about 13:34 and 13:36.
  Repaired: `core.bare = false`, the `[user]` section removed, `git reset` of this worktree's index. The damaged config
  is backed up in the session scratchpad.
  Verified:
  - HEAD, branches and tags are unchanged (reflog, for-each-ref);
  - every commit carries the real author;
  - the main checkout and the hedge worktree show their previous status;
  - `user.email` resolves from `~/.gitconfig`.
  The push was redone with `--no-verify`; the full suite had passed when run normally (92 passed).
  Fix belongs in research-template: the template tests must clear the `GIT_*` variables.

### 2026-11-11 16:10 — Network outage; pilot DONE, assessed, decided NO-GO (f786313)

- ulysses was unreachable from about 14:30 (ssh banner timeouts, no ping). Tim paused all work and talked to the
  admin. The box did not reboot (uptime 77 days at 15:58); the pilot ran through the outage, 1/1 DONE.
- Pulled, snapshotted (`results/api_round1/pilot_20261111-133311`), notebook 06 executed; log §2; `decide` NO_GO.
  Verified: every cell ran without error, and every number in the Observations was re-read from the cell outputs or
  from a recomputation on the same snapshot.
- Result:
  - clairvoyant cross-fitted headroom 0.70 [0.57, 0.84] % of the gap per state;
  - the head has no ranking skill (val Spearman −0.01);
  - π₁ at c\* = 1: −0.8% [−2.5%, +0.2%]; at c = 0: +448%; the null head: 0.
- Cancelled at Tim's request: the 16:00 job (0006 Phase A). It resumes when Tim says so; readout probes come first.

### 2026-11-11 16:20 — Mandate (autonomous, overnight): progress the api experiments

Mandate (Tim, 2026-11-11 ~16:15): "make it work over night and progress the experiments". Priority, as Tim stated:
1. the readout experiments (0006 §1 probes);
2. then policy iteration (0006 value decoder, Phase A noise-free);
3. external baselines (GP-Hedge, ZeroShotOpt) only later.

Planned:
- `api_gate`: the model-based gate of 0003 §8.3 at the pilot's states, plus the model-based rollout policy against EI.
  It needs no learning and answers whether a deployable lookahead finds the headroom.
- `api_probe`: the 0006 §1 probes.
- Then the 0006 decoder.

Allowed launches: sanity and pilot sweeps of these experiments on ulysses while it is idle. Each experiment is
pre-registered in its notebook before its first run. Push from the worktree with --no-verify after running the
tests (template hook incident).

### 2026-11-11 16:27 — api_gate and api_probe implemented, pre-registered; sanity runs on ulysses (4f4c750)

- `api_gate` (eaffd7a) and `api_probe` (9e3c342, 4f4c750): code, configs, notebooks 07 and 08 with their
  descriptions, log entries. Verified: 103 unit tests pass (5 gate tests, 4 probe tests, each including the pipeline
  end to end on a tiny label set); ruff clean. The pyright errors that remain are in files this branch does not touch.
- A local CPU sanity of `api_probe` (60 states) showed heavy-tailed state targets: the gap has a median of 0.17 %
  and a maximum of 182 %; the headroom reaches 18. Raw-scale R² was meaningless. Revised before the pilot: log gap,
  asinh headroom, asinh labels, plus a Spearman metric. Recorded in notebook 08; the decision rule is unchanged.
- ulysses reachable again about 16:25. Its labels.pt sha256 matches the pin (c27d8939…). GPU idle. Launched:
  - `api_gate` sanity (100 states, 20 tasks), pid 1764119;
  - `api_probe` sanity (300 states), pid 1764222.

### 2026-11-11 16:50 — Sanity runs done; pilots of api_probe and api_gate running, api_decoder queued (4274e77)

- `api_decoder` (0006 Phase A) implemented: 1c54997, with notebook 09 pre-registered.
  - 5 unit tests, including query independence (a candidate's score does not depend on the other candidates), the
    density of the bar distribution, and the EI lanes equal π₀.
- Sanity on ulysses, all DONE:
  - `api_gate`: 100 states, 20 tasks;
  - `api_probe`: 300 states;
  - `api_decoder`: 4200 states, 30 + 30 tasks.
  - Snapshots are in results/, and the analysis cells of notebooks 07 and 08 execute on them.
- Revisions before the pilots, each recorded in its notebook with the reason:
  - probes: log or asinh targets plus Spearman (heavy tails);
  - probes: chunked roughness and normal equations, after the first pilot launch failed from GPU memory pressure
    next to two other runs (5ce73d2);
  - decoder: signed-log bar borders (4274e77).
- Running:
  - `api_probe` pilot (`pilot_20261111-163954`);
  - `api_gate` pilot (`pilot_20261111-164326`, pid 1766111).
- Queued: the `api_decoder` pilot, launched by a waiter once the probe pilot exits.

### 2026-11-11 17:05 — api_probe pilot decided INCONCLUSIVE (7e45c42)

- Snapshot `results/api_probe/pilot_20261111-163954` (1/1 DONE). Notebook 08 executed; Observations written;
  log §2; `decide` INCONCLUSIVE. Verified: every number in the Observations and in §2 was re-read from the cell
  outputs.
- Readable from PFN states, with all layers better than the last:
  - x\*: R² 0.53 (all), 0.45 (last), 0.00 (hand);
  - n_relevant, roughness.
- Not readable from any input: the advantage label (within-state Spearman ≤ 0.025; label reliability 0.57) and
  headroom_bin (best AUC 0.68 from hand features).
- Still running on ulysses:
  - `api_gate` pilot (pid 1766111);
  - `api_decoder` pilot (pid 1766691; launched 16:53 after the probe pilot freed GPU memory).

### 2026-11-11 20:55 — api_gate pilot decided INCONCLUSIVE (33190b6)

- Snapshot `results/api_gate/pilot_20261111-164326` (1/1 DONE). Notebook 07 executed; Observations written; log §2; `decide` INCONCLUSIVE.
- Stage 1b: mb − EI is +0.028 [−0.58, 0.74], relative +1.9% [−38%, +49%] over 200 tasks. The CI is too wide to decide.
- Stage 1a: G is 0.080 [−0.025, 0.19], against 0.035 for random deviation. The within-state Spearman between the model advantage and the label is 0.042.
- Conclusion: the PFN's belief barely ranks candidates, which agrees with the probes. Stage 1b is not scaled.
- Still running: the `api_decoder` pilot (pid 1766691).

### 2026-11-11 21:05 — api_probe addendum: direct scores (fc305ea, snapshot pilot_20261111-204422)

- A `direct` table was added and pre-registered in notebook 08, then the pilot was rerun. The original probes reproduce exactly (max |Δ| = 0).
- Within-state Spearman with the label: log EI and PI 0.063, KG 0.048, every |ρ| ≤ 0.075.
- Taken with api_gate §2 (0.042) and the probes (≤ 0.025): the headroom over EI is not in the PFN's belief.

### 2026-11-11 22:05 — api_decoder pilot decided NO-GO (e5a9918); session deliverables and open questions

- Snapshot `results/api_decoder/pilot_20261111-165340` (1/1 DONE). Notebook 09 executed; Observations written; log §2; `decide` NO-GO.
  - rule\* q0.5 deviates at 0.2% of steps.
  - Δ vs EI is −0.1% [−1.1%, +0.9%].
  - Validation ranking is 0.060, the same as untrained log EI (0.063).
- Verified: the EI trajectories are identical across rounds 1 and 2 (max |Δ| 3e-7 over 1000 tasks). Every number above was re-read from the cell outputs.

**Deliverables (branch exp/13-api-round1).** Each notebook has its pre-registration, Observations and verdict.

| Notebook | Experiment | Verdict | Snapshot(s) |
|---|---|---|---|
| 07 | api_gate | INCONCLUSIVE | `pilot_20261111-164326` |
| 08 | api_probe | INCONCLUSIVE, plus the direct-score addendum | `pilot_20261111-163954`, `pilot_20261111-204422` |
| 09 | api_decoder | NO-GO | `pilot_20261111-165340` |

**Overall.** On noise-free P0 (d = 2, T = 20) with the frozen pfn_large:
- the one-step advantage over EI's pick is not predictable from the PFN's belief;
- trained readouts reach ≤ 0.025, the PFN lookahead 0.042, KG 0.048, log EI 0.063, and the decoder 0.060, against a label reliability of 0.57.
- Policy improvement over EI from this belief, as set up here, has nothing to learn.

**Open questions (for the user; these are design decisions, so none were launched).**
1. Is the clairvoyant headroom PFN error, which would make fine-tuning on BO contexts worthwhile (0004/0005 E1)? Or is it irreducible outcome luck? A test: compute labels under the PFN's own predictive (belief labels) and compare their spread with the ground-truth labels.
2. Should the label change from one-step-deviation advantages to multi-step or terminal returns? The policy-iteration loop would then not hinge on a one-step improvement over EI.
3. Would a noisy prior or a higher d (Phase B; the noisy PFN lives in pfn-repro) open headroom that the belief can see?
4. Should external baselines (ZeroShotOpt retrained on our prior, GP-Hedge) wait until one of 1–3 gives a positive signal?

### 2026-11-12 — correction: the gate's lookahead was noise-dominated; decision 0007 (explanations and tests)

- This corrects the 2026-11-11 22:05 entry and api_gate §2. Scratch checks on the pinned labels and the gate's `gate_states`, reproduced in the next experiment:
  - in matching units (% of the gap), the PFN-lookahead advantage has sd 14.2, with a Monte-Carlo noise sd of 20.1 at M = 4;
  - its estimated signal is ≤ 0 and its calibration slope is −0.001 [−0.005, 0.001].
- The gate therefore did not measure the posterior-expected advantage, and "the headroom is not in the belief" is not established by it.
- The gate's decisions are unaffected by the units: they are invariant to a positive per-state scale.
- Resolution:
  - pfn_large has 1000 uniform bins of width 0.001;
  - in 57% of labeled states the incumbent is within one bin of f\*;
  - 91% of the labeled advantages are smaller than one bin.
- The explanations (E1 irreducible, E2 PFN error, E3 resolution, E4 label metric) and the tests that tell them apart: `research_log/decisions/0007-why-the-headroom-is-not-in-the-belief.md`.

### 2026-11-12 08:50 — pfn_bo_inference (0007 tests 1–2) pilot decided GO, narrowly (80fce6b)

- New experiment `pfn_bo_inference`:
  - pre-registered in notebook 10 (8f9ba21);
  - the sanity run exposed two design errors (the control's points were not ignorable; the score used f instead of the PFN's noisy target y). Both were revised before the pilot (03d8de2).
  - The pilot was repeated for the repo's file-size limit (3c475a7) and reproduces exactly.
- Snapshot `results/pfn_bo_inference/pilot_20261112-084114` (1/1 DONE). Notebook 10 executed; Observations written; log §2; `decide` GO.
- Test 2, E2: GO by the pre-registered rule, narrowly.
  - The trigger is cov50 at EI's picks, 0.537 [0.521, 0.551] vs the control's 0.517.
  - Pooled over all points, BO contexts are calibrated within tolerance (cov90 0.881; control 0.900).
  - The BO-specific under-coverage grows late in the run near the incumbent: local cov90 0.831 vs 0.879 at t 15–19.
- Test 1:
  - E3 not supported: log EI Spearman 0.04–0.05 in the ≥ 10-bin strata.
  - E4 does not explain the headroom: the 57% of states with gap < 1 bin carry 2.1% of it, while the ≥ 10-bin strata carry 88%.
- Next: 0007 test 3, a variance-reduced lookahead at the large-gap states.

### 2026-11-12 — scratch checks: f\* is correct; P0 at d = 2 is easy for EI

These are scratch checks (`uv run`, not tracked), run on 200 of the pinned label tasks (every 20th state of `api_round1/pilot_20261111-133311` labels.pt, sha256 c27d8939…).

- **f\* is not the cause.**
  - A 16× stronger search (65536 Sobol points plus 64×500 Adam steps) never improves on the stored f\* by more than 1e-6. It improves at all in 7.5% of the tasks.
  - The share of states within one bin of f\* is unchanged at 0.625.
  - By step t: 0.26 (t < 5), 0.59, 0.80, 0.78 (t ≥ 15).
- **The tasks are easy.**
  - 47% of the tasks are effectively 1-D: each dimension is irrelevant with probability 0.3.
  - Share of the domain within one bin of f\*: median 0.54% (1-D tasks 1.9%, 2-D tasks 0.07%).
  - Within 10 bins: median 3.9%. Within 100 bins: 27%.
- y on the ECDF scale is not quantized: 0.01% of the values lie on a 0.001 grid.

### 2026-11-12 09:50 — ei_opt pilot decided INCONCLUSIVE (snapshot pilot_20261112-091043)

- Mandate (Tim): "go ahead" with step 1, a deterministic π₀ and how suboptimal the sampled EI is. Launched the sanity run (09:04, 5.3 min) and the pilot (09:10, 26 min) on an idle ulysses.
- Sampled EI is stochastic: replicate picks differ in 90% of states, and the replicate sd of the final normalized regret (median 0.0072, mean 0.79) is the round-1 label noise. Its value is near-optimal: EI ratio median 0.995.
- `fixed+refine` vs `sampled`: normalized regret −5.7% [−19%, +7%], but log regret +0.056 [0.013, 0.100] log10 (worse). Unrefined `fixed` is deterministic and indistinguishable from `sampled` on both scales (post hoc).
- Also added, as a scratch figure (not committed): budget vs regret for EI, π₁ and the decoder on the 1000 split-1 tasks.
  - EI's median regret reaches 1e-5.1 in ECDF units at T, far below one PFN bin.
  - The labels' linear normalized regret gives no signal for that late progress. Tim clarified that the reward should reward progress near the optimum (log scale).

### 2026-11-12 — decision 0008: where the headroom is

- Recorded the shared reasoning in `research_log/decisions/0008-where-the-headroom-is.md`:
  - why rounds 1 and 2 found nothing: linear reward, label noise from our own stochastic π₀, log regret mixing basin discovery with polishing, one-step deviations washed out by the horizon;
  - the luck-corrected switching evidence: BBOB decided GO, +66% at d = 2 for the PFN; the prior pilot +37%, indicative only; the belief controller −49% (#6);
  - the pitfalls and scaling of privileged basin labels;
  - the new order of work: decompose by basin, exact option labels (arm for k steps, then π₀), then the belief.
- The candidate-level relabeling round proposed earlier today is paused.

### 2026-11-12 11:55 — option_headroom pilot decided GO (snapshot pilot_20261112-111235)

- Mandate (Tim): plan and run experiments that give an actual signal for policy improvement.
- Ran on ulysses: the sanity run (11:07, 3 min), a revision (ε 1e-6, static comparator), and the pilot (11:12, 26 min).
- First non-zero, luck-corrected advantage over EI in this line of work:
  - option headroom H₃ = 0.195 [0.148, 0.244] log10 (about 36% less regret), against nulls −0.007;
  - context-dependent part 0.109 [0.070, 0.150]; static part 0.086 [0.043, 0.134].
- EI is not the best static arm under log regret: MES and UCB(0.995) are better.
- At d = 2 the gain is mostly polishing, not basin discovery.
- Plot of tasks and trajectories (scratch): about half the tasks are 1-D valleys; 2-D tasks have a median of 12 local minima.
- Verified: notebook numbers re-read from cell outputs; the two descent settings agree 100%; tests pass (70).

### 2026-11-12 15:50 — option_policy pilot: INCONCLUSIVE by the pre-registered rule (snapshot pilot_20261112-123724)

- Mandate (Tim): discover a non-static, context-dependent rule; EI base, MES outside the primary options; "prepare your changes and run on ulysses a comprehensive experiment".
- Ran on ulysses: the sanity run (12:30, 6 min) and the pilot (12:37–14:39, 2000 tasks, code c36d0c3).
- The deployable one-deviation policy beats EI by +0.135 [0.096, 0.174] log10, but the best static schedule (UCB for the rest from t = 2) already gives +0.119; policy over static +0.017 [−0.018, 0.050].
- The context is used (+0.043 [0.020, 0.066] over the shuffled null) without exceeding the schedule. Privileged headroom 0.346.
- Post hoc (scratch on the run artifact `labels.pt`): with EI's terminal regret as a privileged feature the same pipeline gains +0.147 [0.109, 0.183] over static. The missing piece is predicting from context whether EI will end badly.
- Verified: notebook 13 executed on the snapshot, numbers re-read from cell outputs and the `summary` table; tests pass.
- Open: `just decide ... INCONCLUSIVE` not yet run (blocked in the session); the MES-base run was not launched.

### 2026-11-12 15:45 — mandate for the night (autonomous until 2026-11-13 07:30)

- Tim: "i'll leave until tomorrow morning 7:30, so until then do both and give me your conclusions (including standalone notebooks that record the experiment's analysis)".
- "Both" = (1) can EI's failure be predicted from the context, and does gating the UCB schedule with it beat the static schedule; (2) repeated decisions.
- Allowed: new experiments under `experimental/`, sanity and pilot launches on ulysses (idle), each with a stated ETA. No selection on split 1.

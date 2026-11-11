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

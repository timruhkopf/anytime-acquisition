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

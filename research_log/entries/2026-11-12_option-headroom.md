---
experiment: option_headroom
sweeps:
- id: pilot_20261112-111235
  stage: pilot
  commit: 190c3a11d18b
  decision: GO
  reason: arms H3 0.195 [0.148,0.244] log10 (nulls -0.007); context-dependent part
    0.109 [0.070,0.150]; robust at eps 1e-5
  members: 1/1 done
  decided: '2026-11-12'
---
# Where is the headroom over EI? Basin decomposition of static arms and luck-corrected option labels

## Question

See the description at the top of `notebooks/12_2026-11-12_option_headroom.ipynb`; not retyped here.

## §1 Sanity (2026-11-12)

**What ran.** 50 Part A tasks and 60 Part B states on ulysses (`results/option_headroom/sanity_20261112-110719`, code d90eda4). It took 3 min (A 1.6, B 1.4).

**Result** (a pipeline check; not for decisions):
- The two descent settings agree on the basin label for 100% of tasks.
- Regret exactly 0 (the floor) in 9% (A) and 14% (B) of runs.
- Part A, final median log10 regret: UCB(0.995) −5.45, MES −5.35, EI −4.87, PI(0.01) −3.93, mean −3.18. Good basin at T (δ = 1): 0.88–0.94 for every arm except mean (0.65). Median τ = 0 for every arm.
- Part B (ε = 1e-7): H₃ = +0.28 [0.12, 0.51] for the arms and −0.07 for the nulls. Every arm's mean advantage over EI is positive.

**Revision before the pilot** (notebook 12):
- ε = 1e-6, because lanes that set f\* got log10 ε;
- a static comparator, the best single option, plus the context-dependent part arms − static;
- an ε = 1e-5 robustness check.

**Next.** The pilot (500 + 800 tasks).

## §2 Pilot (2026-11-12)

**What ran.**
- 500 Part A tasks (split 0) and 800 Part B states (split 4, offset 20000), d = 2, two candidate-set seeds (`results/option_headroom/pilot_20261112-111235`, code 190c3a1).
- It took 26 min on ulysses: A 7.3, B 18.3.
- Analysis: notebook 12.

**Result** (log10 terminal regret, ε = 1e-6; positive = better than EI):
- Cross-fitted headroom of the arms: H₁ 0.116 [0.086, 0.153], **H₃ 0.195 [0.148, 0.244]**, H₅ 0.262 [0.212, 0.313].
- Nulls ≈ 0: H₃ −0.007 [−0.026, 0.013].
- Best single option: H₃,static 0.086 [0.043, 0.134].
- **Context-dependent part (arms − static):** H₃ 0.109 [0.070, 0.150]; H₁ 0.092, H₅ 0.146, all CIs > 0.
- Robust at ε = 1e-5: H₃ 0.147, context part 0.075 [0.044, 0.108].
- Post hoc, the context part at k = 3: 2-D tasks 0.152 [0.087, 0.218], 1-D tasks 0.068 [0.026, 0.113].
- Part A, mean final log10 regret: MES −4.67, UCB(0.995) −4.64, EI −4.48, PI −4.04, mean −3.57. EI is not the best static arm under log regret.
- Basin gains are small (at most +0.034 net, UCB k = 5). The median τ = 0 for every arm.
- The two descent settings agree on the basin label for 100% of tasks.

**Interpretation.**
- With exact labels and a log reward, the advantage over EI is clearly non-zero and luck-corrected. About half of it is context-dependent: which arm to follow depends on the state and task.
- At d = 2 it is mostly polishing efficiency, not basin discovery.
- It is privileged: an upper bound for a belief-based policy.

**Verdict: GO** (pre-registered rule on the arms' H₃; recorded with `decide`).

**Next.** Can the belief find it? Two parts:
1. the static part, a better default arm than EI under log regret, which needs no prediction;
2. a value model or belief rollouts per (arm, k), trained on these exact labels, with conservative acceptance against EI and the best static arm (0008 §4 step 3).

Then d = 4, where basin discovery should matter more.

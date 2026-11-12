---
experiment: option_headroom
sweeps: []
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

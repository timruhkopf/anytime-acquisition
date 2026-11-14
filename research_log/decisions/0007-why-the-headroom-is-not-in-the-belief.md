<!-- Claude, 2026-11-12, at Tim's request after the api_gate / api_probe / api_decoder pilots (issue #13). Status: open. -->

# Why the clairvoyant headroom over EI is not in the PFN's belief: explanations, and how to tell them apart

Nov 12, 2026 · Claude (requested by Tim)

## 1. Where we stand

Setting: noise-free P0, d = 2, n_init = 4, T = 20. The prior is pfn-repro's BNN prior: random tanh MLPs with random depth and width and spurious dimensions, on the training ECDF scale [0, 1]. The PFN is the frozen pfn_large.

**The label.** For a state (task, D at step t), a candidate x is scored as follows:
1. query x on the true task;
2. follow EI on the true task to T;
3. record R(x), the final regret in % of (f_med − f\*).

adv(x) = mean over 4 continuations of R(x₀) − R(x), where x₀ is EI's pick, with common random numbers. The tasks are noise-free, so the only randomness in a continuation is EI's sampled candidate set.

**What the pilots found.** The within-state rank agreement with this label (Spearman):

| Score | Spearman | Source |
|---|---|---|
| Trained readouts of PFN states | ≤ 0.025 | api_probe §2 |
| PFN lookahead | 0.042 | api_gate §2 |
| KG | 0.048 | api_probe §3 |
| log EI | 0.063 | api_probe §3 |
| 0006 decoder | 0.060 | api_decoder §2 |

The label's split-half reliability is 0.57.

## 2. Candidate explanations

**E1. Irreducible at decision time.**
- The 0.57 reliability measures how repeatable the advantage is *given the true task*. A policy only sees D. Its best possible score is the posterior-expected advantage E_{f ~ p(f | D)}[adv(x)].
- Each task can have a clear winner. If the winner differs across the functions consistent with D, the posterior expectation is nearly flat.
- That is real structure, but not identified at decision time. No policy, even a Bayes-optimal one, could exploit it, and EI would then be close to the best one-step policy.

**E2. PFN approximation error.** The PFN's predictive at BO contexts differs from the Bayes posterior under P0. BO contexts are clustered near the incumbent, which is likely not the distribution of context designs the PFN was trained on. An exact posterior would see the advantage; the PFN does not.

**E3. Resolution: the discretization of the PFN's output.**
- pfn_large predicts on 1000 uniform bins of width 0.001 on the ECDF scale (verified on the checkpoint).
- If the incumbent is within a bin or two of f\*, the predictive cannot represent the improvements that decide the label.
- EI, PI and KG computed from it are then coarse, and so is everything learned from it.

**E4. The label measures what does not matter at the PFN's scale.**
- The normalized regret divides by the task's own range (f_med − f\*).
- Late in a run, differences far below one bin are amplified into "% of gap". The headroom is then a property of the label's metric, not of any decision-relevant difference.
- Neighbouring E3: E3 is about the model's ability, E4 about what the target rewards.

These explanations are not exclusive.

## 3. Evidence so far

All of these are scratch checks: `uv run`, not a tracked run, reproduced in the notebook of the next experiment. Inputs: the pinned labels `api_round1/pilot_20261111-133311/0/tracking/artifacts/labels.pt` (sha256 c27d8939…, 4000 states), and `results/api_gate/pilot_20261111-164326/tables/gate_states.parquet`.

**Resolution (E3, E4).**
- Gap between incumbent and f\* in ECDF units, quantiles q10 / q25 / q50 / q75 / q90: 0 / 0 / 0.00032 / 0.0103 / 0.063.
- In 57% of the labeled states the incumbent is already within **one bin** (0.001) of f\*; in 62%, within two.
- 91% of the non-base labeled advantages are smaller than one bin in ECDF units.
- Most of what the labels rank is therefore below the PFN's output resolution.

**The gate's lookahead was noise-dominated (it does not discriminate E1 from E2).**
- A correction to the reading in api_gate §2 and the 2026-11-11 track record.
- Units: the gate's model advantage is the difference of fantasized best y in ECDF units, while the label is in % of the gap. The gate's decisions (sign, within-state argmax, τ·se) and the Spearman are invariant to this positive per-state scale. The calibration check below converts the model advantage with each state's 100 / (f_med − f\*).
- In matched units, the model advantage, demeaned within state, has sd 14.2 % of the gap. Its Monte-Carlo noise sd (M = 4 continuations, from the per-candidate se) is 20.1.
- Estimated signal variance: ≤ 0, in every gap stratum. The true labels have sd 1.67.
- Calibration slope of the true advantage on the model advantage: −0.001 [−0.005, 0.001].
- Reading: with 4 belief continuations, the lookahead's estimate of the posterior-expected advantage is pure Monte-Carlo noise. The belief-continuation outcome varies much more than the true outcome, because it integrates over f as well. So Stage 1a measured neither E1 nor E2. Resolving a signal of sd ≈ 0.5 against a per-sample noise of about 40 would need thousands of continuations per candidate, so variance reduction is required (§4).

**What the probes "know" about the gap.**
- log_gap is read from PFN states at Spearman 0.775 (all layers, ridge; api_probe §2, 826 test states).
- A trivial data statistic, the second-best minus the best observed y, reaches 0.725 on all 4000 states.
- The incumbent level is uninformative (−0.004), and t gives −0.43.
- So the PFN's reading of the gap adds little beyond what is visible in the raw data.
- The headroom_bin AUC (hand 0.681 vs last layer 0.663) is a tie. Only the 6912-dimensional all-layer probe is worse (0.616), most likely from overfitting with about 300 positive training states.

## 4. Tests that tell the explanations apart

1. **E3 and E4 (cheap, existing data).** Stratify every existing comparison by the gap in bins (< 1, 1–10, 10–100, > 100):
   - the label reliability, the clairvoyant headroom, the ranking scores (log EI, KG, the decoder), and the decoder's gain.
   - If the headroom concentrates where the gap is < 1 bin and vanishes elsewhere, E4/E3 dominate.
   - If ranking signal appears in the resolvable strata, E3 limits the PFN but the decision problem is real there.
2. **E2: the PFN's inference on BO data vs on random designs**, at the *same* tasks and the *same* n.
   - Scores: proper scoring rules (log score of f(x) under the predictive) and PIT calibration at held-out points: uniform, local around the incumbent, and EI's next pick.
   - Why this is a valid check of calibration on BO data: for a design chosen from the observed data only, the design is ignorable. If the PFN were the exact posterior, the PIT of f(x) would be uniform over prior draws for BO and random designs alike.
   - Marginal calibration is necessary, not sufficient. So the comparison that carries the weight is the **paired log-score degradation, BO minus random design**, within task.
   - Report both at the PFN's resolution (bin probability) and on resolvable targets (points whose f lies > 5 bins below the incumbent).
3. **E1 vs E2: a variance-reduced lookahead.** Recompute the posterior-expected advantage under the PFN with enough precision to measure its calibration slope:
   - stratified quantiles of the first fantasized y, as in KG;
   - common random numbers across candidates;
   - many more continuations at fewer states.
   - Under the exact posterior, the calibration slope is 1, so the true advantage is unbiased given the model advantage.
   - Reading: slope ≈ 1 with a flat model advantage → E1. A model advantage with real spread but slope ≈ 0 → E2. Slope ≈ 1 with real spread → the gate was just underpowered.
4. **A reference posterior for E2.** The BNN prior has no closed form. A near-exact reference would need sampling over BNN weights and architecture, for example SMC or HMC with the prior's hyperpriors. This is expensive and deferred until tests 1–3 show whether it is needed.

Fine-tuning the PFN (0004/0005) stays deferred (Tim, 2026-11-12) until these tests say whether E2 matters.

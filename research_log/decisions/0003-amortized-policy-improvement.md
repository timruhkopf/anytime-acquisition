<!-- Design document by Tim, 2026-11-11, added verbatim from the shared doc ("Amortized policy improvement for PFN-based BO: method, training and prior"). Status: adopted as the plan for issue #13; round 1 implemented on exp/13-api-round1 with the scope deviations listed in notebooks/06_2026-11-11_api_round1.ipynb. Companion assessment: 0004-pfn-finetuning-on-bo-trajectories.md. -->

# Amortized policy improvement for PFN-based BO: method, training and prior

Nov 11, 2026 · @Tim

We train a value head on a frozen two-headed PFN to predict, for any candidate query, the ground-truth final regret of querying it and then following latent EI. Acting greedily on it is one step of policy improvement: non-myopic, budget-aware and noise-aware at EI cost. This document supersedes the plan in the earlier formal problem statement and specifies the method, architecture, training and prior.

## 0. Why this is worth doing

PFNs amortized the inference half of Bayesian optimization; the planning half is still a hand-written one-step rule. With a samplable prior and free ground truth, planning is fully simulable and can be amortized too. Before anyone spends the compute to do that with RL, someone should measure how much planning is worth. This project does both: it measures the gap and captures part of it with a guaranteed floor.

### 0.1 The overall picture

BO under a prior is a Bayes-adaptive MDP with two hard parts: inference (the posterior predictive given the data) and planning (the Bellman recursion over future observations).

| Part | Classic BO | Amortized | Status |
| --- | --- | --- | --- |
| Inference | GP posterior, exact for a GP prior | PFN: one forward pass for any samplable prior (Müller et al. 2022, 2023) | done |
| Planning | one-step heuristics: EI, PI, UCB, KG | a policy trained on the same prior | open; this work takes the first, guaranteed step |

The prize is concrete. An ideally trained in-context policy on a prior Π is the Bayes-optimal policy for Π (Ortega et al. 2019): non-myopic, budget-aware and noise-aware, for any prior one can write down, at one forward pass per step.

### 0.2 Why it has not been done convincingly

- **Two effects are confounded.** MetaBO, FSAF, NAP and the OptFormer line learn acquisitions on top of a surrogate whose prior differs from the task family. Their gains mix genuine planning with compensation for the wrong prior. With a PFN trained on the task prior, only planning gain (and correction of PFN error) remains. Its size has never been measured.
- **RL is the bottleneck.** Regret rewards are sparse and terminal; NAP needed an auxiliary model loss to train at all.
- **PFN BO stayed myopic.** PFNs4BO used EI; its one-step KG variant was shown only on spaces of at most three dimensions.

### 0.3 Thought experiment: RL directly on top of a PFN

Suppose someone trains a policy with RL on the frozen PFN, on the infinite task stream, with ground-truth regret as reward.

**If it succeeded completely,** it would converge to the Bayes-optimal policy for the prior. Its gain over EI would be exactly the planning gap plus the share of PFN error that ground truth corrects. BO would become a pretrained model end to end: inference and decisions both amortized.

**What it would have to overcome:**

- A sparse terminal reward spread over T decisions, with variance dominated by instance difficulty rather than by the actions.
- Non-stationary targets: every update changes the policy that generates the returns.
- A continuous, multimodal action space whose optimum is the argmax of a field that changes every step.
- No floor: an untrained policy is far worse than EI, so much compute goes into relearning EI before anything new is learned.

**Why its outcome would be hard to read.** If RL failed, nobody could tell whether the planning gap is small or the optimizer is weak. If it succeeded on one prior, nobody could tell how much of the gain is planning and how much is PFN error.

### 0.4 What this project does instead

We take the first step of policy iteration from latent EI and make it a measurement instrument.

1. **Measure the gap first.** The advantage of the best candidate over EI's pick, estimated on thousands of instances, bounds what any policy can gain over EI to first order (performance difference lemma, §3.6). The measurement is cheap and decisive.
2. **If the gap is small,** EI is near a fixed point of policy iteration for this prior. RL on top of PFNs is then not worth its cost for priors of this kind. That is a useful negative result for the whole learned-acquisition programme.
3. **If the gap is large,** a value head captures part of it by supervised regression on exact returns. It has a floor (never worse than EI with exact advantages, a confidence-bound fallback with estimated ones) and costs one encoding per step.
4. **It is the stepping stone for RL.** The improved policy is a strong initialization, and the value head is a ready critic for later fine-tuning with a KL penalty toward the improved policy.

### 0.5 Expected contributions

- A first measurement of the planning gap over EI on a PFN trained on the task prior, including heterogeneous and heteroscedastic noise.
- Amortized policy improvement on a PFN: non-myopic, budget- and noise-aware, with a floor, at EI cost.
- A prior and normalization scheme intended to transfer from synthetic to real tasks.
- An evaluation protocol that separates legitimate privileged information from clairvoyance.

## 1. Problem setting

BO under a known synthetic prior is a Bayes-adaptive MDP whose transition kernel is the PFN's posterior predictive; the derivations are in the earlier formal problem statement, condensed here.

**Objects.**

- Domain X = \[0,1\]^d. Task ω = (θ, φ, η) \~ Π: BNN weights θ, architecture φ, noise parameters η. Objective f\_ω = F ∘ g\_ω with F the prior-marginal CDF (monotone, argmax-preserving).
- Observations y = f\_ω(x) + ε from a noise kernel P\_ω(dy | x). Data D\_t, size n\_t = n\_0 + t, budget T.
- Latent mean μ\_D(x) = E\[f\_ω(x) | D\]; best latent mean m(D) = sup\_x μ\_D(x).
- The PFN q\_φ has a latent head (target f(x)) and a noisy head (target y). It outputs one-point marginals, not a joint; the joint is reachable only through re-encoding.

**Objective.** Final normalized regret at a known T, with T a policy input:

```latex
R_T(\omega) = \frac{f^\star_\omega - f_\omega(\hat x_T)}{f^\star_\omega - \operatorname{med}_x f_\omega(x)}, \qquad \hat x_T = \arg\max_{i \le n_T} \mu_{D_T}(x_i)
```

The recommendation rule (best latent mean among evaluated points) is fixed for every policy. Anytime curves are reported as secondary.

**BAMDP.** State s\_t = (D\_t, t, T); action x\_t; transition y\_t \~ P\_Π(· | x\_t, D\_t), amortized by the noisy head. The optimal policy satisfies the Bellman recursion below and is intractable; every heuristic is its one-step special case (KG with a free recommendation, EI with an evaluated one).

```latex
V_t(D) = \max_x \int V_{t+1}\big(D \oplus (x, y)\big)\, P_\Pi(dy \mid x, D), \qquad V_T(D) = -\mathbb{E}[R_T \mid D]
```

**What can be gained.** Under a correct model the achievable gain over heuristics is the planning gap (myopia plus budget). Ground truth then adds variance, not information. GT pays where the PFN errs, chiefly on BO-shaped contexts.

**Privileged information rules.** The generator gives f, f\*, x\*, noise, medians and volumes exactly. Their use is legitimate when what reaches deployment is a functional of Law(Z | D):

| Use | Allowed |
| --- | --- |
| Policy input | only F\_t-measurable quantities: data, t, T, own randomness |
| Training target | any Z under a proper loss (squared loss for means, NLL for laws) |
| Reward | any function of ω, including exact regret |
| Critic or baseline | may use ω and the noise stream; must not depend on the action being scored |
| Selection | never evaluated on the same (ω, noise) it was selected on |
| Evaluation | everything |

Two consequences drive this method. Classifying the realized best branch learns win frequency, not expected return, so labels are regressed, never classified. Normalization at deployment uses only data-derived scales.

## 2. Related work and positioning

No prior method learns a non-myopic acquisition by supervised regression on ground-truth returns, on top of a calibrated PFN trained on the same prior; that is the gap we target.

| Line | Work | What it does | Gap relative to us |
| --- | --- | --- | --- |
| Learned acquisition on a GP | MetaBO (Volpp et al. 2020); FSAF (Hsieh et al. 2021) | RL (PPO, Bayesian DQN) over a candidate set; per-candidate state of posterior mean, std, location, t, T | index rule on marginals; gains partly compensate a GP prior that does not encode the task family |
| Joint surrogate and policy | NAP (Maraval et al. 2023) | one transformer as neural process and acquisition, RL plus an auxiliary probabilistic loss because regret rewards are too sparse | RL is the weak point; trained per task family |
| Sequence models on logged runs | OptFormer (Chen et al. 2022); RIBBO (Song et al. 2024); ZeroShotOpt (Meindl et al. 2025) | imitate or offline-RL over trajectories of behaviour optimizers; RIBBO conditions on regret-to-go; ZeroShotOpt pretrains on synthetic GP functions from 12 BO variants | tied to behaviour algorithms; no calibrated surrogate; ground truth used only through logged returns |
| Online adaptation | GP-Hedge (Brochu, Hoffman & de Freitas 2011); SAWEI (Benjamins et al. 2023) | portfolio weights from model-estimated gains; weighted-EI trade-off adjusted on convergence of an upper regret bound | no meta-learning; SAWEI evaluated on noise-free functions; both become baselines |
| PFN surrogates | PFNs4BO (Müller et al. 2023); ifBO (Rakotoarison et al. 2024); Decoupled PFNs (Bergna et al. 2026) | PFN as BO surrogate with EI; a one-step KG variant shown only on spaces of at most 3 dimensions; freeze-thaw PFN for learning curves; latent and noise heads | non-myopic acquisition on a PFN is open beyond toy size; noisy EI unsupported because joint samples are unavailable |

**Positioning.**

- The surrogate is trained on the task prior, so a learned acquisition cannot win by compensating a wrong prior. Any gain is planning gap or correction of PFN error, which is the honest quantity.
- Learning is supervised regression on exact, noise-free returns. It avoids the sparse-reward problem NAP reports.
- The policy improves a strong base policy and contains it, so it has a floor.
- The value head reads the PFN's global state (noise regime, effective dimension, budget), not only per-point marginals.
- The setting includes heterogeneous and heteroscedastic noise, which the learned-acquisition literature mostly skips.

Abstracts of all listed papers and the full text of PFNs4BO and Decoupled PFNs were read; details of the others are to be checked against full text.

## 3. Method: one step of policy improvement, amortized

We learn the advantage of each candidate query over the base policy's own pick, measured in ground-truth final regret, and act greedily on it with a calibrated fallback.

### 3.1 Base policy

The base policy π\_0 is **latent EI**: expected improvement computed on the PFN's latent head, against a latent incumbent.

- **Distribution:** the latent bar distribution of f(x), which carries uncertainty about the function only, not observation noise.
- **Incumbent:** τ = max\_i μ\_D(x\_i), the best posterior latent mean among evaluated points, the same rule as the recommendation.

```latex
\mathrm{EI}_{\mathrm{lat}}(x) = \mathbb{E}_{f(x) \sim q_{\mathrm{lat}}(\cdot \mid x, D)}\big[(f(x) - \tau)^+\big] = \sum_k p_k(x)\, \mathbb{E}\big[(f - \tau)^+ \mid f \in \text{bin } k\big]
```

Each bin is uniform, so every term is closed-form; optimize log EI for numerical stability. Standard EI uses the noisy predictive and the best observed y instead. Under noise it fails twice: noise inflates the predictive variance, so EI stays high in noisy regions and at points already measured, and max y is inflated by the winner's curse. Without noise the two coincide. Decoupled PFNs uses the same rule.

**Why it is the base:** strong, one encoding per step, deterministic given a seed, consistent with the recommendation rule. **Weaknesses:** myopic, and the plug-in incumbent ignores uncertainty about which evaluated point is truly best.

**Why not noisy EI.** Noisy EI (Letham et al. 2019) integrates over the joint posterior of the candidate and all evaluated points:

```latex
\mathrm{NEI}(x) = \mathbb{E}_{(f(x), f(x_1), \dots, f(x_n)) \sim p(\cdot \mid D)}\Big[\big(f(x) - \max_i f(x_i)\big)^+\Big]
```

The joint matters twice: a candidate near the incumbent is correlated with it, so its improvement is small, and the maximum of correlated values is lower than that of independent ones. A PFN's test tokens attend only to the context, never to each other, so it outputs n + 1 separate marginals. Independent draws overestimate the incumbent and ignore the correlation, a bias that varies by candidate. A true joint sample needs the chain rule (sample, append, re-encode, n + 1 times per Monte Carlo sample) and would append noise-free latent values to a context trained on noisy observations, which is out of distribution. PFNs4BO states the same limitation. A dedicated joint head would remove it; until then latent EI is the strongest cheap base.

All components (candidate generator, inner optimizer, §4.3) are deterministic given the state and a seed from the common-random-number stream, so π\_0 is reproducible across branches.

### 3.2 Value of a query under the base policy

For state s = (D, t, T) and candidate x, define the return of querying x now and then following π\_0 to the budget:

```latex
Q^{\pi_0}(s, x) = -\,\mathbb{E}\big[R_T \mid s,\ x_t = x,\ x_{t+1:T-1} \sim \pi_0\big]
```

The expectation runs over the posterior task (in training: the sampled ω) and all future noise. The learning target is the advantage over the base pick x\_0 = π\_0(s):

```latex
A^{\pi_0}(s, x) = Q^{\pi_0}(s, x) - Q^{\pi_0}(s, x_0), \qquad A^{\pi_0}(s, x_0) = 0
```

Advantages remove instance difficulty, which dominates raw returns, and shrink the label scale to the part a decision can change.

**Rollouts are still needed, but of a fixed policy.** Each label is a Monte Carlo estimate of Q^{π\_0}: query x, then roll out π\_0 to T on the true function (§5.2). Because π\_0 never changes, the target never moves: labels are generated once, stored, and regressed like any supervised dataset. RL differs here, since every update changes the policy that generates the returns. Deployment needs no rollouts: one encoding plus one pass of the value head.

**Why Q under this policy.** It is a fixed, well-defined target; acting greedily on it carries the policy improvement guarantee (the rollout algorithm of Bertsekas); the label is final regret after the remaining T − t − 1 steps, so information that pays only later gets credit; and Q truly depends on remaining budget and noise regime, so giving those as inputs lets the head represent budget and noise awareness without any hand-designed schedule. Q\* would need RL or many policy-iteration rounds; Q^{π\_0} is the first round.

### 3.3 Improved policy

With a learned estimate Â over a candidate set C(s) that always contains x\_0:

```latex
\pi_1(s) = \begin{cases} \arg\max_{x \in C(s)} \big[\hat A(s, x) - c\, \hat\sigma(s, x)\big] & \text{if } \max_{x \in C(s)} \big[\hat A(s, x) - c\, \hat\sigma(s, x)\big] > 0 \\ x_0 & \text{otherwise} \end{cases}
```

**Guarantee.** With the exact advantage and τ = 0, π\_1 is never worse than π\_0 (policy improvement theorem). With estimation error, the loss is bounded by twice the expected sup-error over the states π\_1 visits; the confidence bound below trades the gain against this risk (conservative policy improvement). The policy degrades to EI, not below it, when the head is uncertain.

**Why a confidence bound.** With exact advantages, deviating whenever max A > 0 is correct. With estimates, the maximum over about 150 candidates is biased upward: even if every true advantage is at most zero, the largest estimate is usually positive (the optimizer's curse). The rule therefore deviates from EI only when the predicted advantage clearly exceeds its own estimation noise σ̂ from the head's variance output. This is the conservative-policy-improvement idea: take only steps whose improvement can be trusted. The constant c is calibrated on held-out instances (§5.4).

### 3.4 What the improvement step can capture

- **Non-myopia:** the label is the final regret after T − t further steps, so it values information that pays only later, unlike EI.
- **Budget awareness:** t and T are inputs; the same state can call for different queries at different remaining budgets.
- **Noise awareness:** replicating, avoiding noisy regions, and distrusting single high observations are rewarded when they lower final regret.
- **Correction of PFN error:** labels come from ground truth, so systematic PFN errors on BO-shaped contexts are partly corrected.

### 3.5 Iteration

A second round trains Â for Q^{π\_1} on states visited by π\_1 (approximate policy iteration). Each round costs a full label generation, so we plan one round and a second only if the first gains more than half of the brute-force ceiling (§8).

### 3.6 Relation to KG, and what a null result means

**Why it could beat KG.** KG is one-step optimal only if the run stops after the next query; it does not know that T − t − 1 queries follow, while our label does. On a PFN, KG is computed through the model's own fantasy updates and inherits its martingale errors; our labels come from ground truth. KG maximizes μ over the whole space, while our recommendation is restricted to evaluated points, a mismatch that costs KG. **Why it might not:** the guarantee is relative to EI, not to KG, and KG may already capture most of what one improvement step offers.

**Cost.** KG by re-encoding costs |C| · N\_y encodings per step (for example 64 × 8 = 512) plus an inner maximization per fantasy. The value head costs one encoding plus a small network. Matching KG therefore means a speedup of two to three orders of magnitude at decision time; training cost is paid once, offline.

**What a null result means.** The performance difference lemma, with regret as cost, gives:

```latex
J(\pi^\star) - J(\pi_0) = \sum_{t} \mathbb{E}_{s_t \sim \pi^\star}\big[A^{\pi_0}\big(s_t, \pi^\star(s_t)\big)\big] \le T \cdot \sup_{s}\, \max_{x} A^{\pi_0}(s, x)
```

Here J is expected negative regret and A is measured in the same units. If no candidate has a meaningful advantage over EI's pick, latent EI is near a fixed point of policy iteration and hence near-optimal for this prior and candidate set. The planning gap is then small, and gains reported by learned-acquisition work most likely came from compensating a wrong surrogate prior. Caveat: we estimate advantages on states π\_0 visits, while the lemma uses states π\* visits; branch states from other policies (§5.2) partly cover this.

## 4. Architecture

A frozen two-headed PFN encodes the data once per step; a fixed generator proposes candidates; a small learned value head scores them from PFN query states plus global features, anchored so the base pick scores exactly zero.

[embedded content (diagram, not exported): decision loop per BO step · one encoding, one learned head\]

Only the value head is trained in this method; the PFN, readout and generator are fixed, so the base pick x\_0 is always available as the fallback.

### 4.1 PFN backbone (frozen after pretraining)

| Component | Specification (initial; tune in Stage 0) |
| --- | --- |
| Context token | linear embedding of x (d inputs) plus linear embedding of y, summed |
| Query token | linear embedding of x only |
| Layers | 12 blocks: context self-attention, query cross-attention to context, feedforward; pre-LayerNorm; dropout 0.0 at inference |
| Width | d\_model 512, 8 heads, feedforward 1024 |
| Context length | N\_max = 128 (n\_0 + T ≤ 88 plus margin for replicates) |
| Latent head | bar distribution over f(x), K = 1000 bins, bin edges at quantiles of the prior's f |
| Noisy head | bar distribution over y, K = 1000 bins on the support of y under the prior (wider than \[0, 1\] because of noise) |
| Dimension | one model per d in {2, 4, 8} first; later one model with zero-padding to d\_max and scaling by d\_max/d |
| Optional summary heads (later) | learned summary tokens cross-attending to the context, predicting p(f\* \| D) and p(med f \| D) as bars |

The cache C(D) is the per-layer keys and values of the context tokens. One encoding per step serves every query: probe set, candidates and inner-optimizer gradient steps.

### 4.2 State readout (no learned parameters)

- **Probe set P:** 512 scrambled Sobol points plus 128 Gaussian perturbations around the top 4 evaluated points by latent mean, resampled each step from the shared seed.
- **Global features** (groups from the earlier doc): budget (t/T, T − t, log T); progress (m(D) and its recent change, steps since it rose, latent std at the incumbent); opportunity (∫PI and ∫EI over P, max EI, share of P with P(f > m(D)) > 0.05, entropy of softmax(μ) over P); landscape (std of μ over P, mean latent std, roughness, per-input sensitivities pooled symmetrically, effective-dimension estimate); noise (mean noise scale over P from the gap between noisy and latent spread, its coefficient of variation, noise at the incumbent, SNR estimate, replicate count and spread); geometry (fill distance, recent step size, incumbent distance to the boundary).
- **Normalization:** value features relative to m(D) and divided by the std of μ over P; standardized with running training statistics.

### 4.3 Candidate generator (fixed)

| Source | Count | How |
| --- | --- | --- |
| Base pick x\_0 | 1 | argmax of latent EI: 2048 Sobol points, top 16 refined by 20 Adam steps through the query path |
| Global | 64 | fresh scrambled Sobol |
| Local | 64 | Gaussian perturbations around the top 4 points by latent mean, scales {0.01, 0.03, 0.1} × box width, reflected at the boundary |
| Heuristic argmaxes | 4 | UCB (β = 1 and 3 on latent quantiles), PI, posterior-mean argmax; same optimizer as x\_0 |
| Replicate | 1 | argmax of μ over evaluated points |
| Carry-over | 16 | top of the previous step's set by Â |

About 150 candidates per step, deduplicated within 10⁻³ except the replicate. All seeds derive from the common-random-number stream.

### 4.4 Value head (learned)

- **Per-candidate input:** the PFN's final-layer query-token state z(x; C(D)) (512, detached); latent μ, std and quantiles {0.05, 0.5, 0.95}; latent-incumbent EI and PI; noise scale at x; distance to nearest data point and to the incumbent; source one-hot; base-pick flag; within-set ranks of μ, std and EI.
- **Global input:** global features (§4.2) plus attention pooling over the context tokens' final states with 4 learned queries.
- **Body:** candidate MLP (→ 256) → FiLM conditioning on the global vector → 2 set-transformer layers across candidates (4 heads, width 256) → output head giving a mean g(x) and a log standard deviation.
- **Anchoring:** Â(s, x) = g(x) − g(x\_0), so the base pick has advantage exactly zero and the head only learns differences.
- **Size:** about 2M parameters, small next to the PFN.

**Why FiLM.** A candidate's value depends on how its features interact with the global state: high uncertainty is worth more early than late, high EI less under high noise, a replicate only when noise is high. These are multiplicative interactions (budget × σ, noise × EI). FiLM lets the global vector scale and shift each candidate feature, which captures them directly; plain concatenation can learn them but needs more depth and data. The set-transformer layers then make relative judgements across candidates (best EI in this set, redundancy). Start with concatenation plus an MLP, add FiLM, and keep it only if validation NLL improves.

### 4.5 Decision rule and cost

Deviate to argmax (Â − cσ̂) if that lower bound is positive, otherwise query x\_0. Per step: one encoding plus about 640 probe, 150 candidate and 320 optimizer queries on the cache, and one value-head pass.

## 5. Training procedure

Three supervised stages and one calibration step: pretrain the PFN, generate advantage labels by branching ground-truth rollouts, regress them, then calibrate the fallback threshold on held-out instances.

### 5.1 PFN pretraining

1. Sample a task ω from the prior (§7), a context size n \~ U{1..N\_max} and 64 test points.
2. Build the context from a mixture: 50% uniform x; 30% BO-like contexts; 20% contexts with replicates (k points repeated 1 to 4 times, fresh noise each time).
3. Test points: half uniform, half Gaussian perturbations around context points (the region BO queries).
4. Loss: NLL of the noisy head on y plus NLL of the latent head on f(x), equal weights.
5. **BO-like contexts are bootstrapped.** Round A trains on uniform and replicate contexts only. Round B generates BO-like contexts by running latent EI with the round-A model on fresh tasks, and continues training on the full mixture.
6. Validation on held-out tasks, per d: latent PIT histograms and 50/90% coverage on uniform and on BO contexts; the total-variance identity where noise has finite variance; the martingale self-test; BO with latent EI against random search.

### 5.2 Label generation by branching

For each training instance ω and seed:

1. **Base rollout.** Run π\_0 from a Sobol initial design of size n\_0 to T. Record the states.
2. **Branch states.** Sample one step t per rollout, uniform over {0..T−1}, and rebuild the state s\_t exactly. One labeled state per instance maximizes instance diversity at the same cost: diversity comes from instances, while repeated seeds stay within an instance.
3. **Labeled candidates.** From C(s\_t) take x\_0 plus 15 others, stratified by source and by EI rank; record the stratum so training can reweight if needed.
4. **Continuations.** For each labeled candidate c and each continuation seed k = 1..M (M = 4): query c, observe y from the shared noise stream, continue with π\_0 to T, record the GT normalized regret R\_T(c, k).
5. **Label.** The paired advantage and its standard error over k:

```latex
a(s_t, c) = \frac{1}{M} \sum_{k=1}^{M} \big[R_T(x_0, k) - R_T(c, k)\big]
```

**Shared noise.** Noise is a fixed function of (instance, k, x quantized to 10⁻⁶, visit count of that x): a hashed uniform pushed through the noise family's inverse CDF. Every continuation with the same k sees the same noise at the same point, so luck is shared across candidates and cancels in the difference.

**Legitimacy.** Labels use the true ω, but the head is trained across instances to predict E\[a | s\], a functional of the posterior; at deployment it sees only F\_t-measurable inputs. Labels are regressed, never turned into winner classes.

**Coverage.** In round 1, 80% of branch states come from π\_0 rollouts and 20% from rollouts of other heuristics (UCB, PI, Sobol for the first steps). States the improved policy will visit are then partly covered.

### 5.3 Value-head training

- **Loss:** Gaussian NLL of the label given the predicted mean and standard deviation, with the label's own standard error added to the predicted variance, so noisy labels count less.
- **Data:** all labeled candidates of a state form one set, so the set transformer sees the competition it will face; unlabeled candidates are included as context without loss.
- **Splits:** by instance: 80% train, 10% calibration (§5.4), 10% validation. Test instances are fresh draws.
- **Optimization:** AdamW, lr 3·10⁻⁴ with cosine decay, batch 128 states, weight decay 0.01, early stopping on validation NLL. The PFN stays frozen.
- **Label filtering:** instances with f\* − med f below 10⁻³ are dropped before labeling, by a pre-registered rule, because normalized regret is ill-defined on them.

### 5.4 Threshold calibration

On calibration instances, run π\_1 with the confidence-bound rule for c ∈ {0, 0.5, 1, 2}, paired against π\_0 under shared noise. Choose c by mean paired normalized regret and freeze it before any test run. Also report the deviation rate (share of steps where π\_1 leaves EI's pick) per c.

### 5.5 Optional second round

Relabel on states visited by π\_1, with π\_1 as the continuation policy, and retrain. To cut cost, continuations may stop after h = 10 steps and add the round-1 head's value estimate (bootstrapped targets); this trades variance for bias and must be validated against full rollouts on a subset.

## 6. Training setup

Label generation dominates the cost at roughly 30 GPU-hours per input dimension under the assumptions below; measure it in the pilot before committing.

### 6.1 Settings

| Item | Value |
| --- | --- |
| Input dimensions | d ∈ {2, 4, 8}, separate models first |
| Budget | T \~ U{20..80}, given to the policy |
| Initial design | n\_0 = 2 Sobol points (minimal; the Sobol source lets the policy space-fill further) |
| Base policy inner optimizer | 1024 Sobol, top 8 refined by 10 Adam steps (lr 0.02) through the query path |
| Candidates per step | about 150 (§4.3); 16 labeled per branch state |
| Continuation seeds per candidate | M = 4 |
| Branch states | 1 per base rollout: about 50k labeled states from about 50k distinct instances per d in round 1 |
| Instances | fresh per rollout; pilot 500 × 2 seeds per d; advantage-map gate 5000 per d; full ceiling 300 per d; test at least 5000 fresh instances × 2 seeds per d |
| PFN pretraining | 300k steps, batch 128 tasks, AdamW lr 10⁻⁴ with warmup and cosine decay; round A 200k, round B 100k |
| Value head | about 2M parameters; 20 epochs over the label set, early stopping |

### 6.2 Compute estimate (assumptions stated)

- **PFN size:** 12 blocks of about 2.1M parameters, about 25M total, so roughly 50 MFLOP per token forward.
- **One base-policy step:** one encoding (about 60 context tokens) plus about 1300 query-token equivalents including optimizer backward passes, so about 70 GFLOP.
- **One labeled branch state:** 16 candidates × 4 seeds × about 25 remaining steps = 1600 steps, about 110 TFLOP.
- **Throughput:** assumed 50 TFLOP/s effective in bf16 on one modern GPU; this is the most uncertain number.
- **Result:** about 2 s per labeled state, about 30 GPU-hours per d for 50k states, about 90 GPU-hours for three dimensions. The gate experiment (§8) costs about 2 minutes per episode, about 10 GPU-hours for 300 episodes.

### 6.3 Engineering

- **Lockstep batching:** all branches from one state have the same context length at every later step, so they batch without padding. Batch across states and instances at equal n.
- **Query path reuse:** probe set, candidates and optimizer steps share one cache per state.
- **Determinism:** every random draw is keyed by (instance, seed, k, step, purpose). Tests check that a branch replay reproduces the base rollout bit for bit.
- **Storage:** per labeled state store the state reference (instance id, seed, t), candidate coordinates, labels and standard errors, not the PFN features; features are recomputed on load so the readout can change without relabeling.
- **Ground truth:** f\*, x\* and the median are computed once per instance (multi-start gradient ascent checked against dense Sobol; Monte Carlo median) and cached.

## 7. Developing the prior

The prior should be developed in versions, each justified by a measurable improvement on real tasks, and frozen before any label generation. The first change is not noise but the output scale: the global CDF F gives every task an absolute scale that real tasks do not have.

### 7.1 Problems with the current prior (P0)

- **Absolute output scale.** Under F every task lives in \[0, 1\] with a prior-wide meaning: a value of 0.98 says "near the top of the prior". A PFN can exploit that, and a learned policy can too; on real tasks, where y is normalized from data, the cue is gone or misleading.
- **F compresses the top.** Near-optimal differences are tiny in F-units, which distorts both noise levels and the PFN's resolution exactly where BO operates.
- **Realism.** In PFNs4BO a BNN prior matched the best baseline on HPO-B but lost to a HEBO-style GP prior on Bayesmark and clearly on PD1. A BNN-only prior is a risk for any external claim.
- **Missing structure.** No noise, no irrelevant inputs, no input warping, no boundary optima, no plateaus.

### 7.2 Version ladder

| Version | Change | Why | Check that justifies it |
| --- | --- | --- | --- |
| P0 | BNN draws under the global CDF F, noise-free | current | baseline numbers |
| P1 | drop F as the scale: random monotone warp and broad affine map of the output per task, so absolute level carries no information; robust per-context standardization in the PFN, identical in training and deployment (§7.4) | removes the fake absolute scale; matches how real tasks are seen | PFN log score on real tasks does not drop; in-prior BO unchanged |
| P2 | noise: SNR relative to each task's gap, structure mixture (none, homoscedastic, small-BNN heteroscedastic), families (Gaussian, Student-t ν ≥ 3, contaminated, skewed); replicate contexts | the main experimental axis | identifiability probes (§8 Stage 0) |
| P3 | HPO structure: per-input relevance (low effective dimension), per-input Kumaraswamy input warps, boundary optima, additive low-order components | the structures PFNs4BO found to matter on real HPO | log score and EI regret on HPO-B and Bayesmark improve |
| P4 | family mixture: BNN, GP with ARD Matérn and a HEBO-style hyperprior, additive BNN; family as a latent variable | robustness of the surrogate across function classes | external-suite regret improves; in-prior calibration holds |
| P5 | empirical-Bayes weights: fit the mixture weights and hyperprior ranges to maximize the PFN's prequential log score on training splits of real suites | principled realism | held-out real-task log score and regret |

Held out from every version for robustness tests: rotated low-dimensional subspaces, trend plus needle, discontinuities, infinite-variance noise, crashed evaluations.

### 7.3 Generative process (target: P4)

1. Draw d\_eff \~ U{1..d} and per-input relevances: relevant ≈ 1, others 0 or about 0.05.
2. Draw per-input warps: Kumaraswamy(a, b) with log a, log b \~ N(0, 0.5²); with probability 0.2 a log-type warp.
3. Draw the family: BNN (depth 1 to 6, width 16 to 128, activation in {tanh, ReLU, sin}, weight scale log-uniform), GP (ARD Matérn-5/2, length scales log-normal), or additive (sum of 1 to 3 low-dimensional BNNs).
4. With probability 0.15, shift the input box so the optimum falls on a boundary.
5. Compute g on the warped, relevance-scaled inputs. Apply a random monotone output warp (power or log-type), then standardize by the task's own median and IQR. This standardization is only for numerical range; the PFN never sees it, because its inputs are normalized from the context.
6. Compute ground truth on the standardized g: f\*, x\*, median, gap Δ = f\* − med.
7. Draw the noise regime, family and SNR; set the noise scale to Δ / SNR; add noise per the tables below.

| Noise regime | Share | Definition |
| --- | --- | --- |
| Noise-free | 20% | y = f(x) |
| Homoscedastic | 40% | constant robust scale s = Δ / SNR |
| Heteroscedastic | 40% | s(x) = s̄ · exp(a · h(x)), h a standardized one-layer BNN of width 16, optionally mixed with f; a \~ U\[0, ln(10)/4\] |

| Noise family | Share | Parameters |
| --- | --- | --- |
| Gaussian | 60% | none |
| Student-t | 20% | ν \~ U\[3, 10\] |
| Contaminated Gaussian | 10% | outlier rate in \[0.01, 0.1\] |
| Skewed or one-sided failures | 10% | skew-normal, or a rare low value |

log₁₀ SNR \~ U\[log₁₀ 2, log₁₀ 100\], where SNR = Δ / s with s the robust scale (IQR/1.349) of the noise. The range is a starting guess: fix it after the pilot so that BO still clearly beats random search at the lowest SNR, and pre-register it. Results are stratified by SNR and by the top-gap ratio.

### 7.4 Context normalization (P1 and later)

**The concern.** If y is normalized by the data seen so far, the normalization changes during a run, at every step and especially whenever a new extreme is observed. The PFN, the features and the policy would then see a moving scale.

**Resolution in four parts.**

1. **Make the prior invariant first.** Per task, draw a random monotone output warp and a broad affine map y → a + b·y. The absolute level and scale then carry no information about the task, as on real problems.
2. **Standardize with equivariant, robust statistics.** At every forward pass, map y′ = (y − â(D)) / b̂(D) with â the median and b̂ the IQR of the context's y. Min–max scaling is avoided because it moves with every new extreme; median and IQR barely move when one new maximum arrives. Fallbacks: for n < 3, centre on the first observation and use the range; if the IQR is zero (ties, plateaus), use the median absolute deviation, then the range.
3. **Train on every prefix with the same rule.** PFN training contexts have every size n, each standardized by its own statistics, so each forward pass is self-consistent.
4. **Map beliefs back to original units.** Under an output-invariant prior, the standardized data are a maximal invariant and the predictive is equivariant:

```latex
p(y^\ast \mid x^\ast, D) = \frac{1}{\hat b(D)}\, \tilde p\Big(\frac{y^\ast - \hat a(D)}{\hat b(D)} \,\Big|\, x^\ast, \tilde D\Big)
```

This holds exactly under invariance and approximately for the PFN. A change of normalization changes only the units of the output, not the belief. Mapped back to original units, the belief does not jump when â and b̂ move.

**Consequences for the rest of the system.**

- **Features** are computed at each step in that step's units from the current D. History features (change of m(D), steps since it rose, recent step sizes) are recomputed from stored original-unit values, never kept in old units.
- **Both heads** use the same transform: the latent target f is standardized with the context statistics of y.
- **Fantasies and the martingale self-test** are done in original units.
- **Labels and evaluation** are ground-truth normalized regret in instance units and are unaffected.
- **Bar support** must cover heavy tails in standardized units; bins come from quantiles of y′ over the prior, at all context sizes.

**Check.** Verify equivariance empirically: shift and rescale a held-out context and confirm the back-mapped predictive is unchanged up to PFN error, at small and large n.

### 7.5 Diagnostics for prior quality

- **Prior predictive checks against real suites:** distributions of the top-gap ratio (f\* − q₀.₉₅) / IQR, number of local optima on random 2-D slices, effective dimension from sensitivities, volume within 5% of the optimum, and replicate-based noise-to-signal where available.
- **Prequential log score:** the PFN's sequential log predictive density on real optimization traces, per prior version.
- **Fixed-policy regret:** latent EI on HPO-B, Bayesmark and PD1 per prior version, as in PFNs4BO.
- **Rule:** a change is kept only if it improves at least one of these without hurting in-prior calibration.

### 7.6 Interaction with the policy

Labels depend on the PFN through π\_0, so every prior change invalidates them. Develop P0 to P3 with cheap fixed-policy evaluations, freeze the prior, then generate labels. Robustness of the value head is measured leave-one-component-out: train on a version without a component, test with it.

## 8. Evaluation, baselines and gates

The decisive experiment runs before any value head is trained: a model-based brute-force improvement step must beat latent EI by a pre-registered margin, or the line stops.

### 8.1 Protocol

- Paired runs on the same held-out instances with shared noise and shared initial designs; per-instance paired differences in final normalized regret.
- Hierarchical bootstrap over instances, then seeds; report mean, median, win rate and CVaR₀.₉.
- Stratify by SNR, noise regime, effective dimension, d and T.
- Anything tuned (thresholds, selection among checkpoints) is tuned on calibration instances only.

**Generalization is reported explicitly:** across budgets outside the training range of T; across d when one model covers several; leave-one-component-out (train without a noise family or structure, test with it); held-out structures (§7.2); and external suites.

### 8.2 Baselines

| Baseline | Why |
| --- | --- |
| Random search, Sobol | luck floor |
| Latent EI (π\_0) | base policy; the floor of π\_1 |
| EI on best observed y, UCB at two β, PI | standard heuristics |
| KG by re-encoding on the PFN | the strongest one-step non-myopic baseline; expensive |
| GP-Hedge over the heuristic set; SAWEI-style weighted EI | online adaptation baselines |
| Best open-loop schedule (selected on training instances) | budget awareness without state |
| Option selector (fitted Q over heuristics) | ablation: does the value head beat choosing among heuristics? |
| Exact GP acquisitions (EI, noisy EI, KG) on the GP sub-prior | gold standard where the exact posterior is available |

### 8.3 Two ceilings, kept apart

- **Model-based ceiling (deployable, the gate):** at each step, branch the candidates, simulate continuations with y drawn from the PFN's own noisy head, continue with π\_0, and pick the best mean. This uses only F\_t-measurable information, so it is a legitimate (expensive) policy.
- **Clairvoyant ceiling (diagnostic only):** the same with continuations on the true ω. It selects on the instance's own outcomes, so it is an upper bound and never a result. The gap between the two measures how much PFN error the value head could correct.

**Advantage map: the cheap gate.** Instead of running the full model-based ceiling policy, estimate at one sampled state per π\_0 rollout the model-based advantage of the best candidate over x\_0 (PFN-simulated continuations, shared noise, cross-fitted: select the best candidate on one set of continuation seeds, score it on another). By the performance difference lemma (§3.6), the mean of this quantity is a first-order estimate of the achievable gain. It costs about one label per instance, so 5000 instances per d are affordable. The full ceiling policy on 300 instances confirms it.

### 8.4 Gates

| Stage | Experiment | Gate (pre-registered) | Estimate it passes |
| --- | --- | --- | --- |
| 0 | PFN on the frozen prior: calibration, equivariance check, martingale self-test, noise-regime identifiability probes against t/T | 90% coverage within ±5 points; noise regime identifiable by t/T ≈ 0.2 | 80% |
| 1a | Advantage map, 5000 instances per d | cross-fitted mean best advantage implies at least 5% lower median normalized regret over the run | 60% noise-free, 75% with noise |
| 1b | Full model-based ceiling policy against latent EI, 300 instances per d | confirms 1a: paired 95% interval excludes zero | given 1a |
| 2 | Value head (round 1), at least 5000 fresh test instances per d | recovers at least half of the Stage 1 gain | 60% given Stage 1 |
| 3 | Comparison with KG by re-encoding | report either way: beat, match at lower cost, or lose | 45% beat |
| 4 | Round 2, ablations (no budget input, no noise features, concatenation instead of FiLM, option selector), external suites, leave-one-component-out | report | — |

Estimates are judgment calls, not results. A failed Stage 1 is itself a reportable result: by §3.6, latent EI is then near-optimal for this prior, and RL on top of the PFN is not worth its cost for priors of this kind.

## 9. From one step to an optimal policy

Repeating the improvement step is policy iteration, whose fixed point is the Bayes-optimal policy for the prior, conditioned on context, t and T. It is feasible with the same supervised pipeline and far more stable than RL from scratch. With approximation it approaches the optimum without reaching it exactly; how close it gets is measurable.

### 9.1 The loop in plain terms

Each round asks: "if I pick x now and afterwards behave like my *current* policy, how good is the final result?" The answer is learned from ground-truth rollouts, and the new policy picks the best x wherever the gain is credible.

```latex
\pi_{k+1}(s) = \arg\max_{x \in C(s)} \big[\hat Q^{\pi_k}(s, x) - c\,\hat\sigma(s, x)\big] \ \text{if that bound beats } \pi_k(s), \ \text{else } \pi_k(s)
```

Round 0 is latent EI; round 1 is the method of §3. Every later round repeats §5.2–5.4 with π\_k in place of π\_0: rollouts of π\_k on fresh instances, one branch state each, candidates labeled by querying them and continuing with π\_k, regression, calibration. Each round costs about as much as round 1, because a π\_k step (one encoding plus the head) costs about as much as a latent-EI step (one encoding plus the inner optimizer).

[embedded content (diagram, not exported): approximate policy iteration · one round per lap\]

Only the greedy step changes the policy; validation decides whether the change is kept.

### 9.2 Why it converges, and what breaks with approximation

- **Exact case.** With exact Q-values and all actions available, each round is at least as good as the previous one (policy improvement theorem). Over a finite horizon the loop reaches the optimal policy in at most T rounds, and in practice in far fewer.
- **The target is the true prior, not the PFN.** Labels are ground-truth returns on prior samples, so the fixed point is Bayes-optimal under the true prior. PFN errors only shape the inputs; unlike lookahead in the model, they do not compound into the target.
- **Within a round, the target is fixed.** The labels describe a policy that does not change while they are fitted. This is the main stability advantage over RL.
- **Between rounds, instability can appear.** With regression error, the monotone guarantee is lost, and the error bound grows roughly as T² times the per-round regression error. The typical failure is oscillation: the head overestimates an unusual action, the new policy takes it, the next round's labels show it was bad, and the policy switches back.

### 9.3 Making it stable

- **Conservative steps:** deviate only where the lower confidence bound beats the current policy's pick (§3.3).
- **Mixing:** optionally act with π\_{k+1} with probability α and π\_k otherwise, increasing α as validation allows (conservative policy iteration).
- **Accept-or-reject:** a round is kept only if it beats π\_k on held-out instances, paired under shared noise; otherwise π\_k stays. Keep the best policy so far.
- **Fresh labels on the new policy's states:** each round labels states that π\_k visits, which fixes the distribution shift of the previous round.
- **Bootstrapped targets** (k-step returns plus the learned V) cut rollout cost; adopt them only after checking them against full rollouts on a subset, because they trade variance for bias.

### 9.4 What stands between the loop and the true optimum

| Limit | Why it binds | Fix or measurement |
| --- | --- | --- |
| Candidate coverage | the policy can only pick from C(s); the optimum may lie outside it | the head is differentiable in x through the PFN query path: refine the top candidates by gradient ascent on Q̂ (per-candidate head, other candidates fixed); coverage diagnostic against a dense search |
| Readout sufficiency | the cache determines the marginal field, but the optimal Q may need joint information (how a query at x moves beliefs at x′) | the head also sees query states z(x) and pooled context states; probe whether KG or p(f\* \| D) can be decoded from them; add a joint head if not |
| Function approximation | Q must generalize across contexts, budgets, noise regimes and d | scale data and model; report held-out regression error per round |
| Prior | "optimal" means optimal for the training prior | prior versions and external suites (§7); leave-one-component-out |

The defensible claim is therefore: optimal within the candidate class and the readout, up to regression error, for this prior.

### 9.5 Other routes to the same fixed point

| Route | How | Verdict |
| --- | --- | --- |
| RL fine-tuning from π\_k | PPO or leave-one-out policy gradient on a softmax over (Q̂ − cσ̂)/temperature, KL penalty toward π\_k, fitted Q or V as critic | right tool once rounds stop improving; starts from a strong policy, which removes the "no floor" problem of §0.3 |
| Expert iteration | at training time improve actions by a depth-1 fantasy lookahead through the PFN plus V, then distill | useful teacher; depth 1 only, because model error compounds with depth |
| Fitted Q-iteration with max backups | regress Q\* directly with bootstrapped max targets | avoid first: overestimation and instability under function approximation |

### 9.6 How to tell we are close

1. **Rounds stop helping:** paired gains between successive rounds are indistinguishable from zero.
2. **The advantage map goes flat:** under π\_k, the cross-fitted advantage of the best candidate approaches zero. By the performance difference lemma (§3.6) this bounds the remaining gap to first order, with the caveat about which states are visited.
3. **An upper bound from the other side.** The clairvoyant ceiling (continuations on the true task) is a valid but loose upper bound. Information-relaxation duality (Brown, Smith & Sun 2010; from memory, to verify) tightens it: the clairvoyant sees the future but pays a penalty built from the learned V. The result is a certificate of the form "at most X% of regret left to gain", which would be the strongest result of the project.

### 9.7 Ladder after round 1

1. Rounds 2 and 3 of approximate policy iteration with conservative updates and accept-or-reject validation.
2. Gradient refinement of candidates through the head.
3. RL fine-tuning with a KL penalty toward the last accepted round, using the fitted critic.
4. The dual bound, to quantify the remaining distance to the optimum.

Expectation, not a result: most of the gain comes in rounds 1 and 2, followed by a plateau. Only the dual bound can say whether the plateau is near the optimum.

## 10. Risks, open decisions, references

### 10.1 Risks

| Risk | Effect | Mitigation |
| --- | --- | --- |
| Latent EI is already near the one-step improvement | Stage 1 fails | cheap to find out; a negative result is reportable |
| Label noise swamps small advantages | head learns nothing | shared noise; M = 4 seeds; label standard errors in the loss; increase M where advantages are small |
| Coverage: the best query is not among the candidates | ceiling capped | coverage diagnostic: compare the best labeled candidate against a dense search on a subset |
| Distribution shift: π\_1 visits states π\_0 did not | over-optimistic Â | threshold fallback; 20% off-policy branch states; round 2 |
| PFN error on BO contexts | wrong continuations in the model-based ceiling | the clairvoyant ceiling measures the gap; BO-like pretraining contexts |
| Compute underestimated | delay | pilot measures throughput before full label generation |
| Prior realism | small external gains | prior versions P3 to P5; claims scoped to in-prior first |

### 10.2 Open decisions

- [ ] Final SNR range after the pilot.
- [ ] Whether to drop F as the output scale (P1) before or after the first Stage 1 run on P0+noise.
- [ ] PFN size: 25M as specified, or smaller for faster labels.
- [ ] Number of labeled candidates per state (16) and continuation seeds (4) after the pilot's variance estimate.
- [ ] Whether summary heads for f\* and the median are added in Stage 0.
- [ ] Which real suites form the external evaluation: HPO-B, Bayesmark, PD1.
- [ ] (added 2026-11-11) Fine-tune the PFN between rounds on the non-anticipating trajectories that policy iteration produces anyway, mixed with prior contexts and queried anywhere via the ground truth: assessed in [0004](0004-pfn-finetuning-on-bo-trajectories.md) and [0005](0005-assessment-pfn-finetuning-on-rollout-contexts.md) (Tim; the reference version), not acted on.
- [ ] Output-invariance ranges for the affine map (a, b) and the warp family, and the standardization fallbacks for tiny contexts (§7.4).

### 10.3 References

Read in full this session: [Müller et al. (2023), PFNs4BO, ICML](https://arxiv.org/abs/2305.17535); [Bergna, Depeweg & Hernández-Lobato (2026), Decoupled PFNs](https://arxiv.org/abs/2605.06413).

Abstracts read:

- [Volpp et al. (2020), MetaBO, ICLR](https://arxiv.org/abs/1904.02642)
- [Hsieh, Hsieh & Liu (2021), FSAF](https://arxiv.org/abs/2106.04335)
- [Maraval, Zimmer, Grosnit & Bou Ammar (2023), NAP, NeurIPS](https://arxiv.org/abs/2305.15930)
- [Chen et al. (2022), OptFormer, NeurIPS](https://arxiv.org/abs/2205.13320)
- [Song et al. (2024), RIBBO](https://arxiv.org/abs/2402.17423)
- [Meindl et al. (2025), ZeroShotOpt](https://arxiv.org/abs/2510.03051)
- [Benjamins et al. (2023), SAWEI, AutoML](https://arxiv.org/abs/2306.04262)
- [Rakotoarison et al. (2024), ifBO, ICML](https://arxiv.org/abs/2404.16795)
- Brochu, Hoffman & de Freitas (2011), Portfolio Allocation for BO (GP-Hedge), arXiv:1009.5419, not opened.

From memory, to verify: Kakade & Langford (2002), conservative policy iteration; Frazier, Powell & Dayanik (2008), knowledge gradient; Letham et al. (2019), noisy EI; Bossek, Doerr & Kerschke (2020), initial design.

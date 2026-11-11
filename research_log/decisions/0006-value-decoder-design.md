<!-- Design refinement by Tim, 2026-11-11 ("Value decoder design: reading the PFN, candidates, conditioning, noise phases"), converted from .docx with pandoc, otherwise verbatim. Status: adopted as the next step after the api_round1 pilot (#13); Phase A (noise-free) is implemented on this branch, Phase B needs the noisy two-head PFN from pfn-repro. Refines 0003 §3.3, §4.1–4.5, §5.3, §8. -->

# Value decoder design: reading the PFN, candidates, conditioning, noise phases

Nov 11, 2026 · @Tim

The value head becomes a PFN-style query decoder: independent candidate
queries cross-attend to the frozen PFN's per-layer context states plus a
few state tokens, are conditioned on the budget through adaptive
LayerNorm, and output a bar distribution over the paired advantage over
latent EI's pick. The pipeline is built on a noise-free PFN first; the
noisy two-head PFN follows, where the expected headroom is larger.

## 1. Readout: all cached layers, with the decoder's own projections

Read the context tokens' residual-stream states at every PFN layer, not
only the last, and let the decoder learn its own key and value
projections. Whether this carries the information Q needs is an
empirical question, tested with probes before the decoder is built.

**Why all layers.** The final-layer context states are shaped for one
job: producing the predictive distribution at a query. Earlier layers
carry more generic information about each data point and its
neighbourhood. Reading every layer is cheap, because the cache already
exists.

**Why residual states rather than the PFN's cached keys and values.**
The cached keys and values live in a similarity space trained for the
PFN's own queries; a decoder reading them could only re-weight what
those values already carry. With its own projections on the residual
states, it can extract different information from the same
representation.

**Does the readout make sense, given the states were trained only for
the predictive?**

-   **Why it should work:** context self-attention is where the PFN
    builds its representation of the dataset. That representation must
    suffice to produce the predictive at every x, so it encodes much of
    the task: roughness, noise level, where the data lie, where values
    are high. A readout of EI from the PFN state already worked in
    earlier experiments, and PFN-style embeddings (e.g. TabPFN) are
    reused downstream successfully.

-   **Where it can fail:** quantities that need the joint posterior,
    such as KG or the location of x\*. NLL training on marginals does
    not force them to be decodable.

**Probe test (Stage 0, before the decoder).** Train small probes from
pooled context states, last layer only versus all layers, to
ground-truth targets:

| Target                                    | Kind                     | What success means                        |
|-------------------------------------------|--------------------------|-------------------------------------------|
| Noise level, heteroscedasticity           | marginal-level           | noise regime is readable                  |
| Roughness, effective dimension            | landscape                | landscape is readable                     |
| f\*                                       | joint                    | value of the optimum is readable          |
| Location of x\*                           | joint                    | location of the optimum is readable       |
| KG at candidates, computed by re-encoding | joint, decision-relevant | the readout carries lookahead information |

If last-layer probes fail and all-layer probes succeed, the all-layer
readout is confirmed. If both fail on joint targets, that is the case
for adding α-PFN information-gain features or a dedicated joint head.

## 2. Candidates: acquisition argmaxes on the same frozen cache

Every acquisition's argmax is found by gradient ascent through the
frozen query path on the step's single encoding; EI's pick x_0 must
reproduce the rollouts' choice exactly.

-   **How:** latent EI from 2048 Sobol points, the top 16 refined by
    gradient ascent through the query path; UCB, PI and the posterior
    mean the same way. All acquisitions share one encoding and are
    optimized jointly in one batch, so the cost is query-side only.

-   **Hard requirement:** x_0 must be bit for bit the point latent EI
    picks during the rollouts, with the same seeds and the same
    optimizer settings. Otherwise the advantage labels refer to a
    different base policy than the one the decoder is compared against.

-   **Other sources** (Sobol, local perturbations, replicate,
    carry-over) stay as in the main document.

## 3. Decoder: independent queries, memory tokens, budget conditioning

The separate MLP-plus-FiLM head and the set-transformer layers are
dropped. What remains is a PFN-style query decoder whose memory is
extended by a few state tokens, with budget conditioning built into its
normalization.

**No self-attention among candidates.** Q(s, x) depends only on the
state and x, not on which other candidates are in the set.
Candidate–candidate attention would make a score depend on its
neighbours in the set, which breaks consistency across candidate sets,
gradient refinement in x, and the policy-improvement argument.
Candidates are independent queries, exactly like PFN test tokens.

**Components.**

-   **Candidate tokens:** a linear embedding of \[x; the PFN's marginal
    features at x (latent μ, σ, quantiles, EI, PI, noise scale); source
    flag\]. This is the input embedding, replacing the former MLP.

-   **Memory, at every decoder layer:** the PFN's residual context
    states of the matching layer (own projections, §1), plus extra
    memory tokens:

    -   a **state token** with the budget: t/T, T − t, log T, n;

    -   one or two **statistics tokens** with the global readout: ∫PI,
        max EI, noise level, heteroscedasticity, effective dimension,
        progress;

    -   an **x_0 token** with EI's pick and its features, the reference
        for the advantage.

-   **Budget conditioning:** every decoder block is also conditioned on
    the budget through adaptive LayerNorm (scale and shift computed from
    the budget). This is FiLM in transformer form. It is kept for the
    budget because "worth more early, less late" is a multiplicative
    effect that should act on every token at every layer, not only when
    attention happens to read the state token.

-   **Output:** a bar distribution over the paired advantage a(s, x) for
    each candidate. x_0 is trained with target 0, so it stays anchored.

**Target and loss.** For candidate c and continuation seed k, the
single-run paired advantage is a_k(s, c) = R_T(x_0, k) − R_T(c, k), with
shared noise on seed k. The loss is the NLL of each a_k under the
predicted bar distribution (one target per seed, not the seed average),
with signed-log or quantile bins because mass concentrates near zero
with heavy tails. The minimizer is Law(a \| D, t, T, x), a posterior
quantity, so the target is legitimate under the clairvoyance rules.

**Decision rule.** Deviate from x_0 only if a low quantile of the
predicted advantage is positive; the quantile level replaces the
constant c of the current rule and is calibrated on held-out instances.
The quantile reflects run-to-run variability, not estimation error, so
calibration (or a small decoder ensemble) still guards against the
optimizer's curse.

**Size.** About 4 blocks at width 256, roughly 3M parameters; per step
it processes about 150 candidate queries against n ≤ 100 context states
per layer, negligible next to the encoding.

## 4. Noise-free first, then the noisy two-head PFN

Build and debug the pipeline on the existing noise-free PFN; move to the
noisy two-head PFN for the main result, where the expected headroom is
larger.

-   **Phase A, noise-free, single head (existing PFN).** Latent EI is
    plain EI and nothing needs decoupling. Use it to build and debug
    shared-noise rollouts, label generation, the decoder, the probes
    (§1) and the cheap advantage-map gate. The gate result is itself a
    lower bound on the planning gap from non-myopia and budget awareness
    alone.

-   **Phase B, noisy prior, two heads.** The decoupling costs one extra
    bar head on the same backbone and one extra loss term; the generator
    produces f anyway. This is where heuristics fail differently, where
    replication and distrust of single high observations pay, and where
    Decoupled PFNs saw its largest gains.

|                                                            | Noise-free                   | Heterogeneous noise                                                                    |
|------------------------------------------------------------|------------------------------|----------------------------------------------------------------------------------------|
| Stage 1 passes (at least 5% median gain over EI), estimate | about 55–60%                 | about 75%                                                                              |
| Sources of gain                                            | non-myopia, budget awareness | the same, plus noise handling and replication, plus a weaker baseline (EI under noise) |
| Extra cost                                                 | none                         | one head; a choice of signal-to-noise range                                            |

The estimates are judgment calls, not results. The main cost of the
noisy phase is designing the signal-to-noise range, not the
architecture.

## 5. Changes this implies for the main method document

Not yet applied to Amortized policy improvement for PFN-based BO:
method, training and prior:

-   [ ] §4.4: replace the feature MLP, FiLM and set transformer with the
    query decoder of §3; keep the feature-MLP head as an ablation.

-   [ ] §4.1–4.2: cache per-layer residual context states; add the probe
    test of §1 as a Stage 0 gate.

-   [ ] §5.3: replace the Gaussian NLL on seed-averaged labels with
    bar-distribution NLL on single-seed paired advantages.

-   [ ] §3.3 and §4.5: replace Â − cσ̂ with a calibrated low quantile of
    the advantage distribution.

-   [ ] §8: add Phase A (noise-free) and Phase B (noisy two-head) to the
    staged plan.

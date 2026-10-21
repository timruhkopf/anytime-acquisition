# References

Annotated literature for this project. BibTeX for every entry lives in `references.bib` under the same key; `just refs-check` verifies that both files agree.

**Status values:** `to-read` · `skimmed` · `read` · `cited` · `to-verify` (id or BibTeX not yet checked against the source)

**Entry template**

```markdown
### [key] Authors (Year). *Title*. Venue. DOI or arXiv id
- **Status:**
- **Claim:** what the paper shows, in one sentence, in our own words.
- **Relevance:** supports / contradicts / baseline / method source / dataset / evaluation — and why.
- **Used in:** configs, code modules, research log entries that depend on it.
- **Notes:** setup details we must match (splits, metric definitions, seeds, compute).
```

---

## Problem & motivation

### [mueller2022pfn] Müller, Hollmann, Pineda Arango, Grabocka, Hutter (2022). *Transformers Can Do Bayesian Inference*. ICLR 2022. arXiv:2112.10510
- **Status:** read
- **Claim:** A transformer trained on datasets sampled from a prior approximates that prior's posterior predictive in one forward pass (prior-data fitted networks).
- **Relevance:** method source: the reason tasks drawn from the training prior make the PFN a Bayes posterior, and why a policy can be amortized over the same prior.
- **Used in:** pfn-repro (all surrogates); `bo/tasks.py`

### [mueller2023pfns4bo] Müller, Feurer, Hollmann, Hutter (2023). *PFNs4BO: In-Context Learning for Bayesian Optimization*. ICML 2023. arXiv:2305.17535
- **Status:** read
- **Claim:** PFNs as BO surrogates, with acquisition functions computed from the bar distribution; random per-step mixing of acquisition functions beat EI on low-d spaces.
- **Relevance:** method source and closest motivation for switching; its mixing result is a hint the portfolio matters.
- **Used in:** pfn-repro `acquisitions/standard.py`; `bo/engine.py`

### [ngo2026lmabo] Ngo, Phan Trong, Nguyen, Gupta, Venkatesh (2026). *Adaptive Acquisition Selection for Bayesian Optimization with Large Language Models*. ICLR 2026. arXiv:2602.07904
- **Status:** read
- **Claim:** An LLM choosing among acquisition functions from a run summary beats fixed arms and simple portfolios on 50 benchmark problems.
- **Relevance:** contradicts (on the surface) the predecessor's null result; the difference is real heterogeneous benchmarks vs in-prior tasks (pfn-acqpolicy log §20).
- **Used in:** `research_log/decisions/0002-what-carries-over.md`; notebook 02 limits

### [liu2022rlabo] Liu, Qu, Liu, Lyu (2022). *Robust Bayesian optimization with reinforcement learned acquisition functions*. arXiv:2210.00476
- **Status:** skimmed
- **Claim:** PPO learns to switch among UCB exploration weights from GP-derived state features.
- **Relevance:** closest RL-switching baseline; trains and tests on the same function, so its gains are not held-out (pfn-acqpolicy log §20).
- **Used in:** README

## Closest related work

### [volpp2020metabo] Volpp, Fröhlich, Fischer, Doerr, Falkner, Hutter, Daniel (2020). *Meta-Learning Acquisition Functions for Transfer Learning in Bayesian Optimization*. ICLR 2020. arXiv:1904.02642
- **Status:** read
- **Claim:** An acquisition function parameterized by a neural network over GP features, trained with RL on related tasks, transfers within a task family.
- **Relevance:** the original learned acquisition policy; known for instability and overfitting to the training distribution, a risk an amortized-over-the-prior policy shares.
- **Used in:** README

### [maraval2023nap] Maraval, Zimmer, Grosnit, Bou Ammar (2023). *End-to-End Meta-Bayesian Optimisation with Transformer Neural Processes*. NeurIPS 2023. arXiv:2305.15930
- **Status:** skimmed
- **Claim:** A transformer neural process trained end-to-end with RL outputs both the surrogate and the acquisition values.
- **Relevance:** closest amortized acquisition with a transformer; differs from us by training the backbone jointly (ours is frozen) and by not testing switching.
- **Used in:** README

### [chen2022optformer] Chen, Song, Lee, Wang, Zhang, Dohan, Kawakami, Kochanski, Doucet, Ranzato, Perel, de Freitas (2022). *Towards Learning Universal Hyperparameter Optimizers with Transformers*. NeurIPS 2022. arXiv:2205.13320
- **Status:** skimmed
- **Claim:** A transformer trained on optimization traces imitates and can improve on the HPO algorithms that produced them.
- **Relevance:** amortized optimizer from data (not from a prior); context for the amortization argument.
- **Used in:** README

### [brochu2011hedge] Brochu, Hoffman, de Freitas (2011). *Portfolio Allocation for Bayesian Optimization*. UAI 2011. arXiv:1009.5419
- **Status:** read
- **Claim:** GP-Hedge: choose among acquisition functions online with a bandit over their past rewards.
- **Relevance:** the classic non-learned switching baseline; motivates the portfolio formulation.
- **Used in:** README; notebook 02

## Method building blocks

### [black2024pi0] Black et al. (2024). *π0: A Vision-Language-Action Flow Model for General Robot Control*. arXiv:2410.24164
- **Status:** read
- **Claim:** An action expert with its own weights attends jointly to a pretrained VLM's per-layer keys/values and outputs continuous actions.
- **Relevance:** method source for the PFN-VLA readout: the frozen PFN as backbone, an action expert over its context K/V.
- **Used in:** issue #2 (ei_readout)

### [bertsekas1997rollout] Bertsekas, Tsitsiklis, Wu (1997). *Rollout Algorithms for Combinatorial Optimization*. Journal of Heuristics 3, 245–262. doi:10.1023/A:1009635226865
- **Status:** read
- **Claim:** Simulating a base policy from each candidate action and taking the best gives a policy at least as good as the base (policy improvement).
- **Relevance:** method source for the switching controller: why its gain over B* lower-bounds optimal switching and is measured without selection bias.
- **Used in:** `bo/runner.py`; notebook 02

### [rakotoarison2026alphapfn] Rakotoarison, Adriaensen, Viering, Hvarfner, Müller, Hutter, Bakshy (2026). *α-PFN: Fast Entropy Search via In-Context Learning*. arXiv:2606.07134
- **Status:** read
- **Claim:** A second PFN trained on the first PFN's outputs amortizes entropy search into one forward pass.
- **Relevance:** method source for the `es` arm (`alpha_pes_large` in pfn-repro).
- **Used in:** pfn-repro `acquisitions/alpha_pfn`; `bo/engine.py`

### [hollmann2023tabpfn] Hollmann, Müller, Eggensperger, Hutter (2023). *TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second*. ICLR 2023. arXiv:2207.01848
- **Status:** read
- **Claim:** A PFN trained on a structural-causal prior does tabular classification in context, competitive with tuned GBDTs on small data.
- **Relevance:** background for PFN design choices.
- **Used in:** pfn-repro

### [hollmann2025tabpfn] Hollmann, Müller, Purucker, Krishnakumar, Körfer, Hoo, Schirrmeister, Hutter (2025). *Accurate predictions on small data with a tabular foundation model*. Nature 637, 319–326. doi:10.1038/s41586-024-08328-6
- **Status:** read
- **Claim:** TabPFN v2: row- and feature-attention with thinking rows scales PFNs to regression and larger tables.
- **Relevance:** method source for pfn-repro's TabPFN-v2-style surrogate (a possible stronger backbone).
- **Used in:** pfn-repro `surrogates/`

## Baselines

### [jones1998ego] Jones, Schonlau, Welch (1998). *Efficient Global Optimization of Expensive Black-Box Functions*. Journal of Global Optimization 13, 455–492. doi:10.1023/A:1008306431147
- **Status:** cited
- **Claim:** Expected improvement on a GP surrogate (EGO).
- **Relevance:** baseline arm `ei` (and its margin variants).
- **Used in:** pfn-repro `acquisitions/standard.py`; `static_tune`

### [srinivas2010ucb] Srinivas, Krause, Kakade, Seeger (2010). *Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design*. ICML 2010. arXiv:0912.3995
- **Status:** cited
- **Claim:** GP-UCB has sublinear regret with a schedule for the exploration weight.
- **Relevance:** baseline arms `ucb<q>` (here as quantiles of the PFN predictive).
- **Used in:** `static_tune`

### [wang2017mes] Wang, Jegelka (2017). *Max-value Entropy Search for Efficient Bayesian Optimization*. ICML 2017. arXiv:1703.01968
- **Status:** cited
- **Claim:** Entropy search on the optimal value f* instead of the optimizer x* is cheap and competitive.
- **Relevance:** baseline arm `mes`.
- **Used in:** pfn-repro `acquisitions/standard.py`

### [hernandezlobato2014pes] Hernández-Lobato, Hoffman, Ghahramani (2014). *Predictive Entropy Search for Efficient Global Optimization of Black-box Functions*. NeurIPS 2014. arXiv:1406.2541
- **Status:** cited
- **Claim:** Information gain about x* computed in the predictive (output) space.
- **Relevance:** the target the `es` arm (α-PFN, PES variant) amortizes.
- **Used in:** `static_tune`

## Datasets & benchmarks

Tasks are drawn from the PFN's own BNN prior (pfn-repro `priors/bnn.py`); no external datasets.

## Evaluation & statistics

### [bouthillier2021variance] Bouthillier et al. (2021). *Accounting for Variance in Machine Learning Benchmarks*. MLSys 2021. arXiv:2103.03098
- **Status:** read
- **Claim:** Ignoring sources of variation (data sampling, seeds, HPO) leads to wrong conclusions; randomize them and compare paired.
- **Relevance:** evaluation practice: paired comparisons over tasks, re-measuring selected configurations on fresh tasks.
- **Used in:** `analysis/bo.py`; notebooks 01–02

## Tools & software

PyTorch, Hydra, MLflow, uv (cite as each project recommends when a paper is written).

---

## Reading queue

- Lam, Willcox, Wolpert (2016). *Bayesian Optimization with a Finite Budget: An Approximate Dynamic Programming Approach*. NeurIPS 2016. Non-myopic BO via rollouts: the closest precedent for the controller. BibTeX not yet fetched (DBLP blocked the request); to-verify.

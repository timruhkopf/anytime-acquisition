<!-- Claude, 2026-11-13, recording Tim's idea from the discussion after option_policy (notebook 13) and ei_failure (notebook 15); issue #13. Status: open. First step: acq_distill (notebook 16). -->

# An acquisition branch on the frozen PFN: read the train-context activations, learn our own acquisition function

Nov 13, 2026 · Tim's idea, written up by Claude

## 1. The idea in short

The frozen PFN turns the data D into activations of its train-context tokens, one token per observed point, layer by layer. Those activations are enough to compute every predictive quantity at any query x; that is what the PFN's own query path does. So they should also be enough to compute an *acquisition value* at any x.

We add a second, trainable stream beside the PFN and leave the PFN untouched:

1. **Acquisition tokens, one per observed point.** Each starts from the location x_i of its point only.
2. **Layerwise cross-attention into the PFN.** In block b each acquisition token cross-attends to the PFN's train-context activations of layer b. Starting from its location, it can find "its" PFN token and pull out what matters for a decision at that point. The cross-attention weights are ours, so what is extracted is free to differ from what the PFN uses for prediction.
3. **Self-attention among the acquisition tokens.** The tokens talk to each other (and to a budget token). Global, decision-relevant quantities can form here: where the incumbent is, how settled the belief about the optimum is, how much budget is left.
4. **Query cross-attention on the branch.** A query x gets its own token (location + budget) and cross-attends to the acquisition tokens, block by block. This is the PFN's train/test pattern, repeated on the branch. Queries do not see each other, so the value of x depends on D, the budget and x only.
5. **Output: an acquisition value α(x | D, budget).** The policy maximizes it over x.

It is a PFN stacked on a PFN: the first describes the function, the second describes what to do about it. The α-PFN line of work amortizes one given acquisition function this way; the aim here is to amortize a *policy*, trained by a reward.

```
 data (x_i, y_i) ──► frozen PFN: train tokens, layer 1 … L          (unchanged)
                          │  cross-attention, per layer
 locations x_i ────► acquisition tokens ◄──► self-attention (+ budget token)   (trained)
                          │  cross-attention, per layer
 query x, budget ──► query token ──► α(x | D, budget) ──► argmax over x
```

## 2. How it is trained

1. **Distillation, in parallel across arms.** The output has one column per arm (EI, PI, UCB, MES, mean). Each column is trained to *rank* the candidates as its arm does. A ranking loss within the arm is scale-free, so no arm dominates because its values happen to be larger, and no mixture weight is needed. After this step the branch reproduces every reference policy, and its backbone holds the features all of them need.
2. **Improvement.** The per-arm head can be thrown away. A single ranking surface on the same backbone is then optimized towards a better policy reward, starting from the arms rather than from scratch.

## 3. What is decided, what is open

Decided with Tim (2026-11-13):
- Go to harder problems (d = 4); the remaining budget is always part of the policy's input.
- The context-dependent mixture of the arms' scores is a first showcase that the reward and the context can be used at all. It stays inside the convex hull of the arms (an amortized, context-aware GP-Hedge) and is not the end point.
- Horizons: a distribution over horizons as a curriculum. Almost no mass on one step, most on short to medium horizons at first, flattened as the policy improves. Short horizons teach effective sampling, long ones exploration.

Open:
- The reward for step 2: terminal log regret with which floor (recommended: one PFN bin, 1e-3, with 1e-4 and 1e-6 reported), and how efficiency enters (the horizon curriculum is one way).
- How step 2 gets its training signal: direct paired search, policy gradient with a privileged critic, or advantage labels at the candidates the arms pick.
- Whether the branch should also see the PFN's own query state at x (the ablation `query_state` in `experimental/acq_branch.py`). The idea above says the train-context activations suffice.

## 4. Evidence so far that bears on it

- The PFN's pooled last-layer state, reduced to 32 principal components, added nothing to hand-made belief features for predicting option advantages (notebook 13) or EI's outcome (notebook 15). Pooling is not this idea: the branch reads every token of every layer with learned attention.
- Belief features predict EI's outcome better than data statistics early in a run (+0.11 Spearman at t = 2, notebook 15), so the PFN's state does carry information beyond the raw data.

## 5. First step

`acq_distill` (notebook 16): can the branch reproduce every arm's ranking, on candidate sets, on dense query sets, and in closed loop? d = 1 first (plots of the arms and the learned scores over the whole domain), then d = 2 and d = 4.

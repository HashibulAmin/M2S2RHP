# Algorithm fixes implemented

This repository was audited against the supplied `PAPER.md` and the architecture-review attachment. The fixes below are implemented in code, not only documented.

## Data collection / generation

| Issue | Corrective implementation |
|---|---|
| Random independent city-distance matrix was not a metric | Cities are sampled as 2-D coordinates and distances are Euclidean, symmetric, non-negative, and zero on the diagonal. |
| Histories had random holes and could contain inactive padded events | Histories are chronological prefix-valid arrays with explicit masks. |
| Every user was effectively warm | A configurable cold-start cohort has empty training histories and separate test targets. |
| Skips could become BPR positives | BPR positives are sampled only from `hist_weight > 0`. |
| Held-out positives could be sampled as negatives | Negative candidates exclude training, validation, and test positives. |
| Random protected labels made the adversary meaningless | Protected labels are stable user attributes (`user_gender[uid]`). |
| Weak labels were independent of token features | Synthetic situation labels generate correlated marker/lexical token IDs. |
| Random graph/location bits did not represent the paper | City values are Bloom-encoded, locally randomized, then propagated on a city-homophilous social graph. |
| Duplicate graph edges reduced/biased the intended degree | Graph sampling explicitly targets unique undirected edges. |

## Data analysis / signal construction

- Behavioral features are called **engagement-style features**, not inferred OCEAN psychometrics.
- Signed implicit feedback is retained for interest/music vectors; positive-only feedback is used to infer current life situation.
- Temporal decay is applied to interaction history, with separate learned decay rates per situation category.
- Explicit self-declared situations use a floor so onboarding information does not vanish completely.
- Offline evaluation is chronological and excludes seen training positives from ranking candidates.

## Model architecture

| Issue | Corrective implementation |
|---|---|
| Static fusion weights | A learned per-user softmax gate mixes situation, city, interest, and geo signals. |
| Missing direct interest term | `S_interest = cosine(e_interest, reel_embedding)` is present in the final score. |
| Geo score dimensional ambiguity | Implemented as `sigmoid((W_g g_i)^T v_r)`. |
| Only 1-hop propagation | LightGCN-style propagation averages layers 0..K. |
| No cold-network fallback | Isolated users preserve their own layer-0 private city features. |
| Naive multi-bit RR privacy claim | RAPPOR-style permanent + instantaneous randomization is used, and privacy values are reported conservatively rather than overclaimed. |
| Demographics directly in scoring | Demographic variables only enter `MLP_fair`; the gate is adversarially trained. |
| Visual modality missing | Fixed synthetic visual features are fused into the situation classifier, replacing a trainable per-item visual ID embedding. |
| Unbounded trend term | Trend boost is normalized to `[0,1]` before a small mixing coefficient. |

## Training methodology

- BPR is used for pairwise ranking, with BCE for the multi-label situation classifier.
- Fairness uses an explicit minimax update rather than a fragile custom VJP: the adversary minimizes protected-attribute CE, while the fairness gate receives the reversed CE gradient.
- The adversary observes the fair embedding plus a stop-gradient copy of the candidate score, so its fairness signal cannot destabilize unrelated ranking towers.
- Adam is implemented in-repository to keep the prototype runnable with only JAX and NumPy installed.
- Global gradient-norm clipping is applied before each optimizer update.
- Exploration is not injected into the training loss; it belongs to the online serving policy.

## Validation

The repository includes executable invariants covering city-distance structure, history masking, temporal-decay monotonicity, exact-city preference, graph finiteness, and privacy-budget calculations. A full `run_experiment.py` execution produces finite losses and offline ranking diagnostics on the default synthetic world.

## Remaining research limitations

The text/visual encoder is still a tiny synthetic prototype, the LDP mechanism is not a production privacy library, peer matching is only a helper function rather than an ANN service, and the experiment does not establish real-world efficacy. Public-dataset integration with compatible user-item-context metadata remains required before making empirical claims.

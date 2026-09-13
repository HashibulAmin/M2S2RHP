# M²S²-Rec implementation review

## Scope

The project was reviewed module-by-module against the original `PAPER.md`, the supplied architecture-review document, and the behavior of the executable code. The audit covered data generation, feature semantics, model dimensions, privacy processing, ranking loss, fairness training, evaluation, and repository reproducibility.

## Highest-impact corrections

### 1. Pairwise-label integrity
The earlier generator could select a skipped interaction as a BPR positive and could sample held-out positives as negatives. This makes the ranking objective internally contradictory and leaks the evaluation protocol. The repaired generator maintains signed interactions, samples BPR positives only from positive feedback, and excludes all known positives from negative candidates.

### 2. Cold-start coverage
The previous implementation could describe cold-start behavior without actually exercising it. The repaired data generator creates users with empty training histories and evaluates them separately, while city priors and explicit situations remain available.

### 3. Spatial signal integrity
The original random city-distance matrix had no physical interpretation. The repaired world samples city coordinates and derives a valid Euclidean distance matrix. Social edges are undirected and city-homophilous, and the graph layer is explicitly K-hop.

### 4. Privacy claim discipline
Naive independent per-bit randomized response was not sufficient for a multi-dimensional privacy claim. The implementation now uses a RAPPOR-style two-stage mechanism and reports a per-bit epsilon plus a conservative n-bit composition bound. The code deliberately avoids claiming a tight vector-level epsilon.

### 5. Fairness optimization stability
A custom gradient-reversal primitive made the prototype numerically unstable under the available JAX runtime. The repaired code expresses the same minimax intent explicitly: the adversary minimizes cross-entropy, while only the fairness gate receives the reversed adversarial gradient. This makes the training behavior easier to inspect and test.

### 6. Cold-start-safe cosine similarity
The original clamped-norm formulation had an undefined derivative at an exact zero vector. Empty histories make zero vectors a real case, not an edge-case abstraction. Cosine similarity now uses `sqrt(sum(x^2) + eps)`, whose derivative is finite at zero.

### 7. Training/evaluation separation
The repaired experiment uses chronological user holdouts, excludes seen items from ranking, and computes Recall@K, NDCG@K, pairwise AUC, city-hit rate, cold-start recall, warm recall, and gender-group recall diagnostics.

## Current evidence

The default synthetic run completes with finite objective values and produces a non-trivial separation between warm and cold-start evaluation. These values are smoke-test evidence for implementation correctness, not external benchmark results.

## Recommended next scientific step

Integrate a public short-video/interactions dataset through a dedicated adapter, preserve the chronological and leakage controls implemented here, and benchmark against simple baselines (popularity, city-only, interest-only, two-tower, and static-weight fusion) before attributing gains to M²S²-Rec.

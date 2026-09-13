# M²S²-Rec

## Multi-Modal Social-Spatial Reel Recommendation

M²S²-Rec is a research-oriented recommendation architecture for short-form video that combines:

* City-anchored relevance
* Life-situation-aware personalization
* Temporally decayed user interests
* Multi-modal reel situation classification
* K-hop social-spatial graph propagation
* Local Differential Privacy (LDP)
* Adversarial fairness learning
* Dynamic per-user ranking gates
* Cold-start-aware recommendation
* Offline ranking evaluation

The implementation is based on the accompanying research paper included in this repository.

> **Research status:** This repository is a research prototype and experimentation framework. The included results are intended to validate the implementation and methodology on synthetic/public-style experimental data and should not be interpreted as evidence of production performance unless independently reproduced on appropriate real-world datasets.

---

## Repository Structure

```text
m2s2-rec-project/
│
├── README.md
├── LICENSE
├── NOTICE
├── PAPER.md
├── ORIGINAL_PAPER.md
├── ALGORITHM_FIXES.md
├── IMPLEMENTATION_REVIEW.md
├── requirements.txt
│
├── model/
│   ├── __init__.py
│   ├── m2s2rec.py
│   ├── layers.py
│   ├── adversary.py
│   ├── text_encoder.py
│   ├── scoring.py
│   └── ...
│
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   └── ...
│
├── tests/
│   └── ...
│
├── evaluate.py
├── losses.py
├── train.py
└── results/
    └── ...
```

---

## Architecture

The system represents a user using three primary components:

```text
User History
     │
     ├── Temporal Interest Representation
     │
     ├── Temporal Life-Situation Representation
     │
     └── Fairness-Audited Behavioral/Demographic Representation
                         │
                         ▼
                  Dynamic Gate Network
                         │
       ┌─────────────────┼─────────────────┐
       ▼                 ▼                 ▼
 Situation Match    City Match       Interest Match
       │                 │                 │
       └─────────────────┴─────────────────┘
                         │
                    Geo / Graph Signal
                         │
                         ▼
                  Final Reel Score
```

The final recommendation score combines situation relevance, city relevance, direct interest similarity, and graph-based geographic relevance through a learned per-user gate.

---

## Main Components

### 1. Temporal Interest Modeling

Recent positive interactions receive greater weight through exponential temporal decay.

Negative implicit feedback, such as skips or rapid scroll-away events, can carry negative interaction weights.

The model therefore distinguishes between:

```text
positive engagement
negative engagement
old engagement
recent engagement
```

rather than treating every historical interaction equally.

---

### 2. Life-Situation Classification

Each reel is assigned a multi-label probability vector over the repository's life-situation taxonomy.

The prototype supports signals such as:

```text
job_search
new_parent
moving
graduation
```

The classifier is intentionally implemented as a research-friendly lightweight encoder rather than claiming production-level ASR/OCR/vision performance.

Weak supervision can be provided through hashtag-derived labels.

---

### 3. Temporal Life-Situation Modeling

Life situations do not necessarily decay at the same rate.

For example, a short job-search episode may decay faster than a longer-lived life transition.

The implementation therefore supports per-category decay rates instead of a single universal decay coefficient.

---

### 4. City-Anchored Recommendation

The primary spatial signal is an explicit city match.

Exact city matches receive the strongest spatial score, while nearby cities receive a distance-based falloff.

This prevents the system from treating an entire region as equally relevant when the intended product behavior is explicitly city-oriented.

---

### 5. K-Hop Social-Spatial Graph

User location information is propagated through a social graph using a LightGCN-style normalized aggregation.

The implementation supports:

```text
0-hop  → user's own city/location
1-hop  → direct connections
2-hop  → friends-of-friends
...
K-hop  → configurable propagation depth
```

A zero-degree user retains the direct location representation as the cold-network fallback.

---

### 6. Privacy Mechanism

The spatial representation uses a RAPPOR-style randomized-response approach rather than independently perturbing every multi-hot bit under the same nominal epsilon.

This is important because naive independent randomized response across many dimensions creates a composition problem for the overall privacy budget.

The repository documents the privacy assumptions and accounting used by the implementation.

---

### 7. Adversarial Fairness

Protected demographic variables are not directly used as an unrestricted scoring feature.

Instead, they can influence a fairness-audited representation through an adversarial objective.

The implementation uses an explicit minimax-style training formulation so that:

```text
Adversary:
    learns to predict the protected attribute

Fair representation:
    learns to reduce predictable protected information
```

This separates demographic conditioning from direct unrestricted ranking logic.

---

### 8. Dynamic Ranking Gate

The ranking weights are user-dependent instead of fixed global coefficients.

Conceptually:

```text
[α, β, γ, δ] =
    softmax(
        GateNetwork(user_representation)
    )
```

The four signals correspond to:

```text
α → life-situation relevance
β → city relevance
γ → interest relevance
δ → social-spatial / geographic relevance
```

This allows users with limited history to rely more heavily on contextual priors while experienced users can rely more strongly on learned interest signals.

---

## Training

The prototype uses a ranking objective based on Bayesian Personalized Ranking (BPR), together with auxiliary objectives.

The training objective combines:

```text
BPR ranking loss
+
Life-situation classification loss
+
Adversarial fairness loss
+
L2 regularization
```

The implementation also uses gradient clipping and numerically stable similarity calculations to avoid instability for cold-start users and zero representations.

---

## Data Handling

The project deliberately avoids treating randomly generated interaction triples as a valid evaluation protocol.

The data pipeline separates:

```text
training interactions
validation interactions
test interactions
```

using chronological ordering.

Additional safeguards include:

* Positive samples must correspond to actual positive engagement.
* Skipped/negative events are not silently converted into positives.
* Held-out test positives are not reused as negative samples.
* Cold-start users are evaluated separately.
* Warm users are evaluated separately.
* Group-level diagnostic statistics are computed separately from ranking quality.

The included synthetic data generator is intended for reproducible development and software validation, not for claiming real-world effectiveness.

---

## Evaluation

The evaluation framework reports ranking and diagnostic metrics such as:

```text
Recall@K
NDCG@K
Pairwise AUC
City-hit rate
Cold-start Recall@K
Warm-user Recall@K
Group-level recommendation diagnostics
```

Example:

```bash
python evaluate.py
```

Evaluation results should be interpreted as experimental results for the specified dataset/configuration only.

Do not generalize synthetic-data results to production populations without additional validation.

---

## Installation

Python 3.10+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Training Pipeline

Run:

```bash
python train.py
```

The training script will:

1. Construct the experimental world/data.
2. Build the non-learnable context.
3. Initialize model parameters.
4. Train the recommendation model.
5. Report ranking/classification/fairness losses.
6. Report learned gate weights.

---

## Running Tests

Run the complete test suite with:

```bash
pytest -q
```

The tests cover important implementation-level behavior including:

* Temporal decay
* City scoring
* Cosine similarity stability
* Graph propagation
* Sampling constraints
* Ranking losses
* Core model shape consistency

---

## Reproducibility

The project exposes configuration parameters for:

```text
embedding dimensions
number of users
number of reels
number of cities
history length
graph depth
privacy parameters
decay rates
learning rate
batch size
training epochs
random seed
```

For reproducible experiments, always record:

```text
dataset/version
configuration
random seed
code commit
evaluation protocol
```

The repository includes the commit corresponding to the completed implementation.

---

## Important Research Limitations

This repository is not presented as a production-ready TikTok/Instagram-style recommendation system.

The current prototype does not claim to provide:

* Production-scale ASR
* Production OCR
* Production video understanding
* Large-scale distributed training
* Real-time graph recomputation
* Real-world LDP deployment certification
* Regulatory compliance certification
* Guaranteed demographic fairness
* Guaranteed recommendation quality on arbitrary datasets

The architecture is designed as a research implementation that can be extended toward those settings.

---

## Paper

The original research paper is included in:

```text
PAPER.md
```

A preserved copy is also available as:

```text
ORIGINAL_PAPER.md
```

### Paper Copyright

The paper included in this repository is the author's original research work.

**The presence of the paper in this repository does not automatically place the paper under the Apache License 2.0.**

Unless otherwise stated by the author, the paper remains subject to the author's copyright and any separate publication or licensing terms that may apply to it.

The Apache License 2.0 applies to the source code and other repository materials explicitly identified as being licensed under Apache-2.0.

---

## Code License

Copyright © 2026 The M²S²-Rec Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this source code except in compliance with the License.

You may obtain a copy of the License at:

```text
http://www.apache.org/licenses/LICENSE-2.0
```

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the `LICENSE` file for the complete license text.

---

## Citation

When referring to the M²S²-Rec research work, please cite the paper included in this repository according to its publication/citation metadata.

A BibTeX entry can be added here once the paper has an official publication record:

```bibtex
@article{m2s2rec,
  title   = {M²S²-Rec: A City-Anchored, Life-Situation-Aware Architecture for Short-Form Video Recommendation},
  author  = {Md Hashibul Amin},
  year    = {2026},
  note    = {Research manuscript; Copyright © 2026 Md Hashibul Amin. All rights reserved.}
}
```

Replace the placeholder author/publication metadata with the authoritative citation information before publication.

---

## Third-Party Dependencies

This repository may depend on third-party software distributed under their own licenses.

See:

```text
requirements.txt
```

and the corresponding upstream projects for their respective licensing terms.

Apache-2.0 licensing of this repository does not supersede the licenses of third-party dependencies.

---

## Contribution

Contributions to the **code** are welcome under the terms described by the Apache License 2.0.

Contributions that modify or reproduce the accompanying paper should not be assumed to have permission to redistribute the paper unless the author explicitly permits it.

---

## Disclaimer

This repository is intended for research and educational purposes.

Recommendation systems can affect users differently across demographic groups and geographic communities. The fairness, privacy, and safety mechanisms implemented here should be independently audited before any real-world deployment.

The authors make no guarantee that the prototype is suitable for a particular production environment, jurisdiction, dataset, or user population.

---

## License Summary

| Repository component                        | License / terms                     |
| ------------------------------------------- | ----------------------------------- |
| Source code                                 | Apache License 2.0                  |
| Tests                                       | Apache License 2.0                  |
| Evaluation code                             | Apache License 2.0                  |
| Data-generation code                        | Apache License 2.0                  |
| Documentation created as code documentation | Apache License 2.0                  |
| `PAPER.md`                                  | Author's copyright / separate terms |
| `ORIGINAL_PAPER.md`                         | Author's copyright / separate terms |
| Third-party dependencies                    | Their respective licenses           |

---

## Contact

For research questions, implementation issues, or collaboration regarding M²S²-Rec, please use the contact information associated with the paper/repository.

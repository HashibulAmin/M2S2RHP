# M²S²-Rec

**Multi-Modal Social-Spatial Reel Recommendation** — a reproducible JAX prototype of the city-anchored, life-situation-aware architecture described in `PAPER.md`.

The repository is intentionally framed as a **research prototype on synthetic data**. It does not claim public-dataset or production performance.

## What is implemented

- Temporal user interest and music vectors with signed implicit feedback and recency decay.
- A multi-label life-situation classifier over caption/OCR/ASR-style token streams plus fixed synthetic visual features.
- Explicit onboarding situation floors with per-situation learned decay rates.
- Exact city matching plus distance-aware near-city fallback.
- RAPPOR-style local randomization before K-hop LightGCN social-spatial propagation.
- Demographic information isolated in a fairness gate instead of being fed directly to item scoring.
- Explicit minimax adversarial training: the protected-attribute adversary learns normally while the fairness gate receives the reversed objective.
- Learned per-user softmax gating across situation, city, interest, and geo signals.
- A bounded personalization-independent recency/trending feature.
- Training-only exploitation objective; exploration is kept out of BPR optimization to avoid changing the training target.
- Chronological user holdouts, leakage-aware negative sampling, and offline Recall@K / NDCG@K / pairwise AUC diagnostics.

## Important files

```text
m2s2-rec-project/
├── PAPER.md                         # original paper supplied in the archive
├── ALGORITHM_FIXES.md               # detailed issue/fix table
├── IMPLEMENTATION_REVIEW.md         # audit of data, architecture, methodology, training
├── requirements.txt                 # JAX + NumPy only
├── train.py                          # training entry point
├── evaluate.py                       # offline evaluator
├── run_experiment.py                 # train + evaluate + JSON artifact
├── losses.py                          # ranking/classification/fairness losses
├── model/
│   ├── layers.py                     # layers + stable Adam optimizer
│   ├── adversary.py                  # fairness gate/adversary
│   ├── text_encoder.py               # transformer-lite multimodal encoder
│   ├── scoring.py                    # temporal/geographic/trend primitives
│   └── m2s2rec.py                    # full model wiring
├── data/
│   ├── dataset.py                    # synthetic event log + train/val/test split
│   ├── graph.py                      # K-hop LightGCN propagation
│   └── privacy.py                    # Bloom/RAPPOR-style LDP
└── tests/test_core.py                # executable invariants
```

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the unit tests:

```bash
python -m unittest discover -s tests -v
```

Run training only:

```bash
python train.py --epochs 2
```

Run the full reproducible toy experiment:

```bash
python run_experiment.py --epochs 2 --output artifacts/results.json
```

The default configuration is deliberately small enough for CPU execution. Larger public datasets should be integrated through a separate data adapter rather than changing the synthetic generator into a pseudo-real dataset.

## Experimental protocol

Interactions are chronological. Warm users have a train history plus validation/test positive targets; cold-start users have no training history. BPR positives are sampled only from positive-engagement events, and negatives avoid known positive items, including held-out targets. This prevents the original prototype's positive/negative contamination.

Weak situation labels are intentionally correlated with synthetic text markers so the classifier has a learnable signal. The visual features are fixed, situation-correlated features rather than trainable item-ID embeddings, so the multimodal path cannot simply memorize every reel.

The social graph is refreshed once per synthetic world build, representing the paper's batched/offline graph update rather than real-time propagation.

## Privacy note

`data/privacy.py` exposes both the per-bit privacy loss of the composed binary channel and a conservative basic-composition upper bound across the Bloom bits. It does **not** claim that the toy mechanism has a single tight vector-level epsilon. The deterministic memoization surrogate is for reproducibility only; a production service should use a secret keyed PRF.

## Results semantics

The included JSON result is a smoke-test/illustrative run on synthetic data. Metrics must not be presented as MicroLens, TikTok, or production results. The repository is suitable as an architecture implementation and experiment harness; the next empirical step is integration with a genuinely collected public dataset with compatible user/item metadata.
# M2S2RHP

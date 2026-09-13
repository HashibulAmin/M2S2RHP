# M²S²-Rec

**Multi-Modal Social-Spatial Reel Recommendation** — a reproducible JAX prototype of the city-anchored, life-situation-aware architecture described in `PAPER.md`.

The repository is intentionally framed as a **research prototype on synthetic data**. It does not claim public-dataset or production performance.

## What is implemented

- Temporal user interest vectors with signed implicit feedback and recency decay.
- Lyrics + acoustic music encoder and playlist-aware user music profiles.
- A multi-label life-situation classifier over caption/OCR/ASR-style token streams plus fixed synthetic visual features.
- Explicit onboarding situation floors with per-situation learned decay rates.
- Exact city matching plus distance-aware near-city fallback.
- RAPPOR-style local randomization before K-hop LightGCN social-spatial propagation.
- Demographic information isolated in a fairness gate instead of being fed directly to item scoring.
- Explicit minimax adversarial training: the protected-attribute adversary learns normally while the fairness gate receives the reversed objective.
- Learned per-user softmax gating across situation, city, interest, and geo signals.
- A bounded personalization-independent recency/trending feature.
- A fifth dynamic ranking signal for playlist/music relevance.
- Training-only exploitation objective; exploration is kept out of BPR optimization to avoid changing the training target.
- Chronological user holdouts, leakage-aware negative sampling, and offline Recall@K / NDCG@K / pairwise AUC diagnostics.

## Important files

```text
m2s2-rec-project/
├── PAPER.md                         # updated research manuscript
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

The music-specific tests verify stable lyrics tokenization, playlist weighting, playlist-driven music scoring, and prevention of future playlist leakage.

Run training only:

```bash
python train.py --epochs 2
```

Run the full reproducible toy experiment:

```bash
python run_experiment.py --epochs 2 --output artifacts/results.json
```

The default configuration is deliberately small enough for CPU execution. Larger public datasets should be integrated through a separate data adapter rather than changing the synthetic generator into a pseudo-real dataset.

## Music lyrics and playlist recommendation

Music is a first-class signal in the implementation. Each track has a bounded lyrics-like token sequence and acoustic feature vector. `model/music_encoder.py` encodes both modalities into a shared music embedding. The prototype deliberately uses synthetic token sequences and does not distribute copyrighted song lyrics.

Each user also has a playlist representation. The playlist is converted into a weighted mean of track embeddings and combined with music evidence inferred from the user's reel history:

```text
user music profile
    = history music embedding
    + playlist_mix × playlist embedding
```

A candidate reel gets a music score by cosine similarity between its associated track embedding and the user's combined music profile. The dynamic ranking gate now has five weights:

```text
Situation | City | Interest | Geo | Music/Playlist
```

The data generator also includes `track_lyrics`, `track_audio`, `reel_track`, `playlist_track`, and `playlist_weight`. `data/music.py` additionally provides a deterministic lyrics tokenizer and a timestamp-safe playlist builder for real-data adapters. In a real integration, replace these synthetic fields with licensed/consented track metadata, lyrics or lyric embeddings, acoustic embeddings, and timestamped playlist events. Playlist events must be cut off at the prediction timestamp to avoid temporal leakage.

## Experimental protocol

Interactions are chronological. Warm users have a train history plus validation/test positive targets; cold-start users have no training history. BPR positives are sampled only from positive-engagement events, and negatives avoid known positive items, including held-out targets. This prevents the original prototype's positive/negative contamination.

Weak situation labels are intentionally correlated with synthetic text markers so the classifier has a learnable signal. The visual features are fixed, situation-correlated features rather than trainable item-ID embeddings, so the multimodal path cannot simply memorize every reel.

The social graph is refreshed once per synthetic world build, representing the paper's batched/offline graph update rather than real-time propagation.

## Privacy note

`data/privacy.py` exposes both the per-bit privacy loss of the composed binary channel and a conservative basic-composition upper bound across the Bloom bits. It does **not** claim that the toy mechanism has a single tight vector-level epsilon. The deterministic memoization surrogate is for reproducibility only; a production service should use a secret keyed PRF.

## Results semantics

The included JSON result is a smoke-test/illustrative run on synthetic data. Metrics must not be presented as MicroLens, TikTok, or production results. The repository is suitable as an architecture implementation and experiment harness; the next empirical step is integration with a genuinely collected public dataset with compatible user/item metadata, music metadata, and legally usable lyrics/audio representations.


## License and paper ownership

Source code, tests, evaluation code, and implementation documentation are licensed under the Apache License 2.0. See `LICENSE`.

The accompanying research manuscript is authored and owned by **Md Hashibul Amin**. `PAPER.md` and `ORIGINAL_PAPER.md` are not automatically relicensed under Apache-2.0 merely because they are included in this repository. See `NOTICE` for the separation of code and manuscript rights.

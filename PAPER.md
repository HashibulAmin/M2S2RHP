# M²S²-Rec: A City-Anchored, Life-Situation-Aware Architecture for Short-Form Video Recommendation

## Abstract

Short-form video platforms traditionally rely on collaborative filtering and content-based filtering in isolated silos, often failing to leverage social-spatial signals and situational context. This position paper proposes a recommendation architecture that shifts the paradigm from diffuse regional music-affinity to city-anchored, life-situation relevance. By integrating a multi-modal life-situation classifier, temporal decay mechanisms for explicit and implicit interactions, and a $K$-hop spatial graph neural network guarded by Local Differential Privacy (LDP), the proposed framework addresses both cold-start and hyper-local routing challenges. Furthermore, we introduce an adversarially-trained fairness gating mechanism to mitigate demographic bias. By utilizing a dynamically learned per-user gating mechanism rather than static fusion weights, the model proactively adapts to varying user contexts.

## 1. Introduction

The consumption of short-form video content is defined by rapid interactions, heavy reliance on audio-visual synchronicity, and regional trend clusters. Recommendation architectures historically treat user history and visual features as separate entities, neglecting the physical topology of the user base and the immediate life circumstances of the consumer.

This architecture paper establishes a blueprint for M²S²-Rec, a system designed to anchor video delivery to a user's specific city and their transient life situations (e.g., job transition, new parenthood, relocation). By modeling these dynamics, the architecture answers:

1. **Who is watching, and what are they going through?** Evaluated via temporal life-situation vectors.
2. **Where are they consuming?** Modeled through an exact city-match score and a multi-hop social-spatial Graph Neural Network (GNN).
3. **Is the content contextually relevant?** Evaluated via a multi-label life-situation classifier parsing audio, text, and visual signals.

## 2. Related Work

### 2.1 Collaborative Filtering and Sequence Modeling

Standard collaborative filtering and sequence models prioritize recent interactions but can struggle with sparse short-form histories and cold-start routing. The proposed system therefore combines temporally decayed interaction signals with content-derived representations.

### 2.2 Graph Neural Networks and Spatial Recommendation

LightGCN-style propagation provides a simple and efficient mechanism for higher-order graph aggregation. In M²S²-Rec, graph propagation is applied to privacy-preserving location features and is treated as a secondary signal to the primary exact-city match.

### 2.3 Multimodal Recommendation and Music-Lyrics Modeling

Reels contain captions, OCR text, ASR transcripts, audio, and associated music. The proposed architecture uses text and visual context for life-situation classification and a dedicated music encoder for lyrics and acoustic features. User playlists provide an additional explicit signal for modeling persistent music preference.

## 3. Methodology

### 3.1 User Vectorization and Adversarial Fairness

Let $u_i$ denote user $i$. Demographic attributes and behavioral engagement features are transformed into a fairness-audited representation rather than being passed directly to the final scoring function.

$$
\mathbf{b}_i = \mathrm{MLP}_{\mathrm{fair}}\left(\mathbf{d}_i \Vert \mathbf{p}_i\right).
$$

User interest is represented by a temporally decayed aggregation of interacted reel embeddings.

$$
\mathbf{e}_{i,\mathrm{interest}}(t)
=
\sum_{k \in \mathcal{H}_i^T}
 w_{ik}
 \exp\left[-\lambda_t(t-t_k)\right]
 \mathbf{m}_k.
$$

The corresponding style representation uses the same decay process over style embeddings.

$$
\mathbf{e}_{i,\mathrm{style}}(t)
=
\sum_{k \in \mathcal{H}_i^T}
 w_{ik}
 \exp\left[-\lambda_t(t-t_k)\right]
 \mathbf{a}^{\mathrm{style}}_k.
$$

The complete user state is

$$
\mathbf{x}_i
=
\left[
\mathbf{e}_{i,\mathrm{interest}}(t)
\Vert
\mathbf{e}_{i,\mathrm{music}}(t)
\Vert
\mathbf{b}_i
\right].
$$

Negative implicit feedback, such as rapid skips or explicit negative actions, may use $w_{ik}<0$.

### 3.2 Life-Situation Classifier

Let $z_k$ denote the concatenated content available for reel $r_k$ from caption text, OCR text, and ASR transcription. A transformer encoder produces the reel text representation:

$$
\mathbf{e}_{k,\mathrm{text}}
=
\mathrm{MeanPool}\left(\mathrm{Transformer}(z_k)\right).
$$

The life-situation taxonomy is

$$
\mathcal{S}
=
\left\{
\mathrm{job\_search},
\mathrm{new\_parent},
\mathrm{moving},
\mathrm{graduation},
\ldots
\right\}.
$$

The multi-label classifier produces probabilities for each situation:

$$
\mathbf{p}_k
=
\sigma\left(
\mathbf{W}_s\mathbf{e}_{k,\mathrm{text}}
+
\mathbf{b}_s
\right)
\in [0,1]^Q.
$$

The notation $\mathbf{p}_k$ denotes the predicted life-situation probability vector for reel $r_k$.

### 3.3 Multimodal Reel Representation

The final reel representation can combine text, visual, and audio features.

$$
\mathbf{e}_{k,\mathrm{multi}}
=
 f\left(
\mathbf{e}_{k,\mathrm{text}},
\mathbf{e}_{k,\mathrm{visual}},
\mathbf{e}_{k,\mathrm{audio}}
\right).
$$

### 3.4 Music Lyrics and Playlist Representation

Each track $m$ is represented using lyrics, acoustic features, and optional metadata. A music encoder maps these inputs into a shared representation:

$$
\mathbf{e}_{m,\mathrm{music}}
=
 f_{\mathrm{music}}\left(
\mathbf{e}_{m,\mathrm{lyrics}},
\mathbf{e}_{m,\mathrm{audio}}
\right).
$$

For user $u_i$, let $\mathcal{P}_i(t)$ be the playlist and music-history items available up to time $t$. The playlist-aware user music representation is:

$$
\mathbf{e}_{i,\mathrm{music}}(t)
=
\frac{
\sum_{m\in\mathcal{P}_i(t)}
 w_{im}\mathbf{e}_{m,\mathrm{music}}
}{
\sum_{m\in\mathcal{P}_i(t)} |w_{im}| + \varepsilon
}.
$$

This timestamp constraint prevents future playlist edits from leaking into historical evaluation.

The user-to-track music preference is then:

$$
S_{\mathrm{music}}(u_i,r_k)
=
\cos\left(
\mathbf{e}_{i,\mathrm{music}}(t),
\mathbf{e}_{m(r_k),\mathrm{music}}
\right).
$$

A separate situation-to-lyrics consistency term may also be used:

$$
S_{\mathrm{situation-text}}(u_i,m)
=
\cos\left(
W_s\mathbf{s}_i(t),
\mathbf{e}_{m,\mathrm{lyrics}}
\right).
$$

### 3.5 Temporal Life-Situation Vector

Life situations are modeled as transient states with category-specific decay rates. Let $\boldsymbol{\Lambda}=\mathrm{diag}(\lambda_{s_1},\ldots,\lambda_{s_Q})$.

$$
\mathbf{s}_i(t)
=
\sum_{k\in\mathcal{H}_i^T}
 w_{ik}
 \exp\left[-\boldsymbol{\Lambda}(t-t_k)\right]
 \mathbf{p}_k.
$$

For explicit self-declared situations, a floor can be applied:

$$
\mathbf{s}_i(t)
\leftarrow
\max\left(
\mathbf{s}_i(t),
\kappa\mathbf{s}^{\mathrm{explicit}}_i
\right),
\qquad
0<\kappa\le 1.
$$

### 3.6 City Match and Social-Spatial Graph

Let $c_i$ be the city associated with user $u_i$ and $c(r_k)$ the city associated with reel $r_k$. The exact-city score is

$$
S_{\mathrm{city}}(u_i,r_k)
=
\mathbf{1}\left[c_i=c(r_k)\right]
+
\eta
\exp\left(-\frac{d(c_i,c(r_k))}{\tau}\right)
\mathbf{1}\left[c_i\ne c(r_k)\right].
$$

The initial graph feature is the privacy-preserving location representation:

$$
\mathbf{g}_i^{(0)}=\mathbf{h}(l_i).
$$

For a graph propagation depth $K$, the LightGCN-style recurrence is

$$
\mathbf{g}_i^{(\ell)}
=
\sum_{j\in\mathcal{N}(i)}
\frac{
\mathbf{g}_j^{(\ell-1)}
}{
\sqrt{\left|\mathcal{N}(i)\right|\left|\mathcal{N}(j)\right|}
},
\qquad
\ell=1,\ldots,K.
$$

The final graph representation is a weighted combination of propagation depths:

$$
\mathbf{g}_i
=
\sum_{\ell=0}^{K}
\omega_\ell\mathbf{g}_i^{(\ell)},
\qquad
\sum_{\ell=0}^{K}\omega_\ell=1,
\qquad
\omega_\ell\ge 0.
$$

The projected graph representation is

$$
\widetilde{\mathbf{g}}_i
=
\mathbf{W}_g\mathbf{g}_i.
$$

The geographic relevance score is

$$
S_{\mathrm{geo}}(u_i,r_k)
=
\sigma\left(
\widetilde{\mathbf{g}}_i^{\top}\mathbf{v}_{r_k}
\right).
$$

### 3.7 Privacy Mechanism

Independent randomized response over every location bit does not by itself imply an $\varepsilon$ guarantee for the entire multi-hot vector. The implementation therefore uses a RAPPOR-style two-stage mechanism and reports the effective privacy budget under composition.

The privacy mechanism and city-level aggregation operate on coarse location categories rather than raw GPS coordinates. A minimum-support rule can be applied before releasing a neighborhood aggregate.

### 3.8 Dynamic Joint Scoring

The model uses a learned five-way gate over life-situation, city, interest, geographic, and music signals:

$$
[\alpha_i,\beta_i,\gamma_i,\delta_i,\mu_i]
=
\mathrm{softmax}\left(
\mathrm{MLP}_{\mathrm{gate}}(\mathbf{x}_i)
\right).
$$

Life-situation relevance is computed with a numerically stable cosine similarity:

$$
S_{\mathrm{situation}}(u_i,r_k)
=
\frac{
\mathbf{s}_i(t)^{\top}\mathbf{p}_k
}{
\sqrt{\left\|\mathbf{s}_i(t)\right\|_2^2+\epsilon}
\sqrt{\left\|\mathbf{p}_k\right\|_2^2+\epsilon}
}.
$$

The direct interest score is

$$
S_{\mathrm{interest}}(u_i,r_k)
=
\cos\left(
\mathbf{e}_{i,\mathrm{interest}}(t),
\mathbf{v}_{r_k}
\right).
$$

The final prediction is

$$
\hat{y}_{i,k}
=
\alpha_iS_{\mathrm{situation}}(u_i,r_k)
+
\beta_iS_{\mathrm{city}}(u_i,r_k)
+
\gamma_iS_{\mathrm{interest}}(u_i,r_k)
+
\delta_iS_{\mathrm{geo}}(u_i,r_k)
+
\mu_iS_{\mathrm{music}}(u_i,r_k).
$$

### 3.9 Optional Peer-Matching Layer

If the platform provides a separate people-discovery surface, users can be matched using shared life situations and a local-city constraint:

$$
S_{\mathrm{peer}}(u_i,u_j)
=
\cos\left(
\mathbf{s}_i(t),
\mathbf{s}_j(t)
\right)
\mathbf{1}\left[
 d(c_i,c_j)<\delta_{\mathrm{local}}
\right].
$$

## 4. Training Objective

The recommendation component is trained with Bayesian Personalized Ranking (BPR). For batch size $B$:

$$
\mathcal{L}_{\mathrm{BPR}}
=
-\frac{1}{B}
\sum_{i=1}^{B}
\log\sigma\left(
\hat{y}_{i,+}-\hat{y}_{i,-}
\right).
$$

The situation classifier uses multi-label binary cross-entropy over $M$ reels and $Q$ situation categories:

$$
\mathcal{L}_{\mathrm{sit}}
=
-\frac{1}{MQ}
\sum_{k=1}^{M}
\sum_{q=1}^{Q}
\left[
 c_{kq}\log p_{kq}
+
(1-c_{kq})\log(1-p_{kq})
\right].
$$

The adversarial fairness loss is the cross-entropy of the protected-attribute predictor:

$$
\mathcal{L}_{\mathrm{adv}}
=
\mathrm{CE}\left(
A_{\psi}(\phi_i),a_i
\right).
$$

The fairness formulation is written as a minimax objective:

$$
\min_{\Theta}\;\max_{\psi}
\left[
\mathcal{L}_{\mathrm{rank}}(\Theta)
-
\lambda_{\mathrm{fair}}\mathcal{L}_{\mathrm{adv}}(\Theta,\psi)
\right].
$$

The practical implementation uses explicit alternating updates: the adversary minimizes classification loss, while the recommendation representation is updated to reduce the adversary's predictive power.

The complete training objective can be written as

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{BPR}}
+
\lambda_{\mathrm{sit}}\mathcal{L}_{\mathrm{sit}}
+
\lambda_{\mathrm{fair}}\mathcal{L}_{\mathrm{adv}}
+
\lambda_{\mathrm{reg}}\left\|\Theta\right\|_2^2.
$$

## 5. Production Considerations

### 5.1 Candidate Retrieval and Ranking

A two-stage recommender can first retrieve a compact candidate set using approximate nearest-neighbor search and then apply the full contextual score.

### 5.2 Exploration and Trending

Exploration should be applied at serving time rather than injected into the supervised ranking loss. A separate trending signal can capture rapidly changing tracks and topics.

### 5.3 Graph Refresh

Graph-derived embeddings should normally be refreshed in batches rather than recomputed after every interaction. This bounds the online cost while introducing an explicit, measurable graph-staleness interval.

### 5.4 Playlist and Music Updates

Playlist updates should be timestamped and processed incrementally. Music embeddings can be cached and refreshed when lyrics, audio features, or metadata change.

## 6. Evaluation Framework

Evaluation should use chronological train/validation/test splits. Negative samples must not include held-out positives, and skipped content must not silently become a positive interaction.

Recommended ranking metrics include Hit Rate, Recall, NDCG, and Pairwise AUC at $K\in\{5,10,20\}$. Additional system-level diagnostics should include cold-start performance, city-hit rate, coverage, novelty, and fairness-group gaps.

For a user $u$, the Hit Rate at cutoff $K$ can be written as

$$
\mathrm{HR}@K
=
\frac{1}{|\mathcal{U}|}
\sum_{u\in\mathcal{U}}
\mathbf{1}\left[
\mathrm{rank}_u\le K
\right].
$$

Normalized discounted cumulative gain is

$$
\mathrm{NDCG}@K
=
\frac{1}{|\mathcal{U}|}
\sum_{u\in\mathcal{U}}
\frac{\mathrm{DCG}_u@K}{\mathrm{IDCG}_u@K}.
$$

## 7. Limitations and Future Work

The architecture is a research prototype. Public datasets rarely contain aligned reel interactions, city-level social graphs, playlists, lyrics, demographic variables, and multimodal content for the same users. Combining datasets therefore introduces synthetic linkage assumptions that must be disclosed.

Future work should evaluate the complete architecture on real short-form video data where legally and ethically permissible, compare against strong CF, GNN, sequence, and multimodal baselines, and study the trade-offs among recommendation quality, privacy, fairness, and exploration.

## 8. Conclusion

M²S²-Rec combines life-situation relevance, city anchoring, social-spatial propagation, direct user interest, and music/playlist preference into a single dynamically gated recommendation architecture. The music layer explicitly analyzes lyrics and acoustic features while incorporating user playlists as a persistent preference signal. The resulting design is intended to support both warm-user personalization and content-based item cold start while keeping privacy and fairness as first-class architectural constraints.

## Citation

```bibtex
@article{m2s2rec,
  title   = {M²S²-Rec: A City-Anchored, Life-Situation-Aware Architecture for Short-Form Video Recommendation},
  author  = {Md Hashibul Amin},
  year    = {2026},
  note    = {Research manuscript; Copyright © 2026 Md Hashibul Amin. All rights reserved.}
}
```

# M²S²-Rec: A City-Anchored, Life-Situation-Aware Architecture for Short-Form Video Recommendation

**Author:** Md Hashibul Amin  
**Year:** 2026  
**Status:** Research manuscript / architecture and implementation paper

---

## Abstract

Short-form video recommendation must operate under sparse, rapidly changing interaction signals while simultaneously accounting for content semantics, geographic locality, and transient user context. Conventional collaborative filtering (CF) and content-based filtering (CBF) capture parts of this problem, but they do not naturally combine a user's current life situation with city-level relevance and social-spatial signals.

This paper presents **M²S²-Rec**, a city-anchored, life-situation-aware recommendation architecture for short-form video. The framework models user state through temporally decayed interaction embeddings, a fairness-audited behavioral and demographic representation, and a time-varying life-situation vector. Reel content is mapped to a multi-label life-situation taxonomy using textual evidence from captions, OCR, and automatic speech recognition (ASR), with optional visual and acoustic auxiliary features. Geographic relevance is modeled in two stages: an explicit city-match score provides the primary hyper-local signal, while a K-hop LightGCN-style social-spatial propagation layer provides a secondary neighborhood-trend signal. A RAPPOR-style local differential privacy mechanism is used for multi-dimensional location reports rather than independently perturbing every binary location feature under one nominal privacy budget.

The final ranking score is produced by a learned per-user gate over situation, city, interest, and graph-based relevance. Training uses chronological interaction splits, BPR-style pairwise ranking, multi-label situation classification, adversarial fairness optimization, and regularization. Exploration and trending signals are treated as separate production layers rather than being injected into the supervised ranking loss. An executable JAX/NumPy prototype accompanies the manuscript. The prototype has been tested with synthetic data and reports ranking, cold-start, locality, and group-level diagnostic metrics; these results validate implementation behavior but are not claims of production performance or real-world superiority.

---

## 1. Introduction

Short-form video platforms such as TikTok, Instagram Reels, and YouTube Shorts operate in a recommendation regime where a few seconds of interaction can generate meaningful evidence about a user's current interests. At the same time, user intent is not stationary: a person searching for a job, moving to a new city, preparing for graduation, or adapting to parenthood may temporarily consume a different content distribution from the one implied by their long-term history.

Three properties motivate M²S²-Rec:

1. **Interaction sparsity and recency.** Short-form feeds generate large volumes of weak implicit feedback. A skip or fast scroll can be informative, and recent behavior can be more predictive than old behavior. BPR provides a standard framework for learning personalized rankings from implicit feedback [1].

2. **Hyper-local relevance.** A user's home city provides an interpretable and controllable spatial anchor, while a social graph can provide secondary evidence about local trends and cultural diffusion. LightGCN demonstrates that simple normalized neighborhood propagation can be highly effective for recommendation graphs [2].

3. **Transient situational context.** Long-term interest is not sufficient for users whose immediate circumstances have changed. M²S²-Rec therefore treats life situations as time-dependent latent states rather than as fixed personality traits.

The resulting system is intentionally different from an architecture that makes demographic or personality inference the centerpiece of ranking. Earlier versions of this design considered Big Five/OCEAN traits inferred from engagement traces. That formulation is removed from the load-bearing methodology because behavior-to-personality inference has substantial construct-validity concerns. Instead, M²S²-Rec uses a more defensible **behavioral engagement-style representation** and places demographic information behind an adversarially trained fairness gate.

The architecture addresses three questions:

- **Who is watching and what is their current context?**  
  Through temporally decayed interest and life-situation representations.

- **Where is the content relevant?**  
  Through exact city matching plus a K-hop social-spatial graph.

- **Is the reel relevant to the user's situation?**  
  Through a multi-label situation classifier operating over caption, OCR, and ASR evidence, with optional visual and acoustic auxiliary features.

The paper's primary contribution is therefore not a single neural component, but a consistent recommendation pipeline in which **life situation is the principal semantic relevance signal, city is the primary geographic anchor, and social-graph propagation is a secondary locality/trend signal**.

---

## 2. Related Work

### 2.1 Collaborative Filtering and Personalized Ranking

Collaborative filtering methods learn latent relationships between users and items from historical interactions. Neural Collaborative Filtering extends matrix-factorization-style ideas using neural interaction functions [3]. For implicit-feedback recommendation, Bayesian Personalized Ranking (BPR) directly optimizes pairwise preferences between observed positive interactions and sampled alternatives [1].

These methods provide strong foundations but have two limitations for short-form video. First, newly registered users have little or no interaction history. Second, a newly uploaded reel may have no historical engagement data. M²S²-Rec therefore supplements interaction embeddings with content, city, and contextual features.

### 2.2 Graph Neural Networks for Recommendation

Graph neural networks propagate information across connected entities. LightGCN simplifies graph convolution for recommendation by retaining normalized neighborhood aggregation and layer-wise embedding combination while removing unnecessary feature transformations and nonlinearities [2].

M²S²-Rec adapts this principle to a social-spatial graph in which each user has a privacy-protected city or regional representation. The graph does not replace direct city matching; instead, it provides secondary evidence about geographically localized social trends.

### 2.3 Transformer-Based Semantic Representation

Transformer encoders provide contextual text representations and can be fine-tuned for classification and retrieval tasks. BERT is a canonical example of pretrained bidirectional Transformer representations [4].

In M²S²-Rec, text evidence is conceptually formed from:

- user-facing reel caption;
- OCR extracted from frames;
- ASR transcript of spoken narration;
- optional hashtags and structured metadata.

These signals are encoded into a semantic representation from which multi-label life-situation probabilities are predicted.

### 2.4 Cross-Modal and Multimodal Representation Learning

Contrastive learning has demonstrated that aligned representations can connect different modalities, with CLIP providing a well-known text-image example [5]. The proposed system does not make music-to-caption alignment the primary objective. Instead, text and other media signals are used to classify **what life situation a reel expresses**. Audio and visual encoders may be incorporated as auxiliary features or pretrained encoders when production data are available.

### 2.5 Privacy-Preserving Telemetry

Independent randomized response over every bit of a high-dimensional multi-hot vector can create a composition problem if the full vector is intended to satisfy one overall local-DP budget. RAPPOR provides a more appropriate design pattern for privacy-preserving collection of high-dimensional categorical information [6]. M²S²-Rec therefore uses a RAPPOR-style two-stage mechanism and explicitly tracks the privacy parameters used by the implementation.

---

## 3. Problem Formulation

Let:

- $\mathcal{U} = \{u_1,\ldots,u_N\}$ be the user set;
- $\mathcal{R} = \{r_1,\ldots,r_M\}$ be the reel set;
- $\mathcal{C}$ be the supported city vocabulary;
- $\mathcal{S} = \{s_1,\ldots,s_Q\}$ be the life-situation taxonomy;
- $c_i \in \mathcal{C}$ be the city associated with user $u_i$;
- $c(r_k) \in \mathcal{C}$ be the city associated with reel $r_k$ when available;
- $\mathcal{H}_i^T$ be the user's temporally bounded interaction history.

The task is to learn a ranking function

$$
\hat{y}(u_i,r_k)
$$

such that positively engaged reels are ranked above alternatives while respecting contextual, spatial, privacy, and fairness constraints.

The system is designed primarily for **implicit feedback**. Positive events may include qualified watches, completion, rewatch, like, save, share, or comment; negative evidence may include fast scroll-away or explicit negative feedback. The exact event-to-weight mapping is a dataset and product decision and should be calibrated rather than assumed universal.

---

## 4. System Architecture

M²S²-Rec is organized as five principal stages:

1. **User state construction**
2. **Life-situation content encoding**
3. **City and social-spatial representation**
4. **Dynamic multi-signal ranking**
5. **Offline training and evaluation**

A production implementation should use a two-stage serving architecture:

$$
\text{Candidate Retrieval}
\rightarrow
\text{Contextual Reranking}
\rightarrow
\text{Exploration / Policy Layer}
$$

The prototype focuses on the representation, scoring, training, and evaluation stages.

---

## 5. Methodology

### 5.1 User Representation and Fairness Gate

A user is represented by a temporally decayed interest vector, a temporally decayed behavioral style vector, and a fairness-audited representation.

#### 5.1.1 Demographic and Behavioral Inputs

Let

$$
\mathbf{d}_i =
[
\text{age}_i^{norm}
\Vert
\mathbf{v}_{edu}
\Vert
\mathbf{v}_{gender}
\Vert
\mathbf{v}_{occ}
\Vert
\mathbf{v}_{city}
]
$$

be the encoded demographic/contextual vector.

Rather than claiming that interaction behavior measures validated personality traits, we define

$$
\mathbf{p}_i \in \mathbb{R}^{d_p}
$$

as a **behavioral engagement-style embedding** constructed from measurable statistics such as completion ratio, skip behavior, rewatch frequency, session activity, and related engagement features.

The fairness representation is:

$$
\mathbf{b}_i =
\operatorname{MLP}_{fair}
(
\mathbf{d}_i \Vert \mathbf{p}_i
).
$$

Protected-attribute information is then subjected to an adversarial fairness objective rather than being directly exposed as an unrestricted scoring term.

### 5.2 Temporally Decayed Interest Representation

For an interaction event $k$ occurring at time $t_k$, let $w_{ik}$ denote its signed engagement weight. A generic temporally decayed interest embedding is:

$$
\mathbf{e}_{i,interest}(t)
=
\sum_{k \in \mathcal{H}_i^T}
w_{ik}
\exp[-\lambda_t(t-t_k)]
\mathbf{m}_k.
$$

Here:

- $w_{ik}>0$ represents positive engagement;
- $w_{ik}<0$ may represent a skip or other negative signal;
- $\lambda_t>0$ is the interest decay rate;
- $\mathbf{m}_k$ is the reel embedding.

The implementation separates positive and negative evidence when constructing situation state so that a disliked reel does not automatically imply that its detected situation is desirable.

A second embedding can be constructed for content style or audio preference when those features are available:

$$
\mathbf{e}_{i,style}(t)
=
\sum_{k \in \mathcal{H}_i^T}
w_{ik}
\exp[-\lambda_t(t-t_k)]
\mathbf{a}_k^{style}.
$$

---

### 5.3 Life-Situation Classifier

Let each reel expose the textual sequence

$$
z_k =
\operatorname{Caption}(r_k)
\Vert
\operatorname{OCR}(r_k)
\Vert
\operatorname{ASR}(r_k).
$$

A Transformer encoder produces:

$$
\mathbf{e}_{k,text}
=
\operatorname{MeanPool}
(
\operatorname{Transformer}(z_k)
).
$$

The system predicts a multi-label life-situation probability vector:

$$
\mathbf{p}_k
=
P(\mathbf{c}(r_k)\mid r_k)
=
\sigma(
\mathbf{W}_s\mathbf{e}_{k,text}
+
\mathbf{b}_s
)
\in [0,1]^Q.
$$

Possible categories include:

$$
\{
\text{job\_search},
\text{new\_parent},
\text{moving},
\text{graduation},
\ldots
\}.
$$

Hashtags such as `#jobsearch`, `#newmom`, or `#movingcities` can provide weak labels during early bootstrapping. Weak labels should not be treated as ground truth; a later human-annotated validation set is required to estimate classifier quality and label bias.

#### Multimodal Extension

For a production system, visual and acoustic representations can be added:

$$
\mathbf{e}_{k,multi}
=
f(
\mathbf{e}_{k,text},
\mathbf{e}_{k,visual},
\mathbf{e}_{k,audio}
).
$$

The key design choice is that these representations support **situation classification**, rather than forcing a separate music-matching objective to dominate the recommendation score.

---

### 5.4 Temporal Life-Situation State

Life situations are transient and may have different persistence times.

Define:

$$
\Lambda =
\operatorname{diag}
(
\lambda_{s_1},
\dots,
\lambda_{s_Q}
).
$$

The user situation state is:

$$
\mathbf{s}_i(t)
=
\sum_{k\in\mathcal{H}_i^T}
w_{ik}^{+}
\exp[-\Lambda(t-t_k)]
\mathbf{p}_k,
$$

where

$$
w_{ik}^{+}=\max(w_{ik},0).
$$

Using positive engagement for situation-state accumulation prevents negative feedback from strengthening a situation simply because a user disliked content associated with it.

When explicit user-declared states are available, the system may preserve a floor:

$$
\mathbf{s}_i(t)
\leftarrow
\max
\left(
\mathbf{s}_i(t),
\kappa \mathbf{s}_i^{explicit}
\right),
\qquad
0<\kappa\le1.
$$

This is intended as a product-policy mechanism rather than a universally optimal assumption; the value of $\kappa$ should be validated empirically.

---

### 5.5 City Match

The primary spatial signal is city anchored.

For user city $c_i$ and reel city $c(r_k)$:

$$
S_{city}(u_i,r_k)
=
\mathbb{1}[c_i=c(r_k)]
+
\eta
\exp
\left(
-\frac{d(c_i,c(r_k))}{\tau}
\right)
\mathbb{1}[c_i\ne c(r_k)].
$$

Here:

- $d(\cdot,\cdot)$ is a city-centroid distance;
- $\tau$ controls the spatial bandwidth;
- $\eta$ determines how strongly a nearby non-matching city can contribute.

The exact city match is intentionally the dominant locality feature. Regional graph propagation is secondary.

---

### 5.6 Social-Spatial Graph Propagation

Let the social graph be:

$$
\mathcal{G}=(\mathcal{V},\mathcal{E})
$$

where each node corresponds to a user and each edge represents an allowed social connection.

Each user starts with a privacy-protected location representation:

$$
\mathbf{g}^{(0)}_i=\mathbf{h}(c_i).
$$

For K-hop propagation:

$$
\mathbf{g}^{(\ell)}_i
=
\sum_{j\in\mathcal{N}(i)}
\frac{
\mathbf{g}^{(\ell-1)}_j
}{
\sqrt{
|\mathcal{N}(i)|
|\mathcal{N}(j)|
}
},
\qquad
\ell=1,\ldots,K.
$$

The final graph representation is:

$$
\mathbf{g}_i
=
\sum_{\ell=0}^{K}
\omega_\ell
\mathbf{g}^{(\ell)}_i,
\qquad
\sum_{\ell=0}^{K}\omega_\ell=1.
$$

A uniform choice is:

$$
\omega_\ell=\frac{1}{K+1}.
$$

This follows the layer-combination spirit of LightGCN [2].

#### Cold Network

For a user with no graph neighbors, the representation remains:

$$
\mathbf{g}_i=\mathbf{g}^{(0)}_i.
$$

Thus a newly registered user is not forced to inherit information from a nonexistent neighborhood.

---

### 5.7 Local Differential Privacy

Location information can reveal sensitive information when shared across a social graph. A naive approach that applies binary randomized response independently to every location bit with the same nominal $\epsilon$ does not automatically provide an overall $\epsilon$ guarantee for the full vector.

M²S²-Rec therefore uses a RAPPOR-style two-stage approach [6]:

1. a memoized/stable randomized representation for the user's locally held categorical signal;
2. instantaneous randomized response before reporting.

The effective privacy guarantee must be stated in terms of the complete mechanism and composition used by a concrete implementation. The repository records the mechanism parameters rather than claiming that a single per-bit $\epsilon$ describes the privacy loss of the entire multi-dimensional report.

#### Coarse Spatial Resolution

The system should operate on city or similarly coarse administrative units rather than raw GPS coordinates whenever exact coordinates are not required.

A k-anonymity-like minimum cohort rule may additionally be used as a product-policy safeguard:

$$
|\mathcal{N}(i)| < k
\Rightarrow
\mathbf{g}_i\leftarrow\mathbf{g}_{macro}.
$$

This is not equivalent to differential privacy. The two mechanisms address different risks and should not be presented as interchangeable guarantees.

---

### 5.8 Situation Relevance Score

Given the user state $\mathbf{s}_i(t)$ and reel situation distribution $\mathbf{p}_k$:

$$
S_{situation}(u_i,r_k)
=
\frac{
\mathbf{s}_i(t)^\top\mathbf{p}_k
}{
\|\mathbf{s}_i(t)\|_2
\|\mathbf{p}_k\|_2
}.
$$

For numerical stability, the implementation uses an $\epsilon$-stabilized norm rather than a hard clamp whose derivative can become problematic when a cold-start state is exactly zero.

If a user has no inferred situation history, the score is treated as an unavailable/neutral signal and the learned gate can reduce its contribution.

---

### 5.9 Direct Interest Score

The direct collaborative/content relevance term is:

$$
S_{interest}(u_i,r_k)
=
\cos
\left(
\mathbf{e}_{i,interest}(t),
\mathbf{v}_{r_k}
\right).
$$

This term is essential because the model should not rely exclusively on demographic, city, or graph information for warm users.

---

### 5.10 Social-Spatial Reel Score

The graph representation is projected into the reel embedding space:

$$
\tilde{\mathbf{g}}_i
=
\mathbf{W}_g\mathbf{g}_i.
$$

The geographic-social compatibility score is:

$$
S_{geo}(u_i,r_k)
=
\sigma
\left(
\tilde{\mathbf{g}}_i^\top
\mathbf{v}_{r_k}
\right).
$$

The transpose/dot-product formulation is explicit to avoid the dimensional ambiguity of a scalar-vector multiplication.

---

### 5.11 Dynamic Joint Scoring

A fixed global weight vector assumes every user depends on the same evidence.

Instead:

$$
\mathbf{x}_i
=
[
\mathbf{e}_{i,interest}(t)
\Vert
\mathbf{e}_{i,style}(t)
\Vert
\mathbf{b}_i
].
$$

The gate is:

$$
[
\alpha_i,\beta_i,\gamma_i,\delta_i
]
=
\operatorname{softmax}
(
\operatorname{MLP}_{gate}(\mathbf{x}_i)
).
$$

The final score is:

$$
\hat{y}_{i,k}
=
\alpha_iS_{situation}(u_i,r_k)
+
\beta_iS_{city}(u_i,r_k)
+
\gamma_iS_{interest}(u_i,r_k)
+
\delta_iS_{geo}(u_i,r_k).
$$

The interpretation is:

- $\alpha_i$: life-situation relevance;
- $\beta_i$: explicit city relevance;
- $\gamma_i$: direct interest relevance;
- $\delta_i$: social-spatial graph relevance.

This formulation lets the model shift its reliance on signals as a user's interaction history changes.

---

### 5.12 Optional Peer-Matching Surface

If the product goal includes connecting users going through similar circumstances, recommendation of people should be treated as a separate retrieval task.

For two users:

$$
S_{peer}(u_i,u_j)
=
\cos
(
\mathbf{s}_i(t),\mathbf{s}_j(t)
)
\mathbb{1}
[d(c_i,c_j)<\delta_{local}].
$$

This layer should be implemented using a city-scoped approximate nearest-neighbor index rather than being conflated with reel ranking.

---

## 6. Training Algorithm

### 6.1 Chronological Data Split

Randomly splitting every interaction into train and test sets can leak future information into the representation.

The preferred protocol is chronological:

$$
D_{train}<D_{valid}<D_{test}.
$$

For each user, interactions are ordered by timestamp. The most recent eligible positive interaction(s) are reserved for validation/test according to the experiment protocol.

Cold-start cohorts should be defined separately, for example by users whose training history is empty or below a predefined minimum.

---

### 6.2 Positive and Negative Sampling

For BPR, a training example is:

$$
(u_i,r_i^+,r_i^-).
$$

The positive reel must correspond to an observed positive event.

The negative reel should be sampled from candidates that are not known positive interactions in the relevant training/holdout partition.

In particular:

- a skipped reel must not silently become a positive;
- a held-out test positive must not be reused as a negative;
- sampled negatives should not be selected from the user's known positive set.

When the product has meaningful explicit negative feedback, that feedback can be used as weighted supervision rather than being confused with unobserved items.

---

### 6.3 BPR Loss

For a positive and negative candidate:

$$
\mathcal{L}_{BPR}
=
-\frac{1}{B}
\sum_{i=1}^{B}
\log
\sigma
\left(
\hat{y}_{i,+}
-
\hat{y}_{i,-}
\right).
$$

This objective follows the personalized-ranking formulation introduced by Rendle et al. [1].

---

### 6.4 Situation Classification Loss

Given weak or human labels $\mathbf{c}_k$:

$$
\mathcal{L}_{sit}
=
-
\frac{1}{MQ}
\sum_{k=1}^{M}
\sum_{q=1}^{Q}
[
c_{kq}\log p_{kq}
+
(1-c_{kq})\log(1-p_{kq})
].
$$

Weak hashtag labels should be treated as noisy supervision. A human-labeled subset is recommended for model selection and calibration.

---

### 6.5 Adversarial Fairness Objective

Let $\phi$ denote the shared fair representation and $a_i$ a protected attribute.

The adversary minimizes:

$$
\mathcal{L}_{adv}
=
CE
(
A_\psi(\phi_i),
a_i
).
$$

The fairness representation is trained to make protected-attribute prediction difficult. The implementation uses an explicit alternating/minimax update rather than depending on a single opaque gradient-reversal path.

Conceptually:

$$
\min_{\Theta}
\max_{\psi}
\left[
\mathcal{L}_{rank}
-
\lambda_{fair}\mathcal{L}_{adv}
\right]
$$

with the sign convention implemented by the actual alternating optimizer.

Fairness must be evaluated independently using group-level metrics; minimizing an adversary's accuracy is not, by itself, proof of fairness.

---

### 6.6 Regularized Joint Objective

The total training objective is:

$$
\mathcal{L}
=
\mathcal{L}_{BPR}
+
\lambda_{sit}\mathcal{L}_{sit}
+
\lambda_{fair}\mathcal{L}_{adv}
+
\lambda_{reg}\|\Theta\|_2^2.
$$

The repository uses gradient clipping and numerically stable similarity functions to reduce optimization instability.

If a large pretrained multimodal encoder is used, staged training is recommended:

1. pretrain/fine-tune the content representation;
2. freeze or partially freeze the encoder;
3. train the recommendation layers;
4. optionally perform controlled end-to-end fine-tuning after the ranking objective is stable.

This prevents two unrelated objectives from destabilizing the same encoder from the beginning.

---

## 7. Data Collection and Analysis Protocol

A scientifically valid implementation requires an explicit data contract.

### 7.1 User Event Schema

A production event should minimally contain:

```text
user_id
reel_id
timestamp
event_type
watch_duration_ms
reel_duration_ms
city_id (coarse)
optional social-edge reference
```

Derived engagement features should be computed deterministically from raw events.

### 7.2 Reel Content Schema

A reel record should contain:

```text
reel_id
timestamp_created
city_id (if available)
caption
hashtags
ocr_text
asr_text
audio_id
optional visual embedding
optional audio embedding
```

### 7.3 Labeling Pipeline

The recommended situation-label pipeline is:

```text
weak labels
   ↓
cleaning / deduplication
   ↓
human-annotated validation subset
   ↓
classifier training
   ↓
calibration and error analysis
   ↓
production inference
```

Hashtag-derived labels should not be reported as ground-truth annotations.

### 7.4 Leakage Controls

Data analysis should explicitly prevent:

- future interactions entering past user state;
- test positives becoming sampled negatives;
- city information derived from the future test event being available during training;
- labels generated from engagement outcomes that happen after the prediction timestamp.

### 7.5 Synthetic-to-Real Limitation

Public datasets rarely contain the exact combination of user demographics, social connections, city, short-form content, captions, ASR/OCR, audio, and life-situation labels required by this architecture. Combining unrelated public datasets therefore creates a **synthetic integration validity risk**.

Synthetic data are appropriate for testing implementation behavior and controlled ablations. They are not sufficient to establish real-world effectiveness.

---

## 8. Production Considerations

### 8.1 Candidate Retrieval

Applying the complete contextual score to every reel is impractical at platform scale.

A two-stage cascade is recommended:

1. retrieve a few hundred candidates using an ANN index or other scalable retrieval method;
2. apply the complete contextual reranker.

For HNSW, query-time behavior is commonly sub-linear in favorable operating regimes, but $\mathcal{O}(\log |\mathcal{C}|)$ should not be presented as a guaranteed theoretical bound for every implementation. Build cost, memory, graph parameters, and recall/latency trade-offs must be reported experimentally.

### 8.2 Exploration

Pure exploitation can reinforce existing interests and social-spatial homophily.

Exploration should be added after the supervised ranking model through a policy layer such as:

- epsilon-greedy exploration;
- Thompson sampling;
- other contextual-bandit policies.

Exploration should not be confused with the offline supervised ranking objective.

### 8.3 Trending and Recency

The personalization model should be supplemented by an independent trend signal:

$$
S_{trend}(r_k,t)
=
f(
\text{recent interactions},
\text{velocity},
\text{time decay},
\text{regional adoption}
).
$$

This signal can help newly viral audio, creators, or topics receive exposure before they accumulate enough user-specific interaction history.

### 8.4 Graph Refresh

Recomputing social-spatial embeddings after every interaction is expensive.

The graph representation can therefore be refreshed periodically, for example hourly or according to an application-specific freshness SLA. The freshness interval should be measured as an engineering trade-off rather than asserted to be universally optimal.

---

## 9. Evaluation Framework

### 9.1 Ranking Metrics

For top-$K$ recommendation:

$$
HR@K
=
\frac{1}{|\mathcal{U}|}
\sum_{u\in\mathcal{U}}
\mathbb{I}
[
\operatorname{rank}_u\le K
].
$$

NDCG@K is:

$$
NDCG@K
=
\frac{1}{|\mathcal{U}|}
\sum_{u\in\mathcal{U}}
\frac{DCG_u@K}{IDCG_u@K}.
$$

The implementation also reports pairwise AUC-style performance and city-hit rate.

### 9.2 Cold-Start Evaluation

Report at least:

- overall performance;
- user cold-start performance;
- warm-user performance;
- item cold-start performance where item metadata are available.

This prevents a strong warm-user score from hiding poor onboarding behavior.

### 9.3 Fairness Diagnostics

Group-level diagnostics should include, where legally and ethically appropriate:

- Recall@K by group;
- NDCG@K by group;
- exposure/share metrics;
- pairwise performance by group;
- absolute gap and, when appropriate, relative gap.

No single fairness metric is sufficient for all products.

### 9.4 Locality Metrics

The city-aware design motivates:

$$
CityHit@K
=
\frac{
\#\{\text{recommended reels matching user city}\}
}{
K
}.
$$

This should be reported alongside diversity metrics so that stronger locality is not mistaken for universally better recommendation.

### 9.5 Diversity and Coverage

Useful complementary metrics include:

- item coverage;
- creator coverage;
- novelty;
- intra-list diversity;
- regional diversity.

A model that maximizes local homophily may reduce exploration and diversity, so these metrics are particularly important for this architecture.

---

## 10. Prototype Validation

The accompanying repository implements an executable research prototype using JAX and NumPy. The prototype was designed for software validation rather than for claims of production performance.

The tested implementation includes:

- temporal interest decay;
- temporal situation decay;
- city-distance scoring;
- K-hop graph propagation;
- privacy-aware spatial features;
- dynamic four-way gating;
- BPR ranking;
- multi-label situation BCE;
- adversarial fairness optimization;
- chronological evaluation;
- cold/warm evaluation splits.

A smoke-test experiment produced the following illustrative synthetic-data results:

| Metric | Prototype result |
|---|---:|
| Recall@20 | 0.9000 |
| NDCG@20 | 0.4887 |
| Pairwise AUC | 0.9003 |
| City hit rate | 0.8500 |
| Cold Recall@20 | 1.0000 |
| Warm Recall@20 | 0.8889 |

These numbers are **implementation-validation results on synthetic experimental data**. They do not establish that M²S²-Rec outperforms CF, LightGCN, multimodal recommenders, or production systems on real-world datasets.

The repository's automated test suite currently passes **6/6 core tests** for the implemented prototype.

---

## 11. Ablation Plan

A complete empirical study should isolate the contribution of each major signal.

Recommended variants are:

| Variant | Situation | City | Interest | Graph | Fairness gate |
|---|---:|---:|---:|---:|---:|
| CF baseline | ❌ | ❌ | ✅ | ❌ | ❌ |
| + City | ❌ | ✅ | ✅ | ❌ | ❌ |
| + Situation | ✅ | ✅ | ✅ | ❌ | ❌ |
| + Graph | ✅ | ✅ | ✅ | ✅ | ❌ |
| + Dynamic Gate | ✅ | ✅ | ✅ | ✅ | ✅ |
| Full M²S²-Rec | ✅ | ✅ | ✅ | ✅ | ✅ |

Additional ablations should vary:

- K-hop depth;
- privacy budget;
- decay rates;
- explicit situation floor;
- weak-label vs. human-label training;
- cold-start cohort definition;
- exploration strength.

---

## 12. Baselines and Experimental Design

For a future real-data benchmark, comparison should include representatives from:

### Collaborative Filtering

- Matrix Factorization;
- Neural Collaborative Filtering;
- BPR-based matrix factorization.

### Graph Recommendation

- LightGCN;
- a social recommendation model using friendship edges.

### Multimodal Recommendation

- visual/text multimodal models;
- multimodal graph recommendation;
- contrastive multimodal baselines.

### Contextual Variants

- no situation signal;
- no city signal;
- no graph signal;
- fixed-weight fusion;
- dynamic gating.

All models should use the same chronological split, candidate protocol, and evaluation population wherever possible.

---

## 13. Complexity Analysis

Let:

- $M=|\mathcal{R}|$ be the number of reels;
- $N=|\mathcal{U}|$ be the number of users;
- $E=|\mathcal{E}|$ be the number of social edges;
- $L$ be the Transformer token length;
- $d$ be the embedding size;
- $C$ be the candidate set size after retrieval;
- $K$ be the number of graph hops.

### Content Encoding

A dense Transformer encoder has approximate per-reel complexity:

$$
O(L^2d)
$$

so corpus preprocessing is approximately:

$$
O(ML^2d).
$$

### Graph Propagation

With sparse adjacency and dense node representations, K-hop propagation is approximately:

$$
O(KEd)
$$

for a graph representation of width $d$.

If the graph operates directly on a region vector of size $|\mathcal{L}|$, complexity becomes approximately:

$$
O(KE|\mathcal{L}|).
$$

### Online Reranking

For $C$ retrieved candidates:

$$
O(Cd')
$$

is a practical approximation for the dense scoring stage, excluding feature-cache and network overhead.

The full end-to-end latency depends on retrieval, feature access, batching, hardware, model size, and system architecture. Therefore, the manuscript does not claim a universal $\le 50$ ms guarantee.

---

## 14. Limitations and Threats to Validity

### 14.1 Situation Taxonomy Bias

Life-situation categories are not culturally universal. A taxonomy designed in one population may omit relevant situations or encode assumptions about how users describe their lives.

### 14.2 Weak Supervision

Hashtags can be ambiguous, strategic, sarcastic, or unrelated to the actual semantics of a video. Human-annotated validation data are therefore necessary.

### 14.3 Demographic Fairness

Adversarial debiasing can reduce predictability of protected attributes from a representation, but it cannot guarantee fairness in every downstream metric. Fairness evaluation must be performed on the actual recommender outputs.

### 14.4 Privacy

LDP protects reports under the assumptions of the implemented mechanism and threat model. It does not automatically protect against every side channel, re-identification strategy, or inference attack.

### 14.5 Synthetic Integration

Using separate datasets for behavior, city, social links, music, and demographics can create artificial correlations. Such experiments should be labeled as synthetic or simulated rather than as evidence obtained from a naturally integrated population.

### 14.6 Filter-Bubble Risk

Because city and social graph signals can reinforce local homophily, exploration, diversity, and exposure monitoring are essential.

### 14.7 Missing Real-World Benchmark

The current prototype does not constitute a full benchmark against production-scale systems. A public-data or proprietary-data evaluation with reproducible preprocessing and stronger baselines remains necessary.

---

## 15. Reproducibility and Implementation

The accompanying repository contains:

```text
model/
data/
tests/
train.py
evaluate.py
losses.py
README.md
PAPER.md
ORIGINAL_PAPER.md
ALGORITHM_FIXES.md
IMPLEMENTATION_REVIEW.md
requirements.txt
```

The implementation uses:

- JAX;
- NumPy;
- Optax where available in the training environment.

The repository includes deterministic seeds, synthetic data generation, test cases, and evaluation code intended to make the core architecture reproducible.

---

## 16. Conclusion

M²S²-Rec reframes short-form video recommendation around the interaction of **who the user is now, where they are anchored, and what life situation the content expresses**.

The architecture combines:

1. temporally decayed user interests;
2. a behavioral and demographic fairness gate;
3. life-situation classification from multimodal reel evidence;
4. city-anchored spatial relevance;
5. K-hop social-spatial propagation;
6. privacy-aware geographic representations;
7. dynamic per-user ranking gates;
8. explicit cold-start handling;
9. chronological offline evaluation;
10. separate exploration and trend layers for production.

The principal methodological change from the earlier demographic/OCEAN formulation is deliberate: the system no longer treats inferred personality as a validated psychological measurement. Instead, measurable engagement behavior is represented as an embedding, while demographic attributes are constrained by the fairness mechanism.

The resulting architecture is best understood as a **research blueprint with an executable prototype**, not as a demonstrated production recommender. Its strongest future validation path is a controlled real-data experiment with chronological evaluation, meaningful baselines, ablations, privacy analysis, fairness diagnostics, and explicit measurement of locality, diversity, and cold-start performance.

---

## References

[1] Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. 2009. **BPR: Bayesian Personalized Ranking from Implicit Feedback**. Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence, 452–461.

[2] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yong-Dong Zhang, and Meng Wang. 2020. **LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation**. Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, 639–648. DOI: 10.1145/3397271.3401063.

[3] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. **Neural Collaborative Filtering**. Proceedings of the 26th International Conference on World Wide Web, 173–182. DOI: 10.1145/3038912.3052569.

[4] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 4171–4186. DOI: 10.18653/v1/N19-1423.

[5] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. **Learning Transferable Visual Models From Natural Language Supervision**. Proceedings of the 38th International Conference on Machine Learning, 8748–8763.

[6] Úlfar Erlingsson, Vasyl Pihur, and Aleksandra Korolova. 2014. **RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response**. Proceedings of the 21st ACM Conference on Computer and Communications Security.

---

## Author and Copyright

**Md Hashibul Amin** is the author and rights holder of this research manuscript unless a later publication agreement states otherwise.

Suggested citation:

```bibtex
@article{m2s2rec,
  title   = {M²S²-Rec: A City-Anchored, Life-Situation-Aware Architecture for Short-Form Video Recommendation},
  author  = {Md Hashibul Amin},
  year    = {2026},
  note    = {Research manuscript; Copyright © 2026 Md Hashibul Amin. All rights reserved.}
}
```

The accompanying source code may be distributed under the Apache License 2.0 as specified by the repository's `LICENSE` file. The presence of this manuscript in the repository does not, by itself, place the manuscript under the Apache License 2.0. The manuscript remains subject to the author's copyright and any separate publication or licensing terms.

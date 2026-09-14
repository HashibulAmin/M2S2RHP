# M²S²-Rec: A City-Anchored, Life-Situation-Aware Architecture for Short-Form Video Recommendation

**Author:** Md Hashibul Amin  
**Year:** 2026  
**Status:** Research manuscript / architecture and implementation paper (expanded M²S²-Rec v2)

---

## Abstract

Short-form video recommendation must operate under sparse, rapidly changing interaction signals while simultaneously accounting for content semantics, geographic locality, and transient user context. Conventional collaborative filtering (CF) and content-based filtering (CBF) capture parts of this problem, but they do not naturally combine a user's current life situation with city-level relevance and social-spatial signals.

This paper presents **M²S²-Rec v2**, an expanded city-anchored, life-situation-aware recommendation architecture for short-form video. The framework models persistent and transient user state through separate long-term and short-term interest encoders, fine-grained multi-behavior event representations, temporally decayed life-situation states, and a fairness-audited behavioral representation. Reel content is mapped to a multi-label life-situation taxonomy using caption, OCR, ASR, visual, and acoustic evidence, with an adaptive multimodal fusion mechanism that learns modality reliability rather than assuming equal contribution. Geographic relevance is modeled through exact city matching, privacy-aware spatial features, and K-hop social-spatial propagation. A creator/vlogger representation and an optional tripartite user-video-creator graph are added to capture creator affinity and cold-start structure.

The ranking stack is extended with situation confidence, long-versus-short-term interest conflict, fine-grained skip behavior, hard-negative mining, exposure-bias correction, temporal diversity and repetition-fatigue penalties, and creator-exposure monitoring. The learned gate remains five-way over situation, city, direct interest, social-spatial, and music-playlist relevance; fairness is retained as a constraint/regularizer rather than as a sixth gate dimension. A dedicated music encoder and playlist profile remain available, while an optional offline multimodal large language model (MLLM) stage can generate cached semantic descriptions without becoming an online serving dependency. The production design separates candidate retrieval, contextual reranking, diversity/novelty controls, and an optional contextual-bandit policy layer. Training uses chronological splits, weighted BPR, situation classification, calibration, exposure correction, adversarial fairness optimization, and regularization. The accompanying prototype remains an implementation-validation artifact unless real-data experiments are conducted; no synthetic result in this manuscript is presented as evidence of real-world superiority.

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

The expanded v2 design adds a second principle: **current situation should change relevance, but persistent preference, exposure, locality, creator affinity, diversity, and uncertainty should constrain how strongly that situation changes the feed**. The main additions are:

1. separate long-term and short-term interest representations;
2. multi-behavior sequence modeling and fine-grained skip semantics;
3. adaptive multimodal fusion and optional offline MLLM semantic enrichment;
4. situation confidence, calibration, and explicit situation-versus-interest conflict;
5. hard-negative sampling and exposure/propensity correction;
6. temporal diversity, repetition-fatigue, novelty, and creator-exposure controls;
7. optional user-video-creator graph modeling;
8. situation-aware candidate retrieval before contextual reranking;
9. counterfactual and uncertainty-aware evaluation;
10. an optional contextual-bandit policy layer for online exploration.

These additions are designed as modular extensions. The core identity of M²S²-Rec remains the interaction among life situation, city, persistent user preference, and social-spatial context.


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


### 2.6 Long- and Short-Term Interest Modeling

Recent micro-video recommendation work separates stable long-term interests from rapidly changing short-term interests and also considers temporal diversity preferences. This motivates an explicit dual-timescale user state in M²S²-Rec rather than relying on a single exponential decay parameter [7].

### 2.7 Fine-Grained Skip and Multi-Behavior Modeling

Binary positive/negative treatment can discard information in short-form feeds. Fine-grained skip modeling distinguishes early skips from late skips, while multi-behavior sequence modeling can retain the semantic difference between watch, completion, like, save, share, comment, replay, and skip events [8]. M²S²-Rec v2 therefore uses behavior-aware event weights and typed temporal sequences.

### 2.8 Multimodal Contrastive and Semantic Enrichment

Multimodal graph contrastive learning has shown that treating visual, acoustic, and textual channels as equally important can be suboptimal [9]. M²S²-Rec v2 therefore introduces learned modality weights. An optional offline MLLM stage can additionally produce cached high-level descriptions that expose intent, entities, actions, tone, and world knowledge to downstream recommenders [10].

### 2.9 Exposure Bias and Unbiased Evaluation

Observed recommendation logs are affected by what the previous recommender exposed. Randomized-exposure resources such as KuaiRand provide a useful setting for studying debiasing and counterfactual evaluation [11]. M²S²-Rec v2 therefore includes propensity-aware weighting and explicitly distinguishes observed non-interaction from true negative feedback.

### 2.10 Creator-Aware Recommendation

Vlogger/creator information can be represented separately from item identity. A user-video-creator graph can capture creator affinity and provide useful structure for sparse or cold-start videos [12]. M²S²-Rec v2 treats creator affinity as an auxiliary score and retrieval source rather than another independent gate dimension.

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

```math
\hat{y}(u_i,r_k)
```

such that positively engaged reels are ranked above alternatives while respecting contextual, spatial, privacy, and fairness constraints.

The system is designed primarily for **implicit feedback**. Positive events may include qualified watches, completion, rewatch, like, save, share, or comment; negative evidence may include fast scroll-away or explicit negative feedback. The exact event-to-weight mapping is a dataset and product decision and should be calibrated rather than assumed universal.

---

## 4. System Architecture

M²S²-Rec is organized as the following modular stages:

1. **User state construction**
2. **Life-situation content encoding**
3. **City and social-spatial representation**
4. **Music lyrics/audio and playlist profiling**
5. **Dynamic multi-signal ranking**
6. **Offline training and evaluation**

A production implementation should use a staged serving architecture:

```math
\text{Interest Retrieval}
\cup
\text{Situation Retrieval}
\cup
\text{City/Creator Retrieval}
\rightarrow
\text{Candidate Union}
\rightarrow
\text{Contextual Reranking}
\rightarrow
\text{Diversity/Novelty Control}
\rightarrow
\text{Exploration / Policy Layer}
```

The prototype focuses on the representation, scoring, training, and evaluation stages.

---

## 5. Methodology


### 4.1 Expanded M²S²-Rec v2 Architecture

The complete v2 flow is:

```text
                         M²S²-Rec v2
                              │
             ┌────────────────┼────────────────┐
             │                │                │
          USER STATE       CONTENT STATE     CONTEXT
             │                │                │
     ┌───────┴───────┐   ┌────┴──────┐    ┌────┴──────┐
     │               │   │           │    │           │
 Long-term       Short-term       Multimodal       City / Social
 interest         interest        content          context
     │               │            │                  │
     │          Multi-behavior    │             Creator graph
     │             sequence       │                  │
     └───────────────┬────────────┴──────────────────┘
                     │
              Situation state
                     │
        ┌────────────┴────────────┐
        │                         │
 Situation confidence       Interest conflict
        │                         │
        └────────────┬────────────┘
                     │
             Dynamic five-way gate
                     │
                     ▼
          Situation-aware retrieval
                     │
                     ▼
              Candidate union
                     │
                     ▼
            Contextual reranking
                     │
        ┌────────────┼────────────┐
        │            │            │
     Relevance    Diversity    Novelty
        │            │            │
        └────────────┼────────────┘
                     │
             Creator exposure
               monitoring
                     │
                     ▼
            Policy / bandit layer
                     │
                     ▼
                USER FEEDBACK
                     │
                     └──────────────► state update
```

The diagram separates representation learning, retrieval, reranking, and online policy control. It also makes explicit that diversity and novelty are post-ranking controls rather than hidden substitutes for relevance learning.

### 5.1 User Representation and Fairness Gate

A user is represented by a temporally decayed interest vector, a temporally decayed behavioral style vector, and a fairness-audited representation.

#### 5.1.1 Demographic and Behavioral Inputs

Let

```math
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
```

be the encoded demographic/contextual vector.

Rather than claiming that interaction behavior measures validated personality traits, we define

```math
\mathbf{p}_i \in \mathbb{R}^{d_p}
```

as a **behavioral engagement-style embedding** constructed from measurable statistics such as completion ratio, skip behavior, rewatch frequency, session activity, and related engagement features.

The fairness representation is:

```math
\mathbf{b}_i =
\mathrm{MLP}_{\mathrm{fair}}
(
\mathbf{d}_i \Vert \mathbf{p}_i
).
```

Protected-attribute information is then subjected to an adversarial fairness objective rather than being directly exposed as an unrestricted scoring term.


### 5.2 Multi-Behavior Event Representation

A short-form interaction contains more information than a binary click. For event $k$, define:

```math
r_{ik}^{watch}
=
\frac{
\min(watch\_duration_{ik},reel\_duration_k)
}{
\max(reel\_duration_k,\epsilon)
}.
```

The event is represented by a typed tuple:

```math
\mathbf{h}_{ik}
=
[
\mathbf{v}_{r_k}
\Vert
\mathbf{e}_{event(k)}
\Vert
r_{ik}^{watch}
\Vert
\Delta t_{ik}
\Vert
\mathbf{e}_{session(k)}
].
```

A fine-grained engagement weight can be defined as:

```math
w_{ik}
=
a_1r_{ik}^{watch}
+
a_2\mathbb{1}[complete]
+
a_3\mathbb{1}[rewatch]
+
a_4\mathbb{1}[like]
+
a_5\mathbb{1}[save]
+
a_6\mathbb{1}[share]
+
a_7\mathbb{1}[comment]
-
a_8\mathbb{1}[early\_skip]
-
a_9\mathbb{1}[late\_skip].
```

The coefficients are learned or calibrated on validation data. In particular, an early skip should be treated as stronger negative evidence than a late skip, while a late skip can still indicate weak or incomplete satisfaction [8].

### 5.3 Long-Term and Short-Term Interest Representation

The original exponential decay is retained as a useful baseline, but v2 separates stable preference from immediate intent.

Let $\mathcal{H}_i^{long}$ denote a long historical window and $\mathcal{H}_i^{short}(t)$ denote a recent window. Then:

```math
\mathbf{e}_{i,long}
=
\mathrm{Encoder}_{long}
(
\mathcal{H}_i^{long}
).
```

The short-term state is:

```math
\mathbf{e}_{i,short}(t)
=
\mathrm{Encoder}_{short}
(
\mathcal{H}_i^{short}(t)
).
```

A learned balance coefficient is:

```math
\alpha_i(t)
=
\sigma
\left(
\mathbf{w}_{mix}^{\top}
[
\mathbf{e}_{i,long}
\Vert
\mathbf{e}_{i,short}(t)
\Vert
\mathbf{s}_i(t)
]
+
b_{mix}
\right).
```

The unified interest state is:

```math
\mathbf{e}_{i,interest}^{v2}(t)
=
\alpha_i(t)\mathbf{e}_{i,long}
+
(1-\alpha_i(t))\mathbf{e}_{i,short}(t).
```

This formulation allows a user with stable preferences and a user undergoing a temporary interest shift to receive different mixtures of historical and recent evidence [7].

### 5.4 Temporally Decayed Interest Representation

For an interaction event $k$ occurring at time $t_k$, let $w_{ik}$ denote its signed engagement weight. A generic temporally decayed interest embedding is:

```math
\mathbf{e}_{i,interest}(t)
=
\sum_{k \in \mathcal{H}_i^T}
w_{ik}
\exp[-\lambda_t(t-t_k)]
\mathbf{m}_k.
```

Here:

- $w_{ik}>0$ represents positive engagement;
- $w_{ik}<0$ may represent a skip or other negative signal;
- $\lambda_t>0$ is the interest decay rate;
- $\mathbf{m}_k$ is the reel embedding.

The implementation separates positive and negative evidence when constructing situation state so that a disliked reel does not automatically imply that its detected situation is desirable.

A second embedding can be constructed for content style or audio preference when those features are available:

```math
\mathbf{e}_{i,style}(t)
=
\sum_{k \in \mathcal{H}_i^T}
w_{ik}
\exp[-\lambda_t(t-t_k)]
\mathbf{a}_k^{style}.
```

---

### 5.5 Music Lyrics, Audio, and User Playlist Modeling

Music is treated as a first-class recommendation signal rather than an incidental reel attribute. For each track $m$, the system stores a bounded lyrics/text sequence and an acoustic feature vector. The research prototype uses tokenized lyrics-like features and fixed acoustic descriptors; no copyrighted lyric corpus is bundled with the repository.

Let $q_m$ denote the lyrics token sequence and $\boldsymbol{\phi}_m$ the acoustic feature vector. A lightweight Transformer encodes the lyrics:

```math
\mathbf{e}_{m,lyrics} = \mathrm{MeanPool}(\mathrm{Transformer}(q_m)).
```

The acoustic representation is projected into the same latent space and fused with the lyric representation:

```math
\mathbf{e}_{m,music} = W_p\tanh(W_l\mathbf{e}_{m,lyrics} + W_a\boldsymbol{\phi}_m + \mathbf{b}).
```

This representation is content-derived, so a newly published reel can obtain a music representation before it accumulates engagement.

For user $u_i$, let $\mathcal{P}_i$ be the tracks available in the user's playlist. With non-negative playlist weights $\omega_{im}$, the playlist profile is:

```math
\mathbf{e}_{i,playlist} = \frac{\sum_{m\in\mathcal{P}_i}\omega_{im}\mathbf{e}_{m,music}}{\sum_{m\in\mathcal{P}_i}\omega_{im}+\epsilon}.
```

The profile is combined with music evidence extracted from the user's reel history:

```math
\mathbf{e}_{i,music} = \mathbf{e}_{i,music}^{history} + \rho\,\mathbf{e}_{i,playlist}.
```

Here, $\rho$ controls the strength of playlist evidence and should be tuned on validation data. This design means that a user may influence reel ranking through both explicitly curated playlists and observed music interactions.

The dedicated music-match score for a candidate reel $r_k$ with associated track $m(r_k)$ is:

```math
S_{music}(u_i,r_k)=\cos(\mathbf{e}_{i,music},\mathbf{e}_{m(r_k),music}).
```

A separate situation-to-lyrics diagnostic may also be computed as:

```math
S_{situation-music}(u_i,m)=\cos(\mathbf{s}_i(t),\mathbf{z}_{m,lyrics}),
```

when the lyric encoder is projected into the life-situation space. This auxiliary diagnostic is optional; the implementation's primary ranking term is $S_{music}$.

### 5.6 Life-Situation Classifier

Let each reel expose the textual sequence

```math
z_k =
\mathrm{Caption}(r_k)
\Vert
\mathrm{OCR}(r_k)
\Vert
\mathrm{ASR}(r_k).
```

A Transformer encoder produces:

```math
\mathbf{e}_{k,text}
=
\mathrm{MeanPool}
(
\mathrm{Transformer}(z_k)
).
```

The system predicts a multi-label life-situation probability vector:

```math
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
```

Possible categories include:

```math
\{
\text{job\_search},
\text{new\_parent},
\text{moving},
\text{graduation},
\ldots
\}.
```

Hashtags such as `#jobsearch`, `#newmom`, or `#movingcities` can provide weak labels during early bootstrapping. Weak labels should not be treated as ground truth; a later human-annotated validation set is required to estimate classifier quality and label bias.

#### Multimodal Extension

For a production system, visual and acoustic representations can be added:

```math
\mathbf{e}_{k,multi}
=
f(
\mathbf{e}_{k,text},
\mathbf{e}_{k,visual},
\mathbf{e}_{k,audio}
).
```

The key design choice is that these representations support **situation classification**, rather than forcing a separate music-matching objective to dominate the recommendation score.

---


### 5.7 Adaptive Multimodal Fusion

Instead of concatenating text, visual, and acoustic representations with fixed importance, v2 learns modality reliability [9]. Let:

```math
\mathcal{M}_k
=
\{
\mathbf{e}_{k,text},
\mathbf{e}_{k,visual},
\mathbf{e}_{k,audio}
\}.
```

A modality gate is:

```math
\boldsymbol{\alpha}_{k}^{modal}
=
\mathrm{softmax}
\left(
G_{modal}
(
[
\mathbf{e}_{k,text}
\Vert
\mathbf{e}_{k,visual}
\Vert
\mathbf{e}_{k,audio}
]
)
\right).
```

The fused representation is:

```math
\mathbf{e}_{k,multi}
=
\sum_{m\in\mathcal{M}_k}
\alpha_{k,m}^{modal}
\mathbf{e}_{k,m}.
```

Missing or low-confidence modalities receive an availability mask:

```math
\tilde{\alpha}_{k,m}
=
\frac{
\alpha_{k,m}^{modal}a_{k,m}
}{
\sum_j\alpha_{k,j}^{modal}a_{k,j}+\epsilon
}.
```

The normalized weights are then used for fusion. This prevents a missing ASR track, weak OCR, or noisy audio channel from dominating the representation.

#### Optional Offline MLLM Semantic Enrichment

An optional preprocessing stage can use a multimodal large language model to generate a compact semantic description of each reel. The generated description is cached and encoded offline:

```math
\mathbf{e}_{k,MLLM}
=
\mathrm{TextEncoder}
(
\mathrm{MLLM}(r_k)
).
```

The enriched content state is:

```math
\mathbf{e}_{k,content}
=
\mathrm{Fuse}
(
\mathbf{e}_{k,multi},
\mathbf{e}_{k,MLLM}
).
```

The MLLM is not required during online ranking. This keeps serving latency and infrastructure requirements bounded while allowing richer semantics to enter the content representation [10].

### 5.8 Temporal Life-Situation State

Life situations are transient and may have different persistence times.

Define:

```math
\Lambda =
\mathrm{diag}
(
\lambda_{s_1},
\dots,
\lambda_{s_Q}
).
```

The user situation state is:

```math
\mathbf{s}_i(t)
=
\sum_{k\in\mathcal{H}_i^T}
w_{ik}^{+}
\exp[-\Lambda(t-t_k)]
\mathbf{p}_k,
```

where

```math
w_{ik}^{+}=\max(w_{ik},0).
```

Using positive engagement for situation-state accumulation prevents negative feedback from strengthening a situation simply because a user disliked content associated with it.

When explicit user-declared states are available, the system may preserve a floor:

```math
\mathbf{s}_i(t)
\leftarrow
\max
\left(
\mathbf{s}_i(t),
\kappa \mathbf{s}_i^{explicit}
\right),
\qquad
0<\kappa\le1.
```

This is intended as a product-policy mechanism rather than a universally optimal assumption; the value of $\kappa$ should be validated empirically.

---

### 5.9 City Match

The primary spatial signal is city anchored.

For user city $c_i$ and reel city $c(r_k)$:

```math
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
```

Here:

- $d(\cdot,\cdot)$ is a city-centroid distance;
- $\tau$ controls the spatial bandwidth;
- $\eta$ determines how strongly a nearby non-matching city can contribute.

The exact city match is intentionally the dominant locality feature. Regional graph propagation is secondary.

---

### 5.10 Social-Spatial Graph Propagation

Let the social graph be:

```math
\mathcal{G}=(\mathcal{V},\mathcal{E})
```

where each node corresponds to a user and each edge represents an allowed social connection.

Each user starts with a privacy-protected location representation:

```math
\mathbf{g}^{(0)}_i=\mathbf{h}(c_i).
```

For K-hop propagation:

```math
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
```

The final graph representation is:

```math
\mathbf{g}_i
=
\sum_{\ell=0}^{K}
\omega_\ell
\mathbf{g}^{(\ell)}_i,
\qquad
\sum_{\ell=0}^{K}\omega_\ell=1.
```

A uniform choice is:

```math
\omega_\ell=\frac{1}{K+1}.
```

This follows the layer-combination spirit of LightGCN [2].

#### Cold Network

For a user with no graph neighbors, the representation remains:

```math
\mathbf{g}_i=\mathbf{g}^{(0)}_i.
```

Thus a newly registered user is not forced to inherit information from a nonexistent neighborhood.

---

### 5.11 Local Differential Privacy

Location information can reveal sensitive information when shared across a social graph. A naive approach that applies binary randomized response independently to every location bit with the same nominal $\epsilon$ does not automatically provide an overall $\epsilon$ guarantee for the full vector.

M²S²-Rec therefore uses a RAPPOR-style two-stage approach [6]:

1. a memoized/stable randomized representation for the user's locally held categorical signal;
2. instantaneous randomized response before reporting.

The effective privacy guarantee must be stated in terms of the complete mechanism and composition used by a concrete implementation. The repository records the mechanism parameters rather than claiming that a single per-bit $\epsilon$ describes the privacy loss of the entire multi-dimensional report.

#### Coarse Spatial Resolution

The system should operate on city or similarly coarse administrative units rather than raw GPS coordinates whenever exact coordinates are not required.

A k-anonymity-like minimum cohort rule may additionally be used as a product-policy safeguard:

```math
|\mathcal{N}(i)| < k
\Rightarrow
\mathbf{g}_i\leftarrow\mathbf{g}_{macro}.
```

This is not equivalent to differential privacy. The two mechanisms address different risks and should not be presented as interchangeable guarantees.

---

### 5.12 Situation Relevance Score

Given the user state $\mathbf{s}_i(t)$ and reel situation distribution $\mathbf{p}_k$:

```math
S_{situation}(u_i,r_k)
=
\frac{
\mathbf{s}_i(t)^\top\mathbf{p}_k
}{
\|\mathbf{s}_i(t)\|_2
\|\mathbf{p}_k\|_2
}.
```

For numerical stability, the implementation uses an $\epsilon$-stabilized norm rather than a hard clamp whose derivative can become problematic when a cold-start state is exactly zero.

If a user has no inferred situation history, the score is treated as an unavailable/neutral signal and the learned gate can reduce its contribution.

---

### 5.13 Direct Interest Score

The direct collaborative/content relevance term is:

```math
S_{interest}(u_i,r_k)
=
\cos
\left(
\mathbf{e}_{i,interest}(t),
\mathbf{v}_{r_k}
\right).
```

This term is essential because the model should not rely exclusively on demographic, city, or graph information for warm users.

---

### 5.14 Social-Spatial Reel Score

The graph representation is projected into the reel embedding space:

```math
\tilde{\mathbf{g}}_i
=
\mathbf{W}_g\mathbf{g}_i.
```

The geographic-social compatibility score is:

```math
S_{geo}(u_i,r_k)
=
\sigma
\left(
\tilde{\mathbf{g}}_i^\top
\mathbf{v}_{r_k}
\right).
```

The transpose/dot-product formulation is explicit to avoid the dimensional ambiguity of a scalar-vector multiplication.

---


### 5.15 Situation Confidence, Memory, and Transition Modeling

The situation vector can be uncertain when evidence is sparse or contradictory. Define situation entropy:

```math
H_i^{sit}(t)
=
-
\sum_{q=1}^{Q}
\bar{s}_{iq}(t)
\log
(
\bar{s}_{iq}(t)+\epsilon
),
```

where $\bar{\mathbf{s}}_i(t)$ is the normalized situation distribution. A confidence score can be defined as:

```math
C_i^{sit}(t)
=
1-
\frac{
H_i^{sit}(t)
}{
\log Q
}.
```

The confidence value is used to reduce the influence of an uncertain situation state.

A lightweight situation memory can retain historical states:

```math
\mathcal{M}_i^{sit}
=
\{
(
\mathbf{s}_i(t_j),
t_j,
C_i^{sit}(t_j)
)
\}_{j=1}^{J}.
```

For applications requiring proactive adaptation, a transition model may estimate:

```math
P(
s_{t+1}
\mid
s_t,
\mathcal{H}_t
).
```

The transition model is optional and should be evaluated separately from the core ranking task.

### 5.16 Situation-versus-Interest Conflict

A user's current situation may conflict with their stable preference. Define:

```math
C_i^{conflict}(t)
=
1-
\cos
(
\mathbf{e}_{i,long},
\mathbf{s}_i(t)
).
```

The gate can use both situation confidence and conflict:

```math
\mathbf{x}_i^{gate}
=
[
\mathbf{e}_{i,long}
\Vert
\mathbf{e}_{i,short}(t)
\Vert
\mathbf{s}_i(t)
\Vert
C_i^{sit}(t)
\Vert
C_i^{conflict}(t)
\Vert
\mathbf{g}_i
\Vert
\mathbf{e}_{i,music}
].
```

High-confidence situation changes should be able to move the feed away from stable preferences, while low-confidence or highly contradictory situation evidence should be damped.

### 5.17 Creator/Vlogger Representation

Each reel may be associated with a creator $v(r_k)$. A creator embedding is learned from creator-level interactions:

```math
\mathbf{e}_{creator}(v)
=
\mathrm{Encoder}_{creator}
(
\mathcal{H}_{v}
).
```

The creator compatibility score is:

```math
S_{creator}(u_i,r_k)
=
\cos
(
\mathbf{e}_{i,creator},
\mathbf{e}_{creator}(v(r_k))
).
```

An optional tripartite graph contains users, reels, and creators:

```math
\mathcal{G}_{tri}
=
(
\mathcal{U}
\cup
\mathcal{R}
\cup
\mathcal{V},
\mathcal{E}_{ur}
\cup
\mathcal{E}_{rv}
).
```

Creator information is intentionally kept as an auxiliary retrieval/reranking signal rather than adding another independent gate dimension. This avoids uncontrolled growth of the personalized gate while still modeling creator affinity [12].

### 5.18 Situation-Aware Candidate Retrieval

A production recommender should not rely on the final reranker to discover every relevant item. Candidate generation is split into complementary sources:

```math
\mathcal{C}_{retrieval}
=
\mathcal{C}_{interest}
\cup
\mathcal{C}_{situation}
\cup
\mathcal{C}_{city}
\cup
\mathcal{C}_{creator}
\cup
\mathcal{C}_{trend}.
```

The interest pool retrieves items close to $\mathbf{e}_{i,long}$ and $\mathbf{e}_{i,short}(t)$. The situation pool retrieves items close to $\mathbf{s}_i(t)$. The city pool retrieves locally relevant items, while the creator and trend pools provide creator affinity and controlled exploration.

Candidate provenance is retained as a feature so that the final model can distinguish whether an item was retrieved because of situation, interest, city, creator, or trend evidence.

### 5.19 Dynamic Joint Scoring

A fixed global weight vector assumes every user depends on the same evidence. M²S²-Rec therefore uses a five-way per-user gate:

```math
\boldsymbol{\alpha}_i = [\alpha_i,\beta_i,\gamma_i,\delta_i,\mu_i]
=\mathrm{softmax}(\mathrm{MLP}_{\mathrm{gate}}(\mathbf{x}_i)).
```

The final personalized gate remains five-way:

```math
\boldsymbol{\alpha}_i
=
[
\alpha_i,
\beta_i,
\gamma_i,
\delta_i,
\mu_i
]
=
\mathrm{softmax}
(
\mathrm{MLP}_{gate}
(
\mathbf{x}_i^{gate}
)
).
```

The base contextual score is:

```math
S_{base}(u_i,r_k)
=
\alpha_iS_{situation}(u_i,r_k)
+
\beta_iS_{city}(u_i,r_k)
+
\gamma_iS_{interest}(u_i,r_k)
+
\delta_iS_{geo}(u_i,r_k)
+
\mu_iS_{music}(u_i,r_k).
```

The v2 score then applies confidence, conflict, creator, diversity, and exposure corrections:

```math
\hat{y}_{i,k}
=
S_{base}(u_i,r_k)
+
\xi S_{trend}(r_k)
+
\zeta S_{creator}(u_i,r_k)
+
\lambda_{nov}S_{novelty}(r_k)
-
\lambda_{fat}S_{fatigue}(u_i,r_k)
+
\lambda_{conf}C_i^{sit}(t)S_{situation}(u_i,r_k)
-
\lambda_{conflict}C_i^{conflict}(t)
-
\lambda_{expo}B_{exposure}(u_i,r_k).
```

Fairness is not represented as a sixth gate coordinate. It remains an optimization constraint and output-level diagnostic.

The first five terms are personalized through the gate. The trend term is deliberately outside the learned gate and supervised ranking objective so that temporary platform-wide popularity does not become entangled with the user's representation.

The interpretation is:

- $\alpha_i$: life-situation relevance;
- $\beta_i$: explicit city relevance;
- $\gamma_i$: direct interest relevance;
- $\delta_i$: social-spatial graph relevance;
- $\mu_i$: music and playlist relevance.

A playlist-rich user can therefore receive a larger music weight when music preferences are reliable, while a user with little or no playlist/history evidence can rely more heavily on situation, city, and graph signals.


### 5.20 Temporal Diversity, Novelty, and Exposure Control

To reduce repetition and filter-bubble effects, define the similarity of candidate $r_k$ to recently recommended items $\mathcal{R}_i^{recent}$:

```math
S_{fatigue}(u_i,r_k)
=
\frac{1}{|\mathcal{R}_i^{recent}|}
\sum_{r_j\in\mathcal{R}_i^{recent}}
\cos
(
\mathbf{v}_{r_k},
\mathbf{v}_{r_j}
).
```

A novelty score can be defined from inverse historical exposure:

```math
S_{novelty}(r_k)
=
-\log
(
P_{expose}(r_k)+\epsilon
).
```

For a recommendation list $\mathcal{L}_i$, intra-list diversity is:

$$ILD(\mathcal{L}_i)=1-\frac{2}{|\mathcal{L}_i|\left(|\mathcal{L}_i|-1\right)}\sum_{a<b}\cos\left(\mathbf{v}_{r_a},\mathbf{v}_{r_b}\right).$$

The fatigue term is a ranking penalty, while diversity can also be applied as a list-level reranking constraint. This separates relevance learning from the business-policy decision of how much exploration is appropriate [7].

Creator exposure should also be monitored to prevent a popularity feedback loop. For creator $v$:

```math
E_v
=
\frac{
\#\{
r\in\mathcal{L}:creator(r)=v
\}
}{
|\mathcal{L}|
}.
```

Exposure-share concentration can be summarized using a Gini-style diagnostic:

```math
G_{creator}
=
\mathrm{Gini}
(
\{E_v\}_{v\in\mathcal{V}}
).
```

This metric is diagnostic rather than a universal fairness target; product objectives may intentionally prefer different creator-exposure distributions.

### 5.21 Optional Peer-Matching Surface

If the product goal includes connecting users going through similar circumstances, recommendation of people should be treated as a separate retrieval task.

For two users:

```math
S_{peer}(u_i,u_j)
=
\cos
(
\mathbf{s}_i(t),\mathbf{s}_j(t)
)
\mathbb{1}
[d(c_i,c_j)<\delta_{local}].
```

This layer should be implemented using a city-scoped approximate nearest-neighbor index rather than being conflated with reel ranking.

---

## 6. Training Algorithm

### 6.1 Chronological Data Split

Randomly splitting every interaction into train and test sets can leak future information into the representation.

The preferred protocol is chronological:

```math
D_{train}<D_{valid}<D_{test}.
```

For each user, interactions are ordered by timestamp. The most recent eligible positive interaction(s) are reserved for validation/test according to the experiment protocol.

Cold-start cohorts should be defined separately, for example by users whose training history is empty or below a predefined minimum.

---

### 6.2 Positive and Negative Sampling

For BPR, a training example is:

```math
(u_i,r_i^+,r_i^-).
```

The positive reel must correspond to an observed positive event.

The negative reel should be sampled from candidates that are not known positive interactions in the relevant training/holdout partition.

In particular:

- a skipped reel must not silently become a positive;
- a held-out test positive must not be reused as a negative;
- sampled negatives should not be selected from the user's known positive set.

When the product has meaningful explicit negative feedback, that feedback can be used as weighted supervision rather than being confused with unobserved items.

---


### 6.3 Hard-Negative Mining

Random negatives can be too easy. For each positive reel $r_i^+$, construct a hard-negative pool:

```math
\mathcal{N}_{hard}(u_i)
=
\mathrm{TopN}
(
\mathcal{R}
\setminus
\mathcal{R}_{u_i}^{+},
S_{semantic}(u_i,r)
).
```

A hard negative is then sampled from this pool:

```math
r_i^{-}
\sim
\mathcal{N}_{hard}(u_i).
```

This encourages the model to distinguish a user's relevant reel from semantically similar but non-engaged content.

### 6.3 BPR Loss

For a positive and negative candidate:

```math
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
```

This objective follows the personalized-ranking formulation introduced by Rendle et al. [1].

---

### 6.4 BPR Loss

For a positive and negative candidate:

```math
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
```

This objective follows the personalized-ranking formulation introduced by Rendle et al. [1]. The negative sample should preferentially come from the hard-negative mechanism above when enough candidate information is available.

### 6.5 Exposure-Aware Ranking Loss

Observed interactions are conditioned on prior exposure. Let:

```math
p_{expo}(u_i,r_k)
=
P(
r_k\text{ shown}
\mid
u_i
).
```

A clipped inverse-propensity weight is:

```math
w_{IPW}(u_i,r_k)
=
\min
\left(
w_{max},
\frac{1}{
p_{expo}(u_i,r_k)+\epsilon
}
\right).
```

The weighted ranking loss is:

```math
\mathcal{L}_{IPW}
=
-\frac{1}{B}
\sum_{i=1}^{B}
w_{IPW}(u_i,r_i^+)
\log
\sigma
(
\hat{y}_{i,+}
-
\hat{y}_{i,-}
).
```

When randomized-exposure data are available, the propensity model can be estimated more reliably. KuaiRand is an example of a dataset designed to study exposure bias using randomly exposed videos and rich feedback [11].

### 6.6 Situation Classification Loss

Given weak or human labels $\mathbf{c}_k$:

```math
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
```

Weak hashtag labels should be treated as noisy supervision. A human-labeled subset is recommended for model selection and calibration.

---


### 6.7 Multimodal Consistency Loss

When multiple modalities are available, an optional contrastive loss can prevent the fused representation from ignoring useful channels. For paired views $m$ and $n$:

```math
\mathcal{L}_{con}
=
-
\frac{1}{B}
\sum_{i=1}^{B}
\log
\frac{
\exp(
sim(
\mathbf{e}_{i,m},
\mathbf{e}_{i,n}
)/\tau_c
)
}{
\sum_j
\exp(
sim(
\mathbf{e}_{i,m},
\mathbf{e}_{j,n}
)/\tau_c
)
}.
```

The contrastive term is optional because modality quality can vary. It should be used only when reliable positive cross-modal alignment is available [9].

### 6.8 Situation Calibration Objective

The situation classifier should be calibrated rather than evaluated only by BCE. Temperature scaling can be applied after training:

```math
\mathbf{p}_{k}^{cal}
=
\sigma
\left(
\frac{
\mathbf{z}_k
}{
T_{cal}
}
\right).
```

The calibration temperature is selected on a held-out validation set. Expected calibration error (ECE) and Brier score should be reported in addition to classification accuracy.

### 6.9 Adversarial Fairness Objective

Let $\phi$ denote the shared fair representation and $a_i$ a protected attribute.

The adversary minimizes:

```math
\mathcal{L}_{adv}
=
CE
(
A_\psi(\phi_i),
a_i
).
```

The fairness representation is trained to make protected-attribute prediction difficult. The implementation uses an explicit alternating/minimax update rather than depending on a single opaque gradient-reversal path.

Conceptually:

```math
\min_{\Theta}
\max_{\psi}
\left[
\mathcal{L}_{rank}
-
\lambda_{fair}\mathcal{L}_{adv}
\right]
```

with the sign convention implemented by the actual alternating optimizer.

Fairness must be evaluated independently using group-level metrics; minimizing an adversary's accuracy is not, by itself, proof of fairness.

---

### 6.10 Regularized Joint Objective

The v2 training objective is:

```math
\mathcal{L}
=
\mathcal{L}_{BPR}
+
\lambda_{ipw}\mathcal{L}_{IPW}
+
\lambda_{sit}\mathcal{L}_{sit}
+
\lambda_{con}\mathcal{L}_{con}
+
\lambda_{fair}\mathcal{L}_{adv}
+
\lambda_{reg}\|\Theta\|_2^2.
```

The exposure term should be enabled only when a defensible propensity estimate is available. The contrastive term is optional and should not be used to force agreement between inherently unrelated modalities.

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


### 7.1.1 Fine-Grained Interaction Schema

The event log should additionally preserve:

```text
position_in_feed
impression_timestamp
watch_start_timestamp
watch_end_timestamp
skip_timestamp
rewatch_count
like_timestamp
save_timestamp
share_timestamp
comment_timestamp
session_id
candidate_source
exposure_probability (when available)
```

The impression/exposure fields are required for principled exposure-bias analysis. Position is also important because a non-click at the top of a feed is not equivalent to a non-clicked item that was never displayed.

### 7.2 Reel and Music Content Schema

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
```

A music-track record should additionally contain:

```text
audio_id
lyrics_text or lyrics_tokens
artist_id (optional)
genre/features (optional)
audio_embedding or acoustic_features
```

A user playlist record should contain:

```text
user_id
audio_id
playlist_id
playlist_weight or interaction strength
playlist_timestamp (when available)
```

Only playlist tracks available before the prediction timestamp should contribute to a historical user music vector. The prototype uses synthetic lyrics-like tokens and does not distribute licensed song lyrics.


### 7.2.1 Creator/Vlogger Schema

When creator information is available:

```text
creator_id
creator_timestamp
creator_category (optional)
creator_city (optional/coarse)
creator_follow_relationship (optional)
creator_historical_exposure
```

Creator fields should be timestamped where possible to prevent future creator statistics from leaking into earlier predictions.

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
- labels generated from engagement outcomes that happen after the prediction timestamp;
- playlist tracks added after the prediction timestamp entering the user's historical music vector;
- future playlist edits being treated as historical preferences.

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


### 8.1.1 Retrieval Diversity

Candidate retrieval should preserve source diversity. A practical candidate budget can be allocated as:

```text
interest retrieval      -> stable preference candidates
short-term retrieval   -> recent intent candidates
situation retrieval    -> current-life-context candidates
city retrieval         -> local candidates
creator retrieval      -> creator-affinity candidates
trend retrieval        -> controlled exploration candidates
```

The exact quota should be tuned experimentally. The important design property is that situation relevance can introduce candidates that are absent from a purely historical-interest index.

### 8.2 Exploration

Pure exploitation can reinforce existing interests and social-spatial homophily.

Exploration should be added after the supervised ranking model through a policy layer such as:

- epsilon-greedy exploration;
- Thompson sampling;
- other contextual-bandit policies.

Exploration should not be confused with the offline supervised ranking objective.

### 8.3 Trending and Recency

The personalization model should be supplemented by an independent trend signal:

```math
S_{trend}(r_k,t)
=
f(
\text{recent interactions},
\text{velocity},
\text{time decay},
\text{regional adoption}
).
```

This signal can help newly viral audio, creators, or topics receive exposure before they accumulate enough user-specific interaction history.

### 8.4 Graph Refresh

### 8.4 Exposure and Policy Feedback Loop

The online system should log both recommendation decisions and outcomes:

```text
request_id
user_id
candidate_ids
ranked_ids
candidate_source
gate_weights
situation_confidence
propensity/exposure_probability
display_position
feedback
```

This enables offline replay, inverse-propensity evaluation, creator-exposure analysis, and drift monitoring. Logging only clicks is insufficient for distinguishing recommendation quality from exposure opportunity.

### 8.5 Contextual Bandit Policy Layer

The supervised ranker can produce a candidate slate while a contextual bandit selects controlled exploration actions. A reward can combine multiple outcomes:

```math
R_t
=
a_1watch
+
a_2completion
+
a_3save
+
a_4share
-
a_5early\_skip
+
a_6diversity
-
a_7fatigue.
```

The bandit is intentionally separated from the supervised ranker. This prevents online exploration behavior from contaminating the offline training target while still allowing long-term reward optimization.

### 8.6 Counterfactual Situation Evaluation

A key scientific test is whether changing the situation state actually changes recommendation behavior. For the same user history and candidate set, compare:

```math
P(r\mid u,s_A)
\quad
\mathrm{vs.}
\quad
P(r\mid u,s_B).
```

The candidate set, user history, and non-situation features should remain fixed. This is stronger evidence of situation sensitivity than showing only aggregate correlation between situation labels and clicks.



Recomputing social-spatial embeddings after every interaction is expensive.

The graph representation can therefore be refreshed periodically, for example hourly or according to an application-specific freshness SLA. The freshness interval should be measured as an engineering trade-off rather than asserted to be universally optimal.

---

## 9. Evaluation Framework

### 9.1 Ranking Metrics

For top-$K$ recommendation:

```math
HR@K
=
\frac{1}{|\mathcal{U}|}
\sum_{u\in\mathcal{U}}
\mathbb{I}
[
\mathrm{rank}_u\le K
].
```

NDCG@K is:

```math
NDCG@K
=
\frac{1}{|\mathcal{U}|}
\sum_{u\in\mathcal{U}}
\frac{DCG_u@K}{IDCG_u@K}.
```

The implementation also reports pairwise AUC-style performance, city-hit rate, and the mean five-way gate including the music component.

### 9.2 Cold-Start Evaluation

Report at least:

- overall performance;
- user cold-start performance;
- warm-user performance;
- item cold-start performance where item metadata are available.

This prevents a strong warm-user score from hiding poor onboarding behavior.

### 9.3 Music-Aware Diagnostics

Music-aware recommendation should be evaluated separately from generic ranking. Useful diagnostics include:

- playlist-to-reel music similarity among recommended items;
- top-$K$ hit rate for tracks or genres represented in a user's playlist;
- cold-item performance when the associated track has no interaction history;
- ranking sensitivity under playlist ablation;
- novelty and artist/track coverage to ensure playlist personalization does not collapse discovery.

### 9.4 Fairness Diagnostics

Group-level diagnostics should include, where legally and ethically appropriate:

- Recall@K by group;
- NDCG@K by group;
- exposure/share metrics;
- pairwise performance by group;
- absolute gap and, when appropriate, relative gap.

No single fairness metric is sufficient for all products.

### 9.5 Locality Metrics

The city-aware design motivates:

```math
CityHit@K
=
\frac{
\#\{\text{recommended reels matching user city}\}
}{
K
}.
```

This should be reported alongside diversity metrics so that stronger locality is not mistaken for universally better recommendation.


### 9.7 Fine-Grained Behavior Metrics

Report performance separately for:

- early skip;
- late skip;
- qualified watch;
- completion;
- rewatch;
- like;
- save;
- share;
- comment.

A useful diagnostic is the predicted-versus-observed watch ratio:

```math
WatchRatio
=
\frac{
watch\_duration
}{
\max(reel\_duration,\epsilon)
}.
```

The model should not improve aggregate NDCG by increasing recommendations that generate many rapid skips.

### 9.8 Long-Short Interest Diagnostics

Report performance by users with:

- stable long-term interests;
- strong recent-interest shifts;
- sparse histories;
- conflicting long-term and short-term interests.

Measure the learned coefficient $\alpha_i(t)$ and verify whether it responds to recent preference shifts without collapsing stable personalization.

### 9.9 Situation Confidence and Calibration

Report:

- situation classification macro/micro F1;
- ECE;
- Brier score;
- reliability diagrams;
- ranking performance by situation-confidence quartile.

The expected behavior is that high-confidence situation states produce stronger situation influence, while uncertain states are appropriately damped.

### 9.10 Exposure-Bias Diagnostics

When impression or randomized-exposure data are available, report:

- propensity-weighted Recall@K;
- propensity-weighted NDCG@K;
- exposure-stratified performance;
- performance on randomized/intervened impressions;
- calibration of the exposure model.

This separates recommendation quality from the bias introduced by historical exposure [11].

### 9.11 Temporal Diversity and Fatigue Metrics

In addition to ILD, report a temporal diversity metric over recommendation windows:

```math
TD@K
=
1-
\frac{
1
}{
K-1
}
\sum_{j=1}^{K-1}
sim(r_j,r_{j+1}).
```

Also report repetition rate and the fraction of consecutive recommendations from the same creator or narrow semantic cluster.

### 9.12 Counterfactual Situation Sensitivity

For a fixed candidate set, define:

```math
CSS
=
\frac{1}{|\mathcal{U}|}
\sum_{u}
\frac{
1
}{
|\mathcal{R}_u|
}
\sum_{r\in\mathcal{R}_u}
\left|
P(r\mid u,s_A)
-
P(r\mid u,s_B)
\right|.
```

The metric should be paired with utility evaluation to ensure that situation sensitivity is meaningful rather than simply increasing score variance.

### 9.6 Diversity and Coverage

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
- dynamic five-way gating over situation, city, interest, graph, and music;
- BPR ranking;
- multi-label situation BCE;
- adversarial fairness optimization;
- chronological evaluation;
- cold/warm evaluation splits;
- multi-behavior event weighting and fine-grained skip semantics;
- long/short interest representations;
- adaptive multimodal fusion;
- hard-negative sampling;
- exposure-aware weighting when propensity data are available;
- temporal diversity and fatigue diagnostics.

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

The v2 extensions should be reported in two categories in the implementation repository:

- **Implemented and tested:** modules that have executable code and automated tests;
- **Architecture-ready / research extension:** modules specified in the manuscript but not yet validated on a real benchmark.

A manuscript should never describe an architecture-ready component as experimentally validated merely because its equations are defined.


The repository's automated test suite covers the core architecture plus lyrics-tokenization, playlist construction, music scoring, and future-playlist leakage checks.

---

## 11. Ablation Plan

A complete empirical study should isolate the contribution of each major signal.

Recommended variants are:

| Variant | Situation | City | Interest | Graph | Music/Playlist | Fairness gate |
|---|---:|---:|---:|---:|---:|
| CF baseline | No | No | Yes | No | No | No |
| + City | No | Yes | Yes | No | No | No |
| + Situation | Yes | Yes | Yes | No | No | No |
| + Graph | Yes | Yes | Yes | Yes | No | No |
| + Music/Playlist | Yes | Yes | Yes | Yes | Yes | No |
| + Dynamic Gate | Yes | Yes | Yes | Yes | Yes | Yes |
| Full M²S²-Rec | Yes | Yes | Yes | Yes | Yes | Yes |

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

### 12.5 Expanded Sequential and Multimodal Baselines

For a stronger real-data comparison, include representatives of:

- multi-interest recommendation;
- long/short-term sequential recommendation;
- multimodal graph recommendation;
- fine-grained skip-aware recommendation;
- creator-aware recommendation;
- exposure-debiased recommendation.

M²S²-Rec should not be compared only against older CF baselines. The strongest comparison should include modern sequence, multimodal, graph, and behavior-aware models where reproducible implementations are available.

### 12.6 Dataset Strategy

A staged evaluation is recommended:

1. **Synthetic integration:** validate every module and leakage control.
2. **Public short-video benchmark:** evaluate ranking and multimodal components.
3. **Randomized-exposure benchmark:** evaluate exposure correction and counterfactual claims.
4. **Integrated proprietary or consented dataset:** evaluate city, situation, creator, music, privacy, and fairness jointly.

KuaiRand is particularly relevant for exposure-bias experiments because it contains randomly exposed videos and multiple feedback signals [11]. A dataset with the full city-plus-situation-plus-music combination is still required for end-to-end validation.

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

The reel text encoder has approximate per-reel complexity:

```math
O(L^2d)
```

For $M$ reels this is approximately $O(ML^2d)$. The music encoder adds an analogous term for $M_m$ unique tracks and lyric length $L_m$:

```math
O(M_mL_m^2d_m),
```

plus a linear acoustic projection. Playlist profile construction is $O(NPd)$ for $N$ users with average playlist length $P$.

A dense Transformer encoder has approximate per-reel complexity:

```math
O(L^2d)
```

so corpus preprocessing is approximately:

```math
O(ML^2d).
```

### Graph Propagation

With sparse adjacency and dense node representations, K-hop propagation is approximately:

```math
O(KEd)
```

for a graph representation of width $d$.

If the graph operates directly on a region vector of size $|\mathcal{L}|$, complexity becomes approximately:

```math
O(KE|\mathcal{L}|).
```

### Multi-Behavior and Long/Short Interest Encoding

If the long and short behavior windows contain $L_l$ and $L_s$ events, a Transformer-style sequence encoder has approximate attention complexity:

```math
O(L_l^2d + L_s^2d).
```

A lightweight recurrent or linear-attention encoder can reduce this cost when sequence lengths become large.

### Adaptive Multimodal Fusion

For $J$ modalities and embedding width $d$, the gating layer is approximately:

```math
O(Jd^2)
```

per encoded item after modality embeddings are cached. This is normally small relative to the upstream visual or language encoder.

### Candidate Retrieval

If separate ANN indices are used for interest, situation, city, and creator retrieval, the retrieval cost is approximately the sum of the individual index-query costs. The union size should be bounded before contextual reranking:

```math
|\mathcal{C}_{retrieval}|
\le C_{max}.
```


### Online Reranking

For $C$ retrieved candidates:

```math
O(Cd')
```

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

### 14.7 Playlist, Lyrics, and Music Data Availability
Real deployments must obtain playlist and lyrics data under applicable platform terms, copyright constraints, and user-consent requirements. The prototype therefore uses synthetic lyrics-like token sequences rather than distributing licensed song lyrics. Lyrics language, transcription quality, and explicit playlist semantics may materially affect performance.

### 14.8 Missing Real-World Benchmark

### 14.9 Exposure Model Misspecification

Inverse-propensity correction is only as reliable as the exposure model. Poorly estimated or extremely small propensities can increase variance. Clipping, randomized exposure data, and sensitivity analysis are therefore required.

### 14.10 Situation Uncertainty

Situation inference is not ground truth. The system can overreact to ambiguous content, noisy OCR/ASR, or culturally specific expressions. Confidence-aware gating and human-annotated calibration sets reduce but do not eliminate this risk.

### 14.11 Multimodal Missingness

Visual, audio, ASR, and OCR signals are not uniformly available. The adaptive fusion module must be evaluated under realistic missing-modality conditions rather than only on complete examples.

### 14.12 Creator and Popularity Feedback

Creator affinity can reinforce already popular creators. Exposure-share monitoring and creator-level diversity constraints are required to distinguish useful creator personalization from runaway popularity reinforcement.

### 14.13 MLLM Semantic Bias

Offline MLLM descriptions may introduce hallucinations, cultural assumptions, or systematic stylistic bias. MLLM-derived features should therefore be treated as auxiliary representations, validated against human annotations, and versioned for reproducibility.

### 14.14 Counterfactual Validity

Changing a situation vector while holding every other feature fixed is a useful sensitivity test but is not automatically a causal estimate of user behavior. Strong causal claims require randomized or otherwise defensible identification assumptions.



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
PAPER-M2S2-Rec-v2-Enhanced.md
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
8. music lyrics/audio representation and user-playlist profiling;
9. explicit cold-start handling;
10. chronological offline evaluation;
11. separate exploration and trend layers for production.

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
[7] Pan Gu, Haiyang Hu, Dongjing Wang, Dongjin Yu, and Guandong Xu. 2024. **Temporal Diversity-Aware Micro-Video Recommendation with Long- and Short-Term Interests Modeling**. Neural Processing Letters, 56, 194. DOI: 10.1007/s11063-024-11652-7.

[8] Sanghyuck Lee, Sangkeun Park, and Jaesung Lee. 2025. **Exploiting Fine-Grained Skip Behaviors for Micro-Video Recommendation**. Proceedings of the AAAI Conference on Artificial Intelligence, 39(11), 12004–12012. DOI: 10.1609/aaai.v39i11.33307.

[9] Zixuan Yi, Xi Wang, Iadh Ounis, and Craig Macdonald. 2022. **Multi-modal Graph Contrastive Learning for Micro-video Recommendation**. Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, 1807–1811. DOI: 10.1145/3477495.3532027.

[10] Marco De Nadai, Andreas Damianou, and Mounia Lalmas. 2025. **Describe What You See with Multimodal Large Language Models to Enhance Video Recommendations**. arXiv:2508.09789.

[11] Chongming Gao, Shijun Li, Yuan Zhang, Jiawei Chen, Biao Li, Wenqiang Lei, Peng Jiang, and Xiangnan He. 2022. **KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos**. Proceedings of the 31st ACM International Conference on Information & Knowledge Management, 3953–3957. DOI: 10.1145/3511808.3557624.

[12] Weijiang Lai, Beihong Jin, Beibei Li, Yiyuan Zheng, and Rui Zhao. 2024. **A Vlogger-augmented Graph Neural Network Model for Micro-video Recommendation**. arXiv:2405.18260.

[13] Zixuan Yi, Xi Wang, Iadh Ounis, and Craig Macdonald. 2022. **Multi-modal Graph Contrastive Learning for Micro-video Recommendation**. The use of contrastive multimodal representation learning motivates the optional multimodal consistency objective in M²S²-Rec v2.

[14] **MVideoRec: Micro Video Recommendations through Modality Decomposition and Contrastive Learning**. ACM Transactions on Information Systems. DOI: 10.1145/3711855.


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

The accompanying source code may be distributed under the Apache License 2.0 as specified by the repository's `LICENSE` file. The presence of this manuscript in the repository does not, by itself, place the manuscript under the Apache License 2.0. The manuscript remains subject to the author's copyright and any separate publication or licensing terms.### 11. Ablation Plan

The expanded study should isolate the contribution of each major signal and avoid attributing gains to a bundle of simultaneous changes.

| Variant | Situation | Long/Short | Multi-Behavior | City | Graph | Music | Creator | Diversity | Exposure | Fairness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CF baseline | No | No | No | No | No | No | No | No | No | No |
| + City | No | No | No | Yes | No | No | No | No | No | No |
| + Situation | Yes | No | No | Yes | No | No | No | No | No | No |
| + Long/Short | Yes | Yes | No | Yes | No | No | No | No | No | No |
| + Multi-Behavior | Yes | Yes | Yes | Yes | No | No | No | No | No | No |
| + Graph | Yes | Yes | Yes | Yes | Yes | No | No | No | No | No |
| + Music | Yes | Yes | Yes | Yes | Yes | Yes | No | No | No | No |
| + Creator | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | No | No |
| + Diversity | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | No |
| + Exposure Correction | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No |
| Full M²S²-Rec v2 | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |

Additional controlled ablations should vary:

- K-hop depth;
- privacy parameters;
- long/short window lengths;
- temporal decay rates;
- early-versus-late skip weighting;
- hard-negative pool size;
- modality-gate architecture;
- missing-modality masks;
- MLLM semantic enrichment on/off;
- creator graph on/off;
- situation confidence gating on/off;
- situation-interest conflict feature on/off;
- fatigue penalty;
- novelty strength;
- creator-exposure regularization;
- propensity clipping threshold;
- weak-label versus human-label training;
- calibration temperature;
- candidate-source quotas;
- contextual-bandit exploration strength.

Ablations should be evaluated under the same chronological split and candidate protocol. For real-data studies, confidence intervals over multiple seeds or temporal windows should be reported.




---

## v2 Implementation Status and Recommended Build Order

The expanded manuscript intentionally distinguishes architecture from validated implementation. A practical implementation order is:

1. implement long/short interest encoders;
2. implement multi-behavior event weighting and early/late skip handling;
3. implement hard-negative sampling;
4. implement adaptive multimodal fusion with missing-modality masks;
5. implement situation confidence and situation-interest conflict;
6. implement temporal diversity/fatigue reranking;
7. implement exposure logging and propensity-aware loss when exposure probabilities are available;
8. add creator representation;
9. add situation-aware candidate retrieval;
10. add calibration and counterfactual situation-sensitivity evaluation;
11. optionally add offline MLLM semantic enrichment;
12. optionally add the contextual-bandit policy layer.

The manuscript should only move an item from “architecture-ready” to “implemented and tested” after executable code, unit tests, leakage tests, and a reproducible experiment have been added.

The central scientific hypothesis of M²S²-Rec v2 is:

```math
	ext{Current Situation}
+
	ext{Persistent Preference}
+
	ext{Behavior Dynamics}
+
	ext{Local Context}
+
	ext{Content Semantics}
+
	ext{Controlled Diversity}

ightarrow
	ext{Contextual Recommendation}.
```

This formulation preserves the original identity of M²S²-Rec while making the model more explicit about behavior granularity, temporal dynamics, multimodal reliability, exposure bias, uncertainty, diversity, creator effects, and production serving.

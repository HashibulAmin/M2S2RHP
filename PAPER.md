# M²S²-Rec: A City-Anchored, Life-Situation-Aware Architecture for Short-Form Video Recommendation

## Abstract

Short-form video platforms traditionally rely on collaborative filtering and content-based filtering in isolated silos, often failing to leverage social-spatial signals and situational context. This position paper proposes a novel recommendation architecture that shifts the paradigm from diffuse regional music-affinity to city-anchored, life-situation relevance. By integrating a multi-modal life-situation classifier, temporal decay mechanisms for explicit and implicit interactions, and a $K$-hop spatial graph neural network guarded by Local Differential Privacy (LDP), the proposed framework addresses both cold-start and hyper-local routing challenges. Furthermore, we introduce an adversarially-trained fairness gating mechanism to mitigate demographic bias. By utilizing a dynamically learned per-user gating mechanism rather than static fusion weights, the model proactively adapts to varying user contexts.

## 1. Introduction

The consumption of short-form video content is defined by rapid interactions, heavy reliance on audio-visual synchronicity, and regional trend clusters. Recommendation architectures historically treat user history and visual features as separate entities, neglecting the physical topology of the user base and the immediate life circumstances of the consumer.

This architecture paper establishes a blueprint for M²S²-Rec, a system designed to anchor video delivery to a user's specific city and their transient life situations (e.g., job transition, new parenthood, relocation). By modeling these dynamics, the architecture answers:

1. **Who is watching, and what are they going through?** Evaluated via temporal life-situation vectors.

2. **Where are they consuming?** Modeled through an exact city-match score and a multi-hop social-spatial Graph Neural Network (GNN).

3. **Is the content contextually relevant?** Evaluated via a multi-label life-situation classifier parsing audio, text, and visual signals.

## 2. Related Work

* **Collaborative Filtering & Sequence Modeling:** Standard architectures utilize models like SASRec to prioritize recent interactions; however, these models struggle with the extreme data sparsity of short-form video and the cold-start routing of newly uploaded content.

* **Graph Neural Networks (GNNs):** Architectures like LightGCN simplify graph convolutions to capture higher-order connectivities efficiently. Spatial-aware GNNs construct localized sub-graphs, though prior implementations often misapply privacy mechanisms across multi-dimensional data.

* **Fairness in Recommendation:** Raw demographic conditioning risks algorithmic discrimination and feedback-loop stereotyping. Adversarial debiasing techniques are required to ensure compliance with regulatory frameworks.

## 3. Methodology

The proposed framework processes user-item interactions through distinct, dynamically gated representation pipelines.

### 3.1 User Vectorization and Adversarial Fairness

Raw demographic categories (e.g., gender, age, occupation, location) present a direct fairness liability if explicitly concatenated into the scoring path. Instead, demographic vectors $\mathbf{d}_i$ and behavioral engagement embeddings $\mathbf{p}_i$ (derived from engagement styles rather than contested psychometric traits) are passed through a bias-audited gating multi-layer perceptron (MLP):

$$\mathbf{b}_i = \text{MLP}_{\text{fair}}(\mathbf{d}_i \Vert \mathbf{p}_i)$$

The vector $\mathbf{b}_i$ is trained under an adversarial fairness constraint utilizing gradient-reversal against predicting protected attributes from the final prediction $\hat{y}$, allowing demographic signals to inform relevance without serving as a discriminatory lever.

User interest is captured via a temporally decayed interaction history that incorporates negative implicit signals (e.g., fast-scrolls or skips yield $w_{ik} < 0$):

$$\mathbf{e}_{i,\text{interest}}^{\text{temporal}} = \sum_{k \in \mathcal{H}_i^T} w_{ik} e^{-\lambda_t (t_{\text{now}} - t_k)} \mathbf{m}_k$$

The complete user state vector concatenates these representations:

$$\mathbf{x}_i = \big[\mathbf{e}_{i,\text{interest}}^{\text{temporal}} \Vert \mathbf{e}_{i,\text{music}}^{\text{temporal}} \Vert \mathbf{b}_i \big]$$

### 3.2 Life-Situation Classifier

Rather than relying solely on music/audio alignment, content relevance is driven by a life-situation taxonomy $\mathcal{S} = \{s_1,\dots,s_Q\}$. The text tower ingests an Automatic Speech Recognition (ASR) transcript alongside caption and Optical Character Recognition (OCR) data:

$$\mathbf{e}_{k,\text{text}} = \text{MeanPooling}\Big(\text{Transformer}\big(\text{Caption}(r_k)\Vert\text{OCR}(r_k)\Vert\text{ASR}(r_k)\big)\Big)$$

A probability distribution across life situations is generated via a sigmoid classifier, trained initially using weak labels bootstrapped from hashtags (e.g., `#newmom`, `#jobsearch`):

$$P\big(\mathbf{c}(r_k)\mid r_k\big) = \text{sigmoid}\big(\mathbf{W}_s\mathbf{e}_{k,\text{text}} + \mathbf{b}_s\big) \in [0,1]^Q$$

### 3.3 Temporal Life-Situation Vector

Life situations are transient states requiring distinct decay rates. A job search resolves rapidly, whereas new parenthood persists. We apply a per-category decay rate matrix $\Lambda = \text{diag}(\lambda_{s_1},\dots,\lambda_{s_Q})$:

$$\mathbf{s}_i(t) = \sum_{k \in \mathcal{H}_i^T} w_{ik} e^{-\Lambda (t - t_k)} P\big(\mathbf{c}(r_k)\mid r_k\big)$$

If the platform captures explicit self-declared situations (e.g., via onboarding tags), a threshold floor prevents them from fully decaying:

$$\mathbf{s}_i(t) \leftarrow \max\big(\mathbf{s}_i(t), \kappa \cdot \mathbf{s}_i^{\text{explicit}}\big), \quad \kappa \in (0,1]$$

### 3.4 City Match and Social-Spatial Graph

The primary spatial signal is anchored to a defined vocabulary of cities $\mathcal{C}_{\text{cities}}$. The exact city match score penalizes geographical distance $d(c_i, c(r_k))$:

$$S_{\text{city}}(u_i, r_k) = \mathbb{1}[c_i = c(r_k)] + \eta \exp\left(-\frac{d(c_i, c(r_k))}{\tau}\right) \mathbb{1}[c_i \ne c(r_k)]$$

Secondary regional trends propagate through a $K$-hop spatial graph leveraging LightGCN architecture. A brand-new user with zero connections falls back to their own location explicitly:

$$\mathbf{g}_i^{(0)} = \mathbf{h}(l_i), \qquad \mathbf{g}_i^{(k)} = \sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{\vert{}\mathcal{N}(i)\vert{}\vert{}\mathcal{N}(j)\vert{}}} \mathbf{g}_j^{(k-1)}, \qquad \mathbf{g}_i = \sum_{k=0}^{K} \tfrac{1}{K+1}\mathbf{g}_i^{(k)}$$

**Privacy Guarantee:** To solve the Local Differential Privacy (LDP) composition error inherent in multi-hot vectors, we apply a RAPPOR-style two-stage randomized response mechanism (memoization combined with Bloom-filter instantaneous randomization).

### 3.5 Dynamic Joint Scoring

Static weighting coefficients assume identical signal reliance across all users. We implement a learned per-user gating mechanism $\text{MLP}_{\text{gate}}$, ensuring cold-start users lean heavily on city and demographic priors, while tenured users rely on deep interest history:

$$[\alpha_i,\beta_i,\gamma_i,\delta_i] = \text{softmax}\big(\text{MLP}_{\text{gate}}(\mathbf{x}_i)\big)$$

Relevance is calculated by computing the cosine similarity between the user's situation vector and the reel's classification:

$$S_{\text{situation}}(u_i, r_k) = \frac{\mathbf{s}_i(t)^\top P\big(\mathbf{c}(r_k)\mid r_k\big)}{\Vert{}\mathbf{s}_i(t)\Vert{}_2\big\Vert{}P(\mathbf{c}(r_k)\mid r_k)\big\Vert{}_2}$$

The final prediction formulation includes the direct interest term $S_{\text{interest}}$ and the graph term $S_{\text{geo}}$:

$$\hat{y}_{i,k} = \alpha_i S_{\text{situation}}(u_i,r_k) + \beta_i S_{\text{city}}(u_i,r_k) + \gamma_i S_{\text{interest}}(u_i,r_k) + \delta_i S_{\text{geo}}(u_i,r_k)$$

### 3.6 Optional Peer-Matching Layer

If the platform prioritizes user-to-user communication based on shared experiences, a distinct peer-matching surface can be generated via a separate Approximate Nearest Neighbor (ANN) index scoped to the local city:

$$\text{Sim}(u_i, u_j) = \cos\big(\mathbf{s}_i(t), \mathbf{s}_j(t)\big) \cdot \mathbb{1}\big[d(c_i, c_j) < \delta_{\text{local}}\big]$$

## 4. Production Considerations

Transitioning this theoretical design into deployment requires addressing specific feed-product realities:

* **Exploration (Bandit Layer):** Pure exploitative ranking driven by social-spatial homophily risks filter-bubble collapse. The final gating weights are wrapped in an exploration mechanism, such as Thompson sampling or $\epsilon$-greedy.

* **Recency/Trending Boost:** Short-form algorithms require a trending-decay mechanism completely independent of user personalization to capture audio tracks or topics experiencing viral adoption.

* **Live-Edge Invalidation Cost:** Real-time GNN propagation per user interaction is computationally prohibitive. We dictate batch-refreshing the graph embeddings $\mathbf{g}_i$ on an hourly interval, bounding the system's online cost.

## 5. Evaluation Framework

Because this is a position and architecture proposal, empirical validation serves as the immediate future work.

* **Offline Validation:** Experiments will utilize public datasets like MicroLens, though we acknowledge the inherent validity risks in stitching disparate datasets to simulate unified multi-modal user profiles.

* **Online A/B Testing:** Requires measuring the delta in user retention, cold-start drop-off rates, and hyper-local content discovery against standard two-tower models.

## 6. Conclusion

The M²S²-Rec architecture fundamentally shifts short-form video personalization from generic regional assumptions to explicit city-anchored, life-situation contexts. By rectifying previous methodological weaknesses—specifically through adversarial debiasing, dynamically learned gating, and mathematically sound privacy mechanisms—this framework provides a robust blueprint for developing context-aware recommendation systems tailored for hyper-local trend velocity.

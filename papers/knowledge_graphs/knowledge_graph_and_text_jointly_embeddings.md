# pTransE: Knowledge Graph and Text Jointly Embeddings

> Zhen Wang, Jianwen Zhang, Jianlin Feng, Zheng Chen (EMNLP 2014)
> Affiliation: Microsoft & SYSU(CN)
> Paper: [Link](https://www.aclweb.org/anthology/D14-1167.pdf)

## TL;DR

Goal: Score any candidate relational facts between entity and words

## Introduction
Embeddings attempt to preserve the relations between entities in the knowledge graph or concurrences of words in the text corpus, that is learnt by minimizing global loss function.

This paper proposes to embed knowledge graph embeddings and word representation embeddings into the same continuous vector space using a coherent probabilistic model. 

## Related Work
### Knowledge Graph Embeddings
Knowledge Graph expresses relation facts in the form of a $(\text{head, relation, tail})$ triplet and entities are represented as a $k$-dimension vector. ==With the goal of capturing local and global connectivity patterns.== TransE interpretes a **relation** as a translation from the head entity.

$$
\text{object} + \text{relation} \approx \text{subject}
$$

Problems:
- Knowledge graph completion (missing facts of entities/relations, *out-of-kb*)
- Only able to reason from facts within current KGs

### Word Embeddings
Word embeddings are learnt from an unlabeled corpus by predicting the context of each word (SkipGram) or predicting the current word given its context (CBOW). ==Where the goal is to capture the semantic and syntactic relations between words.== A **relation** is also represented as the translation between word embeddings

$$
\text{China} - \text{Beijing} \approx \text{Japan} - \text{Tokyo}
$$

Problem: 
- Does not know the relation between entity pairs

### Relational Facts Extraction
Identifying local text patterns from free text that express a certain relation and making predictions. Knowledge embeddings are complimentary for this task.

Problem:
- Do not utilize the the evidence from knowledge graphs
- Couldn't solve the problem of OOKB

## Methodology
**Preface**
- $(h,r,t) \in \Delta$ represents the head entity $h$, relation $r$ and tail entity $t$ of the knowledge graph $\Delta$
    - $h, t \in \mathcal{E}$ are all the entities within $\Delta$
    - $r \in \mathcal{R}$ are all the relations within $\Delta$
- $\mathcal{V}$ is all the words/phrases within a corpus $\Gamma$
    - $(w,v) \in \mathcal{C}$ represents the concurrence of word pairs 
    - $r_{wv}$ represents the relationship $r_{wv}$ between words $w$ and $v$
- $\mathcal{I} = \mathcal{E} + \mathcal{V}$ is the combination of all words and entities
- $\mathcal{A}$ denotes the anchors words within the corpus that is also a Freebase entity

**Modeling**
- A candidate fact $\mathcal{z}(h, r, t)$ is scored based on $b - \frac{1}{2}|| h + r - t ||^{2}$. Where,
    - $h$ is supervised for the Knowledge Model
    - $h_{wv}$ is hidden for the Text Model
    - $b$ is a constant bias ==($b=7$)==
- The proposed model **pTransE** consists of:
    - Knowledge model
    - Text model
    - Alignment model
- The objective function is maximizing the likelihood of all three models
$$
\mathcal{L} = \mathcal{L_K} + \mathcal{L_T} + \mathcal{L_A}
$$
- Because $|\mathcal{I}|$ and $|\mathcal{V}|$ is very large, we simplify the objective by using Negative Sampling into a Binary Classification problem
    - $\text{Pr}(D= 1|h,r,t) =\sigma(z(h,r,t))$
    - $\text{Pr}(D= 1|w,v) =\sigma(z(w′,v))$
    - Where the probability of the triplet facts being true would lead to $(D=1) D\in \{0,1\}$ and $\sigma(x)$ is the sigmoid function
$$
log \text{Pr}(1|h,r,t)+\sum_{i=1}^c \mathbb{E}_{\tilde{h}_i\tilde{}\text{Pr}_{neg}(\tilde{h}_i)} [Pr(0|\tilde{h}_i,r,t)]
$$
- Where $c$ is the number of negative examples to be discriminated for each positive example
    - For $\Delta$, the corrupted head/relation is sampled by a uniform distribution over $\mathcal{I}$ for corrupted entities and $\mathcal{R}$ for relations
    - For $\Gamma$, $c$ words are sampled fromthe unigram distribution raised to the 3/4rd power (same as Skip-gram)
    - The alignment model is absorbed by the text model and knowledge model 
- Stochastic Gradient Descent is used for optimization, and it is done simultaneously for all models

### Knowledge Model

The goal is to ==maximize the conditional likelihood of existing fact triplets==. Where $\mathcal{z}(h,r,t)$ is considered to be large if the triplet is true.
$$
\mathcal{L}_K = \sum_{(h,r,t)\in\Delta} \mathcal{L}_f (h,r,t)\\
$$

Where, 
$$
\mathcal{L}_f(h,r,t) = 
log \text{Pr}(h|r,t) + 
log \text{Pr}(t|h,r) + 
log \text{Pr}(r|h,t) \\
$$

and

$$
\text{Pr}(h|r,t) = 
\text{Pr}(t|h,r) = 
\text{Pr}(r|h,t) = 
\text{softmax}(\mathcal{z}(h,r,t))
$$

**Notes**:
- Compared to TransE which uses margin ranking loss, we define the knowledge model as a probabilistic model. Thus, there is no need to constrict the norm $\gamma$.
- 

### Text Model
Relational concurrence assumption
: where for two words $w$, $v$ within a context, there is a relation $r_{wv}$ between them. 

The goal is to ==maximize the conditional likelihood of the concurrences of pairs of words== in text windows
$$
\mathcal{L}_f = \sum_{(w,v)\in\mathcal{C}} n_{wv} log \text{Pr}(w|v)
$$

Where, 
- $\mathcal{C}$ is all distinct pairs of words concurring in a fixed window
- $n_{wv}$ is the number of concurrences of the pair $(w,v)$

**Note**:
- This model is almost equivalent to skip-gram
- This problem is actually ill posed as
    - $r_{wv}$ is extremely large $\approx|V| \times \bar{N}$
- Hence, we need to reduce the size of variable by estimating:
  Let $w' = w+r_{wv}$, then

$$
\mathcal{z}(w, r_{wv}, v) \triangleq \mathcal{z}(w', v) = b - \frac{1}{2}||w' -v||^2
$$

and

$$
\text{Pr}(w|r_{wv}, v) \triangleq \text{Pr}(w|v) = \text{softmax}(w', v)
$$

### Alignment Model
Goal: Embed Entities (KG) and Words (Pre-trained Word Representations) into same continuous vector space

There are two alignment methods proposed:
1. Alignment via Wikipedia Anchors
    - the phrase $v = e_v$, a Freebase entity
    - comparatively small, but high quality
$$
\mathcal{L}_{AA} = \sum_{(w,v) \in \mathcal{C},v \in \mathcal{A}} \text{log Pr}(w|e_v)
$$

2. Alignment via Names of Entities
    - if $h$ has a name $w_h$ (or $w_t$) and $w_h \in \mathcal{V}$, we would generate a triplet $(w_h, r, t)$ (or $(h, r, w_t)$ and $(w_h, r, w_t)$)
    - risky, as could contaminate alignment results

$$
\mathcal{L}_{AN} = \sum_{(h,r,t) \in \Delta} \mathbf{I}_{[w_h \in \mathcal{V} \wedge w_t \in \mathcal{V}]} \cdot \mathcal{L}_f(w_h,r,w_t) 
+
\mathbf{I}_{[w_h \in \mathcal{V}]} \cdot \mathcal{L}_f(w_h,r,t)
+
\mathbf{I}_{[w_t \in \mathcal{V}]} \cdot \mathcal{L}_f(h,r,w_t)
$$

Where $\mathcal{L}_A$ could be $\mathcal{L}_{AA}$ or $\mathcal{L}_{AN}$

## Implementation Details
- Datasets
    - KG: Freebase (Graph is divided into:
        - main facts (all)
        - $e-e$ no OOKB
        - $w-e$ head is OOKB
        - $e-w$ tail is OOKB
        - $w-w$ head and tail is OOKB
    - Text: Wikipedia English (Preprocessor: Apache OpenNLP)
        - Sentence Segmentation
        - POS tagging
        - NER tagging (Location/Person/Organization)
    - Alignment: Wikipedia Anchors

**Parameters** (**bold** are optimal)

|  | Meaning | TransE | Skip-gram | pTransE |
| - | - | - | - | - |
| $e$ | epochs | 300 | 6 | 40 |
| $k$ | emb. dim. | {50, **100**, 150} | {50, 100, **150**} | {50, **100**, 150} |
| $\alpha$ | lr | {0.005, **0.01**, 0.05} | **0.025** (linearly decreasing) | {0.01, **0.025**} |
| $c$ | neg/pos | - | {5, **10**} | {5, **10**} |
| $s$ | skip-range | - | {**5**,10} | - | - |

## Experiments
The experiments were done on three tasks:
1. Triplet Classification
2. Improving Relation Extraction
3. Analogical Reasoning Task

### Triplet Classification
- Binary classification if a fact $(h,r,t)$ is true
- Experiment method is same as NTN (Socher et al., 2013)
- vs. TransE (Bordes et al., 2013)

#### Results
![](https://i.imgur.com/qLPxiXE.png =430x)
- pTransE performs better than TransE over non-OOKB triplets
- Because TransE scores are not "normalized", popular relations tend to have higher scores than rare relations
- Assigning a score in a more uniform scale is more advantageous when there is a threshold for separating true and false triplets

![](https://i.imgur.com/u2ywSjl.png)
- Jointly models outperforms the "respectively" (non-jointly) model in all held-out triplet groups
- Alignment by names performs better than alignment by anchors due to the smaller amount of anchors available

### Improving Relation Extraction
- Used Mintz (et al., 2009) and Sm2r (Weston et al., 2013) as extractors (Jointly is used as a feature)
- Dataset: NYT+FB (Riedel et al., 2010)
- vs. TransE (Bordes et al., 2013)

#### Results
![](https://i.imgur.com/Sn8DAJq.png)
- PR curves of both tasks show that the jointly model and knowledge model performs on par over $e-e$.
- Jointly model outperforms knowledge model in OOKB instances

### Analogical Reasoning Task
- The task consists of analogies such as “Germany” : “Berlin” :: “France” : ?,which are solved by finding a vector x such that vec(x) is closest to vec(“Berlin”) - vec(“Germany”) + vec(“France”) according to the cosine distance
- Anologies include:
    - Word analogies
    - Phrase analogies
    - Constructed analogies (from KB)
- vs. Skip-gram (Mikolov et al., 2013b)

#### Results
![](https://i.imgur.com/qZyWEDb.png)
- Using entity names for alignment hurts the performance of analogies of words and phrases because
    - a named graph forces the word embeddings to satisfy both popular and rare facts
    - lack of versatility and resilience in terms of the entity mentioned
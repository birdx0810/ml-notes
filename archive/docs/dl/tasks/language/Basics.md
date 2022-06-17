# NLP Basics
## Introduction

Word Vectors <=> Word Embeddings <=> Word Representations
- a set of language modeling techniques for mapping words to a vector of numbers (turns a text into numbers)
- a numeric vector represents a word
- comparitively sparse: more words == higher dimension

Key properties for Embeddings:
- Dimensionality Reduction: a more efficient representation
- Contextual Similarity: a more expressive representation
  - Syntax(syntactic): Grammatical structure
  - Semantics(Sentiment): Meaning of vocabulary

<!-- TODO: Read
[Neural Language Modeling](https://ofir.io/Neural-Language-Modeling-From-Scratch/)
[Embed, Encode, Attend, Predict](https://explosion.ai/blog/deep-learning-formula-nlp)
[What can we cram into a vector](https://arxiv.org/abs/1805.01070)
 -->

## Word Embeddings

### Word2Vec:
> Mikolov et. al (2013)
> Affiliates: Google
> Paper:
> - [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781): CBOW & SkipGram
> - [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546): Hierarchical Softmax & Negative Sampling
> Code:
> - [GitHub](https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/word2vec/word2vec_basic.py) (Tensorflow)
> - [Original Project](https://code.google.com/archive/p/word2vec/)

Input: corpus
Framework: 2-layer (shallow) NN
Features = the number of neurons in the projection(hidden) layer
<!-- Other implementations: [CBOW](https://hackmd.io/bdlAIXpKS7-J1FZot2DFyQ?both#CBOW), [Skip-gram](https://hackmd.io/bdlAIXpKS7-J1FZot2DFyQ?both#Skip-gram) -->

[Word Embedding Visual Inspector](https://ronxin.github.io/wevi/)
[Word2Vec](https://p.migdal.pl/2017/01/06/king-man-woman-queen-why.html)

![](https://i.imgur.com/veeC83b.png)

#### CBOW
Goal: Predict center word based on context words

![](https://i.imgur.com/0n4bJpZ.png)

$Q = (N \times D) + (D \times log_2(V))$
$Q$ is the model complexity
$(N \times D)$ is the complexity of the hidden layer
$(D \times log_2(V))$ is the output layer

Better syntactic performance

#### Skip-gram
Goal: Predict context words based on center word

![](https://i.imgur.com/9rbBGmp.png)

The input would be a one-hot vector of the center word. Since it is predicting the possibility of the next word, the output vector would be a probability distribution (Problem: High dimension output)
![](https://i.imgur.com/FWCkspb.png)

$Q = C \times D + (D \times log_2(V))$
$Q$ is the model complexity
$C$ is the size of the context window
$D$ is the complexity of the hidden layer
$(D \times log_2(V))$ is the output layer

Better overall performance

#### Hierarchical Softmax
- A Huffman binary tree, where the root node is the hidden layer activations (or context vector $C$ ), and the leaves are the probabilities of each word.
- The goal of this method in word2vec is to reduce the amount of updates required for each learning step in the word vector learning algorithm.
- Rather than update all words in the vocabulary during training, only $log(W)$ words are updated (where $W$ is the size of the vocabulary).
$$
\displaystyle p(w|w_I) = \prod_{j=1}^{L(w)-1}\sigma\bigg(\big[n(w,j+1) = ch(n(w,j))\big] \cdot v'_{n(w,j)} v_{w_I}^{\intercal} \bigg)
$$

![](https://i.imgur.com/JlTkpC1.png)

#### Negative Sampling
- Solves time complexity of Skip-gram.
- Sample part of vocabulary for negative values, rather than whole vocab. (for $k$ negative examples, we have 1 positive example)
- Determine is this word from context? or sampled randomly
- E.g. "the" "bird" might have high co-occurence values, but they might not mean anything useful.

$$
log\ \sigma(v'_{w_O} \top v_{w_I}) + \sum^{k}_{i=1} \mathbb{E}_{w_i \sim P_n(w)} \big[log\ \sigma (-v'_{w_i} \top v_{w_I} \big]
$$

$$
\text{Softmax} = p(t|c) = \frac{e^{\theta^{T}_{t} e_c}}{\sum^{10000}_{j=1} e^{\theta^{T}_{t} e_c}}
$$

Where $\theta_t$ is the target word, and $e_c$ is context word. If the target word is true has probabillity of 1. By reducing the context words from 10000 to $k$ we could reduce the model complexity and runtime.

$$
\begin{aligned}
P(w_i) &= \bigg(\sqrt{\frac{z(w_i)}{0.001}}+1\bigg)⋅\frac{0.001}{z(wi)} \\
P(w_i) &= 1 - \sqrt{\frac{t}{f(w_i)}} \\
\end{aligned}
$$

How to sample negative examples?
- according to empirical frequency
- $\frac{1}{|Vocabulary|}$
- $\frac {f(w_i)^{3/4}}{\sum^{10000}_{j=1}f(w_j)^{3/4}}$

### GloVe: Global Vectors for Word Representations
> Jeffrey Pennington, Richard Socher, Christopher Manning (2014) #ACL
> Affiliates: Stanford University
> Paper: [Link](https://nlp.stanford.edu/pubs/glove.pdf)
> Code: [Link](https://github.com/stanfordnlp/GloVe)
> Official Site: [Link](https://nlp.stanford.edu/projects/glove/)

: Capture global statistics directly through model

2 main models for learning word vectors:
- latent semantic analysis (Good use of statistics, bad anology)
    - e.g. TF-IDF, HAL, COALS
- local context window (Good anology, bad use of statistics)
    - e.g. CBOW, vLBL, PPMI

$X$ is a word-word co-occurence matrix

$X_{ij}$ is the number of times word $j$ occurs in the context of word $i$

$X_i = \sum_k X_{ik}$ is the number of times any word appears in the context of word $i$

$P_{ij} = P(i|j) = \frac{X_{ij}}{X_i}$ be the probability that the word $j$ appear in the context of word $i$

Error Function
$$
J = \sum^{V}_{i,j=1} f(X_{ij})(w^T_i \tilde{w}_j + b_i + \tilde{b_j} - logX_{ij})^2
$$

![](https://i.imgur.com/l2PN1lJ.png)

Although $i$ and $j$ is highly related (e.g. ice, steam), they might not frequently appear together $P_{ij}$. But, through observing neighbouring context words $k$, we could identify the similarity between them through $P_{ik}$ and $P_{ij}$. If $i$ and $j$ is similar, when $P_{ik}$ is small $P_{jk}$ would also be small, and vice versa. Thus, $\frac{P_{ik}}{P_{jk}} \approx 1$.

### fastText: Enriching Word Vectors with Subword Information
> Piotr Bojanowski, Edouard Grave, Armand Joulin, Tomas Mikolov (2016) #TACL
> Affiliation: facebook
> Paper: [Link](https://arxiv.org/abs/1607.04606) & [A bag of tricks](https://arxiv.org/abs/1607.01759)
> Code: [Link](https://github.com/facebookresearch/fastText)

### ELMo: Embedding from Language Models
> Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer (2018) #NAACL
> Affiliation: AllenNLP
> Paper: [Link](https://arxiv.org/abs/1802.05365)
> Code: [Link](https://github.com/allenai/allennlp)
> Official Site: [Link](https://allennlp.org/elmo)

Looks at entire sentence before assigning each word in it an embedding
Bi-directional LSTM

### Google's Universal Sentence Encoder
> Paper: [USE](https://arxiv.org/abs/1803.11175) [MultiUSE](https://arxiv.org/abs/1907.04307)
> Code: [Link](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/semantic_similarity_with_tf_hub_universal_encoder.ipynb)
> Official Site: [Link](https://tfhub.dev/google/universal-sentence-encoder/4)

### Skip-Thought
> Paper: [Link](https://arxiv.org/abs/1506.06726)

---

## Language Models

### Class-Based $n$-gram Models of Natural Language
> n-gram Language Model
> Statistical Language Model
> Brown et al. (1992)
> Affiliates: IBM
> Paper: [Link](https://www.aclweb.org/anthology/J92-4003.pdf)

Language Models have been long used in:
- speech recognition (Bahl et al., 1983)
- machine translation (Brown et al, 1990)
- spelling correction (Mays et al, 1990)

Language models assign probabilities to sequence of words.
$P(w|h)$
- Given context $h$, calculate the probability of the next word being $w$
- An $n$-gram model is a model of the probability distribution of $n$-word (or character) sequences

We assume that production of English text can be characterized by a set of conditional probabilities:
$$
\begin{aligned}
P(w_1^k) &= P(w_1)P(w_2|w_1)...P(w_k|w_{1}^{k-1}) \\
&\approx \prod_{i=1}^k P(w_i|w_{i-1})
\end{aligned}
$$
Where...
- $P(w_k|w_{1}^{k-1})$ is the conditional probability of predicted word $w_k$ given history $w_{1}^{k-1}$
- $w_{1}^{k-1}$ represents the string $w_1w_2...w_{k-1}$

A **trigram** model could be defined as:
$$
P(w_{1:n}) = \prod_{i=1}^k P(w_i|w_{i-2}^{i-1})
$$
Where...
- $w_{k-2}^{k-1}$ is the history taken into context (i.e. the two words before $w_k$)
- In practice, it's more common to use trigram models

Parameter estimation: sequential maximum likelihood estimation
$$
P(w_n|w_1^{n-1}) \approx \frac{C(w_1^{n-1}w_n)}{\sum_w C(w_1^{n-1}w)}
$$
Where...
- $C(w)$ is the occurrence of string $w$ in $t_1^T$
- Maximise $P(t_n^T|t_1^{n-1})$
- Could be thought of as the transition matrix of a Markov model from state $n-1$ to state $n$
- As $n$ increases, the model **accuracy** *increases*, but **reliability** of parameter estimate *decreases*


Besides predicting the probability of next word, this paper suggests that we could also predict word classes (syntactic similarity)
$$
\begin{aligned}
P(w|c) &= \frac{C(w)}{C(c)} \\
P(c) &= \frac{C(c)}{V}
\end{aligned}
$$
This could be "merged" with our 3-gram model into
$$
c = \text{argmax}_c P(c) \prod_{i=1}^k P(w_i|w_{i-2}^{i-1}, c)
$$

Reference:
- [Artificial Intelligence: A Modern Approach](http://aima.cs.berkeley.edu/)
- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/)

### A Neural Probabilistic Language Model
> Bengio et. al (2003) #JMLR
> Affiliates: University of Montreal
> Paper: [Link](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

Curse of Dimensionality: a word or sequence on which the model will be tested is likely to be different from all the word sequences seen during training

- Associate with each word in the vocabulary a distributed word feature vector
- Express the joint probability function of word sequences in terms of the feature word vectors of these words in the sequence
- Learn simultaneously the word feature vectors and the parameters of the probability function

$$
\hat{P}(W^T_1) = \prod_{t=1}^{T} \hat{P}(w_t|w_{1}^{t-1})
$$
Where...
- $w_t$ is the $t$-th word
- $w_i^j$ is the sequence $(w_i, w_{i+1},..., w_{j-1}, w_j)


Objective Function: Maximize Log-Likelihood

### RNN Based Language Model
> Mikolov et. al (2013) #ACL
> Affiliates: Microsoft
> Paper:
> [ACL](https://www.aclweb.org/anthology/N13-1090/)
> [INTERSPEECH](https://www.isca-speech.org/archive/interspeech_2010/i10_1045.html)

![](https://i.imgur.com/8LBdgID.png)

An RNN consists of:
- Input Layer
- Hidden Layer (Recurrent cell)
- Output Layer (Probability distribution)


$U$ is a word matrix where each column represents a word
$s(t-1)$ is the history input, passed from the last $s(t)$
$y(t)$ is a probability distribution over all words in vocab

#### Input Layer
$$
\begin{aligned}
w(t) &= v \times 1 \\
U &= d \times v \\
\end{aligned}
$$

Where
- $w(t)$ is the one-hot vector representation of word at time $t$
- $U$ is the learned weighted matrix
- $v$ is the vocabulary size and $d$ is the embedding dimension size

$$
\begin{aligned}
s(t-1) &= d \times 1 \\
W &= d \times d \\
\end{aligned}
$$

Where
- $s(t-1)$ is the representation of the previous hidden state
- $W$ is the learned weighted matrix

#### Hidden Layer

$$
\begin{aligned}
s(t) &= f(Uw(t) + Ws(t-1)) \\
s(t) &= d \times d \\
\end{aligned}
$$

Where
- $s(t)$ is the representation of the sentence history
- $f( \cdot )$ is the activation function

#### Output Layer
$$
\begin{aligned}
y(t) &= g(V s(t)) \\
V &= v \times d \\
y(t) &= v \times 1 \\
\end{aligned}
$$

Where
- $y(t)$ is the probability distribution of each word within vocab
- $g( \cdot )$ is the activation function

### ULMFiT: Universal Language Model Fine-tuning for Text Classification
> Affiliates: Fast.ai
> Paper: [Link](https://arxiv.org/abs/1801.06146)
> Code: [Link](https://github.com/fastai/fastai/blob/master/examples/ULMFit.ipynb)
> Official Site: [Link](http://nlp.fast.ai/ulmfit)

### UNILM: Unified Language Model Pre-training for Natural Language Understanding and Generation
> Paper: [Link](https://arxiv.org/abs/1905.03197)
> Code: [Link](https://github.com/microsoft/unilm)

---

## Seq2Seq

### NMT & Seq2Seq Models: A tutorial...
> Paper: [Link](https://arxiv.org/abs/1703.01619)
> Neural Machine Translation Code:
> - [Tensorflow](https://github.com/tensorflow/nmt)
> - [Google](https://google.github.io/seq2seq/nmt/)
> - [OpenNMT](http://opennmt.net/)

### On the properties of Neural Machine Translation: Encoder-Decoder Approaches
> Cho et al. (2014)
> Affiliation: University of Montreal
> Paper: [Link](https://arxiv.org/pdf/1409.1259.pdf)

### Sequence to Sequence Learning with Neural Networks
> Affiliation: Google
> Paper: [Link](https://arxiv.org/abs/1409.3215)
> Code: [Link](https://github.com/google/seq2seq)

Source Language $\to$ **encode** $\to$ compressed state (vector) $\to$ **decode** $\to$ Target Language
$V_{src} \text{: \{I love apple\} } \to V_{tgt} \text{: \{我喜歡蘋果\} }$

### Neural Machine Translation by Jointly Learning to Align and Translate
> A.k.a RNNencdec & RNNsearch
> Paper: [Link](https://arxiv.org/abs/1409.0473)

### Google's Neural Machine Translation System
> Affiliation: Google
> Paper:
> [Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144.pdf)
> [Enabling Zero-Shot Translation](https://arxiv.org/abs/1611.04558)

- Human translation: Translating part by part (memorization)
- Attention weight (α parameters) in each hidden state. How much information should we pay attention to in each word in vocab.
- Mapping a query and a set of key-value pairs to an output.
- Reduced sequence computation with parallelization.

![](https://i.imgur.com/KmzSHjm.png)

The Problem with Machine Translation:
1. Do we need to translate $\to V_t +1$
2. Which word to translate (src) $\to$ Classifier
3. What should it be translated to (tgt) $\to$ Attention

$$
a^{<t'>} = \Big( \overrightarrow{a}^{<t'>}, \overleftarrow{a}^{<t'>} \Big) \\
\sum_{t'} \alpha^{<1, t'>} = 1 \\
C^{<1>} = \sum_{t'} \alpha^{<1, t'>} a^{<t'>} \\
$$

$C$ is the context weighted attention sum.
$\alpha^{<t,t'>}$ is the amount of attention $y^{<t>}$ should pay to $a^{<t'>}$.

A larger the scalar (dot) product ($\approx 1$) means higher similarity. Thus, leads to "more attention".

Problem: Slow and still limited by size of context vector of RNN; could we remove the continuous RNN states?


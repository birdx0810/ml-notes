# Get to the Point: Summarization with Pointer-Generator Networks
> ACL 2017
> Abigail See, Peter J. Liu, Christopher D. Manning
> Affiliation: Stanford University
> Paper: [Link](https://nlp.stanford.edu/pubs/see2017get.pdf)
> Code: [GitHub](https://github.com/abisee/pointer-generator)

## Introduction

Two broad approaches to summarization:
- Extractive: Summaries are generated from whole sentences of source document
    - Easier, but not flexible
- Abstractive: Summaries are generated with novel words and phrases that might not be in the source document
    - Higher-quality, but more difficult

Recent seq2seq models have shown promising results in text generation field. Yet they still face a few problems such as:
- inaccurately reproducing factual details
- out-of-vocabulary problem
- repetition of words

Pointer-generator network could be thought of as a balance between extractive/abstractive models, as it **"points back"** to the source text which allows copying, while retaining the architecture of a RNN decoder, which provides the basic ability of generating words.

## TL;DR

Inspired by:
- [Incorporating Copying Mechanism in Sequence-to-Sequence Learning](https://arxiv.org/abs/1603.06393)

Based on:
- [Abstractive Text Summarization Using Sequence-to-Sequence RNNs and Beyond](https://arxiv.org/abs/1602.06023) (Nallapati et al., 2016)
- [Pointer Networks](https://arxiv.org/abs/1506.03134) (Vinyals et al., 2015)
- [Modeling Coverage for Neural Machine Translation](https://arxiv.org/abs/1601.04811) (Tu et al., 2016)

## Model Architecture

### Baseline
![](https://i.imgur.com/8uon7JA.png)

The model (and baseline) is based on an attentional encoder-decoder RNN as proposed by Nallapati et al. ([2016](https://arxiv.org/abs/1602.06023)).
- Encoder: Single Bi-LSTM + Linear Projection
- Decoder: Single Uni-LSTM + 2 Linear Layer
    - Training: Previous word is from referenced summary
    - Testing: Previous word emitted from decoder
- Attention: Distribution is calculated as of Bahdanau et al. (2015)
    - Could be thought of as a probability distribution over the source words

$$
\begin{aligned}
e^t_i &= v^\intercal \tanh (W_hh_i + W_ss_t + b_{attn}) \\
a^t &= \text{softmax}(e^t) \\
\end{aligned}
$$

- Loss Function: Averaged Negative Log Likelihood over timestamp $T$

$$
\begin{aligned}
h^*_t &= \sum_i a^t_i h_i \\
P_\text{vocab} &= \text{softmax}(V'(V[s_t,h^*_t]))
\end{aligned}
$$

Where,
- $h_i$ is a sequence of encoder hidden states
- $s_t$ is the decoder state at time $t$
- $h^*_t$ is the context vector

### Pointer-generator Network
> Resolving OOV issue

![](https://i.imgur.com/FxMEVcf.png)

For each decoder timestep, a generation probability $p_\text{gen} \in [0,1]$ is calculates the probability of generating words from vocabulary or copying words from source text, extending the vocabulary to include words from source text.

$$
p_\text{gen} = \sigma(
    w^\intercal_{h^*}h^* +
    w^\intercal_{s_t}s_t +
    w^\intercal_{x}x_t +
    b_\text{ptr}
)
$$

Hence, the probability distribution of the extended vocabulary is:

$$
\Pr(w) = p_\text{gen} \Pr_\text{vocab}(w) + (1-p_\text{gen}) \sum_{i:w_i=w} a^t_i
$$

Where,
- $\displaystyle \Pr_\text{vocab}(w) = 0$ if $w$ is an OOV word
- $\displaystyle \sum_{i:w_i=w} a^t_i = 0$ if $w$ does not appear in the source text

The loss function is the average negative log likelihood as described in the baseline model but with respect to the modified probability distribution $\Pr(w)$ of the extended vocabulary.

### Coverage Mechanism
> Resolving repetition

The coverage vector $c^t$ represents the degree of coverage the words from the source target have been attended so far. Where $c^0$ is a zero vector as it has not perform any attention, and the attention weights are not normalized (logits). The coverage vector would be added to the attention mechanism, changing it to equation below:

$$
c^t = \sum_{t'=0}^{t-1} a^{t'} \\
e^t_i = v^\intercal\tanh (W_hh_i + W_ss_t + W_cc^t_i + b_\text{attn})
$$

If a word is attended repeatedly, it will be penalized with a coverage loss. The loss is comparably flexible as it does not require one-to-one uniform coverage. The loss is finally reweighted by a $\gamma$ and added to the primary loss function.

$$
\text{covloss}_t = \sum_i\min(a^t_i, c^t_i) \\
\text{loss}_t = -\log\Pr(w^*_t) + \gamma\sum_i \min(a^t_i,c^t_i)
$$

## Experiments

This model was evaluated on the CNN/DM dataset

### Hyperparameters
| Hyperparameter | Details |
| - | - |
| # Parameters              | 21,501,265 |
| Sequence Length (src/tgt) | 400/100,120 (train,test) |
| Hidden Dimension          | 256 |
| Embedding Dimension       | 128 |
| Vocab Size (src/tgt)      | 500/500 |
| Optimizer                 | Adagrad |
| $\lambda$                 | 0.15 + 0.1 |
| Grad Clip Norm            | 2 |
| $\gamma$                  | 1 |
| Iterations/Epochs         | 600,000/33


![](https://i.imgur.com/3osFnFX.png)





# Mem2Seq: Effectively Incorporating Knowledge Bases into End-to-End Task-Oriented Dialog Systems

> Andrea Madotto, Chien-Sheng Wu, Pascale Fung (ACL 2018)
> Affiliation: HKUST
> Paper: [Link](https://arxiv.org/abs/1804.08217)
> Code: [GitHub](https://github.com/HLTCHKUST/Mem2Seq)

## Introduction

Traditional task-oriented dialog systems were built with several pipelined modules:
- language understanding
- dialog management
- knowledge query
- language generation

Current approaches uses Seq2seq models with the inclusion of attention-based copy mechanisms which allows the model to produce relevant results even with OOV words in the dialog history.

MemNNs are recurrent attention models over a large external memory. Yet it only chooses its responses from apredefined candidate

Mem2seq augments the MemNN framework

***Contributions:***
1. Multihop attention + pointer network integration 
2. Learns to generate dynamic queries to control memory access and attend queries
3. Could be trained faster and achieves SOTA results

## Model Architecture
![](https://i.imgur.com/oB9OIor.png)

Consists of:
- **memory encoder** 
    - creates vector representations of the dialog history
- **memory decoder**.
    - reads and copies the memory to generate a response
Where
- $X = \{x_1, ..., x_n,$\$$\}$ are all the words in the dialog history
    - \$ is a special sentinel character, an OOM word representation
- $B = \{b_1, ..., b_l \}$ is the set of tuples in the KB
- $C = \{C^1, ..., C^{k+1}\}$ is a set of trainable embedding matrices, mapping tokens to vectors
- $Y = \{y_1, ..., y_m \}$ are the set of words in the expected response
- $\text{PTR} = \{ptr_1, ..., ptr_m\}$ is the pointer index set
    - $ptr = \begin{cases} max(z) & \text{if } \exists z \text{ s.t. } y_i = u_z \\ n+l+1 & \text{otherwise}\end{cases}$
    - $u_z \in U$ is the input sequence
    - $U = [B;X]$
    - $n+l+1$ is the sentinel position index

### Memory Encoder
- Adjacent weight tying MemNN
- Input is word-level information in $U$

$$
p_i^k = \text{Softmax}((q^k)^\intercal C_i^k)
$$

Where
- $q^k$ is used as the reading head
- $C_k^i = C^k(x_i)$ is the memory content at position $i$
- $K$ is the total hops, and $k$ represents the current hop

$$
o^k = \sum_i p_i^k C_i^{k+1} 
$$

Where
- $p^k$ is the soft memory selector that selects the relevant memory
- $q^{k+1} = q^k + o^k$ is the query vector for the next hop
- $o^K$ is the output of the encoder and input of the decoder

### Memory Decoder
- Uses a GRU as a dynamic query generator for the MemNN
    - First hop focuses more on retrieving memory information
    - Final hop chooses the exact token leveraging the pointer supervision
- Inputs the previous generated word and previous query

$$
h_t = \text{GRU}(C^1(\hat{y}_{t-1}), h_{t-1})
$$

- At each time/hop, two distributions are generated
    - $P_\text{vocab}$ distribution over all words in vocabulary
    - $P_\text{ptr}$ distribution over the memory contents, generated at the last hop of the decoder using the attention weights

$$
\begin{aligned}
P_{\text{vocab}}(\hat{h}_t) &= \text{Softmax}(W_1[h_t;o^1]) \\
P_\text{ptr} &= p^K_t
\end{aligned}
$$

- The decoder generates words by pointing to the input words of the memory
- This approach is similar to the attention in pointer networks
- The objective is to minimize the sum of two standard cross-entropy losses of the two distributions

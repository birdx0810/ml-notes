# Episodic Memory Reader: Learning What to Remember for Question Answering from Streaming Data
###### tags: `MemNN`

> Moonsu Han, Minki Kang, Hyunwoo Jung, Sung Ju Hwang (ACL 2019)
> Affiliation: KAIST
> Paper: [Link](https://arxiv.org/abs/1903.06164)
> Code: [GitHub](https://github.com/h19920918/emr)

###### slide: https://hackmd.io/@birdx0810/rkBcuogwH?type=slide#/

---

## Background

- Document-level context modeling is difficult to acheive due to scalability
- Use reinforment learning to:
    - Learn the relationship between sentences
    - Memory management
- Goal: Learn from streaming data ***without knowing*** when the question would be given
- Note: Streaming data could be Document or Video

----

## TL;DR 
### [DNTM](https://arxiv.org/pdf/1607.00036.pdf) + [LEMN](https://arxiv.org/pdf/1812.04227.pdf)
### Link to [GitHub](https://github.com/h19920918/emr)

---

## Dataset

- bAbI 
- TriviaQA
- ~~TVQA~~

---

## Model Overview

Input: Data Stream 
- $X = \{x^{(1)},...,x^{(T)}\}$

Learning Function: 
- $\mathcal{F}: X \to M$
- $X$ maps itself to the set of memory entries $M$

----

## Model Overview

1. an agent $\mathcal{A}$ based on EMR
2. an external memory $M = [m_1, ..., m_N]$
3. a solver which solves the given task

![](https://i.imgur.com/qipMtyH.png)

----

## Episodic Memory Reader
###### The agent $\mathcal{A}$ based on EMR
1. Data Encoder: **Transform Data into a Memory Vector**
2. Memory Encoder: **Generates replacement probability for memory**
3. Value Network: **Estimates Memory Value**

----

## Data Encoder

**Transform Data into a Memory Vector**
$$
\begin{align*}
&e^{(t)} = \psi(x^{(t)}) \\
\\
\text{Where } &\psi(\cdot) \text{ is the data encoder}, \\
&x^{(t)} \text{ is the data at time } t \\
\end{align*}
$$

<!--
$$
\begin{align*}
&x^{(t)} \text{ could be any data format} \\
&\psi(\cdot) \text{ encodes the data into a k-dimensional vector}
\end{align*}
$$
-->

----

## Memory Encoder

**Input $e^{(t)}$, $M^{(t)}$ and select relevant memory.**
$$
\begin{align*}
\text{When } &t > N \text{ (size of memory)}, \\
& \text{delete memory entry } m^{(t)}_i \\
& \text{append } e^{(t)} \to M \text{, becoming } m^{(t+1)}_N 
\end{align*}
$$

----

## Policy & Value Network

- Actor-Critic RL (A3C)
    - Actor: controls how the agent behaves by learning the **optimal policy** (policy-based)
    - Critic: evaluates the action of the actor computing the **value function** (value based)

![](https://i.imgur.com/KGNVHL3.png)

<!-- ![](https://i.imgur.com/HPCMXAx.png) -->
<!-- Learns a optimal *policy algorithm* to determine what action to take --> 
<!-- Approximate optimal *value function* to evaluate the action of critic -->

----

## Policy Network
###### Determines which memory to replace

- Outputs probability for each memory
- Replace least important encoded memory (sentence)
$$
\pi(i|[M^{(t)},e^{(t)}]) 
$$

- Training: Agent chooses answer and QA model rewards agent
- Testing: QA model only solves the task using instances in memory

----

## Value Network
###### Uses a MLP to estimate Value of Memory

$$
\begin{align*}
&\mathcal{M} = \rho(\sum_{i=1}^N h_i^{(t)}) \\
\text{Where, } \\
&\mathcal{M} \text{ is a holistic representation of the memory} \\
&\rho = max(0,xW_1+b_1)W_2+b_2 \\
&\mathcal{M} \to \text{GRU} \to \text{MLP (Estimate value } V^{(t)})
\end{align*}
$$

---

## Experiment

- Baselines:
    - FIFO
    - Uniform (Random)
    - LIFO
    - EMR-Independent (LRU-DNTM)
- Proposed:
    - EMR-biGRU (R_EMR)
    - EMR-Transformer (T_EMR)

---

## Q&A
Streaming Input & 讀一篇文章有什麼差別？
> 沒什麼差別... 所以才可以拿 TriviaQA 來跑... 作者也有提到希望這個方法可以用來解決 document based QA, without knowing the questions beforehand.

---

## End-to-End Memory Network

![](https://i.imgur.com/SVimnYh.png)
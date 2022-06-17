# Meta-Learning with Memory-Augmented Neural Networks
###### tags: `MemNN`

> Paper: [Link](http://proceedings.mlr.press/v48/santoro16.pdf)

![](https://i.imgur.com/cwJrfBB.png)


### Reading
Read attention is **based on content similarity**.
$$
\begin{aligned}
\mathbf{r}_i &= \sum_{i=1}^{N}w_t^r(i)\mathbf{M}_t(i) \\
w_t^r(i) &= \text{softmax}(\frac{k_t \cdot \mathbf{M}_t(i)}{||k_t|| \cdot ||\mathbf{M}_t(i)||})
\end{aligned}
$$
Where,
- $\mathbf{M}_t(i)$ is the $i$-th row of the memory matrix.
- $\mathbf{k}_t$ is the feature vector i.r.t. input $\mathbf{x}$
- $\mathbf{r}_t$ is the read vector, sum of memory records weighted by
    - $w_t^r$ is the read weighting vector (the cosine similarity of input in reference to the memory)

### Writing
Least Recently Used Access (LRUA)
: write head prefers to write new content to either the *least used* memory location or the *most recently used* memory location

$$
\mathbf{M}_t(i) = \mathbf{M}_{t-1}(i) + w_t^w(i) \mathbf{k}_t, \forall i
$$

1. The usage weight $\mathbf{w}^u_t$ is the sum of current read $\mathbf{w}^r_t$ and write $\mathbf{w}^w_t$ vectors and decayed last usage weight $\gamma \mathbf{w}^u_{t-1}$ where $\gamma$ is the decay factor.
    - $\mathbf{w}^u_t = \gamma \mathbf{w}^u_{t-1} + \mathbf{w}^r_t + \mathbf{w}^w_t$
2. $\mathbf{w}^r_t$ is the weighted cosine similarity above.
    - $\mathbf{w}^r_t = \text{softmax}(\cos(k_t,M_t(i)))$
3. $\mathbf{w}^w_t$ is interpolated between previous read weight ("last used location") and previous least-used weight ("rarely used location").
    - $\mathbf{w}^w_t = \sigma(\alpha)\mathbf{w}^r_{t-1} + (1 - \sigma(\alpha)) \mathbf{w}^{lu}_{t-1}$
4. The least-used weight $\mathbf{w}^{lu}$ is scaled according to usage weights $\mathbf{w}^u_t$. 
    - $w^{lu}_t = \begin{cases} 1 \text{, if } w^u_t(i) \leq m(\mathbf{w^u_t, n}) \\ 0  \text{, otherwise}\end{cases}$
    - Where $m(\mathbf{w}^u_t, n)$ is the $n$-th smallest element in vector $\mathbf{w}^u_t$

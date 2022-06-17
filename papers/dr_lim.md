# Dimensionality Reduction by Learning an Invariant Mapping
> Raja Hadsell, Sumit Chopra, Yann LeCun (CVPR 2006)
> Affiliates: NYU
> Paper: [Link](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)

## Introduction 

Dimensionality reduction
: maps similar vectors to nearby points and dissimilar vectors to distant points from high to low dimensional space

Most dimensionality reduction tecniques have two shortcomings:
1. Function could not be applied to new points
2. Presuppose that there is a meaningful distance metric in input space
3. Cluster points in output space which degenerate solutions

The proposed method DrLIM learns a globally coherent non-linear function.
1. Only needs neighborhood relationships between training samples independent of any distance metric
2. May learn functions that are invariant to complicated non-linear transformations
3. Could be used to map new samples not seen during training
4. The mapping generated is smooth and coherent in the output space

## Methodology

For each input sample $\vec{X}_i$
1. Find the set of samples $S_{\vec{X}_i} = \{\vec{X}_j\}_{j=1}^p$ such that $\vec{X}_j$ is deemed as similar to $\vec{X}_i$
2. Pair the sample $\vec{X}_i$ with other samples such that:

$$
Y_{ij} = \begin{cases}
    0 \quad \text{if } \vec{X}_j \in S_{\vec{X}_i} \\
    1 \quad \text{otherwise}
\end{cases}
$$

For each pair $(\vec{X}_i, \vec{X}_j)$, repeat until converge
1. If $Y_{ij} = 0$, update $W$ to decrease $D_W$
2. If $Y_{ij} = 1$, update $W$ to increase $D_W$

Where $D_W$ is the cosine distance of vectors $X_i$ and $X_j$

$$
D_W = || G_W (\vec{X}_i) - G_W (\vec{X}_j) ||_2
$$

The Contrastive Loss Function

$L(W, Y, \vec{X}_i, \vec{X}_j) = (1-Y) \frac{1}{2}(D_W)^2 + (Y) \frac{1}{2} \{\max(0, m-D_W)\}^2$

Where,
- $W$ is the model parameters (weights)
- $Y$ is the label (similar = 1)
- $\vec{X}$ are the input vectors
- $m$ is a margin that defines the minimal distance for dissimilar pairs

## Model Architecture

Siamese Network
: Consists of two copies of the function $G_W$ that shares the same parameters $W$ (encoder), and a cost function (similarity calculator).


## Experiments

The proposed method is evaluated on the MNIST and NORB datasets.

![](https://i.imgur.com/JKJLR54.png)


![](https://i.imgur.com/GlkIi3A.png)


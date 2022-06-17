# TimeGAN - Time-series Generative Adversarial Networks

> Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar
> NIPS 2019
> Paper: [Link](https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks)
> Code: [GitHub](https://github.com/jsyoon0823/TimeGAN)

TimeGAN
: is a generative time-series model, trained adversarially and jointly via a learned embedding space with both supervised and unsupervised loss that consists of 4 components:
1. embedding function
2. recovery function
3. sequence generator
4. sequence discriminator

Prior methods include: 
- [C-RNN-GAN](https://arxiv.org/abs/1611.09904)
- [RC-GAN](https://arxiv.org/abs/1706.02633)

These approaches only rely on the binary adversarial feedback for learning

## Introduction
Data consists of two elements: static features and temporal features

Let $\mathbf{S} \in \mathcal{S}$ be a vector space of static features with random vectors that can be instantiated with specific values denoted as $s$ and $\mathbf{X} \in \mathcal{X}$ be a vector space of temporal features with random vectors that can be instantiated with specific values denoted as $t$.

The goal is to use training data $\mathcal{D}$ to learn a density $\hat{p}(X_t|S, X_{1:t-1})$ that best approximate $p(X_t|S, X_{1:t-1})$. Which could be broken down into two objectives

1. Global Sequence Objective (Jensen-Shannon Divergence)
    - Measures the similarity between two probability distributions
    - GAN objective
$$
\min_\hat{p} D \Big( p(S,X_{1:T})||\hat{p}(S, X_{1:T}) \Big)
$$
2. Local Stepwise Objective (Kullback-Leibler Divergence)
    - Measures the difference between two probability distributions
    - ML objective
$$
\min_\hat{p} D \Big( p(X_t|S,X_{1:t-1}) || \hat{p}(X_t|S,X_{1:t-1}) \Big)
$$

## Model Architecture
![](https://i.imgur.com/kXhIrEl.png)

1. Embedder: Map input features from feature space to latent space
    - Trained on reconstruction and supervised loss
```
H = embedder(X, T)
```
2. Recovery: Map from latent space to feature space
    - Trained on reconstruction and supervised loss
```
X_tilde = recovery(H, T)
```
3. Generator: Takes in random static and temporal vectors generated via Gaussian Distribution and Wiener Process as static and temporal features and generates synthetic data in latent space
    - Trained on unsupervised and supervised loss
```
E_hat = generator(Z, T)
H_hat = supervisor(E_hat, T)
H_hat_supervise = supervisor(H, T)
```
4. Synthetic data: Takes outputs of generator from latent space and decodes it into feature space
```
X_hat = recovery(H_hat, T)
```
5. Discriminator: Takes the latent space features and determines if static and temporal codes are real or synthetic
    - Trained on unsupervised and supervised loss
```
Y_fake = discriminator(H_hat, T)
Y_real = discriminator(H, T)
Y_fake_e = discriminator(E_hat, T)
```

## Learning to Optimize

![](https://i.imgur.com/e15j7z4.png)

The **reconstruction loss** is used to map the features space and latent space. This loss calculates the $L2$ distance between the generated static and temporal variable $(\tilde{\mathbf{s}}, \tilde{\mathbf{x}}_t)$, and the original variables $(\mathbf{s}, \mathbf{x}_t)$. (Embedding/Recovery Loss)

$$
\mathcal{L}_R = \mathbb{E}_{\text{s, x}_{1:T~P}}
[||s - \tilde{s}||_2 + \sum_t||x_t = \tilde{x}||]
$$

The **supervised loss** lets the model capture more information regarding the stepwise conditional distributions in the data by taking the $t$ step as the ground truth when given $t$. When feeding static and temporal features to both embedding network and generator network; they should project to the same latent space representation $(\mathcal{s}, \mathcal{x} \to \mathbf{h})$.

$$
\mathcal{L}_S = \mathbb{E}_{\text{s, x}_{1:T~P}}
[\sum_t ||h_t - g_\mathcal{X}(h_\mathcal{S}, h_{t-1}, z_t)||_2]
$$

The **unsupervised loss** is used to maximize (discriminator) and minimize (generator) the likelihood of providing the correct classification and synthetic output. (GAN Loss)

$$
\mathcal{L}_U = \mathbb{E}_{\text{s, x}_{1:T~P}}
[\log y_\mathcal{S} + \sum_t \log y_t] + 
\mathbb{E}_{\text{s, x}_{1:T~\hat{P}}}
[\log(1 - \hat{y}_\mathcal{S}) + \sum_t \log(1 - \hat{y}_t)]
$$

# Generative Adversarial Networks

## Loss Function

### Generative Adversarial Networks (Goodfellow et al., NIPS 2014)

Problems with the vanilla GAN:

1. The objective function is designed to optimize the similarity between real and fake sample distributions. Different parameters may yield same same results (non-convex optimization space), and we may not even reach an optimal solution for a fixed set of parameters.
2. Both Generator and Discriminator are updated concurrently, this does not guarantee convergence. In practice, training a discriminator is comparatively easier than the generator, leading to vanishing gradients.

### Wasserstein GAN (Arjovsky et al. 2017)

### Are GANs Created Equal (Lucic et al, 2017)

### Others

* Mutual Information Neural Estimator (Belghazi et al., 2018)

## Conditioning

### Conditional GAN ([Mirza and Osindero, 2014](https://arxiv.org/abs/1411.1784))



### InfoGAN ([Chen et al., 2016](https://arxiv.org/abs/1606.03657))



### Semi-supervised GAN ([Odena, 2016](https://arxiv.org/abs/1606.01583))



### Auxiliary Classifier GAN ([Odena et al., 2016](https://arxiv.org/abs/1610.09585))



## Architecture



### CycleGAN



### EBGAN



### VAEGAN





## Tricks

### Improve Techniques for Training GANs (Salimans et al., NIPS 2016)



### Improved Training of Wasserstein GANs (Gulrajani et al., NIPS 2017)

WGAN with Gradient Penalty

### Progressive Growing of GANs for Improved Quality, Stability, and Variation (Karras et al., ICLR 2018)

### Large Scale GAN Training for High Fidelity Natural Image Synthesis (Brock et al., ICLR 2019)












# Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs

> Stephanie Hyland, Cristobal Esteban, Gunnar Ratsch
> Paper: [arXiv](https://arxiv.org/abs/1706.02633)
> Code: [GitHub](https://github.com/ratschlab/RGAN)

Motivation
: exploit and develop the framework of generative adversarial networks to generate realistic synthetic medical data

Medical data is more noisy, complex, and prediction problems are less clearly defined. Our goal is to generate data that can be used to train models for tasks that are unknown at the moment the GAN is trained.

Contributions:
1. Demonstration of generating time series data using GANs
2. Novel approach in evaluating GANs
3. Privacy analysis of GANs and DP-GANs

## Related Works
Only discrete-valued GANs has been used to generate synthetic EHR data (Choi et al., 2017).

DP-SGD (Abadi et al., 2016) and PATE (Papernot et al., 2016) are two approaches for generating privacy guaranteed data.

## Methodology

![](https://i.imgur.com/zyaMqVw.png)

## Evaluation

### Maximum Mean Discrepancy

### Train on Synthetic, Test on Real

## Future Works
- Multi-model synthetic medical time-series data
- Enforcing Lipschitz constraint on RNNs

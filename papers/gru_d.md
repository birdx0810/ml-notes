# GRU-D: Recurrent Neural Networks for Multivariate Time Series with Missing Values
> Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag, Yan Liu
> ICLR 2017 (Rejected), Nature Scientific Reports (Accepted)
> Paper: [Link](https://www.nature.com/articles/s41598-018-24271-9)
> Open Review: [Link](https://openreview.net/forum?id=BJC8LF9ex)
> Code: [Link](https://github.com/PeterChe1990/GRU-D)

## Introduction
Time-series data usually has missing observations due to reasons such as medical events, anomalies, cost saving, incovenience etc. These missing values sometimes contain valuable information, such as missing value rates of the MIMIC-III dataset is correlated to their respective labels.

Informative missingness
: the missing values and their patterns within a time series data that provides rich information about target labels in supervised learning tasks.

Previous methods for dealing with missing values include:
1. smoothing/interpolation
2. spectral analysis
3. kernel methods
4. imputations
5. EM algorithm

Since these methods of imputation are only assumed to represent the real data, and might not necessarily represent them. This would lead to suboptimal predictions.

> Related Works: `Imputation`, `RNNs`

## Methodology
Proposed a GRU-D model that provides two additional representation for missing values:
1. $M$ `masking`: informs the model which inputs are observed (`true`) and missing (`false`)
2. $\Delta$ `time_interval`: informs the model regarding the time interval between the sampled data

![](https://i.imgur.com/uE5mYA3.png)

### Original GRU

$$
\begin{aligned}
z_t &= \sigma(W_zx_t + U_zh_{t-1} +b_z) \\
r_t &= \sigma(W_rx_t + U_rh_{t-1} +b_r) \\
\tilde{h}_t &= \tanh(Wx_t + U(r_t \odot h_{t-1}) +b) \\
h_t &= (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \\
\end{aligned}
$$

Where,
- $z_t$ denotes the update gate at time $t$ 
- $r_t$ denotes the reset gate at time $t$
- $\tilde{h}_t$ denotes the 

![](https://i.imgur.com/oZgCVvU.png)

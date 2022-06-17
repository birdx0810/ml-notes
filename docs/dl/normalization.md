# Normalization

_**Definition**_:
Normalization adjusts the values with different scales to have similar/compatible statistics or to stay within a certain range.

## Batch Norm

Batch normalization scales each dimension of an input $$x$$ to a succeeding layer such that they have mean zero \($$\mu=0$$\) and unary variance \($$\sigma^2=1$$\), this is often used with CNNs.

![](../.gitbook/assets/image%20%282%29.png)

## Layer Norm

Layer normalization scales input $$x$$ along the feature dimension based on the other features, this is often used with RNNs.

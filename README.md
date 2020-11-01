# RandomFourierFeatures
Implementation of Random Fourier Features using only numpy.

paper: [Random Features for Large-Scale Kernel Machines](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf) (NIPS 2008)
## Overview
Random Fourier Features (RFF) is a finite set of feature maps that approximate the inner product of a continuous and stationary positive definite kernel.  
The existence of this distribution is guaranteed by Bochner's theorem.

In this part, we approximate GP using RBF kernel by Bayesian Linear Regression (BLR) with RFF.

- Number of features : 1000
- Training points : 3, 10, 100, 3000

## Plot
In RFF, the predictions in extrapolation is unstable and cause variance starvation. (see https://arxiv.org/pdf/1706.01445.pdf)
|Gaussian process regression | Bayesian linear regression with RFF|
|:-:|:-:|
| <img src="https://github.com/SK-tklab/RandomFourierFeatures/blob/main/image/RFMGP_3.png" width="500px"> n=3 | <img src="https://github.com/SK-tklab/RandomFourierFeatures/blob/main/image/RFMBLR_3.png" width="500px"> n=3|
| <img src="https://github.com/SK-tklab/RandomFourierFeatures/blob/main/image/RFMGP_10.png" width="500px"> n=10 | <img src="https://github.com/SK-tklab/RandomFourierFeatures/blob/main/image/RFMBLR_10.png" width="500px"> n=10|
| <img src="https://github.com/SK-tklab/RandomFourierFeatures/blob/main/image/RFMGP_100.png" width="500px"> n=100 | <img src="https://github.com/SK-tklab/RandomFourierFeatures/blob/main/image/RFMBLR_100.png" width="500px"> n=100|
| <img src="https://github.com/SK-tklab/RandomFourierFeatures/blob/main/image/RFMGP_3000.png" width="500px"> n=3000 | <img src="https://github.com/SK-tklab/RandomFourierFeatures/blob/main/image/RFMBLR_3000.png" width="500px"> n=3000|

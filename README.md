# Automatic relevance determination in nonnegative matrix factorization (ARD-NMF)

This is a Python 3 implementation of ARD-NMF from Tan & Févotte (IEEE Transactions on Pattern Analysis and Machine Intelligence, 2013). The code here is based on Févotte's MATLAB implementation, in some cases copying code and comments verbatim.

### Set up

The only required software is Python 3.

#### Python dependencies

We include a list of required packages in `requirements.txt`. We recommend using Conda to install Python 3, and then install the required packages.

#### Installation

Beyond installing the dependencies above, no compilation of is required.

### Usage

The implementation of ARD-NMF was written to match the [coding guidelines of `scikit-learn`](http://scikit-learn.org/stable/developers/contributing.html#coding-guidelines), and is provided as a Python module. In that way, the main usage is to import the module (e.g. by running the Python shell in this directory):

    from ardnmf import ARDNMF
    model = ARDNMF(a=10)
    H = model.fit_transform(X)
    W = model.components_

### Parameters

The value for beta determines the cost function and controls the assumed statistics of the observation noise. It can be learned from training data by cross-training but this package expects beta to be a fixed value (default value for beta is 1). beta = 0 implies multiplicative Gamma observation noise, beta = 1 implies Poisson noise and beta = 2 implies Gaussian additive noise. For mutation signature extraction, the assumption of Poisson noise is reasonable.

They impose inverse-Gamma priors on each relevance weight where a is the (non-negative) shape hyperparameter and b is the scale hyperparameter. The value for b is computed using the algorithm originally described by Fevotte et al. Estimating the value for a is more difficult so it is required as an input. Fevotte et al. recommend a small value for a. 

#### SignatureAnalyzer

The PCAWG version of SignatureAnalyzer uses a=10, b=5 and phi=1 as the default parameters. They use an exponential prior for W (the signatures) and a half normal prior for H (the exposures). The algorithm for computing b assumes that the same prior is used for both W and H.


### Examples

We include an example application of ARDNMF to the "swimmer" dataset described in Tan & Févotte in `examples/swimmer`.

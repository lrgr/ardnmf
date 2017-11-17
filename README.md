# Automatic relevance determination in nonnegative matrix factorization (ARD-NMF)

This is a Python 3 implementation of ARD-NMF from Tan & Févotte (IEEE Transactions on Pattern Analysis and Machine Intelligence, 2013). The code here is based on Févotte's MATLAB implementation, in some cases copying code and comments verbatim.

### Set up

The only required software is Python 3.

#### Python dependencies

We include a list of required packages in `requirements.txt`. We recommend using Conda to install Python 3, and then install the required packages.

#### Installation

No compilation is required.

### Usage

The implementation of ARD-NMF was written to match the [coding guidelines of `scikit-learn`](http://scikit-learn.org/stable/developers/contributing.html#coding-guidelines), and is provided as a Python module. In that way, the main usage is to import the module (e.g. by running the Python shell in this directory):

    from ardnmf import ARDNMF
    model = ARDNMF()
    H = model.fit_transform(X)
    W = model.components_

### Examples

We include an example application of ARDNMF to the "swimmer" dataset described in Tan & Févotte in `examples/swimmer`.

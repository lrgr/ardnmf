#!/usr/bin/env python

# Load required modules
import sys, os, numpy as np, logging
from sklearn.base import BaseEstimator, TransformerMixin

# Load our modules
from .constants import *
from .fit import *
from .logger import *

# Define ARDNMF class
class ARDNMF(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, init=None, beta=1, tol=1e-5,
                 max_iter=200, random_state=None, verbose=logging.INFO,
                 prior=EXP_PRIOR, norm_H=True):
        self.n_components = n_components
        self.init = init
        self.beta = beta
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.prior = prior
        self.norm_H = norm_H

    def fit(self, X):
        raise NotImplementedError("ARDNMF.fit is not yet implemented.")

    def fit_transform(self, X, W=None, H=None):
        """Learn an ARDNMF model for the data and return the transformed data
        (i.e. the loadings).

        Note that our X, W, H are transposed compared to sklearn, i.e. compared
        to them we are learning X.T = W.T*H.T
        """
        # Run ARDNMF
        W, H, lambdas, obj, fit, bound = ardnmf(X, prior=self.prior,
            n_components=self.n_components, beta=self.beta,
            max_iter=self.max_iter, tol=self.tol, verbose=self.verbose,
            random_state=self.random_state)

        # Normalize W, H so bases sum to 1 (if necessary)
        if self.norm_H:
            W, H = _rescale(W, H)

        # Update state
        self.components_ = self.W = W
        self.H = H
        self.lambdas_ = lambdas
        self.obj_ = obj
        self.fit_ = fit
        self.bound_ = bound

        # Return the loadings
        return H

    def transform(self, X):
        raise NotImplementedError("ARDNMF.transform is not yet implemented.")

def _rescale(W, H):
    """Rescale so columns of W sum to 1"""
    col_sums = W.sum(axis=0)
    W = W/col_sums
    H = np.diag(col_sums).dot(H)
    return W, H

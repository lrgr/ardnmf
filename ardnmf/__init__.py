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
    def __init__(self, a, init=None, beta=1, tol=1e-5,
                 max_iter=200, random_state=None, verbose=logging.INFO,
                 prior=EXP_PRIOR, norm_H=True, tau=None):
        self.a = a
        self.init = init
        self.beta = beta
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.prior = prior
        self.norm_H = norm_H

        # As they suggest, we can always set tau to be the tolerance
        if tau == None:
            tau = tol
        self.tau = tau

    def fit(self, X):
        raise NotImplementedError("ARDNMF.fit is not yet implemented.")

    def fit_transform(self, X, W=None, H=None):
        """Learn an ARDNMF model for the data and return the transformed data
        (i.e. the loadings).

        Note that our X, W, H are transposed compared to sklearn, i.e. compared
        to them we are learning X.T = W.T*H.T.

        Thus, W represents "signatures" and H represents "exposures".
        """
        # Compute b from a
        L, N = X.shape
        self.b_, self.c_ = _compute_b_and_c(self.a, X, L, self.prior)
        self.B_ = self.b_ / self.c_

        # Run ARDNMF
        W, H, lambdas, obj, fit, bound = ardnmf(X, prior=self.prior, K=L,
            a=self.a, b=self.b_, beta=self.beta, max_iter=self.max_iter,
            tol=self.tol, verbose=self.verbose, random_state=self.random_state)

        # Choose K_effective
        k_eff, W, H = _choose_keff(lambdas[:, -1], self.tau, self.B_, W, H)
        self.k_eff_ = k_eff

        # Normalize W, H so bases sum to 1 (if necessary)
        if self.norm_H:
            W, H = _rescale(W, H)

        # Compute signature association probabilities
        self.p_ = _signature_association_probs(X, W, H)

        # Update state
        self.components_ = self.W_ = W
        self.H_ = H
        self.lambdas_ = lambdas
        self.obj_ = obj
        self.fit_ = fit
        self.bound_ = bound

        # Return the loadings
        return H

    def transform(self, X):
        raise NotImplementedError("ARDNMF.transform is not yet implemented.")

# Equation (34) from Tan & Fevotte
def _choose_keff(lambdas, tau, B, W, H):
    indices = np.where((lambdas-B)/B > tau)[0]
    return len(indices), W[:, indices], H[indices, :]

def _rescale(W, H):
    """Rescale so columns of W sum to 1"""
    col_sums = W.sum(axis=0)
    U = np.diag(col_sums)
    W = W.dot(np.linalg.inv(U))
    H = U.dot(H)
    return W, H

# Choose b with their heuristic (after equation 20 in Tan & Fevotte)
def _compute_b_and_c(a, X, K, prior):
    F, N = X.shape
    mean_X = X.sum() / (F*N) # Data sample mean per component
    if PRIOR_TO_L[prior] == 1:
        b = np.sqrt((a-1.)*(a-2.)*mean_X/K)
        c = F + N + a + 1
    elif PRIOR_TO_L[prior] == 2:
        b = (np.pi/2.)*(a-1.)*mean_X/K
        c = (F+N)/2 + a + 1
    else:
        raise NotImplementedError('Prior "%s" not implemented.' % prior)
    return b, c

def _signature_association_probs(X, W, H):
    """Compute the "association probability" of each mutation context in each
    sample to each signature.

    P[i, j, k] is the association of mutations of category i in sample j to
    signature k.

    TODO: Make this more vectorized."""
    F, N = X.shape
    k_eff = W.shape[1]
    P = np.zeros((F, N, k_eff))
    sig_norm = sum( W[:, k][:, None].dot(H[k][None, :]) for k in range(k_eff) )
    for k in range(k_eff):
        P[:, :, k] = W[:, k][:, None].dot(H[k][None, :])/sig_norm
    return P

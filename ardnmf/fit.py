#!/usr/bin/env python

################################################################################
# License
################################################################################
# Adapted from MATLAB software released from:
# Copyright 2013 Cedric Fevotte & Vincent Y. F. Tan
#
# Some documentation and code is copied verbatim (or as close as possible)
#
# This software is distributed under the terms of the GNU Public License
# version 3 (http://www.gnu.org/licenses/gpl.txt)
#
# The reference for this function is
#
# V. Y. F. Tan and C. Fevotte. Automatic relevance determination in
# nonnegative matrix factorization with the beta-divergence. IEEE
# Transactions on Pattern Analysis and Machine Intelligence,
# 35(7):1592-1605, July 2013.
################################################################################

# Load required modules
import sys, os, numpy as np, logging
from sklearn.utils import check_random_state
from .constants import *
from .logger import logger

################################################################################
# BETA DIVERGENCE
################################################################################
def betadiv(A,B,beta):
    """Computes beta-divergence between matrices
    betadiv(A,B,b) = = \sum \sum d(a_ij,b_ij) with
    - beta \= 0,1
      d(x|y) = (x^beta + (beta-1)*y^beta - beta*x*y^(beta-1))/(beta*(beta-1))
    - beta = 1 (Generalized Kullback-Leibler divergence)
      d(x|y) = x*log(x/y) - x + y
    - beta = 0 (Itakura-Saito divergence)
      d(x|y) = x/y - log(x/y) - 1"""
    # Flatten matrices (just like A(:) in MATLAB)
    A_flat = A.flatten()
    B_flat = B.flatten()

    if beta == 2:
        return np.sum((A_flat-B_flat)**2)/2;
    elif beta == 1:
        ind_0 = np.nonzero(A_flat <= EPS)
        #print(ind_0)
        #print(A_flat.shape[0])
        ind_1 = np.arange(A_flat.shape[0])
        ind_1 = np.delete(ind_1, ind_0)
        return np.sum(A_flat[ind_1]*np.log(A_flat[ind_1]/B_flat[ind_1]) -
                      A_flat[ind_1] + B_flat[ind_1]) + np.sum(B_flat[ind_0])
    elif beta == 0:
        return np.sum(A_flat/B_flat - np.log(A_flat/B_flat)) - (A_flat.shape[0])
    else:
        return np.sum(A_flat**beta + (beta-1)*B_flat**beta -
                      beta*A_flat*B_flat**(beta-1))/(beta*(beta-1))

################################################################################
# INITIALIZE ARDNMF
################################################################################
def init_ardnmf(V, prior, K, init=None, random_state=None):
    """Initialize W,H for ARDNMF randomly and based off mean of components.
    - a: Relevance parameters shape parameter (using ARDNMF default always)
    """
    rng = check_random_state(random_state)
    F, N = V.shape
    mean_V = V.sum() / (F*N) # Data sample mean per component
    W_ini = (rng.rand(F,K) + 1)*(np.sqrt(mean_V/K))
    H_ini = (rng.rand(K,N) + 1)*(np.sqrt(mean_V/K))

    return W_ini, H_ini

################################################################################
# FIT ARDNMF
################################################################################
# ARDNMF main
def ardnmf(V, prior, K, a, b, beta=1, init=None,  max_iter=200, tol=1e-5,
          verbose=logging.INFO, random_state=None):
    # Initialize W and H
    W, H = init_ardnmf(V, prior, K, init, random_state)

    # Fit
    if prior == EXP_PRIOR:
        return l1_ardnmf(V, W=W, H=H, a=a, b=b, beta=beta, max_iter=max_iter,
                         tol=tol, verbose=verbose, random_state=random_state)
    elif prior == HN_PRIOR:
        return l2_ardnmf(V, W=W, H=H, a=a, b=b, beta=beta, max_iter=max_iter,
                         tol=tol, verbose=verbose, random_state=random_state)
    else:
        raise NotImplementedError('Prior "%s" not implemented' % prior)

# L1 Normalization (i.e. exponential priors)
def l1_ardnmf(V, beta, tol, max_iter, W, H, a, b, verbose, random_state):
    F, N = V.shape
    K = W.shape[1]

    # To prevent from possible numerical instabilities, the data is added a
    # small value (Numpy's eps) and fitted to WH+eps instead of WH. You may
    # set eps = 0 to avoid this but divisions by zero may occur.
    V = V + EPS
    V_ap = W.dot(H) + EPS

    cst = F+N+a+1.
    bound = b/cst

    scale_W = np.sum(W,0).T
    scale_H = np.sum(H,1)
    inv_lambda = cst/(scale_W+scale_H+b)

    fit = np.array([0] * (max_iter+1), dtype=np.float64)
    obj = np.array([0] * (max_iter+1), dtype=np.float64)
    lambdas = np.zeros((K, max_iter+1))
    itera = 0
    rel = np.inf
    lambdas[:,itera] = 1./inv_lambda
    fit[itera] = betadiv(V,V_ap,beta)
    obj[itera] = fit[itera] + cst*np.sum(np.log(scale_W+scale_H+b))

    logger.setLevel(verbose)
    logger.info('iter = %4i | obj = %+5.2E | rel = %4.2E (target is %4.2E)' % (itera,obj[itera],rel,tol))

    while rel > tol and itera < max_iter:
        itera = itera + 1

        ## Update H ##
        if  beta > 2:
            R = np.tile(inv_lambda[:, None], (1,N))
            P = W.T.dot(V*V_ap**(beta-2.))
            Q = W.T.dot(V_ap**(beta-1.)) + R
            ex = 1./(beta-1.)
        elif beta == 2: # Euclidean distance
            R = np.tile(inv_lambda[:, None], (1,N))
            P = W.T.dot(V)
            Q = (W.T.dot(W)).dot(H) + R + np.tile(EPS*scale_W[:, None], (1,N)) # Use (V_ap*H.T+R) if K>F
            ex = 1.
        elif (beta < 2) and (beta > 1):
            R = np.tile(inv_lambda[:, None], (1,N))
            P = W.T.dot(V*V_ap**(beta-2.))
            Q = W.T.dot(V_ap**(beta-1.)) + R
            ex = 1.
        elif beta == 1: # Generalized KL divergence
            P = W.T.dot(V/V_ap)
            Q = np.tile((scale_W + inv_lambda)[:, None], (1, N))
            ex = 1.
        elif beta < 1:
            R = np.tile(inv_lambda[:, None], (1,N))
            P = W.T.dot(V*V_ap**(beta-2.))
            Q = W.T.dot(V_ap**(beta-1.)) + R
            ex = 1./(2-beta)

        ind = H>0
        H[ind] = H[ind]*(P[ind]/Q[ind])**ex
        scale_H = np.sum(H,1)

        V_ap = W.dot(H) + EPS

        ## Update W ##
        if  beta > 2:
            R = np.tile(inv_lambda.T, (F,1))
            P = (V*V_ap**(beta-2)).dot(H.T)
            Q = V_ap**(beta-1).dot(H.T) + R
            ex = 1./(beta-1)
        elif beta == 2:
            R = np.tile(inv_lambda.T, (F,1));
            P = V.dot(H.T)
            Q = W.dot((H.dot(H.T))) + R + np.tile(EPS*scale_H.T, (F,1)) # Use (V_ap*H.T+R) if K>N
            ex = 1.
        elif (beta < 2) and (beta > 1):
            R = np.tile(inv_lambda.T, (F,1))
            P = (V*V_ap**(beta-2)).dot(H.T)
            Q = V_ap**(beta-1).dot(H.T) + R
            ex = 1.
        elif beta == 1:
            P = (V/V_ap).dot(H.T)
            Q = np.tile(scale_H.T+inv_lambda.T, (F,1))
            ex = 1.
        elif beta < 1:
            R = np.tile(inv_lambda.T, (F,1))
            P = (V*V_ap**(beta-2))*H.T
            Q = V_ap**(beta-1)*H.T + R
            ex = 1./(2-beta)

        ind = W>0
        W[ind] = W[ind]*(P[ind]/Q[ind])**ex
        scale_W = np.sum(W,0).T

        V_ap = W.dot(H) + EPS

        ## Update lambda ##
        inv_lambda = cst/(scale_W+scale_H+b)

        ## Monitor ##
        fit[itera] = betadiv(V, V_ap, beta)
        obj[itera] = fit[itera] + cst*np.sum(np.log(scale_W+scale_H+b))
        lambdas[:,itera] = 1/inv_lambda;

        # Compute relative change of the relevance parameters
        rel = np.max(np.abs((lambdas[:,itera]-lambdas[:,itera-1])/lambdas[:,itera]))

        # Display objective value and relative change every 500 iterations
        if itera % 500 == 0:
            logger.info('iter = %4i | obj = %+5.2E | rel = %4.2E (target is %4.2E)' % (itera,obj[itera],rel,tol))

    # Trim variables
    fit = fit[:itera+1]
    obj = obj[:itera+1]
    lambdas = lambdas[:,:itera+1]

    # Add constant to optain true minus log posterior value
    obj = obj + (K*cst*(1.-np.log(cst)));

    # Display final values
    logger.info('iter = %4i | obj = %+5.2E | rel = %4.2E (target is %4.2E)'%(itera,obj[itera],rel,tol))
    if itera == max_iter:
        logger.info('Maximum number of iterations reached (n_iter_max = %d)' % max_iter)

    return W, H, lambdas, obj, fit, bound

# L2 normalization (i.e half-normal priors)
def l2_ardnmf(V, beta, tol, max_iter, W, H, a, b, verbose, random_state):
    """
    Documentation for ARD-NMF MATLAB code.

    Majorization-minimization algorithm for ARD-NMF with the beta-divergence
    and L2-norm penalization (half-normal prior)

    Input :
      - V : nonnegative matrix data (F x N)
      - beta : beta-divergence shape parameter value
      - tol : tolerance value for convergence
      - n_iter_max : maximum number of iterations
      - W : dictionary matrix initialization (F x K)
      - H : activation matrix initialization (K x N)
      - a : relevance prior shape parameter
      - b : relevance prior scale parameter

    We recommend experimenting with several values of 'a', using various
    orders of magnitude. Generally, a good start is a small value compared to
    F+N, say 'a = log(F+N)'. Pruning is increasingly aggressive as the value
    of 'a' decreases. Given a value of 'a', we recommend setting the
    other hyperparameter 'b' to '(pi/2)*(a-1)*sum(V(:))/(F*K*N)', see
    paper.

    Output :
      - W and H such that

                  V \approx W * H

      - lambdas : relevance parameters through iterations
      - obj : MAP objective through iterations
      - fit : beta-divergence btw V and WH through iterations
      - bound : lower bound on relevance paramaters value
    """
    F, N = V.shape
    K = W.shape[1]

    # To prevent from possible numerical instabilities, the data is added a
    # small value (Numpy's eps) and fitted to WH+eps instead of WH. You may
    # set eps = 0 to avoid this but divisions by zero may occur.
    V = V + EPS
    V_ap = W.dot(H) + EPS

    cst = (F+N)/2+a+1
    bound = b/cst

    scale_W = 0.5 * np.sum(W**2, axis=0).T
    scale_H = 0.5 * np.sum(H**2, axis=1)
    inv_lambda = cst/(scale_W+scale_H+b)

    fit = np.array([0] * (max_iter+1), dtype=np.float64)
    obj = np.array([0] * (max_iter+1), dtype=np.float64)
    lambdas = np.zeros((K, max_iter), dtype=np.float64)

    itera = 0
    rel = np.inf
    lambdas[:, itera] = 1./inv_lambda
    fit[itera] = betadiv(V,V_ap,beta)
    obj[itera] = fit[itera] + cst * np.sum(np.log(scale_W+scale_H+b))

    logger.setLevel(verbose)
    logger.info('iter = %4i | obj = %+5.2E | rel = %4.2E (target is %4.2E) \n',itera, obj[itera], rel, tol)
    while rel > tol and itera < max_iter:
        itera = itera + 1

        ## Update H ##
        R = H * np.tile(inv_lambda[:, None], (1,N))

        if beta > 2:
            P = W.T.dot(V*V_ap ** (beta-2.))
            Q = W.T.dot(V_ap ** (beta-1.)) + R
            ex = 1./(beta-1.)
        elif beta == 2:
            P = W.T.dot(V)
            Q = (W.T.dot(W)).dot(H) + R + np.tile(EPS*np.sum(W,0).T, (1,N))
            ex = 1.
        elif (beta < 2) and (beta != 1):
            P = W.T.dot(V*V_ap ** (beta-2.))
            Q = W.T.dot(V_ap ** (beta-1.)) + R
            ex = 1./(3.-beta)
        elif beta == 1:
            P = W.T.dot(V/V_ap)
            Q = np.tile(np.sum(W,0).T[:, None], (1,N)) + R
            ex = 1./2

        ind = H>0;
        H[ind] = H[ind] * (P[ind]/Q[ind]) ** ex
        scale_H = 0.5 * np.sum(H ** 2,1);

        V_ap = W.dot(H) + EPS

        ## Update W ##
        R = W * np.tile(inv_lambda.T, (F,1))

        if beta > 2:
            P = (V*V_ap ** (beta-2.)).dot(H.T)
            Q = (V_ap ** (beta-1.)).dot(H.T) + R
            ex = 1./(beta-1)
        elif beta == 2:
            P = V.dot(H.T)
            Q = W.dot((H.dot(H.T))) + R + np.tile(EPS*np.sum(H,axis=1).T, (F,1))
            ex = 1.
        elif (beta < 2) and (beta != 1):
            P = (V*V_ap ** (beta-2.)).dot(H.T)
            Q = (V_ap ** (beta-1.)).dot(H.T) + R
            ex = 1./(3.-beta)
        elif beta == 1:
            P = (V/V_ap).dot(H.T)
            Q = np.tile(np.sum(H,axis=1).T, (F,1)) + R
            ex = 1./2

        ind = W>0
        W[ind] = W[ind] * (P[ind]/Q[ind]) ** ex
        scale_W = 0.5 * np.sum(W**2, axis=0).T

        V_ap = W.dot(H) + EPS

        ## Update lambda ##
        inv_lambda = cst/(scale_W+scale_H+b)

        ## Monitor ##
        fit[itera] = betadiv(V, V_ap, beta)
        obj[itera] = fit[itera] + cst*np.sum(np.log(scale_W+scale_H+b))
        lambdas[:,itera] = 1./inv_lambda

        # Compute relative change of the relevance parameters
        rel = np.max(np.abs((lambdas[:, itera]-lambdas[:, itera-1])/lambdas[:,itera]))

        # Display objective value and relative change every 500 iterations
        if itera % 500 == 0:
            logger.info('iter = %4i | obj = %+5.2E | rel = %4.2E (target is %4.2E) \n',itera,obj[itera],rel,tol)

    # Trim variables
    fit = fit[:itera+1]
    obj = obj[:itera+1]
    lambdas = lambdas[:, :itera+1]

    # Add constant to obtain true minus log posterior value
    obj = obj + (K*cst*(1.-np.log(cst)))

    # Display final values
    logger.info('iter = %4i | obj = %+5.2E | rel = %4.2E (target is %4.2E) \n',itera, obj[itera], rel, tol)
    if itera == max_iter:
        logger.info('Maximum number of iterations reached (n_iter_max = %d) \n',n_iter_max)

    return W, H, lambdas, obj, fit, bound

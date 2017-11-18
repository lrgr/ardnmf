#!/usr/bin/env python

################################################################################
# SETUP
################################################################################
# Load required modules
import sys, os, numpy as np, argparse, matplotlib.pyplot as plt
from scipy.io import loadmat
plt.style.use('ggplot')

# Load our modules
this_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(this_dir + '../../'))
from ardnmf import ARDNMF, PRIORS, EXP_PRIOR, PRIOR_TO_L

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', type=str, required=True)
parser.add_argument('-o', '--output_prefix', type=str, required=True)
parser.add_argument('-p', '--prior', type=str, required=False,
    default=EXP_PRIOR, choices=PRIORS)
parser.add_argument('-v', '--verbose', type=int, required=False, default=1)
parser.add_argument('-b', '--beta', type=float, required=False, default=1)
args = parser.parse_args(sys.argv[1:])

################################################################################
# MAIN
################################################################################
# Load input data
V = loadmat(args.input_file)['V']

# Plot some of the images for fun
F, N = V.shape
for i in range(6):
    plt.subplot(2,3,i+1)
    n = np.random.randint(N)
    plt.imshow(np.reshape(V[:,n], (32, 32)))
    plt.title('sample %s' % n)
plt.savefig(args.output_prefix + '-samples.pdf')
plt.clf()

## Create and run the model
# Set parameters
K = 32 # Nb of components
tol = 1e-5 # Tolerance value for convergence
max_iter = 10000 # Max nb of iterations
L = PRIOR_TO_L[args.prior] # L1-ARD (L=1) or L2-ARD (L = 2)

# Fit
model = ARDNMF(n_components=K, beta=args.beta, tol=tol, max_iter=max_iter,
               verbose=args.verbose, prior=args.prior, norm_H=False)
H = model.fit_transform(V)
W = model.components_

## Display fit and relevance parameters
fig = plt.gcf()
fig.set_size_inches(15, 5)
if not np.any(model.obj_ <= 0):
    plt.subplot(1,3,1)
    plt.loglog(model.obj_,'k')
    plt.title('objective function')

plt.subplot(1,3,2)
plt.loglog(model.fit_,'k')
plt.title('fit to data (beta-divergence)')

plt.subplot(1,3,3)
plt.plot(model.lambdas_.T-model.bound_,'k')
plt.xlim([1, len(model.fit_)])
plt.title('relevance')

plt.savefig(args.output_prefix + '-fit-and-relevance.pdf')

## Display learnt dictionary
order = np.argsort(-model.lambdas_[:,-1])
W_o = W[:,order]
H_o = H[order,:]
lambda_o = model.lambdas_[order,-1]

# Rescale W by prior expectation for improved readability (see paper)
if L == 1:
    W_sc = W_o.dot(np.diag(lambda_o))
elif L == 2:
    W_sc = W_o.dot(np.diag(np.sqrt(2*lambda_o/np.pi)))

# Rescale values to [1:64] to unify colormaps between plots
W_sc = np.floor(64*W_sc/max(W_sc.flatten())+1)

fig = plt.gcf()
fig.set_size_inches(K*5, int(np.ceil(K/4.)*5))
for k in range(K):
    plt.subplot(4,int(np.ceil(K/4.)),k+1)
    plt.imshow(np.reshape(W_sc[:,k],(32,32)))
    plt.title(['Dictionary element %s' % k])

plt.savefig(args.output_prefix + '-dictionary.pdf')

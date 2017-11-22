#!/usr/bin/env python

################################################################################
# SETUP
################################################################################
# Load required modules
import sys, os, argparse, numpy as np, pandas as pd, logging, time
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Load our modules
this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir + '../../'))
from ardnmf import ARDNMF, PRIORS, EXP_PRIOR

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', type=str, required=True,
                   help='Mutation count table in Pandas format('\
                   'categories as rows, samples as columns)')
parser.add_argument('-o', '--output_prefix', type=str, required=True)
parser.add_argument('-v', '--verbose', type=int, required=False,
                    default=logging.INFO)
parser.add_argument('-ni', '--n_initializations', type=int, required=False,
                    default=10)
parser.add_argument('-mi', '--max_iter', type=int, required=False,
                    default=10000000)
parser.add_argument('-p', '--prior', type=str, required=False,
                    default=EXP_PRIOR, choices=PRIORS)
parser.add_argument('-b', '--beta', type=float, required=False, default=1.)
parser.add_argument('-a', type=float, required=False, default=10.)
parser.add_argument('-rs', '--random-seed', type=int, default=int(time.time()),
                    required=False)
args = parser.parse_args(sys.argv[1:])

################################################################################
# INFER SIGNATURES AND TAKE RUN WITH MAXIMUM POSTERIOR PROBABILITY
################################################################################
# Load the input data and convert to a Numpy matrix
df = pd.read_csv(args.input_file, sep='\t', index_col=0)
samples_with_clustering = list(df.columns)
categories = list(df.index)
subs = np.unique([ c.split('[')[1].split(']')[0] for c in categories ]) # maintains order in which they occur
V = df.as_matrix()

# Run ARD-NMF
models = []
for ini in range(args.n_initializations):
    model = ARDNMF(a=args.a, max_iter=args.max_iter, tol=1e-7, norm_H=True,
                   prior=args.prior, beta=args.beta,
                   random_state=args.random_seed+ini, tau=0.001)
    H = model.fit_transform(V)
    models.append(model)

# Get the distribution of Ks, and choose the K that occurs the most often.
# Then choose the model with highest likelihood for that K
from collections import Counter
K_to_count = Counter([ model.k_eff_ for model in models ])
K_eff = max(K_to_count.keys(), key=lambda K: K_to_count[K])
print('Choosing K=%s' % K_eff)
model = sorted([ m for m in models if m.k_eff_ == K_eff ], key=lambda m: m.obj_[-1])[-1]

## Following Kasar, et al, plot number of mutations assigned to each signature
## - We assume that the categories are given in lexicographic order by
##   substitution, with one flanking base (i.e. 16 categories per substitution)
n_subs = len(subs) # 6 possible substitutions
n_cats_per_sub = int(len(categories)/len(subs)) # usually 16; 1 flank on each side
N = V.shape[1]
fig, axes = plt.subplots(K_eff, n_subs, sharex=True, sharey=True)
for s in range(K_eff):
    for m in range(n_subs):
        ax = axes[s][m]
        counts = []
        for i in range(m*n_cats_per_sub, (m+1)*n_cats_per_sub):
            count = 0.
            for j in range(N):
                if model.p_[i, j, s] >= 0.75:
                    count += V[i, j]
            counts.append(count)

        ax.bar(range(n_cats_per_sub), counts)

# Add labels
for i in range(K_eff):
    left_ax = axes[i][0]
    left_ax.set_ylabel('Mutation contributions')

for j, sub in enumerate(subs):
    top_ax = axes[0][j]
    top_ax.set_title(sub)

    bottom_ax = axes[-1][j]
    bottom_ax.set_xlabel('Category')

plt.show()
sys.exit()

## Plot the best signatures
fig, axes = plt.subplots(K_eff, n_subs, sharex=True, sharey=True)
for i in range(K_eff):
    for j in range(n_subs):
        ax = axes[i][j]
        ax.bar(range(n_cats_per_sub), model.W_[j*n_cats_per_sub:(j+1)*n_cats_per_sub, i])

# Add labels
for i in range(K_eff):
    left_ax = axes[i][0]
    left_ax.set_ylabel('Probability')

for j, sub in enumerate(subs):
    top_ax = axes[0][j]
    top_ax.set_title(sub)

    bottom_ax = axes[-1][j]
    bottom_ax.set_xlabel('Category')

plt.show()

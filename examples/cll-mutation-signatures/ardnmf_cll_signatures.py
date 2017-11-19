#!/usr/bin/env python

################################################################################
# SETUP
################################################################################
# Load required modules
import sys, os, argparse, numpy as np, pandas as pd, logging
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
                    default=1)
parser.add_argument('-p', '--prior', type=str, required=False,
                    default=EXP_PRIOR, choices=PRIORS)
parser.add_argument('-b', '--beta', type=float, required=False, default=1.)
parser.add_argument('-a', type=float, required=False, default=10.)
args = parser.parse_args(sys.argv[1:])

################################################################################
# INFER SIGNATURES AND TAKE RUN WITH MAXIMUM POSTERIOR PROBABILITY
################################################################################
# Load the input data and convert to a Numpy matrix
df = pd.read_csv(args.input_file, sep='\t', index_col=0)
samples_with_clustering = list(df.columns)
categories = list(df.index)
V = df.as_matrix()

# Run ARD-NMF
for _ in range(args.n_initializations):
    model = ARDNMF(a=args.a, max_iter=2000, tol=1e-7, norm_H=True,
                   prior=args.prior, beta=args.beta)
    H = model.fit_transform(V)

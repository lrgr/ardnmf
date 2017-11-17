#!/usr/bin/env python

# Load required modules
from numpy import finfo

# Constants
EXP_PRIOR = 'exponential'
HN_PRIOR = 'half-normal'
PRIORS = {EXP_PRIOR, HN_PRIOR}
EPS = finfo(float).eps

PRIOR_TO_L = { EXP_PRIOR: 1, HN_PRIOR: 2 }

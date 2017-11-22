#!/usr/bin/env python

################################################################################
# SET UP
################################################################################
# Load required modules
import sys, os, argparse, pandas as pd, numpy as np
from itertools import permutations

# Constants
BASES       = [ 'A', 'C', 'G', 'T' ]
PURINES     = set('AG')
BASE_PAIR   = dict(A='T', C='G', G='C', T='A')
VAR = 'Variant Base'
REF = 'Reference Base'
SAMPLE = 'Sample'
FPRIME = "5' Flanking Bases"
TPRIME = "3' Flanking Bases"
CAT = 'Category'
CHR = 'Chromosome'
POS = 'Position'
MUT_DIST = 'Distance to Previous Mutation'
NEAREST_MUT = 'Distance to Nearest Mutation'

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', type=str, required=True, help='Excel format')
parser.add_argument('-o', '--output_file', type=str, required=True)
parser.add_argument('--columns', type=str, required=False, nargs='*',
                    default=['Tumor_Sample_Barcode','Variant_Type',
                             'Reference_Allele', 'Tumor_Seq_Allele1',
                             'Tumor_Seq_Allele2', 'ref_context',
                             'Start_position', 'Chromosome'])
parser.add_argument('-sr', '--skip_rows', type=int, required=False, default=0)
parser.add_argument('-lf', '--left_flank', type=int, required=False, default=1,
                    dest='lf')
parser.add_argument('-rf', '--right_flank', type=int, required=False, default=1,
                    dest='rf')
parser.add_argument('-nmt', '--nearest_mutation_threshold', type=int,
                    required=False, default=1000)
args = parser.parse_args(sys.argv[1:])


################################################################################
# LOAD AND FILTER THE MAF
################################################################################
# Load the MAF
maf = pd.read_excel(args.input_file, skiprows=args.skip_rows)
maf = maf[args.columns]
maf = maf.loc[maf['Variant_Type'] == 'SNP']
maf = maf.rename(columns={
    'Tumor_Sample_Barcode': SAMPLE,
    'Reference_Allele': REF,
    'Tumor_Seq_Allele2': VAR,
    'Start_position': POS
})

# Add the 5' and 3' flanking bases (MAFs by default, have 20 flanking
# -- 10 on each side, with original base in the middle)
maf[FPRIME] = maf.apply(lambda x: x['ref_context'][:10].upper(),
                        axis='columns')
maf[TPRIME] = maf.apply(lambda x: x['ref_context'][11:].upper(),
                        axis='columns')

################################################################################
# ADD MUTATION CATEGORIES
################################################################################
# Construct a "category" in the Alexandrov et al. style from a given
# mutation
def category_name( five_prime, ref, variant, three_prime):
    if ref in PURINES:
        ref, variant	= BASE_PAIR[ref], BASE_PAIR[variant]
        five_prime_orig = list(five_prime)
        five_prime	= [ BASE_PAIR[b] for b in three_prime ]
        three_prime	= [ BASE_PAIR[b] for b in five_prime_orig ]
    return '%s[%s>%s]%s' % (''.join(five_prime), ref, variant, ''.join(three_prime))

# Define our list of categories
categories = [ category_name(five, ref, variant, three)
                for ref, variant in ['CA', 'CG', 'CT', 'TA', 'TC', 'TG']
                for five in [ list(t) for t in permutations(BASES, args.lf) ]
                for three in [ list(t) for t in permutations(BASES, args.rf) ] ]

def add_mutation_category_column(df):
    # Make a list of all possible categories
    ncats  = len(categories)
    categoryToIndex = dict(zip(categories, range(ncats)))

    # Add a mutation category column
    nflank = len(sorted(set(df[FPRIME]))[0])
    df[CAT] = df.apply(lambda m: category_name(m[FPRIME][nflank-args.lf:],
                                               m[REF], m[VAR], m[TPRIME][:args.rf]),
                              axis='columns')

    # Filter out mutations with categories not in our lists (usually
    # because the reference genome had a 'N' base)
    all_n_muts = len(df)
    df = df.loc[df[CAT].isin(set(categories))].copy() # necessary to avoid warnings
    n_muts = len(df)

    return df

# Add the distance to the previous mutation, using Infinity
# for mutations on different chromosomes
def add_dist_to_prev_mut_column(df):
    def dist_to_prev_mut(row):
        index = df.index.get_loc(row.name)
        if index == 0:
            return np.inf
        else:
            prev_row = df.iloc[index - 1]
            if row[SAMPLE] == prev_row[SAMPLE] and \
               row[CHR] == prev_row[CHR]:
                return int(row[POS] - prev_row[POS])
            else:
                return np.inf

    # Get distance to nearest mutation (before or after). The
    # basic logic is that the nearest mutation is the minimum
    # of the previous mutation distance, or the next mutation's
    # previous mutation distance.
    def nearest_mut(row):
        j = df.index.get_loc(row.name)+1
        if j > len(df.index)-1:
            return row[MUT_DIST]
        else:
            return min(row[MUT_DIST], df.iloc[j][MUT_DIST])

    df[MUT_DIST] = df.apply(dist_to_prev_mut, axis='columns')
    df[NEAREST_MUT] = df.apply(nearest_mut, axis='columns')

    return df

maf = add_dist_to_prev_mut_column(add_mutation_category_column(maf))

################################################################################
# CLEAN UP AND OUTPUT TO FILE
################################################################################
# Gather some information so we can make a mutation count matrix
samples = sorted(set(maf[SAMPLE]))
N = len(samples)
L = len(categories)
sampleToIndex = dict(zip(samples, range(N)))
catToIndex = dict(zip(categories, range(L)))

# Create a matrix of the mutation counts
V = np.zeros((L, 2*N), dtype=np.int)
for _, r in maf.iterrows():
    i = catToIndex[r[CAT]]
    if r[NEAREST_MUT] <= args.nearest_mutation_threshold:
        j = 2*sampleToIndex[r[SAMPLE]]
    else:
        j = 2*sampleToIndex[r[SAMPLE]] + 1
    V[i, j] += 1

# Turn into dataframe so we have sample/category names
samples_with_clustering = ['%s-%s' % (s, c) for s in samples for c in ['clustered', 'unclustered']]
mut_counts = pd.DataFrame(V, index=categories, columns=samples_with_clustering)

# Output to file
mut_counts.to_csv(args.output_file, sep='\t', index=True)

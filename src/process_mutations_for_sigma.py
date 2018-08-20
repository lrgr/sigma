#!/usr/bin/env python

# Load required modules
import sys, os, argparse, json, pandas as pd, logging
from collections import defaultdict
from itertools import permutations
from pandas import read_csv, Categorical

# Load our modules
from data_utils import get_logger
from constants import PATIENT, CATEGORY, CATEGORY_IDX, MUT_DIST

# Command-line argument parser
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mf', '--mutations_file', type=str, required=True)
    parser.add_argument('-sf', '--signatures_file', type=str, required=True)
    parser.add_argument('-of', '--output_file', type=str, required=True)
    parser.add_argument('-v', '--verbosity', type=int, required=False,
                        default=logging.INFO)
    return parser

# Main
def run( args ):
    # Set up logger
    logger = get_logger(args.verbosity)
    logger.info('[Loading input data]')
    
    # Load the signatures
    sig_df = pd.read_csv(args.signatures_file, sep='\t', index_col=0)
    categories = list(sig_df.columns)
    category_index = dict(zip(categories, range(len(categories))))

    logger.info('- Loaded %s x %s signature matrix' % sig_df.values.shape)
    
    # Load the mutations
    mut_df = pd.read_csv(args.mutations_file, sep='\t', usecols=[PATIENT, CATEGORY, MUT_DIST])
    samples = sorted(set(mut_df[PATIENT]))

    logger.info('- Loaded %s mutations in %s samples' % (len(mut_df), len(samples)))

    # Add the category index and create sequences of mutations
    logger.info('[Processing data into SigMa format]')
    mut_df[CATEGORY_IDX] = mut_df.apply(lambda r: category_index[r[CATEGORY]],
                                    axis='columns')
    
    sampleToSequence = dict( (s, list(map(int, s_df[CATEGORY_IDX])))
                                 for s, s_df in mut_df.groupby([PATIENT]) )
    sampleToPrevMutDists = dict( (s, list(map(float, s_df[MUT_DIST])))
                                     for s, s_df in mut_df.groupby([PATIENT]) )
    
    # Save to JSON
    logger.info('- Saving to file %s' % args.output_file)
    with open(args.output_file, 'w') as OUT:
        output = dict(sampleToSequence=sampleToSequence,
                      sampleToPrevMutDists=sampleToPrevMutDists,
                      samples=samples, categories=categories,
                      params=vars(args))
        json.dump( output, OUT )

if __name__ == '__main__': run( get_parser().parse_args(sys.argv[1:]) )

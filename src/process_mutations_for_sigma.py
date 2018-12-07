#!/usr/bin/env python

# Load required modules
import sys, os, argparse, json, pandas as pd, logging, numpy as np
from collections import defaultdict
from itertools import permutations
from pandas import read_csv, Categorical

# Load our modules
from data_utils import get_logger
from constants import PATIENT, CATEGORY, CATEGORY_IDX, MUT_DIST, CHROMOSOME, START_POS

# Command-line argument parser
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mf', '--mutations_file', type=str, required=True)
    parser.add_argument('-mbf', '--mappability_blacklist_file', type=str,
                        required=False, default=None)
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
    mut_df = pd.read_csv(args.mutations_file, sep='\t',
                         usecols=[PATIENT, CATEGORY, MUT_DIST, CHROMOSOME, START_POS],
                         dtype={PATIENT: str, CATEGORY: str, MUT_DIST: np.float,
                                CHROMOSOME: str, START_POS: int })
    samples = sorted(set(mut_df[PATIENT]))

    logger.info('- Loaded %s mutations in %s samples' % (len(mut_df), len(samples)))

    # If a mappability blacklist is provided, use it remove mutations
    if not (args.mappability_blacklist_file is None):
        # Load the dataframe and process into a dictionary
        mappability_df = pd.read_csv(args.mappability_blacklist_file, sep=',')
        chrom_idx, start_idx, stop_idx = mappability_df.columns[:3]

        map_blacklist = defaultdict(list)
        unmappable_bases = 0
        for _, r in mappability_df.iterrows():
            chrom = r[chrom_idx][3:]
            map_blacklist[chrom].append(( int(r[start_idx]), int(r[stop_idx]) ))
            unmappable_bases += map_blacklist[chrom][-1][1] - map_blacklist[chrom][-1][0]

        logger.info('- Loaded unmappable regions spanning %s bases in %s chromosomes' % (unmappable_bases, len(map_blacklist)))
        
        # Restrict mutations that fall in a blacklisted region
        logger.info('[Removing unmappable mutations]')
        def mappable(r):
            for start, stop in map_blacklist[r[CHROMOSOME]]:
                if start <= r[START_POS] <= stop:
                    return False
            return True
        
        n_muts = len(mut_df)
        mut_df = mut_df[mut_df.apply(mappable, axis='columns')]

        n_unmappable = n_muts-len(mut_df)
        logger.info('\t-> Removed %s mutations that were not mappable' % n_unmappable)

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

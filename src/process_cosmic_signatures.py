#!/usr/bin/env python

# Load required modules
import sys, os, argparse, pandas as pd, logging, numpy as np

# Helpers for parsing categories into substitution, left flanking,
# and right flanking
def sub(c): return c.split('[')[1].split(']')[0]
def lf(c): return c.split('[')[0]
def rf(c): return c.split(']')[-1]

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, required=True)
    parser.add_argument('-o', '--output_file', type=str, required=True)
    parser.add_argument('-v', '--verbosity', type=int, required=False, default=logging.INFO)
    args = parser.parse_args(sys.argv[1:])

    # Set up logger
    logger = logging.getLogger(__name__)
    logger.setLevel(args.verbosity)

    # Load the signatures
    logger.info('[Loading the signatures]')
    with open(args.input_file, 'r') as IN:
        arrs = [ l.rstrip('\n\t').split('\t') for l in IN ]
        header = arrs.pop(0)

        # Get categories and sort according to standard
        categories = [ arr[2] for arr in arrs ]
        categories.sort(key=lambda c: (sub(c), lf(c), rf(c)))
    
        # Create a container for the signatures
        sig_names = header[3:]
        K = len(sig_names)
        L = len(categories)
        sigs = np.zeros((K, L))

        # Parse the lines in the file
        for arr in arrs:
            j = categories.index(arr[2])
            for i, sig_name in enumerate(sig_names):
                sigs[i,j] += float(arr[3+i])

        logger.info('- Loaded %s x %s signature matrix' % sigs.shape)

    # Create dataframe and output to file
    logger.info('[Creating dataframe]')
    df = pd.DataFrame(index=sig_names, columns=categories, data=sigs)

    logger.info('- Saving to %s' % args.output_file)
    df.to_csv(args.output_file, sep='\t')

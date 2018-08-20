#!/usr/bin/env python

# Load required modules
import sys, os, argparse, logging, pandas as pd
from data_utils import get_logger
from constants import PATIENT

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-mf', '--mutations_file', type=str, required=True)
parser.add_argument('-sf', '--samples_file', type=str, required=True)
parser.add_argument('-of', '--output_file', type=str, required=True)
parser.add_argument('-v', '--verbosity', type=int, required=False, default=logging.INFO)
args = parser.parse_args(sys.argv[1:])

# Load the mutations file
mut_df = pd.read_csv(args.mutations_file, sep='\t', low_memory=False)
cols = mut_df.columns

# Load the sample name mapping
sample_df = pd.read_csv(args.samples_file, sep='\t')
sample_ids = [ s.split('a')[0] if s.endswith('a') or s.endswith('a2') else s.split('b')[0]
               for s in sample_df['submitted_sample_id'] ]
patient_map = dict(zip(sample_df['icgc_donor_id'], sample_ids))

# Map the patients and save
mut_df[PATIENT] = [ patient_map[p] for p in mut_df[PATIENT] ]
mut_df.to_csv(args.output_file, sep='\t', index=0)

#!/usr/bin/env python

# Load required modules
import matplotlib
matplotlib.use('Agg')
import sys, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from os import listdir
from os.path import isfile, join
sns.set_style('whitegrid')

# Load our modules
from data_utils import load_json
from constants import MODEL_NAMES, SIGMA_NAME, MMM_NAME

def across_all_dir(path, files):
    scores = np.zeros((len(files), 24))
    for i, f in enumerate(files):
        sample_dict = load_json(join(path, f))
        for j, c in enumerate(sample_dict['chromosomesToResults'].keys()):
            scores[i][j] = sample_dict['chromosomesToResults'][c]['score']
    return scores


def fix_nans(mat):
    """
    returns the matrix with average over models if a model, sample, chromosome had nan in it.
    :param mat: ndarray (model, sample, chromosome)
    :return: mat ndarray (model, sample, chromosome)
    """
    mat = np.nan_to_num(mat)
    idx, idy, idz = np.where(mat == 0)
    for x, y, z in zip(idx, idy, idz):
        mat[x, y, z] = mat[:, y, z].mean()
    return mat


def analyze(par_path, out_fig):
    dirs = [d for d in listdir(par_path)]
    dirs = sorted(dirs)

    thresholds = []
    tmp = []
    names = []
    if MMM_NAME in dirs:
        tmp.append(MMM_NAME)
        names.append(MMM_NAME.upper())
    for d in dirs:
        if SIGMA_NAME in d:
            thresholds.append(int(d.split('a')[1]))

    thresholds = sorted(thresholds)
    print(thresholds)
    for i in thresholds:
        tmp.append(SIGMA_NAME + str(i))
        names.append(str(i // 1000) + 'K')

    dirs = tmp
    files_dict = {}
    for d in dirs:
        curr_path = join(par_path, d)
        files_dict[d] = sorted([f.split('-')[1] for f in listdir(curr_path) if isfile(join(curr_path, f))])

    files_intersection = files_dict[dirs[0]]
    for d in dirs:
        files_intersection = np.intersect1d(files_dict[d], files_intersection)

    print('number of samples in common: {}'.format(len(files_intersection)))

    results_mat = np.zeros((len(dirs), len(files_intersection), 24))
    for i, d in enumerate(dirs):
        path = join(par_path, d)
        if SIGMA_NAME in d:
            prefix = SIGMA_NAME + '-'
        else:
            prefix = MMM_NAME + '-'

        scores = across_all_dir(path, [prefix + f for f in files_intersection])
        results_mat[i] = scores

    results_mat = fix_nans(results_mat)
    sum_results = np.sum(results_mat, axis=(1, 2)).tolist()
    for i in range(len(dirs)):
        print('{}:   {}'.format(dirs[i], sum_results[i]))

    # Create the plot, highlighting the one with maximum log-likelihood
    max_ind = np.argmax(sum_results)
    ind = list(range(0, max_ind)) + list(range(max_ind+1, len(names)))
    plt.bar(ind, sum_results[:max_ind] + sum_results[max_ind+1:])

    plt.bar([max_ind], [sum_results[max_ind]], color=(181./255, 85./255, 85./255))

    plt.xticks(range(len(names)), names)
    plt.ylim((min(sum_results) - 1000, max(sum_results) + 1000))
    plt.xlabel('Model')
    plt.ylabel('Held-out log-likelihood')
    plt.title('Comparing MMM and SigMa with various cloud thresholds')
    
    plt.tight_layout()
    plt.savefig(out_fig)


def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-ld', '--loocv_dir', type=str, required=True)
    parser.add_argument('-of', '--output_file', type=str, required=True)
    return parser


def main(args):
    analyze(args.loocv_dir, args.output_file)


if __name__ == '__main__': main( get_parser().parse_args(sys.argv[1:]) )

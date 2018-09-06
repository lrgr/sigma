from os import listdir
from os.path import isfile, join
import numpy as np
from data_utils import load_json
from constants import MODEL_NAMES, SIGMA_NAME, MMM_NAME
import matplotlib.pyplot as plt
import sys


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
        names.append(MMM_NAME)
    for d in dirs:
        if SIGMA_NAME in d:
            thresholds.append(int(d.split('a')[1]))

    thresholds = sorted(thresholds)
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
    sum_results = np.sum(results_mat, axis=(1, 2))
    for i in range(len(dirs)):
        print('{}:   {}'.format(dirs[i], sum_results[i]))

    plt.bar(names, sum_results)
    plt.ylim((sum_results.min() - 1000, sum_results.max() + 1000))
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

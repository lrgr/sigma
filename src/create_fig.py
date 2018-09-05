from os import listdir
from os.path import isfile, join
import numpy as np
from data_utils import load_json
from constants import MODEL_NAMES, SIGMA_NAME, MMM_NAME
import matplotlib.pyplot as plt
import sys


def across_all_dir(path, files):
    scores = []
    for i, f in enumerate(files):
        result_dict = load_json(join(path, f))
        scores.append(result_dict['results']['score'])
    return scores


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

    results_mat = np.zeros((len(files_intersection), len(dirs)))
    for i, d in enumerate(dirs):
        path = join(par_path, d)
        if SIGMA_NAME in d:
            prefix = SIGMA_NAME + '-'
        else:
            prefix = MMM_NAME + '-'

        scores = across_all_dir(path, [prefix + f for f in files_intersection])
        results_mat[:, i] = scores

    sum_results = np.sum(results_mat, axis=0)
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

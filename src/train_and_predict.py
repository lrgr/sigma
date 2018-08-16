#!/usr/bin/env python

################################################################################
# SETUP
################################################################################
# Load required modules
import sys, os, numpy as np, json, logging
from time import time
from tqdm import tqdm

# Load our modules
from constants import MODEL_NAMES, SIGMA_NAME, MMM_NAME
from models.MMMFrozenEmissions import MMMFrozenEmissions
from models.SigMa import SigMa
from data_utils import get_split_sequences_by_threshold, to_json, get_logger


################################################################################
# HELPERS
################################################################################
def leave_one_out(sample_seqs_tuple, model_name, emissions, max_iterations, epsilon, random_state):
    seqs = sample_seqs_tuple[1]
    n_seq = len(seqs)

    chromosomes_names = ['chromosome%s' % str(i).zfill(2) for i in range(n_seq)]

    chromosome_to_experiment_dict = {}

    total_score = 0
    total_time = 0
    total_test_length = 0
    total_train_length = 0
    total_num_iterations = 0
    for i in range(n_seq):

        train_data = []
        train_length = 0
        test_length = 0

        test_data = seqs[i]
        for k in range(n_seq):
            if k != i:
                train_data.extend(seqs[k])

        for seq in train_data:
            train_length += len(seq)
        for seq in test_data:
            test_length += len(seq)

        tic = time()
        model, num_iterations = get_trained_model(model_name, emissions, train_data, epsilon, max_iterations, random_state)
        train_time = time() - tic

        score = model.log_probability(test_data)

        current_dict = {'score': score, 'time': train_time, 'trainLength': train_length, 'testLength': test_length,
                        'numIterations': num_iterations}
        chromosome_to_experiment_dict[chromosomes_names[i]] = current_dict

        total_time += train_time
        total_score += score
        total_test_length += test_length
        total_train_length += train_length
        total_num_iterations += num_iterations

    summery_dict = {'time': total_time, 'score': total_score,
                    'testLength': total_test_length, 'trainLength': total_train_length,
                    'numIterations': total_num_iterations}
    output_dict = {'results': summery_dict, 'chromosomes': chromosomes_names,
                   'chromosomesToResults': chromosome_to_experiment_dict, 'numberChromosomes': n_seq}
    return output_dict


def get_viterbi(sample_seqs_tuple, model_name, emissions, max_iterations, epsilon, random_state):
    train_data = []
    train_length = 0
    for s in sample_seqs_tuple[1]:
        train_data.extend(s)
        train_length += len(s[0])

    tic = time()
    model, num_iterations = get_trained_model(model_name, emissions, train_data, epsilon, max_iterations, random_state)
    train_time = time() - tic

    score = model.log_probability(train_data)
    out_dict = {'score': score, 'numIterations': num_iterations, 'time': train_time, 'trainLength': train_length}

    viterbi = model.predict(train_data)
    if model_name == MMM_NAME:
        out_dict['viterbi'] = viterbi
    elif model_name == SIGMA_NAME:
        out_dict['viterbi'] = viterbi[0]
        out_dict['cloud_indicator'] = viterbi[1]

    return out_dict


def get_trained_model(model_name, emissions, train_data, epsilon, max_iterations, random_state):
    model = get_model(model_name, emissions, random_state)
    num_iterations = model.fit(train_data, stop_threshold=epsilon, max_iterations=max_iterations)
    return model, num_iterations


def get_model(model_name, emissions, random_state=None):
    if model_name == SIGMA_NAME:
        return SigMa(emissions, random_state)
    elif model_name == MMM_NAME:
        return MMMFrozenEmissions(emissions, random_state=random_state)
    else:
        raise NotImplementedError('Model "%s" not implemented' % args.model_name)

################################################################################
# MAIN
################################################################################
# Parser for command-line arguments
def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-sf', '--signatures_file', type=str, required=True)
    parser.add_argument('-mf', '--mutations_file', type=str, required=True)
    parser.add_argument('-mn', '--model_name', type=str, required=True, choices=MODEL_NAMES)
    parser.add_argument('-ct', '--cloud_threshold', type=int, required=True)
    parser.add_argument('-sn', '--sample_names', type=str, required=False, nargs='*',
                        default=[])
    parser.add_argument('-mi', '--max_iter', type=int, required=False, default=500)
    parser.add_argument('-tol', '--tolerance', type=float, required=False, default=1e-3)
    parser.add_argument('-rs', '--random_state', type=int, required=False, default=5733)
    parser.add_argument('-od', '--output_directory', type=str, required=True)
    parser.add_argument('--cross-validation-mode', action='store_true', default=False,
                        required=False)
    parser.add_argument('-v', '--verbosity', type=int, required=False, default=logging.INFO)
    return parser

# Main
def main(args):
    """
    The main function to reproducing the paper's results
    :param command: 'loo' for leave one out. 'viterbi' for viterbi
    :param model: 'sigma' or 'mmm'
    :param batch: Takes indices from batch * batch_size to (batch + 1) * batch_size
    :param batch_size: see batch
    :param threshold: Define maximal distance (in bp) in clouds. Use 0 to not split (i.e use whole chromosomes)
    :param max_iterations: Maximal number of iterations for training the model
    :param epsilon: Minimum improvement in every iteration of the model, if improvement is lower stop training
    :param random_state: Random state to initialize the models
    :param out_dir: where to save all the files
    :return:
    """
    # Create simple logger
    logger = get_logger(args.verbosity)
    logger.info('[Loading input data]')
    
    # Get the list of samples we're going to run on
    with open(args.mutations_file, 'r') as IN:
        samples = json.load(IN).get('samples')
    if len(args.sample_names) == 0:
        sample_indices = range(len(samples))
    else:
        sample_indices = [ samples.index(s) for s in args.sample_names ]
        
    logger.info('- Loading data for %s samples' % len(sample_indices))

    # Load the emissions matrix
    emissions = np.load(args.signatures_file)
    logger.info('- Loaded %s x %s emissions matrix' % emissions.shape)
    # if threshold <= 0:
    #     out_dir_for_file = os.path.join(out_dir, model)
    #     threshold = 1e99
    # else:
    #     out_dir_for_file = os.path.join(out_dir, model + '_' + str(threshold))

    experiment_tuples = get_split_sequences_by_threshold(args.mutations_file, args.cloud_threshold, sample_indices)

    # Perform the experiments
    logger.info('[Performing experiments]')
    if args.cross_validation_mode:
        logger.info('- Cross-validation mode')
        func = leave_one_out
    else:
        logger.info('- Viterbi mode')
        func = get_viterbi

    for experiment_tuple in tqdm(experiment_tuples, total=len(sample_indices), ncols=80):
        sample = experiment_tuple[0]
        out_file = '%s/%s-%s' % (args.output_directory, args.model_name, sample)
        dict_to_save = func(experiment_tuple, args.model_name, emissions,
                            args.max_iter, args.tolerance, args.random_state)
        to_json(out_file, dict_to_save)
        
    logger.info('- Done')

if __name__ == '__main__': main( get_parser().parse_args(sys.argv[1:]) )

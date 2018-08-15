import os

from models.MMMFrozenEmissions import MMMFrozenEmissions
from models.SigMa import SigMa
from utils.data_utils import get_split_sequences_by_threshold, to_json
import numpy as np
import time


def main(command, model, batch, batch_size, threshold, max_iterations=None, epsilon=None, random_state=5733, out_dir='results'):
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
    start = batch * batch_size
    finish = (batch + 1) * batch_size
    sample_indices = range(start, finish)
    data_file = 'data/nik-zainal2016-wgs-brca-mutations-for-hmm.json'

    if command == 'loo':
        out_dir += '/leave_one_out'
        func = leave_one_out
        if max_iterations is None:
            max_iterations = 100
        if epsilon is None:
            epsilon = 1e-2
    elif command == 'viterbi':
        out_dir += '/viterbi'
        func = get_viterbi
        if max_iterations is None:
            max_iterations = 500
        if epsilon is None:
            epsilon = 1e-3
    else:
        return

    if threshold <= 0:
        out_dir_for_file = os.path.join(out_dir, model)
        threshold = 1e99
    else:
        out_dir_for_file = os.path.join(out_dir, model + '_' + str(threshold))

    experiment_tuples = get_split_sequences_by_threshold(data_file, threshold, sample_indices)

    try:
        os.makedirs(out_dir_for_file)
    except OSError:
        pass

    for experiment_tuple in experiment_tuples:
        sample = experiment_tuple[0]
        out_file = out_dir_for_file + "/" + sample
        if os.path.isfile(out_file + '.json'):
            continue
        dict_to_save = func(experiment_tuple, model, max_iterations, epsilon, random_state)
        to_json(out_file, dict_to_save)


def leave_one_out(sample_seqs_tuple, model_name, max_iterations, epsilon, random_state):
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

        tic = time.clock()
        model, num_iterations = get_trained_model(model_name, train_data, epsilon, max_iterations, random_state)
        train_time = time.clock() - tic

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


def get_viterbi(sample_seqs_tuple, model_name, max_iterations, epsilon, random_state):
    train_data = []
    train_length = 0
    for s in sample_seqs_tuple[1]:
        train_data.extend(s)
        train_length += len(s[0])

    tic = time.clock()
    model, num_iterations = get_trained_model(model_name, train_data, epsilon, max_iterations, random_state)
    train_time = time.clock() - tic

    score = model.log_probability(train_data)
    out_dict = {'score': score, 'numIterations': num_iterations, 'time': train_time, 'trainLength': train_length}

    viterbi = model.predict(train_data)
    if model_name == 'mmm':
        out_dict['viterbi'] = viterbi
    elif model_name == 'sigma':
        out_dict['viterbi'] = viterbi[0]
        out_dict['cloud_indicator'] = viterbi[1]

    return out_dict


def get_trained_model(model_name, train_data, epsilon, max_iterations, random_state):
    model = get_model(model_name, random_state)
    num_iterations = model.fit(train_data, stop_threshold=epsilon, max_iterations=max_iterations)
    return model, num_iterations


def get_model(model_name, random_state=None):

    if model_name == 'sigma':
        emissions = np.load('data/emissions_for_breast_cancer.npy')
        model = SigMa(emissions, random_state)

    elif model_name == 'mmm':
        emissions = np.load('data/emissions_for_breast_cancer.npy')
        model = MMMFrozenEmissions(emissions, random_state=random_state)

    return model

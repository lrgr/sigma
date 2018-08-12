import os

from models.MMMFrozenEmissions import MMMFrozenEmissions
from utils.data_utils import get_split_sequences_by_threshold, get_split_sequences, to_json
import numpy as np
import time
from models.SigMa import SigMa


def chromosomes_experiment(model, sample_indices, threshold, out_dir='results\leave_one_out'):

    if threshold <= 0:
        threshold = 0
        experiment_tuples = get_split_sequences('data/nik-zainal2016-wgs-brca-mutations-for-hmm.json',
                                                sample_indices)
    else:
        experiment_tuples = get_split_sequences_by_threshold('data/nik-zainal2016-wgs-brca-mutations-for-hmm.json',
                                                             threshold, sample_indices)

    out_dir_for_file = os.path.join(out_dir, model + '_' + str(threshold))

    try:
        os.makedirs(out_dir_for_file)
    except OSError:
        pass

    for experiment_tuple in experiment_tuples:
        sample = experiment_tuple[0]
        out_file = out_dir_for_file + "/" + sample
        if os.path.isfile(out_file + '.json'):
            continue
        dict_to_save = leave_one_out(experiment_tuple, model, threshold)
        to_json(out_file, dict_to_save)


def leave_one_out(sample_seqs_tuple, model_name, threshold, epsilon=1e-3):
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

        if threshold > 0:
            test_data = seqs[i]
            for k in range(n_seq):
                if k != i:
                    train_data.extend(seqs[k])

        else:
            test_data = [seqs[i]]
            for k in range(n_seq):
                if k != i:
                    train_data.append(seqs[k])

        for seq in train_data:
            train_length += len(seq)
        for seq in test_data:
            test_length += len(seq)

        model = get_model(model_name)

        tic = time.time()
        num_iterations = model.fit(train_data, stop_threshold=epsilon, max_iterations=100)
        train_time = time.time() - tic

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


def get_model(model_name):

    if model_name == 'sigma':
        emissions = np.load('data/emissions_for_breast_cancer.npy')
        model = SigMa(emissions)

    elif model_name == 'mmm':
        emissions = np.load('data/emissions_for_breast_cancer.npy')
        model = MMMFrozenEmissions(emissions)

    return model


# TODO - make a get_data function for more modular code, it should get the sample, the threshold, the fold,
# TODO - and should return train and test sequences
def get_data():
    pass


def main(model, batch, batch_size, threshold):
    start = batch * batch_size
    finish = (batch + 1) * batch_size
    sample_indices = range(start, finish)
    chromosomes_experiment(model, sample_indices, threshold)

import os

from models.MMMFrozenEmissions import MMMFrozenEmissions
from utils.data_utils import get_split_sequences_by_threshold, get_split_sequences, to_json
import numpy as np
from models.SigMa import SigMa


def all_viterbi(model, sample_indices, threshold, max_iterations, epsilon, random_state, out_dir):

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
        curr_seqs = experiment_tuple[1]
        if threshold > 0:
            seqs = []
            n_seq = len(curr_seqs)
            for i in range(n_seq):
                for s in curr_seqs[i]:
                    seqs.append(s)
        else:
            seqs = curr_seqs

        dict_to_save = create_viterbi(model, seqs, max_iterations, epsilon, random_state)
        out_file = out_dir_for_file + "/" + sample
        to_json(out_file, dict_to_save)


def create_viterbi(model_name, seqs, max_iterations, epsilon, random_state):
    model = get_model(model_name, random_state)

    model.fit(seqs, stop_threshold=epsilon, max_iterations=max_iterations)
    if model_name == 'sigma':
        cloud_indicator, viterbi = model.predict(seqs)
        dict_to_save = {'viterbi': viterbi, 'cloud_indicator': cloud_indicator}

    elif model_name == 'mmm':
        viterbi = model.predict(seqs)
        dict_to_save = {'viterbi': viterbi}

    return dict_to_save


def get_model(model_name, random_state=None):

    if model_name == 'sigma':
        emissions = np.load('data/emissions_for_breast_cancer.npy')
        model = SigMa(emissions, random_state)

    elif model_name == 'mmm':
        emissions = np.load('data/emissions_for_breast_cancer.npy')
        model = MMMFrozenEmissions(emissions, random_state=random_state)

    return model


# TODO - make a get_data function for more modular code, it should get the sample, the threshold, the fold,
# TODO - and should return train and test sequences
def get_data():
    pass


def main(model, batch, batch_size, threshold, max_iterations=500, epsilon=1e-3, random_state=5733, out_dir='results/viterbi'):
    start = batch * batch_size
    finish = (batch + 1) * batch_size
    sample_indices = range(start, finish)
    all_viterbi(model, sample_indices, threshold, max_iterations, epsilon, random_state, out_dir)

import numpy as np
from pomegranate import *
import json

################################################################################
# LOGGING
################################################################################
import logging

# Logging format
FORMAT = '%(asctime)s %(levelname)-10s: %(message)s'
logging.basicConfig(format=FORMAT)

def get_logger(verbosity=logging.INFO):
    '''
    Returns logger object
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(verbosity)
    return logger

################################################################################
# UTILS
################################################################################
def sample_and_noise(model, noise_dist, n_seqs, seqs_len):
    noise_change_dist = DiscreteDistribution(dict(zip(range(96), [1.0 / 96] * 96)))
    seqs = []
    noised_seqs = []
    for i in range(n_seqs):
        seq = np.array(model.sample(seqs_len))
        seqs.append(seq)
        noised_seq = seq.copy()
        hits = noise_dist.sample(seqs_len)
        for j, hit in enumerate(hits):
            if hit == 0:
                noised_seq[j] = noise_change_dist.sample()
        noised_seqs.append(noised_seq)
    return seqs, noised_seqs


def get_emissions(file='data\emissions_for_breast_cancer'):
    return np.load(file + '.npy')


def sample_uniform_between_a_b(n_states, a=0.0, b=1.0):
    return (b - a) * np.random.sample(n_states) + a


def random_seqs_from_json(file_name, n_seqs=10):
    seqs = []
    seqs_names = []
    json_file = json.load(open(file_name))
    samples = json_file[u'samples']
    samples_to_seq = json_file[u'sampleToSequence']
    samples = np.random.permutation(samples)
    for i in range(n_seqs):
        seqs.append(samples_to_seq[samples[i]])
        seqs_names.append(samples[i])
    return seqs, seqs_names


def to_json(file_name, dict_to_save):
    with open(file_name + '.json', 'w') as fp:
        json.dump(dict_to_save, fp)


def full_sample_to_chromosomes_seqs(sample, dists_sample):
    np_sample = np.array(sample)
    starting_chromosome_idxs = np.where(np.array(dists_sample) >= 1e100)[0]
    return np.split(np_sample, starting_chromosome_idxs)[1:]


def load_json(file_name):
    return json.load(open(file_name))


def get_split_sequences(file_name, sample_numbers=None):
    json_file = json.load(open(file_name))
    samples = json_file[u'samples']
    samples_to_seq = json_file[u'sampleToSequence']
    samples_dists = json_file[u'sampleToPrevMutDists']
    out_seqs = []
    out_names = []

    if sample_numbers is None:
        sample_numbers = range(len(samples))

    for i in sample_numbers:
        n = samples[i]
        out_names.append(n)
        out_seqs.append(full_sample_to_chromosomes_seqs(samples_to_seq[n], samples_dists[n]))
    return zip(out_names, out_seqs)


def get_full_sequences(file_name='data/nik-zainal2016-wgs-brca-mutations-for-hmm.json'):
    json_file = json.load(open(file_name))
    samples = json_file[u'samples']
    samples_to_seq = json_file[u'sampleToSequence']
    out_seqs = []
    out_names = []
    for n in samples:
        out_names.append(n)
        out_seqs.append(samples_to_seq[n])
    return zip(out_names, out_seqs)


def get_count_sequences_as_mat(file_name='data/nik-zainal2016-wgs-brca-mutations-for-hmm.json'):
    json_file = json.load(open(file_name))
    samples = json_file[u'samples']
    samples_to_seq = json_file[u'sampleToSequence']

    # finding num_object + counting
    num_objects = 0
    samples_objects = []
    samples_counts = []
    for sample in samples:
        objects, counts = np.unique(samples_to_seq[sample], return_counts=True)
        samples_objects.append(objects)
        samples_counts.append(counts)
        num_objects = max(num_objects, np.max(objects))
    num_objects += 1

    count_mat = np.zeros((len(samples), num_objects))
    for i in range(len(samples)):
        count_mat[i, samples_objects[i]] = samples_counts[i]
    return count_mat


def get_samples_names(file_name='data/nik-zainal2016-wgs-brca-mutations-for-hmm.json'):
    json_file = json.load(open(file_name))
    samples = json_file[u'samples']
    return samples


def get_split_sequences_by_threshold(file_name, threshold, sample_numbers=None):
    json_file = json.load(open(file_name))
    samples = json_file[u'samples']
    samples_to_seq = json_file[u'sampleToSequence']
    samples_dists = json_file[u'sampleToPrevMutDists']
    out_seqs = []
    out_names = []

    if sample_numbers is None:
        sample_numbers = range(len(samples))

    for i in sample_numbers:
        n = samples[i]
        out_names.append(n)
        out_seqs.append(full_sample_to_chromosomes_seqs_by_threshold(samples_to_seq[n], samples_dists[n], threshold))
    return zip(out_names, out_seqs)


def full_sample_to_chromosomes_seqs_by_threshold(sample, dists_sample, threshold):
    np_sample = np.array(sample)
    np_dists = np.array(dists_sample)

    starting_chromosome_idxs = np.where(np_dists >= 1e100)[0]

    chromosomes = np.split(np_sample, starting_chromosome_idxs)[1:]
    chromosomes_dists = np.split(np_dists, starting_chromosome_idxs)[1:]

    out = []
    for i in range(len(chromosomes)):
        chromosome = chromosomes[i]
        chromosome_dists = chromosomes_dists[i]

        starting_seqs_idxs = np.where(chromosome_dists >= threshold)[0]
        seqs = np.split(chromosome, starting_seqs_idxs)[1:]
        out.append(seqs)
    return out


def seqs_to_seq(seqs):
    out = []
    for seq in seqs:
        out.extend(seq)
    return np.array(out)


def seqs_to_seq_of_prefix(seqs):
    out = []
    for seq in seqs:
        out.append(seq[0])
    return np.array(out)


def sample_indices_not_in_dir(dir_path):
    import os
    samples_in_dir = [f[:-5] for f in os.listdir(dir_path)]
    samples = get_samples_names()
    missing_indices = []
    for i in range(len(samples)):
        if samples[i] not in samples_in_dir:
            missing_indices.append(i)
    return missing_indices

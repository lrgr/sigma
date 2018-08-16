import numpy as np
import time
from scipy.misc import logsumexp


class MultinomialMixtureModel:
    def __init__(self, num_states, num_emissions, emissions=None, weights=None, random_state=1000):
        self.num_states = num_states
        self.num_emissions = num_emissions
        if random_state is None:
            random_state = time.time()
        np.random.seed(int(random_state))
        self.weights = np.random.dirichlet([0.5] * num_states) if weights is None else weights
        self.emissions = np.random.dirichlet([0.5] * num_emissions, size=num_states) if emissions is None else emissions
        self.weights = np.log(self.weights)
        self.emissions = np.log(self.emissions)

    def expectation_step(self, log_counts):

        # initializing expected_emissions as log(a[i]) + log(e[i, j])
        expected_emissions = (self.emissions.T + self.weights).T

        # computing the expected emissions
        expected_emissions = expected_emissions + log_counts - logsumexp(expected_emissions, axis=0)
        expected_weights = logsumexp(expected_emissions, axis=1)
        return expected_weights, expected_emissions

    def maximization_step(self, expected_weights, expected_emissions):
        self.emissions = (expected_emissions.T - expected_weights).T
        self.weights = expected_weights

    def projection_step(self):
        return

    def pretrain(self, data):
        return data

    def fit(self, data, stop_threshold=1e-3, max_iterations=1e8):

        count = 0
        iteration = 0

        data = self.pretrain(data)

        # counting the data
        if isinstance(data[0], int):
            counts = np.bincount(data)
        else:
            counts = np.zeros(self.num_emissions)
            for seq in data:
                for l in seq:
                    counts[l] += 1

        normalized_counts = counts / np.sum(counts)
        log_normalized_counts = np.log(normalized_counts)
        prev_score = self.log_probability_counts(counts)
        for iteration in range(int(max_iterations)):
            # expectation step
            expected_weights, expected_emissions = self.expectation_step(log_normalized_counts)

            # maximization step
            self.maximization_step(expected_weights, expected_emissions)

            # projection step
            self.projection_step()

            score = self.log_probability_counts(counts)

            count = count + 1 if score - prev_score < stop_threshold else 0
            prev_score = score

            if count >= 2:
                break

        return iteration

    def get_log_emissions_vector(self):
        out_vec = (self.emissions.T + self.weights).T
        out_vec = logsumexp(out_vec, axis=0)
        return out_vec

    def get_log_weights_vector(self):
        return self.weights

    def get_weights_vector(self):
        return np.exp(self.weights)

    def get_emissions_vector(self):
        return np.exp(self.get_log_emissions_vector())

    def get_emissions_matrix(self):
        return np.exp(self.emissions)

    def log_probability(self, seqs):
        log_prob = 0
        log_prob_vec = self.get_log_emissions_vector()
        for seq in seqs:
            for l in seq:
                log_prob += log_prob_vec[l]
        return log_prob

    def log_probability_counts(self, counts):
        return np.inner(counts, self.get_log_emissions_vector())

    def test_model(self):
        # weight test
        weights_sum = logsumexp(self.weights)
        if not np.isclose(weights_sum, 0.0):
            print('weights are bad')

        # emissions test
        emissions_sum = logsumexp(self.emissions, axis=1)
        for i in range(len(emissions_sum)):
            if not np.isclose(emissions_sum[i], 0.0):
                print('state {} emissions are bad'.format(i))

    def predict(self, seqs):
        gmm_emissions = (self.get_emissions_matrix().T * self.get_weights_vector()).T
        gmm_most_probable_state = np.argmax(gmm_emissions, axis=0).tolist()

        most_probable_seqs = []

        for seq in seqs:
            current_most_probable_seq = []
            for i in range(len(seq)):
                current_most_probable_seq.append(gmm_most_probable_state[seq[i]])

            most_probable_seqs.append(current_most_probable_seq)

        return most_probable_seqs

    def predict_seq(self, seq):
        gmm_emissions = (self.get_emissions_matrix().T * self.get_weights_vector()).T
        gmm_most_probable_state = np.argmax(gmm_emissions, axis=0).tolist()

        current_most_probable_seq = []
        for i in range(len(seq)):
            current_most_probable_seq.append(gmm_most_probable_state[seq[i]])
        return current_most_probable_seq

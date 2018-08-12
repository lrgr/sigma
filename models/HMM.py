import numpy as np
import time
from pomegranate import HiddenMarkovModel, State, DiscreteDistribution
from scipy.misc import logsumexp


class HMM:
    def __init__(self, num_states, num_emissions, laplace=0, random_state=None):
        self.num_states = num_states
        self.num_emissions = num_emissions
        self.laplace = laplace
        if random_state is None:
            random_state = time.time()
        np.random.seed(int(random_state))
        self.transitions = np.random.sample((num_states, num_states))
        self.transitions = (self.transitions.transpose() / np.sum(self.transitions, axis=1)).transpose()
        self.emissions = np.random.dirichlet([0.5] * num_emissions, size=num_states)
        self.start_prob = np.random.dirichlet([0.5] * num_states)
        self.hmm = None
        self.update_hmm()

    def update_hmm(self):
        num_states = self.num_states
        start_prob = self.start_prob
        num_emissions = self.num_emissions

        hmm = HiddenMarkovModel('hmm')
        dist = [DiscreteDistribution(dict(zip(range(num_emissions), self.emissions[i]))) for i in range(num_states)]
        states = [State(dist[i], 's' + str(i).zfill(2)) for i in range(num_states)]
        hmm.add_states(states)
        for i in range(num_states):
            s_i = states[i]
            hmm.add_transition(hmm.start, s_i, start_prob[i])
            for j in range(num_states):
                s_j = states[j]
                p = self.transitions[i, j]
                hmm.add_transition(s_i, s_j, p)

        self.hmm = hmm
        self.hmm.bake()

    def log_fwa_bwa(self, seq):
        forward_matrix = self.hmm.forward(seq)[1:, :self.num_states]
        backward_matrix = self.hmm.backward(seq)[1:, :self.num_states]
        return forward_matrix, backward_matrix

    def pre_train(self, data, stop_threshold, max_iterations):
        return data

    def fit(self, data, stop_threshold=1e-3, max_iterations=1e8, no_emissions=False):
        count = 0
        iteration = 0
        self.update_hmm()
        prev_score = self.log_probability(data)

        data = self.pre_train(data, stop_threshold, max_iterations)

        # scores = []
        for iteration in range(int(max_iterations)):
            # expectation step
            expected_transitions, expected_emissions, expected_start_count = self.expectation_step(data, no_emissions=no_emissions)

            expected_transitions, expected_emissions, expected_start_count =\
                self.expectation_projection_step(expected_transitions, expected_emissions, expected_start_count)

            # maximization step
            self.maximization_step(expected_transitions, expected_emissions, expected_start_count)

            # projection step
            self.projection_step()

            self.update_hmm()

            score = self.log_probability(data)
            # print(score)
            # scores.append(score)

            count = count + 1 if score - prev_score < stop_threshold else 0
            prev_score = score

            if count >= 2:
                break

        # import matplotlib.pyplot as plt
        # plt.plot(scores)
        # plt.show()

        return iteration

    def log_probability(self, seqs):
        score = 0
        for seq in seqs:
            forward_matrix = self.hmm.forward(seq)
            score += logsumexp(forward_matrix[-1])
        return score

    def expectation_projection_step(self, expected_transitions, expected_emissions, expected_start_count):
        return expected_transitions, expected_emissions, expected_start_count

    # need to learn how to compute it better! (using only log fwa bwa)
    def expectation_step(self, seqs, no_emissions=False):
        num_states = self.num_states
        num_emissions = self.num_emissions
        expected_transitions = np.zeros((num_states, num_states))
        expected_emissions = np.zeros((num_states, num_emissions))
        expected_start_count = np.zeros(num_states)
        if self.laplace:
            expected_transitions += self.laplace
            expected_emissions += self.laplace
            expected_start_count += self.laplace
        expected_emissions = np.log(expected_emissions)
        # log_emissions = np.log(self.emissions)
        # log_transitions = np.log(self.transitions)
        for seq in seqs:
            # print(np.max(np.absolute(expected_emissions - np.exp(expected_emissions_2))))
            # fwa, bwa = self.log_fwa_bwa(seq)
            # b = fwa + bwa
            # b -= logsumexp(b[0])
            # temp = np.zeros((num_states, num_states, len(seq)))
            # for l in range(len(seq) - 1):
            #     normalizer = 0
            #     for i in range(num_states):
            #         for j in range(num_states):
            #             temp[i, j, l] =
            #               fwa[l, i] + log_transitions[i, j] + log_emissions[j, seq[l + 1]] + bwa[l + 1, j]
            #             normalizer += normalizer + np.log1p(np.exp(temp[i, j, l] - normalizer))
            #     for i in range(num_states):
            #         for j in range(num_states):
            #             temp[i, j, l] = temp[i, j, l] - normalizer
            if len(seq) == 1:
                a, b = self.hmm.forward_backward(seq)
                expected_start_count += a[num_states, :num_states]
                if not no_emissions:
                    expected_emissions[:, seq[0]] = np.logaddexp(expected_emissions[:, seq[0]], b[0])
            else:
                a, b = self.hmm.forward_backward(seq)
                expected_transitions += a[:num_states, :num_states]
                expected_start_count += a[num_states, :num_states]
                if not no_emissions:
                    for i,  l in enumerate(seq):
                        expected_emissions[:, l] = np.logaddexp(expected_emissions[:, l], b[i])
        if no_emissions:
            expected_emissions = self.emissions
        else:
            expected_emissions = np.exp(expected_emissions)
        return expected_transitions, expected_emissions, expected_start_count

    def maximization_step(self, expected_transitions, expected_emissions, expected_start_prob):
        self.transitions = (expected_transitions.T / np.sum(expected_transitions, axis=1)).T
        self.emissions = (expected_emissions.T / np.sum(expected_emissions, axis=1)).T
        self.start_prob = expected_start_prob / np.sum(expected_start_prob)

    def projection_step(self):
        return

    def sample(self, lengths):
        return self.hmm.sample(lengths)

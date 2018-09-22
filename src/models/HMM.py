import numpy as np
import time
from pomegranate import HiddenMarkovModel, State, DiscreteDistribution
from scipy.misc import logsumexp
np.warnings.filterwarnings('ignore')


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
        prev_score = -np.inf

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

    def log_probability(self, seqs, weights=None):
        score = 0
        if weights is None:
            for seq in seqs:
                score += self.hmm.log_probability(seq)
        else:
            for i, seq in enumerate(seqs):
                score += weights[i] * self.hmm.log_probability(seq)
        return score

    def expectation_projection_step(self, expected_transitions, expected_emissions, expected_start_count):
        return expected_transitions, expected_emissions, expected_start_count

    def expectation_step(self, seqs, weights=None, no_emissions=False):
        num_states = self.num_states
        num_emissions = self.num_emissions
        expected_transitions = np.zeros((num_states, num_states))
        expected_emissions = np.zeros((num_emissions, num_states))
        expected_start_count = np.zeros(num_states)
        if self.laplace:
            expected_transitions += self.laplace
            expected_emissions += self.laplace
            expected_start_count += self.laplace
        expected_emissions = np.log(expected_emissions)
        for i, seq in enumerate(seqs):
            a, b = self.hmm.forward_backward(seq)
            if weights is not None:
                a *= weights[i]
            expected_start_count += a[num_states, :num_states]
            if len(seq) == 1:
                if not no_emissions:
                    if weights is not None:
                        b += np.log(weights[i])
                    expected_emissions[seq[0]] = np.logaddexp(expected_emissions[seq[0]], b[0])
            else:
                expected_transitions += a[:num_states, :num_states]
                if not no_emissions:
                    if weights is not None:
                        b += np.log(weights[i])
                    for i,  l in enumerate(seq):
                        expected_emissions[l] = np.logaddexp(expected_emissions[l], b[i])
        if no_emissions:
            expected_emissions = self.emissions
        else:
            expected_emissions = np.exp(expected_emissions)
        return expected_transitions, expected_emissions.T, expected_start_count

    def expectation_step2(self, seqs, no_emissions=False):
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
        log_emissions = np.log(self.emissions.T)
        log_transitions = np.log(self.transitions)
        log_start_prob = np.log(self.start_prob)
        score = 0

        import time
        for seq in seqs:
            tic = time.clock()
            a2, b2 = self.hmm.forward_backward(seq)
            print(time.clock() - tic)

            tic = time.clock()
            fwa, bwa = self.log_fwa_bwa(seq)
            print(time.clock() - tic)
            seq_score = logsumexp(fwa[-1])
            score += seq_score
            print(time.clock() - tic)

            # emissions estimation
            b = fwa + bwa
            b -= seq_score
            print(time.clock() - tic)

            # transitions estimation
            # tic = time.clock()
            temp = np.zeros((len(seq) - 1, num_states, num_states))
            for l in range(len(seq) - 1):
                np.add.outer(fwa[l], bwa[l + 1] + log_emissions[seq[l + 1]], out=temp[l])
                temp[l] += log_transitions
            print(time.clock() - tic)

            a = logsumexp(temp, 0)
            a -= seq_score
            a = np.exp(a)
            print(time.clock() - tic)

            # start estimation
            expected_start_count += np.exp(log_start_prob + log_emissions[seq[0]] + bwa[1] - seq_score)
            print(time.clock() - tic)
            if len(seq) == 1:
                if not no_emissions:
                    expected_emissions[:, seq[0]] = np.logaddexp(expected_emissions[:, seq[0]], b[0])
            else:
                expected_transitions += a[:num_states, :num_states]
                if not no_emissions:
                    for i,  l in enumerate(seq):
                        expected_emissions[:, l] = np.logaddexp(expected_emissions[:, l], b[i])
        if no_emissions:
            expected_emissions = self.emissions
        else:
            expected_emissions = np.exp(expected_emissions)

        return expected_transitions, expected_emissions, expected_start_count, score

    def maximization_step(self, expected_transitions, expected_emissions, expected_start_prob):
        self.transitions = (expected_transitions.T / np.sum(expected_transitions, axis=1)).T
        self.emissions = (expected_emissions.T / np.sum(expected_emissions, axis=1)).T
        self.start_prob = expected_start_prob / np.sum(expected_start_prob)

    def projection_step(self):
        return

    def sample(self, lengths):
        return self.hmm.sample(lengths)

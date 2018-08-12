import numpy as np
from models.MMMFrozenEmissions import MMMFrozenEmissions
from models.HMM import HMM


class SigMa:
    def __init__(self, emissions):
        self.num_states = len(emissions)
        self.num_emissions = len(emissions[0])

        self.mmm = MMMFrozenEmissions(emissions)
        self.hmm = None
        self.init_hmm()

    def init_hmm(self):
        self.hmm = HMMForSigma(self.mmm.get_emissions_matrix(), self.mmm.get_emissions_vector())

    def fit(self, seqs, max_iterations=1e8, stop_threshold=1e-3):

        # seqs of length = 1
        hmm_seqs = []
        # seqs of length > 1
        gmm_seqs = []

        for seq in seqs:
            if len(seq) == 1:
                gmm_seqs.append(seq)
            else:
                hmm_seqs.append(seq)

        if len(gmm_seqs) != 0:
            self.mmm.fit(gmm_seqs, max_iterations=max_iterations, stop_threshold=stop_threshold)
            self.init_hmm()

        count = 0
        iteration = 0
        prev_score = self.hmm.log_probability(hmm_seqs)
        b = np.zeros((2, 2))
        rho = np.zeros(2)
        transitions = np.zeros((self.num_states, self.num_states))
        hmm_start = np.zeros(self.num_states)

        import time

        for iteration in range(int(max_iterations)):
            # expectation step
            expected_transitions, _, expected_start_count = self.hmm.expectation_step(hmm_seqs, no_emissions=True)

            # expectation projection
            # start
            rho[0] = np.sum(expected_start_count[:-1])
            rho[1] = expected_start_count[-1]

            # rho
            b[0, 0] = np.sum(expected_transitions[:-1, :-1])
            b[0, 1] = np.sum(expected_transitions[:-1, -1])
            b[1, 0] = np.sum(expected_transitions[-1, :-1])
            b[1, 1] = expected_transitions[-1, -1]

            # transitions
            transitions *= 0
            transitions += expected_transitions[:-1, :-1]

            # hmm start
            hmm_start *= 0
            hmm_start += expected_start_count[:-1]
            hmm_start += expected_transitions[-1, :-1]

            # maximization step
            rho /= rho.sum()
            b = (b.T / np.sum(b, axis=1)).T
            transitions = (transitions.T / np.sum(transitions, axis=1)).T
            hmm_start /= hmm_start.sum()

            # fixing start probability
            # hmm start probability
            self.hmm.start_prob[:-1] = hmm_start * rho[0]
            # mmm start probability
            self.hmm.start_prob[-1] = rho[1]

            # fixing transitions
            # hmm-hmm transitions
            self.hmm.transitions[:-1, :-1] = transitions * b[0, 0]
            # mmm-hmm transitions
            self.hmm.transitions[-1, :-1] = hmm_start * b[1, 0]
            # hmm-mmm transitions
            self.hmm.transitions[:-1, -1] = b[0, 1]
            # mmm-mmm transitions
            self.hmm.transitions[-1, -1] = b[1, 1]

            self.hmm.update_hmm()

            score_hmm = self.hmm.log_probability(hmm_seqs)
            score = score_hmm
            count = count + 1 if score - prev_score < stop_threshold else 0
            prev_score = score

            if count >= 2:
                break

        return iteration

    def log_probability(self, seqs):

        # seqs of length = 1
        mmm_seqs = []
        # seqs of length > 1
        hmm_seqs = []

        for seq in seqs:
            if len(seq) == 1:
                mmm_seqs.append(seq)
            else:
                hmm_seqs. append(seq)

        if len(mmm_seqs) == 0:
            if len(hmm_seqs) == 0:
                return 0
            return self.hmm.log_probability(hmm_seqs)

        if len(hmm_seqs) == 0:
            return self.mmm.log_probability(mmm_seqs)

        return self.mmm.log_probability(mmm_seqs) + self.hmm.log_probability(hmm_seqs)

    def predict(self, seqs):
        special_symbol = self.num_states
        gmm_emissions = (self.mmm.get_emissions_matrix().T * self.mmm.get_weights_vector()).T
        mmm_most_probable_path = np.argmax(gmm_emissions, axis=0).tolist()

        cloud_indicator = []
        viterbi = []

        for seq in seqs:
            if len(seq) == 1:
                current_cloud_indicator = [special_symbol]
            else:
                current_cloud_indicator = self.hmm.hmm.predict(seq)

            current_viterbi = []
            for i in range(len(seq)):
                if current_cloud_indicator[i] == self.num_states:
                    current_viterbi.append(mmm_most_probable_path[seq[i]])
                    current_cloud_indicator[i] = 0
                else:
                    current_viterbi.append(current_cloud_indicator[i])
                    current_cloud_indicator[i] = 1

            cloud_indicator.append(current_cloud_indicator)
            viterbi.append(current_viterbi)

        return cloud_indicator, viterbi

    def to_json(self):
        model_dict = {'transitions': self.hmm.transitions.tolist(), 'emissions': self.hmm.emissions.tolist()[:-1],
                      'num_states': self.num_states, 'num_emissions': self.num_emissions,
                      'noise': self.mmm.get_emissions_matrix()[-1].tolist(),
                      'gmmWeights': self.mmm.get_weights_vector().tolist(), 'start_prob': self.hmm.start_prob.tolist()}
        return model_dict


class HMMForSigma(HMM):
    def __init__(self, emissions, sky_emissions):
        self.num_sigs = emissions.shape[0]
        self.weights = np.zeros(self.num_sigs)
        super().__init__(self.num_sigs + 1, emissions.shape[1])

        # fixing parameters
        random_hmm = HMM(self.num_sigs, emissions.shape[1])
        random_b = np.random.random(2)
        random_rho = np.random.random()

        # fixing start probability
        # hmm start probability
        self.start_prob[:-1] = random_hmm.start_prob * random_rho
        # mmm start probability
        self.start_prob[-1] = 1 - random_rho

        # fixing emissions
        self.emissions[:-1] = emissions
        self.emissions[-1] = sky_emissions

        # fixing transitions
        # hmm-hmm transitions
        self.transitions[:-1, :-1] = random_hmm.transitions * (1 - random_b[0])
        # mmm-hmm transitions
        self.transitions[-1, :-1] = random_hmm.start_prob * random_b[1]
        # hmm-mmm transitions
        self.transitions[:-1, -1] = 1 - random_b[0]
        # mmm-mmm transitions
        self.transitions[-1, -1] = 1 - random_b[1]

        self.update_hmm()

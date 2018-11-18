from models.MultinomialMixtureModel import MultinomialMixtureModel
import numpy as np


class MMMFrozenEmissions(MultinomialMixtureModel):
    """
    MMM with frozen emissions
    """
    def __init__(self, emissions):
        num_states, num_emissions = emissions.shape
        super().__init__(num_states, num_emissions, emissions=emissions)

    def maximization_step(self, expected_weights, expected_emissions):
        self.weights = expected_weights

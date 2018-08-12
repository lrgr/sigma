from models.MultinomialMixtureModel import MultinomialMixtureModel
import numpy as np


class MMMFrozenEmissions(MultinomialMixtureModel):
    """
    MMM with frozen emissions
    """
    def __init__(self, emissions):
        num_states, num_emissions = emissions.shape
        self.frozen_emissions = np.log(emissions)
        super().__init__(num_states, num_emissions)
        self.projection_step()

    def projection_step(self):
        self.emissions = self.frozen_emissions

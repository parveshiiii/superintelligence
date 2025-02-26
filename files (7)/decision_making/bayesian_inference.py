import numpy as np

class BayesianInference:
    def __init__(self):
        self.prior = None
        self.likelihood = None

    def set_prior(self, prior: np.ndarray):
        self.prior = prior

    def set_likelihood(self, likelihood: np.ndarray):
        self.likelihood = likelihood

    def infer(self, data: np.ndarray) -> np.ndarray:
        posterior = self.prior * self.likelihood * data
        return posterior / posterior.sum()
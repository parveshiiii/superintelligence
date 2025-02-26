import unittest
import numpy as np
from decision_making.bayesian_inference import BayesianInference

class TestBayesianInference(unittest.TestCase):
    def setUp(self):
        self.model = BayesianInference()
        self.prior = np.array([0.5, 0.5])
        self.likelihood = np.array([0.8, 0.2])
        self.model.set_prior(self.prior)
        self.model.set_likelihood(self.likelihood)

    def test_infer(self):
        data = np.array([1.0, 0.0])
        posterior = self.model.infer(data)
        expected_posterior = np.array([0.8, 0.2])
        np.testing.assert_almost_equal(posterior, expected_posterior)

if __name__ == '__main__':
    unittest.main()
import random
from dnasty.my_utils.config import Config


class MockTrainer:
    def __init__(self, config: Config):
        self.config = config

    def fit(self, model, epochs):
        return self._simulate_forward_pass()

    @staticmethod
    def _simulate_forward_pass():
        """Simulate a forward pass with random metrics for testing."""
        simulated_accuracy = random.uniform(0.5, 1.0)  # Example accuracy range
        return simulated_accuracy

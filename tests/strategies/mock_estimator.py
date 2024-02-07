from mock_trainer import MockTrainer


class MockEarlyStoppingEstimator:
    def __init__(self, config):
        self.fidelity = 1
        self.trainer = MockTrainer(config)

    def fit(self, genome):
        return self.trainer.fit(genome.to_module(), self.fidelity)  # fitness

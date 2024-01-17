import abc
from dnasty.genetics import Genome
from dnasty.defaults.trainer import Trainer


class Estimator(abc.ABC):
    def fit(self, *args, **kwargs):
        raise NotImplementedError(
            "Subclasses must implement __call__()"
        )


class EarlyStoppingEstimator(Estimator):
    def __init__(self, fidelity, config):
        self.fidelity = fidelity
        self.trainer = Trainer(config)

    def fit(self, genome: Genome):
        return self.trainer.fit(genome.to_module(), self.fidelity)  # fitness

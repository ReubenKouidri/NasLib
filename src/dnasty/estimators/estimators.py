from __future__ import annotations

import abc
import random

from dnasty.data.splits import DataModule
from dnasty.defaults.trainer import Trainer
from dnasty.search_space.cbam import Genome
from dnasty.utils import Config


class Estimator(abc.ABC):
    """Maps a genome to a fitness estimate."""

    @abc.abstractmethod
    def fit(self, genome: Genome) -> float: ...


class LowFidelityEstimator(Estimator):
    """Train each genome for a few epochs (``estimator.low_fidelity.epochs``)
    and use the best validation accuracy as its fitness."""

    def __init__(
        self,
        config: Config,
        datamodule: DataModule | None = None,
        trainer: Trainer | None = None,
    ) -> None:
        est_cfg = config.estimator
        section = est_cfg.get("low_fidelity") or est_cfg.get("early_stopping")
        self.fidelity = section.get("epochs", section.get("fidelity", 1))
        if trainer is None:
            if datamodule is None:
                datamodule = DataModule.from_config(config)
            trainer = Trainer(config, datamodule)
        self.trainer = trainer

    def fit(self, genome: Genome) -> float:
        return self.trainer.fit(genome.to_module(), self.fidelity)


# Historical name; it never implemented early stopping.
EarlyStoppingEstimator = LowFidelityEstimator


class MockEstimator(Estimator):
    """Random fitness in [0.5, 1); builds the module to check it expresses."""

    def __init__(self, seed: int = 0, build_module: bool = True) -> None:
        self.rng = random.Random(seed)
        self.build_module = build_module

    def fit(self, genome: Genome) -> float:
        if self.build_module:
            genome.to_module()
        return self.rng.uniform(0.5, 1.0)

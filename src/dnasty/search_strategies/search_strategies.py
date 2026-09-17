from __future__ import annotations

import abc
import copy
import logging

from dnasty.estimators import Estimator, LowFidelityEstimator
from dnasty.search_space.cbam import Genome, is_genome_valid
from dnasty.utils import Config

logger = logging.getLogger(__name__)


class SearchStrategyBase(abc.ABC):
    @abc.abstractmethod
    def generate_population(self, cfg: Config) -> list[Genome]: ...

    @abc.abstractmethod
    def fit(self) -> None: ...


class RandomSearch(SearchStrategyBase):
    """Sample ``population_size`` random genomes per generation and estimate
    each one; the baseline every other strategy must beat."""

    def __init__(self, config: Config, estimator: Estimator | None = None) -> None:
        super().__init__()
        self.search_space_cfg = config.search_space.cbam
        self.perf_metric = config.perf_metric
        self.population_size = config.population_size
        self.generations = config.generations
        self.num_classes = config.num_classes
        self.image_dims = config.get("image_dims", 128)
        self.in_channels = config.get("in_channels", 1)
        self.estimator = estimator or LowFidelityEstimator(config)
        self.history: list[Genome] = []
        self.evaluated: list[Genome] = []
        self.population: list[Genome] = self.generate_population(self.search_space_cfg)

    def fit(self) -> None:
        """Samples the search space for the given number of generations."""
        for i in range(self.generations):
            logger.info("Fitting generation %d/%d", i + 1, self.generations)
            self._step()

    def _step(self) -> None:
        for i, genome in enumerate(self.population):
            logger.info("Fitting genome %d/%d", i + 1, len(self.population))
            genome.fitness = self.estimator.fit(genome)

        self.evaluated.extend(copy.deepcopy(g) for g in self.population)
        self.history.append(copy.deepcopy(self.generation_best))
        self.population = self.generate_population(self.search_space_cfg)

    def generate_population(self, cfg: Config) -> list[Genome]:
        """Randomly initialise a valid population from the search space."""
        population: list[Genome] = []
        while len(population) < self.population_size:
            genome = Genome.from_random(
                cfg,
                num_classes=self.num_classes,
                image_dims=self.image_dims,
                in_channels=self.in_channels,
            )
            if is_genome_valid(genome, cfg):
                population.append(genome)
        return population

    @property
    def generation_best(self) -> Genome:
        """Best genome in the current generation."""
        return max(self.population, key=lambda x: x.fitness)

    @property
    def fittest_genome(self) -> Genome:
        """Best genome seen over all generations."""
        if not self.history:
            raise ValueError("No history - call .fit() first")
        return max(self.history, key=lambda x: x.fitness)

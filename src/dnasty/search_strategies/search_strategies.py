"""Search strategies.

Every strategy consumes the same evaluation budget,
``population_size * generations`` calls to the estimator, so that results are
comparable across strategies. ``RandomSearch`` is the baseline that any other
strategy has to beat (Li & Talwalkar 2019; Yang et al. 2020).
"""

from __future__ import annotations

import abc
import copy
import logging
import random
from collections import deque

from dnasty.estimators import Estimator, LowFidelityEstimator
from dnasty.search_space.cbam import Genome, is_genome_valid
from dnasty.utils import Config

from .operators import crossover_genomes, mutate_genome

logger = logging.getLogger(__name__)


class SearchStrategyBase(abc.ABC):
    """Bookkeeping shared by all strategies.

    Attributes:
        evaluated: a copy of every evaluated genome, in evaluation order.
        history: the best genome after each generation; ``generations``
            entries once ``fit`` has run.
        budget: total number of estimator calls, ``population_size *
            generations``.
    """

    name = "base"

    def __init__(self, config: Config, estimator: Estimator | None = None) -> None:
        self.config = config
        self.search_space_cfg = config.search_space.cbam
        self.population_size = int(config.population_size)
        self.generations = int(config.generations)
        self.num_classes = config.num_classes
        self.image_dims = config.get("image_dims", 128)
        self.in_channels = config.get("in_channels", 1)
        self.estimator = estimator or LowFidelityEstimator(config)
        self.evaluated: list[Genome] = []
        self.history: list[Genome] = []

    @property
    def budget(self) -> int:
        return self.population_size * self.generations

    def random_genome(self) -> Genome:
        """A random genome that satisfies the search-space constraints."""
        while True:
            genome = Genome.from_random(
                self.search_space_cfg,
                num_classes=self.num_classes,
                image_dims=self.image_dims,
                in_channels=self.in_channels,
            )
            if is_genome_valid(genome, self.search_space_cfg):
                return genome

    def generate_population(self, cfg: Config | None = None) -> list[Genome]:
        """``population_size`` random valid genomes."""
        return [self.random_genome() for _ in range(self.population_size)]

    def _evaluate(self, genome: Genome) -> Genome:
        genome.fitness = float(self.estimator.fit(genome))
        self.evaluated.append(copy.deepcopy(genome))
        return genome

    @property
    def best_so_far(self) -> Genome:
        if not self.evaluated:
            raise ValueError("Nothing evaluated yet - call .fit() first")
        return max(self.evaluated, key=lambda g: g.fitness)

    @property
    def fittest_genome(self) -> Genome:
        """Best genome seen over all generations."""
        if not self.history:
            raise ValueError("No history - call .fit() first")
        return max(self.history, key=lambda g: g.fitness)

    @abc.abstractmethod
    def fit(self) -> None: ...


class RandomSearch(SearchStrategyBase):
    """Sample ``population_size`` random genomes per generation and estimate
    each one; the baseline every other strategy must beat."""

    name = "random"

    def __init__(self, config: Config, estimator: Estimator | None = None) -> None:
        super().__init__(config, estimator)
        self.population: list[Genome] = self.generate_population()

    def fit(self) -> None:
        """Samples the search space for the given number of generations."""
        for i in range(self.generations):
            logger.info("Fitting generation %d/%d", i + 1, self.generations)
            self._step()

    def _step(self) -> None:
        for i, genome in enumerate(self.population):
            logger.info("Fitting genome %d/%d", i + 1, len(self.population))
            self._evaluate(genome)
        self.history.append(copy.deepcopy(self.generation_best))
        self.population = self.generate_population()

    @property
    def generation_best(self) -> Genome:
        """Best genome in the current generation."""
        return max(self.population, key=lambda x: x.fitness)


class RegularizedEvolution(SearchStrategyBase):
    """Aging evolution (Real et al. 2019).

    The population is a queue of ``population_size`` genomes. Each step
    tournament-selects a parent from ``sample_size`` random members, makes one
    child by mutation (or, with probability ``crossover_prob``, by one-point
    crossover with a second tournament winner), evaluates it, appends it and
    discards the *oldest* member. Removing by age rather than by fitness keeps
    the population moving and is what "regularised" refers to.

    Config (``evolution`` section, all optional)::

        evolution:
          sample_size: 3          # tournament size; default round(P / 4), min 2
          crossover_prob: 0.0     # 0 = mutation only, as in the paper
          mutation:               # relative weights of the mutation operators
            hparam: 0.7
            insert_conv: 0.15
            delete_conv: 0.15

    ``history`` receives the best-so-far genome after every
    ``population_size`` evaluations, so it has ``generations`` entries like
    ``RandomSearch``.
    """

    name = "regularized_evolution"

    def __init__(self, config: Config, estimator: Estimator | None = None) -> None:
        super().__init__(config, estimator)
        evo = config.get("evolution") or Config({})
        default_sample = max(2, round(0.25 * self.population_size))
        self.sample_size = min(
            self.population_size, int(evo.get("sample_size", default_sample))
        )
        self.crossover_prob = float(evo.get("crossover_prob", 0.0))
        mutation = evo.get("mutation")
        self.mutation_weights: dict[str, float] | None = (
            mutation.to_dict() if isinstance(mutation, Config) else mutation
        )
        self.population: deque[Genome] = deque()
        self.fallbacks = 0  # children that had to be random genomes

    def fit(self) -> None:
        while len(self.population) < self.population_size:
            self.population.append(self._evaluate(self.random_genome()))
        evaluations = len(self.population)
        self._snapshot(evaluations)

        while evaluations < self.budget:
            child = self._make_child()
            self._evaluate(child)
            self.population.append(child)
            self.population.popleft()
            evaluations += 1
            if evaluations % self.population_size == 0 or evaluations == self.budget:
                self._snapshot(evaluations)

    def _snapshot(self, evaluations: int) -> None:
        best = self.best_so_far
        logger.info(
            "Evaluations %d/%d, best fitness %.4f",
            evaluations,
            self.budget,
            best.fitness,
        )
        self.history.append(copy.deepcopy(best))

    def _tournament(self) -> Genome:
        sample = random.sample(list(self.population), self.sample_size)
        return max(sample, key=lambda g: g.fitness)

    def _make_child(self, max_tries: int = 5) -> Genome:
        for _ in range(max_tries):
            parent = self._tournament()
            if self.crossover_prob > 0 and random.random() < self.crossover_prob:
                child = crossover_genomes(
                    parent, self._tournament(), self.search_space_cfg
                )
                if child is not None and random.random() < 0.5:
                    child = (
                        mutate_genome(
                            child, self.search_space_cfg, self.mutation_weights
                        )
                        or child
                    )
            else:
                child = mutate_genome(
                    parent, self.search_space_cfg, self.mutation_weights
                )
            if child is not None:
                return child
        self.fallbacks += 1
        logger.warning(
            "No valid child after %d tries; sampling a random genome", max_tries
        )
        return self.random_genome()


STRATEGIES: dict[str, type[SearchStrategyBase]] = {
    RandomSearch.name: RandomSearch,
    RegularizedEvolution.name: RegularizedEvolution,
    "re": RegularizedEvolution,
}


def build_strategy(
    config: Config,
    estimator: Estimator | None = None,
    name: str | None = None,
) -> SearchStrategyBase:
    """Instantiate the strategy named by ``name`` or ``config.search_strategy``."""
    key = (name or config.get("search_strategy", "random")).lower()
    try:
        cls = STRATEGIES[key]
    except KeyError:
        raise ValueError(
            f"Unknown search strategy '{key}'; choose from {sorted(STRATEGIES)}"
        ) from None
    return cls(config, estimator=estimator)

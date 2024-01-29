import abc
import copy
import logging
from dnasty.genetics import *
from dnasty.estimators import EarlyStoppingEstimator
from dnasty.my_utils import Config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%m-%Y %H:%M:%S')


class SearchStrategyBase(abc.ABC):

    def generate_population(self, config) -> list[Genome]:
        raise NotImplementedError(
            "Subclasses must implement generate_population()"
        )

    def fit(self):
        raise NotImplementedError(
            "Subclasses must implement fit()"
        )


class RandomSearch(SearchStrategyBase):

    def __init__(self,
                 config: Config):
        """
        Initialize a random search optimizer.

        Args:
            config (Config): configuration for NAS
        """
        super(RandomSearch, self).__init__()
        self.search_space_cfg = config.search_space.cbam
        self.perf_metric = config.perf_metric
        self.population_size = config.population_size
        self.generations = config.generations
        self.num_classes = config.num_classes
        self.sampled: set[Genome] = set()
        self.history = []
        self.estimator = EarlyStoppingEstimator(config=config)
        self.population: list[Genome] = self.generate_population(
            config.search_space.cbam)

    def fit(self) -> None:
        """Samples the search space for the given number of generations"""
        for i, generation in enumerate(range(self.generations)):
            logging.log(logging.INFO,
                        f"Fitting generation: {i+1}/{self.generations}")
            self._step()

    def _step(self) -> None:
        """Sample and estimate performance of a new genome"""
        for i, genome in enumerate(self.population):
            logging.log(logging.INFO,
                        f"Fitting genome: {i+1}/{len(self.population)}")
            genome.fitness = self.estimator.fit(genome)

        self.sampled.update(
            {copy.deepcopy(genome) for genome in self.population})
        self.history.append(
            copy.deepcopy(max(self.population, key=lambda x: x.fitness)))
        self.population = self.generate_population(self.search_space_cfg)

    def generate_population(self, cfg: Config) -> list[Genome]:
        """
        Randomly initialise the population from the search space

        Args:
            cfg (Config): the config for the search space
                          e.g. cfg = nas.search_space.cbam

        Returns:
            list[Genome]: the initialised population of genomes
        """
        population = []
        while len(population) < self.population_size:
            genome = Genome.from_random(cfg)
            if is_genome_valid(genome, cfg):
                population.append(genome)
        return population

    @property
    def generation_best(self) -> Genome:
        """Returns best genome in the current generation"""
        return max(self.population, key=lambda x: x.fitness)

    @property
    def fittest_genome(self) -> Genome:
        """Returns the sampled architecture with the lowest validation error"""
        if len(self.history) == 0:
            raise ValueError("No history - call .fit() first")
        return max(self.history, key=lambda x: x.fitness)

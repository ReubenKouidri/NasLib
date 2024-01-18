import abc
import copy
import random
from dnasty.genetics import *
from dnasty.estimators import EarlyStoppingEstimator
from dnasty.my_utils import Config


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
            config: total Config object
        """
        super(RandomSearch, self).__init__()
        self.perf_metric = config.perf_metric
        self.population_size = config.population_size
        self.population: list[Genome] = self._generate_population(
            config.search_space.cbam)
        self.sampled: set[Genome] = set()
        self.history = []
        self.estimator = EarlyStoppingEstimator(config=config)
        self.generations = config.generations

    def fit(self):
        for generation in range(self.generations):
            self._step()

    def _step(self):
        """
        Sample a new architecture to train.
        """
        population_best = []
        for genome in self.population:
            genome.fitness = self.estimator.fit(genome)
            self.sampled.add(copy.deepcopy(genome))
            population_best.append(copy.deepcopy(genome))

        self.history.append(
            copy.deepcopy(max(population_best, key=lambda x: x.fitness)))

    def _generate_population(self, cfg: Config) -> list[Genome]:
        """
        Randomly initialise the population from the search space
        cfg is the config for the search space e.g. cfg = nas.search_space.cbam
        """
        population = []
        while len(population) < self.population_size:
            genes = []
            num_cbam_genes = random.randint(1, cfg.cells)
            for _ in range(num_cbam_genes):
                cbam_gene = CBAMGene.from_random()
                num_conv = random.randint(1, cfg.conv)
                genes.extend(
                    ConvBlock2dGene.from_random() for _ in range(num_conv))
                genes.append(MaxPool2dGene.from_random())
                genes.append(cbam_gene)

            genes.append(FlattenGene())
            genes.extend(
                LinearBlockGene.from_random() for _ in range(cfg.linear))
            genome = Genome.from_sequence(genes)
            print(genome)

            # TODO: lazy way to constrain num_params - needs investigation
            if 2 < genome.outdims < 50:
                population.append(Genome.from_sequence(genes))

        return population

    @property
    def generation_best(self):
        """Returns best genome in the current generation"""
        return max(self.population, key=lambda x: x.fitness)

    @property
    def fittest_genome(self):
        """Returns the sampled architecture with the lowest validation error"""
        if len(self.history) == 0:
            return None
        return max(self.history, key=lambda x: x.fitness)

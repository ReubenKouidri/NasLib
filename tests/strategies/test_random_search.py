import unittest
from dnasty.my_utils.config import Config
from dnasty.search_strategies import RandomSearch
import dnasty
from mock_estimator import MockEarlyStoppingEstimator


class TestRandomSearch(unittest.TestCase):

    def setUp(self):
        self.config = dnasty.my_utils.config.Config.from_file(
            "../../dnasty/my_utils/config.json")

        # Patch the dependent modules/methods used by RandomSearch
        self.strategy = RandomSearch(self.config.nas)
        self.strategy.estimator = MockEarlyStoppingEstimator(self.config)

    def test_initialization(self):
        """Test RandomSearch initialization with config."""
        self.assertEqual(self.strategy.population_size,
                         self.config.nas.population_size)

    def test_fit_generates_history(self):
        """Test that the fit method generates a history of genomes."""
        self.strategy.fit()
        self.assertTrue(len(self.strategy.history) > 0)

    def test_ranking_of_genomes(self):
        """Test that genomes are ranked based on fitness."""
        self.strategy.fit()
        self.assertEqual(self.strategy.generation_best,
                         self.strategy.population[0])
        genomes = sorted(self.strategy.history, key=lambda x: x.fitness,
                         reverse=True)

        self.assertEqual(len(genomes), self.config.nas.population_size)
        self.assertEqual(genomes[0], self.strategy.fittest_genome)
        self.assertEqual(genomes[0].fitness,
                         self.strategy.fittest_genome.fitness)


if __name__ == '__main__':
    unittest.main()

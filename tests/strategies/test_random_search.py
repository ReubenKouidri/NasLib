import unittest
from dnasty.my_utils.config import Config
from dnasty.search_strategies import RandomSearch


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.config = Config.from_file("../../dnasty/my_utils/config.json")
        self.nas = self.config.nas
        self.strat = RandomSearch(self.nas)

    def test__generate_population(self):
        self.assertEqual(len(self.strat.population), self.nas.population_size)

    def test_fit(self):
        self.strat._step()
        history = self.strat.history


if __name__ == '__main__':
    unittest.main()

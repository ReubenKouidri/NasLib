from dnasty.estimators import MockEstimator
from dnasty.search_strategies import RandomSearch
from dnasty.utils import seed_everything


def test_initialization(tiny_config):
    strategy = RandomSearch(tiny_config, estimator=MockEstimator())
    assert strategy.population_size == tiny_config.population_size
    assert len(strategy.population) == tiny_config.population_size


def test_fit_generates_history(tiny_config):
    strategy = RandomSearch(tiny_config, estimator=MockEstimator())
    strategy.fit()
    assert len(strategy.history) == tiny_config.generations
    assert len(strategy.evaluated) == (
        tiny_config.generations * tiny_config.population_size
    )


def test_ranking_of_genomes(tiny_config):
    strategy = RandomSearch(tiny_config, estimator=MockEstimator())
    strategy.fit()
    genomes = sorted(strategy.history, key=lambda x: x.fitness, reverse=True)
    assert genomes[0] is strategy.fittest_genome
    assert all(g.fitness >= 0.5 for g in strategy.evaluated)


def test_search_is_reproducible(tiny_config):
    def run():
        seed_everything(123)
        s = RandomSearch(tiny_config, estimator=MockEstimator(seed=123))
        s.fit()
        return [g.to_dict() for g in s.history]

    assert run() == run()

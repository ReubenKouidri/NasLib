import statistics

import pytest

from dnasty.estimators import MockEstimator, SyntheticEstimator
from dnasty.search_strategies import (
    RandomSearch,
    RegularizedEvolution,
    build_strategy,
)
from dnasty.utils import Config, seed_everything


def test_budget_and_history(tiny_config):
    strategy = RegularizedEvolution(tiny_config, estimator=MockEstimator())
    strategy.fit()
    assert len(strategy.evaluated) == strategy.budget
    assert len(strategy.history) == tiny_config.generations
    assert len(strategy.population) == tiny_config.population_size
    assert strategy.fittest_genome.fitness == max(g.fitness for g in strategy.evaluated)


def test_history_is_monotone(tiny_config):
    strategy = RegularizedEvolution(tiny_config, estimator=MockEstimator())
    strategy.fit()
    fitness = [g.fitness for g in strategy.history]
    assert fitness == sorted(fitness)


def test_reproducible(tiny_config):
    def run():
        seed_everything(11)
        s = RegularizedEvolution(tiny_config, estimator=SyntheticEstimator())
        s.fit()
        return [g.arch_key() for g in s.evaluated]

    assert run() == run()


def test_build_strategy_registry(tiny_config):
    assert isinstance(build_strategy(tiny_config, MockEstimator()), RandomSearch)
    assert isinstance(
        build_strategy(tiny_config, MockEstimator(), name="regularized_evolution"),
        RegularizedEvolution,
    )
    assert isinstance(
        build_strategy(tiny_config, MockEstimator(), name="re"), RegularizedEvolution
    )
    with pytest.raises(ValueError, match="Unknown search strategy"):
        build_strategy(tiny_config, MockEstimator(), name="nope")


def test_crossover_config_is_used(tiny_config):
    cfg = tiny_config.to_dict()
    cfg["evolution"] = {
        "sample_size": 2,
        "crossover_prob": 0.5,
        "mutation": {"hparam": 1},
    }
    strategy = RegularizedEvolution(Config(cfg), estimator=MockEstimator())
    assert strategy.crossover_prob == 0.5
    assert strategy.sample_size == 2
    assert strategy.mutation_weights == {"hparam": 1}
    strategy.fit()
    assert len(strategy.evaluated) == strategy.budget


@pytest.mark.slow
def test_evolution_beats_random_on_synthetic_landscape(tiny_config):
    """On a landscape with structure, evolution should climb faster than
    random sampling at an equal budget (mean over seeds)."""
    cfg = tiny_config.to_dict()
    cfg.update(population_size=8, generations=8)
    config = Config(cfg)

    def best(strategy_cls, seed):
        seed_everything(seed)
        s = strategy_cls(config, estimator=SyntheticEstimator(seed=0))
        s.fit()
        return s.best_so_far.fitness

    seeds = [0, 1, 2]
    evolution = statistics.fmean(best(RegularizedEvolution, s) for s in seeds)
    random_search = statistics.fmean(best(RandomSearch, s) for s in seeds)
    assert evolution > random_search

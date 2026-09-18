import copy

from dnasty.estimators import CachedEstimator, MockEstimator, SyntheticEstimator
from dnasty.search_space.cbam import Genome, is_genome_valid
from dnasty.utils import seed_everything


def valid_genome(cfg):
    while True:
        genome = Genome.from_random(cfg, num_classes=9)
        if is_genome_valid(genome, cfg):
            return genome


def test_synthetic_is_deterministic_and_bounded(tiny_config):
    seed_everything(0)
    space = tiny_config.search_space.cbam
    estimator = SyntheticEstimator(seed=3)
    for _ in range(10):
        genome = valid_genome(space)
        first = estimator.fit(genome)
        assert 0.0 <= first <= 1.0
        assert estimator.fit(copy.deepcopy(genome)) == first
        assert abs(first - estimator.score(genome)) <= 5 * estimator.noise


def test_synthetic_landscape_seed_changes_noise_only(tiny_config):
    seed_everything(0)
    genome = valid_genome(tiny_config.search_space.cbam)
    a, b = SyntheticEstimator(seed=0), SyntheticEstimator(seed=1)
    assert a.score(genome) == b.score(genome)
    assert abs(a.fit(genome) - b.fit(genome)) <= 10 * a.noise


def test_cached_estimator_counts_hits(tiny_config):
    seed_everything(0)
    genome = valid_genome(tiny_config.search_space.cbam)
    cached = CachedEstimator(MockEstimator(build_module=False))
    first = cached.fit(genome)
    assert cached.fit(copy.deepcopy(genome)) == first
    assert (cached.hits, cached.misses) == (1, 1)
    other = valid_genome(tiny_config.search_space.cbam)
    cached.fit(other)
    assert cached.misses == 2

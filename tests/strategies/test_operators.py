import copy

import pytest

from dnasty.search_space.cbam import (
    CBAMGene,
    ConvBlock2dGene,
    Genome,
    LinearBlockGene,
    is_genome_valid,
)
from dnasty.search_strategies import crossover_genomes, mutate_genome
from dnasty.search_strategies.operators import (
    mutate_delete_conv,
    mutate_hparam,
    mutate_insert_conv,
)
from dnasty.utils import seed_everything


def random_valid_genome(cfg):
    while True:
        genome = Genome.from_random(cfg, num_classes=9)
        if is_genome_valid(genome, cfg):
            return genome


def conv_count(genome):
    return sum(isinstance(g, ConvBlock2dGene) for g in genome.genes.values())


@pytest.fixture
def space(tiny_config):
    return tiny_config.search_space.cbam


def test_mutate_genome_returns_valid_different_child(space):
    seed_everything(1)
    for _ in range(30):
        parent = random_valid_genome(space)
        before = copy.deepcopy(parent.to_dict())
        child = mutate_genome(parent, space)
        assert child is not None
        assert child.arch_key() != parent.arch_key()
        assert is_genome_valid(child, space)
        assert child.fitness == 0.0
        assert parent.to_dict() == before, "parent must not be modified"


def test_mutate_child_expresses_and_runs(space):
    import torch

    seed_everything(2)
    parent = random_valid_genome(space)
    child = mutate_genome(parent, space)
    out = child.to_module()(torch.randn(2, 1, 128, 128))
    assert out.shape == (2, 9)


def test_hparam_mutation_never_touches_output_layer(space):
    seed_everything(3)
    parent = random_valid_genome(space)
    for _ in range(20):
        child = mutate_hparam(parent)
        child.sync_genes()
        last = list(child.genes.values())[-1]
        assert isinstance(last, LinearBlockGene)
        assert last.out_features == 9
        assert last.activation is None


def test_insert_conv_respects_cell_limit(space):
    seed_everything(4)
    parent = random_valid_genome(space)
    child = mutate_insert_conv(parent, max_convs_per_cell=space.conv)
    if child is None:
        # every cell already holds the maximum number of conv blocks
        assert conv_count(parent) == space.conv * sum(
            isinstance(g, CBAMGene) for g in parent.genes.values()
        )
    else:
        assert conv_count(child) == conv_count(parent) + 1
        child.sync_genes()  # channels chain through the new block
        assert child.outdims <= parent.outdims


def test_delete_conv_keeps_one_per_cell(space):
    seed_everything(5)
    for _ in range(10):
        parent = random_valid_genome(space)
        child = mutate_delete_conv(parent)
        if child is None:
            assert conv_count(parent) == sum(
                isinstance(g, CBAMGene) for g in parent.genes.values()
            )
        else:
            assert conv_count(child) == conv_count(parent) - 1
            assert isinstance(next(iter(child.genes.values())), ConvBlock2dGene)


def test_mutation_weights_select_operator(space):
    seed_everything(6)
    parent = random_valid_genome(space)
    child = mutate_genome(
        parent, space, weights={"hparam": 1, "insert_conv": 0, "delete_conv": 0}
    )
    assert child is not None
    assert conv_count(child) == conv_count(parent)


def test_crossover_child_is_valid_and_mixes_parents(space):
    seed_everything(7)
    produced = 0
    for _ in range(20):
        a = random_valid_genome(space)
        b = random_valid_genome(space)
        child = crossover_genomes(a, b, space)
        if child is None:
            continue
        produced += 1
        assert is_genome_valid(child, space)
        assert child.arch_key() not in {a.arch_key(), b.arch_key()}
        assert isinstance(next(iter(child.genes.values())), ConvBlock2dGene)
        # the head (last linear block) comes from the second parent
        head_child = list(child.genes.values())[-2]
        head_b = list(b.genes.values())[-2]
        assert head_child.exons["dropout"] == head_b.exons["dropout"]
    assert produced > 0


def test_operators_are_reproducible(space):
    def run():
        seed_everything(8)
        parent = random_valid_genome(space)
        child = mutate_genome(parent, space)
        return child.arch_key()

    assert run() == run()

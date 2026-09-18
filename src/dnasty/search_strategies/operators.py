"""Variation operators for the ``cbam`` chain genome.

Mutation applies exactly one change per child, as in regularised evolution
(Real et al. 2019):

- ``hparam``: step one hyper-parameter of one gene to a neighbouring allowed
  value (channels, kernel size, pooling window, attention ratio, linear width,
  dropout);
- ``insert_conv``: add a random conv block to a cell that has room;
- ``delete_conv``: remove a conv block from a cell that keeps at least one.

Crossover is one-point at a cell boundary (CNN-GA, Sun et al. 2020): the
leading cells of the first parent followed by the trailing cells and the
classifier head of the second.

All operators return a *new*, valid genome (validated with
:func:`is_genome_valid`, which may shrink the first linear block to respect
the parameter budget) or ``None`` when no valid child was found. Randomness
comes from the ``random`` module, so ``seed_everything`` makes them
reproducible.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Mapping

from dnasty.search_space.cbam import (
    CBAMGene,
    ConvBlock2dGene,
    FlattenGene,
    GeneBase,
    Genome,
    is_genome_valid,
)
from dnasty.utils import Config

MUTATION_OPS = ("hparam", "insert_conv", "delete_conv")
DEFAULT_MUTATION_WEIGHTS: dict[str, float] = {
    "hparam": 0.7,
    "insert_conv": 0.15,
    "delete_conv": 0.15,
}


def _conv_indices(genes: list[GeneBase]) -> list[int]:
    return [i for i, g in enumerate(genes) if isinstance(g, ConvBlock2dGene)]


def _cell_convs(genes: list[GeneBase], index: int) -> list[int]:
    """Indices of the contiguous run of conv genes containing ``index``."""
    low = index
    while low > 0 and isinstance(genes[low - 1], ConvBlock2dGene):
        low -= 1
    high = index
    while high + 1 < len(genes) and isinstance(genes[high + 1], ConvBlock2dGene):
        high += 1
    return list(range(low, high + 1))


def _head_start(genes: list[GeneBase]) -> int:
    for i, gene in enumerate(genes):
        if isinstance(gene, FlattenGene):
            return i
    raise ValueError("Genome has no FlattenGene; cannot locate the classifier head")


def _cell_starts(genes: list[GeneBase]) -> list[int]:
    """Indices where a cell after the first begins (a conv that follows a CBAM)."""
    return [
        i
        for i in range(1, len(genes))
        if isinstance(genes[i], ConvBlock2dGene) and isinstance(genes[i - 1], CBAMGene)
    ]


def mutate_hparam(genome: Genome) -> Genome | None:
    """Copy ``genome`` and step one hyper-parameter of one random gene.

    The output layer is excluded: ``sync_genes`` fixes its width, activation
    and dropout, so mutating it would be a no-op.
    """
    child = copy.deepcopy(genome)
    genes = list(child.genes.values())
    candidates = [g for g in genes[:-1] if hasattr(g, "mutate")]
    if not candidates:
        return None
    random.choice(candidates).mutate()
    return child


def mutate_insert_conv(genome: Genome, max_convs_per_cell: int) -> Genome | None:
    """Insert a random conv block after an existing one in a cell with room."""
    genes = genome.to_sequence()
    options = [
        i
        for i in _conv_indices(genes)
        if len(_cell_convs(genes, i)) < max_convs_per_cell
    ]
    if not options:
        return None
    index = random.choice(options)
    genes.insert(index + 1, ConvBlock2dGene.from_random())
    return genome.spawn(genes)


def mutate_delete_conv(genome: Genome) -> Genome | None:
    """Delete a conv block from a cell that has more than one."""
    genes = genome.to_sequence()
    options = [i for i in _conv_indices(genes) if len(_cell_convs(genes, i)) > 1]
    if not options:
        return None
    del genes[random.choice(options)]
    return genome.spawn(genes)


def mutate_genome(
    genome: Genome,
    space_cfg: Config,
    weights: Mapping[str, float] | None = None,
    max_tries: int = 20,
) -> Genome | None:
    """One random mutation of ``genome``; returns a valid, different child.

    Args:
        genome: parent; never modified.
        space_cfg: the ``search_space.cbam`` section (``conv`` bounds the
            number of conv blocks per cell; the validity limits come from it).
        weights: relative weights of :data:`MUTATION_OPS`; missing entries use
            :data:`DEFAULT_MUTATION_WEIGHTS`.
        max_tries: attempts before giving up and returning ``None``.
    """
    merged = {**DEFAULT_MUTATION_WEIGHTS, **dict(weights or {})}
    ops = list(MUTATION_OPS)
    op_weights = [float(merged[op]) for op in ops]
    parent_key = genome.arch_key()

    for _ in range(max_tries):
        op = random.choices(ops, weights=op_weights, k=1)[0]
        if op == "hparam":
            child = mutate_hparam(genome)
        elif op == "insert_conv":
            child = mutate_insert_conv(genome, int(space_cfg.conv))
        else:
            child = mutate_delete_conv(genome)
        if child is None:
            continue
        child.sync_genes()
        if is_genome_valid(child, space_cfg) and child.arch_key() != parent_key:
            child.fitness = 0.0
            return child
    return None


def crossover_genomes(
    first: Genome,
    second: Genome,
    space_cfg: Config,
    max_tries: int = 10,
) -> Genome | None:
    """One-point crossover at cell boundaries.

    The child is ``first[:i] + second[j:]`` where ``i`` is a cell start (or
    the head start) of ``first`` and ``j`` one of ``second``; the classifier
    head therefore always comes from ``second``. Children with no cell, more
    than ``space_cfg.cells`` cells, or identical to a parent are rejected.
    """
    genes_a = first.to_sequence()
    genes_b = second.to_sequence()
    cuts_a = _cell_starts(genes_a) + [_head_start(genes_a)]
    cuts_b = _cell_starts(genes_b) + [_head_start(genes_b)]
    max_cells = int(space_cfg.cells)
    parent_keys = {first.arch_key(), second.arch_key()}

    for _ in range(max_tries):
        i = random.choice(cuts_a)
        j = random.choice(cuts_b)
        genes = [copy.deepcopy(g) for g in genes_a[:i] + genes_b[j:]]
        cells = sum(isinstance(g, CBAMGene) for g in genes)
        if cells == 0 or cells > max_cells:
            continue
        child = first.spawn(genes)
        if is_genome_valid(child, space_cfg) and child.arch_key() not in parent_keys:
            child.fitness = 0.0
            return child
    return None

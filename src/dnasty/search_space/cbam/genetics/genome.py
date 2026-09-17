from __future__ import annotations

import collections.abc as abc
import copy
from collections import OrderedDict
from typing import Any

import torch.nn as nn

from dnasty.search_space.common import (
    ConvBlock2dGene,
    GeneBase,
    LinearBlockGene,
    MaxPool2dGene,
)
from dnasty.utils import Config

from .genes import CBAMGene, create_gene_sequence


def adjust_linear_genes(genome: Genome, max_num_params: int) -> bool:
    """
    Adjusts the number of out_features in the first linear gene so the genome
    fits within ``max_num_params``.

    Returns:
        bool: True if the adjustment is successful, otherwise False.

    Derivation:
        N := in_features to first linear layer
           = genome.outdims ** 2 * out_chans | out_dims = size of the 2D
            feature map following all feature extraction and pooling layers.

        L := num parameters in linear blocks following feature extraction
        if num linear blocks == 2
            L = (N + c)n | c = num_classes,
                           n = neurons in first lin layer
        C := num parameters in all feature extraction blocks (conv, cbam, ...)
        T = L + C = total params
        M := max allowed params in model

        If T < M we do not need to adjust linear genes, otherwise we adjust
        the number of neurons in the linear blocks in the following way:

        T -> T' | T' = C + L' | L' = (N + c)n', n' = adjusted number of neurons

        Constraint on C: with n_min = c, Cmax = M - (N + c)c; C > Cmax => invalid.
        Constraint on L: n' = floor((M - C) / (N + c)), additionally n' <= N.
    """
    linear_gene = genome.genes["LinearBlockGene0"]
    numerator = max_num_params - (genome.num_params - genome.linear_params)
    denominator = linear_gene.in_features + genome.num_classes + 1
    new_out_features = min(int(numerator // denominator), linear_gene.in_features)

    linear_gene.out_features = new_out_features
    genome.sync_genes()
    if genome.num_params > max_num_params:
        return False
    return new_out_features > genome.num_classes


def _exceeds_param_limit(genome: Genome, cfg: Config) -> bool:
    """See 'adjust_linear_genes' for logic on 'limit'."""
    limit = cfg.max_num_params - genome.num_classes * (
        genome.linear_block_input_size + genome.num_classes
    )
    return (genome.num_params - genome.linear_params) > limit


def is_genome_valid(genome: Genome, cfg: Config) -> bool:
    """
    Checks if the given genome is valid for the NAS search space.
    1. check the outdims match:
        outdims < min_outdims => image squashed too small
        outdims > max_outdims => too few extraction/pooling layers
    2. check the number of parameters:
        constrain the number of parameters for minimal search and
        training speed.
    3. check if it's possible to constrain (might not be possible):
        if so adjust the number of parameters in the linear block

    Note: step 3 modifies the genome in place.
    """
    if not cfg.min_outdims < genome.outdims < cfg.max_outdims:
        return False
    if _exceeds_param_limit(genome, cfg):
        return False
    if genome.num_params > cfg.max_num_params:
        return adjust_linear_genes(genome, cfg.max_num_params)
    return True


class Genome:
    """
    An ordered sequence of genes that expresses to an ``nn.Sequential``.

    Attributes:
        genes (OrderedDict): gene name -> gene.
        fitness (float): estimated performance, used to rank genomes.
        num_classes (int): output size of the final linear layer.
        image_dims (int): side length of the (square) input image.
        in_channels (int): channels of the input image.
    """

    def __init__(
        self,
        genes: OrderedDict | None = None,
        num_classes: int = 9,
        image_dims: int = 128,
        in_channels: int = 1,
    ) -> None:
        if genes is not None and not isinstance(genes, OrderedDict):
            raise TypeError(
                f"Genes must be an OrderedDict, not {type(genes).__name__}."
            )
        self.genes = genes if genes else OrderedDict()
        self.fitness = 0.0
        self.num_classes = num_classes
        self.image_dims = image_dims
        self.in_channels = in_channels
        if self.genes:
            self.sync_genes()

    @classmethod
    def from_random(cls, cfg: Config, **kwargs: Any) -> Genome:
        """Random genome from the search-space config (e.g. ``nas.search_space.cbam``)."""
        return cls.from_sequence(create_gene_sequence(cfg), **kwargs)

    @classmethod
    def from_sequence(cls, genes: abc.MutableSequence, **kwargs: Any) -> Genome:
        """Constructs a Genome from a sequence of genes, naming them ``<Type><i>``."""
        new_genes: OrderedDict = OrderedDict()
        mapping: dict[str, int] = {}
        for gene in genes:
            gene_name = type(gene).__name__
            mapping.setdefault(gene_name, -1)
            mapping[gene_name] += 1
            new_genes[f"{gene_name}{mapping[gene_name]}"] = gene
        return cls(new_genes, **kwargs)

    def to_module(self) -> nn.Sequential:
        """Synchronise gene shapes and express the genome as ``nn.Sequential``."""
        self.sync_genes()
        return nn.Sequential(*(gene.to_module() for gene in self.genes.values()))

    def sync_genes(self) -> None:
        """
        Propagate in_/out_channels and in_/out_features along the gene chain
        and configure the output layer (``num_classes`` logits, no activation,
        no dropout).
        """
        genes_iter = iter(self.genes.values())
        first_gene = next(genes_iter)

        if not isinstance(first_gene, ConvBlock2dGene):
            raise ValueError("First gene must be a ConvBlock2dGene.")

        first_gene.in_channels = self.in_channels
        prev_significant_gene: GeneBase = first_gene
        first_linear = True

        for next_gene in genes_iter:
            if hasattr(next_gene, "in_channels"):
                next_gene.in_channels = prev_significant_gene.out_channels
                prev_significant_gene = next_gene

            if isinstance(next_gene, LinearBlockGene):
                if first_linear:
                    next_gene.__setattr__(
                        "in_features",
                        self.outdims**2 * prev_significant_gene.out_channels,
                        True,
                    )
                    first_linear = False
                else:
                    next_gene.in_features = prev_significant_gene.out_features
                prev_significant_gene = next_gene

            if isinstance(next_gene, CBAMGene):
                next_gene.sync()

        last_gene = next(reversed(self.genes.values()))
        if isinstance(last_gene, LinearBlockGene):
            # Raw logits: the loss applies log-softmax.
            last_gene.activation = None
            last_gene.dropout = False
            last_gene.out_features = self.num_classes

    @property
    def outdims(self) -> int:
        """Side length of the feature map after all conv and pooling genes."""

        def reduce_func(d: int, f: int, p: int, s: int) -> int:
            return 1 + (d - f + 2 * p) // s

        dims = self.image_dims
        for gene in self.genes.values():
            if isinstance(gene, ConvBlock2dGene):
                dims = reduce_func(dims, gene.kernel_size, 0, 1)
            elif isinstance(gene, MaxPool2dGene):
                dims = reduce_func(dims, gene.kernel_size, 0, gene.kernel_size)
        return dims

    @property
    def num_params(self) -> int:
        return sum(
            g.num_params for g in self.genes.values() if hasattr(g, "num_params")
        )

    @property
    def linear_params(self) -> int:
        return sum(
            gene.num_params
            for gene in self.genes.values()
            if isinstance(gene, LinearBlockGene)
        )

    @property
    def linear_block_input_size(self) -> int:
        return self.genes["LinearBlockGene0"].in_features

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable description (for run logs)."""
        return {
            "fitness": self.fitness,
            "num_params": self.num_params,
            "genes": {name: dict(gene.exons) for name, gene in self.genes.items()},
        }

    def __deepcopy__(self, memo: dict) -> Genome:
        if id(self) in memo:
            return memo[id(self)]
        cls = self.__class__
        new_genome = cls.__new__(cls)
        memo[id(self)] = new_genome
        new_genome.fitness = self.fitness
        new_genome.num_classes = self.num_classes
        new_genome.image_dims = self.image_dims
        new_genome.in_channels = self.in_channels
        new_genome.genes = copy.deepcopy(self.genes, memo)
        return new_genome

    def __len__(self) -> int:
        return len(self.genes)

    def __getitem__(self, item):
        return self.genes[item]

    def __le__(self, other) -> bool:
        return self.fitness <= other.fitness

    def __ge__(self, other) -> bool:
        return self.fitness >= other.fitness

    def __lt__(self, other) -> bool:
        return self.fitness < other.fitness

    def __gt__(self, other) -> bool:
        return self.fitness > other.fitness

    def __repr__(self) -> str:
        gene_summary = ",\n".join(
            f"  {name}: {gene.exons}" for name, gene in self.genes.items()
        )
        return f"Genome(\n{gene_summary},\n  Fitness: {self.fitness}\n)"

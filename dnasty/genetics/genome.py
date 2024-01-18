from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
import collections.abc as abc
import copy
import torch.nn as nn
from dnasty import genetics


class Genome:
    """
    Represents a genome storing a sequence of genes used to create a
    neural network.

    Attributes:
        genes (OrderedDict): An ordered dictionary of genes.
        fitness (float): A prediction for the performance of the model,
        used to rank models.
        image_dims (int): Image dimensions (default is 128).
        __num_classes (int): Number of classes for classification (default is
        9).
    """

    __num_classes = 9
    image_dims = 128

    def __init__(self, genes: OrderedDict = None):
        if not isinstance(genes, OrderedDict):
            raise TypeError(
                f"Genes must be an OrderedDict, not {type(genes).__name__}.")

        self.genes = genes if genes else OrderedDict()
        self.fitness = 0.0

    def append(self, gene: genetics.GeneBase) -> None:
        self.genes.update({type(gene).__name__: gene})

    @classmethod
    def from_sequence(cls, genes: abc.MutableSequence) -> Genome:
        """Constructs a Genome instance from a mutable sequence of genes."""
        new_genes = OrderedDict()
        mapping = {}
        for gene in genes:
            gene_name = gene.__name__
            mapping.setdefault(gene_name, -1)
            mapping[gene_name] += 1
            unique_name = f"{gene_name}{mapping[gene_name]}"
            new_genes[unique_name] = gene
        return cls(new_genes)

    @property
    def genes_iter(self) -> Iterator[genetics.GeneBase]:
        """
        Provides an iterator over the genes in the genome.

        Returns:
            An iterator over the gene objects.
        """
        return iter(self.genes.values())

    def to_module(self) -> nn.Sequential:
        """
        Converts the genome into a PyTorch Sequential model.

        This method synchronizes the genes and then creates corresponding
        PyTorch modules for each gene in the sequence.

        Returns:
            nn.Sequential: The constructed PyTorch Sequential model.
        """
        self._sync_genes()
        modules = [gene.to_module() for gene in self.genes.values()]
        return nn.Sequential(*modules)

    def _sync_genes(self) -> None:
        """
        Synchronizes in_/out_channels and in_/out_features for each gene.

        This method ensures that the dimensions match between consecutive
        layers in the generated neural network.
        """
        genes_iter = self.genes_iter
        first_gene = next(genes_iter)

        if not isinstance(first_gene, genetics.ConvBlock2dGene) or \
                not hasattr(first_gene, 'in_channels'):
            raise ValueError(
                "First gene must be a ConvBlock2dGene with an out_channels "
                "attribute.")

        first_gene.in_channels = 1
        prev_significant_gene = first_gene
        first_linear = True

        for next_gene in genes_iter:
            if hasattr(next_gene, 'in_channels'):
                next_gene.in_channels = prev_significant_gene.out_channels
                prev_significant_gene = next_gene

            if isinstance(next_gene, genetics.LinearBlockGene):
                if first_linear:
                    next_gene.__setattr__(
                        "in_features",
                        self.outdims ** 2 * prev_significant_gene.out_channels,
                        True)
                    first_linear = False
                else:
                    next_gene.in_features = prev_significant_gene.out_features
                prev_significant_gene = next_gene

        # Special handling for the last
        last_gene = next(reversed(self.genes.values()))
        if isinstance(last_gene, genetics.LinearBlockGene):
            last_gene.activation = "Softmax"
            last_gene.dropout = False
            last_gene.out_features = self.__num_classes

    @property
    def outdims(self) -> int:
        """
        Calculates the output dimensions of the neural network
        based on the genes.

        This method computes the resulting dimensions after applying all
        convolutional and pooling layers in the gene sequence.

        reduce_func: A utility function to calculate the reduced dimension
        after applying a conv or pooling layer with given parameters:
            d (int): Original dimension.
            f (int): Filter size.
            p (int): Padding.
            s (int): Stride.

        Returns:
            int: The output dimensions.
        """

        def reduce_func(d, f, p, s):
            return 1 + (d - f + 2 * p) // s

        dims = self.image_dims
        for gene in self.genes.values():
            if isinstance(gene, genetics.ConvBlock2dGene):
                dims = reduce_func(dims, gene.kernel_size, 0, 1)
            elif isinstance(gene, genetics.MaxPool2dGene):
                dims = reduce_func(dims, gene.kernel_size, 0,
                                   gene.kernel_size)
        return dims

    def __deepcopy__(self, memo) -> Genome:
        """
        Creates a deep copy of the Genome instance.

        Args:
            memo (dict): A dictionary for memoization during deep copy.

        Returns:
            Genome: A deep copy of the Genome
        """
        if id(self) in memo:
            return memo[id(self)]
        cls = self.__class__
        new_genome = cls.__new__(cls)
        memo[id(self)] = new_genome
        new_genome.fitness = copy.deepcopy(self.fitness, memo)
        new_genome.genes = copy.deepcopy(self.genes, memo)
        return new_genome

    def __len__(self) -> int:
        return len(self.genes)

    def __le__(self, other) -> bool:
        return self.fitness <= other.fitness

    def __ge__(self, other) -> bool:
        return self.fitness >= other.fitness

    def __lt__(self, other) -> bool:
        return self.fitness < other.fitness

    def __gt__(self, other) -> bool:
        return self.fitness > other.fitness

    def __repr__(self):
        gene_summary = ',\n'.join(
            [f"  {name}: {gene.exons}" for name, gene in self.genes.items()])
        return (f"Genome(\n"
                f"{gene_summary},\n"
                f"  Fitness: {self.fitness}\n)")

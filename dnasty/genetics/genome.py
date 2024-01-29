from __future__ import annotations
from collections import OrderedDict
import collections.abc as abc
import copy
import torch.nn as nn
from dnasty import genetics
from dnasty.my_utils import Config


def adjust_linear_genes(genome: Genome, max_num_params: int) -> bool:
    """
    Adjusts the number of out_features in the linear gene based on the given
    genome and maximum number of parameters.

    Args:
        genome (Genome): The genome object to be adjusted.
        max_num_params (int): The maximum number of allowed model parameters.

    Returns:
        bool: True if the adjustment is successful, otherwise False.

    Example:
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

        Before performing this adjustment we need to ask whether it will
        result in a new genome with T' > M.

        Consider the following case:

        Under what constraints will T' be < M?
        Finding constraint on C:
        If we adjust L to the lowest possible value Lmin, what is Cmax such
        that C > Cmax => T' > M?
            T' < M
            => C + L' < M
            => C < M - L'
            => Cmax = M - Lmin
            Lmin = (N + c)n_min | n_min = min allowed number of neurons = c
            * we have previously restricted n to be >= c
            => Lmin = (N + c)c
            => Cmax = M - (N + c)c
            Therefore, C > Cmax => T' > M => genome invalid!

        Finding constraint on L:
        Given T > M and C < Cmax, what L' gives T' = M?
            L' = (N + c)n' | n' = adjusted number of neurons
            T' = C + L' = C + (N + c)n'
            T' = M
            => C + (N + c)n' = M
            => n' = floor((M - C) / (N + c)) | type(n') = int

        We apply the additional constraint n' <= N
        This adds human bias to the search by following 'best practices',
        but can be removed later.
    """
    linear_gene = genome.genes["LinearBlockGene0"]
    numerator = max_num_params - (genome.num_params - genome.linear_params)
    denominator = linear_gene.in_features + Genome.num_classes + 1
    new_out_features = min(int(numerator // denominator),
                           linear_gene.in_features)

    linear_gene.out_features = new_out_features
    genome.sync_genes()
    if genome.num_params > max_num_params:
        return False
    return new_out_features > Genome.num_classes


def _exceeds_param_limit(genome: Genome, cfg: Config) -> bool:
    """
    See 'adjust_linear_genes' for logic on 'limit'.
    """
    limit = cfg.max_num_params - Genome.num_classes * (
            genome.linear_block_input_size + Genome.num_classes)
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

    Args:
        genome (Genome): The genome object to be checked.
        cfg (Config): The configuration object.

    Returns:
        bool: True if the genome is valid, otherwise False.
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
    Represents a genome storing a sequence of genes used to create a
    neural network.

    Attributes:
        genes (OrderedDict): An ordered dictionary of genes.
        fitness (float): A prediction for the performance of the model,
        used to rank models.
        image_dims (int): Image dimensions (default is 128).
        num_classes (int): Number of classes for classification (default is
        9).
    """

    num_classes = 9
    image_dims = 128

    def __init__(self, genes: OrderedDict = None):
        if not isinstance(genes, OrderedDict):
            raise TypeError(
                f"Genes must be an OrderedDict, not {type(genes).__name__}.")

        self.genes = genes if genes else OrderedDict()
        self.fitness = 0.0
        self.sync_genes()

    @classmethod
    def from_random(cls, cfg: Config) -> Genome:
        return cls.from_sequence(genetics.create_gene_sequence(cfg))

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

    def append(self, gene: genetics.GeneBase) -> None:
        self.genes.update({type(gene).__name__: gene})

    def to_module(self) -> nn.Sequential:
        """
        Converts the genome into a PyTorch Sequential model.

        This method synchronizes the genes and then creates corresponding
        PyTorch modules for each gene in the sequence.

        Returns:
            nn.Sequential: The constructed PyTorch Sequential model.
        """
        self.sync_genes()
        modules = [gene.to_module() for gene in self.genes.values()]
        return nn.Sequential(*modules)

    def sync_genes(self) -> None:
        """
        Synchronizes in_/out_channels and in_/out_features for each gene.

        This method ensures that the dimensions match between consecutive
        layers in the generated neural network.
        """
        genes_iter = iter(self.genes.values())
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

            if isinstance(next_gene, genetics.CBAMGene):
                next_gene.sync()

        # Special handling for the last
        last_gene = next(reversed(self.genes.values()))
        if isinstance(last_gene, genetics.LinearBlockGene):
            last_gene.activation = "Softmax"
            last_gene.dropout = False
            last_gene.out_features = self.num_classes

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

    @property
    def num_params(self) -> int:
        s = 0
        for g in self.genes.values():
            if hasattr(g, "num_params"):
                s += g.num_params
        return s

    @property
    def linear_params(self) -> int:
        return sum(gene.num_params for gene in self.genes.values() if
                   isinstance(gene, genetics.LinearBlockGene))

    @property
    def linear_block_input_size(self) -> int:
        return self.genes["LinearBlockGene0"].in_features

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

    def __repr__(self):
        gene_summary = ',\n'.join(
            [f"  {name}: {gene.exons}" for name, gene in self.genes.items()])
        return (f"Genome(\n"
                f"{gene_summary},\n"
                f"  Fitness: {self.fitness}\n)")

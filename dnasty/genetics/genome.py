from __future__ import annotations

import copy
import numpy as np
from dnasty import genes


# TODO:
#   - This is old code! Refactor genome using the new Gene interface
#   - Add tests for this class
#   - Crossover will be implemented in strategy.py

class Genome:
    @staticmethod
    def reduce_func(d: int, f: int, p: int, s: int) -> int:
        """
        Calculate the output size.

        :param d: Dimension size.
        :param f: Filter size.
        :param p: Padding.
        :param s: Stride.
        :return: The calculated output size.
        """
        return 1 + (d - f + 2 * p) // s

    def __deepcopy__(self, memo):
        cls = self.__class__
        genes_copy = copy.deepcopy(self.genes, memo)
        new_genome = cls(genes_copy)
        memo[id(self)] = new_genome
        return new_genome

    def __len__(self):
        return len(self.genes)

    def __init__(self, genes):
        self.genes = list(genes)
        self.fitness = 0.0
        self.image_dims = 128

    @property
    def outdims(self):
        dims = self.image_dims
        for gene in self.genes:
            if isinstance(gene, genes.ConvBlock2dGene):
                dims = self.reduce_func(dims, gene.kernel_size, 0, 1)
            elif isinstance(gene, genes.MaxPool2dGene):
                dims = self.reduce_func(dims, gene.kernel_size, 0,
                                        gene.kernel_size)
        return dims

    def _sync_layers(self):
        prev_out_channels = 1
        for gene in self.genes:
            if isinstance(gene, genes.ConvBlock2dGene):
                in_chans = gene.in_channels
                gene.in_channels = prev_out_channels if (in_chans !=
                                                         prev_out_channels) \
                    else in_chans
                prev_out_channels = gene.out_channels

    def crossover(self, other: Genome, generation: int, operator: str):
        if operator == 'mean':
            keys = list(self.genes.exons.keys())
            sequence_1 = np.array(list(self.sequence.values()))
            sequence_2 = np.array(list(other.chromosomes.values()))
            new_sequence = sequence_1 + ((np.exp(generation / self.alpha) *
                                          (
                                                  sequence_2 - sequence_1))
                                         // 2).astype(
                int)
            new_sequence = {key: val for key, val in zip(keys, new_sequence)}
            self.sequence = new_sequence
            # print("AFTER mix", self.chromosomes)

        else:
            print("WARNING: gene1 mixing not implemented!")

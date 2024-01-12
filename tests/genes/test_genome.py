import unittest
import copy

from dnasty.genes.genes import ConvBlock2dGene, MaxPool2dGene
from dnasty.genes.genome import Genome


class TestGenome(unittest.TestCase):
    def setUp(self):
        """
        TODO:
            - setup genome using every gene type
            - remove old test code
            - use new Genome class once implemented with new Gene interface
        """

        self.conv = ConvBlock2dGene(1, 32, 10)
        self.conv2 = ConvBlock2dGene(1, 50, 10)
        self.mp = MaxPool2dGene(2, 2)
        self.genes = [self.conv, self.conv2, self.mp]
        self.genome = Genome(self.genes)

    def test_init(self):
        self.assertEqual(self.conv, self.genome.genes[0])
        self.assertEqual(self.conv2, self.genome.genes[1])
        self.assertEqual(self.mp, self.genome.genes[2])
        self.assertEqual(self.genome.fitness, 0.0)
        self.assertIsInstance(self.genome.genes, list)

    def test_crossover(self):
        pass

    def test_len(self):
        self.assertEqual(len(self.genome), 3)

    def test_sync(self):
        self.assertNotEqual(self.conv.out_channels, self.conv2.in_channels)
        self.genome._sync_layers()
        self.assertEqual(self.conv.out_channels, self.conv2.in_channels)

    def test_outdims(self):
        self.assertEqual(self.genome.outdims, 55)

    def test_deepcopy(self):
        genome_copy = copy.deepcopy(self.genome)
        self.assertNotEqual(genome_copy, self.genome)
        self.assertNotEqual(genome_copy.genes, self.genome.genes)
        for g1, g2 in zip(genome_copy.genes, self.genome.genes):
            self.assertNotEqual(g1, g2)  # check the list of genes are not
            # at the same memory address
            for v1, v2 in zip(g1.exons.values(), g2.exons.values()):
                self.assertEqual(v1, v2)  # check the values are the same
            for k1, k2 in zip(g1.exons.keys(), g2.exons.keys()):
                self.assertEqual(k1, k2)  # check the keys are the same


if __name__ == "__main__":
    unittest.main()

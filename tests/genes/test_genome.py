import unittest
import copy
import torch

from dnasty.genetics import *


class TestGenome(unittest.TestCase):
    def setUp(self):
        self.conv1 = ConvBlock2dGene(1, 31, 10)
        self.conv2 = ConvBlock2dGene(1, 50, 11)
        self.conv3 = ConvBlock2dGene(1, 127, 12)
        self.mp = MaxPool2dGene(2, 2)
        self.cbam = CBAMGene(ChannelAttentionGene(se_ratio=5),
                             SpatialAttentionGene(kernel_size=3))
        self.linear1 = LinearBlockGene(100, 200)
        self.linear2 = LinearBlockGene(300, 400)
        self.flatten = FlattenGene()

        self.genes = [
            self.conv1,
            self.conv2,
            self.conv3,
            self.mp,
            self.cbam,
            self.flatten,
            self.linear1,
            self.linear2
        ]

        self.genome = Genome.from_sequence(self.genes)
        self.genome.__num_classes = 9
        self.img_size = 128
        # 128 -> 119 -> 109 -> 98 -> 49
        # 49^2 * CHANNELS
        self.outdims = (49 ** 2) * 128

    def test_init(self):
        genes_iter = iter(
            self.genome.genes.values())  # Create an iterator for genes

        first_gene = next(genes_iter)
        self.assertEqual(self.conv1, first_gene)
        self.assertEqual(first_gene.in_channels, 1)
        self.assertEqual(first_gene.out_channels, 32)
        self.assertEqual(first_gene.kernel_size, 10)

        second_gene = next(genes_iter)
        self.assertEqual(self.conv2, second_gene)
        self.assertEqual(second_gene.in_channels, 1)
        self.assertEqual(second_gene.out_channels, 64)
        self.assertEqual(second_gene.kernel_size, 11)

        third_gene = next(genes_iter)
        self.assertEqual(self.conv3, third_gene)
        self.assertEqual(third_gene.in_channels, 1)
        self.assertEqual(third_gene.out_channels, 128)
        self.assertEqual(third_gene.kernel_size, 12)

        fourth_gene = next(genes_iter)
        self.assertEqual(self.mp, fourth_gene)
        self.assertEqual(fourth_gene.kernel_size, 2)
        self.assertEqual(fourth_gene.stride, 2)

        fifth_gene = next(genes_iter)
        self.assertEqual(self.cbam, fifth_gene)
        self.assertEqual(fifth_gene.se_ratio, 5)
        self.assertEqual(fifth_gene.kernel_size, 3)
        self.assertEqual(fifth_gene.in_channels, 1)  # default

        sixtht_gene = next(genes_iter)
        self.assertEqual(self.flatten, sixtht_gene)

        seventh_gene = next(genes_iter)
        self.assertEqual(self.linear1, seventh_gene)
        self.assertEqual(seventh_gene.in_features, 100)
        self.assertEqual(seventh_gene.out_features, 200)

        eigth_gene = next(genes_iter)
        self.assertEqual(self.linear2, eigth_gene)
        self.assertEqual(eigth_gene.in_features, 300)
        self.assertEqual(eigth_gene.out_features, 400)

        self.assertEqual(len(self.genome), len(self.genes))
        self.assertEqual(self.genome.fitness, 0.0)

    def test_len(self):
        self.assertEqual(len(self.genome), len(self.genes))

    def test_sync(self):
        self.genome._sync_genes()
        genes_iter = iter(self.genome.genes.values())
        # check conv genes are synchronised
        g1 = next(genes_iter)
        self.assertEqual(g1.in_channels, 1)
        self.assertEqual(g1.out_channels, 32)
        g2 = next(genes_iter)
        self.assertEqual(g2.in_channels, 32)
        self.assertEqual(g2.out_channels, 64)
        g3 = next(genes_iter)
        self.assertEqual(g3.in_channels, 64)
        self.assertEqual(g3.out_channels, 128)
        # check cbam synchronised
        _ = next(genes_iter)  # mp
        g5 = next(genes_iter)
        self.assertEqual(g5.in_channels, 128)
        self.assertEqual(g5.out_channels, 128)
        # check linear genes are synchronised
        _ = next(genes_iter)  # flatten
        g7 = next(genes_iter)
        self.assertEqual(g7.in_features, self.outdims)
        self.assertEqual(g7.out_features, 200)

        g8 = next(genes_iter)
        self.assertEqual(g8.in_features, 200)
        self.assertEqual(g8.out_features, 9)

    def test_deepcopy(self):
        genome_copy = copy.deepcopy(self.genome)
        self.assertNotEqual(genome_copy, self.genome)
        self.assertNotEqual(genome_copy.genes, self.genome.genes)

        for g1, g2 in zip(genome_copy.genes.values(),
                          self.genome.genes.values()):
            self.assertNotEqual(g1, g2, f"Genes {g1} and {g2} reference "
                                        f"the same object")
            for v1, v2 in zip(g1.exons.values(), g2.exons.values()):
                self.assertEqual(v1, v2)  # check the values are the same
            for k1, k2 in zip(g1.exons.keys(), g2.exons.keys()):
                self.assertEqual(k1, k2)  # check the keys are the same

    def test_to_module(self):
        self.genome._sync_genes()
        model = self.genome.to_module()
        self.assertIsInstance(model, torch.nn.Sequential)
        x = torch.randn(16, 1, 128, 128)
        out = model(x)
        self.assertEqual(out.shape, torch.Size([16, 9]))


if __name__ == "__main__":
    unittest.main()

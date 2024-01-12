import unittest
import torch
import copy
from dnasty.genes.genes import ChannelAttentionGene, SpatialAttentionGene
from dnasty.genes.genes import CBAMGene
from dnasty.components import CBAM


class TestCBAMGene(unittest.TestCase):
    def setUp(self):
        self.channel_gene = ChannelAttentionGene(in_channels=16, se_ratio=4)
        self.spatial_gene = SpatialAttentionGene(kernel_size=3)
        self.cbam_gene = CBAMGene(channel_gene=self.channel_gene,
                                  spatial_gene=self.spatial_gene)

    def test_initialization(self):
        self.assertEqual(self.cbam_gene.in_channels, 16)
        self.assertEqual(self.cbam_gene.se_ratio, 4)
        self.assertEqual(self.cbam_gene.kernel_size, 3)

    def test_express(self):
        cbam_module = self.cbam_gene.to_module()
        self.assertIsInstance(cbam_module, CBAM)
        x = torch.randn(12, 16, 32, 32)
        y = cbam_module(x)
        self.assertEqual(y.shape, x.shape)

    def test_deepcopy(self):
        copied_gene = copy.deepcopy(self.cbam_gene)

        self.assertIsNot(copied_gene, self.cbam_gene,
                         "Deep copy resulted in the same object reference.")

        self.assertIsNot(copied_gene.channel_gene, self.channel_gene,
                         "Deep copy resulted in the same object reference.")

        self.assertIsNot(copied_gene.spatial_gene, self.spatial_gene,
                         "Deep copy resulted in the same object reference.")

        self.assertEqual(copied_gene.exons, self.cbam_gene.exons,
                         "Attributes of the deep copied object do not match "
                         "the original.")

        self.assertEqual(copied_gene.channel_gene.exons,
                         self.cbam_gene.channel_gene.exons,
                         "Attributes of the deep copied object do not match "
                         "the original.")

        self.assertEqual(copied_gene.spatial_gene.exons,
                         self.cbam_gene.spatial_gene.exons,
                         "Attributes of the deep copied object do not match "
                         "the original.")


if __name__ == '__main__':
    unittest.main()

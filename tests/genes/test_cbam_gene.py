import unittest
import torch
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


if __name__ == '__main__':
    unittest.main()

import unittest
import random
import torch
from dnasty.genes.genes import ChannelAttentionGene
from dnasty.components import ChannelAttention


class TestChannelAttentionGene(unittest.TestCase):
    def setUp(self):
        self.se_ratio = 4
        self.in_channels = 2
        self.gene = ChannelAttentionGene(se_ratio=self.se_ratio,
                                         in_channels=self.in_channels)

    def test_initialization(self):
        self.assertEqual(self.gene.se_ratio, 4)

        # Test initialization with invalid se_ratio (outside allowed range)
        with self.assertWarns(UserWarning):
            ChannelAttentionGene(se_ratio=1)

    def test_mutate(self):
        random.seed(0)  # Set seed for reproducibility
        self.gene.mutate()
        # Check if mutation occurred within allowed range
        self.assertIn(self.gene.se_ratio, ChannelAttentionGene.allowed_values)

    def test_express(self):
        # TODO: explicit test for forward meth with known input and output
        module = self.gene.express()
        self.assertIsInstance(module, ChannelAttention)
        self.assertEqual(module.mlp[1].in_features, self.gene.in_channels)
        self.assertEqual(module.mlp[1].out_features,
                         max((self.gene.in_channels // 4), 1))
        self.assertEqual(module.mlp[3].in_features,
                         max((self.gene.in_channels // 4), 1))
        test_input = torch.randn(16, self.in_channels, 32, 32)
        test_output = module(test_input)
        self.assertEqual(test_output.shape, test_input.shape)

    def test_validate_feature(self):
        adjusted_value = self.gene._validate_feature("se_ratio", 1,
                                                     self.gene.allowed_values)
        self.assertIn(adjusted_value, self.gene.allowed_values)


if __name__ == '__main__':
    unittest.main()

import unittest
import random
import torch
import copy
from dnasty.genetics import ChannelAttentionGene, ChannelAttention


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
        self.assertIn(self.gene.se_ratio,
                      ChannelAttentionGene._feature_ranges["se_ratio"])

    def test_express(self):
        # TODO: explicit test for forward meth with known input and output
        module = self.gene.to_module()
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
        self.gene.se_ratio = 1
        self.assertIn(self.gene.se_ratio, self.gene._feature_ranges["se_ratio"])

    def test_deepcopy(self):
        copied_gene = copy.deepcopy(self.gene)

        self.assertIsNot(copied_gene, self.gene,
                         "Deep copy resulted in the same object reference.")

        # valid for non-composite genetics (do not contain other genetics)
        self.assertEqual(copied_gene.__dict__, self.gene.__dict__,
                         "Attributes of the deep copied object do not match "
                         "the original.")


if __name__ == '__main__':
    unittest.main()

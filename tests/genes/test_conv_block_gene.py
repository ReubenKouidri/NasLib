import torch
from torch import nn
import copy

from dnasty.genes import ConvBlock2dGene
import unittest


class TestConvBlockGene(unittest.TestCase):
    def setUp(self):
        self.gene = ConvBlock2dGene(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            activation="ReLU"
        )

    def test_init(self):
        self.assertIsInstance(self.gene, ConvBlock2dGene)
        self.assertEqual(self.gene.in_channels, 32)
        self.assertEqual(self.gene.out_channels, 64)
        self.assertEqual(self.gene.kernel_size, 5)
        self.assertEqual(self.gene.activation, "ReLU")
        self.assertTrue(self.gene.batch_norm)

    def test_getattr(self):
        with self.assertRaises(AttributeError):
            _ = self.gene.invalid

    def test_express(self):
        module = self.gene.to_module()
        self.assertIsInstance(module, nn.Sequential)
        self.assertIsInstance(module[0], nn.Conv2d)
        self.assertEqual(module[0].in_channels, 32)
        self.assertEqual(module[0].out_channels, 64)
        self.assertEqual(module[0].kernel_size, (5, 5))
        self.assertEqual(module[0].stride, (1, 1))
        self.assertEqual(module[0].padding, (0, 0))
        self.assertEqual(module[0].groups, 1)
        self.assertEqual(module[0].bias.shape, torch.Size([64]))
        self.assertIsInstance(module[1], nn.ReLU)
        self.assertIsInstance(module[2], nn.BatchNorm2d)

    def test_forward(self):
        module = self.gene.to_module()
        x = torch.randn(2, 32, 32, 32)
        out = module(x)
        self.assertEqual(out.shape, torch.Size([2, 64, 28, 28]))

    def test_deepcopy(self):
        copied_gene = copy.deepcopy(self.gene)

        self.assertIsNot(copied_gene, self.gene,
                         "Deep copy resulted in the same object reference.")

        # this is valid for ConvBlock2dGene as it is not composite (c.f. CBAM)
        self.assertEqual(copied_gene.__dict__, self.gene.__dict__,
                         "Attributes of the deep copied object do not match "
                         "the original.")

    def test_len(self):
        self.assertEqual(len(self.gene), 5)


if __name__ == "__main__":
    unittest.main()

import torch

from dnasty.genes.genes import SpatialAttentionGene
import unittest


class TestSpatialAttentionGene(unittest.TestCase):
    def setUp(self):
        self.gene = SpatialAttentionGene(7)

    def test_init(self):
        self.assertIsInstance(self.gene, SpatialAttentionGene)
        self.assertEqual(self.gene.kernel_size, 7)

    def test_getattr(self):
        with self.assertRaises(AttributeError):
            _ = self.gene.invalid

    def test_express(self):
        module = self.gene.express()
        x = torch.randn(16, 64, 32, 32)
        y = module(x)
        self.assertEqual(y.shape, torch.Size((16, 64, 32, 32)))

    def test_len(self):
        self.assertEqual(len(self.gene), 1)


if __name__ == "__main__":
    unittest.main()

import torch
import copy
import unittest
from dnasty.search_space.cbam import SpatialAttentionGene


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
        module = self.gene.to_module()
        x = torch.randn(16, 64, 32, 32)
        y = module(x)
        self.assertEqual(y.shape, torch.Size((16, 64, 32, 32)))

    def test_len(self):
        self.assertEqual(len(self.gene), 1)

    def test_deepcopy(self):
        copied_gene = copy.deepcopy(self.gene)

        self.assertIsNot(copied_gene, self.gene,
                         "Deep copy resulted in the same object reference.")

        self.assertEqual(copied_gene.__dict__, self.gene.__dict__,
                         "Attributes of the deep copied object do not match "
                         "the original.")


if __name__ == "__main__":
    unittest.main()

import torch

from dnasty.components import Flatten
import unittest


class TestSpatialAttentionGene(unittest.TestCase):
    def test_forward(self):
        module = Flatten()
        x = torch.randn(4, 2, 5, 5)
        y = module(x)
        self.assertEqual(y.shape[1], 2 * 5 * 5)


if __name__ == "__main__":
    unittest.main()

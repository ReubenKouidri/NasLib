import unittest
import warnings

import torch.nn as nn
from dnasty.genes.genes import MaxPool2dGene


class TestMaxPool2dGene(unittest.TestCase):
    def setUp(self):
        self.gene1 = MaxPool2dGene(kernel_size=2)
        self.gene2 = MaxPool2dGene(kernel_size=(2, 2))

    def test_init(self):
        self.assertIsInstance(self.gene1, MaxPool2dGene)
        self.assertIsInstance(self.gene2, MaxPool2dGene)
        self.assertEqual(self.gene1.kernel_size, 2)
        self.assertEqual(self.gene2.kernel_size, [2, 2])
        self.assertEqual(self.gene1.stride, 2)
        self.assertEqual(self.gene2.stride, [2, 2])
        self.assertEqual(len(self.gene1), 2)
        self.assertEqual(len(self.gene2), 2)

    def test_express_method(self):
        max_pool_layer = self.gene1.express()
        self.assertIsInstance(max_pool_layer, nn.MaxPool2d)
        self.assertEqual(max_pool_layer.kernel_size, 2)
        self.assertEqual(max_pool_layer.stride, 2)

    def test_invalid_integer_kernel_size(self):
        kernel_size = max(MaxPool2dGene.allowed_values) + 1
        with self.assertWarns(UserWarning):
            g = MaxPool2dGene(kernel_size=kernel_size)

        self.assertEqual(g.kernel_size, max(MaxPool2dGene.allowed_values))
        self.assertEqual(g.stride, g.kernel_size)

    def test_invalid_sequence_kernel_size(self):
        max_val = max(MaxPool2dGene.allowed_values)
        min_val = min(MaxPool2dGene.allowed_values)
        kernel_size = (min_val - 1, max_val + 1)

        with self.assertWarns(UserWarning):
            g = MaxPool2dGene(kernel_size=kernel_size)

        self.assertEqual(g.kernel_size, g.stride)
        self.assertIsInstance(g.kernel_size, list)
        self.assertEqual(g.kernel_size[0], min(MaxPool2dGene.allowed_values))
        self.assertEqual(g.kernel_size[1], max(MaxPool2dGene.allowed_values))

    def test_mutate(self):
        # TODO: Implement when method implemented
        pass


if __name__ == '__main__':
    unittest.main()

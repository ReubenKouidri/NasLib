import unittest
import copy
import torch.nn as nn
from dnasty.search_space.common import MaxPool2dGene


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
        max_pool_layer = self.gene1.to_module()
        self.assertIsInstance(max_pool_layer, nn.MaxPool2d)
        self.assertEqual(max_pool_layer.kernel_size, 2)
        self.assertEqual(max_pool_layer.stride, 2)

    def test_invalid_integer_kernel_size(self):
        kernel_size = 0
        with self.assertWarns(UserWarning):
            g = MaxPool2dGene(kernel_size=kernel_size)

        self.assertEqual(g.kernel_size,
                         min(MaxPool2dGene._feature_ranges["kernel_size"]))
        self.assertEqual(g.stride, g.kernel_size)

    def test_invalid_sequence_kernel_size(self):
        max_val = max(MaxPool2dGene._feature_ranges["kernel_size"])
        min_val = min(MaxPool2dGene._feature_ranges["kernel_size"])
        kernel_size = (min_val - 1, max_val + 1)

        with self.assertWarns(UserWarning):
            g = MaxPool2dGene(kernel_size=kernel_size)

        self.assertEqual(g.kernel_size, g.stride)
        self.assertIsInstance(g.kernel_size, list)
        self.assertEqual(g.kernel_size[0], min_val)
        self.assertEqual(g.kernel_size[1], max_val)

    def test_deepcopy(self):
        copied_gene = copy.deepcopy(self.gene1)

        self.assertIsNot(copied_gene, self.gene1,
                         "Deep copy resulted in the same object reference.")

        # valid for non-composite genetics (do not contain other genetics)
        self.assertEqual(copied_gene.__dict__, self.gene1.__dict__,
                         "Attributes of the deep copied object do not match "
                         "the original.")


if __name__ == '__main__':
    unittest.main()

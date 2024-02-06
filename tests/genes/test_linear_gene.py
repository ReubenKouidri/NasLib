import torch.nn as nn
import unittest
import copy
from dnasty.search_space.common import LinearBlockGene


class TestLinearGene(unittest.TestCase):
    def setUp(self):
        self.gene = LinearBlockGene(in_features=10,
                                    out_features=20,
                                    activation='ReLU',
                                    dropout=True)

    def _test_within_ranges(self, gene):
        self.assertIn(gene.in_features,
                      LinearBlockGene._feature_ranges["in_features"])
        self.assertIn(gene.out_features,
                      LinearBlockGene._feature_ranges["out_features"])

    def test_init(self):
        self.assertIsInstance(self.gene, LinearBlockGene)
        self.assertEqual(self.gene.in_features, 10)
        self.assertEqual(self.gene.out_features, 20)
        self.assertEqual(self.gene.activation, 'ReLU')
        self.assertTrue(self.gene.dropout)

    def test_invalid_activation(self):
        with self.assertRaises(ValueError):
            LinearBlockGene(in_features=10, out_features=20,
                            activation='InvalidActivation', dropout=True)

    def test_express_method(self):
        module = self.gene.to_module()
        self.assertIsInstance(module, nn.Sequential)

        # Check if the correct modules are added
        self.assertIsInstance(module[0], nn.Linear)
        self.assertEqual(module[0].in_features, 10)
        self.assertEqual(module[0].out_features, 20)
        self.assertIsInstance(module[1], nn.ReLU)
        self.assertIsInstance(module[2], nn.Dropout)

    def test_mutate(self):
        pre_drop = self.gene.dropout
        self.gene.mutate()
        self._test_within_ranges(self.gene)
        self.assertNotEqual(self.gene.dropout, pre_drop)

    def test_len(self):
        self.assertEqual(len(self.gene), 4)

    def test_deepcopy(self):
        copied_gene = copy.deepcopy(self.gene)

        self.assertIsNot(copied_gene, self.gene,
                         "Deep copy resulted in the same object reference.")

        # valid for non-composite genetics (do not contain other genetics)
        self.assertEqual(copied_gene.__dict__, self.gene.__dict__,
                         "Attributes of the deep copied object do not match "
                         "the original.")

    def test_from_random(self):
        gene = LinearBlockGene.from_random()
        self.assertIsInstance(gene, LinearBlockGene)
        self._test_within_ranges(gene)


if __name__ == "__main__":
    unittest.main()

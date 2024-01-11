import torch.nn as nn
from dnasty.genes.genes import LinearBlockGene
import unittest


class TestLinearGene(unittest.TestCase):
    def setUp(self):
        self.gene = LinearBlockGene(in_features=10,
                                    out_features=20,
                                    activation='ReLU',
                                    dropout=True)

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
        module = self.gene.express()
        self.assertIsInstance(module, nn.Sequential)

        # Check if the correct modules are added
        self.assertIsInstance(module[0], nn.Linear)
        self.assertEqual(module[0].in_features, 10)
        self.assertEqual(module[0].out_features, 20)
        self.assertIsInstance(module[1], nn.Dropout)
        self.assertIsInstance(module[2], nn.ReLU)

    def test_mutate(self):
        pre_drop = self.gene.dropout
        self.gene.mutate()
        self.assertIn(self.gene.out_features, LinearBlockGene.allowed_features)
        self.assertIn(self.gene.in_features, LinearBlockGene.allowed_features)
        self.assertNotEqual(self.gene.dropout, pre_drop)

    def test_len(self):
        self.assertEqual(len(self.gene), 4)


if __name__ == "__main__":
    unittest.main()

from dnasty.genetics import ConvBlock2dGene
import unittest


class TestConvBlockGene(unittest.TestCase):
    def setUp(self):
        self.exons = {"in_channels": 1, "out_channels": 2, "kernel_size": 3, "activation": "ReLU", "batch_norm": True}
        self.gene = ConvBlock2dGene(in_channels=1, out_channels=2, kernel_size=3, activation="ReLU", batch_norm=True)

    def test_init(self):
        self.assertEqual(self.exons, self.gene.exons)

    def test_getattr(self):
        self.assertEqual(self.gene.in_channels, 1)
        self.assertEqual(self.gene.out_channels, 2)
        self.assertEqual(self.gene.kernel_size, 3)
        self.assertEqual(self.gene.activation, "ReLU")
        self.assertEqual(self.gene.batch_norm, True)
        self.assertRaises(self.gene.unknown, AttributeError)

    def test_len(self):
        self.assertEqual(len(self.gene), len(self.exons))


if __name__ == "__main__":
    unittest.main()